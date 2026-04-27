"""``wf-nmr-aggregate`` — Boltzmann-averaged NMR shifts + couplings.

Consumes the manifest of an upstream :mod:`scripps_workflow.nodes.thermo_aggregate`
call, walks each conformer's task directory, parses chemical shieldings and
J-coupling tables from the (compound) ORCA output, applies population-weighted
averaging using the Boltzmann weights the thermo aggregator already computed,
then applies linear-scaling correction (cheshire / Bally-Rablen) to produce
predicted experimental observables. Writes two CSVs:

    * ``predicted_shifts.csv`` — per atom: index, element, σ_iso_avg,
      δ_predicted, calibration source.
    * ``predicted_couplings.csv`` — per (i, j) pair: indices, elements,
      J_total_avg, J_predicted, calibration source.

The compute path lives in :mod:`scripps_workflow.nodes.orca_thermo_array`,
extended to chain freq + high-level SP + NMR shielding(s) + J-coupling jobs
inside ONE ORCA invocation per conformer (separated by ``$new_job``). That
keeps SLURM allocation cost down — large queue times mean reusing a granted
node for several jobs is much cheaper than queuing each job separately.

Linear scaling is applied AFTER Boltzmann averaging — mathematically
equivalent to scaling each conformer first then averaging, since both
operations are linear. The order chosen here keeps the calibration
lookup to once-per-nucleus rather than once-per-(conformer × nucleus).

Failure modes:

    * ``upstream_manifest_unreadable`` / ``upstream_manifest_not_found`` —
      framework-level (Node base class).
    * ``upstream_missing_conformers_bucket`` — upstream is not a thermo
      aggregator (or its conformers bucket is empty).
    * ``no_finite_weights`` — every conformer was None-weighted (the
      upstream had no parseable Gibbs).
    * ``no_shielding_data_in_any_conformer`` — every conformer's .out
      file failed to produce shielding rows.
    * ``calibration_not_found`` — surfaced as a structured failure
      (NOT an exception) when a configured (functional, basis, solvent,
      nucleus) tuple isn't in the calibration table. Raw σ values are
      still written to CSV under ``sigma_iso_ppm``.

Config keys (``key=value`` tokens or one JSON object):

    solvent                  [CHCl3]   used in calibration lookup
    shielding_method_h       [WP04]
    shielding_basis_h        [6-311++G(2d,p)]
    shielding_method_c       [wB97X-D]
    shielding_basis_c        [6-31G(d,p)]
    coupling_method          [mPW1PW91]
    coupling_basis           [pcJ-2]
    output_shifts_csv        [predicted_shifts.csv]
    output_couplings_csv     [predicted_couplings.csv]
    skip_couplings           [false]   skip J table parse + write
    fail_policy              [soft]
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional

from .. import logging_utils
from ..hashing import sha256_file
from ..nmr_calibration import (
    lookup_calibration,
    predict_chemical_shift,
    predict_coupling_constant,
)
from ..node import Node, NodeContext
from ..orca import (
    parse_orca_couplings,
    parse_orca_shieldings,
    pick_orca_outputs,
)
from ..parsing import normalize_optional_str, parse_bool


DEFAULT_SOLVENT: str = "CHCl3"

DEFAULT_SHIELDING_METHOD_H: str = "WP04"
DEFAULT_SHIELDING_BASIS_H: str = "6-311++G(2d,p)"

DEFAULT_SHIELDING_METHOD_C: str = "wB97X-D"
DEFAULT_SHIELDING_BASIS_C: str = "6-31G(d,p)"

DEFAULT_COUPLING_METHOD: str = "mPW1PW91"
DEFAULT_COUPLING_BASIS: str = "pcJ-2"

DEFAULT_OUTPUT_SHIFTS_CSV: str = "predicted_shifts.csv"
DEFAULT_OUTPUT_COUPLINGS_CSV: str = "predicted_couplings.csv"


SHIFT_CSV_COLUMNS: tuple[str, ...] = (
    "atom_index",
    "element",
    "sigma_iso_avg_ppm",
    "delta_predicted_ppm",
    "calibration_source",
    "calibration_method",
    "calibration_basis",
    "calibration_solvent",
    "n_conformers_used",
)

COUPLING_CSV_COLUMNS: tuple[str, ...] = (
    "i",
    "elem_i",
    "j",
    "elem_j",
    "J_total_avg_hz",
    "J_predicted_hz",
    "calibration_source",
    "calibration_method",
    "calibration_basis",
    "calibration_solvent",
    "n_conformers_used",
)


# --------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------


def collect_conformer_records(
    upstream_manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    """Pull the per-conformer records (task_dir + boltzmann_weight) out
    of an upstream :mod:`thermo_aggregate` manifest dict.

    Returns the raw list (1:1 with task indices). Empty list if the
    bucket is missing or empty.
    """
    arts = upstream_manifest.get("artifacts") if isinstance(upstream_manifest, dict) else None
    if not isinstance(arts, dict):
        return []
    confs = arts.get("conformers")
    return list(confs) if isinstance(confs, list) else []


def renormalize_weights(
    weights: list[Optional[float]],
) -> list[Optional[float]]:
    """Renormalize a weight list so finite entries sum to 1.0.

    Entries that are ``None`` (or non-finite) keep ``None`` — useful
    when some conformers had no parseable thermochemistry but the rest
    of the ensemble should still average cleanly.

    Returns ``[]`` for an all-None / empty input.
    """
    finite_total = 0.0
    any_finite = False
    for w in weights:
        if isinstance(w, (int, float)):
            finite_total += float(w)
            any_finite = True
    if not any_finite or finite_total <= 0.0:
        return [None] * len(weights)
    out: list[Optional[float]] = []
    for w in weights:
        if isinstance(w, (int, float)):
            out.append(float(w) / finite_total)
        else:
            out.append(None)
    return out


def boltzmann_average_shieldings(
    *,
    per_conformer: list[Optional[list[dict[str, Any]]]],
    weights: list[Optional[float]],
) -> tuple[dict[int, dict[str, Any]], int]:
    """Population-weight chemical shieldings across conformers.

    Returns ``(by_atom_index, n_used)`` where:

        * ``by_atom_index[k]`` is
          ``{"element": str, "sigma_iso_avg_ppm": float}`` and the keys
          are sorted ascending atom indices.
        * ``n_used`` is the number of conformers that contributed
          shielding data AND had a finite weight.

    Conformers with ``None`` weight or empty shielding lists are
    skipped. Weights are renormalized over the contributing subset, so
    a single failed conformer doesn't bias the average. When a given
    atom index appears in only some conformers (shouldn't happen in
    practice — same molecule, same atom order — but defensively) the
    average is over the conformers in which it does appear.
    """
    if len(per_conformer) != len(weights):
        raise ValueError(
            "boltzmann_average_shieldings: per_conformer / weights length mismatch"
        )

    contributing: list[tuple[float, list[dict[str, Any]]]] = []
    for shi, w in zip(per_conformer, weights):
        if not shi or not isinstance(w, (int, float)):
            continue
        contributing.append((float(w), shi))

    if not contributing:
        return {}, 0

    total_w = sum(w for w, _ in contributing)
    if total_w <= 0.0:
        return {}, 0

    by_atom: dict[int, dict[str, Any]] = {}
    weight_per_atom: dict[int, float] = {}
    for w, rows in contributing:
        rw = w / total_w
        for r in rows:
            idx = int(r["atom_index"])
            sigma = float(r["sigma_iso_ppm"])
            entry = by_atom.setdefault(
                idx,
                {"element": r["element"], "sigma_iso_avg_ppm": 0.0},
            )
            entry["sigma_iso_avg_ppm"] += rw * sigma
            weight_per_atom[idx] = weight_per_atom.get(idx, 0.0) + rw

    # If an atom appeared only in a strict subset of conformers,
    # renormalize that atom's accumulated value back to a "weighted
    # average over the conformers it appeared in". This branch is
    # defensive — in well-behaved runs every conformer reports the
    # same atom set and this is a no-op.
    for idx, sub_w in weight_per_atom.items():
        if abs(sub_w - 1.0) > 1e-9 and sub_w > 0.0:
            by_atom[idx]["sigma_iso_avg_ppm"] /= sub_w

    return by_atom, len(contributing)


def boltzmann_average_couplings(
    *,
    per_conformer: list[Optional[list[dict[str, Any]]]],
    weights: list[Optional[float]],
) -> tuple[dict[tuple[int, int], dict[str, Any]], int]:
    """Population-weight J-couplings across conformers.

    Returns ``(by_pair, n_used)`` where ``by_pair[(i,j)]`` =
    ``{"elem_i": str, "elem_j": str, "J_total_avg_hz": float}``.
    ``n_used`` counts conformers that contributed at least one pair.
    Same renormalization-on-subset behavior as
    :func:`boltzmann_average_shieldings`.

    Pairs with no parseable ``J_total_hz`` in any conformer are
    dropped — there's no meaningful partial average for J.
    """
    if len(per_conformer) != len(weights):
        raise ValueError(
            "boltzmann_average_couplings: per_conformer / weights length mismatch"
        )

    contributing: list[tuple[float, list[dict[str, Any]]]] = []
    for cps, w in zip(per_conformer, weights):
        if not cps or not isinstance(w, (int, float)):
            continue
        contributing.append((float(w), cps))

    if not contributing:
        return {}, 0

    total_w = sum(w for w, _ in contributing)
    if total_w <= 0.0:
        return {}, 0

    by_pair: dict[tuple[int, int], dict[str, Any]] = {}
    weight_per_pair: dict[tuple[int, int], float] = {}
    for w, rows in contributing:
        rw = w / total_w
        for r in rows:
            j_total = r.get("J_total_hz")
            if j_total is None:
                continue
            key = (int(r["i"]), int(r["j"]))
            entry = by_pair.setdefault(
                key,
                {
                    "elem_i": r["elem_i"],
                    "elem_j": r["elem_j"],
                    "J_total_avg_hz": 0.0,
                },
            )
            entry["J_total_avg_hz"] += rw * float(j_total)
            weight_per_pair[key] = weight_per_pair.get(key, 0.0) + rw

    for key, sub_w in weight_per_pair.items():
        if abs(sub_w - 1.0) > 1e-9 and sub_w > 0.0:
            by_pair[key]["J_total_avg_hz"] /= sub_w

    return by_pair, len(contributing)


def write_shifts_csv(
    *,
    out_path: Path,
    by_atom: dict[int, dict[str, Any]],
    n_used: int,
    cal_h: Optional[dict[str, Any]],
    cal_c: Optional[dict[str, Any]],
    cfg: dict[str, Any],
) -> None:
    """Write the per-atom predicted-shifts CSV.

    For each atom, look up the appropriate calibration (H vs C). When
    no calibration is available for the element, ``delta_predicted_ppm``
    is left empty and the calibration provenance columns are blank —
    the raw ``sigma_iso_avg_ppm`` is still emitted so downstream
    consumers can apply their own scaling.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(SHIFT_CSV_COLUMNS)
        for idx in sorted(by_atom):
            entry = by_atom[idx]
            elem = entry["element"]
            sigma = float(entry["sigma_iso_avg_ppm"])

            if elem == "H" and cal_h is not None:
                cal = cal_h
                method = cfg["shielding_method_h"]
                basis = cfg["shielding_basis_h"]
            elif elem == "C" and cal_c is not None:
                cal = cal_c
                method = cfg["shielding_method_c"]
                basis = cfg["shielding_basis_c"]
            else:
                cal = None
                method = ""
                basis = ""

            if cal is not None:
                delta = predict_chemical_shift(
                    sigma, slope=cal["slope"], intercept=cal["intercept"]
                )
                source = cal.get("source", "")
            else:
                delta = None
                source = ""

            w.writerow(
                [
                    idx,
                    elem,
                    sigma,
                    delta,
                    source,
                    method,
                    basis,
                    cfg["solvent"],
                    n_used,
                ]
            )


def write_couplings_csv(
    *,
    out_path: Path,
    by_pair: dict[tuple[int, int], dict[str, Any]],
    n_used: int,
    cal_jhh: Optional[dict[str, Any]],
    cfg: dict[str, Any],
) -> None:
    """Write the per-pair predicted-couplings CSV.

    Currently only ¹H-¹H pairs are linearly scaled (the Bally/Rablen
    calibration in the default table is for H-H J's). Heteronuclear
    pairs are emitted with raw ``J_total_avg_hz`` and an empty
    ``J_predicted_hz``.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COUPLING_CSV_COLUMNS)
        for (i, j) in sorted(by_pair):
            entry = by_pair[(i, j)]
            ei, ej = entry["elem_i"], entry["elem_j"]
            j_avg = float(entry["J_total_avg_hz"])

            is_hh = (ei == "H" and ej == "H")
            if is_hh and cal_jhh is not None:
                j_pred = predict_coupling_constant(
                    j_avg,
                    slope=cal_jhh["slope"],
                    intercept=cal_jhh["intercept"],
                )
                source = cal_jhh.get("source", "")
                method = cfg["coupling_method"]
                basis = cfg["coupling_basis"]
            else:
                j_pred = None
                source = ""
                method = ""
                basis = ""

            w.writerow(
                [
                    i,
                    ei,
                    j,
                    ej,
                    j_avg,
                    j_pred,
                    source,
                    method,
                    basis,
                    cfg["solvent"],
                    n_used,
                ]
            )


# --------------------------------------------------------------------
# Node class
# --------------------------------------------------------------------


class NmrAggregate(Node):
    """Boltzmann-average NMR observables over a thermo-aggregated ensemble."""

    step = "nmr_aggregate"
    accepts_upstream = True
    requires_upstream = True

    def parse_config(self, raw: dict[str, Any]) -> dict[str, Any]:
        solvent = (
            normalize_optional_str(raw.get("solvent")) or DEFAULT_SOLVENT
        )

        shielding_method_h = (
            normalize_optional_str(raw.get("shielding_method_h"))
            or DEFAULT_SHIELDING_METHOD_H
        )
        shielding_basis_h = (
            normalize_optional_str(raw.get("shielding_basis_h"))
            or DEFAULT_SHIELDING_BASIS_H
        )
        shielding_method_c = (
            normalize_optional_str(raw.get("shielding_method_c"))
            or DEFAULT_SHIELDING_METHOD_C
        )
        shielding_basis_c = (
            normalize_optional_str(raw.get("shielding_basis_c"))
            or DEFAULT_SHIELDING_BASIS_C
        )
        coupling_method = (
            normalize_optional_str(raw.get("coupling_method"))
            or DEFAULT_COUPLING_METHOD
        )
        coupling_basis = (
            normalize_optional_str(raw.get("coupling_basis"))
            or DEFAULT_COUPLING_BASIS
        )

        output_shifts_csv = (
            normalize_optional_str(raw.get("output_shifts_csv"))
            or DEFAULT_OUTPUT_SHIFTS_CSV
        )
        output_couplings_csv = (
            normalize_optional_str(raw.get("output_couplings_csv"))
            or DEFAULT_OUTPUT_COUPLINGS_CSV
        )
        for name, val in (
            ("output_shifts_csv", output_shifts_csv),
            ("output_couplings_csv", output_couplings_csv),
        ):
            if "/" in val or val.startswith("."):
                raise ValueError(f"{name} must be a basename, got {val!r}")

        return {
            "solvent": solvent,
            "shielding_method_h": shielding_method_h,
            "shielding_basis_h": shielding_basis_h,
            "shielding_method_c": shielding_method_c,
            "shielding_basis_c": shielding_basis_c,
            "coupling_method": coupling_method,
            "coupling_basis": coupling_basis,
            "output_shifts_csv": output_shifts_csv,
            "output_couplings_csv": output_couplings_csv,
            "skip_couplings": parse_bool(raw.get("skip_couplings"), False),
        }

    def run(self, ctx: NodeContext) -> None:
        cfg = ctx.config
        ctx.set_inputs(**cfg)

        if ctx.upstream_manifest is None:
            ctx.fail("no_upstream_manifest")
            return

        upm_dict = ctx.upstream_manifest.to_dict()
        confs = collect_conformer_records(upm_dict)
        if not confs:
            ctx.fail("upstream_missing_conformers_bucket")
            return

        # ---- Walk conformers + parse ----
        weights: list[Optional[float]] = []
        per_conformer_shieldings: list[Optional[list[dict[str, Any]]]] = []
        per_conformer_couplings: list[Optional[list[dict[str, Any]]]] = []

        for c in confs:
            wt = c.get("boltzmann_weight")
            weights.append(wt if isinstance(wt, (int, float)) else None)

            task_dir_str = c.get("task_dir_abs") or c.get("path_abs")
            if not isinstance(task_dir_str, str):
                per_conformer_shieldings.append(None)
                per_conformer_couplings.append(None)
                continue
            task_dir = Path(task_dir_str)
            thermo_out, _sp_out = pick_orca_outputs(task_dir)
            if thermo_out is None:
                per_conformer_shieldings.append(None)
                per_conformer_couplings.append(None)
                continue

            sh = parse_orca_shieldings(thermo_out)
            per_conformer_shieldings.append(sh or None)
            if cfg["skip_couplings"]:
                per_conformer_couplings.append(None)
            else:
                cp = parse_orca_couplings(thermo_out)
                per_conformer_couplings.append(cp or None)

        if not any(isinstance(w, (int, float)) for w in weights):
            ctx.fail("no_finite_weights")
            return

        norm_weights = renormalize_weights(weights)

        logging_utils.log_info(
            f"nmr-aggregate: walking {len(confs)} conformers "
            f"({sum(1 for w in norm_weights if w is not None)} with finite weight)"
        )

        # ---- Boltzmann-average shieldings ----
        by_atom, n_used_sh = boltzmann_average_shieldings(
            per_conformer=per_conformer_shieldings,
            weights=norm_weights,
        )
        if not by_atom:
            # No conformer produced parseable shielding data — surface
            # the failure and return. We deliberately skip the rest of
            # the pipeline (calibration lookups, CSV emission) because
            # there is nothing to write; doing the work anyway would
            # only emit empty CSVs and a cascade of irrelevant
            # ``calibration_not_found`` records that mask the real
            # problem in the manifest.
            ctx.fail("no_shielding_data_in_any_conformer")
            return

        # Detect which elements actually appear in the averaged data;
        # only look up (and possibly fail on) calibrations for elements
        # we will need. This keeps the manifest's failure list signal-
        # rich — e.g. a ¹H-only run won't emit a spurious
        # ``calibration_not_found`` for ¹³C.
        elements_present = {entry["element"] for entry in by_atom.values()}

        # ---- Calibration lookups ----
        cal_h = (
            lookup_calibration(
                functional=cfg["shielding_method_h"],
                basis=cfg["shielding_basis_h"],
                solvent=cfg["solvent"],
                nucleus="1H",
            )
            if "H" in elements_present
            else None
        )
        cal_c = (
            lookup_calibration(
                functional=cfg["shielding_method_c"],
                basis=cfg["shielding_basis_c"],
                solvent=cfg["solvent"],
                nucleus="13C",
            )
            if "C" in elements_present
            else None
        )
        cal_jhh = (
            None
            if cfg["skip_couplings"]
            else lookup_calibration(
                functional=cfg["coupling_method"],
                basis=cfg["coupling_basis"],
                solvent=cfg["solvent"],
                nucleus="1H-1H_J",
            )
        )

        # Surface missing-calibration as structured failures (not
        # exceptions). The CSVs are still written with raw σ. Only
        # emit the failure when the element is actually present in
        # the data — see ``elements_present`` above.
        if "H" in elements_present and cal_h is None:
            ctx.fail(
                "calibration_not_found",
                nucleus="1H",
                functional=cfg["shielding_method_h"],
                basis=cfg["shielding_basis_h"],
                solvent=cfg["solvent"],
            )
        if "C" in elements_present and cal_c is None:
            ctx.fail(
                "calibration_not_found",
                nucleus="13C",
                functional=cfg["shielding_method_c"],
                basis=cfg["shielding_basis_c"],
                solvent=cfg["solvent"],
            )
        if not cfg["skip_couplings"] and cal_jhh is None:
            # We can't tell from ``by_atom`` alone whether a coupling
            # CSV will have any H-H rows — coupling parsing happens
            # below. Emit the calibration failure preemptively only
            # when ¹H is present; if no H atoms exist at all, the H-H
            # calibration is moot.
            if "H" in elements_present:
                ctx.fail(
                    "calibration_not_found",
                    nucleus="1H-1H_J",
                    functional=cfg["coupling_method"],
                    basis=cfg["coupling_basis"],
                    solvent=cfg["solvent"],
                )

        # ---- Write CSVs + summary ----
        outputs_dir = ctx.outputs_dir
        outputs_dir.mkdir(parents=True, exist_ok=True)

        shifts_path = outputs_dir / cfg["output_shifts_csv"]
        write_shifts_csv(
            out_path=shifts_path,
            by_atom=by_atom,
            n_used=n_used_sh,
            cal_h=cal_h,
            cal_c=cal_c,
            cfg=cfg,
        )
        ctx.add_artifact(
            "files",
            {
                "label": "predicted_shifts_csv",
                "path_abs": str(shifts_path.resolve()),
                "sha256": sha256_file(shifts_path),
                "format": "csv",
            },
        )

        n_used_cp = 0
        if not cfg["skip_couplings"]:
            by_pair, n_used_cp = boltzmann_average_couplings(
                per_conformer=per_conformer_couplings,
                weights=norm_weights,
            )
            couplings_path = outputs_dir / cfg["output_couplings_csv"]
            write_couplings_csv(
                out_path=couplings_path,
                by_pair=by_pair,
                n_used=n_used_cp,
                cal_jhh=cal_jhh,
                cfg=cfg,
            )
            ctx.add_artifact(
                "files",
                {
                    "label": "predicted_couplings_csv",
                    "path_abs": str(couplings_path.resolve()),
                    "sha256": sha256_file(couplings_path),
                    "format": "csv",
                },
            )

        summary = {
            "n_conformers_total": len(confs),
            "n_conformers_with_shielding": n_used_sh,
            "n_conformers_with_couplings": n_used_cp,
            "n_atoms": len(by_atom),
            "calibration_h_used": bool(cal_h),
            "calibration_c_used": bool(cal_c),
            "calibration_jhh_used": bool(cal_jhh),
            "solvent": cfg["solvent"],
        }
        summary_path = outputs_dir / "nmr_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        ctx.add_artifact(
            "files",
            {
                "label": "nmr_summary_json",
                "path_abs": str(summary_path.resolve()),
                "sha256": sha256_file(summary_path),
                "format": "json",
            },
        )


__all__ = [
    "COUPLING_CSV_COLUMNS",
    "DEFAULT_COUPLING_BASIS",
    "DEFAULT_COUPLING_METHOD",
    "DEFAULT_OUTPUT_COUPLINGS_CSV",
    "DEFAULT_OUTPUT_SHIFTS_CSV",
    "DEFAULT_SHIELDING_BASIS_C",
    "DEFAULT_SHIELDING_BASIS_H",
    "DEFAULT_SHIELDING_METHOD_C",
    "DEFAULT_SHIELDING_METHOD_H",
    "DEFAULT_SOLVENT",
    "NmrAggregate",
    "SHIFT_CSV_COLUMNS",
    "boltzmann_average_couplings",
    "boltzmann_average_shieldings",
    "collect_conformer_records",
    "main",
    "renormalize_weights",
    "write_couplings_csv",
    "write_shifts_csv",
]


main = NmrAggregate.invoke_factory()


if __name__ == "__main__":
    raise SystemExit(main())
