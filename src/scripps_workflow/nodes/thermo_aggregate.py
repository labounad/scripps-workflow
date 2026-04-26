"""``wf-thermo-aggregate`` — composite Gibbs energies from a freq + SP array.

Consumes the manifest of an upstream
:mod:`scripps_workflow.nodes.orca_thermo_array` (or a legacy
``orca_thermo_freq_array``) call, walks every per-task ``orca.out``,
parses the thermochemistry block, and writes a single CSV that is the
canonical "what's the conformer ensemble's free-energy distribution"
deliverable for the workflow.

Composite Gibbs protocol::

    G_composite = E_SP_high + (G - E_el)_low

Where ``E_SP_high`` is the high-level single-point energy (e.g.
``wB97M-V/def2-TZVPP``) read from the *last* ``FINAL SINGLE POINT
ENERGY`` line in the file (the SP job in a compound output), and
``(G - E_el)_low`` is the thermal correction to the electronic energy
read from ORCA's freq block (typically at ``r2scan-3c``).

When no high-level SP is available (e.g. ``singlepoint_keywords=none``
upstream) the aggregator falls back to the low-level total Gibbs::

    G_composite = G_low

Per-conformer rows in the CSV carry:

    * absolute energies: ``E_high_eh``, ``G_minus_Eel_low_eh``,
      ``G_low_eh``, ``H_low_eh``, ``TS_corr_low_eh``, ``G_composite_eh``.
    * derived: ``deltaG_kcal`` (relative to lowest finite
      ``G_composite_eh``, with optional 1 atm → 1 M standard-state
      shift), ``boltzmann_weight``, ``boltzmann_percent``,
      ``cumulative_weight``, ``cumulative_percent``.
    * provenance: ``task_dir``, ``thermo_out``, ``sp_out``,
      ``temperature_k_used``, ``T_low_k_parsed``, ``standard_state``,
      ``std_state_corr_kcal``.

Failure modes (per-conformer, surfaced as structured ``failures``):

    * ``missing_task_dir`` — upstream said n_tasks=N but task_XXXX is
      gone.
    * ``no_orca_out_files_found`` — task dir exists but contains no
      ``*.out``.
    * ``missing_thermochem_or_energy_for_G`` — neither the
      high-level + ΔG-correction pair nor the fallback ``G_low`` was
      parseable. The CSV row is still written but with ``None`` in the
      Gibbs columns.

Top-level failure modes:

    * ``upstream_manifest_unreadable``/``upstream_manifest_not_found``
      surfaced by the framework via :class:`Node`.
    * ``could_not_locate_tasks_root_abs_in_upstream_manifest`` — the
      upstream manifest's ``artifacts.array.tasks_root_abs`` is missing
      AND the conventional ``<upstream_cwd>/outputs/array/tasks``
      fallback also doesn't exist.
    * ``tasks_root_missing`` — the located path doesn't exist on disk.
    * ``n_tasks_invalid_or_zero`` — upstream reported zero tasks.

Config keys (``key=value`` tokens or one JSON object):

    temperature_k    K used for ΔG / weight columns               [298.15]
    standard_state   "1atm" or "1M" — only "1M" applies the
                     RT·ln(24.46) correction to ``deltaG_kcal``.   ["1atm"]
    output_csv       basename of the per-conformer CSV in
                     ``outputs/``                          ["conformer_thermo.csv"]
    fail_policy      "soft" or "hard" — soft (default) keeps
                     exit-code 0 and embeds ok=false; hard
                     exits 1 on any failure.                       ["soft"]
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional

from .. import logging_utils
from ..hashing import sha256_file
from ..node import Node, NodeContext
from ..orca import (
    HARTREE_TO_KCAL,
    parse_orca_thermochem,
    pick_orca_outputs,
)
from ..parsing import normalize_optional_str, parse_float, parse_int
from ..thermo import (
    boltzmann_weights,
    cumulative_weights_by_dg,
    rt_ln_24_46_kcal,
)


#: Default thermochem temperature (K). Matches ORCA's own default and the
#: legacy ``orca_thermo_aggregator`` script.
DEFAULT_TEMPERATURE_K: float = 298.15

#: Default standard-state token. ``1atm`` keeps ΔG values referenced to
#: the gas-phase 1 atm convention (no correction); ``1M`` applies the
#: Ben-Naim RT·ln(24.46) shift to the relative ΔG column.
DEFAULT_STANDARD_STATE: str = "1atm"

#: Default basename of the per-conformer thermo CSV. Lives at
#: ``outputs/<this>`` in the call's directory.
DEFAULT_OUTPUT_CSV: str = "conformer_thermo.csv"


CSV_COLUMNS: tuple[str, ...] = (
    "index",
    "task_dir",
    "thermo_out",
    "sp_out",
    "E_high_eh",
    "G_minus_Eel_low_eh",
    "G_low_eh",
    "G_composite_eh",
    "deltaG_kcal",
    "boltzmann_weight",
    "boltzmann_percent",
    "cumulative_weight",
    "cumulative_percent",
    "standard_state",
    "std_state_corr_kcal",
    "temperature_k_used",
    "T_low_k_parsed",
    "H_low_eh",
    "TS_corr_low_eh",
)


# --------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------


def parse_task_dir(task_dir: Path) -> dict[str, Any]:
    """Walk one ``task_XXXX`` and return a thermo record (no failure code).

    The returned dict carries:

        * ``task_dir`` — absolute path (or the unresolved input if the
          dir is missing — :func:`aggregate_thermo_records` decides
          whether to surface a failure).
        * ``thermo_out`` / ``sp_out`` — absolute paths chosen by
          :func:`scripps_workflow.orca.pick_orca_outputs`, or ``None``.
        * ``E_high_eh`` / ``G_minus_Eel_low_eh`` / ``G_low_eh`` /
          ``H_low_eh`` / ``TS_corr_low_eh`` / ``T_low_k`` — parsed
          values (any may be ``None``).

    This function is intentionally non-fatal — a missing dir, missing
    out files, or unparseable values just produce ``None`` slots. The
    caller is responsible for emitting failure records based on which
    specific combinations are missing.
    """
    rec: dict[str, Any] = {
        "task_dir": str(task_dir),
        "thermo_out": None,
        "sp_out": None,
        "E_high_eh": None,
        "G_minus_Eel_low_eh": None,
        "G_low_eh": None,
        "H_low_eh": None,
        "TS_corr_low_eh": None,
        "T_low_k": None,
    }

    if not task_dir.is_dir():
        return rec

    rec["task_dir"] = str(task_dir.resolve())

    thermo_out, sp_out = pick_orca_outputs(task_dir)
    if thermo_out is not None:
        rec["thermo_out"] = str(thermo_out.resolve())
        tdat = parse_orca_thermochem(thermo_out)
        rec["G_minus_Eel_low_eh"] = tdat["g_minus_e_el_eh"]
        rec["G_low_eh"] = tdat["final_g_eh"]
        rec["H_low_eh"] = tdat["total_h_eh"]
        rec["TS_corr_low_eh"] = tdat["total_entropy_corr_eh"]
        rec["T_low_k"] = tdat["temperature_k"]
    if sp_out is not None:
        rec["sp_out"] = str(sp_out.resolve())
        sdat = parse_orca_thermochem(sp_out)
        # For pure-SP files this is the SCF energy. For a compound (freq +
        # SP) file, ``parse_orca_thermochem`` already returns the LAST
        # FINAL E match, which is the SP from the second job — so this
        # reads the high-level energy correctly without any branching.
        rec["E_high_eh"] = sdat["final_sp_energy_eh"]

    return rec


def composite_gibbs_for_record(rec: dict[str, Any]) -> Optional[float]:
    """Combine high-level SP + low-level thermal correction into ``G_composite``.

    Three branches, in priority order:

        1. ``E_high + (G - E_el)_low`` — the canonical composite protocol.
        2. ``G_low`` — fallback when the SP energy is missing but the
           freq job finished and reported a total Gibbs.
        3. ``None`` — neither the composite pair nor the fallback are
           parseable; the aggregator emits
           ``missing_thermochem_or_energy_for_G`` and writes ``None`` in
           the row.
    """
    e_high = rec.get("E_high_eh")
    g_corr = rec.get("G_minus_Eel_low_eh")
    if e_high is not None and g_corr is not None:
        return float(e_high) + float(g_corr)
    g_low = rec.get("G_low_eh")
    if g_low is not None:
        return float(g_low)
    return None


def aggregate_thermo_records(
    *,
    tasks_root: Path,
    n_tasks: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Walk ``task_0001`` … ``task_{n:04d}`` and collect thermo records.

    Returns ``(records, failures)`` where ``records`` is always 1:1 with
    task indices (length ``n_tasks``) and ``failures`` is a list of
    structured failure dicts ready to hand to :meth:`NodeContext.fail`.

    Failure codes emitted here:

        * ``missing_task_dir`` — task directory absent.
        * ``no_orca_out_files_found`` — task dir present but no
          ``*.out`` inside.
        * ``missing_thermochem_or_energy_for_G`` — neither composite
          pair nor ``G_low`` parseable.

    Each record gets a ``G_composite_eh`` slot (possibly ``None``).
    """
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for i in range(1, int(n_tasks) + 1):
        task_dir = tasks_root / f"task_{i:04d}"
        rec: dict[str, Any] = {"index": i, "G_composite_eh": None}

        if not task_dir.exists():
            rec.update(parse_task_dir(task_dir))  # records the unresolved path
            failures.append(
                {
                    "error": "missing_task_dir",
                    "index": i,
                    "task_dir": str(task_dir),
                }
            )
            records.append(rec)
            continue

        parsed = parse_task_dir(task_dir)
        rec.update(parsed)

        if parsed["thermo_out"] is None and parsed["sp_out"] is None:
            failures.append(
                {
                    "error": "no_orca_out_files_found",
                    "index": i,
                    "task_dir": parsed["task_dir"],
                }
            )
            records.append(rec)
            continue

        g_comp = composite_gibbs_for_record(parsed)
        rec["G_composite_eh"] = g_comp
        if g_comp is None:
            failures.append(
                {
                    "error": "missing_thermochem_or_energy_for_G",
                    "index": i,
                    "task_dir": parsed["task_dir"],
                }
            )
        records.append(rec)

    return records, failures


def compute_relative_gibbs(
    *,
    g_composite: list[Optional[float]],
    standard_state: str,
    temperature_k: float,
) -> tuple[list[Optional[float]], Optional[float], float]:
    """Compute ΔG_kcal from absolute composite Gibbs values.

    Returns ``(deltaG_kcal, g_min, ss_corr_kcal)``:

        * ``deltaG_kcal`` — per-conformer (G - G_min) in kcal/mol, with
          the standard-state correction applied if requested.
        * ``g_min`` — the minimum finite ``G_composite`` (Hartree), or
          ``None`` if every entry was missing.
        * ``ss_corr_kcal`` — the constant additive RT·ln(24.46) shift
          (kcal/mol) applied to every entry; 0.0 unless
          ``standard_state`` is in ``{"1m", "1mol", "1molar"}``.

    Note: because the standard-state correction is a constant additive
    shift, it does NOT affect the relative ranking, Boltzmann weights,
    or cumulative weights. It is included anyway so downstream
    comparisons against experimental free energies of solvation see the
    right absolute number.
    """
    finite = [g for g in g_composite if isinstance(g, (int, float))]
    g_min = float(min(finite)) if finite else None

    ss_token = standard_state.strip().lower()
    if ss_token in {"1m", "1mol", "1molar"}:
        ss_corr_kcal = rt_ln_24_46_kcal(temperature_k)
    else:
        ss_corr_kcal = 0.0

    out: list[Optional[float]] = []
    for g in g_composite:
        if g is None or g_min is None:
            out.append(None)
        else:
            out.append((float(g) - g_min) * HARTREE_TO_KCAL + ss_corr_kcal)
    return out, g_min, ss_corr_kcal


def write_thermo_csv(
    *,
    out_path: Path,
    records: list[dict[str, Any]],
    delta_g_kcal: list[Optional[float]],
    weights: list[Optional[float]],
    cumulative: list[Optional[float]],
    standard_state: str,
    ss_corr_kcal: float,
    temperature_k: float,
) -> None:
    """Write the per-conformer CSV at ``out_path``.

    Columns are exactly :data:`CSV_COLUMNS`. Float values are written
    via Python's default repr, which is round-trippable for ``float``
    types; ``None`` becomes the empty cell. Order matches the input
    record order (task index ascending, 1..N).
    """
    if not (
        len(records) == len(delta_g_kcal) == len(weights) == len(cumulative)
    ):
        raise ValueError(
            "write_thermo_csv: input lists must have equal length "
            f"(records={len(records)}, dG={len(delta_g_kcal)}, "
            f"w={len(weights)}, cum={len(cumulative)})"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for r, dG, wt, cum in zip(
            records, delta_g_kcal, weights, cumulative
        ):
            w.writerow(
                [
                    int(r["index"]),
                    r.get("task_dir"),
                    r.get("thermo_out"),
                    r.get("sp_out"),
                    r.get("E_high_eh"),
                    r.get("G_minus_Eel_low_eh"),
                    r.get("G_low_eh"),
                    r.get("G_composite_eh"),
                    dG,
                    wt,
                    None if wt is None else 100.0 * float(wt),
                    cum,
                    None if cum is None else 100.0 * float(cum),
                    standard_state,
                    ss_corr_kcal,
                    temperature_k,
                    r.get("T_low_k"),
                    r.get("H_low_eh"),
                    r.get("TS_corr_low_eh"),
                ]
            )


def locate_tasks_root(
    *,
    upstream_manifest: dict[str, Any],
) -> tuple[Optional[Path], Optional[int]]:
    """Find ``tasks_root_abs`` + ``n_tasks`` in an upstream array manifest.

    Strategy:

        1. Read ``artifacts.array.tasks_root_abs`` /
           ``artifacts.array.n_tasks`` if present (the new framework
           always populates these).
        2. Fall back to ``<upstream.cwd>/outputs/array/tasks`` if
           that directory exists (legacy layout).
        3. If ``n_tasks`` is unknown, count ``task_*`` directories at
           the located root.

    Returns ``(None, None)`` if neither path resolved. The caller is
    responsible for surfacing a structured failure.
    """
    tasks_root: Optional[Path] = None
    n_tasks: Optional[int] = None

    arts = upstream_manifest.get("artifacts") if isinstance(upstream_manifest, dict) else None
    array_info = arts.get("array") if isinstance(arts, dict) else None
    if isinstance(array_info, dict):
        tr = array_info.get("tasks_root_abs")
        if isinstance(tr, str) and tr:
            candidate = Path(tr)
            if candidate.exists():
                tasks_root = candidate
        nt = array_info.get("n_tasks")
        if isinstance(nt, int):
            n_tasks = nt

    if tasks_root is None:
        cwd = upstream_manifest.get("cwd") if isinstance(upstream_manifest, dict) else None
        if isinstance(cwd, str) and cwd:
            guess = Path(cwd).resolve() / "outputs" / "array" / "tasks"
            if guess.exists():
                tasks_root = guess

    if tasks_root is not None and n_tasks is None:
        n_tasks = len(
            [p for p in tasks_root.glob("task_*") if p.is_dir()]
        )

    return tasks_root, n_tasks


# --------------------------------------------------------------------
# Node class
# --------------------------------------------------------------------


class ThermoAggregate(Node):
    """Aggregate per-conformer thermo from an upstream freq+SP array."""

    step = "thermo_aggregate"
    accepts_upstream = True
    requires_upstream = True

    def parse_config(self, raw: dict[str, Any]) -> dict[str, Any]:
        temperature_k = parse_float(
            raw.get("temperature_k"), DEFAULT_TEMPERATURE_K
        )
        if temperature_k <= 0:
            raise ValueError(
                f"temperature_k must be positive, got {temperature_k!r}"
            )

        ss = normalize_optional_str(raw.get("standard_state"))
        standard_state = (ss or DEFAULT_STANDARD_STATE).lower()
        if standard_state not in {"1atm", "1m", "1mol", "1molar"}:
            raise ValueError(
                f"standard_state must be '1atm' or '1M', got {raw.get('standard_state')!r}"
            )

        output_csv = (
            normalize_optional_str(raw.get("output_csv"))
            or DEFAULT_OUTPUT_CSV
        )
        if "/" in output_csv or output_csv.startswith("."):
            raise ValueError(
                f"output_csv must be a basename, got {output_csv!r}"
            )

        return {
            "temperature_k": temperature_k,
            "standard_state": standard_state,
            "output_csv": output_csv,
            # parse_int isn't used here but exposing the int default to
            # the manifest is useful for debugging operators reading
            # ``inputs``. Keep the explicit cast so JSON round-trips.
            "n_tasks_override": parse_int(raw.get("n_tasks"), 0),
        }

    def run(self, ctx: NodeContext) -> None:
        cfg = ctx.config

        ctx.set_inputs(
            temperature_k=cfg["temperature_k"],
            standard_state=cfg["standard_state"],
            output_csv=cfg["output_csv"],
        )

        if ctx.upstream_manifest is None:
            ctx.fail("no_upstream_manifest")
            return

        # Locate tasks_root via the upstream array bucket; fall back to
        # the conventional <upstream_cwd>/outputs/array/tasks layout. We
        # serialize the upstream manifest back through json so the dict
        # shape is identical to what the legacy aggregator saw — the
        # locate helper is dict-only and easier to test that way.
        upm_dict = ctx.upstream_manifest.to_dict()
        tasks_root, n_tasks = locate_tasks_root(upstream_manifest=upm_dict)
        if tasks_root is None:
            ctx.fail("could_not_locate_tasks_root_abs_in_upstream_manifest")
            return
        if not tasks_root.exists():
            ctx.fail("tasks_root_missing", tasks_root=str(tasks_root))
            return

        if cfg["n_tasks_override"] > 0:
            n_tasks = cfg["n_tasks_override"]
        if not n_tasks:
            ctx.fail("n_tasks_invalid_or_zero")
            return

        ctx.set_input("n_tasks", int(n_tasks))
        ctx.set_input("tasks_root_abs", str(tasks_root.resolve()))

        logging_utils.log_info(
            f"thermo-aggregate: walking {n_tasks} task dirs under {tasks_root}"
        )

        # ---- 1) Walk tasks + parse ----
        records, failures = aggregate_thermo_records(
            tasks_root=tasks_root, n_tasks=n_tasks
        )
        for fail in failures:
            ctx.fail(fail.pop("error"), **fail)

        # ---- 2) Derived columns ----
        g_composite = [r["G_composite_eh"] for r in records]
        delta_g_kcal, g_min, ss_corr_kcal = compute_relative_gibbs(
            g_composite=g_composite,
            standard_state=cfg["standard_state"],
            temperature_k=cfg["temperature_k"],
        )
        weights = boltzmann_weights(delta_g_kcal, cfg["temperature_k"])
        cumulative = cumulative_weights_by_dg(delta_g_kcal, weights)

        # ---- 3) Write CSV ----
        outputs_dir = ctx.outputs_dir
        outputs_dir.mkdir(parents=True, exist_ok=True)
        csv_path = outputs_dir / cfg["output_csv"]
        write_thermo_csv(
            out_path=csv_path,
            records=records,
            delta_g_kcal=delta_g_kcal,
            weights=weights,
            cumulative=cumulative,
            standard_state=cfg["standard_state"],
            ss_corr_kcal=ss_corr_kcal,
            temperature_k=cfg["temperature_k"],
        )
        ctx.add_artifact(
            "files",
            {
                "label": "conformer_thermo_csv",
                "path_abs": str(csv_path.resolve()),
                "sha256": sha256_file(csv_path),
                "format": "csv",
            },
        )

        # ---- 4) Summary JSON ----
        summary = {
            "n_conformers": len(records),
            "n_with_G": len(
                [r for r in records if r.get("G_composite_eh") is not None]
            ),
            "Gmin_eh": g_min,
            "temperature_k": cfg["temperature_k"],
            "standard_state": cfg["standard_state"],
            "std_state_corr_kcal": ss_corr_kcal,
        }
        summary_path = outputs_dir / "thermo_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        ctx.add_artifact(
            "files",
            {
                "label": "thermo_summary_json",
                "path_abs": str(summary_path.resolve()),
                "sha256": sha256_file(summary_path),
                "format": "json",
            },
        )

        # ---- 5) Per-conformer records bucket ----
        # The CSV is the deliverable, but downstream nodes (e.g. a
        # plotting node) may want structured records too. Mirror the
        # CSV row shape into the manifest's ``conformers`` bucket.
        for r, dG, wt, cum in zip(
            records, delta_g_kcal, weights, cumulative
        ):
            # ``path_abs`` is the schema's required canonical-path field.
            # For per-conformer records the *task directory* is the
            # natural "where this conformer's data lives" anchor — fall
            # back to the CSV path itself if (very rarely) we somehow
            # got here without a task_dir, just to satisfy the schema.
            task_dir_abs = r.get("task_dir") or str(csv_path)
            ctx.add_artifact(
                "conformers",
                {
                    "path_abs": task_dir_abs,
                    "index": int(r["index"]),
                    "label": f"conf_{int(r['index']):04d}",
                    "task_dir_abs": task_dir_abs,
                    "thermo_out_abs": r.get("thermo_out"),
                    "sp_out_abs": r.get("sp_out"),
                    "E_high_eh": r.get("E_high_eh"),
                    "G_minus_Eel_low_eh": r.get("G_minus_Eel_low_eh"),
                    "G_low_eh": r.get("G_low_eh"),
                    "H_low_eh": r.get("H_low_eh"),
                    "TS_corr_low_eh": r.get("TS_corr_low_eh"),
                    "T_low_k": r.get("T_low_k"),
                    "G_composite_eh": r.get("G_composite_eh"),
                    "deltaG_kcal": dG,
                    "boltzmann_weight": wt,
                    "cumulative_weight": cum,
                },
            )


__all__ = [
    "CSV_COLUMNS",
    "DEFAULT_OUTPUT_CSV",
    "DEFAULT_STANDARD_STATE",
    "DEFAULT_TEMPERATURE_K",
    "ThermoAggregate",
    "aggregate_thermo_records",
    "composite_gibbs_for_record",
    "compute_relative_gibbs",
    "locate_tasks_root",
    "main",
    "parse_task_dir",
    "write_thermo_csv",
]


main = ThermoAggregate.invoke_factory()


if __name__ == "__main__":
    raise SystemExit(main())
