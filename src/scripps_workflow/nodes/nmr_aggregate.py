"""``wf-nmr-aggregate`` — Boltzmann-averaged NMR shifts + couplings.

Consumes the manifest of an upstream :mod:`scripps_workflow.nodes.thermo_aggregate`
call, walks each conformer's task directory, parses chemical shieldings and
J-coupling tables from the (compound) ORCA output, applies population-weighted
averaging using the Boltzmann weights the thermo aggregator already computed,
then applies linear-scaling correction (cheshire / Bally-Rablen) to produce
predicted experimental observables. Writes two CSVs and (optionally) up to
four mnova-spinsim XML files:

    * ``predicted_shifts.csv`` — per atom: index, element, σ_iso_avg,
      δ_predicted, calibration source.
    * ``predicted_couplings.csv`` — per (i, j) pair: indices, elements,
      J_total_avg, J_predicted, calibration source.
    * ``predicted_mnova_1h.xml`` / ``predicted_mnova_13c.xml`` —
      pre-averaged spin-system files for the mnova spin-simulator,
      with chemical equivalence detected via
      :mod:`scripps_workflow.equivalence` (homotopic groups collapse,
      AA'BB'-style magnetically inequivalent groups stay split).
    * ``predicted_mnova_*_per_conformer.xml`` — one
      ``<spin-system>`` per conformer with Boltzmann weight as
      ``<population>``, lets mnova render the conformational
      ensemble directly. Emission is gated by ``mnova_per_conformer``.
    * ``predicted_structure_*.svg`` — 2D molecule diagram with each
      equivalence group's atoms annotated by predicted shift + group
      name. Browser-viewable, embeddable in reports.
    * ``predicted_structure_*.html`` — standalone HTML page with an
      embedded 3Dmol.js viewer; user can drag to rotate, scroll to
      zoom, and read all the labels in 3D. Loads 3Dmol.js from CDN.

mnova XML emission and the diagrams BOTH need a SMILES (or, as
fallback, a parseable xyz) to drive the equivalence detector and
shape the visualization. Without either, all visualization paths are
skipped with a structured ``mnova_xml_skipped:topology_unavailable``
failure record; the CSV path keeps running unaffected. The XML and
diagram outputs are independently gated by ``mnova_enabled`` and
``diagrams_enabled`` respectively.

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
    * ``no_hh_couplings_parsed`` — ``skip_couplings=false`` and the
      molecule has ≥2 H atoms but zero H–H pairs came back. Normally
      means the parser missed the format or the ORCA J-coupling jobs
      crashed. A *sparse* H–H table is expected and is NOT a failure —
      ORCA's ``SpinSpinRThresh`` (default 8 Å) deliberately excludes
      long-range pairs because their J's are negligible.
    * ``coupling_parse_empty`` — per-conformer, ``orca_nmr_j.out``
      exists but ``parse_orca_couplings`` returned no rows. Distinct
      from a missing file; surfaces parser/format mismatches.
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

    smiles                   [None]    canonical SMILES driving the
                                       equivalence detector. When
                                       absent the emitter falls back
                                       to xyz perception of any
                                       conformer's geometry.
    mnova_enabled            [true]    master toggle for XML emission.
    mnova_per_conformer      [true]    also emit per-conformer XMLs.
    mnova_nuclei             [1H,13C]  comma-separated; "1H", "13C",
                                       or "1H,13C". Each requested
                                       nucleus produces its own file.
    mnova_field_mhz_h        [400.13]  ¹H Larmor frequency.
    mnova_field_mhz_c        [100.61]  ¹³C Larmor frequency.
    mnova_line_width_hz_h    [1.0]     simulated linewidth, ¹H.
    mnova_line_width_hz_c    [5.0]     simulated linewidth, ¹³C.
    mnova_from_ppm_h         [0.0]     spectrum window low edge, ¹H.
    mnova_to_ppm_h           [12.0]    spectrum window high edge, ¹H.
    mnova_from_ppm_c         [0.0]     spectrum window low edge, ¹³C.
    mnova_to_ppm_c           [220.0]   spectrum window high edge, ¹³C.
    mnova_points             [16384]   FID points for the simulation.
    mnova_tol_jcoupling_hz   [0.5]     magnetic-equivalence tolerance.
    mnova_tol_shift_ppm      [0.05]    data-aware refinement tolerance
                                       (splits topological classes
                                       whose DFT shifts spread above
                                       this — catches diastereotopic
                                       CH₂ in chiral environments).

    diagrams_enabled         [true]    emit SVG + HTML molecule diagrams
                                       per requested nucleus. Reuses
                                       ``mnova_nuclei`` for nucleus
                                       selection and ``smiles`` /
                                       xyz fallback for the structure.
    diagrams_width           [900]     diagram canvas width (px).
    diagrams_height          [600]     diagram canvas height (px).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional

from .. import logging_utils
from ..equivalence import (
    EquivalenceGroup,
    compute_equivalence_groups,
    mol_from_smiles_or_xyz,
)
from ..hashing import sha256_file
from ..mnova_xml import SpectrumConfig, SpinSystem, render_mnova_xml
from ..molecule_diagram import render_shift_html, render_shift_svg
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
from ..parsing import normalize_optional_str, parse_bool, parse_float, parse_int


DEFAULT_SOLVENT: str = "CHCl3"

DEFAULT_SHIELDING_METHOD_H: str = "WP04"
DEFAULT_SHIELDING_BASIS_H: str = "6-311++G(2d,p)"

DEFAULT_SHIELDING_METHOD_C: str = "wB97X-D"
DEFAULT_SHIELDING_BASIS_C: str = "6-31G(d,p)"

DEFAULT_COUPLING_METHOD: str = "mPW1PW91"
DEFAULT_COUPLING_BASIS: str = "pcJ-2"

DEFAULT_OUTPUT_SHIFTS_CSV: str = "predicted_shifts.csv"
DEFAULT_OUTPUT_COUPLINGS_CSV: str = "predicted_couplings.csv"

# --------------------------------------------------------------------
# mnova-spinsim XML emission
# --------------------------------------------------------------------

#: Comma-separated list of nuclei the XML emitter is willing to render.
#: Order is irrelevant — each nucleus produces an independent XML file.
DEFAULT_MNOVA_NUCLEI: str = "1H,13C"

#: Larmor frequency presets in MHz on a 400 MHz spectrometer (¹H Larmor
#: ≈ field × 42.577; ¹³C ≈ field × 10.708). Override when the user's
#: spectrometer is different (e.g., 600 MHz → 600.13 / 150.92).
DEFAULT_MNOVA_FIELD_MHZ_H: float = 400.13
DEFAULT_MNOVA_FIELD_MHZ_C: float = 100.61

#: Per-nucleus simulated linewidth (Hz). ¹H signals are typically narrow
#: (1 Hz at half-height in fluid solution); broadband-decoupled ¹³C is
#: deliberately broader to reflect the spectrum's actual visual.
DEFAULT_MNOVA_LINE_WIDTH_HZ_H: float = 1.0
DEFAULT_MNOVA_LINE_WIDTH_HZ_C: float = 5.0

#: Spectrum window (ppm) per nucleus. Defaults bracket organic
#: molecules' typical chemical-shift ranges with a small margin.
DEFAULT_MNOVA_FROM_PPM_H: float = 0.0
DEFAULT_MNOVA_TO_PPM_H: float = 12.0
DEFAULT_MNOVA_FROM_PPM_C: float = 0.0
DEFAULT_MNOVA_TO_PPM_C: float = 220.0

#: FID points for the simulated spectrum. 16k matches the reference
#: file (SF_M_001.xml); higher values give finer resolution at the
#: cost of mnova render time.
DEFAULT_MNOVA_POINTS: int = 16384

#: Equivalence detection tolerances. ``tol_jcoupling_hz`` controls
#: the magnetic-equivalence J-vector test (HARD vs SOFT dispatch).
#: ``tol_shift_ppm`` controls the data-aware refinement that splits
#: topological classes whose DFT shifts spread above noise (e.g.,
#: diastereotopic CH₂ in chiral environments).
DEFAULT_MNOVA_TOL_JCOUPLING_HZ: float = 0.5
DEFAULT_MNOVA_TOL_SHIFT_PPM: float = 0.05

#: Output filenames. Per nucleus we emit one pre-averaged file and
#: optionally one per-conformer file (controlled by mnova_per_conformer).
DEFAULT_MNOVA_FILENAME_FMT: str = "predicted_mnova_{nucleus}.xml"
DEFAULT_MNOVA_PER_CONFORMER_FILENAME_FMT: str = (
    "predicted_mnova_{nucleus}_per_conformer.xml"
)

# --------------------------------------------------------------------
# Molecule-diagram artifacts (2D SVG + 3D HTML viewer)
# --------------------------------------------------------------------

#: Per-nucleus 2D SVG depiction of the molecule with each
#: equivalence-group's atoms annotated by predicted shift + group name.
DEFAULT_DIAGRAM_SVG_FILENAME_FMT: str = "predicted_structure_{nucleus}.svg"

#: Per-nucleus standalone HTML page with an embedded 3Dmol.js viewer.
#: Loads the molecule's optimized 3D geometry and labels each group at
#: its centroid. The user can drag to rotate, scroll to zoom.
DEFAULT_DIAGRAM_HTML_FILENAME_FMT: str = "predicted_structure_{nucleus}.html"

#: SVG / HTML canvas dimensions in pixels. 900x600 fits comfortably on
#: a laptop screen and matches a typical lab notebook export size.
DEFAULT_DIAGRAM_WIDTH: int = 900
DEFAULT_DIAGRAM_HEIGHT: int = 600


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
# mnova XML helpers (shape adapters between aggregator state +
# scripps_workflow.equivalence / scripps_workflow.mnova_xml inputs)
# --------------------------------------------------------------------


def _shifts_by_atom_from_by_atom(
    *,
    by_atom: dict[int, dict[str, Any]],
    element: str,
    cal: dict[str, Any],
) -> dict[int, float]:
    """Convert the aggregator's averaged ``by_atom`` map to the
    ``{atom_idx → predicted δ_ppm}`` shape that
    :func:`compute_equivalence_groups` expects.

    Filters to the requested ``element`` and applies the cheshire-
    style linear scaling ``δ = (σ − intercept) / slope`` per atom.
    Atoms whose entry has no parseable σ are silently dropped.
    """
    out: dict[int, float] = {}
    for atom_idx, entry in by_atom.items():
        if entry.get("element") != element:
            continue
        sigma = entry.get("sigma_iso_avg_ppm")
        if not isinstance(sigma, (int, float)):
            continue
        out[int(atom_idx)] = predict_chemical_shift(
            float(sigma), slope=cal["slope"], intercept=cal["intercept"]
        )
    return out


def _shifts_by_atom_from_per_conformer(
    *,
    shieldings: list[dict[str, Any]],
    element: str,
    cal: dict[str, Any],
) -> dict[int, float]:
    """Same as :func:`_shifts_by_atom_from_by_atom` but operating on
    a single conformer's parsed shielding rows (raw σ per atom).

    Linear scaling commutes with averaging, so applying the
    calibration here (per conformer) is mathematically equivalent
    to applying it post-Boltzmann; we apply early because the
    equivalence detector wants δ for its data-aware refinement
    threshold (also in ppm).
    """
    out: dict[int, float] = {}
    for row in shieldings:
        if row.get("element") != element:
            continue
        sigma = row.get("sigma_iso_ppm")
        if not isinstance(sigma, (int, float)):
            continue
        out[int(row["atom_index"])] = predict_chemical_shift(
            float(sigma), slope=cal["slope"], intercept=cal["intercept"]
        )
    return out


def _calibrated_jhh_matrix_from_by_pair(
    by_pair: dict[tuple[int, int], dict[str, Any]],
    cal_jhh: Optional[dict[str, Any]],
) -> dict[tuple[int, int], float]:
    """Convert aggregator's averaged ``by_pair`` map to the
    canonical-pair-keyed J matrix the equivalence detector wants.

    Filters to ¹H–¹H pairs (the only ones our default coupling
    pipeline computes) and applies the Bally/Rablen scaling
    ``J_pred = slope · J_calc + intercept`` when available. When
    no calibration is loaded, raw J's are passed through.
    """
    out: dict[tuple[int, int], float] = {}
    for key, entry in by_pair.items():
        if not (
            entry.get("elem_i") == "H" and entry.get("elem_j") == "H"
        ):
            continue
        j_avg = entry.get("J_total_avg_hz")
        if not isinstance(j_avg, (int, float)):
            continue
        if cal_jhh is not None:
            out[key] = predict_coupling_constant(
                float(j_avg),
                slope=cal_jhh["slope"],
                intercept=cal_jhh["intercept"],
            )
        else:
            out[key] = float(j_avg)
    return out


def _calibrated_jhh_matrix_from_per_conformer(
    couplings: list[dict[str, Any]],
    cal_jhh: Optional[dict[str, Any]],
) -> dict[tuple[int, int], float]:
    """Per-conformer version of :func:`_calibrated_jhh_matrix_from_by_pair`.

    Operates on parsed coupling rows from one ORCA J-coupling output.
    """
    out: dict[tuple[int, int], float] = {}
    for row in couplings:
        if not (row.get("elem_i") == "H" and row.get("elem_j") == "H"):
            continue
        j_total = row.get("J_total_hz")
        if not isinstance(j_total, (int, float)):
            continue
        i, j = int(row["i"]), int(row["j"])
        key = (min(i, j), max(i, j))
        if cal_jhh is not None:
            out[key] = predict_coupling_constant(
                float(j_total),
                slope=cal_jhh["slope"],
                intercept=cal_jhh["intercept"],
            )
        else:
            out[key] = float(j_total)
    return out


def _spectrum_config_for(
    cfg: dict[str, Any], *, nucleus: str
) -> SpectrumConfig:
    """Build a :class:`SpectrumConfig` for the requested nucleus.

    The aggregator stores per-nucleus knobs as flat ``mnova_*_h`` /
    ``mnova_*_c`` keys for GUI ergonomics; this helper picks the
    right pair and packages them into the renderer's dataclass.
    """
    if nucleus == "1H":
        return SpectrumConfig(
            frequency_mhz=float(cfg["mnova_field_mhz_h"]),
            points=int(cfg["mnova_points"]),
            from_ppm=float(cfg["mnova_from_ppm_h"]),
            to_ppm=float(cfg["mnova_to_ppm_h"]),
            line_width_hz=float(cfg["mnova_line_width_hz_h"]),
        )
    if nucleus == "13C":
        return SpectrumConfig(
            frequency_mhz=float(cfg["mnova_field_mhz_c"]),
            points=int(cfg["mnova_points"]),
            from_ppm=float(cfg["mnova_from_ppm_c"]),
            to_ppm=float(cfg["mnova_to_ppm_c"]),
            line_width_hz=float(cfg["mnova_line_width_hz_c"]),
        )
    raise ValueError(f"_spectrum_config_for: unsupported nucleus {nucleus!r}")


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

        # mnova-spinsim XML emission knobs. SMILES is the topology
        # source the equivalence detector needs; if absent, the
        # emitter falls back to xyz perception (charge=0 assumed)
        # and skips XML emission only if both routes fail.
        smiles = normalize_optional_str(raw.get("smiles"))
        mnova_enabled = parse_bool(raw.get("mnova_enabled"), True)
        mnova_per_conformer = parse_bool(raw.get("mnova_per_conformer"), True)
        mnova_nuclei = (
            normalize_optional_str(raw.get("mnova_nuclei"))
            or DEFAULT_MNOVA_NUCLEI
        )
        # Validate: each token must be one of the supported nuclei.
        nuclei_tokens = [t.strip() for t in mnova_nuclei.split(",") if t.strip()]
        for tok in nuclei_tokens:
            if tok not in ("1H", "13C"):
                raise ValueError(
                    f"mnova_nuclei: unsupported token {tok!r}; "
                    f"expected one of '1H', '13C' (comma-separated)"
                )

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
            # mnova XML emission
            "smiles": smiles,
            "mnova_enabled": mnova_enabled,
            "mnova_per_conformer": mnova_per_conformer,
            "mnova_nuclei": ",".join(nuclei_tokens),
            "mnova_field_mhz_h": parse_float(
                raw.get("mnova_field_mhz_h"), DEFAULT_MNOVA_FIELD_MHZ_H
            ),
            "mnova_field_mhz_c": parse_float(
                raw.get("mnova_field_mhz_c"), DEFAULT_MNOVA_FIELD_MHZ_C
            ),
            "mnova_line_width_hz_h": parse_float(
                raw.get("mnova_line_width_hz_h"), DEFAULT_MNOVA_LINE_WIDTH_HZ_H
            ),
            "mnova_line_width_hz_c": parse_float(
                raw.get("mnova_line_width_hz_c"), DEFAULT_MNOVA_LINE_WIDTH_HZ_C
            ),
            "mnova_from_ppm_h": parse_float(
                raw.get("mnova_from_ppm_h"), DEFAULT_MNOVA_FROM_PPM_H
            ),
            "mnova_to_ppm_h": parse_float(
                raw.get("mnova_to_ppm_h"), DEFAULT_MNOVA_TO_PPM_H
            ),
            "mnova_from_ppm_c": parse_float(
                raw.get("mnova_from_ppm_c"), DEFAULT_MNOVA_FROM_PPM_C
            ),
            "mnova_to_ppm_c": parse_float(
                raw.get("mnova_to_ppm_c"), DEFAULT_MNOVA_TO_PPM_C
            ),
            "mnova_points": parse_int(
                raw.get("mnova_points"), DEFAULT_MNOVA_POINTS
            ),
            "mnova_tol_jcoupling_hz": parse_float(
                raw.get("mnova_tol_jcoupling_hz"),
                DEFAULT_MNOVA_TOL_JCOUPLING_HZ,
            ),
            "mnova_tol_shift_ppm": parse_float(
                raw.get("mnova_tol_shift_ppm"),
                DEFAULT_MNOVA_TOL_SHIFT_PPM,
            ),
            # Molecule-diagram artifacts (2D SVG + 3D HTML viewer).
            # Reuses ``mnova_nuclei`` to decide which nuclei to depict.
            "diagrams_enabled": parse_bool(raw.get("diagrams_enabled"), True),
            "diagrams_width": parse_int(
                raw.get("diagrams_width"), DEFAULT_DIAGRAM_WIDTH
            ),
            "diagrams_height": parse_int(
                raw.get("diagrams_height"), DEFAULT_DIAGRAM_HEIGHT
            ),
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

            # Shieldings live in the dedicated standalone NMR ``.out``
            # files (``orca_nmr_h.out`` for ¹H, ``orca_nmr_c.out`` for
            # ¹³C). Couplings live in ``orca_nmr_j.out``. These are
            # produced by orca_thermo_array as separate ORCA invocations
            # so the GIAO functionals (WP04, wB97X-D3, mPW1PW91) don't
            # inherit method-state flags from the wB97M-V SP that ran
            # earlier in the task. For backward compat with older runs
            # that used a single compound ``orca_thermo.out`` we fall
            # back to that file when the dedicated NMR outputs aren't
            # present, picking up shieldings/couplings from any
            # ``$new_job`` blocks the parser finds inside.
            sh: list[dict[str, Any]] = []
            for out_name in ("orca_nmr_h.out", "orca_nmr_c.out"):
                p = task_dir / out_name
                if p.exists():
                    sh.extend(parse_orca_shieldings(p))

            # Couplings: parse the dedicated ``orca_nmr_j.out`` if
            # present. Distinguish three failure modes for diagnostics:
            #   1. file missing       -> upstream workflow problem
            #   2. file present but parser returned []  -> parser problem
            #   3. file present but empty contents      -> ORCA crash
            # Case (2) is the one this run hit before the parser
            # rewrite; surface it as a structured failure so the next
            # regression doesn't slip past silently.
            cp: list[dict[str, Any]] = []
            j_out = task_dir / "orca_nmr_j.out"
            j_out_exists = j_out.exists()
            if j_out_exists and not cfg["skip_couplings"]:
                cp.extend(parse_orca_couplings(j_out))
                if not cp:
                    ctx.fail(
                        "coupling_parse_empty",
                        task_dir=str(task_dir),
                        file=str(j_out),
                        reason=(
                            "orca_nmr_j.out exists but parse_orca_couplings "
                            "returned zero rows — likely a parser/format "
                            "mismatch"
                        ),
                    )

            # Backward-compat: if no dedicated NMR outputs were found,
            # try the legacy single-compound ``orca_thermo.out`` shape.
            if not sh and not cp:
                thermo_out, _sp_out = pick_orca_outputs(task_dir)
                if thermo_out is None:
                    per_conformer_shieldings.append(None)
                    per_conformer_couplings.append(None)
                    continue
                sh = parse_orca_shieldings(thermo_out)
                if not cfg["skip_couplings"]:
                    cp = parse_orca_couplings(thermo_out)

            per_conformer_shieldings.append(sh or None)
            if cfg["skip_couplings"]:
                per_conformer_couplings.append(None)
            else:
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
        # Per-element scaled counts — distinguish "calibration table
        # entry exists" from "calibration was actually applied to a
        # row". The old ``calibration_*_used = bool(cal_*)`` shape
        # produced misleading summaries when cal_jhh existed but zero
        # coupling rows were parsed.
        n_h_shifts_scaled = sum(
            1 for entry in by_atom.values()
            if entry["element"] == "H" and cal_h is not None
        )
        n_c_shifts_scaled = sum(
            1 for entry in by_atom.values()
            if entry["element"] == "C" and cal_c is not None
        )
        n_h_atoms = sum(
            1 for entry in by_atom.values() if entry["element"] == "H"
        )

        ctx.add_artifact(
            "files",
            {
                "label": "predicted_shifts_csv",
                "path_abs": str(shifts_path.resolve()),
                "sha256": sha256_file(shifts_path),
                "format": "csv",
                "row_count": len(by_atom),
                "h_row_count": n_h_atoms,
                "c_row_count": sum(
                    1 for e in by_atom.values() if e["element"] == "C"
                ),
            },
        )

        n_used_cp = 0
        n_hh_pairs = 0
        n_hh_pairs_scaled = 0
        by_pair: dict[tuple[int, int], dict[str, Any]] = {}
        if not cfg["skip_couplings"]:
            by_pair, n_used_cp = boltzmann_average_couplings(
                per_conformer=per_conformer_couplings,
                weights=norm_weights,
            )
            hh_pairs = {
                k: v for k, v in by_pair.items()
                if v["elem_i"] == "H" and v["elem_j"] == "H"
            }
            n_hh_pairs = len(hh_pairs)
            n_hh_pairs_scaled = n_hh_pairs if cal_jhh is not None else 0

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
                    "row_count": len(by_pair),
                    "hh_row_count": n_hh_pairs,
                },
            )

            # Failure contract: when the user asked for couplings AND
            # the molecule has at least 2 H atoms, an empty H-H table
            # is a real failure — either the parser missed the format
            # or the ORCA jobs didn't actually compute couplings. At
            # least one short-range H-H pair will always exist in any
            # molecule small enough to do NMR on, so n_hh_pairs == 0
            # under those conditions reliably indicates breakage.
            #
            # A *sparse* H-H table (some pairs present, but fewer than
            # the complete-graph count of n_h*(n_h-1)/2) is normal:
            # ORCA's SpinSpinRThresh caps the inter-nucleus distance
            # for which couplings are computed, and J's outside that
            # cutoff are negligible enough to not need a row. We do
            # NOT fail in that case — the sparse table is the design.
            if n_h_atoms >= 2 and n_hh_pairs == 0:
                ctx.fail(
                    "no_hh_couplings_parsed",
                    n_h_atoms=n_h_atoms,
                    n_conformers_with_couplings=n_used_cp,
                    expected_reason=(
                        "skip_couplings=false and >=2 H atoms present"
                    ),
                )
            else:
                # Informational only — surface what fraction of the
                # complete H-H graph actually came back so the operator
                # has a sanity-check signal in the logs without it
                # turning into a manifest-level failure.
                expected_hh_pairs = n_h_atoms * (n_h_atoms - 1) // 2
                logging_utils.log_info(
                    f"nmr-aggregate: {n_hh_pairs} H-H pair(s) parsed "
                    f"out of {expected_hh_pairs} possible "
                    f"(SpinSpinRThresh excludes long-range pairs by design)"
                )

        summary = {
            "n_conformers_total": len(confs),
            "n_conformers_with_shielding": n_used_sh,
            "n_conformers_with_couplings": n_used_cp,
            "n_atoms": len(by_atom),
            "n_h_atoms": n_h_atoms,
            "n_hh_pairs": n_hh_pairs,
            "n_h_shifts_scaled": n_h_shifts_scaled,
            "n_c_shifts_scaled": n_c_shifts_scaled,
            "n_hh_pairs_scaled": n_hh_pairs_scaled,
            # ``_found``: a calibration entry exists in the table.
            # ``_used``: that entry was actually applied to at least
            # one output row. The two diverge whenever an upstream
            # parser / data-availability problem leaves the input
            # table empty for a nucleus class.
            "calibration_h_found": cal_h is not None,
            "calibration_h_used": n_h_shifts_scaled > 0,
            "calibration_c_found": cal_c is not None,
            "calibration_c_used": n_c_shifts_scaled > 0,
            "calibration_jhh_found": cal_jhh is not None,
            "calibration_jhh_used": n_hh_pairs_scaled > 0,
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

        # ---- mnova-spinsim XML + diagram artifacts (optional) ----
        if cfg["mnova_enabled"] or cfg["diagrams_enabled"]:
            self._emit_visualizations(
                ctx=ctx,
                cfg=cfg,
                confs=confs,
                norm_weights=norm_weights,
                per_conformer_shieldings=per_conformer_shieldings,
                per_conformer_couplings=per_conformer_couplings,
                by_atom=by_atom,
                by_pair=by_pair,
                cal_h=cal_h,
                cal_c=cal_c,
                cal_jhh=cal_jhh,
            )

    # ------------------------------------------------------------------
    # Visualization emission (mnova XML + molecule diagrams)
    # ------------------------------------------------------------------

    def _emit_visualizations(
        self,
        *,
        ctx: NodeContext,
        cfg: dict[str, Any],
        confs: list[dict[str, Any]],
        norm_weights: list[Optional[float]],
        per_conformer_shieldings: list[Optional[list[dict[str, Any]]]],
        per_conformer_couplings: list[Optional[list[dict[str, Any]]]],
        by_atom: dict[int, dict[str, Any]],
        by_pair: dict[tuple[int, int], dict[str, Any]],
        cal_h: Optional[dict[str, Any]],
        cal_c: Optional[dict[str, Any]],
        cal_jhh: Optional[dict[str, Any]],
    ) -> None:
        """Render mnova XMLs + molecule-diagram artifacts.

        Both visualization paths share the same equivalence-group
        computation (one call per nucleus), so we keep them in one
        method to avoid recomputing the structural classification.
        Each output type is independently gated:

        * ``mnova_enabled`` controls XML emission (pre-averaged + the
          optional per-conformer file when ``mnova_per_conformer``).
        * ``diagrams_enabled`` controls the SVG + HTML diagrams.

        Failure-mode contract:

        * ``mnova_xml_skipped:topology_unavailable`` — neither the
          provided SMILES nor the xyz fallback yields a usable Mol.
          ALL visualization artifacts are skipped (XML and diagrams
          both need the Mol). The CSV path keeps running.
        * Per-nucleus skips are silent (logged at INFO level): a
          molecule with no H atoms simply doesn't get ¹H artifacts;
          a missing calibration entry blocks XML for that nucleus
          but the SVG/HTML diagrams emit raw shifts so you still get
          a structure visualization.

        Per-conformer SVG/HTML diagrams are NOT emitted (would
        produce N files per nucleus for N conformers — too noisy).
        Diagrams use Boltzmann-averaged shifts only.
        """
        nuclei = [t for t in cfg["mnova_nuclei"].split(",") if t]

        # Build the Mol once — equivalence detection is purely structural,
        # so the same Mol works for both ¹H and ¹³C and for both XML
        # and diagram emission.
        mol = self._build_mnova_mol(cfg, confs)
        if mol is None:
            ctx.fail(
                "mnova_xml_skipped",
                reason="topology_unavailable",
                detail=(
                    "neither smiles= nor xyz fallback produced a usable Mol; "
                    "set smiles=... in node config to enable mnova XML / "
                    "diagram emission"
                ),
            )
            return

        # XYZ text for the 3D HTML viewer. Drawn from the first
        # conformer with a readable xyz at path_abs (any conformer
        # works — the molecule is the same; only the geometry differs
        # per conformer and any geometry visualizes the structure).
        # ``None`` means HTML emission is silently skipped; SVG still
        # works since it computes its own 2D layout.
        xyz_text = self._first_conformer_xyz_text(confs)

        outputs_dir = ctx.outputs_dir
        outputs_dir.mkdir(parents=True, exist_ok=True)

        for nuc in nuclei:
            element, cal_shift = ("H", cal_h) if nuc == "1H" else ("C", cal_c)
            # Skip silently when the molecule has no atoms of this
            # nucleus (e.g., a perfluorinated molecule asking for ¹H).
            has_atoms = any(
                entry.get("element") == element for entry in by_atom.values()
            )
            if not has_atoms:
                continue
            if cal_shift is None:
                # The CSV path already surfaces calibration_not_found
                # for nuclei that have atoms in the data — no need to
                # double-fire from the XML/diagram paths. Just log + skip.
                logging_utils.log_warn(
                    f"nmr-aggregate: skipping {nuc} visualizations "
                    f"(no calibration entry for {nuc})"
                )
                continue

            # Pre-averaged groups (shared between XML and diagrams).
            avg_shifts = _shifts_by_atom_from_by_atom(
                by_atom=by_atom, element=element, cal=cal_shift
            )
            if not avg_shifts:
                # Defensive: has_atoms said we have atoms of this
                # element, but the calibration filter dropped all of
                # them (shouldn't happen unless the dict shape is bad).
                logging_utils.log_info(
                    f"nmr-aggregate: skipping {nuc} visualizations "
                    f"(no {element} atoms with parseable shielding)"
                )
                continue
            j_matrix_avg = (
                _calibrated_jhh_matrix_from_by_pair(by_pair, cal_jhh)
                if nuc == "1H"
                else {}
            )
            avg_groups = compute_equivalence_groups(
                mol=mol,
                element=element,
                shifts_by_atom=avg_shifts,
                j_matrix=j_matrix_avg,
                tol_jcoupling_hz=cfg["mnova_tol_jcoupling_hz"],
                tol_shift_ppm=cfg["mnova_tol_shift_ppm"],
            )
            if not avg_groups:
                continue

            # ---- mnova XML (gated by mnova_enabled) ----
            if cfg["mnova_enabled"]:
                self._emit_mnova_xml_for_nucleus(
                    ctx=ctx,
                    cfg=cfg,
                    nucleus=nuc,
                    element=element,
                    avg_groups=avg_groups,
                    confs=confs,
                    norm_weights=norm_weights,
                    per_conformer_shieldings=per_conformer_shieldings,
                    per_conformer_couplings=per_conformer_couplings,
                    cal_shift=cal_shift,
                    cal_jhh=cal_jhh,
                    mol=mol,
                    outputs_dir=outputs_dir,
                )

            # ---- Molecule diagrams (gated by diagrams_enabled) ----
            if cfg["diagrams_enabled"]:
                self._emit_diagrams_for_nucleus(
                    ctx=ctx,
                    cfg=cfg,
                    nucleus=nuc,
                    avg_groups=avg_groups,
                    mol=mol,
                    xyz_text=xyz_text,
                    outputs_dir=outputs_dir,
                )

    def _emit_mnova_xml_for_nucleus(
        self,
        *,
        ctx: NodeContext,
        cfg: dict[str, Any],
        nucleus: str,
        element: str,
        avg_groups: list[EquivalenceGroup],
        confs: list[dict[str, Any]],
        norm_weights: list[Optional[float]],
        per_conformer_shieldings: list[Optional[list[dict[str, Any]]]],
        per_conformer_couplings: list[Optional[list[dict[str, Any]]]],
        cal_shift: dict[str, Any],
        cal_jhh: Optional[dict[str, Any]],
        mol: Any,
        outputs_dir: Path,
    ) -> None:
        """Render and write the mnova XML(s) for one nucleus.

        Always emits the pre-averaged file. Optionally emits the
        per-conformer file when ``mnova_per_conformer`` is true and
        any conformer has usable data.
        """
        avg_system = SpinSystem(groups=tuple(avg_groups), population=1.0)
        spec = _spectrum_config_for(cfg, nucleus=nucleus)

        xml_text = render_mnova_xml([avg_system], spectrum=spec)
        file_path = outputs_dir / DEFAULT_MNOVA_FILENAME_FMT.format(
            nucleus=nucleus.lower()
        )
        file_path.write_text(xml_text, encoding="utf-8")
        ctx.add_artifact(
            "files",
            {
                "label": f"predicted_mnova_{nucleus.lower()}",
                "path_abs": str(file_path.resolve()),
                "sha256": sha256_file(file_path),
                "format": "xml",
                "n_groups": len(avg_groups),
                "nucleus": nucleus,
                "mode": "pre_averaged",
            },
        )
        logging_utils.log_info(
            f"nmr-aggregate: wrote {file_path.name} "
            f"({len(avg_groups)} group(s))"
        )

        if not cfg["mnova_per_conformer"]:
            return
        per_systems = self._build_per_conformer_systems(
            mol=mol,
            element=element,
            nucleus=nucleus,
            cfg=cfg,
            confs=confs,
            norm_weights=norm_weights,
            per_conformer_shieldings=per_conformer_shieldings,
            per_conformer_couplings=per_conformer_couplings,
            cal_shift=cal_shift,
            cal_jhh=cal_jhh,
        )
        if not per_systems:
            logging_utils.log_info(
                f"nmr-aggregate: skipping {nucleus} per-conformer mnova XML "
                f"(no conformers with usable data)"
            )
            return
        per_xml = render_mnova_xml(per_systems, spectrum=spec)
        per_path = (
            outputs_dir
            / DEFAULT_MNOVA_PER_CONFORMER_FILENAME_FMT.format(
                nucleus=nucleus.lower()
            )
        )
        per_path.write_text(per_xml, encoding="utf-8")
        ctx.add_artifact(
            "files",
            {
                "label": f"predicted_mnova_{nucleus.lower()}_per_conformer",
                "path_abs": str(per_path.resolve()),
                "sha256": sha256_file(per_path),
                "format": "xml",
                "n_spin_systems": len(per_systems),
                "nucleus": nucleus,
                "mode": "per_conformer",
            },
        )
        logging_utils.log_info(
            f"nmr-aggregate: wrote {per_path.name} "
            f"({len(per_systems)} conformer(s))"
        )

    def _emit_diagrams_for_nucleus(
        self,
        *,
        ctx: NodeContext,
        cfg: dict[str, Any],
        nucleus: str,
        avg_groups: list[EquivalenceGroup],
        mol: Any,
        xyz_text: Optional[str],
        outputs_dir: Path,
    ) -> None:
        """Write the SVG (always) and HTML (if xyz available) diagrams.

        SVG comes from RDKit's :mod:`rdMolDraw2D` and uses a fresh 2D
        layout, so it works whether or not we have 3D coords. HTML
        needs a real xyz to drive the 3Dmol.js viewer; if no
        conformer's xyz is readable, the HTML is silently skipped
        (just logged at INFO level).
        """
        title = f"Predicted {nucleus} NMR shifts"
        # SVG.
        svg_text = render_shift_svg(
            mol=mol,
            groups=avg_groups,
            width=int(cfg["diagrams_width"]),
            height=int(cfg["diagrams_height"]),
        )
        svg_path = outputs_dir / DEFAULT_DIAGRAM_SVG_FILENAME_FMT.format(
            nucleus=nucleus.lower()
        )
        svg_path.write_text(svg_text, encoding="utf-8")
        ctx.add_artifact(
            "files",
            {
                "label": f"predicted_structure_{nucleus.lower()}_svg",
                "path_abs": str(svg_path.resolve()),
                "sha256": sha256_file(svg_path),
                "format": "svg",
                "n_groups": len(avg_groups),
                "nucleus": nucleus,
            },
        )
        logging_utils.log_info(f"nmr-aggregate: wrote {svg_path.name}")

        # HTML (3Dmol.js). Needs a real xyz; silently skip if absent.
        if not xyz_text:
            logging_utils.log_info(
                f"nmr-aggregate: skipping {nucleus} HTML viewer "
                f"(no readable xyz on any conformer's path_abs)"
            )
            return
        html_text = render_shift_html(
            groups=avg_groups,
            xyz_text=xyz_text,
            title=title,
            width=int(cfg["diagrams_width"]),
            height=int(cfg["diagrams_height"]),
        )
        html_path = outputs_dir / DEFAULT_DIAGRAM_HTML_FILENAME_FMT.format(
            nucleus=nucleus.lower()
        )
        html_path.write_text(html_text, encoding="utf-8")
        ctx.add_artifact(
            "files",
            {
                "label": f"predicted_structure_{nucleus.lower()}_html",
                "path_abs": str(html_path.resolve()),
                "sha256": sha256_file(html_path),
                "format": "html",
                "n_groups": len(avg_groups),
                "nucleus": nucleus,
            },
        )
        logging_utils.log_info(f"nmr-aggregate: wrote {html_path.name}")

    def _first_conformer_xyz_text(
        self, confs: list[dict[str, Any]]
    ) -> Optional[str]:
        """Return the first readable xyz text from the conformer list.

        Used as the 3D-coordinate source for :func:`render_shift_html`.
        Any conformer's xyz is fine since they're all the same molecule
        — geometry differs per conformer but the visual structure is
        the same. Returns ``None`` if no conformer has a readable xyz
        (e.g., the legacy test fixture where ``path_abs`` is a directory
        rather than a file).
        """
        for c in confs:
            path = c.get("path_abs")
            if not isinstance(path, str):
                continue
            try:
                text = Path(path).read_text(encoding="utf-8")
            except Exception:
                continue
            # Sanity: an xyz starts with an integer atom count.
            head = text.lstrip().splitlines()[:1]
            if not head:
                continue
            try:
                int(head[0].split()[0])
            except (ValueError, IndexError):
                continue
            return text
        return None

    def _build_mnova_mol(
        self, cfg: dict[str, Any], confs: list[dict[str, Any]]
    ) -> Optional[Any]:
        """Build the RDKit Mol used for equivalence detection.

        Tries SMILES first (clean canonical structure, deterministic
        atom ordering matched with smiles_to_3d's xyz). Falls back to
        xyz perception via :func:`mol_from_smiles_or_xyz` on any
        conformer's staged xyz (atom indices match ORCA's output by
        construction). Returns ``None`` only if both routes fail.
        """
        smi = cfg.get("smiles")
        if smi:
            mol = mol_from_smiles_or_xyz(smiles=smi)
            if mol is not None:
                return mol
            logging_utils.log_warn(
                "nmr-aggregate: SMILES failed to parse; falling back to xyz"
            )

        for c in confs:
            path = c.get("path_abs")
            if not isinstance(path, str):
                continue
            try:
                xyz_text = Path(path).read_text(encoding="utf-8")
            except Exception:
                continue
            mol = mol_from_smiles_or_xyz(xyz_text=xyz_text, charge=0)
            if mol is not None:
                return mol
        return None

    def _build_per_conformer_systems(
        self,
        *,
        mol: Any,
        element: str,
        nucleus: str,
        cfg: dict[str, Any],
        confs: list[dict[str, Any]],
        norm_weights: list[Optional[float]],
        per_conformer_shieldings: list[Optional[list[dict[str, Any]]]],
        per_conformer_couplings: list[Optional[list[dict[str, Any]]]],
        cal_shift: dict[str, Any],
        cal_jhh: Optional[dict[str, Any]],
    ) -> list[SpinSystem]:
        """Build one :class:`SpinSystem` per conformer with weight populations.

        Conformers with ``None`` weight or empty shielding data are
        skipped. The returned list keeps source-conformer order
        (the orchestrator stage 5 sort happens INSIDE
        :func:`compute_equivalence_groups` via min atom index).
        """
        systems: list[SpinSystem] = []
        for conf, w, shieldings, couplings in zip(
            confs,
            norm_weights,
            per_conformer_shieldings,
            per_conformer_couplings,
        ):
            if not isinstance(w, (int, float)):
                continue
            if not shieldings:
                continue
            shifts = _shifts_by_atom_from_per_conformer(
                shieldings=shieldings, element=element, cal=cal_shift
            )
            if not shifts:
                continue
            j_matrix = (
                _calibrated_jhh_matrix_from_per_conformer(
                    couplings or [], cal_jhh
                )
                if nucleus == "1H"
                else {}
            )
            groups = compute_equivalence_groups(
                mol=mol,
                element=element,
                shifts_by_atom=shifts,
                j_matrix=j_matrix,
                tol_jcoupling_hz=cfg["mnova_tol_jcoupling_hz"],
                tol_shift_ppm=cfg["mnova_tol_shift_ppm"],
            )
            if not groups:
                continue
            systems.append(
                SpinSystem(groups=tuple(groups), population=float(w))
            )
        return systems


__all__ = [
    "COUPLING_CSV_COLUMNS",
    "DEFAULT_COUPLING_BASIS",
    "DEFAULT_COUPLING_METHOD",
    "DEFAULT_DIAGRAM_HEIGHT",
    "DEFAULT_DIAGRAM_HTML_FILENAME_FMT",
    "DEFAULT_DIAGRAM_SVG_FILENAME_FMT",
    "DEFAULT_DIAGRAM_WIDTH",
    "DEFAULT_MNOVA_FIELD_MHZ_C",
    "DEFAULT_MNOVA_FIELD_MHZ_H",
    "DEFAULT_MNOVA_FILENAME_FMT",
    "DEFAULT_MNOVA_FROM_PPM_C",
    "DEFAULT_MNOVA_FROM_PPM_H",
    "DEFAULT_MNOVA_LINE_WIDTH_HZ_C",
    "DEFAULT_MNOVA_LINE_WIDTH_HZ_H",
    "DEFAULT_MNOVA_NUCLEI",
    "DEFAULT_MNOVA_PER_CONFORMER_FILENAME_FMT",
    "DEFAULT_MNOVA_POINTS",
    "DEFAULT_MNOVA_TO_PPM_C",
    "DEFAULT_MNOVA_TO_PPM_H",
    "DEFAULT_MNOVA_TOL_JCOUPLING_HZ",
    "DEFAULT_MNOVA_TOL_SHIFT_PPM",
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
