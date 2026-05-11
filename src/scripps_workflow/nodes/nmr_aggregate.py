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
    mnova_field_mhz_h        [600.15]  ¹H Larmor frequency.
    mnova_field_mhz_c        [150.94]  ¹³C Larmor frequency.
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
    mnova_j_round_threshold_hz [0.25]  any predicted |J| ≤ this gets
                                       set to 0 before emission.
                                       Cleans near-zero noise from
                                       the simulated spectrum.

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
from ..config_schema import ConfigField, NodeSchema, apply_schema
from ..equivalence import (
    EquivalenceGroup,
    compute_equivalence_groups,
    mol_from_smiles_or_xyz,
)
from ..hashing import sha256_file
from ..mnova_xml import SpectrumConfig, SpinSystem, render_mnova_xml
from ..molecule_diagram import render_shift_html, render_shift_svg
from ..nmr_calibration import (
    canonical_j_nucleus_label,
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

#: Larmor frequency presets in MHz for the lab's 600 MHz spectrometer
#: (¹H Larmor ≈ field × 42.577; ¹³C ≈ field × 10.708). Override when
#: the user's spectrometer is different (e.g., 400 MHz → 400.13 / 100.61).
DEFAULT_MNOVA_FIELD_MHZ_H: float = 600.15
DEFAULT_MNOVA_FIELD_MHZ_C: float = 150.94

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

#: J-coupling round-down threshold (Hz). After calibration, any J with
#: |J| ≤ this gets snapped to 0 before being emitted to mnova XML or
#: used in equivalence detection. Removes near-zero noise (typically
#: long-range / through-space artifacts) that would otherwise clutter
#: the simulated spectrum with sub-linewidth splittings. Set to 0 to
#: disable the round-down entirely.
DEFAULT_MNOVA_J_ROUND_THRESHOLD_HZ: float = 0.25

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


def _calibrated_j_matrix_from_by_pair(
    by_pair: dict[tuple[int, int], dict[str, Any]],
    *,
    allowed_element_pairs: Optional[set[tuple[str, str]]] = None,
    cals_by_label: dict[str, Optional[dict[str, Any]]],
) -> dict[tuple[int, int], float]:
    """Convert aggregator's averaged ``by_pair`` map to the
    canonical-pair-keyed J matrix the equivalence detector wants,
    with per-element-pair calibration dispatch.

    ``allowed_element_pairs`` is a set of frozen element pairs (each
    a 2-tuple of element symbols, in either order) — pairs whose
    elements aren't in any permitted pair are filtered out.
    ``None`` (the default) means "include every pair regardless of
    element".

    ``cals_by_label`` maps canonical-isotope-pair labels (the output
    of :func:`canonical_j_nucleus_label`, e.g., ``"1H-1H_J"``,
    ``"1H-19F_J"``) to calibration entries (slope/intercept dicts) or
    ``None`` for "no calibration loaded → pass raw J through".

    Implementation note: ``allowed_element_pairs`` is checked using
    canonical (sorted-by-mass) pair labels so the caller can pass
    ``{("H", "F")}`` and we'll match both orientations of the
    underlying coupling row.
    """
    out: dict[tuple[int, int], float] = {}
    allowed_labels: Optional[set[str]] = None
    if allowed_element_pairs is not None:
        allowed_labels = {
            canonical_j_nucleus_label(a, b) for (a, b) in allowed_element_pairs
        }

    for key, entry in by_pair.items():
        ei = entry.get("elem_i")
        ej = entry.get("elem_j")
        if ei is None or ej is None:
            continue
        label = canonical_j_nucleus_label(ei, ej)
        if allowed_labels is not None and label not in allowed_labels:
            continue
        j_avg = entry.get("J_total_avg_hz")
        if not isinstance(j_avg, (int, float)):
            continue
        cal = cals_by_label.get(label)
        if cal is not None:
            out[key] = predict_coupling_constant(
                float(j_avg),
                slope=cal["slope"],
                intercept=cal["intercept"],
            )
        else:
            out[key] = float(j_avg)
    return out


def _calibrated_j_matrix_from_per_conformer(
    couplings: list[dict[str, Any]],
    *,
    allowed_element_pairs: Optional[set[tuple[str, str]]] = None,
    cals_by_label: dict[str, Optional[dict[str, Any]]],
) -> dict[tuple[int, int], float]:
    """Per-conformer version of :func:`_calibrated_j_matrix_from_by_pair`."""
    out: dict[tuple[int, int], float] = {}
    allowed_labels: Optional[set[str]] = None
    if allowed_element_pairs is not None:
        allowed_labels = {
            canonical_j_nucleus_label(a, b) for (a, b) in allowed_element_pairs
        }

    for row in couplings:
        ei = row.get("elem_i")
        ej = row.get("elem_j")
        if ei is None or ej is None:
            continue
        label = canonical_j_nucleus_label(ei, ej)
        if allowed_labels is not None and label not in allowed_labels:
            continue
        j_total = row.get("J_total_hz")
        if not isinstance(j_total, (int, float)):
            continue
        i, j = int(row["i"]), int(row["j"])
        key = (min(i, j), max(i, j))
        cal = cals_by_label.get(label)
        if cal is not None:
            out[key] = predict_coupling_constant(
                float(j_total),
                slope=cal["slope"],
                intercept=cal["intercept"],
            )
        else:
            out[key] = float(j_total)
    return out


def _round_small_js_to_zero(
    j_matrix: dict[tuple[int, int], float],
    threshold_hz: float,
) -> dict[tuple[int, int], float]:
    """Snap |J| ≤ ``threshold_hz`` to 0 in place-equivalent fashion.

    Returns a new dict; doesn't mutate the input. Threshold ≤ 0
    disables the round-down (passes everything through unchanged).
    """
    if threshold_hz <= 0:
        return dict(j_matrix)
    return {
        key: (0.0 if abs(j) <= threshold_hz else j)
        for key, j in j_matrix.items()
    }


def _parse_partner_list(raw: Any) -> list[str]:
    """Parse the ``mnova_heteronuclear_partners`` config value.

    Accepts a list (from JSON config) or a comma-separated string
    (from key=value tokens). Strips whitespace, drops empty tokens,
    and validates that each entry is a known element symbol the rest
    of the pipeline can canonicalize. Unknown elements raise so a
    typo (e.g., ``"Fl"`` for fluorine) surfaces as ``argv_parse_failed``
    rather than silently producing wrong-element groups.
    """
    if raw is None or raw == "":
        return []
    if isinstance(raw, (list, tuple)):
        items = [str(x).strip() for x in raw]
    else:
        items = [s.strip() for s in str(raw).split(",")]
    out: list[str] = []
    valid = {"H", "C", "N", "O", "F", "P", "Si"}
    for item in items:
        if not item:
            continue
        if item not in valid:
            # No field-name prefix here — apply_schema (and the legacy
            # caller) wrap with the field name, so a self-prefix would
            # double up to ``foo: foo: unknown element...``.
            raise ValueError(
                f"unknown element {item!r}; "
                f"expected one of {sorted(valid)}"
            )
        if item not in out:
            out.append(item)
    return out


def _build_pair_set_and_cals(
    *,
    primary_element: str,
    partner_elements: list[str],
    cfg: dict[str, Any],
) -> tuple[set[tuple[str, str]], dict[str, Optional[dict[str, Any]]]]:
    """Compute the J-coupling pair filter and per-pair calibration map.

    For a spin-system that includes ``primary_element`` plus zero or
    more ``partner_elements``, generate every distinct unordered
    element-pair (including homonuclear within partners, e.g. F-F)
    and look up the matching calibration entry in NMR_CALIBRATION.

    Pairs without a calibration entry land in the dict as ``None``,
    which the J-matrix builders interpret as "pass raw J through".
    Calibration lookups use the aggregator's ``coupling_method`` /
    ``coupling_basis`` / ``solvent`` config triple — heteronuclear
    pairs assume the same DFT method/basis as the homonuclear pair,
    which is what ORCA actually computes in a single J-coupling job.
    """
    elements_in_xml = [primary_element] + list(partner_elements)
    pair_set: set[tuple[str, str]] = set()
    cals_by_label: dict[str, Optional[dict[str, Any]]] = {}
    for ea in elements_in_xml:
        for eb in elements_in_xml:
            # Canonical (sorted-by-mass) tuple so we don't double-add
            # (H,F) and (F,H).
            from ..nmr_calibration import _MASS_BY_ELEMENT
            ma = _MASS_BY_ELEMENT.get(ea, 999)
            mb = _MASS_BY_ELEMENT.get(eb, 999)
            canonical_pair = (ea, eb) if ma <= mb else (eb, ea)
            if canonical_pair in pair_set:
                continue
            pair_set.add(canonical_pair)
            label = canonical_j_nucleus_label(*canonical_pair)
            cal = lookup_calibration(
                functional=cfg["coupling_method"],
                basis=cfg["coupling_basis"],
                solvent=cfg["solvent"],
                nucleus=label,
            )
            cals_by_label[label] = cal
    return pair_set, cals_by_label


def _partner_shifts_by_atom(
    *,
    mol: Any,
    partner_elements: list[str],
    by_atom: Optional[dict[int, dict[str, Any]]],
    stub_ppm: float,
) -> dict[int, float]:
    """Fetch placeholder chemical shifts for partner-element atoms.

    For each atom of a partner element in ``mol``:

    * If ``by_atom`` has a parsed σ for that atom (i.e., the upstream
      ORCA job ran shielding on that nucleus), use the raw σ as a
      placeholder δ. This is wrong as published-ppm chemistry but
      lands the atom far outside the primary nucleus's window in
      mnova's simulated spectrum (e.g., a typical ¹⁹F σ of 200-400
      ppm sits well outside the 0-12 ppm ¹H window).
    * Otherwise, use ``stub_ppm`` (default 100 ppm — also outside
      typical ¹H or ¹³C windows). Each partner element gets a
      slightly different stub via element-index offset so atoms
      within a partner-element class still register as equivalent
      while different partner elements are visually distinguishable.

    Iteration 1 limitation: this is intentionally NOT a real shift
    prediction. To get correct ¹⁹F δ values, add a 19F shielding
    calibration entry and a ``shielding_method_f`` / ``shielding_basis_f``
    config path (similar to the existing H/C plumbing). For now the
    placeholder is enough to make the J-coupling-driven splitting
    appear in the primary nucleus's spectrum.
    """
    shifts: dict[int, float] = {}
    for elem_idx, elem in enumerate(partner_elements):
        # Element-specific offset so multiple partner elements don't
        # all stack at the same stub value (visually distinguishable
        # in the rendered spectrum even though both are out-of-window).
        elem_stub = stub_ppm + 50.0 * elem_idx
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != elem:
                continue
            atom_idx = atom.GetIdx()
            entry = by_atom.get(atom_idx) if by_atom else None
            if entry and isinstance(entry.get("sigma_iso_avg_ppm"), (int, float)):
                shifts[atom_idx] = float(entry["sigma_iso_avg_ppm"])
            else:
                shifts[atom_idx] = elem_stub
    return shifts


def _partner_shifts_per_conformer(
    *,
    mol: Any,
    partner_elements: list[str],
    shieldings: list[dict[str, Any]],
    stub_ppm: float,
) -> dict[int, float]:
    """Per-conformer version of :func:`_partner_shifts_by_atom`.

    Builds an atom_idx → σ map from a single conformer's parsed
    shielding rows for the requested partner elements; falls back to
    ``stub_ppm`` for partner atoms whose σ wasn't parsed.
    """
    by_idx_sigma: dict[int, float] = {}
    for row in shieldings:
        sym = row.get("element")
        if sym not in partner_elements:
            continue
        sigma = row.get("sigma_iso_ppm")
        if isinstance(sigma, (int, float)):
            by_idx_sigma[int(row["atom_index"])] = float(sigma)

    shifts: dict[int, float] = {}
    for elem_idx, elem in enumerate(partner_elements):
        elem_stub = stub_ppm + 50.0 * elem_idx
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != elem:
                continue
            atom_idx = atom.GetIdx()
            shifts[atom_idx] = by_idx_sigma.get(atom_idx, elem_stub)
    return shifts


def _build_multinucleus_groups(
    *,
    mol: Any,
    primary_element: str,
    partner_elements: list[str],
    primary_shifts: dict[int, float],
    partner_shifts: dict[int, float],
    j_matrix: dict[tuple[int, int], float],
    tol_jcoupling_hz: float,
    tol_shift_ppm: float,
) -> list[EquivalenceGroup]:
    """Build one merged spin-system covering primary + partner elements.

    Calls :func:`compute_equivalence_groups` once per element (so each
    element's equivalence dispatch is independent — primary nucleus's
    AA'BB' tier classification doesn't get confused by cross-element
    coupling, and vice versa). Then merges the per-element group
    lists, sorts by minimum atom index, re-assigns Excel-style names
    A-Z over the combined list, and re-derives every inter-group J
    coupling from the supplied ``j_matrix``.

    The j_matrix passed here SHOULD include all relevant pair types
    (primary-primary, primary-partner, partner-partner) — typically
    built via :func:`_calibrated_j_matrix_from_by_pair` with an
    ``allowed_element_pairs`` set that covers every pair from
    :func:`_build_pair_set_and_cals`.

    When ``partner_elements`` is empty this is equivalent to a single
    :func:`compute_equivalence_groups` call (no merge step needed).

    Iteration-1 caveat: per-element compute_equivalence_groups calls
    pass the FULL j_matrix, but the magnetic-equivalence test inside
    only uses J's to atoms of the same element (since ``other_atoms``
    is restricted to same-nucleus atoms by the orchestrator). So an
    AA'XX' pattern where the X's are partner atoms (e.g., two H's
    coupled differently to two F's) will be wrongly classified as
    HARD. Fix in iteration 2 by adding a ``cross_element_other_atoms``
    parameter to compute_equivalence_groups.
    """
    primary_groups = compute_equivalence_groups(
        mol=mol,
        element=primary_element,
        shifts_by_atom=primary_shifts,
        j_matrix=j_matrix,
        tol_jcoupling_hz=tol_jcoupling_hz,
        tol_shift_ppm=tol_shift_ppm,
    )
    if not partner_elements:
        return primary_groups

    partner_groups: list[EquivalenceGroup] = []
    for elem in partner_elements:
        # Filter partner_shifts to just this element's atoms.
        elem_shifts = {
            idx: val
            for idx, val in partner_shifts.items()
            if mol.GetAtomWithIdx(idx).GetSymbol() == elem
        }
        if not elem_shifts:
            continue
        partner_groups.extend(
            compute_equivalence_groups(
                mol=mol,
                element=elem,
                shifts_by_atom=elem_shifts,
                j_matrix=j_matrix,
                tol_jcoupling_hz=tol_jcoupling_hz,
                tol_shift_ppm=tol_shift_ppm,
            )
        )

    if not partner_groups:
        return primary_groups

    # Merge + sort by min atom index for stable A-Z labeling.
    combined: list[EquivalenceGroup] = list(primary_groups) + partner_groups
    combined.sort(key=lambda g: min(g.atom_indices))
    new_names = _assign_group_labels_local(len(combined))

    # Re-derive J couplings between all groups using the new names.
    relabeled: list[EquivalenceGroup] = []
    for new_name, g in zip(new_names, combined):
        new_j: dict[str, float] = {}
        for other_new_name, other_g in zip(new_names, combined):
            if new_name == other_new_name:
                continue
            j_avg = _avg_pairwise_j_local(
                g.atom_indices, other_g.atom_indices, j_matrix
            )
            if j_avg is not None:
                new_j[other_new_name] = j_avg
        relabeled.append(
            EquivalenceGroup(
                name=new_name,
                element=g.element,
                atom_indices=g.atom_indices,
                shift_avg_ppm=g.shift_avg_ppm,
                tier=g.tier,
                j_couplings=new_j,
            )
        )
    return relabeled


def _assign_group_labels_local(n: int) -> list[str]:
    """Local re-import of equivalence.assign_group_labels to avoid an
    aggregator-side name collision and keep the dependency explicit."""
    from ..equivalence import assign_group_labels
    return assign_group_labels(n)


def _avg_pairwise_j_local(
    atoms_a: tuple[int, ...],
    atoms_b: tuple[int, ...],
    j_matrix: dict[tuple[int, int], float],
) -> Optional[float]:
    """Average J(a, b) over a ∈ atoms_a, b ∈ atoms_b; ``None`` if no
    pair has a parseable J. Mirrors equivalence._avg_pairwise_j but
    kept local since that helper is module-private."""
    vals: list[float] = []
    for a in atoms_a:
        for b in atoms_b:
            if a == b:
                continue
            v = j_matrix.get((min(a, b), max(a, b)))
            if v is not None:
                vals.append(float(v))
    if not vals:
        return None
    return sum(vals) / len(vals)


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


# --------------------------------------------------------------------
# Schema (source of truth for parse_config + auto-generated docs)
# --------------------------------------------------------------------


def _basename_validator(value: str) -> str:
    """Reject path-like strings (with ``/`` or leading ``.``).

    Output filenames must be basenames — the aggregator joins them with
    its own outputs_dir; allowing path components would let a workflow
    write artifacts outside the engine's expected location.
    """
    if "/" in value or value.startswith("."):
        raise ValueError(f"must be a basename, got {value!r}")
    return value


def _mnova_nuclei_coercer(raw: Any) -> str:
    """Parse + validate the mnova_nuclei CSV, returning a canonical
    comma-joined string (e.g., ``"1H,13C"``)."""
    if isinstance(raw, (list, tuple)):
        tokens = [str(s).strip() for s in raw]
    else:
        tokens = [s.strip() for s in str(raw).split(",")]
    tokens = [t for t in tokens if t]
    for tok in tokens:
        if tok not in ("1H", "13C"):
            raise ValueError(
                f"unsupported token {tok!r}; expected one of '1H', '13C'"
            )
    return ",".join(tokens)


SCHEMA = NodeSchema(
    step_name="nmr_aggregate",
    cli_entrypoint="wf-nmr-aggregate",
    module_path="scripps_workflow.nodes.nmr_aggregate",
    overview=(
        "Boltzmann-averaged ¹H/¹³C chemical shifts and ¹H-¹H coupling "
        "predictions over a conformer ensemble. Consumes a "
        "``thermo_aggregate`` upstream and emits CSVs, mnova-spinsim "
        "XMLs (pre-averaged + per-conformer), and 2D/3D molecule "
        "diagrams annotated with predicted shifts."
    ),
    fields=(
        # ----- General -----
        ConfigField(
            name="solvent",
            type="str",
            default=DEFAULT_SOLVENT,
            section="general",
            description=(
                "Solvent name (used for calibration table lookup and "
                "embedded in artifact metadata). SMD aliases are "
                "resolved per ``orca.solvent_to_orca_smd``."
            ),
        ),
        # ----- Shielding methods -----
        ConfigField(
            name="shielding_method_h",
            type="str",
            default=DEFAULT_SHIELDING_METHOD_H,
            section="shielding",
            description=(
                "DFT functional label for ¹H shielding calibration "
                "lookup. ``WP04`` is cheshire's recommended choice "
                "(Wiitala/Hoye/Cramer 2006)."
            ),
        ),
        ConfigField(
            name="shielding_basis_h",
            type="str",
            default=DEFAULT_SHIELDING_BASIS_H,
            section="shielding",
            description="Basis set for ¹H shielding calibration lookup.",
        ),
        ConfigField(
            name="shielding_method_c",
            type="str",
            default=DEFAULT_SHIELDING_METHOD_C,
            section="shielding",
            description=(
                "DFT functional for ¹³C shielding calibration lookup. "
                "``wB97X-D`` is cheshire's default."
            ),
        ),
        ConfigField(
            name="shielding_basis_c",
            type="str",
            default=DEFAULT_SHIELDING_BASIS_C,
            section="shielding",
            description="Basis set for ¹³C shielding calibration lookup.",
        ),
        # ----- Coupling -----
        ConfigField(
            name="coupling_method",
            type="str",
            default=DEFAULT_COUPLING_METHOD,
            section="coupling",
            description=(
                "DFT functional for J-coupling calibration lookup. "
                "``mPW1PW91`` is the cheshire / Bally-Rablen default."
            ),
        ),
        ConfigField(
            name="coupling_basis",
            type="str",
            default=DEFAULT_COUPLING_BASIS,
            section="coupling",
            description="Basis set for J-coupling calibration lookup.",
        ),
        ConfigField(
            name="skip_couplings",
            type="bool",
            default=False,
            section="coupling",
            description=(
                "If true, skip parsing the ``orca_nmr_j.out`` files "
                "and emit only chemical-shift CSVs / XMLs. Useful when "
                "the upstream didn't run coupling jobs."
            ),
        ),
        # ----- CSV outputs -----
        ConfigField(
            name="output_shifts_csv",
            type="str",
            default=DEFAULT_OUTPUT_SHIFTS_CSV,
            section="outputs",
            validator=_basename_validator,
            description="Basename for the predicted-shifts CSV.",
        ),
        ConfigField(
            name="output_couplings_csv",
            type="str",
            default=DEFAULT_OUTPUT_COUPLINGS_CSV,
            section="outputs",
            validator=_basename_validator,
            description="Basename for the predicted-couplings CSV.",
        ),
        # ----- mnova XML emission -----
        ConfigField(
            name="smiles",
            type="str",
            default=None,
            section="mnova",
            depends_on=("mnova_enabled", "diagrams_enabled"),
            description=(
                "Canonical SMILES. Drives the equivalence detector "
                "for both mnova XML and diagram emission. When absent, "
                "the aggregator falls back to xyz perception of any "
                "conformer's geometry; if that also fails, XML and "
                "diagrams are skipped with "
                "``mnova_xml_skipped:topology_unavailable``."
            ),
        ),
        ConfigField(
            name="mnova_enabled",
            type="bool",
            default=True,
            section="mnova",
            description="Master toggle for mnova-spinsim XML emission.",
        ),
        ConfigField(
            name="mnova_per_conformer",
            type="bool",
            default=True,
            section="mnova",
            description=(
                "Also emit ``predicted_mnova_<nuc>_per_conformer.xml`` "
                "files (one ``<spin-system>`` per conformer with the "
                "Boltzmann weight as ``<population>``)."
            ),
        ),
        ConfigField(
            name="mnova_nuclei",
            type="str",
            default=DEFAULT_MNOVA_NUCLEI,
            section="mnova",
            coercer=_mnova_nuclei_coercer,
            choices=("1H", "13C", "1H,13C", "13C,1H"),
            description=(
                "Comma-separated list of nuclei to emit XML files for. "
                "Tokens: ``1H``, ``13C``."
            ),
        ),
        ConfigField(
            name="mnova_field_mhz_h",
            type="float",
            default=DEFAULT_MNOVA_FIELD_MHZ_H,
            section="mnova",
            description="¹H Larmor frequency (MHz). Default targets a 600 MHz spectrometer.",
        ),
        ConfigField(
            name="mnova_field_mhz_c",
            type="float",
            default=DEFAULT_MNOVA_FIELD_MHZ_C,
            section="mnova",
            description="¹³C Larmor frequency (MHz).",
        ),
        ConfigField(
            name="mnova_line_width_hz_h",
            type="float",
            default=DEFAULT_MNOVA_LINE_WIDTH_HZ_H,
            section="mnova",
            min_value=0.0,
            description="Simulated peak linewidth (Hz) in ¹H spectra.",
        ),
        ConfigField(
            name="mnova_line_width_hz_c",
            type="float",
            default=DEFAULT_MNOVA_LINE_WIDTH_HZ_C,
            section="mnova",
            min_value=0.0,
            description=(
                "Simulated peak linewidth (Hz) in ¹³C spectra. "
                "Broader by default to reflect broadband-decoupled "
                "¹³C peak appearance."
            ),
        ),
        ConfigField(
            name="mnova_from_ppm_h",
            type="float",
            default=DEFAULT_MNOVA_FROM_PPM_H,
            section="mnova",
            description="Lower edge of the simulated ¹H spectrum (ppm).",
        ),
        ConfigField(
            name="mnova_to_ppm_h",
            type="float",
            default=DEFAULT_MNOVA_TO_PPM_H,
            section="mnova",
            description="Upper edge of the simulated ¹H spectrum (ppm).",
        ),
        ConfigField(
            name="mnova_from_ppm_c",
            type="float",
            default=DEFAULT_MNOVA_FROM_PPM_C,
            section="mnova",
            description="Lower edge of the simulated ¹³C spectrum (ppm).",
        ),
        ConfigField(
            name="mnova_to_ppm_c",
            type="float",
            default=DEFAULT_MNOVA_TO_PPM_C,
            section="mnova",
            description="Upper edge of the simulated ¹³C spectrum (ppm).",
        ),
        ConfigField(
            name="mnova_points",
            type="int",
            default=DEFAULT_MNOVA_POINTS,
            section="mnova",
            min_value=64,
            description=(
                "FID points in the simulated spectrum. Higher = finer "
                "resolution but slower mnova render."
            ),
        ),
        ConfigField(
            name="mnova_tol_jcoupling_hz",
            type="float",
            default=DEFAULT_MNOVA_TOL_JCOUPLING_HZ,
            section="mnova",
            min_value=0.0,
            description=(
                "Magnetic-equivalence J-vector tolerance (Hz). "
                "Topological classes whose member J-vectors agree to "
                "within this are classified HARD; mismatches → SOFT "
                "(AA'BB' patterns)."
            ),
        ),
        ConfigField(
            name="mnova_tol_shift_ppm",
            type="float",
            default=DEFAULT_MNOVA_TOL_SHIFT_PPM,
            section="mnova",
            min_value=0.0,
            description=(
                "Data-aware refinement tolerance (ppm). Within a "
                "topological class in a chiral molecule, members whose "
                "DFT shifts spread above this get split into NONE "
                "singletons (catches diastereotopic CH₂)."
            ),
        ),
        ConfigField(
            name="mnova_j_round_threshold_hz",
            type="float",
            default=DEFAULT_MNOVA_J_ROUND_THRESHOLD_HZ,
            section="mnova",
            min_value=0.0,
            description=(
                "Predicted |J| at or below this threshold gets "
                "rounded to 0 before emission. Removes near-zero "
                "long-range coupling noise. Set to 0 to disable."
            ),
        ),
        ConfigField(
            name="mnova_heteronuclear_partners",
            type="csv",
            default=[],
            section="mnova",
            coercer=_parse_partner_list,
            depends_on=("smiles",),
            description=(
                "Additional elements (e.g., ``F``, ``P``) to include "
                "as groups in EVERY primary-nucleus spin-system XML. "
                "Required for ¹H-¹⁹F coupling to render in the ¹H "
                "spectrum. Must coordinate with "
                "``orca_thermo_array.coupling_pairs`` so ORCA actually "
                "computes the cross-element J's."
            ),
        ),
        ConfigField(
            name="mnova_partner_shift_stub_ppm",
            type="float",
            default=100.0,
            section="mnova",
            description=(
                "Placeholder shift (ppm) for partner-element atoms "
                "whose σ wasn't parsed. Lands them outside typical "
                "primary-nucleus windows so the J-driven splitting on "
                "the primary spectrum is what shows."
            ),
        ),
        # ----- Molecule diagrams -----
        ConfigField(
            name="diagrams_enabled",
            type="bool",
            default=True,
            section="diagrams",
            description=(
                "Toggle for ``predicted_structure_<nuc>.svg`` and "
                "``.html`` diagram artifacts. Reuses ``mnova_nuclei`` "
                "to pick which nuclei to depict."
            ),
        ),
        ConfigField(
            name="diagrams_width",
            type="int",
            default=DEFAULT_DIAGRAM_WIDTH,
            section="diagrams",
            min_value=100,
            description="Diagram canvas width (px).",
        ),
        ConfigField(
            name="diagrams_height",
            type="int",
            default=DEFAULT_DIAGRAM_HEIGHT,
            section="diagrams",
            min_value=100,
            description="Diagram canvas height (px).",
        ),
    ),
)


class NmrAggregate(Node):
    """Boltzmann-average NMR observables over a thermo-aggregated ensemble."""

    step = "nmr_aggregate"
    accepts_upstream = True
    requires_upstream = True

    def parse_config(self, raw: dict[str, Any]) -> dict[str, Any]:
        return apply_schema(raw, SCHEMA)

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
            # Build the J-matrix with all element pairs we want in
            # the spin-system. Default behavior preserved: ¹H XML has
            # H-H J's; ¹³C XML has none. Configuring heteronuclear
            # partners adds cross-element pairs to BOTH nuclei's XMLs
            # (so a fluorinated molecule run with partners=["F"]
            # gets H-H + H-F in the ¹H XML and C-F in the ¹³C XML).
            partners = list(cfg["mnova_heteronuclear_partners"])
            include_primary_primary = (nuc == "1H")
            if include_primary_primary or partners:
                pair_set, cals_by_label = _build_pair_set_and_cals(
                    primary_element=element,
                    partner_elements=partners,
                    cfg=cfg,
                )
                if not include_primary_primary:
                    # Drop the primary-primary entry to preserve the
                    # "no ¹³C-¹³C J's" default; cross + partner-
                    # partner entries stay so partner couplings show.
                    pair_set.discard((element, element))
                j_matrix_avg = _calibrated_j_matrix_from_by_pair(
                    by_pair,
                    allowed_element_pairs=pair_set,
                    cals_by_label=cals_by_label,
                )
                j_matrix_avg = _round_small_js_to_zero(
                    j_matrix_avg,
                    float(cfg["mnova_j_round_threshold_hz"]),
                )
            else:
                j_matrix_avg = {}

            partner_shifts = _partner_shifts_by_atom(
                mol=mol,
                partner_elements=partners,
                by_atom=by_atom,
                stub_ppm=float(cfg["mnova_partner_shift_stub_ppm"]),
            )
            avg_groups = _build_multinucleus_groups(
                mol=mol,
                primary_element=element,
                partner_elements=partners,
                primary_shifts=avg_shifts,
                partner_shifts=partner_shifts,
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
        # SVG. Pass xyz_text so the renderer produces a 3D-snapshot
        # (PCA-aligned projection of the actual geometry) rather than
        # RDKit's schematic 2D layout. When xyz is absent it falls
        # back to Compute2DCoords automatically.
        svg_text = render_shift_svg(
            mol=mol,
            groups=avg_groups,
            xyz_text=xyz_text,
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
        the same. Returns ``None`` if no conformer has a readable xyz.

        Tries ``path_abs`` first (the canonical convention from
        ``orca_thermo_array``), then falls back to
        ``task_dir_abs/input.xyz`` (the legacy/recovered convention
        for upstream nodes that point ``path_abs`` at a task directory
        instead of a file). Older ``thermo_aggregate`` builds shipped
        directories in ``path_abs``; the fallback keeps those runs
        compatible without a re-run.
        """
        for c in confs:
            for candidate in self._xyz_path_candidates(c):
                text = self._read_as_xyz(candidate)
                if text is not None:
                    return text
        return None

    @staticmethod
    def _xyz_path_candidates(conf: dict[str, Any]) -> list[str]:
        """Ordered list of paths to try for a conformer's xyz."""
        paths: list[str] = []
        primary = conf.get("path_abs")
        if isinstance(primary, str):
            paths.append(primary)
        task_dir = conf.get("task_dir_abs")
        if isinstance(task_dir, str):
            # Convention: orca_thermo_array stages each conformer's
            # input as <task_dir>/input.xyz.
            paths.append(str(Path(task_dir) / "input.xyz"))
        return paths

    @staticmethod
    def _read_as_xyz(path: str) -> Optional[str]:
        """Read ``path`` and return its content if it parses as xyz format.

        Returns ``None`` when the path doesn't exist, is a directory,
        or doesn't start with an integer atom count.
        """
        try:
            text = Path(path).read_text(encoding="utf-8")
        except Exception:
            return None
        head = text.lstrip().splitlines()[:1]
        if not head:
            return None
        try:
            int(head[0].split()[0])
        except (ValueError, IndexError):
            return None
        return text

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
            partners = list(cfg["mnova_heteronuclear_partners"])
            include_primary_primary = (nucleus == "1H")
            if include_primary_primary or partners:
                pair_set, cals_by_label = _build_pair_set_and_cals(
                    primary_element=element,
                    partner_elements=partners,
                    cfg=cfg,
                )
                if not include_primary_primary:
                    pair_set.discard((element, element))
                j_matrix = _calibrated_j_matrix_from_per_conformer(
                    couplings or [],
                    allowed_element_pairs=pair_set,
                    cals_by_label=cals_by_label,
                )
                j_matrix = _round_small_js_to_zero(
                    j_matrix,
                    float(cfg["mnova_j_round_threshold_hz"]),
                )
            else:
                j_matrix = {}
            partner_shifts = _partner_shifts_per_conformer(
                mol=mol,
                partner_elements=partners,
                shieldings=shieldings,
                stub_ppm=float(cfg["mnova_partner_shift_stub_ppm"]),
            )
            groups = _build_multinucleus_groups(
                mol=mol,
                primary_element=element,
                partner_elements=partners,
                primary_shifts=shifts,
                partner_shifts=partner_shifts,
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
    "DEFAULT_MNOVA_J_ROUND_THRESHOLD_HZ",
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
