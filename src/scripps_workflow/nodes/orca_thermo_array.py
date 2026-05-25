"""``wf-orca-thermo-array`` — composite freq + high-level single point.

Sister node to :mod:`scripps_workflow.nodes.orca_dft_array`. Where the
DFT array node optimizes geometries (``! r2scan-3c TightSCF TightOpt``)
and emits *new* coordinates, this node runs a composite ORCA job over
an *already-optimized* ensemble: a low-level frequency calculation
(``! r2scan-3c TightSCF Freq``) followed by a high-level single point
(``! wB97M-V def2-TZVPP TightSCF``) on the SAME geometry, in one ORCA
input file separated by ``$new_job``. The downstream
:mod:`scripps_workflow.nodes.thermo_aggregate` reads ``G - E(el)`` from
the low-level freq block and ``FINAL SINGLE POINT ENERGY`` from the
high-level SP, combining them into a composite Gibbs energy::

    G_composite = E_SP_high + (G - E_el)_low

Key differences from ``orca_dft_array``:

    * Default keywords for the FREQ job: ``r2scan-3c TightSCF Freq``
      (no geometry optimization — frequencies on the input coords).
    * Default keywords for the SP job: ``wB97M-V def2-TZVPP TightSCF``.
      Set ``singlepoint_keywords=none`` to disable the SP step and run
      a single-job freq.
    * Per-task input/output files are named ``orca_thermo.{inp,out}``.
    * No new geometry artifact: the input geometry IS the geometry the
      thermochemistry refers to. The ``conformers[]`` records point at
      the staged input xyz file plus an ``orca_out_abs`` field for the
      thermo aggregator to parse.
    * The ensemble published as ``xyz_ensemble[input_ensemble]`` is the
      INPUT conformers concatenated; ``best.xyz`` is copied from the
      input staging dir, not from new optimization output.
    * Per-task failures include ``missing_or_unparsed_energy`` and
      ``orca_not_terminated_normally``. The latter catches
      walltime-killed jobs that got far enough to print a FINAL E line
      but stopped before the SP / Hessian completed — these are
      silently worthless to the thermo aggregator and we surface them
      upfront.

Config keys (``key=value`` tokens or one JSON object) — same shape as
``orca_dft_array`` plus the new ``singlepoint_keywords`` knob:

    max_concurrency        ``%M`` in ``--array=1-N%M`` (also accepts
                           ``batchsize``/``max_nodes`` aliases)        [10]
    charge                 int                                          [0]
    unpaired_electrons     int (multiplicity = unpaired + 1)            [0]
    multiplicity           override int (wins over unpaired_electrons)  [None]
    solvent                SMD solvent token, or null/none for vacuum   [None]
    smd_solvent            verbatim SMDsolvent override                 [None]
    keywords               First-job ``!`` line (the freq calc)         ["r2scan-3c TightSCF Freq"]
    singlepoint_keywords   Second-job ``!`` line, or null/none/"" to
                           skip the SP step entirely.                   ["wB97M-V def2-TZVPP TightSCF DEFGRID3"]
    temperature_k          Thermo temperature (K). Must match
                           ``thermo_aggregate.temperature_k``
                           downstream or the v6.5 thermo cache
                           drifts.                                       [298.15]
    maxcore                MB per ORCA process (clamped to >= 500)      [4000]
    nprocs                 ``%pal nprocs`` and ``--ntasks``              [8]
    time_limit             SBATCH ``-t``                                ["12:00:00"]
    partition              optional SBATCH ``-p``                       [None]
    job_name               SBATCH ``-J`` (defaults to
                           ``orca_thermo_array_<n>``)
    orca_module            module-load string                           ["orca/6.0.0"]
    submit                 actually call ``sbatch``?                    [true]
    monitor                actually wait for the job?                   [true]
    monitor_interval_s     polling interval (seconds; clamped >= 5)    [60]
    monitor_timeout_min    wall-clock cap (0 = no cap)                  [0]
    silence_openib         set ``OMPI_MCA_btl=^openib`` in the array   [true]

Optional NMR section (all default ON because this node is the
shielding/coupling generator for the NMR Predictor pipeline — set all
three flags to ``false`` to degrade back to a pure freq[+SP] run):

    run_shielding_h           append a 1H shielding job                [true]
    run_shielding_c           append a 13C shielding job               [true]
    run_couplings             append a 1H-1H coupling job              [true]
    shielding_method_h        functional for the 1H job                ["WP04"]
    shielding_basis_h         basis set for the 1H job                 ["6-311++G(2d,p)"]
    shielding_method_c        functional for the 13C job               ["wB97X-D"]
    shielding_basis_c         basis set for the 13C job                ["6-31G(d,p)"]
    coupling_method           functional for the J job                 ["mPW1PW91"]
    coupling_basis            basis set for the J job                  ["pcJ-2"]
    coupling_pairs            list/csv of ORCA nuclei selectors        [["all H"]]
    coupling_thresh_angstrom  ``SpinSpinRThresh`` cap                  [8.0]
    nmr_aux_keywords          extra ``!`` tokens added to every NMR    ["TightSCF"]
                              job line

Each enabled NMR job is appended after the freq+SP via ``$new_job``,
producing a single ``orca_thermo.out`` containing every block. The
matching :mod:`scripps_workflow.nodes.nmr_aggregate` reads back the
shielding / coupling tables, Boltzmann-averages over the conformer
ensemble, and applies the cheshire / Bally-Rablen linear scaling.

Ports the legacy ``orca_thermo_freq_array`` script onto the new
framework, and extends it with the composite SP step. The on-disk
artifact layout (``outputs/array/tasks/...``,
``outputs/thermo/thermo.energies``) is preserved so the matching
thermo aggregator port can consume either the new or the legacy
output.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from .. import logging_utils
from ..hashing import sha256_file
from ..node import Node, NodeContext
from ..parsing import parse_bool
from ..schema import Manifest


# --------------------------------------------------------------------
# v6.5b — optional cache imports. Best-effort: a missing nmr_data /
# rdkit / DB just falls through to compute.
# --------------------------------------------------------------------

_NMR_DATA_IMPORT_ERROR: str | None = None
_RDKIT_IMPORT_ERROR: str | None = None

try:
    from nmr_data.cache import (  # noqa: F401
        build_crest_ensemble_key,
        build_dft_run_key,
        build_predicted_run_key,
        build_thermo_run_key,
        find_dft_run,
        find_ensemble,
        find_predicted_run,
        find_thermo_run,
        fingerprint as _cache_fingerprint,
        inchikey_from_smiles,
    )
    from nmr_data.db import get_session
    from nmr_data.models import Molecule
    _HAS_NMR_DATA = True
except Exception as _e:
    _HAS_NMR_DATA = False
    _NMR_DATA_IMPORT_ERROR = f"{type(_e).__name__}: {_e}"

try:
    from rdkit import Chem  # noqa: F401
    _HAS_RDKIT = True
except Exception as _e:
    _HAS_RDKIT = False
    _RDKIT_IMPORT_ERROR = f"{type(_e).__name__}: {_e}"
from ..orca import (
    concat_xyz_files,
    make_orca_compound_input,
    make_orca_simple_input,
    nmr_coupling_block,
    nmr_shielding_block,
    orca_terminated_normally,
    parse_orca_final_energy,
    resolve_functional_alias,
    write_energy_file,
)
from ..config_schema import ConfigField, NodeSchema, apply_schema
from ..parsing import normalize_optional_str
from ..slurm import (
    MonitorResult,
    ProgressCounts,
    SlurmExecutables,
    count_task_progress,
    discover_slurm_executables,
    make_array_slurm_text,
    monitor_array_job,
    sacct_failures_for_array,
    sacct_states,
    sbatch_submit,
    squeue_has_any,
    standard_orca_per_task_body,
)

# Re-use the same input-staging primitives that the DFT array node uses
# — the conformer source discovery rule + multi-xyz splitting are
# identical.
from .orca_dft_array import (
    _make_slurm_array_fields,
    _reject_empty_keywords,
    normalize_max_concurrency,
    resolve_multiplicity,
    stage_conformer_inputs,
)


# --------------------------------------------------------------------
# v6.5b cache helpers — upstream-chain walkers + .gz hydration
# --------------------------------------------------------------------
#
# Same pattern as the CREST cache check (build a typed key from
# upstream-recorded inputs, look up via nmr_data.cache, on hit
# materialize the central-tree files locally). orca_thermo_array's
# hit path is more involved than CREST's because thermo_aggregate and
# nmr_aggregate downstream walk a per-conformer task_dir tree that
# this node normally builds via SLURM submission. We reconstruct that
# tree from the central tree's gzipped .out files.


def _walk_upstream_for_step(ctx: NodeContext, step_name: str) -> dict | None:
    """Walk back through ``upstream.manifest_path`` references looking
    for a manifest whose ``step`` matches. Bounded to a handful of hops
    so a pathological chain can't spin."""
    visited: set[str] = set()
    current = ctx.upstream_manifest.to_dict() if ctx.upstream_manifest else None
    for _ in range(12):
        if current is None:
            return None
        if (current.get("step") or "") == step_name:
            return current
        upstream = current.get("upstream") or {}
        mp = upstream.get("manifest_path")
        if not mp or mp in visited:
            return None
        visited.add(mp)
        p = Path(mp)
        if not p.exists():
            return None
        try:
            current = Manifest.read(p).to_dict()
        except Exception:
            return None
    return None


def _walk_upstream_for_smiles(ctx: NodeContext) -> str | None:
    """Same chain walk, looking for ``inputs.smiles`` (carried by
    smiles_to_3d at the top of the chain)."""
    visited: set[str] = set()
    current = ctx.upstream_manifest.to_dict() if ctx.upstream_manifest else None
    for _ in range(12):
        if current is None:
            return None
        s = (current.get("inputs") or {}).get("smiles")
        if isinstance(s, str) and s:
            return s
        upstream = current.get("upstream") or {}
        mp = upstream.get("manifest_path")
        if not mp or mp in visited:
            return None
        visited.add(mp)
        p = Path(mp)
        if not p.exists():
            return None
        try:
            current = Manifest.read(p).to_dict()
        except Exception:
            return None
    return None


def _gunzip_to(src_gz: Path, dst: Path) -> None:
    """Decompress ``src_gz`` (a .gz file) into ``dst``. Creates parent
    dirs; idempotent on dst existence (skips if dst already there)."""
    import gzip
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(src_gz, "rb") as f_in, open(dst, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


#: ORCA module name to ``module load`` inside the SLURM array script.
DEFAULT_ORCA_MODULE: str = "orca/6.0.0"

#: Default ORCA simple-input keywords for the freq/thermo calculation.
#: ``Freq`` does an analytic-Hessian frequency calculation on top of an
#: SCF; combined with ``r2scan-3c`` this is the lab's standard "thermo
#: at the optimization level" recipe.
DEFAULT_KEYWORDS: str = "r2scan-3c TightSCF Freq"

#: Default ORCA simple-input keywords for the high-level single-point
#: that follows the freq job (separated by ``$new_job``). Combined with
#: the ``DEFAULT_KEYWORDS`` low-level freq, the downstream thermo
#: aggregator computes a composite Gibbs energy
#: ``G_composite = E_SP_high + (G - E_el)_low``. Set
#: ``singlepoint_keywords=none`` (or ``""``) at config time to disable
#: the SP step entirely and run a plain single-job freq calculation.
#:
#: Note: ``RIJCOSX`` is intentionally absent. Under ORCA 6.0.0 the
#: conventional-integral default rejects RIJCOSX with "Conventional
#: integral handling with RIJCOSX approximation does not yet work."
#: If you need the RIJCOSX speedup, add ``DIRECT`` (forces direct-mode
#: SCF) to the keywords explicitly — e.g.
#: ``wB97M-V def2-TZVPP TightSCF RIJCOSX DEFGRID3 DIRECT``.
DEFAULT_SINGLEPOINT_KEYWORDS: str = "wB97M-V def2-TZVPP TightSCF DEFGRID3"

#: Default ORCA ``%pal nprocs`` (and SLURM ``--ntasks``). Matches the
#: rest of the array nodes' default of 8 — Frequency calculations
#: parallelize well, but 8 is the cluster-wide sweet spot.
DEFAULT_NPROCS: int = 8

#: Temperature (K) used for thermochemistry. Matches
#: :mod:`scripps_workflow.nodes.thermo_aggregate`'s default so the two
#: nodes line up on the same value when neither is overridden. Pinned
#: separately rather than imported to keep the inter-module coupling
#: one-way (orca_thermo_array → thermo_aggregate, never the reverse).
#: When this differs from thermo_aggregate's ``temperature_k`` the
#: thermo cache key drifts and the cache misses (intentional — the
#: thermal corrections aren't the same physics).
DEFAULT_TEMPERATURE_K: float = 298.15

#: Per-task ORCA input/output filenames. Note there is no
#: ``ORCA_OPT_XYZ`` analogue — frequency runs preserve the input
#: geometry rather than producing a new one.
ORCA_INP_NAME: str = "orca_thermo.inp"
ORCA_OUT_NAME: str = "orca_thermo.out"

#: Optional high-level single-point input/output filenames. The SP is
#: run as a separate ORCA process and appended into ``orca_thermo.out``
#: after successful completion, so downstream parsers still see the
#: composite high-level final energy without exposing ORCA to ``$new_job``
#: method-state leakage between r2scan-3c and wB97M-V.
ORCA_SP_INP_NAME: str = "orca_thermo_sp.inp"
ORCA_SP_OUT_NAME: str = "orca_thermo_sp.out"

#: Per-task NMR input/output filenames. Each NMR job runs as a SEPARATE
#: ORCA invocation (not via ``$new_job``) so that method-state flags
#: like VV10/NL, D3/D4, gCP, etc. cannot leak from the freq+SP compound
#: into the chemically-unrelated NMR calculations. The cost is one
#: extra ORCA boot per job (~5–10 s of overhead); the benefit is
#: bulletproof state isolation for any future functional combination.
ORCA_NMR_H_INP_NAME: str = "orca_nmr_h.inp"
ORCA_NMR_H_OUT_NAME: str = "orca_nmr_h.out"
ORCA_NMR_C_INP_NAME: str = "orca_nmr_c.inp"
ORCA_NMR_C_OUT_NAME: str = "orca_nmr_c.out"
ORCA_NMR_J_INP_NAME: str = "orca_nmr_j.inp"
ORCA_NMR_J_OUT_NAME: str = "orca_nmr_j.out"


# --------------------------------------------------------------------
# NMR defaults — kept in sync with
# :mod:`scripps_workflow.nodes.nmr_aggregate` so an operator who
# configures one node sees the same recipe in the other. Override at
# config time when running a non-cheshire calibration.
# --------------------------------------------------------------------

#: Functional/basis defaults for the ¹H GIAO shielding job. Combined
#: with the cheshire ¹H calibration table in
#: ``scripps_workflow.nmr_calibration``.
DEFAULT_SHIELDING_METHOD_H: str = "WP04"
DEFAULT_SHIELDING_BASIS_H: str = "6-311++G(2d,p)"

#: Functional/basis defaults for the ¹³C GIAO shielding job.
DEFAULT_SHIELDING_METHOD_C: str = "wB97X-D"
DEFAULT_SHIELDING_BASIS_C: str = "6-31G(d,p)"

#: Functional/basis defaults for the ¹H–¹H J-coupling job (Bally/Rablen).
DEFAULT_COUPLING_METHOD: str = "mPW1PW91"
DEFAULT_COUPLING_BASIS: str = "pcJ-2"

#: Per-job ``! NMR`` keyword fragment. The shielding/coupling jobs all
#: need ``! NMR`` to trigger ORCA's GIAO + spin-spin machinery. The
#: ``TightSCF`` ride-along matches the freq/SP defaults so the SCF
#: thresholds are uniform across the chain.
DEFAULT_NMR_AUX_KEYWORDS: str = "TightSCF"

#: Default coupling-block configuration. ``["all H"]`` requests every
#: ¹H–¹H pair; the ``SpinSpinRThresh`` cap keeps the O(N²) cost down
#: by skipping pairs farther apart than the threshold.
DEFAULT_COUPLING_PAIRS: tuple[str, ...] = ("all H",)
DEFAULT_COUPLING_THRESH_ANGSTROM: float = 8.0


# --------------------------------------------------------------------
# System-class profiles for the NMR shielding + coupling jobs
# --------------------------------------------------------------------
#
# The geometry-opt + freq + high-level SP steps don't change between
# system classes — the def2 family of basis sets ships with Stuttgart
# ECPs for all elements Z >= 37 (Rb–Rn), so r2scan-3c (which builds on
# def2-mTZVPP) and wB97M-V/def2-TZVPP both handle Pd correctly out of
# the box for energies and geometries.
#
# What does change is the *NMR* step. ECPs capture scalar relativistic
# contraction of the core but DO NOT capture spin-orbit. For ¹H and
# ¹³C shieldings on atoms near a heavy metal, the spin-orbit-driven
# HALA effect contributes ~0.5–2 ppm on ¹H and ~5–20 ppm on ¹³C —
# above noise, sometimes well above. Capturing it requires a
# relativistic Hamiltonian (ZORA or DKH) plus a basis recontracted
# for that Hamiltonian (def2-ZORA-TZVPP for ORCA's ZORA path).
#
# Profiles encode this choice:
#
#   organic   — current default. Organic basis sets (6-311++G(2d,p)
#               on H, 6-31G(d,p) on C, pcJ-2 for J). No relativistic
#               Hamiltonian. Matches the cheshire + Bally/Rablen
#               calibration tuples.
#   organopd  — for Pd-containing molecules. Switches all three NMR
#               bases to def2-ZORA-TZVPP and prepends ``ZORA`` to the
#               NMR job's ``!`` line so ORCA enables scalar
#               relativistic with HALA contributions.
#
# Adding more profiles (organotm_3d, organotm_4d, etc.) is purely
# additive — register a new entry in this dict + extend
# :func:`detect_system_class` if you want auto-detection for that
# element class. The cache fingerprint for PredictedRun naturally
# diverges between profiles because shielding_basis_* are part of
# the PredictedRunKey payload — no schema change required.

SYSTEM_CLASS_PROFILES: dict[str, dict[str, Any]] = {
    "organic": {
        "nmr_keywords_prefix": "",
        "shielding_basis_h": DEFAULT_SHIELDING_BASIS_H,
        "shielding_basis_c": DEFAULT_SHIELDING_BASIS_C,
        "coupling_basis": DEFAULT_COUPLING_BASIS,
    },
    "organopd": {
        "nmr_keywords_prefix": "ZORA",
        "shielding_basis_h": "def2-ZORA-TZVPP",
        "shielding_basis_c": "def2-ZORA-TZVPP",
        "coupling_basis": "def2-ZORA-TZVPP",
    },
}

#: Recognized profile names. ``"auto"`` triggers detection; anything
#: else must match a key in :data:`SYSTEM_CLASS_PROFILES`.
SYSTEM_CLASSES_KNOWN: frozenset[str] = frozenset(SYSTEM_CLASS_PROFILES.keys())

#: Element symbols that map to each non-organic profile. Used by
#: :func:`detect_system_class` to pick a profile from input geometry.
#: First match wins, in dict-iteration order — keep more specific /
#: heavier profiles first as new ones land.
_PROFILE_ELEMENT_TRIGGERS: dict[str, frozenset[str]] = {
    "organopd": frozenset({"Pd"}),
}


def detect_system_class(xyz_paths: list[Path]) -> str:
    """Scan input geometries for heavy-element triggers; return the
    matching :data:`SYSTEM_CLASS_PROFILES` key.

    First-match-wins on the trigger dict. Returns ``"organic"`` when
    no triggers fire. Reads at most the first few xyz files — the
    element composition is the same across all conformers of one
    molecule, so we don't need to scan a 50-conformer ensemble.

    Robust to per-file read errors: a single unreadable xyz just gets
    skipped, the rest of the scan continues.
    """
    elements_seen: set[str] = set()
    for xyz in xyz_paths[:3]:
        try:
            text = xyz.read_text(encoding="utf-8")
        except OSError:
            continue
        # XYZ format: line 1 = atom count, line 2 = comment, lines 3+
        # are ``Elem  x  y  z`` rows. We accept any whitespace-leading
        # token as the element symbol; numbers / blank lines get
        # filtered out naturally because they don't appear as keys in
        # any trigger set.
        for line in text.splitlines():
            tokens = line.split()
            if tokens:
                elements_seen.add(tokens[0])
    for profile_name, triggers in _PROFILE_ELEMENT_TRIGGERS.items():
        if elements_seen & triggers:
            return profile_name
    return "organic"


def _upstream_xyz_paths(ctx: NodeContext) -> list[Path]:
    """Pull conformer xyz paths from the upstream manifest.

    Used by :func:`detect_system_class` before staging — element
    composition is the same on the upstream's xyzs as on whatever
    we'd stage locally, so we can resolve system_class before the
    cache check fires (which depends on the resolved profile).

    Returns an empty list if no upstream manifest is present (the
    node will fail later via the existing ``no_upstream_manifest``
    failure path; this helper just falls through cleanly).
    """
    if ctx.upstream_manifest is None:
        return []
    confs = ctx.upstream_manifest.artifacts.get("conformers") or []
    paths: list[Path] = []
    for rec in confs:
        if not isinstance(rec, dict):
            continue
        p = rec.get("path_abs")
        if isinstance(p, str) and p:
            paths.append(Path(p))
    return paths


def _resolve_system_class_profile(
    cfg: dict[str, Any], *, ctx: NodeContext,
) -> str:
    """Resolve ``cfg['system_class']`` against the profile registry.

    Mutates ``cfg`` in place: writes the resolved profile name back to
    ``cfg['system_class']`` (so ``"auto"`` is replaced with the
    detected concrete class), and swaps any NMR-basis fields that still
    hold their organic defaults to the profile's value. Operator-
    supplied basis values pass through unchanged.

    Returns the resolved class for logging convenience.

    Raises :class:`ValueError` on an unknown class string. Auto-detect
    failure (no upstream, can't read xyzs) falls back to ``"organic"``
    — the conservative choice.
    """
    raw = str(cfg.get("system_class", "auto")).strip().lower()
    if raw == "auto":
        resolved = detect_system_class(_upstream_xyz_paths(ctx))
        logging_utils.log_info(
            f"orca-thermo: system_class=auto resolved to {resolved!r} "
            "based on upstream element scan"
        )
    elif raw in SYSTEM_CLASSES_KNOWN:
        resolved = raw
        logging_utils.log_info(
            f"orca-thermo: system_class={resolved!r} (operator-set)"
        )
    else:
        raise ValueError(
            f"unknown system_class={raw!r}; expected 'auto' or one of "
            f"{sorted(SYSTEM_CLASSES_KNOWN)}"
        )

    profile = SYSTEM_CLASS_PROFILES[resolved]
    organic = SYSTEM_CLASS_PROFILES["organic"]
    cfg["system_class"] = resolved

    # Swap NMR basis defaults to the profile's values when the operator
    # left them at the organic defaults. Explicit operator overrides
    # (different value from the organic default) pass through unchanged.
    for key in ("shielding_basis_h", "shielding_basis_c", "coupling_basis"):
        profile_value = profile[key]
        if cfg.get(key) == organic[key] and profile_value != organic[key]:
            cfg[key] = profile_value
            logging_utils.log_info(
                f"orca-thermo: profile {resolved!r} swapped {key} to "
                f"{profile_value!r}"
            )
    return resolved


def build_nmr_input_files(
    *,
    cfg: dict[str, Any],
    multiplicity: int,
    xyz_filename: str = "input.xyz",
) -> dict[str, str]:
    """Translate the NMR config knobs into standalone ORCA ``.inp`` files.

    Returns a mapping ``{filename: input_text}`` with one entry per
    enabled NMR job. Empty dict when none are enabled.

    Each file is a complete, standalone ORCA simple input — *not* a
    ``$new_job`` chain. Running them as separate ORCA invocations
    eliminates method-state leakage (DFT-NL/VV10, D3/D4, gCP, …) from
    the freq+SP compound that ran upstream. The trade-off is one
    extra ORCA boot per job (~5–10 s of basis-set / functional-table
    init); the win is bulletproof isolation for any future
    functional combination, including the WP04 ¹H + wB97X-D3 ¹³C +
    mPW1PW91 J-coupling chain whose cross-functional state mismatch
    used to abort the run.

    Filenames returned (only those whose ``run_*`` flag is True):

        * :data:`ORCA_NMR_H_INP_NAME` — ¹H GIAO shielding
        * :data:`ORCA_NMR_C_INP_NAME` — ¹³C GIAO shielding
        * :data:`ORCA_NMR_J_INP_NAME` — ¹H–¹H J-couplings

    Method/basis come from the matching ``shielding_method_*`` /
    ``shielding_basis_*`` / ``coupling_method`` / ``coupling_basis``
    config keys; functional aliases (e.g. ``WP04``, ``wB97X-D``) are
    resolved via :func:`scripps_workflow.orca.resolve_functional_alias`.

    Pure function — no I/O — so it's easy to unit-test against
    config dicts.
    """
    files: dict[str, str] = {}

    aux = str(cfg.get("nmr_aux_keywords", DEFAULT_NMR_AUX_KEYWORDS)).strip()
    aux_suffix = f" {aux}" if aux else ""

    # Profile keyword prefix (e.g. ``ZORA`` for organopd). Prepended
    # to every NMR job's ``!`` line so ORCA picks up the relativistic
    # Hamiltonian alongside the basis swap that the resolver did.
    # Empty string for the organic default — no prefix added.
    prefix = (cfg.get("nmr_keywords_prefix") or "").strip()
    prefix_part = f"{prefix} " if prefix else ""

    common = {
        "nprocs": cfg["nprocs"],
        "maxcore": cfg["maxcore"],
        "charge": cfg["charge"],
        "multiplicity": multiplicity,
        "solvent": cfg["solvent"],
        "smd_solvent_override": cfg["smd_solvent"],
        "xyz_filename": xyz_filename,
    }

    if cfg.get("run_shielding_h"):
        h_method, h_extras = resolve_functional_alias(cfg["shielding_method_h"])
        files[ORCA_NMR_H_INP_NAME] = make_orca_simple_input(
            keywords=f"{prefix_part}NMR {h_method} {cfg['shielding_basis_h']}{aux_suffix}",
            extra_blocks=[*h_extras, nmr_shielding_block("all H")],
            **common,
        )

    if cfg.get("run_shielding_c"):
        c_method, c_extras = resolve_functional_alias(cfg["shielding_method_c"])
        files[ORCA_NMR_C_INP_NAME] = make_orca_simple_input(
            keywords=f"{prefix_part}NMR {c_method} {cfg['shielding_basis_c']}{aux_suffix}",
            extra_blocks=[*c_extras, nmr_shielding_block("all C")],
            **common,
        )

    if cfg.get("run_couplings"):
        pairs = list(cfg.get("coupling_pairs") or DEFAULT_COUPLING_PAIRS)
        thresh = cfg.get("coupling_thresh_angstrom")
        j_method, j_extras = resolve_functional_alias(cfg["coupling_method"])
        files[ORCA_NMR_J_INP_NAME] = make_orca_simple_input(
            keywords=f"{prefix_part}NMR {j_method} {cfg['coupling_basis']}{aux_suffix}",
            extra_blocks=[
                *j_extras,
                nmr_coupling_block(
                    pairs,
                    ssall=True,
                    spinspin_thresh=(
                        float(thresh) if thresh is not None else None
                    ),
                ),
            ],
            **common,
        )

    return files


# --------------------------------------------------------------------
# Pure helpers (testable)
# --------------------------------------------------------------------


def build_thermo_task_dirs(
    *,
    staged_paths: list[Path],
    tasks_root: Path,
    inp_text: str,
    inp_name: str = ORCA_INP_NAME,
    extra_inputs: dict[str, str] | None = None,
) -> None:
    """Per-conformer task dir, each with ``input.xyz`` + ``<inp_name>``.

    Same shape as :func:`orca_dft_array.build_task_dirs` modulo the
    default ``inp_name``. Kept as a thin wrapper for clarity at the
    call site.

    :param extra_inputs: Optional ``{filename: text}`` map of extra
        ORCA inputs to materialize in each task dir. Used by the NMR
        pipeline to drop standalone ``orca_nmr_h.inp`` /
        ``orca_nmr_c.inp`` / ``orca_nmr_j.inp`` next to the freq+SP
        compound; the SLURM per-task body runs each as its own
        invocation so method-state flags can't leak between them.
    """
    tasks_root.mkdir(parents=True, exist_ok=True)
    for i, src_xyz in enumerate(staged_paths, start=1):
        task_dir = tasks_root / f"task_{i:04d}"
        task_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_xyz, task_dir / "input.xyz")
        (task_dir / inp_name).write_text(inp_text, encoding="utf-8")
        for name, text in (extra_inputs or {}).items():
            (task_dir / name).write_text(text, encoding="utf-8")


def collect_thermo_outputs(
    *,
    n_tasks: int,
    tasks_root: Path,
    staged_dir: Path,
    out_name: str = ORCA_OUT_NAME,
) -> tuple[
    list[dict[str, Any]],
    list[float | None],
    list[dict[str, Any]],
]:
    """Walk each ``task_XXXX`` and gather thermo.out + energy.

    Unlike :func:`orca_dft_array.collect_optimized_outputs`, this node
    does NOT publish a new per-conformer xyz — the geometry is whatever
    was staged on the input side. Each per-conformer record references:

        * ``path_abs`` — the staged INPUT xyz
        * ``orca_out_abs`` — the per-task ``orca_thermo.out`` (the
          field that :mod:`scripps_workflow.nodes.thermo_aggregate`
          parses to extract Gibbs / enthalpy / ZPVE).

    Per-task failures are surfaced with two distinct error codes:

        * ``missing_or_unparsed_energy`` — no ``FINAL E`` line in the
          .out file (or the file is missing entirely).
        * ``orca_not_terminated_normally`` — the .out file exists and
          has a parseable FINAL E, but ``ORCA TERMINATED NORMALLY`` is
          missing. The downstream thermo aggregator NEEDS the Freq
          section to actually finish, so we treat this as a hard fail.

    Returns ``(conformer_records, energies_h, failure_records)``.
    Records are always 1:1 with task indices (no skipping) — the array
    is a list of length ``n_tasks``.
    """
    conformer_records: list[dict[str, Any]] = []
    energies_h: list[float | None] = []
    failure_records: list[dict[str, Any]] = []

    for i in range(1, int(n_tasks) + 1):
        task_dir = tasks_root / f"task_{i:04d}"
        out_path = task_dir / out_name
        staged_xyz = staged_dir / f"conf_{i:04d}.xyz"

        e_h = parse_orca_final_energy(out_path) if out_path.exists() else None
        energies_h.append(e_h)

        rec: dict[str, Any] = {
            "index": i,
            "label": f"conf_{i:04d}",
            "path_abs": str(staged_xyz.resolve()),
            "format": "xyz",
            "task_dir_abs": str(task_dir.resolve()),
            "orca_out_abs": (
                str(out_path.resolve()) if out_path.exists() else None
            ),
        }
        if staged_xyz.exists():
            rec["sha256"] = sha256_file(staged_xyz)

        if e_h is not None:
            rec["energy_hartree"] = float(e_h)
        else:
            failure_records.append(
                {
                    "error": "missing_or_unparsed_energy",
                    "index": i,
                    "task_dir": str(task_dir.resolve()),
                    "orca_out": (
                        str(out_path.resolve())
                        if out_path.exists()
                        else None
                    ),
                }
            )

        if not orca_terminated_normally(out_path):
            failure_records.append(
                {
                    "error": "orca_not_terminated_normally",
                    "index": i,
                    "task_dir": str(task_dir.resolve()),
                }
            )

        conformer_records.append(rec)

    return conformer_records, energies_h, failure_records


# --------------------------------------------------------------------
# Schema (source of truth for parse_config + auto-generated docs)
# --------------------------------------------------------------------


SCHEMA = NodeSchema(
    step_name="orca_thermo_array",
    cli_entrypoint="wf-orca-thermo-array",
    module_path="scripps_workflow.nodes.orca_thermo_array",
    overview=(
        "SLURM-array freq+SP runner with optional NMR shielding / "
        "J-coupling jobs chained after. Heart of the NMR Predictor "
        "workflow: defaults follow the cheshire & Bally-Rablen "
        "calibration recipes so the downstream ``nmr_aggregate`` "
        "sees the geometry/method combination its lookup tables "
        "were fit for."
    ),
    fields=(
        *_make_slurm_array_fields(
            default_keywords=DEFAULT_KEYWORDS,
            default_nprocs=DEFAULT_NPROCS,
        ),
        # ----- Compound-job knob -----
        ConfigField(
            name="singlepoint_keywords",
            type="str",
            default=DEFAULT_SINGLEPOINT_KEYWORDS,
            description=(
                "ORCA simple-input line for the high-level SP appended "
                "after the freq job via ``$new_job``. Combined with "
                "``keywords`` (low-level freq), the downstream thermo "
                "aggregator computes a composite Gibbs energy. "
                "Set to ``none`` (or any of ``null``, ``auto``, ``\"\"``) "
                "to disable the SP step entirely — the run degrades to "
                "pure freq+thermo at ``keywords``. For combobox widgets "
                "in the GUI use the literal string ``none`` rather than "
                "an empty option, since the engine drops empty tag "
                "tokens before they reach this node."
            ),
        ),
        ConfigField(
            name="temperature_k",
            type="float",
            default=DEFAULT_TEMPERATURE_K,
            min_value=0.0,
            description=(
                "Temperature (K) at which the thermochemistry is "
                "evaluated. Injected into ORCA's ``%freq Temp`` block "
                "so the printed thermal corrections (H_corr, TS, …) "
                "are computed at this T, and also used to build the "
                "v6.5 ThermoRun + PredictedRun cache keys at node "
                "entry. Must match ``thermo_aggregate.temperature_k`` "
                "downstream — wire a shared tag node into both ports "
                "when overriding, or the downstream Boltzmann math "
                "will use a different T than the ORCA freq calc."
            ),
        ),
        # ----- NMR section -----
        ConfigField(
            name="run_shielding_h",
            type="bool",
            default=True,
            section="nmr",
            description=(
                "Whether to append a ¹H GIAO shielding job after the "
                "freq+SP."
            ),
        ),
        ConfigField(
            name="run_shielding_c",
            type="bool",
            default=True,
            section="nmr",
            description=(
                "Whether to append a ¹³C GIAO shielding job."
            ),
        ),
        ConfigField(
            name="run_couplings",
            type="bool",
            default=True,
            section="nmr",
            description=(
                "Whether to append a J-coupling (SSCC) job. Requires "
                "``coupling_pairs`` to be non-empty when true; "
                "otherwise ``parse_config`` raises and the run lands "
                "as ``argv_parse_failed``."
            ),
        ),
        ConfigField(
            name="shielding_method_h",
            type="str",
            default=DEFAULT_SHIELDING_METHOD_H,
            section="nmr",
            coercer=normalize_optional_str,
            description=(
                "DFT functional for the ¹H shielding job. ``WP04`` "
                "matches the cheshire ¹H calibration."
            ),
        ),
        ConfigField(
            name="shielding_basis_h",
            type="str",
            default=DEFAULT_SHIELDING_BASIS_H,
            section="nmr",
            coercer=normalize_optional_str,
            description="Basis set for the ¹H shielding job.",
        ),
        ConfigField(
            name="shielding_method_c",
            type="str",
            default=DEFAULT_SHIELDING_METHOD_C,
            section="nmr",
            coercer=normalize_optional_str,
            description=(
                "DFT functional for the ¹³C shielding job. "
                "``wB97X-D`` matches the cheshire ¹³C calibration."
            ),
        ),
        ConfigField(
            name="shielding_basis_c",
            type="str",
            default=DEFAULT_SHIELDING_BASIS_C,
            section="nmr",
            coercer=normalize_optional_str,
            description="Basis set for the ¹³C shielding job.",
        ),
        ConfigField(
            name="coupling_method",
            type="str",
            default=DEFAULT_COUPLING_METHOD,
            section="nmr",
            coercer=normalize_optional_str,
            description=(
                "DFT functional for the J-coupling job. "
                "``mPW1PW91`` matches the Bally-Rablen calibration."
            ),
        ),
        ConfigField(
            name="coupling_basis",
            type="str",
            default=DEFAULT_COUPLING_BASIS,
            section="nmr",
            coercer=normalize_optional_str,
            description="Basis set for the J-coupling job.",
        ),
        ConfigField(
            name="coupling_pairs",
            type="csv",
            default=list(DEFAULT_COUPLING_PAIRS),
            section="nmr",
            description=(
                "ORCA nuclei selectors for the J-coupling block. "
                "Accepts either a list (JSON config) or comma-"
                "separated string (key=value config). Tokens are "
                "ORCA-syntax (e.g. ``\"all H\"``, ``\"1, 4, 7\"``, "
                "``\"all C\"``). Must be non-empty when "
                "``run_couplings=true``."
            ),
            depends_on=("run_couplings",),
        ),
        ConfigField(
            name="coupling_thresh_angstrom",
            type="float",
            default=DEFAULT_COUPLING_THRESH_ANGSTROM,
            section="nmr",
            min_value=0.0,
            description=(
                "``SpinSpinRThresh`` cap (Å) — pairs farther apart "
                "than this are skipped to keep the O(N²) cost down."
            ),
        ),
        ConfigField(
            name="nmr_aux_keywords",
            type="str",
            default=DEFAULT_NMR_AUX_KEYWORDS,
            section="nmr",
            coercer=normalize_optional_str,
            description=(
                "Extra simple-input fragment applied to each NMR "
                "job (alongside ``! NMR``). ``TightSCF`` keeps the "
                "SCF thresholds uniform with the freq/SP jobs."
            ),
        ),
        ConfigField(
            name="system_class",
            type="str",
            default="auto",
            section="nmr",
            coercer=normalize_optional_str,
            description=(
                "Chemistry class profile for the NMR shielding + "
                "coupling jobs. ``auto`` scans the input geometry for "
                "heavy-element triggers and picks the matching profile "
                "(currently: Pd -> 'organopd', else 'organic'). The "
                "'organopd' profile prepends ``ZORA`` to NMR keyword "
                "lines and swaps the default shielding/coupling bases "
                "to ``def2-ZORA-TZVPP`` so HALA contributions on H/C "
                "near Pd are captured correctly. Explicit operator-set "
                "bases pass through unchanged."
            ),
        ),
        ConfigField(
            name="use_cache",
            type="bool",
            default=True,
            coercer=parse_bool,
            description=(
                "Consult the nmr-data ThermoRun + PredictedRun caches "
                "on entry. On hit, skip the whole SLURM array — "
                "decompress central-tree .out files into a local "
                "task_dir tree so thermo_aggregate + nmr_aggregate "
                "downstream walk them transparently. Set to false to "
                "always re-run the freq+SP and NMR jobs."
            ),
        ),
        ConfigField(
            name="force_recompute",
            type="bool",
            default=False,
            coercer=parse_bool,
            description=(
                "Force a fresh SLURM array submission even when the "
                "cache would hit. Useful when investigating ORCA non-"
                "determinism or after a functional-alias bug fix that "
                "should change the values written to the database."
            ),
        ),
    ),
)


# --------------------------------------------------------------------
# Node class
# --------------------------------------------------------------------


class OrcaThermoArray(Node):
    """SLURM-array Freq/thermo runner for an upstream conformer ensemble."""

    step = "orca_thermo_array"
    accepts_upstream = True
    requires_upstream = True

    def parse_config(self, raw: dict[str, Any]) -> dict[str, Any]:
        _reject_empty_keywords(raw)
        cfg = apply_schema(raw, SCHEMA)

        # Two fields share an "absent ≠ explicitly-empty" semantics that
        # the schema layer can't directly express (its empty-→-default
        # short-circuit collapses both into "use the default"). Handle
        # the explicit-empty case here:

        # singlepoint_keywords="" / "none" → disable the SP step.
        if "singlepoint_keywords" in raw:
            cfg["singlepoint_keywords"] = normalize_optional_str(
                raw.get("singlepoint_keywords")
            )

        # coupling_pairs="" → []; the cross-field invariant below catches
        # the "run_couplings=true but explicitly empty" case the legacy
        # parse_config also rejected.
        if "coupling_pairs" in raw:
            raw_cp = raw["coupling_pairs"]
            if isinstance(raw_cp, list):
                cfg["coupling_pairs"] = [
                    str(s).strip() for s in raw_cp if str(s).strip()
                ]
            else:
                cfg["coupling_pairs"] = [
                    s.strip() for s in str(raw_cp).split(",") if s.strip()
                ]

        # Cross-field invariant: J-coupling requested ⇒ non-empty pairs.
        if cfg["run_couplings"] and not cfg["coupling_pairs"]:
            raise ValueError(
                "run_couplings=true but coupling_pairs is empty"
            )
        return cfg

    def run(self, ctx: NodeContext) -> None:
        cfg = ctx.config
        multiplicity = resolve_multiplicity(
            multiplicity=cfg["multiplicity"],
            unpaired_electrons=cfg["unpaired_electrons"],
        )

        # Resolve the NMR system_class profile EARLY — before set_inputs
        # records the basis fields and before the cache check builds
        # the PredictedRunKey. The resolver mutates cfg in place: maps
        # ``system_class=auto`` to a concrete class via element scan of
        # the upstream conformer xyzs, then swaps any NMR-basis fields
        # still at their organic defaults to the profile's values. Doing
        # this before set_inputs and before the cache check guarantees
        # the manifest and the fingerprint both reflect what the NMR
        # jobs actually receive.
        try:
            _resolve_system_class_profile(cfg, ctx=ctx)
        except ValueError as e:
            ctx.fail(f"bad_system_class: {e}")
            return

        # Record resolved config in manifest.inputs FIRST — before
        # any cache check or compute. Anything that fails later (cache
        # helper raises, sbatch errors, ORCA crashes mid-array) still
        # leaves a self-describing manifest behind. Sibling nodes
        # (crest, orca_goat, orca_dft_array) already follow this
        # ordering; bug #56 was that this one didn't.
        ctx.set_inputs(
            max_concurrency=cfg["max_concurrency"],
            charge=cfg["charge"],
            unpaired_electrons=cfg["unpaired_electrons"],
            multiplicity=multiplicity,
            solvent=cfg["solvent"],
            smd_solvent=cfg["smd_solvent"],
            keywords=cfg["keywords"],
            singlepoint_keywords=cfg["singlepoint_keywords"],
            temperature_k=cfg["temperature_k"],
            maxcore=cfg["maxcore"],
            nprocs=cfg["nprocs"],
            time_limit=cfg["time_limit"],
            partition=cfg["partition"],
            orca_module=cfg["orca_module"],
            submit=cfg["submit"],
            monitor=cfg["monitor"],
            monitor_interval_s=cfg["monitor_interval_s"],
            monitor_timeout_min=cfg["monitor_timeout_min"],
            silence_openib=cfg["silence_openib"],
            run_shielding_h=cfg["run_shielding_h"],
            run_shielding_c=cfg["run_shielding_c"],
            run_couplings=cfg["run_couplings"],
            shielding_method_h=cfg["shielding_method_h"],
            shielding_basis_h=cfg["shielding_basis_h"],
            shielding_method_c=cfg["shielding_method_c"],
            shielding_basis_c=cfg["shielding_basis_c"],
            coupling_method=cfg["coupling_method"],
            coupling_basis=cfg["coupling_basis"],
            coupling_pairs=list(cfg["coupling_pairs"]),
            coupling_thresh_angstrom=cfg["coupling_thresh_angstrom"],
            nmr_aux_keywords=cfg["nmr_aux_keywords"],
            system_class=cfg["system_class"],
            # Provenance: the operator-supplied / calibration-table
            # functional name (``shielding_method_*`` / ``coupling_method``)
            # may not be the literal ORCA 6 keyword we end up putting on
            # the ``!`` line. ``resolve_functional_alias`` translates
            # things like WP04 -> B3LYP/G + %method block, wB97X-D ->
            # wB97X-D3, and mPW1PW91 -> mPW1PW. Record both forms so the
            # manifest is unambiguous about which functional definition
            # ORCA actually saw vs. which calibration label
            # ``nmr-aggregate`` will key against downstream.
            shielding_method_h_orca_keyword=resolve_functional_alias(
                cfg["shielding_method_h"]
            )[0],
            shielding_method_c_orca_keyword=resolve_functional_alias(
                cfg["shielding_method_c"]
            )[0],
            coupling_method_orca_keyword=resolve_functional_alias(
                cfg["coupling_method"]
            )[0],
        )

        # v6.5b cache check fires BEFORE any compute work (input
        # staging, SLURM array build, sbatch). On hit, the manifest
        # is fully populated from central-tree .gz files and we
        # return; the whole SLURM round-trip is skipped. Cache
        # infrastructure is deliberately best-effort: a down/offline
        # database is treated exactly like a cache miss so calculations
        # can proceed.
        try:
            cache_hit = self._maybe_emit_cached_manifest_thermo(
                ctx, cfg, multiplicity
            )
        except Exception as e:
            logging_utils.log_warn(
                "orca-thermo cache: lookup unavailable/raised; treating as "
                f"cache miss and continuing: {type(e).__name__}: {e}"
            )
            cache_hit = False
        if cache_hit:
            return

        if ctx.upstream_manifest is None:
            ctx.fail("no_upstream_manifest")
            return

        outputs_dir = ctx.outputs_dir
        outputs_dir.mkdir(parents=True, exist_ok=True)

        # ----- 1) Stage upstream conformers -----
        staged_dir = outputs_dir / "input_conformers"
        try:
            staged_paths = stage_conformer_inputs(
                upstream_artifacts=dict(ctx.upstream_manifest.artifacts),
                staged_dir=staged_dir,
            )
        except Exception as e:
            ctx.fail(f"stage_inputs_failed: {e}")
            return

        n_tasks = len(staged_paths)
        ctx.set_input("n_input_conformers", n_tasks)

        for i, p in enumerate(staged_paths, start=1):
            ctx.add_artifact(
                "files",
                {
                    "label": f"input_conf_{i:04d}",
                    "path_abs": str(p.resolve()),
                    "sha256": sha256_file(p),
                    "format": "xyz",
                    "index": i,
                },
            )

        # Convenience: stage a multi-xyz of all input conformers and
        # publish it as ``xyz_ensemble[input_ensemble]``. This mirrors
        # the legacy node so downstream tooling that walks
        # ``xyz_ensemble`` keeps finding what it expects.
        staged_ensemble = outputs_dir / "input_conformers.xyz"
        concat_xyz_files(staged_paths, staged_ensemble)
        ctx.add_artifact(
            "xyz_ensemble",
            {
                "label": "input_ensemble",
                "path_abs": str(staged_ensemble.resolve()),
                "sha256": sha256_file(staged_ensemble),
                "format": "xyz",
            },
        )

        # ----- 2) Build per-task dirs -----
        array_root = outputs_dir / "array"
        tasks_root = array_root / "tasks"
        slurm_logs = array_root / "slurm_logs"
        array_root.mkdir(parents=True, exist_ok=True)
        tasks_root.mkdir(parents=True, exist_ok=True)
        slurm_logs.mkdir(parents=True, exist_ok=True)

        # Run each method family as a separate ORCA process. Earlier
        # versions kept the r2scan-3c freq and wB97M-V single point in
        # one ``$new_job`` compound input, but ORCA 6.0.0 can leak
        # method-state / exchange-algorithm flags across that boundary
        # (for example RIJONX + RIJCOSX both set when the SP includes
        # ``DIRECT``). Keeping the SP as a fresh process is slightly more
        # expensive but much more robust. The SLURM body appends the SP
        # output into ``orca_thermo.out`` after successful completion so
        # downstream parsers still find the high-level final energy by
        # reading the primary thermo output.
        inp_text = make_orca_compound_input(
            keywords=cfg["keywords"],
            singlepoint_keywords=None,
            post_jobs=None,
            nprocs=cfg["nprocs"],
            maxcore=cfg["maxcore"],
            charge=cfg["charge"],
            multiplicity=multiplicity,
            solvent=cfg["solvent"],
            smd_solvent_override=cfg["smd_solvent"],
            xyz_filename="input.xyz",
            # Thread the operator's T into ORCA's ``%freq Temp`` so the
            # printed thermal corrections (H_corr, TS, etc.) are computed
            # at the same T thermo_aggregate uses for ΔG / Boltzmann
            # weights. Without this, overriding ``temperature_k`` only
            # affected the downstream math, not the freq calc itself.
            freq_temperature_k=cfg["temperature_k"],
        )
        extra_inputs: dict[str, str] = {}
        if cfg["singlepoint_keywords"] is not None:
            extra_inputs[ORCA_SP_INP_NAME] = make_orca_simple_input(
                keywords=cfg["singlepoint_keywords"],
                nprocs=cfg["nprocs"],
                maxcore=cfg["maxcore"],
                charge=cfg["charge"],
                multiplicity=multiplicity,
                solvent=cfg["solvent"],
                smd_solvent_override=cfg["smd_solvent"],
                xyz_filename="input.xyz",
            )

        nmr_inputs = build_nmr_input_files(
            cfg=cfg, multiplicity=multiplicity, xyz_filename="input.xyz",
        )
        extra_inputs.update(nmr_inputs)
        build_thermo_task_dirs(
            staged_paths=staged_paths,
            tasks_root=tasks_root,
            inp_text=inp_text,
            inp_name=ORCA_INP_NAME,
            extra_inputs=extra_inputs,
        )

        # ----- 3) Render SLURM array script -----
        job_name = cfg["job_name"] or f"orca_thermo_array_{n_tasks}"
        # Build the (inp, out) sequence: freq first, optional high-level
        # SP second, then each enabled NMR job. The SP output is appended
        # into ``orca_thermo.out`` by ``multi_orca_per_task_body`` so the
        # thermo aggregator's legacy single-file parser still sees both
        # the low-level thermal corrections and high-level final energy.
        nmr_out_map = {
            ORCA_NMR_H_INP_NAME: ORCA_NMR_H_OUT_NAME,
            ORCA_NMR_C_INP_NAME: ORCA_NMR_C_OUT_NAME,
            ORCA_NMR_J_INP_NAME: ORCA_NMR_J_OUT_NAME,
        }
        orca_jobs: list[tuple[str, str]] = [(ORCA_INP_NAME, ORCA_OUT_NAME)]
        sp_outputs_to_append: list[str] = []
        if cfg["singlepoint_keywords"] is not None:
            orca_jobs.append((ORCA_SP_INP_NAME, ORCA_SP_OUT_NAME))
            sp_outputs_to_append.append(ORCA_SP_OUT_NAME)
        for inp_name in nmr_inputs:
            orca_jobs.append((inp_name, nmr_out_map[inp_name]))
        if len(orca_jobs) == 1:
            per_task_body = standard_orca_per_task_body(
                inp_filename=ORCA_INP_NAME,
                out_filename=ORCA_OUT_NAME,
            )
        else:
            from ..slurm import multi_orca_per_task_body
            per_task_body = multi_orca_per_task_body(
                jobs=orca_jobs,
                append_outputs_to=ORCA_OUT_NAME,
                append_output_names=sp_outputs_to_append,
            )
        slurm_text = make_array_slurm_text(
            job_name=job_name,
            n_tasks=n_tasks,
            max_concurrency=cfg["max_concurrency"],
            nprocs=cfg["nprocs"],
            time_limit=cfg["time_limit"],
            partition=cfg["partition"],
            tasks_root_abs=str(tasks_root.resolve()),
            slurm_logs_abs=str(slurm_logs.resolve()),
            orca_module=cfg["orca_module"],
            silence_openib=cfg["silence_openib"],
            per_task_body=per_task_body,
        )
        slurm_path = array_root / "submit_array.slurm"
        slurm_path.write_text(slurm_text, encoding="utf-8")
        ctx.add_artifact(
            "files",
            {
                "label": "submit_array_slurm",
                "path_abs": str(slurm_path.resolve()),
                "sha256": sha256_file(slurm_path),
                "format": "slurm",
            },
        )

        ctx.manifest.set_array_info(
            tasks_root_abs=str(tasks_root.resolve()),
            n_tasks=n_tasks,
            array_root_abs=str(array_root.resolve()),
            slurm_logs_abs=str(slurm_logs.resolve()),
            submit_slurm_abs=str(slurm_path.resolve()),
            max_concurrency=cfg["max_concurrency"],
            job_name=job_name,
            jobid=None,
            submit_ok=False,
            submit_msg=None,
            progress_last=ProgressCounts.empty(n_tasks).to_dict(),
        )

        # ----- 4) Submit + monitor -----
        execs = discover_slurm_executables()
        ctx.manifest.environment["sbatch"] = execs.sbatch
        ctx.manifest.environment["squeue"] = execs.squeue
        ctx.manifest.environment["sacct"] = execs.sacct

        jobid: str | None = None
        if cfg["submit"]:
            jobid = self._submit(
                ctx,
                execs=execs,
                slurm_path=slurm_path,
                array_root=array_root,
            )

        if cfg["monitor"] and jobid is not None:
            self._monitor(
                ctx,
                jobid=jobid,
                execs=execs,
                tasks_root=tasks_root,
                n_tasks=n_tasks,
                monitor_interval_s=cfg["monitor_interval_s"],
                monitor_timeout_min=cfg["monitor_timeout_min"],
            )

        # ----- 5) Aggregate outputs -----
        do_aggregate = bool(jobid) and bool(cfg["monitor"])
        ctx.manifest.artifacts["array"]["aggregated"] = bool(do_aggregate)
        if do_aggregate:
            self._aggregate(
                ctx,
                outputs_dir=outputs_dir,
                tasks_root=tasks_root,
                staged_dir=staged_dir,
                n_tasks=n_tasks,
                jobid=jobid,
                execs=execs,
            )

        # ----- 6) Self-register the thermo_run + central-tree copy -----
        # Writes the ThermoRun row + copies per-task orca_thermo.out.gz
        # into the central tree. The thermo_aggregate manifest + summary
        # CSV don't exist yet (that node runs downstream); those land
        # via wf-db-ingest as a follow-up populator. Same use_cache gating
        # as the read-side check.
        if ctx.manifest.ok and cfg.get("use_cache", True):
            self._maybe_register_thermo_run(
                ctx, cfg, multiplicity=multiplicity, tasks_root=tasks_root,
            )

    # ------------------------------------------------------------------
    # Producer-side self-registration
    # ------------------------------------------------------------------

    def _maybe_register_thermo_run(
        self,
        ctx: NodeContext,
        cfg: dict[str, Any],
        *,
        multiplicity: int,
        tasks_root: Path,
    ) -> None:
        """Self-register this thermo run to nmr-data, best-effort.

        Mirrors the read-side ``_maybe_emit_cached_manifest_thermo`` key
        derivation: walk upstream for SMILES + CREST + DFT inputs, build
        the parent :class:`DftRunKey` fingerprint, then this node's
        :class:`ThermoKey`. Delegates the registration + per-task log
        copy to the Node base-class helper. Same caveats: GOAT upstreams
        aren't supported here (the cache check is CREST-only too, see
        ``_maybe_emit_cached_manifest_thermo`` line 1213).
        """
        if not _HAS_NMR_DATA:
            logging_utils.log_info(
                f"orca-thermo registry: nmr_data not importable "
                f"({_NMR_DATA_IMPORT_ERROR}), skipping registration"
            )
            return
        if not _HAS_RDKIT:
            logging_utils.log_info(
                f"orca-thermo registry: rdkit not importable "
                f"({_RDKIT_IMPORT_ERROR}), skipping registration"
            )
            return

        smiles = _walk_upstream_for_smiles(ctx)
        crest_dict = _walk_upstream_for_step(ctx, "crest")
        dft_dict = _walk_upstream_for_step(ctx, "orca_dft_array")
        if not smiles or not crest_dict:
            logging_utils.log_info(
                "orca-thermo registry: SMILES or wf_crest manifest not "
                "found in upstream chain, skipping registration"
            )
            return

        crest_inputs = crest_dict.get("inputs") or {}
        dft_inputs = (dft_dict.get("inputs") if dft_dict else None) or {}

        # Build inputs dict the same way the writer side will see it.
        own_inputs: dict[str, Any] = dict(cfg)
        own_inputs["multiplicity"] = multiplicity

        try:
            ens_key = build_crest_ensemble_key(crest_inputs, smiles)
            if ens_key is None:
                logging_utils.log_info(
                    "orca-thermo registry: could not build EnsembleKey, "
                    "skipping registration"
                )
                return
            ens_fp = _cache_fingerprint(ens_key)
            dft_key = build_dft_run_key(dft_inputs, ensemble_fingerprint=ens_fp)
            dft_fp = _cache_fingerprint(dft_key)
            thermo_key = build_thermo_run_key(
                own_inputs,
                dft_run_fingerprint=dft_fp,
                thermo_aggregate_inputs={
                    "temperature_k": cfg["temperature_k"],
                },
            )
        except Exception as e:
            logging_utils.log_warn(
                f"orca-thermo registry: key build raised, "
                f"skipping registration: {type(e).__name__}: {e}"
            )
            return

        # Build conformer_task_dirs from the local tasks_root. Each
        # conf_NNNN holds the orca_thermo.out that copy_thermo_run_artifacts
        # gzip-copies into the central tree.
        conformer_task_dirs: dict[int, Path] = {}
        if tasks_root.is_dir():
            for d in tasks_root.iterdir():
                if d.is_dir() and d.name.startswith("conf_"):
                    try:
                        idx = int(d.name.removeprefix("conf_"))
                        conformer_task_dirs[idx] = d
                    except ValueError:
                        continue

        conformer_records = list(
            ctx.manifest.artifacts.get("conformers", []) or []
        )
        self._try_register_to_nmr_data(
            "thermo_run",
            ctx=ctx,
            parent_dft_run_fingerprint=dft_fp,
            thermo_key=thermo_key,
            conformer_records=conformer_records,
            conformer_task_dirs=conformer_task_dirs or None,
        )

    # ------------------------------------------------------------------
    # v6.5b cache hit path
    # ------------------------------------------------------------------

    def _maybe_emit_cached_manifest_thermo(
        self, ctx: NodeContext, cfg: dict[str, Any], multiplicity: int
    ) -> bool:
        """Joint ThermoRun + PredictedRun cache check.

        Both must hit (this node runs both thermo SP/freq AND NMR
        shielding/coupling jobs in one SLURM array, so a thermo-only
        hit doesn't save the NMR work and a NMR-only hit doesn't save
        the thermo work — only a *both-hit* short-circuits cleanly).

        On hit:
          * Walks central tree at ``<hpc_data_root>/<thermo_run_path>``
            and ``<predicted_run_path>`` (set on the cache rows by
            db_ingest's v6.3 copy step).
          * Gunzips each conformer's ``orca_thermo.out.gz`` from the
            thermo dir + ``orca_nmr_h.out.gz`` / ``_c.gz`` / ``_j.gz``
            from the predicted_run dir into a local
            ``<outputs>/array/tasks/conf_NNNN/`` task_dir.
          * Stages the input.xyz from the ensemble dir into each
            task_dir so anything that reads it still works.
          * Populates ``artifacts.array``, ``artifacts.conformers`` so
            thermo_aggregate + nmr_aggregate downstream walk the
            local tree transparently.

        Returns True if the manifest is in a publishable cache-hit
        state; False on any miss (cache disabled, infra unavailable,
        DB row missing, central-tree dir missing). On False the
        caller runs the SLURM array normally.
        """
        if not cfg.get("use_cache", True):
            logging_utils.log_info(
                "orca-thermo cache: disabled via use_cache=false"
            )
            return False
        if cfg.get("force_recompute", False):
            logging_utils.log_info(
                "orca-thermo cache: force_recompute=true → ignoring cache"
            )
            return False
        if not _HAS_NMR_DATA:
            logging_utils.log_info(
                f"orca-thermo cache: nmr_data not importable "
                f"({_NMR_DATA_IMPORT_ERROR}), skipping check"
            )
            return False
        if not _HAS_RDKIT:
            logging_utils.log_info(
                f"orca-thermo cache: rdkit not importable "
                f"({_RDKIT_IMPORT_ERROR}), skipping check"
            )
            return False
        db_url = os.environ.get("NMR_DATABASE_URL")
        hpc_root = os.environ.get("NMR_HPC_DATA_ROOT")
        if not db_url or not hpc_root:
            logging_utils.log_info(
                "orca-thermo cache: NMR_DATABASE_URL or NMR_HPC_DATA_ROOT "
                "unset, skipping check"
            )
            return False

        # Walk upstream chain for the inputs we need.
        smiles = _walk_upstream_for_smiles(ctx)
        crest_dict = _walk_upstream_for_step(ctx, "crest")
        dft_dict = _walk_upstream_for_step(ctx, "orca_dft_array")
        if not smiles or not crest_dict:
            logging_utils.log_info(
                "orca-thermo cache: SMILES or wf_crest manifest not found "
                "in upstream chain, skipping check"
            )
            return False

        crest_inputs = (crest_dict.get("inputs") or {})
        dft_inputs = (dft_dict.get("inputs") if dft_dict else None) or {}

        # Construct the same set of inputs the writer side will see in
        # this node's manifest. set_inputs hasn't fired yet (we return
        # *before* it), so build the dict from cfg in the same shape.
        own_inputs: dict[str, Any] = dict(cfg)
        own_inputs["multiplicity"] = multiplicity

        # Build the four keys + their fingerprints in dependency order.
        # ConformerEnsemble (from CREST/GOAT) → DftRun (from this
        # node's upstream orca_dft_array) → ThermoRun (from own inputs)
        # → PredictedRun (NMR side, also from own inputs).
        try:
            ens_key = build_crest_ensemble_key(crest_inputs, smiles)
            if ens_key is None:
                logging_utils.log_info(
                    "orca-thermo cache: could not build EnsembleKey "
                    "(rdkit / SMILES parse failure), skipping check"
                )
                return False
            ens_fp = _cache_fingerprint(ens_key)
            dft_key = build_dft_run_key(dft_inputs, ensemble_fingerprint=ens_fp)
            dft_fp = _cache_fingerprint(dft_key)
            # ``thermo_aggregate_inputs`` is downstream and not visible
            # here, but the v6.5 thermo-key build needs ``temperature_k``
            # + ``standard_state`` from it. We forward the values from
            # *this* node's config — the operator is responsible for
            # piping the same numbers to both nodes via a tag node, and
            # the writer side prefers this node's value when present
            # (see ``ingest.py``'s thermo-key build). standard_state is
            # left to the build_thermo_run_key default ("1m") since this
            # node has no opinion on it.
            thermo_key = build_thermo_run_key(
                own_inputs,
                dft_run_fingerprint=dft_fp,
                thermo_aggregate_inputs={
                    "temperature_k": cfg["temperature_k"],
                },
            )
            thermo_fp = _cache_fingerprint(thermo_key)
            predicted_key = build_predicted_run_key(
                own_inputs,
                thermo_run_fingerprint=thermo_fp,
                temperature_k=cfg["temperature_k"],
            )
        except Exception as e:
            logging_utils.log_warn(
                f"orca-thermo cache: key build raised, falling through: {e}"
            )
            return False

        # DB lookup — all four rows must exist for the joint hit.
        _orig_url = os.environ.get("NMR_DATABASE_URL")
        os.environ["NMR_DATABASE_URL"] = db_url
        try:
            with get_session() as session:
                inchikey = inchikey_from_smiles(smiles)
                if not inchikey:
                    return False
                mol = session.query(Molecule).filter_by(
                    inchikey=inchikey
                ).one_or_none()
                if mol is None:
                    logging_utils.log_info(
                        f"orca-thermo cache: no molecule row for {inchikey}, "
                        f"skipping"
                    )
                    return False
                ensemble = find_ensemble(session, molecule_id=mol.id, key=ens_key)
                if ensemble is None or not ensemble.ensemble_path:
                    logging_utils.log_info(
                        "orca-thermo cache: no matching ensemble row"
                    )
                    return False
                dft_row = find_dft_run(
                    session, ensemble_id=ensemble.id, key=dft_key,
                )
                if dft_row is None or not dft_row.dft_run_path:
                    logging_utils.log_info(
                        "orca-thermo cache: no matching dft_run row"
                    )
                    return False
                thermo_row = find_thermo_run(
                    session, dft_run_id=dft_row.id, key=thermo_key,
                )
                if thermo_row is None or not thermo_row.thermo_run_path:
                    logging_utils.log_info(
                        "orca-thermo cache: no matching thermo_run row"
                    )
                    return False
                predicted_row = find_predicted_run(
                    session, molecule_id=mol.id, key=predicted_key,
                )
                if predicted_row is None or not predicted_row.run_root_path:
                    logging_utils.log_info(
                        "orca-thermo cache: no matching predicted_run row"
                    )
                    return False
                # Snapshot paths since the row detaches when the session ends.
                ensemble_path_rel = ensemble.ensemble_path
                dft_path_rel = dft_row.dft_run_path
                thermo_path_rel = thermo_row.thermo_run_path
                predicted_path_rel = predicted_row.run_root_path
        finally:
            if _orig_url is None:
                os.environ.pop("NMR_DATABASE_URL", None)
            else:
                os.environ["NMR_DATABASE_URL"] = _orig_url

        hpc_root_path = Path(hpc_root)
        thermo_abs = hpc_root_path / thermo_path_rel
        predicted_abs = hpc_root_path / predicted_path_rel
        ensemble_abs = hpc_root_path / ensemble_path_rel
        dft_abs = hpc_root_path / dft_path_rel
        for label, p in (
            ("thermo", thermo_abs),
            ("predicted", predicted_abs),
            ("ensemble", ensemble_abs),
            ("dft_run", dft_abs),
        ):
            if not p.is_dir():
                logging_utils.log_warn(
                    f"orca-thermo cache: {label} dir missing on disk at "
                    f"{p} — falling through to compute"
                )
                return False

        # Enumerate cached conformers via thermo_runs/<uuid>/conformers/.
        thermo_confs_dir = thermo_abs / "conformers"
        conf_dirs = sorted(
            d for d in thermo_confs_dir.iterdir()
            if d.is_dir() and d.name.startswith("conf_")
        )
        if not conf_dirs:
            logging_utils.log_warn(
                f"orca-thermo cache: thermo dir {thermo_confs_dir} has no "
                f"conformers, falling through"
            )
            return False

        # Materialize the local task_dir tree by decompressing .gz files
        # from the three central-tree subtrees.
        outputs_dir = ctx.outputs_dir
        outputs_dir.mkdir(parents=True, exist_ok=True)
        array_root = outputs_dir / "array"
        tasks_root = array_root / "tasks"
        tasks_root.mkdir(parents=True, exist_ok=True)

        n_tasks = 0
        for thermo_conf in conf_dirs:
            try:
                idx = int(thermo_conf.name.removeprefix("conf_"))
            except ValueError:
                continue
            local_task = tasks_root / f"conf_{idx:04d}"
            local_task.mkdir(parents=True, exist_ok=True)
            n_tasks += 1

            # 1) thermo .out.gz
            t_gz = thermo_conf / "orca_thermo.out.gz"
            if t_gz.exists():
                _gunzip_to(t_gz, local_task / ORCA_OUT_NAME)

            # 2) NMR .out.gz files from predicted_runs/<uuid>/conformers/<same>/
            pr_conf = predicted_abs / "conformers" / thermo_conf.name
            if pr_conf.is_dir():
                for nmr_name, dst_name in (
                    ("orca_nmr_h.out.gz", ORCA_NMR_H_OUT_NAME),
                    ("orca_nmr_c.out.gz", ORCA_NMR_C_OUT_NAME),
                    ("orca_nmr_j.out.gz", ORCA_NMR_J_OUT_NAME),
                ):
                    src = pr_conf / nmr_name
                    if src.exists():
                        _gunzip_to(src, local_task / dst_name)

            # 3) input.xyz from dft_runs/<uuid>/conformers/<same>/
            # (DFT-optimized geometries — what this node's freq + SP
            # jobs actually run on). Note: ensembles/<uuid>/ now
            # contains xtb-level geometries from CREST/GOAT — wrong
            # source for thermo.
            dft_conf = dft_abs / "conformers" / thermo_conf.name
            if dft_conf.is_dir():
                xyzs = sorted(dft_conf.glob("*.xyz"))
                if xyzs:
                    shutil.copy2(xyzs[0], local_task / "input.xyz")

            ctx.add_artifact(
                "conformers",
                {
                    "index": idx,
                    "label": f"conf_{idx:04d}",
                    "task_dir_abs": str(local_task.resolve()),
                    "path_abs": str((local_task / "input.xyz").resolve()),
                    "format": "xyz",
                    "cache_hit": True,
                },
            )

        # Top-level array metadata mirrors what set_array_info would write
        # on a fresh run, minus SLURM fields (jobid, submit_ok, etc.)
        # which don't apply on a hit.
        ctx.manifest.set_array_info(
            tasks_root_abs=str(tasks_root.resolve()),
            n_tasks=n_tasks,
            array_root_abs=str(array_root.resolve()),
            aggregated=True,
            cache_hit=True,
        )

        # Manifest-level inputs that the cache fan-out makes visible.
        ctx.set_input("cache_hit", True)
        ctx.set_input("cached_thermo_run_path", thermo_path_rel)
        ctx.set_input("cached_predicted_run_path", predicted_path_rel)
        ctx.set_input("cached_dft_run_path", dft_path_rel)
        ctx.set_input("cached_ensemble_path", ensemble_path_rel)
        ctx.set_input("n_cached_conformers", n_tasks)

        logging_utils.log_info(
            f"orca-thermo cache: HIT — thermo_run={thermo_path_rel}, "
            f"predicted_run={predicted_path_rel} ({n_tasks} conformer(s)). "
            f"Skipping SLURM array."
        )
        return True

    # ------------------------------------------------------------------
    # Step handlers
    # ------------------------------------------------------------------

    def _submit(
        self,
        ctx: NodeContext,
        *,
        execs: SlurmExecutables,
        slurm_path: Path,
        array_root: Path,
    ) -> str | None:
        if not execs.sbatch:
            ctx.fail("sbatch_not_found_on_PATH")
            return None

        ok, jobid, msg = sbatch_submit(execs.sbatch, slurm_path, cwd=array_root)
        ctx.manifest.artifacts["array"]["submit_ok"] = bool(ok)
        ctx.manifest.artifacts["array"]["submit_msg"] = msg
        ctx.manifest.artifacts["array"]["jobid"] = jobid

        if not ok or not jobid:
            ctx.fail("sbatch_failed", details=msg)
            return None

        n_tasks = ctx.manifest.artifacts["array"]["n_tasks"]
        max_conc = ctx.manifest.artifacts["array"]["max_concurrency"]
        logging_utils.log_info(
            f"orca-thermo-array: submitted SLURM array job -> jobid {jobid} "
            f"(array 1-{n_tasks}%{max_conc})"
        )
        return jobid

    def _monitor(
        self,
        ctx: NodeContext,
        *,
        jobid: str,
        execs: SlurmExecutables,
        tasks_root: Path,
        n_tasks: int,
        monitor_interval_s: int,
        monitor_timeout_min: int,
    ) -> None:
        if not execs.squeue:
            ctx.fail("monitor_requested_but_squeue_not_found")
            return

        squeue_exe = execs.squeue

        def _squeue_check(j: str) -> bool:
            return squeue_has_any(squeue_exe, j)

        def _progress(root: Path, n: int) -> ProgressCounts:
            # The thermo task body, like the DFT body, only writes the
            # success/failed sentinels AFTER ORCA exits. While ORCA is
            # still running the only evidence of "task started" is the
            # in-place orca_thermo.out file. Pass it through as an
            # additional started-signal so partial-output tasks are
            # counted as in_progress rather than left.
            return count_task_progress(
                root, n, started_extra_signals=(ORCA_OUT_NAME,)
            )

        result: MonitorResult = monitor_array_job(
            jobid=jobid,
            tasks_root=tasks_root,
            n_tasks=n_tasks,
            monitor_interval_s=monitor_interval_s,
            monitor_timeout_min=monitor_timeout_min,
            squeue_check=_squeue_check,
            progress_fn=_progress,
            log_fn=logging_utils.log_info,
        )

        ctx.manifest.artifacts["array"]["progress_final"] = (
            result.final_progress.to_dict()
        )
        ctx.manifest.artifacts["array"]["progress_last"] = (
            result.final_progress.to_dict()
        )
        ctx.manifest.artifacts["array"]["monitor_iterations"] = (
            result.iterations
        )

        if result.timed_out:
            ctx.fail(
                "monitor_timeout",
                jobid=jobid,
                progress=result.final_progress.to_dict(),
            )

        # If the SLURM-side sentinel walker reported any failed tasks
        # (``.wf_status/done_failed`` written by the per-task body),
        # surface that at the manifest level immediately. Without this
        # the downstream ``_aggregate`` step can declare the run "ok"
        # whenever the freq+SP outputs parse successfully — even when
        # one of the *separate* NMR ORCA invocations crashed and left
        # a ``done_failed`` marker. Keeps ``ok``/``failures`` honest
        # against ``progress_final``.
        if result.final_progress.failed > 0:
            ctx.fail(
                "tasks_marked_failed",
                jobid=jobid,
                n_failed=result.final_progress.failed,
                n_total=n_tasks,
                progress=result.final_progress.to_dict(),
            )

    def _aggregate(
        self,
        ctx: NodeContext,
        *,
        outputs_dir: Path,
        tasks_root: Path,
        staged_dir: Path,
        n_tasks: int,
        jobid: str | None,
        execs: SlurmExecutables,
    ) -> None:
        thermo_dir = outputs_dir / "thermo"
        thermo_dir.mkdir(parents=True, exist_ok=True)

        conf_records, energies_h, failure_records = collect_thermo_outputs(
            n_tasks=n_tasks,
            tasks_root=tasks_root,
            staged_dir=staged_dir,
            out_name=ORCA_OUT_NAME,
        )

        for rec in conf_records:
            ctx.add_artifact("conformers", rec)
        for fail_rec in failure_records:
            ctx.fail(fail_rec.pop("error"), **fail_rec)

        # 3-column thermo.energies file (index, abs_Eh, rel_kcal). The
        # file lives in outputs/thermo/ rather than
        # outputs/optimized_conformers/ so a single workflow
        # (opt → thermo) writes both ``orca.energies`` AND
        # ``thermo.energies`` and the consumer can disambiguate.
        energies_path = thermo_dir / "thermo.energies"
        rel_kcal, e_min = write_energy_file(
            energies_h=energies_h, out_path=energies_path
        )
        ctx.add_artifact(
            "files",
            {
                "label": "thermo_energies",
                "path_abs": str(energies_path.resolve()),
                "sha256": sha256_file(energies_path),
                "format": "txt",
            },
        )

        # Attach rel_energy_kcal in-place. Records are 1:1 with the
        # task index, so we can index rel_kcal by (rec.index - 1).
        for rec in ctx.manifest.artifacts.get("conformers", []):
            idx = rec.get("index")
            if isinstance(idx, int) and 1 <= idx <= n_tasks:
                rk = rel_kcal[idx - 1]
                if rk is not None:
                    rec["rel_energy_kcal"] = float(rk)

        # ``best`` = lowest absolute energy, sourced from the staged
        # input geometry (the freq run preserves coordinates so this
        # IS the geometry the thermochemistry refers to).
        if e_min is not None:
            finite_pairs = [
                (i + 1, e) for i, e in enumerate(energies_h) if e is not None
            ]
            if finite_pairs:
                best_idx, _ = min(finite_pairs, key=lambda t: t[1])
                best_src = staged_dir / f"conf_{best_idx:04d}.xyz"
                if best_src.exists():
                    best_dst = thermo_dir / "best.xyz"
                    shutil.copy2(best_src, best_dst)
                    ctx.add_artifact(
                        "xyz",
                        {
                            "label": "best",
                            "path_abs": str(best_dst.resolve()),
                            "sha256": sha256_file(best_dst),
                            "format": "xyz",
                            "index": int(best_idx),
                        },
                    )

        # sacct post-mortem — surface per-task failures as structured
        # records on top of whatever the sentinels said. Best-effort.
        if execs.sacct and jobid:
            states = sacct_states(execs.sacct, jobid)
            ctx.manifest.artifacts["array"]["sacct"] = {
                k: {"state": v[0], "exitcode": v[1]} for k, v in states.items()
            }
            for fail_rec in sacct_failures_for_array(
                states, jobid=jobid, n_tasks=n_tasks
            ):
                ctx.fail(fail_rec.pop("error"), **fail_rec)


__all__ = [
    "DEFAULT_COUPLING_BASIS",
    "DEFAULT_COUPLING_METHOD",
    "DEFAULT_COUPLING_PAIRS",
    "DEFAULT_COUPLING_THRESH_ANGSTROM",
    "DEFAULT_KEYWORDS",
    "DEFAULT_NMR_AUX_KEYWORDS",
    "DEFAULT_NPROCS",
    "DEFAULT_ORCA_MODULE",
    "DEFAULT_SHIELDING_BASIS_C",
    "DEFAULT_SHIELDING_BASIS_H",
    "DEFAULT_SHIELDING_METHOD_C",
    "DEFAULT_SHIELDING_METHOD_H",
    "DEFAULT_SINGLEPOINT_KEYWORDS",
    "ORCA_INP_NAME",
    "ORCA_OUT_NAME",
    "OrcaThermoArray",
    "build_nmr_input_files",
    "ORCA_NMR_H_INP_NAME",
    "ORCA_NMR_H_OUT_NAME",
    "ORCA_NMR_C_INP_NAME",
    "ORCA_NMR_C_OUT_NAME",
    "ORCA_NMR_J_INP_NAME",
    "ORCA_NMR_J_OUT_NAME",
    "build_thermo_task_dirs",
    "collect_thermo_outputs",
    "main",
]


main = OrcaThermoArray.invoke_factory()


if __name__ == "__main__":
    raise SystemExit(main())
