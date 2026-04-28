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
                           skip the SP step entirely.                   ["wB97M-V def2-TZVPP TightSCF"]
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

import shutil
from pathlib import Path
from typing import Any

from .. import logging_utils
from ..hashing import sha256_file
from ..node import Node, NodeContext
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
from ..parsing import (
    normalize_optional_str,
    parse_bool,
    parse_float,
    parse_int,
    parse_optional_int,
)
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
    normalize_max_concurrency,
    resolve_multiplicity,
    stage_conformer_inputs,
)


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
DEFAULT_SINGLEPOINT_KEYWORDS: str = "wB97M-V def2-TZVPP TightSCF RIJCOSX DEFGRID3"

#: Default ORCA ``%pal nprocs`` (and SLURM ``--ntasks``). Matches the
#: rest of the array nodes' default of 8 — Frequency calculations
#: parallelize well, but 8 is the cluster-wide sweet spot.
DEFAULT_NPROCS: int = 8

#: Per-task ORCA input/output filenames. Note there is no
#: ``ORCA_OPT_XYZ`` analogue — frequency runs preserve the input
#: geometry rather than producing a new one.
ORCA_INP_NAME: str = "orca_thermo.inp"
ORCA_OUT_NAME: str = "orca_thermo.out"

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
            keywords=f"NMR {h_method} {cfg['shielding_basis_h']}{aux_suffix}",
            extra_blocks=[*h_extras, nmr_shielding_block("all H")],
            **common,
        )

    if cfg.get("run_shielding_c"):
        c_method, c_extras = resolve_functional_alias(cfg["shielding_method_c"])
        files[ORCA_NMR_C_INP_NAME] = make_orca_simple_input(
            keywords=f"NMR {c_method} {cfg['shielding_basis_c']}{aux_suffix}",
            extra_blocks=[*c_extras, nmr_shielding_block("all C")],
            **common,
        )

    if cfg.get("run_couplings"):
        pairs = list(cfg.get("coupling_pairs") or DEFAULT_COUPLING_PAIRS)
        thresh = cfg.get("coupling_thresh_angstrom")
        j_method, j_extras = resolve_functional_alias(cfg["coupling_method"])
        files[ORCA_NMR_J_INP_NAME] = make_orca_simple_input(
            keywords=f"NMR {j_method} {cfg['coupling_basis']}{aux_suffix}",
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
# Node class
# --------------------------------------------------------------------


class OrcaThermoArray(Node):
    """SLURM-array Freq/thermo runner for an upstream conformer ensemble."""

    step = "orca_thermo_array"
    accepts_upstream = True
    requires_upstream = True

    def parse_config(self, raw: dict[str, Any]) -> dict[str, Any]:
        max_concurrency = normalize_max_concurrency(raw)

        unpaired = parse_int(raw.get("unpaired_electrons"), 0)
        mult_override = parse_optional_int(raw.get("multiplicity"))

        keywords = str(raw.get("keywords", DEFAULT_KEYWORDS)).strip()
        if not keywords:
            raise ValueError("keywords must be non-empty")

        # The SP step keywords. Distinguish "user did not set the key at
        # all" (use the default) from "user explicitly set it to a
        # null-ish value" (disable the SP step). ``normalize_optional_str``
        # collapses ``None``/``""``/``"none"``/``"null"`` to ``None``,
        # which is the disable signal.
        if "singlepoint_keywords" in raw:
            singlepoint_keywords = normalize_optional_str(
                raw.get("singlepoint_keywords")
            )
        else:
            singlepoint_keywords = DEFAULT_SINGLEPOINT_KEYWORDS

        maxcore = max(500, parse_int(raw.get("maxcore"), 4000))
        nprocs = max(1, parse_int(raw.get("nprocs"), DEFAULT_NPROCS))

        time_limit = str(raw.get("time_limit", "12:00:00")).strip() or "12:00:00"
        partition = normalize_optional_str(raw.get("partition"))

        orca_module = str(
            raw.get("orca_module", DEFAULT_ORCA_MODULE)
        ).strip() or DEFAULT_ORCA_MODULE

        monitor_interval_s = max(5, parse_int(raw.get("monitor_interval_s"), 60))
        monitor_timeout_min = max(0, parse_int(raw.get("monitor_timeout_min"), 0))

        # ---- NMR section ----
        # Three booleans gate the optional shielding/coupling jobs
        # appended after the freq+SP. Defaults are ``True`` because
        # this node is the heart of the ``NMR Predictor`` workflow —
        # the downstream ``nmr_aggregate`` aborts with
        # ``no_shielding_data_in_any_conformer`` whenever none of the
        # three are run. Method/basis fall back to the cheshire
        # defaults so an operator who flips one off (e.g.
        # ``run_couplings=false`` to skip J-couplings on a big system)
        # gets the same recipe the matching ``nmr_aggregate``
        # calibration was fit for. Set all three to ``false`` to
        # degrade gracefully back to a pure freq+SP run.
        run_shielding_h = parse_bool(raw.get("run_shielding_h"), True)
        run_shielding_c = parse_bool(raw.get("run_shielding_c"), True)
        run_couplings = parse_bool(raw.get("run_couplings"), True)

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

        # ``coupling_pairs`` accepts either a list (from JSON config)
        # or a comma-separated string (from key=value config). Each
        # entry is an ORCA nuclei selector (``"all H"``, ``"1, 4, 7"``,
        # ``"all C"``, ...).
        raw_cp = raw.get("coupling_pairs")
        coupling_pairs: list[str]
        if raw_cp is None:
            coupling_pairs = list(DEFAULT_COUPLING_PAIRS)
        elif isinstance(raw_cp, list):
            coupling_pairs = [str(s).strip() for s in raw_cp if str(s).strip()]
        else:
            coupling_pairs = [
                s.strip() for s in str(raw_cp).split(",") if s.strip()
            ]
        if run_couplings and not coupling_pairs:
            raise ValueError(
                "run_couplings=true but coupling_pairs is empty"
            )

        coupling_thresh_angstrom = parse_float(
            raw.get("coupling_thresh_angstrom"),
            DEFAULT_COUPLING_THRESH_ANGSTROM,
        )

        nmr_aux_keywords = (
            normalize_optional_str(raw.get("nmr_aux_keywords"))
            or DEFAULT_NMR_AUX_KEYWORDS
        )

        return {
            "max_concurrency": max_concurrency,
            "charge": parse_int(raw.get("charge"), 0),
            "unpaired_electrons": unpaired,
            "multiplicity": mult_override,
            "solvent": normalize_optional_str(raw.get("solvent")),
            "smd_solvent": normalize_optional_str(raw.get("smd_solvent")),
            "keywords": keywords,
            "singlepoint_keywords": singlepoint_keywords,
            "maxcore": maxcore,
            "nprocs": nprocs,
            "time_limit": time_limit,
            "partition": partition,
            "job_name": normalize_optional_str(raw.get("job_name")),
            "orca_module": orca_module,
            "submit": parse_bool(raw.get("submit"), True),
            "monitor": parse_bool(raw.get("monitor"), True),
            "monitor_interval_s": monitor_interval_s,
            "monitor_timeout_min": monitor_timeout_min,
            "silence_openib": parse_bool(raw.get("silence_openib"), True),
            # NMR knobs (forwarded to build_nmr_input_files in run()):
            "run_shielding_h": run_shielding_h,
            "run_shielding_c": run_shielding_c,
            "run_couplings": run_couplings,
            "shielding_method_h": shielding_method_h,
            "shielding_basis_h": shielding_basis_h,
            "shielding_method_c": shielding_method_c,
            "shielding_basis_c": shielding_basis_c,
            "coupling_method": coupling_method,
            "coupling_basis": coupling_basis,
            "coupling_pairs": coupling_pairs,
            "coupling_thresh_angstrom": coupling_thresh_angstrom,
            "nmr_aux_keywords": nmr_aux_keywords,
        }

    def run(self, ctx: NodeContext) -> None:
        cfg = ctx.config
        multiplicity = resolve_multiplicity(
            multiplicity=cfg["multiplicity"],
            unpaired_electrons=cfg["unpaired_electrons"],
        )

        ctx.set_inputs(
            max_concurrency=cfg["max_concurrency"],
            charge=cfg["charge"],
            unpaired_electrons=cfg["unpaired_electrons"],
            multiplicity=multiplicity,
            solvent=cfg["solvent"],
            smd_solvent=cfg["smd_solvent"],
            keywords=cfg["keywords"],
            singlepoint_keywords=cfg["singlepoint_keywords"],
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

        # The freq+SP compound stays as ONE ORCA invocation (the
        # r2scan-3c → wB97M-V transition is well-covered by the
        # auto-injected DFTDOPT/DoGCP reset block). The NMR jobs,
        # however, are emitted as SEPARATE standalone ORCA inputs —
        # one ``orca_nmr_*.inp`` each — so method-state flags like
        # VV10/NL can't leak from the wB97M-V SP into the chemically
        # unrelated WP04/wB97X-D3/mPW1PW91 NMR functionals. Each NMR
        # job runs as a fresh ORCA process, paying ~5–10 s of init
        # overhead for bulletproof state isolation.
        inp_text = make_orca_compound_input(
            keywords=cfg["keywords"],
            singlepoint_keywords=cfg["singlepoint_keywords"],
            post_jobs=None,
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
        build_thermo_task_dirs(
            staged_paths=staged_paths,
            tasks_root=tasks_root,
            inp_text=inp_text,
            inp_name=ORCA_INP_NAME,
            extra_inputs=nmr_inputs,
        )

        # ----- 3) Render SLURM array script -----
        job_name = cfg["job_name"] or f"orca_thermo_array_{n_tasks}"
        # Build the (inp, out) sequence: freq+SP first, then each
        # enabled NMR job as its own ORCA call. mapping is locked to
        # ORCA_NMR_*_INP_NAME / ORCA_NMR_*_OUT_NAME constants so
        # nmr_aggregate can find them by name without rummaging.
        nmr_out_map = {
            ORCA_NMR_H_INP_NAME: ORCA_NMR_H_OUT_NAME,
            ORCA_NMR_C_INP_NAME: ORCA_NMR_C_OUT_NAME,
            ORCA_NMR_J_INP_NAME: ORCA_NMR_J_OUT_NAME,
        }
        orca_jobs: list[tuple[str, str]] = [(ORCA_INP_NAME, ORCA_OUT_NAME)]
        for inp_name in nmr_inputs:
            orca_jobs.append((inp_name, nmr_out_map[inp_name]))
        if len(orca_jobs) == 1:
            per_task_body = standard_orca_per_task_body(
                inp_filename=ORCA_INP_NAME,
                out_filename=ORCA_OUT_NAME,
            )
        else:
            from ..slurm import multi_orca_per_task_body
            per_task_body = multi_orca_per_task_body(jobs=orca_jobs)
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
