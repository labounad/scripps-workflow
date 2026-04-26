"""``wf-orca-dft-array`` — DFT geometry optimization for an ensemble.

This node fans an upstream conformer ensemble out across one SLURM
array job, runs ORCA in each task directory, monitors progress, and
aggregates the optimized geometries + energies back into a manifest
shaped like the smaller chain nodes' (``conformers`` / ``xyz_ensemble``
/ ``xyz`` / ``files``) so downstream nodes (``prism_screen``,
``marc_screen``, ``thermo_aggregate``) consume it identically to the
crest / orca_goat outputs.

Why an array node? A 50-conformer optimization on a HPC cluster is
~50 × ``r2scan-3c TightOpt`` runs that are each independent: scheduling
them as separate SLURM jobs floods the controller with bookkeeping for
no parallelism gain, while wrapping them in one ``--array=1-N%M`` job
lets the cluster's fair-share scheduler do the right thing. The ``%M``
concurrency cap is the operator's knob for "how rude are we willing to
be to other lab users."

The SLURM glue (``sbatch_submit``, ``squeue_has_any``, monitor loop,
sentinel-file progress counting, array-script generation) lives in
:mod:`scripps_workflow.slurm` and is shared with the upcoming
``orca_thermo_array`` node. The ORCA-input-file rendering and energy
parsing live in :mod:`scripps_workflow.orca`.

Config keys (``key=value`` tokens or one JSON object):

    max_concurrency        ``%M`` in ``--array=1-N%M`` (also accepts
                           ``batchsize``/``max_nodes`` aliases)        [10]
    charge                 int                                          [0]
    unpaired_electrons     int (multiplicity = unpaired + 1)            [0]
    multiplicity           override int (wins over unpaired_electrons)  [None]
    solvent                SMD solvent token, or null/none for vacuum   [None]
    smd_solvent            verbatim SMDsolvent override (bypasses
                           the alias map)                               [None]
    keywords               ORCA simple-input ``!`` line (no leading
                           ``!`` required)                              ["r2scan-3c TightSCF TightOpt"]
    maxcore                MB per ORCA process (clamped to >= 500)      [4000]
    nprocs                 ``%pal nprocs`` and ``--ntasks``             [8]
    time_limit             SBATCH ``-t``                                ["12:00:00"]
    partition              optional SBATCH ``-p``                       [None]
    job_name               SBATCH ``-J`` (defaults to
                           ``orca_opt_array_<n>``)
    orca_module            module-load string                           ["orca/6.0.0"]
    submit                 actually call ``sbatch``?                    [true]
    monitor                actually wait for the job?                   [true]
    monitor_interval_s     polling interval (seconds; clamped >= 5)    [60]
    monitor_timeout_min    wall-clock cap (0 = no cap)                  [0]
    silence_openib         set ``OMPI_MCA_btl=^openib`` in the array   [true]
    fail_policy            ``"soft"`` (default) or ``"hard"``

Ports the legacy ``geometry_opt_orca_v1`` script onto the new framework
verbatim — the on-disk artifact layout (``outputs/array/tasks/...``,
``outputs/optimized_conformers/...``) is preserved so any in-flight
runs already on the cluster keep working when an operator switches
their wrapper to ``wf-orca-dft-array``.
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
    make_orca_simple_input,
    parse_orca_final_energy,
    write_energy_file,
)
from ..parsing import (
    normalize_optional_str,
    parse_bool,
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

# Re-use the conformer source discovery helper that prism / marc share.
from .crest import XyzBlock, split_multixyz, write_xyz_block
from .prism_screen import discover_conformer_sources


#: ORCA module name to ``module load`` inside the SLURM array script.
DEFAULT_ORCA_MODULE: str = "orca/6.0.0"

#: Default ORCA simple-input keywords for geometry optimization.
DEFAULT_KEYWORDS: str = "r2scan-3c TightSCF TightOpt"

#: ORCA writes the optimized geometry to ``<basename>.xyz`` (single
#: frame, the "last optimization step" coordinates) regardless of
#: which composite or DFT functional was used. We collect that file
#: as the per-conformer output.
ORCA_INP_NAME: str = "orca_opt.inp"
ORCA_OUT_NAME: str = "orca_opt.out"
ORCA_OPT_XYZ: str = "orca_opt.xyz"


# --------------------------------------------------------------------
# Pure helpers (testable)
# --------------------------------------------------------------------


def normalize_max_concurrency(raw_cfg: dict[str, Any]) -> int:
    """Read ``max_concurrency`` (with ``batchsize`` / ``max_nodes``
    aliases) and clamp to ``>= 1``."""
    v = raw_cfg.get(
        "max_concurrency",
        raw_cfg.get("max_nodes", raw_cfg.get("batchsize", 10)),
    )
    return max(1, parse_int(v, 10))


def resolve_multiplicity(
    *, multiplicity: int | None, unpaired_electrons: int
) -> int:
    """Either an explicit ``multiplicity`` override or
    ``unpaired_electrons + 1``."""
    if multiplicity is not None:
        return int(multiplicity)
    return int(unpaired_electrons) + 1


def stage_conformer_inputs(
    *,
    upstream_artifacts: dict[str, Any],
    staged_dir: Path,
) -> list[Path]:
    """Materialize a list of single-frame xyz files into ``staged_dir``.

    Uses :func:`prism_screen.discover_conformer_sources` to pick the
    highest-priority conformer bucket on the upstream manifest. If the
    upstream produced an ensemble (single multi-xyz file), the file is
    split into per-conformer frames; otherwise per-conformer files are
    copied across verbatim.

    Returns the staged paths in 1-based order. Raises ``RuntimeError``
    if no usable xyz inputs were found, or if the result is empty.
    """
    staged_dir.mkdir(parents=True, exist_ok=True)
    mode, items = discover_conformer_sources(upstream_artifacts)
    if mode == "none" or not items:
        raise RuntimeError("no_xyz_inputs_found_in_upstream_manifest")

    staged_paths: list[Path] = []

    if mode in {"ensemble", "single"} and len(items) == 1:
        src = Path(items[0]["path_abs"])
        text = src.read_text(encoding="utf-8", errors="replace")
        blocks = split_multixyz(text)
        if len(blocks) >= 2:
            for i, blk in enumerate(blocks, start=1):
                p = staged_dir / f"conf_{i:04d}.xyz"
                write_xyz_block(p, blk)
                staged_paths.append(p)
        else:
            p = staged_dir / "conf_0001.xyz"
            shutil.copy2(src, p)
            staged_paths.append(p)
    else:
        # mode == "many" — copy each per-conformer xyz across verbatim.
        for i, it in enumerate(items, start=1):
            src = Path(it["path_abs"])
            dst = staged_dir / f"conf_{i:04d}.xyz"
            shutil.copy2(src, dst)
            staged_paths.append(dst)

    if not staged_paths:
        raise RuntimeError("no_staged_conformers")
    return staged_paths


def build_task_dirs(
    *,
    staged_paths: list[Path],
    tasks_root: Path,
    inp_text: str,
    inp_name: str = ORCA_INP_NAME,
) -> None:
    """Per-conformer task dir, each with ``input.xyz`` + ``<inp_name>``.

    ``inp_text`` is the same for every task — the per-frame difference
    is only the ``input.xyz`` geometry; the ORCA input file references
    that file by relative name, so a single-shared text is correct.
    """
    tasks_root.mkdir(parents=True, exist_ok=True)
    for i, src_xyz in enumerate(staged_paths, start=1):
        task_dir = tasks_root / f"task_{i:04d}"
        task_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_xyz, task_dir / "input.xyz")
        (task_dir / inp_name).write_text(inp_text, encoding="utf-8")


def collect_optimized_outputs(
    *,
    n_tasks: int,
    tasks_root: Path,
    opt_root: Path,
    out_name: str = ORCA_OUT_NAME,
    opt_xyz_name: str = ORCA_OPT_XYZ,
) -> tuple[
    list[dict[str, Any]],
    list[float | None],
    list[Path],
    list[dict[str, Any]],
]:
    """Walk each ``task_XXXX`` and gather opt geometries + energies.

    Returns ``(conformer_records, energies_h, optimized_paths,
    missing_records)``:

        conformer_records  per-task dicts shaped for the manifest's
                           ``conformers`` bucket (always include
                           ``index`` and ``path_abs`` for tasks that
                           produced an ``orca_opt.xyz``; include
                           ``energy_hartree`` when parseable).
        energies_h         length-``n_tasks`` list parallel to the
                           per-task indices; ``None`` for tasks that
                           produced no parseable FINAL E.
        optimized_paths    file paths in the destination dir (same
                           order as ``conformer_records``); empty list
                           when nothing was collected.
        missing_records    structured failure dicts for tasks where
                           ``orca_opt.xyz`` was missing — caller
                           appends these to the manifest's failures.
    """
    opt_root.mkdir(parents=True, exist_ok=True)

    conformer_records: list[dict[str, Any]] = []
    energies_h: list[float | None] = []
    optimized_paths: list[Path] = []
    missing_records: list[dict[str, Any]] = []

    for i in range(1, int(n_tasks) + 1):
        task_dir = tasks_root / f"task_{i:04d}"
        out_xyz = task_dir / opt_xyz_name
        out_out = task_dir / out_name

        e_h = parse_orca_final_energy(out_out) if out_out.exists() else None
        energies_h.append(e_h)

        if out_xyz.exists():
            dst = opt_root / f"conf_{i:04d}.xyz"
            shutil.copy2(out_xyz, dst)
            optimized_paths.append(dst)

            rec: dict[str, Any] = {
                "index": i,
                "label": f"conf_{i:04d}",
                "path_abs": str(dst.resolve()),
                "sha256": sha256_file(dst),
                "format": "xyz",
            }
            if e_h is not None:
                rec["energy_hartree"] = float(e_h)
            conformer_records.append(rec)
        else:
            missing_records.append(
                {
                    "error": "missing_orca_opt_xyz",
                    "index": i,
                    "task_dir": str(task_dir.resolve()),
                }
            )

    return conformer_records, energies_h, optimized_paths, missing_records


# --------------------------------------------------------------------
# Node class
# --------------------------------------------------------------------


class OrcaDftArray(Node):
    """SLURM-array DFT optimizer for an upstream conformer ensemble."""

    step = "orca_dft_array"
    accepts_upstream = True
    requires_upstream = True

    def parse_config(self, raw: dict[str, Any]) -> dict[str, Any]:
        max_concurrency = normalize_max_concurrency(raw)

        unpaired = parse_int(raw.get("unpaired_electrons"), 0)
        mult_override = parse_optional_int(raw.get("multiplicity"))

        keywords = str(raw.get("keywords", DEFAULT_KEYWORDS)).strip()
        if not keywords:
            raise ValueError("keywords must be non-empty")

        maxcore = max(500, parse_int(raw.get("maxcore"), 4000))
        nprocs = max(1, parse_int(raw.get("nprocs"), 8))

        time_limit = str(raw.get("time_limit", "12:00:00")).strip() or "12:00:00"
        partition = normalize_optional_str(raw.get("partition"))

        orca_module = str(
            raw.get("orca_module", DEFAULT_ORCA_MODULE)
        ).strip() or DEFAULT_ORCA_MODULE

        monitor_interval_s = max(5, parse_int(raw.get("monitor_interval_s"), 60))
        monitor_timeout_min = max(0, parse_int(raw.get("monitor_timeout_min"), 0))

        return {
            "max_concurrency": max_concurrency,
            "charge": parse_int(raw.get("charge"), 0),
            "unpaired_electrons": unpaired,
            "multiplicity": mult_override,
            "solvent": normalize_optional_str(raw.get("solvent")),
            "smd_solvent": normalize_optional_str(raw.get("smd_solvent")),
            "keywords": keywords,
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
        }

    def run(self, ctx: NodeContext) -> None:
        cfg = ctx.config
        multiplicity = resolve_multiplicity(
            multiplicity=cfg["multiplicity"],
            unpaired_electrons=cfg["unpaired_electrons"],
        )

        # Echo resolved config into manifest.inputs early.
        ctx.set_inputs(
            max_concurrency=cfg["max_concurrency"],
            charge=cfg["charge"],
            unpaired_electrons=cfg["unpaired_electrons"],
            multiplicity=multiplicity,
            solvent=cfg["solvent"],
            smd_solvent=cfg["smd_solvent"],
            keywords=cfg["keywords"],
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

        # ----- 2) Build per-task dirs -----
        array_root = outputs_dir / "array"
        tasks_root = array_root / "tasks"
        slurm_logs = array_root / "slurm_logs"
        array_root.mkdir(parents=True, exist_ok=True)
        tasks_root.mkdir(parents=True, exist_ok=True)
        slurm_logs.mkdir(parents=True, exist_ok=True)

        inp_text = make_orca_simple_input(
            keywords=cfg["keywords"],
            nprocs=cfg["nprocs"],
            maxcore=cfg["maxcore"],
            charge=cfg["charge"],
            multiplicity=multiplicity,
            solvent=cfg["solvent"],
            smd_solvent_override=cfg["smd_solvent"],
            xyz_filename="input.xyz",
        )
        build_task_dirs(
            staged_paths=staged_paths,
            tasks_root=tasks_root,
            inp_text=inp_text,
            inp_name=ORCA_INP_NAME,
        )

        # ----- 3) Render SLURM array script -----
        job_name = cfg["job_name"] or f"orca_opt_array_{n_tasks}"
        per_task_body = standard_orca_per_task_body(
            inp_filename=ORCA_INP_NAME,
            out_filename=ORCA_OUT_NAME,
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

        # Initial array bookkeeping. ``set_array_info`` overwrites; we
        # update it in place once we have a jobid + final progress.
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
            f"orca-dft-array: submitted SLURM array job -> jobid {jobid} "
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
            # The DFT task body only writes the success/failed sentinels
            # AFTER ORCA exits; while ORCA is mid-run, the orca_opt.out
            # file is the only evidence the task actually started. Pass
            # it through as an additional "started" signal.
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

    def _aggregate(
        self,
        ctx: NodeContext,
        *,
        outputs_dir: Path,
        tasks_root: Path,
        n_tasks: int,
        jobid: str | None,
        execs: SlurmExecutables,
    ) -> None:
        opt_root = outputs_dir / "optimized_conformers"
        opt_root.mkdir(parents=True, exist_ok=True)

        conf_records, energies_h, optimized_paths, missing_records = (
            collect_optimized_outputs(
                n_tasks=n_tasks,
                tasks_root=tasks_root,
                opt_root=opt_root,
                out_name=ORCA_OUT_NAME,
                opt_xyz_name=ORCA_OPT_XYZ,
            )
        )

        for rec in conf_records:
            ctx.add_artifact("conformers", rec)
        for fail_rec in missing_records:
            ctx.fail(fail_rec.pop("error"), **fail_rec)

        # 3-column orca.energies file (index, abs_Eh, rel_kcal).
        energies_path = opt_root / "orca.energies"
        rel_kcal, e_min = write_energy_file(
            energies_h=energies_h, out_path=energies_path
        )
        ctx.add_artifact(
            "files",
            {
                "label": "orca_energies",
                "path_abs": str(energies_path.resolve()),
                "sha256": sha256_file(energies_path),
                "format": "txt",
            },
        )

        # Attach rel_energy_kcal in-place on the just-published records.
        # We can only do this after the rel_kcal list is known, and we
        # need to walk both lists in lockstep — but missing tasks were
        # skipped from conf_records, so rebuild the (index, rel) map by
        # 1-based task index.
        rel_by_index = {
            i: rel_kcal[i - 1]
            for i in range(1, n_tasks + 1)
            if rel_kcal[i - 1] is not None
        }
        for rec in ctx.manifest.artifacts.get("conformers", []):
            idx = rec.get("index")
            if isinstance(idx, int) and idx in rel_by_index:
                rec["rel_energy_kcal"] = float(rel_by_index[idx])

        # Multi-xyz ensemble of optimized geometries.
        if optimized_paths:
            ensemble_path = opt_root / "optimized_conformers.xyz"
            concat_xyz_files(optimized_paths, ensemble_path)
            ctx.add_artifact(
                "xyz_ensemble",
                {
                    "label": "optimized_ensemble",
                    "path_abs": str(ensemble_path.resolve()),
                    "sha256": sha256_file(ensemble_path),
                    "format": "xyz",
                },
            )

        # ``best`` = lowest absolute energy among the tasks that
        # produced a geometry. If no energies were parseable, no best.
        if e_min is not None and optimized_paths:
            finite_pairs = [
                (i + 1, e) for i, e in enumerate(energies_h) if e is not None
            ]
            if finite_pairs:
                best_idx, _ = min(finite_pairs, key=lambda t: t[1])
                best_src = opt_root / f"conf_{best_idx:04d}.xyz"
                if best_src.exists():
                    best_dst = opt_root / "best.xyz"
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

        if not optimized_paths:
            ctx.fail("no_optimized_geometries_collected")

        # sacct post-mortem — surface per-task failures as structured
        # records on top of whatever the sentinels said. Best-effort:
        # if sacct isn't available we just skip.
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
    "DEFAULT_KEYWORDS",
    "DEFAULT_ORCA_MODULE",
    "ORCA_INP_NAME",
    "ORCA_OPT_XYZ",
    "ORCA_OUT_NAME",
    "OrcaDftArray",
    "build_task_dirs",
    "collect_optimized_outputs",
    "main",
    "normalize_max_concurrency",
    "resolve_multiplicity",
    "stage_conformer_inputs",
]


main = OrcaDftArray.invoke_factory()


if __name__ == "__main__":
    raise SystemExit(main())
