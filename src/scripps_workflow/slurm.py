"""SLURM array-job primitives shared across array nodes.

Two nodes (``orca_dft_array`` and ``orca_thermo_array``) currently fan
the same conformer ensemble out across an ``--array=1-N%M`` job, run
ORCA in each task directory, and aggregate results. The SLURM glue is
identical between them; this module owns it.

Design notes:

    * The high-level :func:`monitor_array_job` loop is a pure function
      modulo three injected dependencies (``squeue_has_any``,
      ``count_progress``, ``sleep_fn``). Tests can replace ``sleep_fn``
      with an immediate-return stub and ``squeue_has_any`` with a
      step-counting fake, so the loop is exercised without ever
      sleeping or shelling out.

    * :func:`make_array_slurm_text` takes the per-task body as a
      free-form string. The DFT-array vs thermo-array difference (which
      .inp file ORCA runs, which .out it captures) is therefore a
      one-line caller-side concern; the module structure (SBATCH
      headers, module load, sentinel files) is shared.

    * Sentinel files under each ``task_XXXX/.wf_status/`` (``started``,
      ``done_success``, ``done_failed``, ``exit_code.txt``) are the
      authoritative signal we use to count progress. ``squeue`` is only
      used to detect that the array job has cleared the queue; ``sacct``
      is consulted post-hoc to surface per-task states in the manifest.

This module is stdlib-only.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional


SBATCH_JOBID_RE: re.Pattern[str] = re.compile(
    r"Submitted batch job\s+(\d+)\s*$"
)


# --------------------------------------------------------------------
# Executable discovery
# --------------------------------------------------------------------


@dataclass(frozen=True)
class SlurmExecutables:
    """Resolved absolute paths for SLURM client tools.

    Each field is ``None`` if the executable is not on PATH. ``has_all``
    is ``True`` iff every binary needed for submit + monitor was found.
    The submit / monitor branches degrade gracefully when one is
    missing (recording a structured failure).
    """

    sbatch: str | None = None
    squeue: str | None = None
    sacct: str | None = None

    @property
    def has_all(self) -> bool:
        return all((self.sbatch, self.squeue, self.sacct))


def discover_slurm_executables() -> SlurmExecutables:
    """Locate ``sbatch`` / ``squeue`` / ``sacct`` on the current PATH."""
    return SlurmExecutables(
        sbatch=shutil.which("sbatch"),
        squeue=shutil.which("squeue"),
        sacct=shutil.which("sacct"),
    )


# --------------------------------------------------------------------
# Submit / queue / accounting
# --------------------------------------------------------------------


def sbatch_submit(
    sbatch_exe: str,
    slurm_path: Path,
    *,
    cwd: Path,
) -> tuple[bool, Optional[str], str]:
    """Submit an array job. Returns ``(ok, jobid, combined_stdout_stderr)``.

    On success the combined message is preserved for the manifest so
    operators can read SLURM's full reply. On failure ``jobid`` is
    ``None`` and the message contains whatever ``sbatch`` wrote to
    stderr.
    """
    proc = subprocess.run(
        [sbatch_exe, str(slurm_path)],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    msg = (out + ("\n" + err if err else "")).strip()

    if proc.returncode != 0:
        return False, None, msg

    m = SBATCH_JOBID_RE.search(out)
    if not m:
        return False, None, f"could_not_parse_jobid: {out!r}"
    return True, m.group(1), msg


def squeue_has_any(squeue_exe: str, jobid: str) -> bool:
    """Return ``True`` iff ``squeue`` still reports any task of ``jobid``.

    Uses ``-h -o %i`` so the output is one line per still-queued task,
    no header. Empty stdout = nothing left in the queue.
    """
    proc = subprocess.run(
        [squeue_exe, "-j", str(jobid), "-h", "-o", "%i"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        # squeue can transiently fail (controller hiccups). Treat as
        # "still in queue" so the monitor loop doesn't bail early on a
        # blip.
        return True
    return (proc.stdout or "").strip() != ""


def sacct_states(
    sacct_exe: str,
    jobid: str,
) -> dict[str, tuple[Optional[str], Optional[str]]]:
    """Parse ``sacct -P -o JobIDRaw,State,ExitCode``.

    Returns ``{job_id_raw: (state, exit_code)}``. On any sacct error or
    parse failure the dict is empty — a missing entry is treated by the
    caller as "no information", not "failed".
    """
    proc = subprocess.run(
        [sacct_exe, "-j", str(jobid), "-n", "-P", "-o", "JobIDRaw,State,ExitCode"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        return {}
    out: dict[str, tuple[Optional[str], Optional[str]]] = {}
    for raw in (proc.stdout or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) >= 3:
            out[parts[0]] = (parts[1] or None, parts[2] or None)
    return out


# --------------------------------------------------------------------
# Per-task progress (sentinel files)
# --------------------------------------------------------------------


@dataclass(frozen=True)
class ProgressCounts:
    """Per-array snapshot of how many tasks have reached each state.

    The fields are derived from sentinel files written by the per-task
    body in :func:`make_array_slurm_text` — :func:`count_task_progress`
    converts those sentinels into this snapshot in one pass.
    """

    total: int
    started: int
    success: int
    failed: int
    processed: int
    in_progress: int
    left: int

    def to_dict(self) -> dict[str, int]:
        return {
            "total": self.total,
            "started": self.started,
            "success": self.success,
            "failed": self.failed,
            "processed": self.processed,
            "in_progress": self.in_progress,
            "left": self.left,
        }

    @classmethod
    def empty(cls, n_tasks: int) -> "ProgressCounts":
        return cls(
            total=int(n_tasks),
            started=0,
            success=0,
            failed=0,
            processed=0,
            in_progress=0,
            left=int(n_tasks),
        )


def count_task_progress(
    tasks_root: Path,
    n_tasks: int,
    *,
    started_extra_signals: tuple[str, ...] = (),
) -> ProgressCounts:
    """Walk ``task_0001..task_NNNN`` and tally sentinel-file presence.

    Sentinels (under ``task_XXXX/.wf_status/``):

        ``started``        the per-task script ran ``touch started``
        ``done_success``   ORCA exited 0
        ``done_failed``    ORCA exited non-zero (or was killed)
        ``exit_code.txt``  decimal exit code (informational; not used
                           for tallying)

    A task that has neither ``done_*`` sentinel but DOES have either the
    ``started`` flag OR an entry in ``started_extra_signals`` (relative
    filenames inside ``task_XXXX/`` that count as evidence the task was
    launched, e.g. ``orca_opt.out`` for the dft node) is counted as
    ``in_progress``.

    The returned :class:`ProgressCounts` is JSON-serializable via
    ``to_dict()`` for the manifest.
    """
    started = 0
    success = 0
    failed = 0

    for i in range(1, int(n_tasks) + 1):
        task_dir = tasks_root / f"task_{i:04d}"
        status_dir = task_dir / ".wf_status"

        if (status_dir / "done_success").exists():
            success += 1
            started += 1
            continue
        if (status_dir / "done_failed").exists():
            failed += 1
            started += 1
            continue

        if (status_dir / "started").exists():
            started += 1
            continue
        for sig in started_extra_signals:
            if (task_dir / sig).exists():
                started += 1
                break

    processed = success + failed
    in_progress = max(0, started - processed)
    left = max(0, int(n_tasks) - processed)
    return ProgressCounts(
        total=int(n_tasks),
        started=started,
        success=success,
        failed=failed,
        processed=processed,
        in_progress=in_progress,
        left=left,
    )


# --------------------------------------------------------------------
# Monitor loop
# --------------------------------------------------------------------


@dataclass
class MonitorResult:
    """Outcome of the monitor loop.

    ``timed_out`` is True iff the loop exited because
    ``monitor_timeout_min`` elapsed (caller should record a structured
    failure). ``progress_history`` retains the per-iteration snapshot
    so the manifest can include a trace of how the array drained — set
    ``record_history=True`` in :func:`monitor_array_job` to enable it.
    """

    final_progress: ProgressCounts
    timed_out: bool = False
    iterations: int = 0
    progress_history: list[dict[str, int]] = field(default_factory=list)


def monitor_array_job(
    *,
    jobid: str,
    tasks_root: Path,
    n_tasks: int,
    monitor_interval_s: int,
    monitor_timeout_min: int,
    squeue_check: Callable[[str], bool],
    progress_fn: Callable[[Path, int], ProgressCounts] = count_task_progress,
    sleep_fn: Callable[[float], None] = time.sleep,
    log_fn: Callable[[str], None] | None = None,
    record_history: bool = False,
) -> MonitorResult:
    """Poll ``squeue`` + sentinel files until the array drains or times out.

    The function is deterministic given its dependencies: tests inject
    a deterministic ``squeue_check`` (e.g. "True for the first 3 calls,
    then False forever") and a ``sleep_fn`` no-op.

    Args:
        jobid: SLURM job id from sbatch.
        tasks_root: Directory containing ``task_0001..``.
        n_tasks: Total array width.
        monitor_interval_s: Seconds between polls (also the sleep
            interval). ``<= 0`` is treated as 1.
        monitor_timeout_min: Wall-clock cap in minutes; ``<= 0`` = no
            cap.
        squeue_check: Function ``(jobid) -> bool`` — True when squeue
            still has tasks queued/running.
        progress_fn: Sentinel-file tallier. Defaults to
            :func:`count_task_progress`.
        sleep_fn: ``time.sleep``-compatible callable.
        log_fn: Optional callback for human-readable progress lines.
        record_history: If True, ``MonitorResult.progress_history``
            gets a snapshot per iteration. Off by default to keep
            manifests slim.
    """
    interval = max(1, int(monitor_interval_s))
    timeout_s = (
        max(0, int(monitor_timeout_min)) * 60
        if monitor_timeout_min and int(monitor_timeout_min) > 0
        else 0
    )

    history: list[dict[str, int]] = []
    iterations = 0
    start_t = time.monotonic()

    progress = progress_fn(tasks_root, n_tasks)

    while True:
        iterations += 1
        progress = progress_fn(tasks_root, n_tasks)
        if record_history:
            history.append(progress.to_dict())
        if log_fn is not None:
            log_fn(
                "Progress: "
                f"processed={progress.processed}/{progress.total} | "
                f"success={progress.success} | "
                f"failed={progress.failed} | "
                f"in_progress={progress.in_progress} | "
                f"left={progress.left}"
            )

        if timeout_s > 0 and (time.monotonic() - start_t) > timeout_s:
            return MonitorResult(
                final_progress=progress,
                timed_out=True,
                iterations=iterations,
                progress_history=history,
            )

        if not squeue_check(jobid):
            # Drained. Take one more snapshot in case sentinels landed
            # after the queue cleared.
            progress = progress_fn(tasks_root, n_tasks)
            if record_history:
                history.append(progress.to_dict())
            return MonitorResult(
                final_progress=progress,
                timed_out=False,
                iterations=iterations,
                progress_history=history,
            )

        sleep_fn(float(interval))


# --------------------------------------------------------------------
# Array script generator
# --------------------------------------------------------------------


def make_array_slurm_text(
    *,
    job_name: str,
    n_tasks: int,
    max_concurrency: int,
    nprocs: int,
    time_limit: str,
    partition: str | None,
    tasks_root_abs: str,
    slurm_logs_abs: str,
    orca_module: str,
    silence_openib: bool,
    per_task_body: str,
) -> str:
    """Render an array-job SBATCH script.

    Per-task body is injected verbatim, so the script that runs in each
    task directory is whatever the caller specifies (e.g. ``orca
    orca_opt.inp > orca_opt.out`` for dft, ``orca orca_thermo.inp >
    orca_thermo.out`` for thermo). The body runs after ``cd "${TASK_DIR}"``
    and after ``mark_success`` / ``mark_failure`` shell functions are
    in scope.

    Args:
        job_name: SBATCH ``-J`` value.
        n_tasks: Array width (``--array=1-N``).
        max_concurrency: ``%M`` in ``--array=1-N%M``; clamped to
            ``[1, n_tasks]``.
        nprocs: ``--ntasks`` (and the ORCA ``%pal nprocs`` value
            chosen by the caller; we just allocate cores here).
        time_limit: SBATCH ``-t`` (e.g. ``"12:00:00"``).
        partition: SBATCH ``-p`` (omitted if None).
        tasks_root_abs: Absolute path to the dir containing
            ``task_0001/``…
        slurm_logs_abs: Absolute path for ``%x.%A_%a.{out,err}``.
        orca_module: Module name to load if ``module`` is available
            (e.g. ``"orca/6.0.0"``).
        silence_openib: If True, set ``OMPI_MCA_btl="^openib"``.
        per_task_body: Shell snippet executed inside each
            ``task_XXXX`` directory. The standard pattern wraps the
            ORCA invocation in ``if`` so success/failure sentinels are
            written:

                if "${ORCA_BIN}" "orca_opt.inp" > "orca_opt.out"; then
                  mark_success
                else
                  ...
                fi
    """
    width = max(1, min(int(max_concurrency), int(n_tasks)))
    array_spec = f"1-{int(n_tasks)}%{int(width)}"

    lines: list[str] = []
    lines.append("#!/bin/bash")
    lines.append(f"#SBATCH -J {job_name}")
    lines.append("#SBATCH -N 1")
    lines.append(f"#SBATCH --ntasks={int(nprocs)}")
    lines.append("#SBATCH --cpus-per-task=1")
    lines.append(f"#SBATCH -t {time_limit}")
    lines.append(f"#SBATCH --array={array_spec}")
    lines.append(f"#SBATCH -o {slurm_logs_abs}/%x.%A_%a.out")
    lines.append(f"#SBATCH -e {slurm_logs_abs}/%x.%A_%a.err")
    if partition:
        lines.append(f"#SBATCH -p {partition}")
    lines.append("")
    lines.append("set -euo pipefail")
    lines.append("")
    lines.append("# Make 'module' available in non-interactive shells without tripping set -u")
    lines.append("if ! type module &>/dev/null; then")
    lines.append("  set +u")
    lines.append("  [[ -f /etc/profile.d/modules.sh ]] && source /etc/profile.d/modules.sh || true")
    lines.append("  [[ -f /usr/share/Modules/init/bash ]] && source /usr/share/Modules/init/bash || true")
    lines.append("  set -u")
    lines.append("fi")
    lines.append("")
    lines.append("if type module &>/dev/null; then")
    lines.append("  module purge || true")
    lines.append(f"  module load {orca_module}")
    lines.append("else")
    lines.append('  echo "[wf-array] module command not found; assuming orca is already on PATH" >&2')
    lines.append("fi")
    lines.append("")
    lines.append('if ! command -v orca >/dev/null 2>&1; then')
    lines.append('  echo "[wf-array] ERROR: orca not found on PATH after module setup" >&2')
    lines.append("  exit 127")
    lines.append("fi")
    lines.append('ORCA_BIN="$(readlink -f "$(command -v orca)")"')
    lines.append("export OMP_NUM_THREADS=1")
    if silence_openib:
        lines.append('export OMPI_MCA_btl="^openib"')
    lines.append("")
    lines.append('TASK_ID="${SLURM_ARRAY_TASK_ID}"')
    lines.append('TASK_DIR="' + tasks_root_abs + '/task_$(printf "%04d" "${TASK_ID}")"')
    lines.append('cd "${TASK_DIR}"')
    lines.append('echo "[wf-array] task=${TASK_ID} cwd=$(pwd)"')
    lines.append("")
    lines.append('STATUS_DIR="${TASK_DIR}/.wf_status"')
    lines.append('mkdir -p "${STATUS_DIR}"')
    lines.append('touch "${STATUS_DIR}/started"')
    lines.append('rm -f "${STATUS_DIR}/done_success" "${STATUS_DIR}/done_failed"')
    lines.append("")
    lines.append("mark_success() {")
    lines.append('  printf "0\\n" > "${STATUS_DIR}/exit_code.txt"')
    lines.append('  touch "${STATUS_DIR}/done_success"')
    lines.append("}")
    lines.append("")
    lines.append("mark_failure() {")
    lines.append('  local rc="${1:-1}"')
    lines.append('  printf "%s\\n" "${rc}" > "${STATUS_DIR}/exit_code.txt"')
    lines.append('  touch "${STATUS_DIR}/done_failed"')
    lines.append("}")
    lines.append("")
    body = per_task_body.rstrip("\n")
    lines.append(body)
    lines.append("")
    return "\n".join(lines)


def standard_orca_per_task_body(
    *,
    inp_filename: str,
    out_filename: str,
) -> str:
    """Return the canonical "run ORCA + record sentinels" task body.

    Both DFT-array and thermo-array share the same shape; only the
    filenames differ::

        if "${ORCA_BIN}" "orca_opt.inp" > "orca_opt.out"; then
          mark_success
        else
          ...
        fi
    """
    return (
        f'if "${{ORCA_BIN}}" "{inp_filename}" > "{out_filename}"; then\n'
        f"  mark_success\n"
        f"else\n"
        f"  rc=$?\n"
        f'  echo "[wf-array] task=${{TASK_ID}} ORCA failed rc=${{rc}}" >&2\n'
        f'  mark_failure "${{rc}}"\n'
        f'  exit "${{rc}}"\n'
        f"fi\n"
    )


def multi_orca_per_task_body(
    *,
    jobs: list[tuple[str, str]],
) -> str:
    """Run multiple ORCA invocations sequentially in one SLURM task.

    Each ``(inp_filename, out_filename)`` pair becomes a fresh ORCA
    process — completely isolated from its neighbors — so method-state
    flags (DFT-NL/VV10, D3/D4, gCP, …) cannot leak between them. Used
    by the thermo-array node to keep the freq+SP compound separate
    from the WP04 ¹H, wB97X-D3 ¹³C, and mPW1PW91 J-coupling NMR jobs.

    Fail-fast: any non-zero ORCA exit aborts the chain, records the
    return code, and marks the task failed. The first job is the
    "primary" — its sentinel filename is what the manifest's
    ``ORCA_OUT_NAME`` points at — so when only one job is supplied
    this helper produces a body equivalent to
    :func:`standard_orca_per_task_body`.

    :param jobs: Ordered list of ``(inp, out)`` filename pairs
        (relative to the task dir). Must contain at least one entry.
    """
    if not jobs:
        raise ValueError("multi_orca_per_task_body: jobs must be non-empty")

    parts: list[str] = []
    for i, (inp, out) in enumerate(jobs, start=1):
        parts.append(
            f'echo "[wf-array] task=${{TASK_ID}} job {i}/{len(jobs)}: '
            f'{inp} -> {out}"\n'
            f'if ! "${{ORCA_BIN}}" "{inp}" > "{out}"; then\n'
            f"  rc=$?\n"
            f'  echo "[wf-array] task=${{TASK_ID}} ORCA job {i}/{len(jobs)} '
            f'({inp}) failed rc=${{rc}}" >&2\n'
            f'  mark_failure "${{rc}}"\n'
            f'  exit "${{rc}}"\n'
            f"fi\n"
        )
    parts.append("mark_success\n")
    return "\n".join(parts)


# --------------------------------------------------------------------
# sacct → per-task failure aggregation
# --------------------------------------------------------------------


def sacct_failures_for_array(
    states: dict[str, tuple[Optional[str], Optional[str]]],
    *,
    jobid: str,
    n_tasks: int,
) -> list[dict[str, Any]]:
    """Synthesize a list of per-task failure records from a sacct map.

    A task is "failed" iff its state does NOT start with ``COMPLETED``
    or its exit code does NOT start with ``0:0``. Tasks not present in
    ``states`` are silently skipped (no information ≠ failure). The
    returned dicts are shaped for ``Manifest.add_failure`` (``error``
    is always set; ``task``, ``jobid``, ``state``, ``exitcode`` are
    surfaced as extras).
    """
    out: list[dict[str, Any]] = []
    for i in range(1, int(n_tasks) + 1):
        key = f"{jobid}_{i}"
        entry = states.get(key)
        if entry is None:
            continue
        state, exitcode = entry
        state_ok = bool(state) and str(state).upper().startswith("COMPLETED")
        exit_ok = bool(exitcode) and str(exitcode).startswith("0:0")
        if state_ok and exit_ok:
            continue
        out.append(
            {
                "error": "array_task_not_completed",
                "task": i,
                "jobid": str(jobid),
                "state": state,
                "exitcode": exitcode,
            }
        )
    return out


__all__ = [
    "MonitorResult",
    "ProgressCounts",
    "SBATCH_JOBID_RE",
    "SlurmExecutables",
    "count_task_progress",
    "discover_slurm_executables",
    "make_array_slurm_text",
    "monitor_array_job",
    "sacct_failures_for_array",
    "sacct_states",
    "sbatch_submit",
    "squeue_has_any",
    "standard_orca_per_task_body",
    "multi_orca_per_task_body",
]
