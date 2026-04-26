"""Tests for ``scripps_workflow.slurm`` array-job primitives.

We don't have SLURM in CI / the test sandbox, so the tests inject fakes
for the three places that would otherwise shell out
(:func:`squeue_has_any` lookup, :func:`time.sleep`, log callback).

Coverage:

    * :func:`count_task_progress` — sentinel-file tally with and without
      extra "started" signals.
    * :func:`monitor_array_job` — drain-on-empty-queue, timeout, history
      recording, log callback.
    * :func:`make_array_slurm_text` / :func:`standard_orca_per_task_body`
      — SBATCH headers + per-task body composition.
    * :func:`sacct_failures_for_array` — per-task failure synthesis.
    * :class:`SlurmExecutables` / :func:`discover_slurm_executables` —
      PATH lookup with monkeypatched ``shutil.which``.
    * :class:`ProgressCounts` — empty + to_dict shape.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytest

from scripps_workflow.slurm import (
    MonitorResult,
    ProgressCounts,
    SBATCH_JOBID_RE,
    SlurmExecutables,
    count_task_progress,
    discover_slurm_executables,
    make_array_slurm_text,
    monitor_array_job,
    sacct_failures_for_array,
    sbatch_submit,
    standard_orca_per_task_body,
)


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------


def _make_status_dir(task_dir: Path) -> Path:
    """Create the ``.wf_status`` dir under a per-task root and return it."""
    status = task_dir / ".wf_status"
    status.mkdir(parents=True, exist_ok=True)
    return status


def _write_tasks(
    tasks_root: Path,
    states: list[str],
    *,
    extra_files: dict[int, list[str]] | None = None,
) -> None:
    """Build ``task_0001..task_NNNN`` with the given sentinel states.

    ``states[i]`` is one of:
        ``"empty"``     — task dir exists but no sentinels.
        ``"started"``   — ``.wf_status/started`` exists.
        ``"success"``   — ``.wf_status/done_success`` exists.
        ``"failed"``    — ``.wf_status/done_failed`` exists.
        ``"missing"``   — task dir does not exist at all.

    ``extra_files`` maps 1-based index → list of relative filenames to
    create inside the task dir (used for ``started_extra_signals``
    coverage).
    """
    extra_files = extra_files or {}
    for i, st in enumerate(states, start=1):
        task_dir = tasks_root / f"task_{i:04d}"
        if st == "missing":
            continue
        task_dir.mkdir(parents=True, exist_ok=True)
        if st in {"started", "success", "failed"}:
            status = _make_status_dir(task_dir)
            if st == "started":
                (status / "started").touch()
            elif st == "success":
                (status / "done_success").touch()
            elif st == "failed":
                (status / "done_failed").touch()
        for fname in extra_files.get(i, []):
            (task_dir / fname).write_text("")


# --------------------------------------------------------------------
# SBATCH_JOBID_RE
# --------------------------------------------------------------------


class TestSbatchJobidRegex:
    def test_basic_match(self):
        m = SBATCH_JOBID_RE.search("Submitted batch job 12345")
        assert m is not None
        assert m.group(1) == "12345"

    def test_with_trailing_whitespace(self):
        m = SBATCH_JOBID_RE.search("Submitted batch job 12345  \n")
        assert m is not None
        assert m.group(1) == "12345"

    def test_no_match_in_unrelated_text(self):
        assert SBATCH_JOBID_RE.search("squeue says nothing here") is None


# --------------------------------------------------------------------
# ProgressCounts
# --------------------------------------------------------------------


class TestProgressCounts:
    def test_empty(self):
        p = ProgressCounts.empty(7)
        assert p.total == 7
        assert p.started == 0
        assert p.success == 0
        assert p.failed == 0
        assert p.processed == 0
        assert p.in_progress == 0
        assert p.left == 7

    def test_to_dict_keys(self):
        p = ProgressCounts.empty(3)
        d = p.to_dict()
        assert set(d.keys()) == {
            "total",
            "started",
            "success",
            "failed",
            "processed",
            "in_progress",
            "left",
        }

    def test_to_dict_round_trip(self):
        p = ProgressCounts(
            total=5,
            started=4,
            success=2,
            failed=1,
            processed=3,
            in_progress=1,
            left=2,
        )
        d = p.to_dict()
        assert d == {
            "total": 5,
            "started": 4,
            "success": 2,
            "failed": 1,
            "processed": 3,
            "in_progress": 1,
            "left": 2,
        }


# --------------------------------------------------------------------
# count_task_progress
# --------------------------------------------------------------------


class TestCountTaskProgress:
    def test_all_empty(self, tmp_path):
        _write_tasks(tmp_path, ["empty"] * 4)
        p = count_task_progress(tmp_path, 4)
        assert p.total == 4
        assert p.started == 0
        assert p.processed == 0
        assert p.left == 4

    def test_mixed_states(self, tmp_path):
        _write_tasks(
            tmp_path,
            ["success", "failed", "started", "empty", "missing"],
        )
        p = count_task_progress(tmp_path, 5)
        assert p.total == 5
        assert p.success == 1
        assert p.failed == 1
        assert p.processed == 2
        assert p.started == 3  # success + failed + started counted as started
        assert p.in_progress == 1  # 3 started − 2 processed
        assert p.left == 3  # 5 total − 2 processed

    def test_started_extra_signals(self, tmp_path):
        # No status sentinels at all, but task 1 has orca_opt.out which
        # the extra_signals tuple flags as evidence of a started task.
        _write_tasks(
            tmp_path,
            ["empty", "empty", "empty"],
            extra_files={1: ["orca_opt.out"]},
        )
        p = count_task_progress(
            tmp_path, 3, started_extra_signals=("orca_opt.out",)
        )
        assert p.started == 1
        assert p.in_progress == 1
        assert p.processed == 0

    def test_extra_signal_does_not_double_count_with_started(self, tmp_path):
        # Both ``started`` sentinel AND ``orca_opt.out`` exist for the
        # same task — should still only contribute 1 to ``started``.
        _write_tasks(
            tmp_path,
            ["started"],
            extra_files={1: ["orca_opt.out"]},
        )
        p = count_task_progress(
            tmp_path, 1, started_extra_signals=("orca_opt.out",)
        )
        assert p.started == 1

    def test_done_success_implies_started(self, tmp_path):
        # If only the success sentinel was written (no separate
        # ``started``), the task still counts as started. The per-task
        # body always touches ``started`` first, but if the file is
        # somehow missing (race, manual cleanup) the success sentinel
        # is sufficient.
        _write_tasks(tmp_path, ["success", "success"])
        p = count_task_progress(tmp_path, 2)
        assert p.started == 2
        assert p.success == 2
        assert p.processed == 2
        assert p.in_progress == 0
        assert p.left == 0

    def test_n_tasks_smaller_than_dirs_truncates(self, tmp_path):
        # Build 5 task dirs but only ask about 3 — the function should
        # only walk the first 3.
        _write_tasks(tmp_path, ["success"] * 5)
        p = count_task_progress(tmp_path, 3)
        assert p.success == 3
        assert p.total == 3

    def test_zero_tasks(self, tmp_path):
        p = count_task_progress(tmp_path, 0)
        assert p.total == 0
        assert p.left == 0


# --------------------------------------------------------------------
# monitor_array_job
# --------------------------------------------------------------------


class TestMonitorArrayJob:
    def test_drains_when_squeue_returns_false(self, tmp_path):
        # 3 tasks, all already success on disk. Inject a squeue_check
        # that returns False on the very first call → loop exits in
        # iteration 1.
        _write_tasks(tmp_path, ["success", "success", "success"])

        squeue_calls = {"n": 0}

        def _sq(jobid: str) -> bool:
            squeue_calls["n"] += 1
            return False

        slept: list[float] = []

        def _sleep(s: float) -> None:
            slept.append(s)

        result = monitor_array_job(
            jobid="12345",
            tasks_root=tmp_path,
            n_tasks=3,
            monitor_interval_s=10,
            monitor_timeout_min=0,
            squeue_check=_sq,
            sleep_fn=_sleep,
        )
        assert result.timed_out is False
        assert result.iterations == 1
        # Drain branch: the function does NOT sleep before returning.
        assert slept == []
        assert squeue_calls["n"] == 1
        assert result.final_progress.success == 3
        assert result.final_progress.processed == 3
        assert result.final_progress.left == 0

    def test_squeue_true_then_false(self, tmp_path):
        # 2 tasks. squeue returns True for the first 2 calls, False
        # for the 3rd. The loop should sleep twice, then drain.
        _write_tasks(tmp_path, ["success", "success"])

        seq = [True, True, False]
        idx = {"n": 0}

        def _sq(jobid: str) -> bool:
            v = seq[idx["n"]]
            idx["n"] += 1
            return v

        slept: list[float] = []
        result = monitor_array_job(
            jobid="42",
            tasks_root=tmp_path,
            n_tasks=2,
            monitor_interval_s=5,
            monitor_timeout_min=0,
            squeue_check=_sq,
            sleep_fn=lambda s: slept.append(s),
        )
        assert result.timed_out is False
        assert result.iterations == 3
        # Slept once per "still queued" iteration.
        assert slept == [5.0, 5.0]

    def test_timeout_triggers(self, tmp_path, monkeypatch):
        # squeue always reports True → loop would never drain. Cap the
        # walltime via fake time.monotonic. We start the clock at t=0
        # and step 60s on each .now() call so the second iteration
        # exceeds the 1-minute cap.
        _write_tasks(tmp_path, ["empty", "empty"])

        ticks = iter([0.0, 30.0, 90.0, 150.0, 210.0])

        def _now():
            return next(ticks)

        # Patch time.monotonic in slurm module to drive timeout.
        from scripps_workflow import slurm as slurm_mod

        monkeypatch.setattr(slurm_mod.time, "monotonic", _now)

        result = monitor_array_job(
            jobid="99",
            tasks_root=tmp_path,
            n_tasks=2,
            monitor_interval_s=5,
            monitor_timeout_min=1,
            squeue_check=lambda j: True,
            sleep_fn=lambda s: None,
        )
        assert result.timed_out is True

    def test_timeout_zero_means_no_cap(self, tmp_path):
        # monitor_timeout_min=0 → never time out. squeue returns False
        # on the first poll so the loop exits cleanly.
        _write_tasks(tmp_path, ["empty"])
        result = monitor_array_job(
            jobid="1",
            tasks_root=tmp_path,
            n_tasks=1,
            monitor_interval_s=10,
            monitor_timeout_min=0,
            squeue_check=lambda j: False,
            sleep_fn=lambda s: None,
        )
        assert result.timed_out is False

    def test_log_callback_invoked_per_iteration(self, tmp_path):
        _write_tasks(tmp_path, ["empty", "empty"])

        seq = [True, False]
        idx = {"n": 0}

        def _sq(jobid: str) -> bool:
            v = seq[idx["n"]]
            idx["n"] += 1
            return v

        logs: list[str] = []
        monitor_array_job(
            jobid="1",
            tasks_root=tmp_path,
            n_tasks=2,
            monitor_interval_s=1,
            monitor_timeout_min=0,
            squeue_check=_sq,
            sleep_fn=lambda s: None,
            log_fn=logs.append,
        )
        # Two iterations → two log lines (then drain — drain branch
        # does not log again).
        assert len(logs) == 2
        for line in logs:
            assert "Progress:" in line
            assert "processed=" in line

    def test_record_history_off_by_default(self, tmp_path):
        _write_tasks(tmp_path, ["success"])
        result = monitor_array_job(
            jobid="1",
            tasks_root=tmp_path,
            n_tasks=1,
            monitor_interval_s=1,
            monitor_timeout_min=0,
            squeue_check=lambda j: False,
            sleep_fn=lambda s: None,
        )
        assert result.progress_history == []

    def test_record_history_on(self, tmp_path):
        _write_tasks(tmp_path, ["success", "success"])
        seq = [True, False]
        idx = {"n": 0}

        def _sq(jobid: str) -> bool:
            v = seq[idx["n"]]
            idx["n"] += 1
            return v

        result = monitor_array_job(
            jobid="1",
            tasks_root=tmp_path,
            n_tasks=2,
            monitor_interval_s=1,
            monitor_timeout_min=0,
            squeue_check=_sq,
            sleep_fn=lambda s: None,
            record_history=True,
        )
        # Each iteration appends; drain appends one more snapshot.
        assert len(result.progress_history) >= 2
        for snap in result.progress_history:
            assert snap["total"] == 2

    def test_progress_fn_injected(self, tmp_path):
        # A custom progress_fn replaces sentinel-walking entirely.
        calls = {"n": 0}

        def _fake_progress(root: Path, n: int) -> ProgressCounts:
            calls["n"] += 1
            return ProgressCounts.empty(n)

        monitor_array_job(
            jobid="1",
            tasks_root=tmp_path,
            n_tasks=4,
            monitor_interval_s=1,
            monitor_timeout_min=0,
            squeue_check=lambda j: False,
            progress_fn=_fake_progress,
            sleep_fn=lambda s: None,
        )
        # The injected fn was called at least once.
        assert calls["n"] >= 1


# --------------------------------------------------------------------
# make_array_slurm_text
# --------------------------------------------------------------------


class TestMakeArraySlurmText:
    def _base(self, **overrides):
        defaults = dict(
            job_name="test_job",
            n_tasks=4,
            max_concurrency=2,
            nprocs=8,
            time_limit="06:00:00",
            partition=None,
            tasks_root_abs="/tasks/root",
            slurm_logs_abs="/slurm/logs",
            orca_module="orca/6.0.0",
            silence_openib=True,
            per_task_body='echo "hello"\n',
        )
        defaults.update(overrides)
        return make_array_slurm_text(**defaults)

    def test_starts_with_shebang(self):
        text = self._base()
        assert text.startswith("#!/bin/bash\n")

    def test_sbatch_headers_present(self):
        text = self._base()
        assert "#SBATCH -J test_job" in text
        assert "#SBATCH --ntasks=8" in text
        assert "#SBATCH -t 06:00:00" in text
        assert "#SBATCH --array=1-4%2" in text
        assert "#SBATCH -o /slurm/logs/%x.%A_%a.out" in text
        assert "#SBATCH -e /slurm/logs/%x.%A_%a.err" in text

    def test_partition_omitted_when_none(self):
        text = self._base(partition=None)
        assert "#SBATCH -p " not in text

    def test_partition_included_when_set(self):
        text = self._base(partition="gpu")
        assert "#SBATCH -p gpu" in text

    def test_max_concurrency_clamped_to_n_tasks(self):
        # Asking for %M=100 with N=4 → clamp to 4.
        text = self._base(n_tasks=4, max_concurrency=100)
        assert "#SBATCH --array=1-4%4" in text

    def test_max_concurrency_clamped_to_one_floor(self):
        text = self._base(n_tasks=4, max_concurrency=0)
        assert "#SBATCH --array=1-4%1" in text

    def test_max_concurrency_clamped_negative(self):
        text = self._base(n_tasks=4, max_concurrency=-5)
        assert "#SBATCH --array=1-4%1" in text

    def test_module_load_line(self):
        text = self._base(orca_module="orca/6.0.0")
        assert "module load orca/6.0.0" in text

    def test_silence_openib_emits_export(self):
        text = self._base(silence_openib=True)
        assert 'OMPI_MCA_btl="^openib"' in text

    def test_no_silence_openib(self):
        text = self._base(silence_openib=False)
        assert "OMPI_MCA_btl" not in text

    def test_per_task_body_injected_verbatim(self):
        body = '# my custom body\necho "task ${TASK_ID}"\n'
        text = self._base(per_task_body=body)
        # Body appears verbatim (sans trailing newline normalization).
        assert "# my custom body" in text
        assert 'echo "task ${TASK_ID}"' in text

    def test_status_dir_setup(self):
        text = self._base()
        assert 'STATUS_DIR="${TASK_DIR}/.wf_status"' in text
        assert 'touch "${STATUS_DIR}/started"' in text
        assert "mark_success() {" in text
        assert "mark_failure() {" in text

    def test_task_dir_uses_printf_format(self):
        text = self._base(tasks_root_abs="/abs/tasks")
        # %04d formatting preserves the 4-digit task index.
        assert (
            'TASK_DIR="/abs/tasks/task_$(printf "%04d" "${TASK_ID}")"' in text
        )

    def test_set_euo_pipefail(self):
        text = self._base()
        assert "set -euo pipefail" in text

    def test_orca_not_on_path_falls_back_to_127(self):
        text = self._base()
        # The script exits 127 (POSIX "command not found") if ORCA
        # isn't reachable after module setup.
        assert "exit 127" in text


# --------------------------------------------------------------------
# standard_orca_per_task_body
# --------------------------------------------------------------------


class TestStandardOrcaPerTaskBody:
    def test_invokes_orca_with_inp_and_redirects_to_out(self):
        body = standard_orca_per_task_body(
            inp_filename="orca_opt.inp",
            out_filename="orca_opt.out",
        )
        assert '"${ORCA_BIN}" "orca_opt.inp" > "orca_opt.out"' in body

    def test_calls_mark_success_and_mark_failure(self):
        body = standard_orca_per_task_body(
            inp_filename="x.inp",
            out_filename="x.out",
        )
        assert "mark_success" in body
        assert "mark_failure" in body
        # Failure path captures rc and propagates exit code.
        assert "rc=$?" in body
        assert 'exit "${rc}"' in body

    def test_thermo_filenames_pass_through(self):
        # Thermo array node will use different filenames. Make sure
        # they're injected correctly.
        body = standard_orca_per_task_body(
            inp_filename="orca_thermo.inp",
            out_filename="orca_thermo.out",
        )
        assert '"orca_thermo.inp"' in body
        assert '"orca_thermo.out"' in body


# --------------------------------------------------------------------
# sacct_failures_for_array
# --------------------------------------------------------------------


class TestSacctFailuresForArray:
    def test_all_completed_no_failures(self):
        states = {
            "1234_1": ("COMPLETED", "0:0"),
            "1234_2": ("COMPLETED", "0:0"),
        }
        out = sacct_failures_for_array(states, jobid="1234", n_tasks=2)
        assert out == []

    def test_failed_state(self):
        states = {
            "1234_1": ("COMPLETED", "0:0"),
            "1234_2": ("FAILED", "1:0"),
            "1234_3": ("CANCELLED", "0:0"),
        }
        out = sacct_failures_for_array(states, jobid="1234", n_tasks=3)
        assert len(out) == 2
        # Each failure record carries task index, jobid, state, exitcode.
        idx_to_state = {rec["task"]: rec["state"] for rec in out}
        assert idx_to_state == {2: "FAILED", 3: "CANCELLED"}
        for rec in out:
            assert rec["error"] == "array_task_not_completed"
            assert rec["jobid"] == "1234"

    def test_nonzero_exit_code_counts_as_failure(self):
        # State is COMPLETED but the exit code says non-zero — flag it.
        states = {
            "1_1": ("COMPLETED", "0:0"),
            "1_2": ("COMPLETED", "127:0"),
        }
        out = sacct_failures_for_array(states, jobid="1", n_tasks=2)
        assert len(out) == 1
        assert out[0]["task"] == 2
        assert out[0]["exitcode"] == "127:0"

    def test_missing_state_silently_skipped(self):
        # If sacct returned no info for task 2, we don't fabricate a
        # failure — no information ≠ failure.
        states = {"5_1": ("COMPLETED", "0:0")}
        out = sacct_failures_for_array(states, jobid="5", n_tasks=2)
        assert out == []

    def test_completed_with_zero_zero_format(self):
        # Some clusters report "0:0" for the success exit code (signal:rc).
        states = {"1_1": ("COMPLETED", "0:0")}
        out = sacct_failures_for_array(states, jobid="1", n_tasks=1)
        assert out == []

    def test_empty_states(self):
        assert sacct_failures_for_array({}, jobid="1", n_tasks=4) == []


# --------------------------------------------------------------------
# SlurmExecutables / discover_slurm_executables
# --------------------------------------------------------------------


class TestSlurmExecutables:
    def test_has_all_true_when_set(self):
        e = SlurmExecutables(
            sbatch="/bin/sbatch",
            squeue="/bin/squeue",
            sacct="/bin/sacct",
        )
        assert e.has_all is True

    def test_has_all_false_when_any_missing(self):
        assert SlurmExecutables(sbatch="/bin/sbatch").has_all is False
        assert SlurmExecutables(squeue="/bin/squeue").has_all is False
        assert SlurmExecutables(sacct="/bin/sacct").has_all is False
        assert SlurmExecutables().has_all is False

    def test_discover_uses_shutil_which(self, monkeypatch):
        from scripps_workflow import slurm as slurm_mod

        fake = {
            "sbatch": "/usr/bin/sbatch",
            "squeue": "/usr/bin/squeue",
            "sacct": "/usr/bin/sacct",
        }

        def _which(name: str) -> Optional[str]:
            return fake.get(name)

        monkeypatch.setattr(slurm_mod.shutil, "which", _which)
        e = discover_slurm_executables()
        assert e.sbatch == "/usr/bin/sbatch"
        assert e.squeue == "/usr/bin/squeue"
        assert e.sacct == "/usr/bin/sacct"
        assert e.has_all is True

    def test_discover_returns_none_when_missing(self, monkeypatch):
        from scripps_workflow import slurm as slurm_mod

        monkeypatch.setattr(slurm_mod.shutil, "which", lambda name: None)
        e = discover_slurm_executables()
        assert e.sbatch is None
        assert e.squeue is None
        assert e.sacct is None
        assert e.has_all is False


# --------------------------------------------------------------------
# sbatch_submit (subprocess-fenced)
# --------------------------------------------------------------------


class _FakeProc:
    def __init__(self, *, returncode: int, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class TestSbatchSubmit:
    def test_success_parses_jobid(self, monkeypatch, tmp_path):
        from scripps_workflow import slurm as slurm_mod

        def _fake_run(cmd, **kwargs):
            return _FakeProc(returncode=0, stdout="Submitted batch job 99887\n")

        monkeypatch.setattr(slurm_mod.subprocess, "run", _fake_run)
        ok, jobid, msg = sbatch_submit(
            "/bin/sbatch", tmp_path / "submit.slurm", cwd=tmp_path
        )
        assert ok is True
        assert jobid == "99887"
        assert "Submitted batch job 99887" in msg

    def test_nonzero_returncode_is_failure(self, monkeypatch, tmp_path):
        from scripps_workflow import slurm as slurm_mod

        def _fake_run(cmd, **kwargs):
            return _FakeProc(
                returncode=1, stdout="", stderr="error: bad partition\n"
            )

        monkeypatch.setattr(slurm_mod.subprocess, "run", _fake_run)
        ok, jobid, msg = sbatch_submit(
            "/bin/sbatch", tmp_path / "submit.slurm", cwd=tmp_path
        )
        assert ok is False
        assert jobid is None
        assert "bad partition" in msg

    def test_unparseable_output_is_failure(self, monkeypatch, tmp_path):
        from scripps_workflow import slurm as slurm_mod

        def _fake_run(cmd, **kwargs):
            return _FakeProc(returncode=0, stdout="<weird>\n")

        monkeypatch.setattr(slurm_mod.subprocess, "run", _fake_run)
        ok, jobid, msg = sbatch_submit(
            "/bin/sbatch", tmp_path / "submit.slurm", cwd=tmp_path
        )
        assert ok is False
        assert jobid is None
        assert "could_not_parse_jobid" in msg


# --------------------------------------------------------------------
# Module surface
# --------------------------------------------------------------------


class TestPublicSurface:
    def test_all_exports(self):
        from scripps_workflow import slurm as s

        for name in (
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
        ):
            assert hasattr(s, name), f"missing public export: {name}"

    def test_monitor_result_default_history_empty(self):
        r = MonitorResult(final_progress=ProgressCounts.empty(0))
        assert r.timed_out is False
        assert r.iterations == 0
        assert r.progress_history == []
