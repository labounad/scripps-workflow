"""Tests for the orca_dft_array SLURM-array geometry-optimizer node.

ORCA + SLURM are not available in CI / the test sandbox, so we
monkeypatch:

    * :func:`scripps_workflow.nodes.orca_dft_array.sbatch_submit` — fake
      a successful submit AND populate the per-task output files
      (``orca_opt.xyz``, ``orca_opt.out``, sentinel files) so the
      aggregator finds something to aggregate.
    * :func:`scripps_workflow.nodes.orca_dft_array.squeue_has_any` —
      always returns False (queue drained on first poll).
    * :func:`scripps_workflow.nodes.orca_dft_array.sacct_states` —
      returns a synthetic per-task state map.
    * :func:`shutil.which` (in :mod:`scripps_workflow.slurm`) — to make
      sbatch / squeue / sacct discoverable.

Coverage:

    * Pure helpers: normalize_max_concurrency, resolve_multiplicity,
      stage_conformer_inputs (many / ensemble / single / none),
      build_task_dirs, collect_optimized_outputs.
    * Happy path: conformers fanned out, sbatch invoked, queue drained,
      optimized geometries collected, ensemble + best published,
      orca.energies written, rel_energy_kcal attached.
    * Failures: no upstream manifest, no xyz inputs, sbatch missing,
      sbatch returns non-zero, missing per-task outputs, monitor
      timeout.
    * Manifest: array bucket populated, environment records executable
      paths, sacct surfaced as failure records when state != COMPLETED.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import pytest

from scripps_workflow.nodes import orca_dft_array as oda
from scripps_workflow.nodes.orca_dft_array import (
    OrcaDftArray,
    build_task_dirs,
    collect_optimized_outputs,
    normalize_max_concurrency,
    resolve_multiplicity,
    stage_conformer_inputs,
)
from scripps_workflow.pointer import Pointer
from scripps_workflow.schema import Manifest


# --------------------------------------------------------------------
# Pure helpers — config normalization
# --------------------------------------------------------------------


class TestNormalizeMaxConcurrency:
    def test_default_when_missing(self):
        assert normalize_max_concurrency({}) == 10

    def test_explicit_max_concurrency(self):
        assert normalize_max_concurrency({"max_concurrency": "5"}) == 5

    def test_batchsize_alias(self):
        assert normalize_max_concurrency({"batchsize": "7"}) == 7

    def test_max_nodes_alias(self):
        assert normalize_max_concurrency({"max_nodes": "3"}) == 3

    def test_max_concurrency_wins_over_aliases(self):
        cfg = {"max_concurrency": "20", "batchsize": "5", "max_nodes": "8"}
        assert normalize_max_concurrency(cfg) == 20

    def test_clamped_to_one(self):
        assert normalize_max_concurrency({"max_concurrency": "0"}) == 1
        assert normalize_max_concurrency({"max_concurrency": "-3"}) == 1

    def test_garbled_value_falls_back_to_default(self):
        # parse_int returns the default on bad input.
        assert normalize_max_concurrency({"max_concurrency": "not a number"}) == 10


class TestResolveMultiplicity:
    def test_explicit_override(self):
        assert resolve_multiplicity(multiplicity=3, unpaired_electrons=0) == 3

    def test_unpaired_zero_means_singlet(self):
        assert resolve_multiplicity(multiplicity=None, unpaired_electrons=0) == 1

    def test_unpaired_one_means_doublet(self):
        assert resolve_multiplicity(multiplicity=None, unpaired_electrons=1) == 2

    def test_unpaired_two_means_triplet(self):
        assert resolve_multiplicity(multiplicity=None, unpaired_electrons=2) == 3

    def test_override_wins_over_unpaired(self):
        # Caller explicitly set multiplicity=5 — that wins even if
        # unpaired_electrons would imply something else.
        assert resolve_multiplicity(multiplicity=5, unpaired_electrons=10) == 5


# --------------------------------------------------------------------
# Pure helpers — stage_conformer_inputs
# --------------------------------------------------------------------


def _xyz_block(comment: str = "frame") -> str:
    """Tiny well-formed 3-atom xyz frame with a custom comment line."""
    return f"3\n{comment}\nC 0 0 0\nO 0 0 1.4\nH 0 0 -1.0\n"


class TestStageConformerInputs:
    def test_many_mode_copies_each(self, tmp_path):
        # Build per-conformer files and reference them from a fake
        # upstream artifacts dict (the prism/marc shape).
        src = tmp_path / "src"
        src.mkdir()
        for i in range(1, 4):
            (src / f"c{i}.xyz").write_text(_xyz_block(f"frame {i}"))

        artifacts = {
            "conformers": [
                {"path_abs": str((src / f"c{i}.xyz").resolve())}
                for i in range(1, 4)
            ]
        }
        staged_dir = tmp_path / "staged"
        paths = stage_conformer_inputs(
            upstream_artifacts=artifacts, staged_dir=staged_dir
        )
        assert len(paths) == 3
        # Files renamed with 1-based zero-padded conf_NNNN naming.
        assert [p.name for p in paths] == [
            "conf_0001.xyz",
            "conf_0002.xyz",
            "conf_0003.xyz",
        ]
        for p in paths:
            assert p.exists()

    def test_ensemble_mode_splits_multixyz(self, tmp_path):
        # Single multi-xyz file in xyz_ensemble bucket → split into
        # individual frames.
        src = tmp_path / "ensemble.xyz"
        src.write_text(
            "3\nframe 1\nC 0 0 0\nO 0 0 1.4\nH 0 0 -1.0\n"
            "3\nframe 2\nC 0 0 0\nO 0 0 1.5\nH 0 0 -1.0\n"
            "3\nframe 3\nC 0 0 0\nO 0 0 1.6\nH 0 0 -1.0\n"
        )
        artifacts = {"xyz_ensemble": [{"path_abs": str(src.resolve())}]}
        staged_dir = tmp_path / "staged"
        paths = stage_conformer_inputs(
            upstream_artifacts=artifacts, staged_dir=staged_dir
        )
        assert len(paths) == 3
        # Each staged frame should contain its own comment.
        for i, p in enumerate(paths, start=1):
            assert f"frame {i}" in p.read_text()

    def test_single_mode_one_frame(self, tmp_path):
        src = tmp_path / "single.xyz"
        src.write_text(_xyz_block("only one"))
        artifacts = {"xyz": [{"path_abs": str(src.resolve())}]}
        staged_dir = tmp_path / "staged"
        paths = stage_conformer_inputs(
            upstream_artifacts=artifacts, staged_dir=staged_dir
        )
        assert len(paths) == 1
        assert paths[0].name == "conf_0001.xyz"
        assert "only one" in paths[0].read_text()

    def test_no_xyz_inputs_raises(self, tmp_path):
        artifacts: dict[str, Any] = {}
        with pytest.raises(RuntimeError, match="no_xyz_inputs"):
            stage_conformer_inputs(
                upstream_artifacts=artifacts,
                staged_dir=tmp_path / "staged",
            )

    def test_priority_accepted_over_conformers(self, tmp_path):
        # ``accepted`` should be picked over ``conformers``.
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.xyz").write_text(_xyz_block("accepted file"))
        (src / "c.xyz").write_text(_xyz_block("conformers file"))

        artifacts = {
            "accepted": [{"path_abs": str((src / "a.xyz").resolve())}],
            "conformers": [{"path_abs": str((src / "c.xyz").resolve())}],
        }
        staged_dir = tmp_path / "staged"
        paths = stage_conformer_inputs(
            upstream_artifacts=artifacts, staged_dir=staged_dir
        )
        assert len(paths) == 1
        assert "accepted file" in paths[0].read_text()


# --------------------------------------------------------------------
# Pure helpers — build_task_dirs
# --------------------------------------------------------------------


class TestBuildTaskDirs:
    def test_creates_per_task_dirs(self, tmp_path):
        # Pre-stage two conformer xyz files.
        staged = tmp_path / "staged"
        staged.mkdir()
        for i in range(1, 3):
            (staged / f"conf_{i:04d}.xyz").write_text(_xyz_block(f"frame {i}"))

        tasks_root = tmp_path / "tasks"
        build_task_dirs(
            staged_paths=[staged / "conf_0001.xyz", staged / "conf_0002.xyz"],
            tasks_root=tasks_root,
            inp_text="! r2scan-3c\n",
        )

        for i in (1, 2):
            d = tasks_root / f"task_{i:04d}"
            assert d.is_dir()
            assert (d / "input.xyz").exists()
            assert (d / "orca_opt.inp").exists()
            assert (d / "orca_opt.inp").read_text() == "! r2scan-3c\n"
            # Per-task input.xyz contains the right frame.
            assert f"frame {i}" in (d / "input.xyz").read_text()

    def test_inp_text_shared_across_tasks(self, tmp_path):
        staged = tmp_path / "staged"
        staged.mkdir()
        (staged / "conf_0001.xyz").write_text(_xyz_block("a"))
        (staged / "conf_0002.xyz").write_text(_xyz_block("b"))
        tasks_root = tmp_path / "tasks"
        shared = "shared input text\n"
        build_task_dirs(
            staged_paths=[staged / "conf_0001.xyz", staged / "conf_0002.xyz"],
            tasks_root=tasks_root,
            inp_text=shared,
        )
        for i in (1, 2):
            assert (
                (tasks_root / f"task_{i:04d}" / "orca_opt.inp").read_text()
                == shared
            )

    def test_custom_inp_name(self, tmp_path):
        staged = tmp_path / "staged"
        staged.mkdir()
        (staged / "conf_0001.xyz").write_text(_xyz_block("x"))
        tasks_root = tmp_path / "tasks"
        build_task_dirs(
            staged_paths=[staged / "conf_0001.xyz"],
            tasks_root=tasks_root,
            inp_text="!\n",
            inp_name="orca_thermo.inp",
        )
        assert (tasks_root / "task_0001" / "orca_thermo.inp").exists()


# --------------------------------------------------------------------
# Pure helpers — collect_optimized_outputs
# --------------------------------------------------------------------


def _populate_task_outputs(
    tasks_root: Path,
    *,
    energies: list[Optional[float]],
    write_xyz: list[bool] | None = None,
) -> None:
    """Create ``task_XXXX/orca_opt.{xyz,out}`` for testing collection.

    ``energies[i]`` is None → skip the .out file (parse returns None).
    ``write_xyz[i]`` False → skip the .xyz file (collect treats as missing).
    """
    write_xyz = (
        write_xyz
        if write_xyz is not None
        else [True] * len(energies)
    )
    assert len(write_xyz) == len(energies)
    for i, (e, has_xyz) in enumerate(zip(energies, write_xyz), start=1):
        d = tasks_root / f"task_{i:04d}"
        d.mkdir(parents=True, exist_ok=True)
        if has_xyz:
            (d / "orca_opt.xyz").write_text(
                f"3\nopt_{i}\nC 0 0 0\nO 0 0 1.4\nH 0 0 -1.0\n"
            )
        if e is not None:
            (d / "orca_opt.out").write_text(
                f"FINAL SINGLE POINT ENERGY  {e:.9f}\n"
            )


class TestCollectOptimizedOutputs:
    def test_all_present(self, tmp_path):
        tasks_root = tmp_path / "tasks"
        opt_root = tmp_path / "opt"
        _populate_task_outputs(
            tasks_root, energies=[-100.0, -99.5, -99.9]
        )

        records, energies, paths, missing = collect_optimized_outputs(
            n_tasks=3,
            tasks_root=tasks_root,
            opt_root=opt_root,
        )
        assert len(records) == 3
        assert energies == [
            pytest.approx(-100.0),
            pytest.approx(-99.5),
            pytest.approx(-99.9),
        ]
        assert len(paths) == 3
        assert missing == []

        # Records carry index, label, energy_hartree.
        assert [r["index"] for r in records] == [1, 2, 3]
        assert [r["label"] for r in records] == [
            "conf_0001",
            "conf_0002",
            "conf_0003",
        ]
        assert records[0]["energy_hartree"] == pytest.approx(-100.0)
        # Files copied into opt_root.
        for p in paths:
            assert p.parent == opt_root
            assert p.exists()

    def test_missing_xyz_becomes_missing_record(self, tmp_path):
        tasks_root = tmp_path / "tasks"
        opt_root = tmp_path / "opt"
        _populate_task_outputs(
            tasks_root,
            energies=[-100.0, -99.0, -98.0],
            write_xyz=[True, False, True],
        )
        records, energies, paths, missing = collect_optimized_outputs(
            n_tasks=3,
            tasks_root=tasks_root,
            opt_root=opt_root,
        )
        # Task 2 has no xyz → missing record + no path; energy still
        # parsed from orca_opt.out.
        assert len(records) == 2
        assert [r["index"] for r in records] == [1, 3]
        assert len(paths) == 2
        assert missing == [
            {
                "error": "missing_orca_opt_xyz",
                "index": 2,
                "task_dir": str((tasks_root / "task_0002").resolve()),
            }
        ]
        # energies list is full-length and parallel.
        assert energies[1] == pytest.approx(-99.0)

    def test_missing_out_yields_none_energy(self, tmp_path):
        tasks_root = tmp_path / "tasks"
        opt_root = tmp_path / "opt"
        # xyz present but out missing.
        _populate_task_outputs(tasks_root, energies=[None])
        records, energies, paths, missing = collect_optimized_outputs(
            n_tasks=1,
            tasks_root=tasks_root,
            opt_root=opt_root,
        )
        assert energies == [None]
        # Record published (xyz exists) but no energy_hartree key.
        assert len(records) == 1
        assert "energy_hartree" not in records[0]

    def test_zero_tasks(self, tmp_path):
        records, energies, paths, missing = collect_optimized_outputs(
            n_tasks=0,
            tasks_root=tmp_path / "tasks",
            opt_root=tmp_path / "opt",
        )
        assert records == []
        assert energies == []
        assert paths == []
        assert missing == []


# --------------------------------------------------------------------
# End-to-end harness
# --------------------------------------------------------------------


def _make_upstream_with_conformers(
    tmp_path: Path, n: int = 3
) -> Path:
    """Build a fake upstream outputs/manifest.json with ``conformers``."""
    up_dir = tmp_path / "upstream"
    out_dir = up_dir / "outputs"
    conf_dir = out_dir / "conformers"
    conf_dir.mkdir(parents=True)

    m = Manifest.skeleton(step="crest", cwd=str(up_dir))
    for i in range(1, n + 1):
        path = conf_dir / f"conf_{i:04d}.xyz"
        path.write_text(_xyz_block(f"input frame {i}"))
        m.artifacts["conformers"].append(
            {
                "label": f"conf_{i:04d}",
                "path_abs": str(path.resolve()),
                "sha256": "0" * 64,
                "format": "xyz",
                "index": i,
            }
        )
    m_path = out_dir / "manifest.json"
    m.write(m_path)
    return m_path


def _pointer_text(manifest_path: Path, ok: bool = True) -> str:
    return Pointer.of(ok=ok, manifest_path=manifest_path).to_json_line()


def _make_fake_sbatch(
    *,
    jobid: str = "111222",
    return_ok: bool = True,
    fake_msg: str = "Submitted batch job 111222",
    populate_tasks: bool = True,
    energies: list[Optional[float]] | None = None,
    write_xyz: list[bool] | None = None,
):
    """Build a fake ``sbatch_submit`` that also drops in synthetic
    per-task outputs so the aggregator finds them.

    The real sbatch would SLURM the array, the cluster would run ORCA,
    and the per-task body would write ``orca_opt.xyz`` / ``orca_opt.out``
    on success. We short-circuit all that: when called, the fake walks
    the tasks_root sibling of ``slurm_path`` and creates synthetic
    output files directly.
    """

    def _fake(sbatch_exe, slurm_path, *, cwd):
        if populate_tasks:
            tasks_root = Path(cwd) / "tasks"
            if tasks_root.is_dir():
                # Count existing task dirs.
                existing = sorted(tasks_root.glob("task_*"))
                n = len(existing)
                if n:
                    # Default energies if not provided. Increase with
                    # index so task 1 is lowest (best.index == 1).
                    es = (
                        energies
                        if energies is not None
                        else [-100.0 + 0.01 * (i - 1) for i in range(1, n + 1)]
                    )
                    wxs = (
                        write_xyz
                        if write_xyz is not None
                        else [True] * n
                    )
                    for i, task_dir in enumerate(existing, start=1):
                        if i > len(es):
                            break
                        # Sentinels so progress counter sees "success".
                        status = task_dir / ".wf_status"
                        status.mkdir(exist_ok=True)
                        (status / "started").touch()
                        (status / "done_success").touch()
                        if wxs[i - 1]:
                            (task_dir / "orca_opt.xyz").write_text(
                                f"3\nopt {i}\nC 0 0 0\nO 0 0 1.4\nH 0 0 -1.0\n"
                            )
                        if es[i - 1] is not None:
                            (task_dir / "orca_opt.out").write_text(
                                f"FINAL SINGLE POINT ENERGY  {es[i - 1]:.9f}\n"
                            )
        if return_ok:
            return True, jobid, fake_msg
        return False, None, fake_msg

    return _fake


@pytest.fixture
def cluster_stub(monkeypatch):
    """Make sbatch / squeue / sacct discoverable AND provide deterministic
    fakes for sbatch_submit / squeue_has_any / sacct_states."""
    from scripps_workflow import slurm as slurm_mod

    fake_paths = {
        "sbatch": "/fake/bin/sbatch",
        "squeue": "/fake/bin/squeue",
        "sacct": "/fake/bin/sacct",
    }

    def _which(name):
        return fake_paths.get(name)

    monkeypatch.setattr(slurm_mod.shutil, "which", _which)

    # Default fakes — happy path.
    monkeypatch.setattr(oda, "sbatch_submit", _make_fake_sbatch())
    # Drained queue → monitor returns immediately.
    monkeypatch.setattr(oda, "squeue_has_any", lambda exe, jobid: False)
    # All tasks COMPLETED 0:0.
    monkeypatch.setattr(
        oda,
        "sacct_states",
        lambda exe, jobid: {
            f"{jobid}_1": ("COMPLETED", "0:0"),
            f"{jobid}_2": ("COMPLETED", "0:0"),
            f"{jobid}_3": ("COMPLETED", "0:0"),
        },
    )


def _run_node(
    tmp_path: Path,
    *config_tokens: str,
    n_conformers: int = 3,
    ok_pointer: bool = True,
) -> dict:
    """Invoke OrcaDftArray against a freshly-built upstream."""
    up_manifest_path = _make_upstream_with_conformers(tmp_path, n=n_conformers)
    pointer_text = _pointer_text(up_manifest_path, ok=ok_pointer)

    call_dir = tmp_path / "calls" / "orca_dft"
    call_dir.mkdir(parents=True)

    cwd = os.getcwd()
    os.chdir(call_dir)
    try:
        rc = OrcaDftArray().invoke(
            ["orca_dft_array", pointer_text, *config_tokens]
        )
    finally:
        os.chdir(cwd)
    assert rc == 0, "soft-fail invariant violated"
    m_path = call_dir / "outputs" / "manifest.json"
    assert m_path.exists()
    return json.loads(m_path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------


class TestHappyPath:
    def test_writes_expected_artifacts(self, tmp_path, cluster_stub):
        m = _run_node(tmp_path)
        assert m["ok"] is True
        assert m["step"] == "orca_dft_array"

        # Input conformers + submit script staged into files bucket.
        files_labels = {f["label"] for f in m["artifacts"]["files"]}
        assert "input_conf_0001" in files_labels
        assert "input_conf_0002" in files_labels
        assert "input_conf_0003" in files_labels
        assert "submit_array_slurm" in files_labels
        assert "orca_energies" in files_labels

        # Optimized ensemble published.
        ens = m["artifacts"]["xyz_ensemble"]
        assert len(ens) == 1
        assert ens[0]["label"] == "optimized_ensemble"
        assert Path(ens[0]["path_abs"]).exists()

        # Per-conformer optimized records.
        confs = m["artifacts"]["conformers"]
        assert [c["index"] for c in confs] == [1, 2, 3]
        for c in confs:
            assert "energy_hartree" in c
            assert Path(c["path_abs"]).exists()

        # rel_energy_kcal attached and the lowest is 0.0.
        rels = [c["rel_energy_kcal"] for c in confs]
        assert min(rels) == pytest.approx(0.0)
        # The fake energies decrease per-task, so task 1 is lowest.
        assert rels[0] == pytest.approx(0.0)

        # Best xyz published — index points at the lowest-energy task.
        xyz = m["artifacts"]["xyz"]
        assert len(xyz) == 1
        assert xyz[0]["label"] == "best"
        assert xyz[0]["index"] == 1

    def test_array_bucket_populated(self, tmp_path, cluster_stub):
        m = _run_node(tmp_path)
        arr = m["artifacts"]["array"]
        assert arr["n_tasks"] == 3
        assert arr["max_concurrency"] == 10
        assert arr["job_name"] == "orca_opt_array_3"
        assert arr["jobid"] == "111222"
        assert arr["submit_ok"] is True
        # progress_final reflects the drained-success state.
        prog = arr["progress_final"]
        assert prog["success"] == 3
        assert prog["processed"] == 3
        assert prog["left"] == 0
        # Aggregation flag.
        assert arr["aggregated"] is True

    def test_inputs_block_typed(self, tmp_path, cluster_stub):
        m = _run_node(
            tmp_path,
            "max_concurrency=5",
            "charge=-1",
            "unpaired_electrons=1",
            "solvent=ch2cl2",
            "keywords=r2scan-3c TightOpt",
            "maxcore=2000",
            "nprocs=4",
        )
        ins = m["inputs"]
        assert ins["max_concurrency"] == 5
        assert ins["charge"] == -1
        assert ins["unpaired_electrons"] == 1
        # Multiplicity = unpaired + 1 = 2.
        assert ins["multiplicity"] == 2
        assert ins["solvent"] == "ch2cl2"
        assert ins["keywords"] == "r2scan-3c TightOpt"
        assert ins["maxcore"] == 2000
        assert ins["nprocs"] == 4
        assert ins["n_input_conformers"] == 3

    def test_environment_records_executables(self, tmp_path, cluster_stub):
        m = _run_node(tmp_path)
        env = m["environment"]
        assert env["sbatch"] == "/fake/bin/sbatch"
        assert env["squeue"] == "/fake/bin/squeue"
        assert env["sacct"] == "/fake/bin/sacct"

    def test_orca_energies_file_written(self, tmp_path, cluster_stub):
        m = _run_node(tmp_path)
        files = {f["label"]: f for f in m["artifacts"]["files"]}
        e_path = Path(files["orca_energies"]["path_abs"])
        assert e_path.exists()
        text = e_path.read_text()
        # 3 rows, no NaN — all energies finite in the happy stub.
        lines = text.strip().splitlines()
        assert len(lines) == 3
        for line in lines:
            assert "NaN" not in line

    def test_maxcore_clamped(self, tmp_path, cluster_stub):
        # 100 < 500 floor → clamps to 500.
        m = _run_node(tmp_path, "maxcore=100")
        assert m["inputs"]["maxcore"] == 500

    def test_explicit_multiplicity_wins(self, tmp_path, cluster_stub):
        m = _run_node(tmp_path, "unpaired_electrons=1", "multiplicity=4")
        assert m["inputs"]["multiplicity"] == 4


# --------------------------------------------------------------------
# Failure paths
# --------------------------------------------------------------------


class TestFailures:
    def test_sbatch_not_found(self, tmp_path, monkeypatch):
        # which() returns None for sbatch.
        from scripps_workflow import slurm as slurm_mod

        monkeypatch.setattr(slurm_mod.shutil, "which", lambda name: None)
        # sbatch_submit shouldn't be reached, but stub it just in case.
        monkeypatch.setattr(
            oda, "sbatch_submit", lambda *a, **kw: (False, None, "boom")
        )
        monkeypatch.setattr(oda, "squeue_has_any", lambda exe, jobid: False)
        monkeypatch.setattr(oda, "sacct_states", lambda exe, jobid: {})

        m = _run_node(tmp_path)
        assert m["ok"] is False
        errors = [f["error"] for f in m["failures"]]
        assert "sbatch_not_found_on_PATH" in errors

    def test_sbatch_returns_failure(self, tmp_path, monkeypatch):
        from scripps_workflow import slurm as slurm_mod

        monkeypatch.setattr(
            slurm_mod.shutil,
            "which",
            lambda n: {
                "sbatch": "/fake/sbatch",
                "squeue": "/fake/squeue",
                "sacct": "/fake/sacct",
            }.get(n),
        )
        monkeypatch.setattr(
            oda,
            "sbatch_submit",
            _make_fake_sbatch(
                return_ok=False, fake_msg="error: bad partition"
            ),
        )
        monkeypatch.setattr(oda, "squeue_has_any", lambda exe, jobid: False)
        monkeypatch.setattr(oda, "sacct_states", lambda exe, jobid: {})

        m = _run_node(tmp_path)
        assert m["ok"] is False
        errors = {f["error"] for f in m["failures"]}
        assert "sbatch_failed" in errors

    def test_no_xyz_in_upstream_manifest(self, tmp_path, cluster_stub):
        # Build an upstream manifest with EMPTY conformer / xyz buckets.
        up_dir = tmp_path / "upstream"
        out_dir = up_dir / "outputs"
        out_dir.mkdir(parents=True)
        m = Manifest.skeleton(step="crest", cwd=str(up_dir))
        m_path = out_dir / "manifest.json"
        m.write(m_path)

        pointer_text = _pointer_text(m_path)
        call_dir = tmp_path / "calls" / "orca_dft"
        call_dir.mkdir(parents=True)
        cwd = os.getcwd()
        os.chdir(call_dir)
        try:
            rc = OrcaDftArray().invoke(["orca_dft_array", pointer_text])
        finally:
            os.chdir(cwd)
        assert rc == 0
        result = json.loads(
            (call_dir / "outputs" / "manifest.json").read_text()
        )
        assert result["ok"] is False
        errors = [f["error"] for f in result["failures"]]
        assert any(e.startswith("stage_inputs_failed") for e in errors)

    def test_monitor_timeout(self, tmp_path, monkeypatch):
        from scripps_workflow import slurm as slurm_mod

        monkeypatch.setattr(
            slurm_mod.shutil,
            "which",
            lambda n: {
                "sbatch": "/fake/sbatch",
                "squeue": "/fake/squeue",
                "sacct": "/fake/sacct",
            }.get(n),
        )
        # sbatch succeeds but does NOT populate per-task outputs.
        monkeypatch.setattr(
            oda,
            "sbatch_submit",
            _make_fake_sbatch(populate_tasks=False),
        )
        # squeue keeps the queue "busy" forever.
        monkeypatch.setattr(oda, "squeue_has_any", lambda exe, jobid: True)
        # Drive the timeout by stepping monotonic clock.
        ticks = iter([0.0] + [1e9] * 100)
        monkeypatch.setattr(slurm_mod.time, "monotonic", lambda: next(ticks))
        # Skip sleeping.
        monkeypatch.setattr(slurm_mod.time, "sleep", lambda s: None)
        monkeypatch.setattr(oda, "sacct_states", lambda exe, jobid: {})

        m = _run_node(tmp_path, "monitor_timeout_min=1")
        assert m["ok"] is False
        errors = [f["error"] for f in m["failures"]]
        assert "monitor_timeout" in errors

    def test_missing_per_task_xyz_surfaces_failures(
        self, tmp_path, monkeypatch
    ):
        from scripps_workflow import slurm as slurm_mod

        monkeypatch.setattr(
            slurm_mod.shutil,
            "which",
            lambda n: {
                "sbatch": "/fake/sbatch",
                "squeue": "/fake/squeue",
                "sacct": "/fake/sacct",
            }.get(n),
        )
        # Only task 1 produces an xyz; tasks 2 and 3 don't.
        monkeypatch.setattr(
            oda,
            "sbatch_submit",
            _make_fake_sbatch(
                energies=[-100.0, -99.0, -98.0],
                write_xyz=[True, False, False],
            ),
        )
        monkeypatch.setattr(oda, "squeue_has_any", lambda exe, jobid: False)
        monkeypatch.setattr(oda, "sacct_states", lambda exe, jobid: {})

        m = _run_node(tmp_path)
        # ok=False because of missing_orca_opt_xyz failures.
        assert m["ok"] is False
        errors = [f["error"] for f in m["failures"]]
        assert errors.count("missing_orca_opt_xyz") == 2
        # Task 1 should still be reported as a conformer.
        confs = m["artifacts"]["conformers"]
        assert [c["index"] for c in confs] == [1]

    def test_no_optimized_geometries_collected(self, tmp_path, monkeypatch):
        from scripps_workflow import slurm as slurm_mod

        monkeypatch.setattr(
            slurm_mod.shutil,
            "which",
            lambda n: {
                "sbatch": "/fake/sbatch",
                "squeue": "/fake/squeue",
                "sacct": "/fake/sacct",
            }.get(n),
        )
        # sbatch "succeeds" but every task fails to produce xyz.
        monkeypatch.setattr(
            oda,
            "sbatch_submit",
            _make_fake_sbatch(
                energies=[None, None, None],
                write_xyz=[False, False, False],
            ),
        )
        monkeypatch.setattr(oda, "squeue_has_any", lambda exe, jobid: False)
        monkeypatch.setattr(oda, "sacct_states", lambda exe, jobid: {})

        m = _run_node(tmp_path)
        assert m["ok"] is False
        errors = [f["error"] for f in m["failures"]]
        assert "no_optimized_geometries_collected" in errors

    def test_sacct_failure_records_surfaced(self, tmp_path, monkeypatch):
        from scripps_workflow import slurm as slurm_mod

        monkeypatch.setattr(
            slurm_mod.shutil,
            "which",
            lambda n: {
                "sbatch": "/fake/sbatch",
                "squeue": "/fake/squeue",
                "sacct": "/fake/sacct",
            }.get(n),
        )
        monkeypatch.setattr(oda, "sbatch_submit", _make_fake_sbatch())
        monkeypatch.setattr(oda, "squeue_has_any", lambda exe, jobid: False)
        # Task 2 reported as FAILED by sacct even though our fake
        # populated a successful xyz/out for it. The sacct path
        # surfaces the discrepancy as a structured failure.
        monkeypatch.setattr(
            oda,
            "sacct_states",
            lambda exe, jobid: {
                f"{jobid}_1": ("COMPLETED", "0:0"),
                f"{jobid}_2": ("FAILED", "1:0"),
                f"{jobid}_3": ("COMPLETED", "0:0"),
            },
        )

        m = _run_node(tmp_path)
        # Each conformer xyz exists, but the FAILED sacct entry pushes
        # ok=False.
        assert m["ok"] is False
        errors = [f for f in m["failures"] if f["error"] == "array_task_not_completed"]
        assert len(errors) == 1
        assert errors[0]["task"] == 2
        # The array bucket also has a sacct sub-dict.
        assert m["artifacts"]["array"]["sacct"]["111222_2"]["state"] == "FAILED"

    def test_bad_keywords_returns_one(self, tmp_path, cluster_stub):
        # Empty keywords → ValueError in parse_config → soft-fail, but
        # the engine still writes a manifest with ok=False.
        up_manifest_path = _make_upstream_with_conformers(tmp_path)
        pointer_text = _pointer_text(up_manifest_path)
        call_dir = tmp_path / "calls" / "orca_dft_bad"
        call_dir.mkdir(parents=True)
        cwd = os.getcwd()
        os.chdir(call_dir)
        try:
            rc = OrcaDftArray().invoke(
                ["orca_dft_array", pointer_text, "keywords="]
            )
        finally:
            os.chdir(cwd)
        # Soft-fail invariant: returncode 0; ok=False on the manifest.
        assert rc == 0
        result = json.loads(
            (call_dir / "outputs" / "manifest.json").read_text()
        )
        assert result["ok"] is False


# --------------------------------------------------------------------
# Node wiring (smoke)
# --------------------------------------------------------------------


class TestNodeWiring:
    def test_step_name(self):
        assert OrcaDftArray.step == "orca_dft_array"

    def test_requires_upstream(self):
        assert OrcaDftArray.requires_upstream is True
        assert OrcaDftArray.accepts_upstream is True

    def test_main_factory_attached(self):
        assert callable(oda.main)

    def test_default_keywords_constant(self):
        assert oda.DEFAULT_KEYWORDS == "r2scan-3c TightSCF TightOpt"

    def test_default_module_constant(self):
        assert oda.DEFAULT_ORCA_MODULE == "orca/6.0.0"

    def test_filename_constants(self):
        assert oda.ORCA_INP_NAME == "orca_opt.inp"
        assert oda.ORCA_OUT_NAME == "orca_opt.out"
        assert oda.ORCA_OPT_XYZ == "orca_opt.xyz"
