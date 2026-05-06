"""Tests for the orca_thermo_array SLURM-array freq/thermo node.

The node is the sister of orca_dft_array. ORCA + SLURM are not
available in CI / the test sandbox so the same monkeypatch trick is
used:

    * :func:`scripps_workflow.nodes.orca_thermo_array.sbatch_submit` —
      fake a successful submit AND populate ``orca_thermo.out`` per task
      (FINAL E line + ``ORCA TERMINATED NORMALLY`` footer + sentinel
      files) so the aggregator finds something to aggregate.
    * :func:`scripps_workflow.nodes.orca_thermo_array.squeue_has_any` —
      always returns False (queue drained on first poll).
    * :func:`scripps_workflow.nodes.orca_thermo_array.sacct_states` —
      returns synthetic per-task state map.
    * :func:`shutil.which` (in :mod:`scripps_workflow.slurm`) — sbatch /
      squeue / sacct discoverable.

Coverage:

    * Pure helpers: build_thermo_task_dirs, collect_thermo_outputs
      (happy path, missing energy, no NORMALLY footer).
    * Happy path: conformers staged, sbatch invoked, queue drained,
      thermo.energies written, best.xyz copied from staged input,
      input_ensemble published, conformer records reference
      ``orca_out_abs``.
    * Failures: sbatch missing, sbatch returns non-zero, monitor
      timeout, missing FINAL E, missing NORMALLY footer.
    * Manifest: array bucket, environment records, sacct surfaced as
      ``array_task_not_completed`` failures.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import pytest

from scripps_workflow.nodes import orca_thermo_array as ota
from scripps_workflow.nodes.orca_thermo_array import (
    OrcaThermoArray,
    build_thermo_task_dirs,
    collect_thermo_outputs,
)
from scripps_workflow.pointer import Pointer
from scripps_workflow.schema import Manifest


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------


def _xyz_block(comment: str = "frame") -> str:
    """Tiny well-formed 3-atom xyz frame with a custom comment line."""
    return f"3\n{comment}\nC 0 0 0\nO 0 0 1.4\nH 0 0 -1.0\n"


def _thermo_out(
    energy: float | None,
    *,
    terminated_normally: bool = True,
) -> str:
    """Synthesize an ``orca_thermo.out`` body for tests.

    ``energy=None`` produces a file with no FINAL E line; the parser
    returns None, the collector emits ``missing_or_unparsed_energy``.
    ``terminated_normally=False`` omits the ``ORCA TERMINATED
    NORMALLY`` footer; the collector emits
    ``orca_not_terminated_normally``.
    """
    lines: list[str] = ["...lots of header noise...", ""]
    if energy is not None:
        lines.append(f"FINAL SINGLE POINT ENERGY  {energy:.9f}")
    lines.append("...thermochemistry block would go here...")
    if terminated_normally:
        lines.append("                     ****ORCA TERMINATED NORMALLY****")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------
# Pure helpers — build_thermo_task_dirs
# --------------------------------------------------------------------


class TestBuildThermoTaskDirs:
    def test_creates_per_task_dirs(self, tmp_path):
        staged = tmp_path / "staged"
        staged.mkdir()
        for i in range(1, 3):
            (staged / f"conf_{i:04d}.xyz").write_text(_xyz_block(f"frame {i}"))

        tasks_root = tmp_path / "tasks"
        build_thermo_task_dirs(
            staged_paths=[
                staged / "conf_0001.xyz",
                staged / "conf_0002.xyz",
            ],
            tasks_root=tasks_root,
            inp_text="! r2scan-3c TightSCF Freq\n",
        )

        for i in (1, 2):
            d = tasks_root / f"task_{i:04d}"
            assert d.is_dir()
            assert (d / "input.xyz").exists()
            # NOTE: default inp_name is orca_thermo.inp, NOT orca_opt.inp.
            assert (d / "orca_thermo.inp").exists()
            assert (
                (d / "orca_thermo.inp").read_text()
                == "! r2scan-3c TightSCF Freq\n"
            )
            assert f"frame {i}" in (d / "input.xyz").read_text()

    def test_default_inp_name_is_thermo(self, tmp_path):
        # Lock the default — this is the exact filename the SLURM body
        # will reference, so a typo here would silently break the job.
        staged = tmp_path / "staged"
        staged.mkdir()
        (staged / "conf_0001.xyz").write_text(_xyz_block("a"))
        tasks_root = tmp_path / "tasks"
        build_thermo_task_dirs(
            staged_paths=[staged / "conf_0001.xyz"],
            tasks_root=tasks_root,
            inp_text="!\n",
        )
        assert (tasks_root / "task_0001" / "orca_thermo.inp").exists()
        # Sanity: NOT the dft-array name.
        assert not (tasks_root / "task_0001" / "orca_opt.inp").exists()


# --------------------------------------------------------------------
# Pure helpers — collect_thermo_outputs
# --------------------------------------------------------------------


def _populate_thermo_task_outputs(
    tasks_root: Path,
    staged_dir: Path,
    *,
    energies: list[Optional[float]],
    terminated_normally: list[bool] | None = None,
) -> None:
    """Create ``task_XXXX/orca_thermo.out`` AND staged input xyz files
    for testing :func:`collect_thermo_outputs`."""
    if terminated_normally is None:
        terminated_normally = [True] * len(energies)
    assert len(terminated_normally) == len(energies)

    staged_dir.mkdir(parents=True, exist_ok=True)
    for i, (e, ok_term) in enumerate(
        zip(energies, terminated_normally), start=1
    ):
        d = tasks_root / f"task_{i:04d}"
        d.mkdir(parents=True, exist_ok=True)
        (staged_dir / f"conf_{i:04d}.xyz").write_text(
            _xyz_block(f"input {i}")
        )
        (d / "orca_thermo.out").write_text(
            _thermo_out(e, terminated_normally=ok_term)
        )


class TestCollectThermoOutputs:
    def test_all_present_and_normal(self, tmp_path):
        tasks_root = tmp_path / "tasks"
        staged = tmp_path / "staged"
        _populate_thermo_task_outputs(
            tasks_root, staged, energies=[-100.0, -99.5, -99.9]
        )

        records, energies, failures = collect_thermo_outputs(
            n_tasks=3,
            tasks_root=tasks_root,
            staged_dir=staged,
        )

        assert len(records) == 3
        # Records are 1:1 with task indices (no skipping).
        assert [r["index"] for r in records] == [1, 2, 3]
        assert [r["label"] for r in records] == [
            "conf_0001",
            "conf_0002",
            "conf_0003",
        ]
        # Records reference the STAGED INPUT xyz, not a new geometry.
        for r in records:
            assert r["path_abs"].endswith(f"conf_{r['index']:04d}.xyz")
            # And the orca_out_abs field for the thermo aggregator.
            assert r["orca_out_abs"].endswith(
                f"task_{r['index']:04d}/orca_thermo.out"
            )
            assert r["format"] == "xyz"
            # task_dir_abs lets the aggregator walk into the task dir
            # for additional artifacts (gradients, hessian, etc.).
            assert r["task_dir_abs"].endswith(f"task_{r['index']:04d}")

        assert energies == [
            pytest.approx(-100.0),
            pytest.approx(-99.5),
            pytest.approx(-99.9),
        ]
        # All terminated normally + all parseable → no failures.
        assert failures == []

        # energy_hartree attached to each record on success.
        assert records[0]["energy_hartree"] == pytest.approx(-100.0)

    def test_missing_energy_emits_failure(self, tmp_path):
        tasks_root = tmp_path / "tasks"
        staged = tmp_path / "staged"
        # Task 2 has an .out file but no FINAL E line.
        _populate_thermo_task_outputs(
            tasks_root,
            staged,
            energies=[-100.0, None, -99.0],
        )
        records, energies, failures = collect_thermo_outputs(
            n_tasks=3, tasks_root=tasks_root, staged_dir=staged
        )
        assert energies == [
            pytest.approx(-100.0),
            None,
            pytest.approx(-99.0),
        ]
        # Records still all 3 present — index 2 just has no
        # energy_hartree key.
        assert len(records) == 3
        assert "energy_hartree" not in records[1]
        # missing_or_unparsed_energy surfaced for index 2.
        errs = [(f["error"], f["index"]) for f in failures]
        assert ("missing_or_unparsed_energy", 2) in errs

    def test_no_normal_footer_emits_failure(self, tmp_path):
        # Task 3 has a parseable energy but the run didn't terminate
        # normally — walltime kill is the typical cause.
        tasks_root = tmp_path / "tasks"
        staged = tmp_path / "staged"
        _populate_thermo_task_outputs(
            tasks_root,
            staged,
            energies=[-100.0, -99.5, -99.0],
            terminated_normally=[True, True, False],
        )
        records, energies, failures = collect_thermo_outputs(
            n_tasks=3, tasks_root=tasks_root, staged_dir=staged
        )
        assert energies == [
            pytest.approx(-100.0),
            pytest.approx(-99.5),
            pytest.approx(-99.0),
        ]
        # Energy is present BUT a not_terminated_normally fail is
        # recorded. The aggregator NEEDS the Freq section to actually
        # finish, so we treat this as a hard fail.
        errs = [(f["error"], f["index"]) for f in failures]
        assert ("orca_not_terminated_normally", 3) in errs
        assert ("missing_or_unparsed_energy", 3) not in errs

    def test_missing_out_file_emits_both_failures(self, tmp_path):
        # No .out file at all → both "missing energy" AND "not
        # terminated normally" fire; the aggregator will see the
        # ``missing_or_unparsed_energy`` first but both are valuable
        # signals for the operator.
        tasks_root = tmp_path / "tasks"
        staged = tmp_path / "staged"
        staged.mkdir()
        (staged / "conf_0001.xyz").write_text(_xyz_block("input 1"))
        # Don't create the task dir at all.
        records, energies, failures = collect_thermo_outputs(
            n_tasks=1, tasks_root=tasks_root, staged_dir=staged
        )
        assert energies == [None]
        # Single record — orca_out_abs is None because the file
        # doesn't exist.
        assert len(records) == 1
        assert records[0]["orca_out_abs"] is None
        errs = [f["error"] for f in failures]
        assert "missing_or_unparsed_energy" in errs
        assert "orca_not_terminated_normally" in errs

    def test_zero_tasks(self, tmp_path):
        records, energies, failures = collect_thermo_outputs(
            n_tasks=0,
            tasks_root=tmp_path / "tasks",
            staged_dir=tmp_path / "staged",
        )
        assert records == []
        assert energies == []
        assert failures == []


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

    m = Manifest.skeleton(step="orca_dft_array", cwd=str(up_dir))
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
    terminated_normally: list[bool] | None = None,
):
    """Build a fake ``sbatch_submit`` that drops in synthetic
    ``orca_thermo.out`` files (with FINAL E + NORMAL footer) so the
    aggregator finds them. Default energies make task 1 the lowest, so
    ``best.index == 1`` in the happy path.
    """

    def _fake(sbatch_exe, slurm_path, *, cwd):
        if populate_tasks:
            tasks_root = Path(cwd) / "tasks"
            if tasks_root.is_dir():
                existing = sorted(tasks_root.glob("task_*"))
                n = len(existing)
                if n:
                    es = (
                        energies
                        if energies is not None
                        else [-100.0 + 0.01 * (i - 1) for i in range(1, n + 1)]
                    )
                    norms = (
                        terminated_normally
                        if terminated_normally is not None
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
                        # Note: the thermo node does NOT produce a new
                        # xyz. Only the .out file is dropped.
                        (task_dir / "orca_thermo.out").write_text(
                            _thermo_out(
                                es[i - 1],
                                terminated_normally=norms[i - 1],
                            )
                        )
        if return_ok:
            return True, jobid, fake_msg
        return False, None, fake_msg

    return _fake


@pytest.fixture
def cluster_stub(monkeypatch):
    """Make sbatch / squeue / sacct discoverable AND provide
    deterministic fakes for sbatch_submit / squeue_has_any /
    sacct_states. Default: 3-task happy path."""
    from scripps_workflow import slurm as slurm_mod

    fake_paths = {
        "sbatch": "/fake/bin/sbatch",
        "squeue": "/fake/bin/squeue",
        "sacct": "/fake/bin/sacct",
    }

    monkeypatch.setattr(
        slurm_mod.shutil, "which", lambda name: fake_paths.get(name)
    )
    monkeypatch.setattr(ota, "sbatch_submit", _make_fake_sbatch())
    monkeypatch.setattr(ota, "squeue_has_any", lambda exe, jobid: False)
    monkeypatch.setattr(
        ota,
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
    """Invoke OrcaThermoArray against a freshly-built upstream."""
    up_manifest_path = _make_upstream_with_conformers(tmp_path, n=n_conformers)
    pointer_text = _pointer_text(up_manifest_path, ok=ok_pointer)

    call_dir = tmp_path / "calls" / "orca_thermo"
    call_dir.mkdir(parents=True)

    cwd = os.getcwd()
    os.chdir(call_dir)
    try:
        rc = OrcaThermoArray().invoke(
            ["orca_thermo_array", pointer_text, *config_tokens]
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
        assert m["step"] == "orca_thermo_array"

        # Input conformers + submit script + thermo.energies in the
        # files bucket.
        files_labels = {f["label"] for f in m["artifacts"]["files"]}
        assert "input_conf_0001" in files_labels
        assert "input_conf_0002" in files_labels
        assert "input_conf_0003" in files_labels
        assert "submit_array_slurm" in files_labels
        assert "thermo_energies" in files_labels

        # Input ensemble published (as the ONLY xyz_ensemble).
        ens = m["artifacts"]["xyz_ensemble"]
        assert len(ens) == 1
        assert ens[0]["label"] == "input_ensemble"
        assert Path(ens[0]["path_abs"]).exists()

        # Per-conformer records — one per task, even if some failed.
        confs = m["artifacts"]["conformers"]
        assert [c["index"] for c in confs] == [1, 2, 3]
        for c in confs:
            assert "energy_hartree" in c
            # path_abs points at the staged INPUT xyz (NOT a new
            # geometry, as the freq calc preserves coordinates).
            assert "input_conformers" in c["path_abs"]
            assert Path(c["path_abs"]).exists()
            # orca_out_abs points at the per-task .out for the thermo
            # aggregator to parse.
            assert c["orca_out_abs"].endswith("orca_thermo.out")
            assert Path(c["orca_out_abs"]).exists()

        # rel_energy_kcal attached and the lowest is 0.0.
        rels = [c["rel_energy_kcal"] for c in confs]
        assert min(rels) == pytest.approx(0.0)
        # Default fake energies make task 1 the lowest.
        assert rels[0] == pytest.approx(0.0)

        # Best xyz published — index points at the lowest-energy task.
        xyz = m["artifacts"]["xyz"]
        assert len(xyz) == 1
        assert xyz[0]["label"] == "best"
        assert xyz[0]["index"] == 1
        # And the best.xyz file LIVES in outputs/thermo/, not in a new
        # optimization output dir.
        assert "/thermo/" in xyz[0]["path_abs"]

    def test_array_bucket_populated(self, tmp_path, cluster_stub):
        m = _run_node(tmp_path)
        arr = m["artifacts"]["array"]
        assert arr["n_tasks"] == 3
        assert arr["max_concurrency"] == 10
        assert arr["job_name"] == "orca_thermo_array_3"
        assert arr["jobid"] == "111222"
        assert arr["submit_ok"] is True
        prog = arr["progress_final"]
        assert prog["success"] == 3
        assert prog["processed"] == 3
        assert prog["left"] == 0
        assert arr["aggregated"] is True

    def test_inputs_block_typed(self, tmp_path, cluster_stub):
        m = _run_node(
            tmp_path,
            "max_concurrency=5",
            "charge=-1",
            "unpaired_electrons=1",
            "solvent=ch2cl2",
            "keywords=r2scan-3c TightSCF Freq",
            "maxcore=2000",
            "nprocs=8",
        )
        ins = m["inputs"]
        assert ins["max_concurrency"] == 5
        assert ins["charge"] == -1
        assert ins["unpaired_electrons"] == 1
        assert ins["multiplicity"] == 2
        assert ins["solvent"] == "ch2cl2"
        assert ins["keywords"] == "r2scan-3c TightSCF Freq"
        assert ins["maxcore"] == 2000
        assert ins["nprocs"] == 8
        assert ins["n_input_conformers"] == 3

    def test_environment_records_executables(self, tmp_path, cluster_stub):
        m = _run_node(tmp_path)
        env = m["environment"]
        assert env["sbatch"] == "/fake/bin/sbatch"
        assert env["squeue"] == "/fake/bin/squeue"
        assert env["sacct"] == "/fake/bin/sacct"

    def test_thermo_energies_file_written(self, tmp_path, cluster_stub):
        m = _run_node(tmp_path)
        files = {f["label"]: f for f in m["artifacts"]["files"]}
        e_path = Path(files["thermo_energies"]["path_abs"])
        assert e_path.exists()
        # Lives under outputs/thermo/ (NOT optimized_conformers/).
        assert "/thermo/" in str(e_path)
        text = e_path.read_text()
        lines = text.strip().splitlines()
        assert len(lines) == 3
        for line in lines:
            assert "NaN" not in line

    def test_default_keywords_are_freq(self, tmp_path, cluster_stub):
        # The DFT array node defaults to TightOpt; this node must
        # default to Freq. If the default ever drifts the SLURM job
        # would silently re-optimize geometries instead of running
        # frequencies.
        m = _run_node(tmp_path)
        assert m["inputs"]["keywords"] == "r2scan-3c TightSCF Freq"
        # And the per-task .inp on disk reflects this.
        arr = m["artifacts"]["array"]
        task1_inp = (
            Path(arr["tasks_root_abs"]) / "task_0001" / "orca_thermo.inp"
        )
        assert task1_inp.exists()
        text = task1_inp.read_text()
        assert text.startswith("! r2scan-3c TightSCF Freq\n")

    def test_default_nprocs_is_8(self, tmp_path, cluster_stub):
        m = _run_node(tmp_path)
        # All array nodes share an 8-cpu default — freq calcs
        # parallelize well, but 8 is the cluster-wide sweet spot.
        assert m["inputs"]["nprocs"] == 8

    def test_default_singlepoint_keywords_in_input(self, tmp_path, cluster_stub):
        # Default behavior: composite freq + high-level SP separated by
        # ``$new_job``. The rendered orca_thermo.inp contains BOTH the
        # low-level freq keywords AND the wB97M-V/def2-TZVPP SP block.
        # The default carries RIJCOSX + DEFGRID3 because wB97M-V's VV10
        # nonlocal kernel needs RI-J for the Coulomb integrals to be
        # tractable at TZVPP and DEFGRID3 to get the thermo right
        # (commit 99f8f85).
        m = _run_node(tmp_path)
        assert (
            m["inputs"]["singlepoint_keywords"]
            == "wB97M-V def2-TZVPP TightSCF RIJCOSX DEFGRID3"
        )
        arr = m["artifacts"]["array"]
        task1_inp = (
            Path(arr["tasks_root_abs"]) / "task_0001" / "orca_thermo.inp"
        )
        text = task1_inp.read_text()
        assert "$new_job" in text
        assert "! wB97M-V def2-TZVPP TightSCF RIJCOSX DEFGRID3" in text
        # Both jobs share the staged xyz geometry.
        assert text.count("* xyzfile 0 1 input.xyz") == 2

    def test_singlepoint_keywords_disabled(self, tmp_path, cluster_stub):
        # ``singlepoint_keywords=none`` collapses the compound input to
        # a plain single-job freq calculation — no $new_job separator,
        # no SP block.
        m = _run_node(tmp_path, "singlepoint_keywords=none")
        assert m["inputs"]["singlepoint_keywords"] is None
        arr = m["artifacts"]["array"]
        task1_inp = (
            Path(arr["tasks_root_abs"]) / "task_0001" / "orca_thermo.inp"
        )
        text = task1_inp.read_text()
        assert "$new_job" not in text
        assert "wB97M-V" not in text
        # Only ONE xyzfile line in the single-job form.
        assert text.count("* xyzfile 0 1 input.xyz") == 1

    def test_singlepoint_keywords_custom(self, tmp_path, cluster_stub):
        # Custom SP-keywords override survives parse_config and lands
        # in the rendered .inp.
        m = _run_node(
            tmp_path, "singlepoint_keywords=B3LYP D4 def2-TZVP TightSCF"
        )
        assert (
            m["inputs"]["singlepoint_keywords"]
            == "B3LYP D4 def2-TZVP TightSCF"
        )
        arr = m["artifacts"]["array"]
        text = (
            Path(arr["tasks_root_abs"]) / "task_0001" / "orca_thermo.inp"
        ).read_text()
        assert "$new_job" in text
        assert "! B3LYP D4 def2-TZVP TightSCF" in text
        assert "wB97M-V" not in text

    def test_explicit_multiplicity_wins(self, tmp_path, cluster_stub):
        m = _run_node(tmp_path, "unpaired_electrons=1", "multiplicity=4")
        assert m["inputs"]["multiplicity"] == 4

    def test_input_ensemble_concat_in_order(self, tmp_path, cluster_stub):
        m = _run_node(tmp_path)
        ens = m["artifacts"]["xyz_ensemble"][0]
        text = Path(ens["path_abs"]).read_text()
        # Comments preserved and in order.
        i1 = text.index("input frame 1")
        i2 = text.index("input frame 2")
        i3 = text.index("input frame 3")
        assert i1 < i2 < i3


# --------------------------------------------------------------------
# Failure paths
# --------------------------------------------------------------------


class TestFailures:
    def test_sbatch_not_found(self, tmp_path, monkeypatch):
        from scripps_workflow import slurm as slurm_mod

        monkeypatch.setattr(slurm_mod.shutil, "which", lambda name: None)
        monkeypatch.setattr(
            ota, "sbatch_submit", lambda *a, **kw: (False, None, "boom")
        )
        monkeypatch.setattr(ota, "squeue_has_any", lambda exe, jobid: False)
        monkeypatch.setattr(ota, "sacct_states", lambda exe, jobid: {})

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
            ota,
            "sbatch_submit",
            _make_fake_sbatch(
                return_ok=False, fake_msg="error: bad partition"
            ),
        )
        monkeypatch.setattr(ota, "squeue_has_any", lambda exe, jobid: False)
        monkeypatch.setattr(ota, "sacct_states", lambda exe, jobid: {})

        m = _run_node(tmp_path)
        assert m["ok"] is False
        errors = {f["error"] for f in m["failures"]}
        assert "sbatch_failed" in errors

    def test_no_xyz_in_upstream_manifest(self, tmp_path, cluster_stub):
        up_dir = tmp_path / "upstream"
        out_dir = up_dir / "outputs"
        out_dir.mkdir(parents=True)
        m = Manifest.skeleton(step="orca_dft_array", cwd=str(up_dir))
        m_path = out_dir / "manifest.json"
        m.write(m_path)

        pointer_text = _pointer_text(m_path)
        call_dir = tmp_path / "calls" / "orca_thermo"
        call_dir.mkdir(parents=True)
        cwd = os.getcwd()
        os.chdir(call_dir)
        try:
            rc = OrcaThermoArray().invoke(["orca_thermo_array", pointer_text])
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
        monkeypatch.setattr(
            ota,
            "sbatch_submit",
            _make_fake_sbatch(populate_tasks=False),
        )
        monkeypatch.setattr(ota, "squeue_has_any", lambda exe, jobid: True)
        ticks = iter([0.0] + [1e9] * 100)
        monkeypatch.setattr(slurm_mod.time, "monotonic", lambda: next(ticks))
        monkeypatch.setattr(slurm_mod.time, "sleep", lambda s: None)
        monkeypatch.setattr(ota, "sacct_states", lambda exe, jobid: {})

        m = _run_node(tmp_path, "monitor_timeout_min=1")
        assert m["ok"] is False
        errors = [f["error"] for f in m["failures"]]
        assert "monitor_timeout" in errors

    def test_missing_energy_surfaces_failure(self, tmp_path, monkeypatch):
        # Tasks 2 and 3 produce .out files with NO FINAL E line.
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
            ota,
            "sbatch_submit",
            _make_fake_sbatch(
                energies=[-100.0, None, None],
                terminated_normally=[True, True, True],
            ),
        )
        monkeypatch.setattr(ota, "squeue_has_any", lambda exe, jobid: False)
        monkeypatch.setattr(ota, "sacct_states", lambda exe, jobid: {})

        m = _run_node(tmp_path)
        assert m["ok"] is False
        errors = [f["error"] for f in m["failures"]]
        assert errors.count("missing_or_unparsed_energy") == 2
        # No "not terminated normally" failures here — the footer was
        # written even though FINAL E was missing.
        assert "orca_not_terminated_normally" not in errors
        # Conformer records still all 3 published — only task 1 has
        # an energy_hartree.
        confs = m["artifacts"]["conformers"]
        assert [c["index"] for c in confs] == [1, 2, 3]
        assert "energy_hartree" in confs[0]
        assert "energy_hartree" not in confs[1]
        assert "energy_hartree" not in confs[2]

    def test_no_normal_footer_surfaces_failure(self, tmp_path, monkeypatch):
        # All three tasks parse a FINAL E, but task 2 was killed before
        # the Freq finished — no NORMAL footer. The aggregator NEEDS
        # the freq to finish, so this is a hard fail.
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
            ota,
            "sbatch_submit",
            _make_fake_sbatch(
                energies=[-100.0, -99.5, -99.0],
                terminated_normally=[True, False, True],
            ),
        )
        monkeypatch.setattr(ota, "squeue_has_any", lambda exe, jobid: False)
        monkeypatch.setattr(ota, "sacct_states", lambda exe, jobid: {})

        m = _run_node(tmp_path)
        assert m["ok"] is False
        errors = [(f["error"], f.get("index")) for f in m["failures"]]
        assert ("orca_not_terminated_normally", 2) in errors
        # No missing_or_unparsed_energy — the FINAL E line WAS there.
        assert ("missing_or_unparsed_energy", 2) not in errors

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
        monkeypatch.setattr(ota, "sbatch_submit", _make_fake_sbatch())
        monkeypatch.setattr(ota, "squeue_has_any", lambda exe, jobid: False)
        monkeypatch.setattr(
            ota,
            "sacct_states",
            lambda exe, jobid: {
                f"{jobid}_1": ("COMPLETED", "0:0"),
                f"{jobid}_2": ("FAILED", "1:0"),
                f"{jobid}_3": ("COMPLETED", "0:0"),
            },
        )

        m = _run_node(tmp_path)
        assert m["ok"] is False
        errors = [
            f for f in m["failures"] if f["error"] == "array_task_not_completed"
        ]
        assert len(errors) == 1
        assert errors[0]["task"] == 2
        assert m["artifacts"]["array"]["sacct"]["111222_2"]["state"] == "FAILED"

    def test_bad_keywords_returns_one(self, tmp_path, cluster_stub):
        up_manifest_path = _make_upstream_with_conformers(tmp_path)
        pointer_text = _pointer_text(up_manifest_path)
        call_dir = tmp_path / "calls" / "orca_thermo_bad"
        call_dir.mkdir(parents=True)
        cwd = os.getcwd()
        os.chdir(call_dir)
        try:
            rc = OrcaThermoArray().invoke(
                ["orca_thermo_array", pointer_text, "keywords="]
            )
        finally:
            os.chdir(cwd)
        # Soft-fail invariant: returncode 0; ok=False.
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
        assert OrcaThermoArray.step == "orca_thermo_array"

    def test_requires_upstream(self):
        assert OrcaThermoArray.requires_upstream is True
        assert OrcaThermoArray.accepts_upstream is True

    def test_main_factory_attached(self):
        assert callable(ota.main)

    def test_default_keywords_constant(self):
        assert ota.DEFAULT_KEYWORDS == "r2scan-3c TightSCF Freq"

    def test_default_module_constant(self):
        assert ota.DEFAULT_ORCA_MODULE == "orca/6.0.0"

    def test_default_nprocs_constant(self):
        # All array nodes share an 8-cpu default.
        assert ota.DEFAULT_NPROCS == 8

    def test_default_singlepoint_keywords_constant(self):
        # The composite Gibbs protocol pairs the r2scan-3c freq with a
        # wB97M-V/def2-TZVPP single point on the same geometry. RIJCOSX
        # is required for wB97M-V's VV10 nonlocal kernel to be tractable
        # at TZVPP; DEFGRID3 makes the SCF convergence + thermochemistry
        # numerically stable. (commit 99f8f85)
        assert (
            ota.DEFAULT_SINGLEPOINT_KEYWORDS
            == "wB97M-V def2-TZVPP TightSCF RIJCOSX DEFGRID3"
        )

    def test_filename_constants(self):
        assert ota.ORCA_INP_NAME == "orca_thermo.inp"
        assert ota.ORCA_OUT_NAME == "orca_thermo.out"
        # Sanity: NOT the dft-array filenames.
        assert ota.ORCA_INP_NAME != "orca_opt.inp"
        assert ota.ORCA_OUT_NAME != "orca_opt.out"
