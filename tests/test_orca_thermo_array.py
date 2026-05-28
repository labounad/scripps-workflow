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


# --------------------------------------------------------------------
# system_class profile — element detection + basis swap
# --------------------------------------------------------------------


class TestDetectSystemClass:
    """``detect_system_class`` scans element symbols out of xyz files
    and returns a profile name. First-match-wins on the trigger dict;
    no triggers → 'organic'."""

    def _write_xyz(self, path: Path, atoms: list[tuple[str, float, float, float]]) -> None:
        lines = [str(len(atoms)), "comment"]
        for sym, x, y, z in atoms:
            lines.append(f"{sym} {x:.4f} {y:.4f} {z:.4f}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_pure_organic_returns_organic(self, tmp_path):
        xyz = tmp_path / "ethanol.xyz"
        self._write_xyz(xyz, [
            ("C", 0.0, 0.0, 0.0), ("C", 1.5, 0.0, 0.0),
            ("O", 2.4, 0.0, 0.0),
            ("H", -0.5, 0.9, 0.0), ("H", -0.5, -0.9, 0.0), ("H", -0.5, 0.0, 0.9),
        ])
        assert ota.detect_system_class([xyz]) == "organic"

    def test_pd_present_returns_organopd(self, tmp_path):
        xyz = tmp_path / "pd_complex.xyz"
        self._write_xyz(xyz, [
            ("Pd", 0.0, 0.0, 0.0),
            ("C", 1.8, 0.0, 0.0), ("C", -1.8, 0.0, 0.0),
            ("H", 2.5, 0.8, 0.0), ("H", -2.5, -0.8, 0.0),
        ])
        assert ota.detect_system_class([xyz]) == "organopd"

    def test_unreadable_file_does_not_raise(self, tmp_path):
        """A missing / unreadable xyz is skipped silently — detection
        falls back to whatever the remaining files say."""
        missing = tmp_path / "nope.xyz"
        present = tmp_path / "ok.xyz"
        self._write_xyz(present, [("Pd", 0, 0, 0), ("C", 1.8, 0, 0)])
        # Should not raise on the missing file, should still see the Pd
        # in the present file.
        assert ota.detect_system_class([missing, present]) == "organopd"

    def test_empty_list_returns_organic(self):
        assert ota.detect_system_class([]) == "organic"

    def test_only_scans_first_few_conformers(self, tmp_path):
        """Element composition is the same across conformers of one
        molecule; the helper caps at 3 to avoid O(N) work on 50-conf
        ensembles. Verify the cap by hiding a Pd-containing conformer
        at index 5."""
        for i in range(5):
            self._write_xyz(tmp_path / f"conf_{i}.xyz", [("C", 0, 0, 0)])
        # The "secret" Pd conformer at index 5 is BEYOND the scan cap.
        self._write_xyz(tmp_path / "conf_5.xyz", [("Pd", 0, 0, 0)])
        paths = sorted(tmp_path.glob("conf_*.xyz"))
        # The first 3 only contain C; the late Pd file is ignored.
        assert ota.detect_system_class(paths) == "organic"


class TestSystemClassProfileBasisSwap:
    """``_resolve_system_class_profile`` mutates cfg in place: maps
    ``auto`` to a concrete class via element scan, then swaps basis
    fields still at the organic defaults to the profile's values.
    Operator-set bases pass through unchanged."""

    def _ctx_with_upstream_xyzs(self, tmp_path: Path, *, has_pd: bool):
        """Build a minimal ctx with an upstream manifest whose
        conformer artifacts point at a Pd or pure-organic xyz."""
        from scripps_workflow.node import NodeContext

        xyz = tmp_path / "input.xyz"
        if has_pd:
            xyz.write_text("2\nfoo\nPd 0 0 0\nC 1.8 0 0\n", encoding="utf-8")
        else:
            xyz.write_text("2\nfoo\nC 0 0 0\nC 1.5 0 0\n", encoding="utf-8")

        upstream = Manifest.skeleton(step="orca_dft_array", cwd=tmp_path)
        upstream.artifacts["conformers"] = [
            {"path_abs": str(xyz.resolve()), "index": 1}
        ]
        outputs_dir = tmp_path / "outputs"
        outputs_dir.mkdir()
        return NodeContext(
            cwd=tmp_path,
            outputs_dir=outputs_dir,
            manifest_path=outputs_dir / "manifest.json",
            raw_argv=[],
            config={},
            upstream_pointer=None,
            upstream_manifest=upstream,
            manifest=Manifest.skeleton(step="orca_thermo_array", cwd=tmp_path),
            started_at_unix=0,
            started_at_perf=0.0,
        )

    def test_auto_organic_keeps_organic_defaults(self, tmp_path):
        ctx = self._ctx_with_upstream_xyzs(tmp_path, has_pd=False)
        cfg = {
            "system_class": "auto",
            "shielding_basis_h": ota.DEFAULT_SHIELDING_BASIS_H,
            "shielding_basis_c": ota.DEFAULT_SHIELDING_BASIS_C,
            "coupling_basis": ota.DEFAULT_COUPLING_BASIS,
        }
        resolved = ota._resolve_system_class_profile(cfg, ctx=ctx)
        assert resolved == "organic"
        assert cfg["system_class"] == "organic"
        # Bases unchanged.
        assert cfg["shielding_basis_h"] == ota.DEFAULT_SHIELDING_BASIS_H
        assert cfg["coupling_basis"] == ota.DEFAULT_COUPLING_BASIS

    def test_auto_pd_resolves_to_organopd_and_swaps_bases(self, tmp_path):
        ctx = self._ctx_with_upstream_xyzs(tmp_path, has_pd=True)
        cfg = {
            "system_class": "auto",
            "shielding_basis_h": ota.DEFAULT_SHIELDING_BASIS_H,
            "shielding_basis_c": ota.DEFAULT_SHIELDING_BASIS_C,
            "coupling_basis": ota.DEFAULT_COUPLING_BASIS,
        }
        resolved = ota._resolve_system_class_profile(cfg, ctx=ctx)
        assert resolved == "organopd"
        assert cfg["system_class"] == "organopd"
        # All three NMR bases swapped to def2-ZORA-TZVPP.
        assert cfg["shielding_basis_h"] == "def2-ZORA-TZVPP"
        assert cfg["shielding_basis_c"] == "def2-ZORA-TZVPP"
        assert cfg["coupling_basis"] == "def2-ZORA-TZVPP"

    def test_explicit_organopd_works_without_pd_in_geometry(self, tmp_path):
        """Operator can force the organopd profile even on a non-Pd
        molecule (e.g., for method development)."""
        ctx = self._ctx_with_upstream_xyzs(tmp_path, has_pd=False)
        cfg = {
            "system_class": "organopd",
            "shielding_basis_h": ota.DEFAULT_SHIELDING_BASIS_H,
            "shielding_basis_c": ota.DEFAULT_SHIELDING_BASIS_C,
            "coupling_basis": ota.DEFAULT_COUPLING_BASIS,
        }
        resolved = ota._resolve_system_class_profile(cfg, ctx=ctx)
        assert resolved == "organopd"
        assert cfg["shielding_basis_h"] == "def2-ZORA-TZVPP"

    def test_explicit_organic_on_pd_geometry_keeps_organic_bases(self, tmp_path):
        """Operator override beats auto-detection, when paired with
        ``on_uncovered_heavy_metal=warn`` (the default ``fail`` would
        raise — see :class:`TestOnUncoveredHeavyMetal`)."""
        ctx = self._ctx_with_upstream_xyzs(tmp_path, has_pd=True)
        cfg = {
            "system_class": "organic",
            "shielding_basis_h": ota.DEFAULT_SHIELDING_BASIS_H,
            "shielding_basis_c": ota.DEFAULT_SHIELDING_BASIS_C,
            "coupling_basis": ota.DEFAULT_COUPLING_BASIS,
            "on_uncovered_heavy_metal": "warn",
        }
        resolved = ota._resolve_system_class_profile(cfg, ctx=ctx)
        assert resolved == "organic"
        assert cfg["shielding_basis_h"] == ota.DEFAULT_SHIELDING_BASIS_H

    def test_operator_override_basis_passes_through(self, tmp_path):
        """A non-default basis the operator passed in must NOT be
        clobbered by the profile."""
        ctx = self._ctx_with_upstream_xyzs(tmp_path, has_pd=True)
        cfg = {
            "system_class": "auto",
            "shielding_basis_h": "pcSseg-2",     # operator's choice
            "shielding_basis_c": ota.DEFAULT_SHIELDING_BASIS_C,
            "coupling_basis": ota.DEFAULT_COUPLING_BASIS,
        }
        ota._resolve_system_class_profile(cfg, ctx=ctx)
        # H basis preserved (operator-set, not the organic default).
        assert cfg["shielding_basis_h"] == "pcSseg-2"
        # C and J bases swapped (they were at organic defaults).
        assert cfg["shielding_basis_c"] == "def2-ZORA-TZVPP"
        assert cfg["coupling_basis"] == "def2-ZORA-TZVPP"

    def test_unknown_class_raises(self, tmp_path):
        ctx = self._ctx_with_upstream_xyzs(tmp_path, has_pd=False)
        cfg = {
            "system_class": "uranium_carbide_v2",
            "shielding_basis_h": ota.DEFAULT_SHIELDING_BASIS_H,
            "shielding_basis_c": ota.DEFAULT_SHIELDING_BASIS_C,
            "coupling_basis": ota.DEFAULT_COUPLING_BASIS,
        }
        with pytest.raises(ValueError, match="unknown system_class"):
            ota._resolve_system_class_profile(cfg, ctx=ctx)


class TestOnUncoveredHeavyMetal:
    """Tier 2 element + explicit ``system_class=organic``: the
    ``on_uncovered_heavy_metal`` knob controls fail / warn /
    auto-promote behavior. Default is ``fail``."""

    def _ctx_with_upstream_xyzs(self, tmp_path: Path, *, heavy_metal: str = "Pd"):
        from scripps_workflow.node import NodeContext

        xyz = tmp_path / "input.xyz"
        xyz.write_text(
            f"2\nfoo\n{heavy_metal} 0 0 0\nC 1.8 0 0\n",
            encoding="utf-8",
        )
        upstream = Manifest.skeleton(step="orca_dft_array", cwd=tmp_path)
        upstream.artifacts["conformers"] = [
            {"path_abs": str(xyz.resolve()), "index": 1}
        ]
        outputs_dir = tmp_path / "outputs"
        outputs_dir.mkdir()
        return NodeContext(
            cwd=tmp_path,
            outputs_dir=outputs_dir,
            manifest_path=outputs_dir / "manifest.json",
            raw_argv=[],
            config={},
            upstream_pointer=None,
            upstream_manifest=upstream,
            manifest=Manifest.skeleton(step="orca_thermo_array", cwd=tmp_path),
            started_at_unix=0,
            started_at_perf=0.0,
        )

    def _organic_cfg(self, **over):
        cfg = {
            "system_class": "organic",
            "shielding_basis_h": ota.DEFAULT_SHIELDING_BASIS_H,
            "shielding_basis_c": ota.DEFAULT_SHIELDING_BASIS_C,
            "coupling_basis": ota.DEFAULT_COUPLING_BASIS,
        }
        cfg.update(over)
        return cfg

    def test_fail_policy_raises(self, tmp_path):
        ctx = self._ctx_with_upstream_xyzs(tmp_path)
        cfg = self._organic_cfg(on_uncovered_heavy_metal="fail")
        with pytest.raises(ValueError, match="Tier 2 element"):
            ota._resolve_system_class_profile(cfg, ctx=ctx)

    def test_default_policy_is_fail(self, tmp_path):
        ctx = self._ctx_with_upstream_xyzs(tmp_path)
        cfg = self._organic_cfg()    # no on_uncovered_heavy_metal set
        with pytest.raises(ValueError, match="Tier 2 element"):
            ota._resolve_system_class_profile(cfg, ctx=ctx)

    def test_warn_policy_continues(self, tmp_path, caplog):
        ctx = self._ctx_with_upstream_xyzs(tmp_path)
        cfg = self._organic_cfg(on_uncovered_heavy_metal="warn")
        resolved = ota._resolve_system_class_profile(cfg, ctx=ctx)
        assert resolved == "organic"
        # Bases unchanged — organic profile, no swap.
        assert cfg["shielding_basis_h"] == ota.DEFAULT_SHIELDING_BASIS_H

    def test_auto_switch_promotes_to_organopd_for_pd(self, tmp_path):
        ctx = self._ctx_with_upstream_xyzs(tmp_path, heavy_metal="Pd")
        cfg = self._organic_cfg(on_uncovered_heavy_metal="auto_switch_profile")
        resolved = ota._resolve_system_class_profile(cfg, ctx=ctx)
        assert resolved == "organopd"
        assert cfg["system_class"] == "organopd"
        # Bases got swapped after promotion.
        assert cfg["shielding_basis_h"] == "def2-ZORA-TZVPP"

    def test_auto_switch_promotes_to_organopt_for_pt(self, tmp_path):
        ctx = self._ctx_with_upstream_xyzs(tmp_path, heavy_metal="Pt")
        cfg = self._organic_cfg(on_uncovered_heavy_metal="auto_switch_profile")
        resolved = ota._resolve_system_class_profile(cfg, ctx=ctx)
        assert resolved == "organopt"

    def test_no_tier2_in_geometry_is_noop(self, tmp_path):
        """Pure organic geometry — the policy never fires."""
        from scripps_workflow.node import NodeContext

        xyz = tmp_path / "input.xyz"
        xyz.write_text("2\nfoo\nC 0 0 0\nC 1.5 0 0\n", encoding="utf-8")
        upstream = Manifest.skeleton(step="orca_dft_array", cwd=tmp_path)
        upstream.artifacts["conformers"] = [
            {"path_abs": str(xyz.resolve()), "index": 1}
        ]
        outputs_dir = tmp_path / "outputs"
        outputs_dir.mkdir()
        ctx = NodeContext(
            cwd=tmp_path,
            outputs_dir=outputs_dir,
            manifest_path=outputs_dir / "manifest.json",
            raw_argv=[],
            config={},
            upstream_pointer=None,
            upstream_manifest=upstream,
            manifest=Manifest.skeleton(step="orca_thermo_array", cwd=tmp_path),
            started_at_unix=0,
            started_at_perf=0.0,
        )
        cfg = self._organic_cfg(on_uncovered_heavy_metal="fail")
        resolved = ota._resolve_system_class_profile(cfg, ctx=ctx)
        assert resolved == "organic"


class TestBuildNmrInputFilesCoverage:
    """``build_nmr_input_files`` injects ``%basis newgto`` blocks when
    given an ``elements`` set containing atoms uncovered by the
    configured basis. Without ``elements`` (legacy callers), no
    coverage check runs and inputs render exactly as before."""

    def _cfg(self, **over):
        base = {
            "nprocs": 8, "maxcore": 4000, "charge": 0, "smd_solvent": None,
            "solvent": "CHCl3", "run_shielding_h": True,
            "run_shielding_c": True, "run_couplings": True,
            "shielding_method_h": "WP04",
            "shielding_basis_h": ota.DEFAULT_SHIELDING_BASIS_H,
            "shielding_method_c": "wB97X-D",
            "shielding_basis_c": ota.DEFAULT_SHIELDING_BASIS_C,
            "coupling_method": "mPW1PW91",
            "coupling_basis": ota.DEFAULT_COUPLING_BASIS,
            "coupling_pairs": ["all H"],
            "coupling_thresh_angstrom": 8.0,
            "nmr_aux_keywords": "TightSCF",
            "nmr_keywords_prefix": "",
            "heavy_atom_basis": ota.DEFAULT_HEAVY_ATOM_BASIS,
        }
        base.update(over)
        return base

    def test_no_elements_arg_is_legacy_behavior(self):
        """Passing ``elements=None`` (the default) disables the coverage
        check entirely — rendered input must match the old shape."""
        files = ota.build_nmr_input_files(
            cfg=self._cfg(), multiplicity=1,
        )
        for text in files.values():
            assert "newgto" not in text

    def test_organic_elements_organic_basis_is_noop(self):
        """C/H/N/O molecule in organic basis: no %basis block injected."""
        files = ota.build_nmr_input_files(
            cfg=self._cfg(), multiplicity=1,
            elements={"C", "H", "N", "O"},
        )
        for text in files.values():
            assert "newgto" not in text

    def test_bromine_supplements_h_and_j_but_not_c(self):
        """The aryl-bromide case. Per-basis coverage:
          * H basis (6-311++G(2d,p)) — diffuse Pople, stops at Ar.
            Br uncovered → supplement.
          * C basis (6-31G(d,p)) — Pople H–Kr. Br covered → no block.
          * J basis (pcJ-2) — organic main-group. Br uncovered → supplement.
        """
        files = ota.build_nmr_input_files(
            cfg=self._cfg(), multiplicity=1,
            elements={"C", "H", "Br"},
        )
        h_text = files[ota.ORCA_NMR_H_INP_NAME]
        c_text = files[ota.ORCA_NMR_C_INP_NAME]
        j_text = files[ota.ORCA_NMR_J_INP_NAME]
        assert 'newgto Br "def2-TZVPP" end' in h_text
        assert "newgto" not in c_text
        assert 'newgto Br "def2-TZVPP" end' in j_text

    def test_iodine_supplements_all_three_pople_jobs(self):
        """I (Z=53) sits above Kr — outside every Pople / pcJ coverage
        set. All three NMR inputs need supplementation."""
        cfg = self._cfg()
        files = ota.build_nmr_input_files(
            cfg=cfg, multiplicity=1, elements={"C", "H", "I"},
        )
        for name in (
            ota.ORCA_NMR_H_INP_NAME,
            ota.ORCA_NMR_C_INP_NAME,
            ota.ORCA_NMR_J_INP_NAME,
        ):
            assert 'newgto I "def2-TZVPP" end' in files[name]

    def test_supplement_block_appears_before_eprnmr(self):
        """The %basis block must appear pre-xyz (before %eprnmr can
        come post-xyz). Sanity check by string position."""
        files = ota.build_nmr_input_files(
            cfg=self._cfg(), multiplicity=1, elements={"C", "H", "Br"},
        )
        j_text = files[ota.ORCA_NMR_J_INP_NAME]
        # %basis comes before * xyzfile, %eprnmr comes after.
        assert j_text.index("%basis") < j_text.index("* xyzfile")
        assert j_text.index("* xyzfile") < j_text.index("%eprnmr")

    def test_custom_heavy_atom_basis_threads_through(self):
        """Operator-set ``heavy_atom_basis`` is used for the supplement."""
        files = ota.build_nmr_input_files(
            cfg=self._cfg(heavy_atom_basis="SARC-ZORA-TZVPP"),
            multiplicity=1,
            elements={"C", "H", "Br"},
        )
        j_text = files[ota.ORCA_NMR_J_INP_NAME]
        assert 'newgto Br "SARC-ZORA-TZVPP" end' in j_text


class TestComputeBasisFingerprints:
    """``compute_basis_fingerprints`` returns supplemented strings
    suitable for set_inputs / cache fingerprint use. The base cfg is
    NOT mutated."""

    def test_organic_elements_returns_empty_dict(self):
        cfg = {
            "shielding_basis_h": "6-311++G(2d,p)",
            "shielding_basis_c": "6-31G(d,p)",
            "coupling_basis": "pcJ-2",
            "heavy_atom_basis": "def2-TZVPP",
        }
        fps = ota.compute_basis_fingerprints(cfg, elements={"H", "C", "F"})
        assert fps == {}

    def test_bromine_supplementation_per_basis(self):
        """Br is the canonical Tier 1 case. Coverage outcome depends on
        which basis is on each NMR job:

          * 6-311++G(2d,p) — diffuse Pople, stops at Ar. Br uncovered.
          * 6-31G(d,p)    — Pople H–Kr. Br covered.
          * pcJ-2         — organic main-group. Br uncovered.
        """
        cfg = {
            "shielding_basis_h": "6-311++G(2d,p)",
            "shielding_basis_c": "6-31G(d,p)",
            "coupling_basis": "pcJ-2",
            "heavy_atom_basis": "def2-TZVPP",
        }
        fps = ota.compute_basis_fingerprints(cfg, elements={"H", "C", "Br"})
        assert fps["shielding_basis_h"] == "6-311++G(2d,p)+def2-TZVPP/heavy"
        assert "shielding_basis_c" not in fps   # 6-31G(d,p) covers Br
        assert fps["coupling_basis"] == "pcJ-2+def2-TZVPP/heavy"

    def test_iodine_supplements_all_three(self):
        """I (Z=53) is above Kr — outside every Pople / pcJ set."""
        cfg = {
            "shielding_basis_h": "6-311++G(2d,p)",
            "shielding_basis_c": "6-31G(d,p)",
            "coupling_basis": "pcJ-2",
            "heavy_atom_basis": "def2-TZVPP",
        }
        fps = ota.compute_basis_fingerprints(cfg, elements={"H", "C", "I"})
        for key in ("shielding_basis_h", "shielding_basis_c", "coupling_basis"):
            assert key in fps
            assert fps[key].endswith("+def2-TZVPP/heavy")

    def test_cfg_is_not_mutated(self):
        cfg = {
            "shielding_basis_h": "6-311++G(2d,p)",
            "shielding_basis_c": "6-31G(d,p)",
            "coupling_basis": "pcJ-2",
            "heavy_atom_basis": "def2-TZVPP",
        }
        snapshot = dict(cfg)
        _ = ota.compute_basis_fingerprints(cfg, elements={"H", "C", "Br"})
        assert cfg == snapshot


class TestBuildNmrInputFilesZora:
    """``build_nmr_input_files`` prepends ``ZORA`` to NMR keyword lines
    when ``cfg['nmr_keywords_prefix']`` is set (which the organopd
    profile does). Organic stays unchanged."""

    def _cfg(self, **over):
        base = {
            "nprocs": 8, "maxcore": 4000, "charge": 0, "smd_solvent": None,
            "solvent": "CHCl3", "run_shielding_h": True,
            "run_shielding_c": True, "run_couplings": True,
            "shielding_method_h": "WP04",
            "shielding_basis_h": ota.DEFAULT_SHIELDING_BASIS_H,
            "shielding_method_c": "wB97X-D",
            "shielding_basis_c": ota.DEFAULT_SHIELDING_BASIS_C,
            "coupling_method": "mPW1PW91",
            "coupling_basis": ota.DEFAULT_COUPLING_BASIS,
            "coupling_pairs": ["all H"],
            "coupling_thresh_angstrom": 8.0,
            "nmr_aux_keywords": "TightSCF",
            "nmr_keywords_prefix": "",
        }
        base.update(over)
        return base

    def test_organic_no_prefix(self):
        files = ota.build_nmr_input_files(cfg=self._cfg(), multiplicity=1)
        h_text = files[ota.ORCA_NMR_H_INP_NAME]
        # First non-empty ``!`` line should start with NMR, not ZORA.
        first = next(ln for ln in h_text.splitlines() if ln.startswith("!"))
        assert first.startswith("! NMR")
        assert "ZORA" not in first

    def test_organopd_prefix_zora(self):
        files = ota.build_nmr_input_files(
            cfg=self._cfg(
                nmr_keywords_prefix="ZORA",
                shielding_basis_h="def2-ZORA-TZVPP",
                shielding_basis_c="def2-ZORA-TZVPP",
                coupling_basis="def2-ZORA-TZVPP",
            ),
            multiplicity=1,
        )
        for inp_name in (
            ota.ORCA_NMR_H_INP_NAME,
            ota.ORCA_NMR_C_INP_NAME,
            ota.ORCA_NMR_J_INP_NAME,
        ):
            text = files[inp_name]
            first = next(ln for ln in text.splitlines() if ln.startswith("!"))
            # ZORA appears before NMR.
            assert "ZORA" in first
            assert first.index("ZORA") < first.index("NMR")
            # Basis is the ZORA-recontracted variant.
            assert "def2-ZORA-TZVPP" in first


# --------------------------------------------------------------------
# Heteronuclear J auto-expansion (¹⁹F / ³¹P)
# --------------------------------------------------------------------


class TestDetectNmrJPartners:
    """``detect_nmr_j_partners`` scans xyz files for ¹⁹F / ³¹P element
    symbols and returns the subset present, sorted. Empty when neither
    is found."""

    def _write_xyz(self, path: Path, atoms: list[tuple[str, float, float, float]]) -> None:
        lines = [str(len(atoms)), "comment"]
        for sym, x, y, z in atoms:
            lines.append(f"{sym} {x:.4f} {y:.4f} {z:.4f}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_pure_organic_returns_empty(self, tmp_path):
        xyz = tmp_path / "ethane.xyz"
        self._write_xyz(xyz, [("C", 0, 0, 0), ("C", 1.5, 0, 0), ("H", -0.5, 0, 0)])
        assert ota.detect_nmr_j_partners([xyz]) == []

    def test_fluorine_present_returns_f(self, tmp_path):
        xyz = tmp_path / "fluorobenzene.xyz"
        self._write_xyz(xyz, [("C", 0, 0, 0), ("F", 1.3, 0, 0)])
        assert ota.detect_nmr_j_partners([xyz]) == ["F"]

    def test_phosphorus_present_returns_p(self, tmp_path):
        xyz = tmp_path / "pme3.xyz"
        self._write_xyz(xyz, [("P", 0, 0, 0), ("C", 1.8, 0, 0)])
        assert ota.detect_nmr_j_partners([xyz]) == ["P"]

    def test_both_returns_sorted(self, tmp_path):
        xyz = tmp_path / "fluorophos.xyz"
        self._write_xyz(xyz, [("P", 0, 0, 0), ("F", 1.6, 0, 0)])
        # Sorted alphabetically: F before P.
        assert ota.detect_nmr_j_partners([xyz]) == ["F", "P"]

    def test_pd_present_does_not_trigger(self, tmp_path):
        """Pd is intentionally NOT a J partner here — quadrupolar nuclei
        need different J-coupling methodology than mPW1PW91/pcJ-2."""
        xyz = tmp_path / "pd_complex.xyz"
        self._write_xyz(xyz, [("Pd", 0, 0, 0), ("C", 1.8, 0, 0)])
        assert ota.detect_nmr_j_partners([xyz]) == []


class TestExpandCouplingPairsForHeteronuclear:
    """``_expand_coupling_pairs_for_heteronuclear`` appends ``\"all
    <X>\"`` selectors for detected partners, leaving operator-set
    entries (and any existing entry for the same partner) alone."""

    def test_appends_F_when_not_present(self):
        out = ota._expand_coupling_pairs_for_heteronuclear(
            ["all H"], detected_partners=["F"],
        )
        assert out == ["all H", "all F"]

    def test_appends_both_when_neither_present(self):
        out = ota._expand_coupling_pairs_for_heteronuclear(
            ["all H"], detected_partners=["F", "P"],
        )
        assert out == ["all H", "all F", "all P"]

    def test_skips_existing_partner(self):
        """Operator-set ``\"all F\"`` is not duplicated."""
        out = ota._expand_coupling_pairs_for_heteronuclear(
            ["all H", "all F"], detected_partners=["F"],
        )
        assert out == ["all H", "all F"]

    def test_case_insensitive_dedup(self):
        out = ota._expand_coupling_pairs_for_heteronuclear(
            ["all h", "all f"], detected_partners=["F"],
        )
        assert out == ["all h", "all f"]

    def test_preserves_atom_index_selectors(self):
        """Non-``\"all\"`` selectors (e.g., ``\"1, 3, 5\"``) pass through."""
        out = ota._expand_coupling_pairs_for_heteronuclear(
            ["1, 3, 5", "all H"], detected_partners=["F"],
        )
        assert out == ["1, 3, 5", "all H", "all F"]

    def test_empty_detected_returns_unchanged(self):
        out = ota._expand_coupling_pairs_for_heteronuclear(
            ["all H"], detected_partners=[],
        )
        assert out == ["all H"]


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
        # Default behavior: low-level freq and high-level SP are staged
        # as separate ORCA input files. This avoids ORCA method-state
        # leakage across a ``$new_job`` boundary while the SLURM wrapper
        # still appends the SP output into orca_thermo.out for the legacy
        # thermo parser.
        m = _run_node(tmp_path)
        assert (
            m["inputs"]["singlepoint_keywords"]
            == "wB97M-V def2-TZVPP TightSCF DEFGRID3"
        )
        arr = m["artifacts"]["array"]
        task_dir = Path(arr["tasks_root_abs"]) / "task_0001"
        thermo_text = (task_dir / "orca_thermo.inp").read_text()
        sp_text = (task_dir / "orca_thermo_sp.inp").read_text()
        assert "$new_job" not in thermo_text
        assert "! r2scan-3c TightSCF Freq" in thermo_text
        assert "! wB97M-V def2-TZVPP TightSCF DEFGRID3" in sp_text
        assert thermo_text.count("* xyzfile 0 1 input.xyz") == 1
        assert sp_text.count("* xyzfile 0 1 input.xyz") == 1
        submit = Path(arr["submit_slurm_abs"]).read_text()
        assert 'run_orca_job 2 ' in submit
        assert 'orca_thermo_sp.inp' in submit
        assert 'WF CONCATENATED ORCA OUTPUT: orca_thermo_sp.out' in submit

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
        assert not (Path(arr["tasks_root_abs"]) / "task_0001" / "orca_thermo_sp.inp").exists()
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
        task_dir = Path(arr["tasks_root_abs"]) / "task_0001"
        thermo_text = (task_dir / "orca_thermo.inp").read_text()
        sp_text = (task_dir / "orca_thermo_sp.inp").read_text()
        assert "$new_job" not in thermo_text
        assert "! B3LYP D4 def2-TZVP TightSCF" in sp_text
        assert "wB97M-V" not in sp_text

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
        # wB97M-V/def2-TZVPP single point on the same geometry. The node
        # now runs the SP as a separate ORCA process to avoid method-state
        # leakage across ORCA ``$new_job`` boundaries.
        assert (
            ota.DEFAULT_SINGLEPOINT_KEYWORDS
            == "wB97M-V def2-TZVPP TightSCF DEFGRID3"
        )

    def test_filename_constants(self):
        assert ota.ORCA_INP_NAME == "orca_thermo.inp"
        assert ota.ORCA_OUT_NAME == "orca_thermo.out"
        # Sanity: NOT the dft-array filenames.
        assert ota.ORCA_INP_NAME != "orca_opt.inp"
        assert ota.ORCA_OUT_NAME != "orca_opt.out"
