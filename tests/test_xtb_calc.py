"""Tests for the xtb_calc node.

xtb is not available in CI / the test sandbox, so we monkeypatch
:func:`scripps_workflow.nodes.xtb_calc.run_xtb` (the only place we shell out
to xtb) and :func:`shutil.which`. The stub creates plausible scratch files
in the run directory so the orchestration code's "did xtb produce
xtbopt.xyz / xtbout.json" branches can be exercised.

Coverage:

    * Pure helpers: normalize_theory, normalize_solvent, normalize_opt_level,
      parse_calculations, base_xtb_cmd, resolve_threads, find_first_xyz_path.
    * Happy path: optimize-only, optimize-then-SP chain (uses xtbopt.xyz),
      SP-only (uses upstream geometry).
    * Manifest population: artifacts.xyz / files / xtbout_json / logs /
      operations buckets, inputs block, environment block.
    * Failure paths: xtb missing, no upstream xyz, no operations requested,
      xtb returns nonzero, optimize produces no xtbopt.xyz.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from scripps_workflow.nodes import xtb_calc as xc
from scripps_workflow.nodes.xtb_calc import (
    ALPB_SOLVENTS,
    OPT_LEVELS,
    OPS_ORDER,
    XtbCalc,
    base_xtb_cmd,
    find_first_xyz_path,
    normalize_opt_level,
    normalize_solvent,
    normalize_theory,
    parse_calculations,
    resolve_threads,
)
from scripps_workflow.pointer import Pointer
from scripps_workflow.schema import Manifest


# --------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------


class TestNormalizeTheory:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("GFN2-XTB", "GFN2-XTB"),
            ("gfn2-xtb", "GFN2-XTB"),
            ("GFN2", "GFN2-XTB"),
            ("gfn2xtb", "GFN2-XTB"),
            ("", "GFN2-XTB"),
            (None, "GFN2-XTB"),
            ("GFN1", "GFN1-XTB"),
            ("gfn1-xtb", "GFN1-XTB"),
            ("GFN-FF", "GFN-FF"),
            ("gff", "GFN-FF"),
            ("gfnff", "GFN-FF"),
        ],
    )
    def test_aliases(self, raw, expected):
        assert normalize_theory(raw) == expected

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown theory"):
            normalize_theory("DFT-D4")


class TestNormalizeSolvent:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            (None, None),
            ("", None),
            ("none", None),
            ("vacuum", None),
            ("water", "water"),
            ("WATER", "water"),
            (" Water ", "water"),
            ("ch2cl2", "ch2cl2"),
        ],
    )
    def test_cases(self, raw, expected):
        assert normalize_solvent(raw) == expected

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown ALPB solvent"):
            normalize_solvent("liquid_helium")

    def test_alpb_solvents_well_known(self):
        # If someone tweaks the table, these load-bearing names must stay.
        for s in ("water", "methanol", "ch2cl2", "thf", "dmso"):
            assert s in ALPB_SOLVENTS


class TestNormalizeOptLevel:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("", "tight"),
            (None, "tight"),
            ("tight", "tight"),
            ("VERYTIGHT", "verytight"),
            ("Crude", "crude"),
        ],
    )
    def test_cases(self, raw, expected):
        assert normalize_opt_level(raw) == expected

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown opt level"):
            normalize_opt_level("supertight")

    def test_full_set(self):
        # Spot-check: ensure all canonical names round-trip.
        for lvl in OPT_LEVELS:
            assert normalize_opt_level(lvl) == lvl


class TestParseCalculations:
    def test_default_is_optimize(self):
        assert parse_calculations(None) == ["optimize"]
        assert parse_calculations("") == ["optimize"]

    def test_csv(self):
        # Order is canonical (optimize first), regardless of input order.
        assert parse_calculations("sp_energy,optimize") == ["optimize", "sp_energy"]

    def test_list(self):
        assert parse_calculations(["optimize", "sp_hessian"]) == [
            "optimize",
            "sp_hessian",
        ]

    def test_aliases(self):
        # Mix of UI labels, lower-case, short forms.
        assert parse_calculations(["Geometry Optimization", "sp", "grad"]) == [
            "optimize",
            "sp_energy",
            "sp_gradient",
        ]

    def test_dedup(self):
        assert parse_calculations(["optimize", "optimize", "opt"]) == ["optimize"]

    def test_json_object_gui_payload(self):
        s = json.dumps(
            {
                "Geometry Optimization": True,
                "SP Energy": True,
                "SP Gradient": False,
                "SP Hessian": False,
            }
        )
        assert parse_calculations(s) == ["optimize", "sp_energy"]

    def test_json_object_all_false(self):
        s = json.dumps({"Geometry Optimization": False, "SP Energy": False})
        assert parse_calculations(s) == []

    def test_json_array(self):
        s = json.dumps(["optimize", "sp_hessian"])
        assert parse_calculations(s) == ["optimize", "sp_hessian"]

    def test_unknown_token_raises(self):
        with pytest.raises(ValueError, match="Unknown calculation token"):
            parse_calculations("vibrational_modes")


class TestBaseXtbCmd:
    def test_gfn2_default(self):
        cmd = base_xtb_cmd(
            xtb_exe="/bin/xtb",
            theory="GFN2-XTB",
            charge=0,
            uhf=0,
            solvent=None,
            threads=1,
            write_json=True,
        )
        # Anchor the prefix; downstream we just want to know flags are present.
        assert cmd[0] == "/bin/xtb"
        assert cmd[1] == "input.xyz"
        assert "--gfn" in cmd and cmd[cmd.index("--gfn") + 1] == "2"
        assert "--chrg" in cmd and cmd[cmd.index("--chrg") + 1] == "0"
        assert "--uhf" in cmd and cmd[cmd.index("--uhf") + 1] == "0"
        assert "--alpb" not in cmd
        assert "--json" in cmd
        assert "-P" in cmd and cmd[cmd.index("-P") + 1] == "1"

    def test_gfnff_drops_gfn_number(self):
        cmd = base_xtb_cmd(
            xtb_exe="xtb",
            theory="GFN-FF",
            charge=0,
            uhf=0,
            solvent=None,
            threads=1,
            write_json=False,
        )
        assert "--gfnff" in cmd
        assert "--gfn" not in cmd
        assert "--json" not in cmd

    def test_solvent_emits_alpb(self):
        cmd = base_xtb_cmd(
            xtb_exe="xtb",
            theory="GFN2-XTB",
            charge=-1,
            uhf=1,
            solvent="water",
            threads=4,
            write_json=True,
        )
        i = cmd.index("--alpb")
        assert cmd[i + 1] == "water"
        assert cmd[cmd.index("--chrg") + 1] == "-1"
        assert cmd[cmd.index("--uhf") + 1] == "1"
        assert cmd[cmd.index("-P") + 1] == "4"

    def test_unknown_theory_raises(self):
        with pytest.raises(ValueError, match="Unsupported theory"):
            base_xtb_cmd(
                xtb_exe="xtb",
                theory="BOGUS",
                charge=0,
                uhf=0,
                solvent=None,
                threads=1,
                write_json=False,
            )


class TestResolveThreads:
    def test_positive_passthrough(self):
        assert resolve_threads(8) == 8

    def test_zero_uses_slurm(self, monkeypatch):
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "16")
        assert resolve_threads(0) == 16

    def test_zero_no_slurm_falls_back_to_one(self, monkeypatch):
        for k in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "OMP_NUM_THREADS"):
            monkeypatch.delenv(k, raising=False)
        assert resolve_threads(0) == 1


class TestFindFirstXyzPath:
    def _manifest_with_xyz(self, path_abs: str) -> Manifest:
        m = Manifest.skeleton(step="up", cwd="/tmp")
        m.artifacts["xyz"] = [
            {
                "label": "embed_xyz",
                "path_abs": path_abs,
                "sha256": "0" * 64,
                "format": "xyz",
            }
        ]
        return m

    def test_returns_existing_path(self, tmp_path):
        xyz = tmp_path / "molecule.xyz"
        xyz.write_text("3\n\nC 0 0 0\nO 0 0 1\nH 0 0 -1\n")
        m = self._manifest_with_xyz(str(xyz))
        assert find_first_xyz_path(m) == xyz

    def test_no_xyz_bucket(self):
        m = Manifest.skeleton(step="up", cwd="/tmp")
        with pytest.raises(ValueError, match="no .xyz. artifacts"):
            find_first_xyz_path(m)

    def test_path_must_be_absolute(self):
        m = self._manifest_with_xyz("relative/path.xyz")
        with pytest.raises(ValueError, match="not absolute"):
            find_first_xyz_path(m)

    def test_path_must_exist(self, tmp_path):
        m = self._manifest_with_xyz(str(tmp_path / "nope.xyz"))
        with pytest.raises(FileNotFoundError):
            find_first_xyz_path(m)


# --------------------------------------------------------------------
# End-to-end node fixtures + stub
# --------------------------------------------------------------------


def _make_upstream(tmp_path: Path, xyz_text: str = None) -> tuple[Path, Path]:
    """Create an upstream outputs/ tree with manifest + xyz. Returns
    ``(upstream_manifest_path, upstream_xyz_path)``."""
    up_dir = tmp_path / "upstream"
    out_dir = up_dir / "outputs"
    xyz_dir = out_dir / "xyz"
    xyz_dir.mkdir(parents=True)
    xyz_path = xyz_dir / "input.xyz"
    xyz_path.write_text(
        xyz_text
        if xyz_text is not None
        else "3\nethanol\nC 0 0 0\nO 0 0 1.4\nH 0 0 -1.0\n"
    )

    m = Manifest.skeleton(step="smiles_to_3d", cwd=str(up_dir))
    m.artifacts["xyz"] = [
        {
            "label": "embed_xyz",
            "path_abs": str(xyz_path.resolve()),
            "sha256": "f" * 64,
            "format": "xyz",
            "name": "ethanol",
            "smiles": "CCO",
            "num_atoms": 9,
            "num_heavy_atoms": 3,
        }
    ]
    m_path = out_dir / "manifest.json"
    m.write(m_path)
    return m_path, xyz_path


def _pointer_text(manifest_path: Path, ok: bool = True) -> str:
    return Pointer.of(ok=ok, manifest_path=manifest_path).to_json_line()


def _stub_run_xtb_factory(produce_xtbopt: bool = True, write_xtbout: bool = True, rc: int = 0):
    """Build a stub for ``run_xtb`` that fakes the side-effects xtb produces."""

    def _stub(cmd, *, cwd, stdout_path, stderr_path):
        cwd.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text(
            "fake xtb output\n  | TOTAL ENERGY  -1.234567 Eh   |\nnormal termination of xtb\n"
        )
        stderr_path.write_text("")
        if produce_xtbopt and "--opt" in cmd:
            xtbopt = Path(cwd) / "xtbopt.xyz"
            xtbopt.write_text("3\noptimized\nC 0 0 0\nO 0 0 1.42\nH 0 0 -1.0\n")
        if write_xtbout and "--json" in cmd:
            xtbout = Path(cwd) / "xtbout.json"
            xtbout.write_text(json.dumps({"total energy": -1.234567}))
        return rc, 0.05

    return _stub


@pytest.fixture
def stub_xtb(monkeypatch):
    """Make ``shutil.which('xtb')`` succeed and ``run_xtb`` produce fake outputs."""
    import shutil as _shutil

    def _which(name):
        if name == "xtb":
            return "/fake/bin/xtb"
        return _shutil.which(name)  # delegate for any other lookups

    monkeypatch.setattr(xc.shutil, "which", _which)
    monkeypatch.setattr(xc, "run_xtb", _stub_run_xtb_factory())
    # Wipe SLURM env so resolve_threads is deterministic.
    for k in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "OMP_NUM_THREADS"):
        monkeypatch.delenv(k, raising=False)


def _run_node(tmp_path: Path, *config_tokens: str, ok_pointer: bool = True) -> dict:
    """Invoke XtbCalc against a freshly-built upstream. Returns the manifest dict.

    The node's call directory is ``tmp_path / "calls" / "xtb"`` so the upstream
    tree at ``tmp_path / "upstream" / "outputs"`` is untouched.
    """
    up_manifest_path, _ = _make_upstream(tmp_path)
    pointer_text = _pointer_text(up_manifest_path, ok=ok_pointer)

    call_dir = tmp_path / "calls" / "xtb"
    call_dir.mkdir(parents=True)

    cwd = os.getcwd()
    os.chdir(call_dir)
    try:
        rc = XtbCalc().invoke(["xtb_calc", pointer_text, *config_tokens])
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
    def test_optimize_only_writes_expected_artifacts(self, tmp_path, stub_xtb):
        m = _run_node(tmp_path, "calculations=optimize")
        assert m["ok"] is True
        assert m["step"] == "xtb_calc"

        # input_xyz staged as a 'files' artifact.
        files = m["artifacts"]["files"]
        assert any(f.get("label") == "input_xyz" for f in files)

        # xtbopt.xyz published into the xyz bucket.
        xyz = m["artifacts"]["xyz"]
        assert len(xyz) == 1
        assert xyz[0]["label"] == "xtbopt"
        assert Path(xyz[0]["path_abs"]).exists()
        assert xyz[0]["format"] == "xyz"

        # xtbout.json published.
        xj = m["artifacts"]["xtbout_json"]
        assert len(xj) == 1 and xj[0]["op"] == "optimize"

        # Logs: stdout + stderr for the one op.
        logs = m["artifacts"]["logs"]
        assert {l["label"] for l in logs} == {"optimize_stdout", "optimize_stderr"}

        # Operations bucket: one record with rc=0.
        ops = m["artifacts"]["operations"]
        assert len(ops) == 1
        assert ops[0]["op"] == "optimize"
        assert ops[0]["returncode"] == 0
        assert ops[0]["cmd"][0] == "/fake/bin/xtb"
        assert "--opt" in ops[0]["cmd"]

    def test_inputs_block_is_typed(self, tmp_path, stub_xtb):
        # The framework's parse_config -> ctx.config wiring is what makes
        # threads_requested an int (0) rather than the string "0".
        m = _run_node(
            tmp_path,
            "calculations=optimize",
            "theory=gfn1-xtb",
            "solvent=water",
            "charge=-1",
            "unpaired_electrons=1",
            "opt_level=verytight",
            "threads=4",
            "write_json=false",
        )
        ins = m["inputs"]
        assert ins["theory"] == "GFN1-XTB"
        assert ins["solvent"] == "water"
        assert ins["charge"] == -1
        assert ins["unpaired_electrons"] == 1
        assert ins["opt_level"] == "verytight"
        assert ins["threads_requested"] == 4
        assert ins["write_json"] is False
        assert ins["calculations"] == ["optimize"]

    def test_optimize_then_sp_chain_uses_xtbopt(self, tmp_path, stub_xtb):
        # The crucial chain semantics: SP ops should run on the optimized
        # geometry (xtbopt.xyz), not the upstream geometry.
        m = _run_node(
            tmp_path,
            "calculations=optimize,sp_energy",
        )
        ops = {op["op"]: op for op in m["artifacts"]["operations"]}
        assert set(ops.keys()) == {"optimize", "sp_energy"}

        # The SP op's input geometry should point at outputs/xtbopt.xyz, not
        # at the upstream input.xyz path.
        sp_input = Path(ops["sp_energy"]["input_geom_abs"])
        assert sp_input.name == "xtbopt.xyz"
        assert sp_input.parent.name == "outputs"

    def test_sp_only_uses_upstream_geometry(self, tmp_path, stub_xtb):
        m = _run_node(tmp_path, "calculations=sp_energy")
        ops = m["artifacts"]["operations"]
        assert len(ops) == 1
        assert ops[0]["op"] == "sp_energy"
        # No optimize, no xtbopt.xyz — SP should run on the staged input.xyz.
        sp_input = Path(ops[0]["input_geom_abs"])
        assert sp_input.name == "input.xyz"

        # No xyz artifact (we didn't optimize anything to publish).
        assert m["artifacts"]["xyz"] == []

    def test_environment_records_xtb_path(self, tmp_path, stub_xtb):
        m = _run_node(tmp_path, "calculations=optimize")
        assert m["environment"]["xtb"] == "/fake/bin/xtb"
        assert m["environment"]["threads"] == 1  # auto fallback

    def test_upstream_block_filled(self, tmp_path, stub_xtb):
        m = _run_node(tmp_path, "calculations=optimize")
        up = m["upstream"]
        assert up["pointer_schema"] == "wf.pointer.v1"
        assert up["ok"] is True
        assert up["manifest_path"].endswith("manifest.json")

    def test_full_op_set(self, tmp_path, stub_xtb):
        m = _run_node(
            tmp_path,
            "calculations=optimize,sp_energy,sp_gradient,sp_hessian",
        )
        ops = [o["op"] for o in m["artifacts"]["operations"]]
        # Run order: optimize first, then SP ops in canonical order.
        assert ops == ["optimize", "sp_energy", "sp_gradient", "sp_hessian"]

        # Each op contributes its own xtbout_json + 2 log records.
        assert len(m["artifacts"]["xtbout_json"]) == 4
        assert len(m["artifacts"]["logs"]) == 8


# --------------------------------------------------------------------
# Failure paths
# --------------------------------------------------------------------


class TestFailures:
    def test_xtb_not_on_path(self, tmp_path, monkeypatch):
        # Don't activate stub_xtb; instead make which return None.
        monkeypatch.setattr(xc.shutil, "which", lambda name: None)
        monkeypatch.setattr(xc, "run_xtb", _stub_run_xtb_factory())

        m = _run_node(tmp_path, "calculations=optimize")
        assert m["ok"] is False
        assert any("xtb_not_found" in f["error"] for f in m["failures"])

    def test_no_operations_requested(self, tmp_path, stub_xtb):
        # JSON object with everything false should yield 0 ops.
        m = _run_node(
            tmp_path,
            'calculations={"Geometry Optimization": false, "SP Energy": false}',
        )
        assert m["ok"] is False
        assert any("no_operations_requested" in f["error"] for f in m["failures"])

    def test_no_upstream_xyz(self, tmp_path, stub_xtb):
        # Build an upstream manifest with NO xyz artifacts.
        up_dir = tmp_path / "upstream"
        out_dir = up_dir / "outputs"
        out_dir.mkdir(parents=True)
        m_up = Manifest.skeleton(step="up", cwd=str(up_dir))
        m_up_path = out_dir / "manifest.json"
        m_up.write(m_up_path)

        call_dir = tmp_path / "call"
        call_dir.mkdir()
        cwd = os.getcwd()
        os.chdir(call_dir)
        try:
            rc = XtbCalc().invoke(
                ["xtb_calc", _pointer_text(m_up_path), "calculations=optimize"]
            )
        finally:
            os.chdir(cwd)
        assert rc == 0
        m = json.loads((call_dir / "outputs" / "manifest.json").read_text())
        assert m["ok"] is False
        assert any("upstream_xyz_error" in f["error"] for f in m["failures"])

    def test_xtb_returns_nonzero(self, tmp_path, monkeypatch):
        import shutil as _shutil

        monkeypatch.setattr(xc.shutil, "which", lambda name: "/fake/bin/xtb" if name == "xtb" else _shutil.which(name))
        monkeypatch.setattr(xc, "run_xtb", _stub_run_xtb_factory(rc=1))

        m = _run_node(tmp_path, "calculations=sp_energy")
        assert m["ok"] is False
        assert any("xtb_op_failed: sp_energy" in f["error"] for f in m["failures"])
        # Operation record should still be present, recording the rc.
        ops = m["artifacts"]["operations"]
        assert len(ops) == 1 and ops[0]["returncode"] == 1

    def test_optimize_no_xtbopt(self, tmp_path, monkeypatch):
        # xtb returns rc=0 but produces no xtbopt.xyz — that's the "embedded
        # but didn't converge" failure mode.
        import shutil as _shutil

        monkeypatch.setattr(xc.shutil, "which", lambda name: "/fake/bin/xtb" if name == "xtb" else _shutil.which(name))
        monkeypatch.setattr(
            xc, "run_xtb", _stub_run_xtb_factory(produce_xtbopt=False)
        )

        m = _run_node(tmp_path, "calculations=optimize")
        assert m["ok"] is False
        assert any("optimize_no_xtbopt" in f["error"] for f in m["failures"])

    def test_bad_pointer_is_soft_fail(self, tmp_path, stub_xtb):
        call_dir = tmp_path / "call"
        call_dir.mkdir()
        cwd = os.getcwd()
        os.chdir(call_dir)
        try:
            rc = XtbCalc().invoke(["xtb_calc", "this-is-not-json", "calculations=optimize"])
        finally:
            os.chdir(cwd)
        assert rc == 0
        m = json.loads((call_dir / "outputs" / "manifest.json").read_text())
        assert m["ok"] is False

    def test_unknown_theory_is_argv_parse_failed(self, tmp_path, stub_xtb):
        m = _run_node(tmp_path, "theory=BOGUS")
        assert m["ok"] is False
        assert any("argv_parse_failed" in f["error"] for f in m["failures"])

    def test_hard_fail_policy_returns_one(self, tmp_path, monkeypatch):
        import shutil as _shutil

        monkeypatch.setattr(xc.shutil, "which", lambda name: None)  # no xtb
        monkeypatch.setattr(xc, "run_xtb", _stub_run_xtb_factory())

        up_manifest_path, _ = _make_upstream(tmp_path)
        call_dir = tmp_path / "call"
        call_dir.mkdir()
        cwd = os.getcwd()
        os.chdir(call_dir)
        try:
            rc = XtbCalc().invoke(
                [
                    "xtb_calc",
                    _pointer_text(up_manifest_path),
                    "calculations=optimize",
                    "fail_policy=hard",
                ]
            )
        finally:
            os.chdir(cwd)
        assert rc == 1


# --------------------------------------------------------------------
# Wiring sanity
# --------------------------------------------------------------------


class TestNodeWiring:
    def test_class_attributes(self):
        # Chain semantics: requires upstream pointer.
        assert XtbCalc.step == "xtb_calc"
        assert XtbCalc.accepts_upstream is True
        assert XtbCalc.requires_upstream is True

    def test_main_factory_returns_callable(self):
        # ``main`` is what pyproject's ``wf-xtb`` console script targets.
        assert callable(xc.main)

    def test_ops_order_canonical(self):
        # If anyone reorders OPS_ORDER, the chain semantics break.
        assert OPS_ORDER == ("optimize", "sp_energy", "sp_gradient", "sp_hessian")
