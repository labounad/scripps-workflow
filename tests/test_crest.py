"""Tests for the crest conformer-search node.

crest is not available in CI / the test sandbox, so we monkeypatch
:func:`scripps_workflow.nodes.crest.run_crest` (the only place we shell
out to crest) and :func:`shutil.which`. The stub creates plausible
output files (``crest_conformers.xyz``, ``crest_rotamers.xyz``,
``crest_best.xyz``, ``crest.energies``) in the run directory so the
collection / multi-xyz-splitting code can be exercised.

Coverage:

    * Pure helpers: normalize_theory, normalize_mode, crest_theory_flag,
      crest_mode_flag, build_crest_cmd, split_multixyz,
      parse_crest_energies, write_xyz_block.
    * Happy path: ensemble published, per-conformer files produced,
      energies attached as ``rel_energy_kcal``, best xyz published.
    * Manifest population: artifacts (xyz / xyz_ensemble / conformers /
      files / logs / operations), inputs, environment, upstream.
    * Failure paths: crest missing, xtb missing, no upstream xyz, crest
      returns nonzero, ensemble not produced, bad config tokens, hard
      fail policy.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripps_workflow.nodes import crest as cr
from scripps_workflow.nodes.crest import (
    CREST_MODES,
    CREST_THEORIES,
    CrestConformerSearch,
    XyzBlock,
    build_crest_cmd,
    crest_mode_flag,
    crest_theory_flag,
    normalize_mode,
    normalize_solvent,
    normalize_theory,
    parse_crest_energies,
    split_multixyz,
    write_xyz_block,
)
from scripps_workflow.pointer import Pointer
from scripps_workflow.schema import Manifest


# --------------------------------------------------------------------
# Pure helpers — theory / mode normalization
# --------------------------------------------------------------------


class TestNormalizeTheory:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("GFN2-XTB", "GFN2-XTB"),
            ("gfn2-xtb", "GFN2-XTB"),
            ("GFN2", "GFN2-XTB"),
            ("gfn2xtb", "GFN2-XTB"),
            ("2", "GFN2-XTB"),
            ("", "GFN2-XTB"),
            (None, "GFN2-XTB"),
            ("GFN1", "GFN1-XTB"),
            ("gfn1-xtb", "GFN1-XTB"),
            ("1", "GFN1-XTB"),
            ("GFN-FF", "GFN-FF"),
            ("gff", "GFN-FF"),
            ("gfnff", "GFN-FF"),
            ("GFN-FF (gff)", "GFN-FF"),
            ("GFN2//GFN-FF", "GFN2//GFN-FF"),
            ("gfn2//gfnff", "GFN2//GFN-FF"),
            ("gfn2/gfnff", "GFN2//GFN-FF"),
        ],
    )
    def test_aliases(self, raw, expected):
        assert normalize_theory(raw) == expected

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown theory"):
            normalize_theory("DFT-D4")

    def test_canonical_set(self):
        # All canonical names round-trip through their alias table.
        for t in CREST_THEORIES:
            assert normalize_theory(t) == t


class TestNormalizeMode:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            (None, "standard"),
            ("", "standard"),
            ("default", "standard"),
            ("standard", "standard"),
            ("STANDARD", "standard"),
            ("quick", "quick"),
            ("Quick", "quick"),
            ("squick", "squick"),
            ("mquick", "mquick"),
        ],
    )
    def test_cases(self, raw, expected):
        assert normalize_mode(raw) == expected

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown CREST mode"):
            normalize_mode("ultraquick")

    def test_full_set(self):
        for m in CREST_MODES:
            assert normalize_mode(m) == m


# --------------------------------------------------------------------
# Pure helpers — flag emission
# --------------------------------------------------------------------


class TestCrestTheoryFlag:
    def test_gfn2_emits_no_flag(self):
        # GFN2-XTB is crest's default; absence of a flag is canonical.
        assert crest_theory_flag("GFN2-XTB") is None

    def test_gfn1_flag(self):
        assert crest_theory_flag("GFN1-XTB") == "--gfn1"

    def test_gfnff_flag(self):
        assert crest_theory_flag("GFN-FF") == "--gfnff"

    def test_composite_flag(self):
        assert crest_theory_flag("GFN2//GFN-FF") == "--gfn2//gfnff"

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            crest_theory_flag("DFT")


class TestCrestModeFlag:
    def test_standard_emits_no_flag(self):
        assert crest_mode_flag("standard") is None

    @pytest.mark.parametrize("mode", ["quick", "squick", "mquick"])
    def test_speed_modes(self, mode):
        assert crest_mode_flag(mode) == f"--{mode}"

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            crest_mode_flag("turbo")


# --------------------------------------------------------------------
# Pure helpers — command construction
# --------------------------------------------------------------------


class TestBuildCrestCmd:
    def _base(self, **kwargs):
        defaults = dict(
            crest_exe="/bin/crest",
            input_xyz_name="input.xyz",
            theory="GFN2-XTB",
            charge=0,
            uhf=0,
            solvent=None,
            mode="standard",
            ewin_kcal=10.0,
            threads=1,
        )
        defaults.update(kwargs)
        return build_crest_cmd(**defaults)

    def test_default_neutral_singlet(self):
        cmd = self._base()
        # Anchor: exe and input come first.
        assert cmd[0] == "/bin/crest"
        assert cmd[1] == "input.xyz"
        # GFN2-XTB is implicit, charge 0 is implicit, no solvent, no mode flag.
        assert "--gfn1" not in cmd and "--gfnff" not in cmd
        assert "--chrg" not in cmd
        assert "--uhf" not in cmd
        assert "--alpb" not in cmd
        assert "--quick" not in cmd
        # ewin and -T are always present.
        assert "--ewin" in cmd
        assert cmd[cmd.index("--ewin") + 1] == "10"
        assert "--T" in cmd and cmd[cmd.index("--T") + 1] == "1"

    def test_charge_uhf_solvent_emit_when_set(self):
        cmd = self._base(charge=-1, uhf=1, solvent="water")
        assert "--chrg" in cmd and cmd[cmd.index("--chrg") + 1] == "-1"
        assert "--uhf" in cmd and cmd[cmd.index("--uhf") + 1] == "1"
        assert "--alpb" in cmd and cmd[cmd.index("--alpb") + 1] == "water"

    def test_theory_flag_appears(self):
        cmd_ff = self._base(theory="GFN-FF")
        assert "--gfnff" in cmd_ff

        cmd_1 = self._base(theory="GFN1-XTB")
        assert "--gfn1" in cmd_1

        cmd_comp = self._base(theory="GFN2//GFN-FF")
        assert "--gfn2//gfnff" in cmd_comp

    def test_speed_mode_flag_appears(self):
        cmd = self._base(mode="quick")
        assert "--quick" in cmd

    def test_ewin_formatted_as_float(self):
        cmd = self._base(ewin_kcal=5.5)
        assert cmd[cmd.index("--ewin") + 1] == "5.5"

    def test_threads_passed(self):
        cmd = self._base(threads=8)
        assert cmd[cmd.index("--T") + 1] == "8"


# --------------------------------------------------------------------
# Pure helpers — multi-xyz splitting / energies
# --------------------------------------------------------------------


_TWO_FRAME_MULTIXYZ = (
    "3\n"
    "frame 1\n"
    "C 0 0 0\n"
    "O 0 0 1.4\n"
    "H 0 0 -1.0\n"
    "3\n"
    "frame 2\n"
    "C 0.1 0 0\n"
    "O 0 0 1.42\n"
    "H 0 0 -0.99\n"
)


class TestSplitMultixyz:
    def test_two_frames(self):
        blocks = split_multixyz(_TWO_FRAME_MULTIXYZ)
        assert len(blocks) == 2
        assert all(b.nat == 3 for b in blocks)
        assert blocks[0].comment == "frame 1"
        assert blocks[1].comment == "frame 2"
        # Each block has nat+2 lines (header + comment + atoms).
        assert len(blocks[0].lines) == 5

    def test_truncated_tail_is_tolerated(self):
        # Crest occasionally writes a partial trailing block when killed.
        text = _TWO_FRAME_MULTIXYZ + "3\nframe 3\nC 0 0 0\n"  # only 1 of 3 atoms
        blocks = split_multixyz(text)
        # We get the two complete frames; the partial third is dropped.
        assert len(blocks) == 2

    def test_blank_separator_lines_ok(self):
        text = _TWO_FRAME_MULTIXYZ.replace("3\nframe 2\n", "\n3\nframe 2\n")
        assert len(split_multixyz(text)) == 2

    def test_empty_input(self):
        assert split_multixyz("") == []

    def test_garbage_first_line(self):
        # No leading int -> nothing parseable.
        assert split_multixyz("not a number\n") == []


class TestWriteXyzBlock:
    def test_round_trip(self, tmp_path):
        blocks = split_multixyz(_TWO_FRAME_MULTIXYZ)
        out = tmp_path / "single.xyz"
        write_xyz_block(out, blocks[0])
        # The written file should re-parse to a single-frame ensemble.
        round_trip = split_multixyz(out.read_text())
        assert len(round_trip) == 1
        assert round_trip[0].nat == 3
        assert round_trip[0].comment == "frame 1"


class TestParseCrestEnergies:
    def test_last_float_wins(self, tmp_path):
        p = tmp_path / "crest.energies"
        # Realistic shape: index, abs_energy, rel_energy. We want the last.
        p.write_text("1   -1.234  0.000\n2   -1.230  2.51\n3   -1.225  4.12\n")
        e = parse_crest_energies(p)
        assert e == [0.0, 2.51, 4.12]

    def test_blank_lines_skipped(self, tmp_path):
        p = tmp_path / "e"
        p.write_text("\n1 -1.234 0.0\n\n2 -1.230 2.51\n")
        e = parse_crest_energies(p)
        assert e == [0.0, 2.51]

    def test_unparseable_line_yields_none(self, tmp_path):
        p = tmp_path / "e"
        p.write_text("1 -1.234 0.0\nbogus garbage line\n2 -1.230 2.51\n")
        e = parse_crest_energies(p)
        assert e == [0.0, None, 2.51]

    def test_csv_tokens_handled(self, tmp_path):
        # Some legacy crest builds emit comma-separated values.
        p = tmp_path / "e"
        p.write_text("1, -1.234, 0.0\n2, -1.230, 2.51\n")
        e = parse_crest_energies(p)
        assert e == [0.0, 2.51]


# --------------------------------------------------------------------
# Imported helper sanity (cross-module reuse)
# --------------------------------------------------------------------


class TestNormalizeSolventReuse:
    """``normalize_solvent`` is imported from xtb_calc — confirm it works
    through the crest module's namespace so a future refactor that
    silently breaks the import is loud."""

    def test_water(self):
        assert normalize_solvent("water") == "water"

    def test_vacuum(self):
        assert normalize_solvent("vacuum") is None


# --------------------------------------------------------------------
# End-to-end node fixtures + stub
# --------------------------------------------------------------------


def _make_upstream(tmp_path: Path, xyz_text: str | None = None) -> tuple[Path, Path]:
    """Create an upstream outputs/ tree with manifest + xyz."""
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
            "smiles": "CCO",
        }
    ]
    m_path = out_dir / "manifest.json"
    m.write(m_path)
    return m_path, xyz_path


def _pointer_text(manifest_path: Path, ok: bool = True) -> str:
    return Pointer.of(ok=ok, manifest_path=manifest_path).to_json_line()


def _stub_run_crest_factory(
    *,
    n_conformers: int = 3,
    write_rotamers: bool = True,
    write_best: bool = True,
    write_energies: bool = True,
    rc: int = 0,
):
    """Build a stub for ``run_crest`` that fakes the side effects crest produces.

    The stub writes a ``crest_conformers.xyz`` multi-xyz with
    ``n_conformers`` frames into ``cwd`` (the staged run dir), plus
    optionally rotamers / best / energies. Setting ``n_conformers=0``
    skips the conformers file entirely (failure-path stub).
    """

    def _stub(cmd, *, cwd, stdout_path, stderr_path):
        cwd.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text("fake crest stdout\nnormal termination of crest\n")
        stderr_path.write_text("")

        if n_conformers > 0:
            frames = []
            for i in range(1, n_conformers + 1):
                frames.append(
                    f"3\nconf {i}\nC {0.01*i:.3f} 0 0\nO 0 0 1.4\nH 0 0 -1.0"
                )
            (Path(cwd) / "crest_conformers.xyz").write_text("\n".join(frames) + "\n")

        if write_rotamers:
            (Path(cwd) / "crest_rotamers.xyz").write_text(
                "3\nrot1\nC 0 0 0\nO 0 0 1.4\nH 0 0 -1.0\n"
            )
        if write_best:
            (Path(cwd) / "crest_best.xyz").write_text(
                "3\nbest\nC 0 0 0\nO 0 0 1.41\nH 0 0 -1.0\n"
            )
        if write_energies:
            lines = []
            for i in range(1, n_conformers + 1):
                # Realistic shape: index, abs_energy, rel_energy_kcal.
                lines.append(f"{i} -1.234 {0.5*(i-1):.3f}")
            (Path(cwd) / "crest.energies").write_text("\n".join(lines) + "\n")

        return rc, 0.05

    return _stub


@pytest.fixture
def stub_crest(monkeypatch):
    """Make ``shutil.which('crest'/'xtb')`` succeed and ``run_crest``
    produce fake outputs."""
    import shutil as _shutil

    def _which(name):
        if name == "crest":
            return "/fake/bin/crest"
        if name == "xtb":
            return "/fake/bin/xtb"
        return _shutil.which(name)

    monkeypatch.setattr(cr.shutil, "which", _which)
    monkeypatch.setattr(cr, "run_crest", _stub_run_crest_factory())
    # Wipe SLURM env so resolve_threads is deterministic.
    for k in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "OMP_NUM_THREADS"):
        monkeypatch.delenv(k, raising=False)


def _run_node(tmp_path: Path, *config_tokens: str, ok_pointer: bool = True) -> dict:
    """Invoke CrestConformerSearch against a freshly-built upstream.

    Returns the parsed manifest dict. Call directory is
    ``tmp_path/calls/crest`` so the upstream tree is left alone.
    """
    up_manifest_path, _ = _make_upstream(tmp_path)
    pointer_text = _pointer_text(up_manifest_path, ok=ok_pointer)

    call_dir = tmp_path / "calls" / "crest"
    call_dir.mkdir(parents=True)

    cwd = os.getcwd()
    os.chdir(call_dir)
    try:
        rc = CrestConformerSearch().invoke(["crest", pointer_text, *config_tokens])
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
    def test_writes_expected_artifacts(self, tmp_path, stub_crest):
        m = _run_node(tmp_path)
        assert m["ok"] is True
        assert m["step"] == "crest"

        # input_xyz staged.
        assert any(
            f.get("label") == "input_xyz" for f in m["artifacts"]["files"]
        )

        # Whole multi-xyz published.
        ens = m["artifacts"]["xyz_ensemble"]
        assert len(ens) == 1
        assert ens[0]["label"] == "crest_conformers"
        assert Path(ens[0]["path_abs"]).exists()

        # Per-conformer records: 3 frames in the stub, indexed 1..3.
        confs = m["artifacts"]["conformers"]
        assert [c["index"] for c in confs] == [1, 2, 3]
        for c in confs:
            assert Path(c["path_abs"]).exists()
            assert c["format"] == "xyz"

        # Best xyz published (from crest_best.xyz, not conf_0001).
        xyz = m["artifacts"]["xyz"]
        assert len(xyz) == 1 and xyz[0]["label"] == "best"
        # The fake crest_best.xyz has comment "best".
        assert "best" in Path(xyz[0]["path_abs"]).read_text()

        # Logs.
        labels = {l["label"] for l in m["artifacts"]["logs"]}
        assert labels == {"crest_stdout", "crest_stderr"}

        # Operations: single record.
        ops = m["artifacts"]["operations"]
        assert len(ops) == 1
        assert ops[0]["op"] == "crest"
        assert ops[0]["returncode"] == 0
        assert ops[0]["cmd"][0] == "/fake/bin/crest"

    def test_rel_energy_kcal_attached(self, tmp_path, stub_crest):
        m = _run_node(tmp_path)
        confs = m["artifacts"]["conformers"]
        # The fake crest.energies emits 0.0, 0.5, 1.0 (rel).
        rels = [c.get("rel_energy_kcal") for c in confs]
        assert rels == [0.0, 0.5, 1.0]

    def test_rotamers_and_energies_in_files(self, tmp_path, stub_crest):
        m = _run_node(tmp_path)
        labels = {f["label"] for f in m["artifacts"]["files"]}
        # input_xyz staged, plus the two informational outputs.
        assert "crest_rotamers" in labels
        assert "crest_energies" in labels

    def test_inputs_block_is_typed(self, tmp_path, stub_crest):
        m = _run_node(
            tmp_path,
            "theory=gfn-ff",
            "solvent=ch2cl2",
            "charge=-1",
            "unpaired_electrons=1",
            "mode=quick",
            "ewin_kcal=5.5",
            "max_conformers=2",
            "threads=4",
        )
        ins = m["inputs"]
        assert ins["theory"] == "GFN-FF"
        assert ins["solvent"] == "ch2cl2"
        assert ins["charge"] == -1
        assert ins["unpaired_electrons"] == 1
        assert ins["mode"] == "quick"
        assert ins["ewin_kcal"] == 5.5
        assert ins["max_conformers"] == 2
        assert ins["threads_requested"] == 4

    def test_max_conformers_truncates(self, tmp_path, monkeypatch):
        """Stub produces 5 frames; max_conformers=2 should keep 2."""
        import shutil as _shutil

        def _which(name):
            if name in {"crest", "xtb"}:
                return f"/fake/bin/{name}"
            return _shutil.which(name)

        monkeypatch.setattr(cr.shutil, "which", _which)
        monkeypatch.setattr(
            cr, "run_crest", _stub_run_crest_factory(n_conformers=5)
        )
        for k in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "OMP_NUM_THREADS"):
            monkeypatch.delenv(k, raising=False)

        m = _run_node(tmp_path, "max_conformers=2")
        confs = m["artifacts"]["conformers"]
        assert [c["index"] for c in confs] == [1, 2]

    def test_best_falls_back_to_conf_0001(self, tmp_path, monkeypatch):
        """When crest_best.xyz is missing, ``best`` should be conf_0001 copy."""
        import shutil as _shutil

        def _which(name):
            if name in {"crest", "xtb"}:
                return f"/fake/bin/{name}"
            return _shutil.which(name)

        monkeypatch.setattr(cr.shutil, "which", _which)
        monkeypatch.setattr(
            cr,
            "run_crest",
            _stub_run_crest_factory(n_conformers=2, write_best=False),
        )
        for k in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "OMP_NUM_THREADS"):
            monkeypatch.delenv(k, raising=False)

        m = _run_node(tmp_path)
        xyz = m["artifacts"]["xyz"]
        assert len(xyz) == 1 and xyz[0]["label"] == "best"
        # conf_0001 in the fake stub has comment "conf 1".
        assert "conf 1" in Path(xyz[0]["path_abs"]).read_text()

    def test_environment_records_binary_paths(self, tmp_path, stub_crest):
        m = _run_node(tmp_path)
        env = m["environment"]
        assert env["crest"] == "/fake/bin/crest"
        assert env["xtb"] == "/fake/bin/xtb"
        assert env["threads"] == 1  # SLURM env scrubbed -> auto fallback

    def test_upstream_block_filled(self, tmp_path, stub_crest):
        m = _run_node(tmp_path)
        up = m["upstream"]
        assert up["pointer_schema"] == "wf.pointer.v1"
        assert up["ok"] is True
        assert up["manifest_path"].endswith("manifest.json")


# --------------------------------------------------------------------
# Failure paths
# --------------------------------------------------------------------


class TestFailures:
    def test_crest_not_on_path(self, tmp_path, monkeypatch):
        # which returns None for crest specifically; xtb is fine.
        import shutil as _shutil

        def _which(name):
            if name == "crest":
                return None
            if name == "xtb":
                return "/fake/bin/xtb"
            return _shutil.which(name)

        monkeypatch.setattr(cr.shutil, "which", _which)
        monkeypatch.setattr(cr, "run_crest", _stub_run_crest_factory())

        m = _run_node(tmp_path)
        assert m["ok"] is False
        assert any("crest_not_found" in f["error"] for f in m["failures"])

    def test_xtb_not_on_path(self, tmp_path, monkeypatch):
        # crest is found, xtb missing — crest can't actually run without xtb,
        # so we fail fast before launching crest.
        import shutil as _shutil

        def _which(name):
            if name == "crest":
                return "/fake/bin/crest"
            if name == "xtb":
                return None
            return _shutil.which(name)

        monkeypatch.setattr(cr.shutil, "which", _which)
        monkeypatch.setattr(cr, "run_crest", _stub_run_crest_factory())

        m = _run_node(tmp_path)
        assert m["ok"] is False
        assert any("xtb_not_found" in f["error"] for f in m["failures"])

    def test_no_upstream_xyz(self, tmp_path, stub_crest):
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
            rc = CrestConformerSearch().invoke(
                ["crest", _pointer_text(m_up_path)]
            )
        finally:
            os.chdir(cwd)
        assert rc == 0
        m = json.loads((call_dir / "outputs" / "manifest.json").read_text())
        assert m["ok"] is False
        assert any("upstream_xyz_error" in f["error"] for f in m["failures"])

    def test_crest_returns_nonzero(self, tmp_path, monkeypatch):
        import shutil as _shutil

        def _which(name):
            if name in {"crest", "xtb"}:
                return f"/fake/bin/{name}"
            return _shutil.which(name)

        monkeypatch.setattr(cr.shutil, "which", _which)
        # Even with rc=1 we still emit the conformers file (so the test
        # exercises both the failure record AND the partial-collection
        # path).
        monkeypatch.setattr(
            cr, "run_crest", _stub_run_crest_factory(rc=1, n_conformers=2)
        )
        for k in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "OMP_NUM_THREADS"):
            monkeypatch.delenv(k, raising=False)

        m = _run_node(tmp_path)
        assert m["ok"] is False
        assert any(
            "crest_failed_returncode_1" in f["error"] for f in m["failures"]
        )
        ops = m["artifacts"]["operations"]
        assert len(ops) == 1 and ops[0]["returncode"] == 1
        # We still collected what crest did write.
        assert len(m["artifacts"]["conformers"]) == 2

    def test_no_conformers_produced(self, tmp_path, monkeypatch):
        # rc=0 but crest produces no crest_conformers.xyz — must mark ok=false.
        import shutil as _shutil

        def _which(name):
            if name in {"crest", "xtb"}:
                return f"/fake/bin/{name}"
            return _shutil.which(name)

        monkeypatch.setattr(cr.shutil, "which", _which)
        monkeypatch.setattr(
            cr,
            "run_crest",
            _stub_run_crest_factory(
                n_conformers=0, write_best=False, write_rotamers=False, write_energies=False
            ),
        )
        for k in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "OMP_NUM_THREADS"):
            monkeypatch.delenv(k, raising=False)

        m = _run_node(tmp_path)
        assert m["ok"] is False
        assert any(
            "no_best_xyz_artifact_produced" in f["error"] for f in m["failures"]
        )
        assert m["artifacts"]["conformers"] == []
        assert m["artifacts"]["xyz_ensemble"] == []

    def test_bad_pointer_is_soft_fail(self, tmp_path, stub_crest):
        call_dir = tmp_path / "call"
        call_dir.mkdir()
        cwd = os.getcwd()
        os.chdir(call_dir)
        try:
            rc = CrestConformerSearch().invoke(
                ["crest", "this-is-not-json"]
            )
        finally:
            os.chdir(cwd)
        assert rc == 0
        m = json.loads((call_dir / "outputs" / "manifest.json").read_text())
        assert m["ok"] is False

    def test_unknown_theory_is_argv_parse_failed(self, tmp_path, stub_crest):
        m = _run_node(tmp_path, "theory=BOGUS")
        assert m["ok"] is False
        assert any("argv_parse_failed" in f["error"] for f in m["failures"])

    def test_unknown_mode_is_argv_parse_failed(self, tmp_path, stub_crest):
        m = _run_node(tmp_path, "mode=ultraquick")
        assert m["ok"] is False
        assert any("argv_parse_failed" in f["error"] for f in m["failures"])

    def test_zero_ewin_is_argv_parse_failed(self, tmp_path, stub_crest):
        m = _run_node(tmp_path, "ewin_kcal=0")
        assert m["ok"] is False
        assert any("argv_parse_failed" in f["error"] for f in m["failures"])

    def test_negative_max_conformers_is_argv_parse_failed(self, tmp_path, stub_crest):
        m = _run_node(tmp_path, "max_conformers=-3")
        assert m["ok"] is False
        assert any("argv_parse_failed" in f["error"] for f in m["failures"])

    def test_hard_fail_policy_returns_one(self, tmp_path, monkeypatch):
        # No crest on PATH -> ok=false. With fail_policy=hard, exit 1.
        monkeypatch.setattr(cr.shutil, "which", lambda name: None)
        monkeypatch.setattr(cr, "run_crest", _stub_run_crest_factory())

        up_manifest_path, _ = _make_upstream(tmp_path)
        call_dir = tmp_path / "call"
        call_dir.mkdir()
        cwd = os.getcwd()
        os.chdir(call_dir)
        try:
            rc = CrestConformerSearch().invoke(
                [
                    "crest",
                    _pointer_text(up_manifest_path),
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
        assert CrestConformerSearch.step == "crest"
        assert CrestConformerSearch.accepts_upstream is True
        assert CrestConformerSearch.requires_upstream is True

    def test_main_factory_returns_callable(self):
        # ``main`` is what pyproject's ``wf-crest`` console script targets.
        assert callable(cr.main)

    def test_xyz_block_is_dataclass(self):
        # Sanity: tests below depend on field access.
        b = XyzBlock(nat=1, comment="x", lines=["1", "x", "H 0 0 0"])
        assert b.nat == 1 and b.comment == "x"
