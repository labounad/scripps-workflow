"""Tests for the orca_goat global-conformer-search node.

ORCA is not available in CI / the test sandbox, so we monkeypatch
:func:`scripps_workflow.nodes.orca_goat.run_orca_goat` (the only place
we shell out to ORCA) and :func:`shutil.which`. The stub creates
plausible output files (``orca.finalensemble.xyz``,
``orca.globalminimum.xyz``, optional ``orca_property.json``) in the run
directory so the collection / multi-xyz-splitting code can be
exercised.

Coverage:

    * Pure helpers: normalize_theory (aliases + pass-through +
      shell-injection rejection), normalize_mode, normalize_solvent,
      goat_simple_input_keyword, build_orca_input,
      parse_goat_ensemble_energies, find_orca_outputs.
    * Happy path: ensemble published, per-conformer files produced,
      energies attached as ``rel_energy_kcal``, best xyz published.
    * Manifest population: artifacts (xyz / xyz_ensemble / conformers /
      files / logs / operations), inputs, environment, upstream.
    * Failure paths: orca missing on PATH, no upstream xyz, orca returns
      nonzero, no ensemble produced, bad config tokens, hard fail
      policy.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripps_workflow.nodes import orca_goat as og
from scripps_workflow.nodes.crest import XyzBlock
from scripps_workflow.nodes.orca_goat import (
    CPCM_SOLVENTS,
    GOAT_MODES,
    OrcaGoat,
    build_orca_input,
    find_orca_outputs,
    goat_simple_input_keyword,
    normalize_mode,
    normalize_solvent,
    normalize_theory,
    parse_goat_ensemble_energies,
)
from scripps_workflow.pointer import Pointer
from scripps_workflow.schema import Manifest


# --------------------------------------------------------------------
# Pure helpers — normalize_theory
# --------------------------------------------------------------------


class TestNormalizeTheory:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            (None, "XTB"),
            ("", "XTB"),
            ("  ", "XTB"),
            ("xtb", "XTB"),
            ("XTB", "XTB"),
            ("xtb2", "XTB"),
            ("gfn2", "XTB"),
            ("GFN2-XTB", "XTB"),
            ("gfn2xtb", "XTB"),
            ("gfn1", "XTB1"),
            ("GFN1-XTB", "XTB1"),
            ("gfn-ff", "GFNFF"),
            ("gfnff", "GFNFF"),
            ("r2scan-3c", "r2SCAN-3c"),
            ("r2SCAN-3c", "r2SCAN-3c"),
            ("r2scan3c", "r2SCAN-3c"),
            ("b97-3c", "B97-3c"),
            ("b973c", "B97-3c"),
            ("PBEh-3c", "PBEh-3c"),
            ("pbeh3c", "PBEh-3c"),
            ("hf-3c", "HF-3c"),
            ("hf3c", "HF-3c"),
        ],
    )
    def test_aliases(self, raw, expected):
        assert normalize_theory(raw) == expected

    def test_passthrough_unknown(self):
        # ORCA accepts a long tail of DFT functionals; normalize_theory
        # should pass anything plausible through verbatim.
        assert normalize_theory("B3LYP D3") == "B3LYP D3"
        assert normalize_theory("PBE0 D4 def2-SVP") == "PBE0 D4 def2-SVP"
        assert normalize_theory("wB97X-D3 def2-TZVP") == "wB97X-D3 def2-TZVP"

    def test_passthrough_strips_whitespace(self):
        assert normalize_theory("  B3LYP D3  ") == "B3LYP D3"

    @pytest.mark.parametrize(
        "metachar",
        ["\n", "\r", "!", "*", "%", "&", "|", ";", "`", "$", "<", ">"],
    )
    def test_shell_injection_metacharacters_rejected(self, metachar):
        with pytest.raises(ValueError, match="forbidden metacharacters"):
            normalize_theory(f"B3LYP{metachar}rm -rf /")


class TestNormalizeMode:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            (None, "regular"),
            ("", "regular"),
            ("  ", "regular"),
            ("default", "regular"),
            ("regular", "regular"),
            ("REGULAR", "regular"),
            ("quick", "quick"),
            ("Quick", "quick"),
            ("explore", "explore"),
            ("accurate", "accurate"),
        ],
    )
    def test_cases(self, raw, expected):
        assert normalize_mode(raw) == expected

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown GOAT mode"):
            normalize_mode("turbo")

    def test_full_set(self):
        for m in GOAT_MODES:
            assert normalize_mode(m) == m


class TestNormalizeSolvent:
    @pytest.mark.parametrize(
        "raw",
        [None, "", "   ", "none", "null", "vacuum", "gas", "gas_phase"],
    )
    def test_vacuum_inputs(self, raw):
        assert normalize_solvent(raw) is None

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("water", "water"),
            ("Water", "water"),
            ("h2o", "water"),
            ("H2O", "water"),
            ("diethylether", "ether"),
            ("dichloromethane", "ch2cl2"),
            ("ch2cl2", "ch2cl2"),
            ("DMSO", "dmso"),
            ("toluene", "toluene"),
            ("methanol", "methanol"),
        ],
    )
    def test_aliases(self, raw, expected):
        assert normalize_solvent(raw) == expected

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown CPCM solvent"):
            normalize_solvent("supercritical_co2")

    def test_full_set_round_trip(self):
        # Every canonical solvent should normalize to itself or to a
        # canonical alias — never raise.
        for s in CPCM_SOLVENTS:
            normalize_solvent(s)


# --------------------------------------------------------------------
# Pure helpers — simple-input keyword
# --------------------------------------------------------------------


class TestGoatSimpleInputKeyword:
    def test_regular(self):
        assert goat_simple_input_keyword("regular") == "GOAT"

    @pytest.mark.parametrize(
        "mode, expected",
        [
            ("quick", "GOAT-QUICK"),
            ("explore", "GOAT-EXPLORE"),
            ("accurate", "GOAT-ACCURATE"),
        ],
    )
    def test_speed_modes(self, mode, expected):
        assert goat_simple_input_keyword(mode) == expected

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown GOAT mode"):
            goat_simple_input_keyword("turbo")


# --------------------------------------------------------------------
# Pure helpers — build_orca_input
# --------------------------------------------------------------------


class TestBuildOrcaInput:
    def _base(self, **kwargs):
        defaults = dict(
            theory="XTB",
            mode="regular",
            charge=0,
            multiplicity=1,
            solvent=None,
            ewin_kcal=6.0,
            max_conformers=0,
            threads=1,
            maxcore_mb=2000,
            xyz_filename="input.xyz",
        )
        defaults.update(kwargs)
        return build_orca_input(**defaults)

    def test_default_neutral_singlet(self):
        text = self._base()
        # Simple-input line carries GOAT keyword + theory.
        assert "! GOAT XTB" in text
        # No CPCM line for vacuum.
        assert "CPCM" not in text
        # Pal + maxcore always emitted.
        assert "%pal nprocs 1 end" in text
        assert "%maxcore 2000" in text
        # GOAT block carries maxEn.
        assert "%goat" in text
        assert "maxEn 6" in text
        # No maxConfs line when max_conformers=0.
        assert "maxConfs" not in text
        # Coordinate spec: charge / multiplicity / xyz filename.
        assert "* xyzfile 0 1 input.xyz" in text

    def test_charge_and_multiplicity_emit(self):
        text = self._base(charge=-1, multiplicity=3)
        assert "* xyzfile -1 3 input.xyz" in text

    def test_solvent_emits_cpcm_line(self):
        text = self._base(solvent="water")
        assert "! CPCM(water)" in text

    def test_speed_mode_emits_keyword(self):
        text_quick = self._base(mode="quick")
        assert "! GOAT-QUICK XTB" in text_quick

        text_explore = self._base(mode="explore")
        assert "! GOAT-EXPLORE XTB" in text_explore

    def test_max_conformers_emits_maxconfs(self):
        text = self._base(max_conformers=50)
        assert "maxConfs 50" in text

    def test_ewin_formatted(self):
        text = self._base(ewin_kcal=5.5)
        assert "maxEn 5.5" in text

    def test_threads_passed(self):
        text = self._base(threads=8)
        assert "%pal nprocs 8 end" in text

    def test_maxcore_passed(self):
        text = self._base(maxcore_mb=4096)
        assert "%maxcore 4096" in text

    def test_theory_passes_through_verbatim(self):
        # Composite + DFT functional strings should appear unchanged.
        text = self._base(theory="r2SCAN-3c")
        assert "! GOAT r2SCAN-3c" in text

        text2 = self._base(theory="B3LYP D3")
        assert "! GOAT B3LYP D3" in text2


# --------------------------------------------------------------------
# Pure helpers — parse_goat_ensemble_energies
# --------------------------------------------------------------------


def _make_block(comment: str, nat: int = 3) -> XyzBlock:
    lines = [str(nat), comment, "C 0 0 0", "O 0 0 1.4", "H 0 0 -1.0"][: nat + 2]
    return XyzBlock(nat=nat, comment=comment, lines=lines)


class TestParseGoatEnsembleEnergies:
    def test_explicit_erel_token(self):
        blocks = [
            _make_block("Erel: 0.000 kcal/mol Etot: -123.45 Eh"),
            _make_block("Erel: 1.234 kcal/mol Etot: -123.43 Eh"),
            _make_block("Erel: 4.20 kcal/mol Etot: -123.40 Eh"),
        ]
        assert parse_goat_ensemble_energies(blocks) == [0.0, 1.234, 4.20]

    def test_first_float_with_kcal_unit(self):
        # Common GOAT shape: leading float + kcal/mol unit, no Erel.
        blocks = [
            _make_block("   0.000000000 kcal/mol"),
            _make_block("   2.510000000 kcal/mol"),
        ]
        assert parse_goat_ensemble_energies(blocks) == [0.0, 2.51]

    def test_first_float_no_unit_treated_as_kcal(self):
        # Bare number — assume kcal/mol (the most common GOAT default).
        blocks = [
            _make_block("0.000"),
            _make_block("3.142"),
        ]
        assert parse_goat_ensemble_energies(blocks) == [0.0, 3.142]

    def test_eh_only_returns_none(self):
        # Absolute energy in Hartree without an Erel token — we can't
        # convert to relative without a reference, so return None.
        blocks = [
            _make_block("Energy: -123.456 Eh"),
            _make_block("E= -123.450 Hartree"),
        ]
        assert parse_goat_ensemble_energies(blocks) == [None, None]

    def test_empty_comment_returns_none(self):
        blocks = [_make_block("")]
        assert parse_goat_ensemble_energies(blocks) == [None]

    def test_no_floats_returns_none(self):
        blocks = [_make_block("no numbers here")]
        assert parse_goat_ensemble_energies(blocks) == [None]

    def test_erel_takes_precedence_over_eh(self):
        # If Erel is present, we should pick that even if Eh appears too.
        blocks = [_make_block("Etot: -123.45 Eh    Erel: 0.5 kcal/mol")]
        assert parse_goat_ensemble_energies(blocks) == [0.5]

    def test_negative_relative_energy_ok(self):
        # Some shapes write the lowest as "0.0" and rel can technically
        # be negative if the reference is the median or similar.
        blocks = [_make_block("Erel: -0.123 kcal/mol")]
        assert parse_goat_ensemble_energies(blocks) == [-0.123]


# --------------------------------------------------------------------
# Pure helpers — find_orca_outputs
# --------------------------------------------------------------------


class TestFindOrcaOutputs:
    def test_all_present(self, tmp_path):
        (tmp_path / "orca.finalensemble.xyz").write_text("3\nx\nH 0 0 0\n")
        (tmp_path / "orca.globalminimum.xyz").write_text("3\nx\nH 0 0 0\n")
        (tmp_path / "orca.out").write_text("normal termination\n")
        (tmp_path / "orca.err").write_text("")
        (tmp_path / "orca_property.json").write_text("{}")

        located = find_orca_outputs(tmp_path, "orca")
        assert located["ensemble"] == tmp_path / "orca.finalensemble.xyz"
        assert located["best"] == tmp_path / "orca.globalminimum.xyz"
        assert located["out"] == tmp_path / "orca.out"
        assert located["err"] == tmp_path / "orca.err"
        assert located["property_json"] == tmp_path / "orca_property.json"

    def test_partial_missing_returns_none(self, tmp_path):
        # Only ensemble exists — others should be None.
        (tmp_path / "orca.finalensemble.xyz").write_text("3\nx\nH 0 0 0\n")

        located = find_orca_outputs(tmp_path, "orca")
        assert located["ensemble"] is not None
        assert located["best"] is None
        assert located["out"] is None
        assert located["err"] is None
        assert located["property_json"] is None

    def test_basename_respected(self, tmp_path):
        (tmp_path / "myrun.finalensemble.xyz").write_text("3\nx\nH 0 0 0\n")
        located = find_orca_outputs(tmp_path, "myrun")
        assert located["ensemble"] is not None
        # Different basename — same file should not match.
        located2 = find_orca_outputs(tmp_path, "orca")
        assert located2["ensemble"] is None


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


def _stub_run_orca_goat_factory(
    *,
    n_conformers: int = 3,
    write_best: bool = True,
    write_property_json: bool = True,
    rc: int = 0,
    energy_shape: str = "kcal_explicit",
):
    """Build a stub for ``run_orca_goat`` that fakes the side effects.

    The stub writes ``orca.finalensemble.xyz`` (multi-xyz with
    ``n_conformers`` frames), optionally ``orca.globalminimum.xyz`` and
    ``orca_property.json`` into ``cwd`` (the run dir). ``energy_shape``
    controls the comment line format so tests can exercise different
    parser branches.
    """

    def _comment_for(i: int) -> str:
        rel = 0.5 * (i - 1)
        if energy_shape == "kcal_explicit":
            return f"Erel: {rel:.3f} kcal/mol Etot: -123.4{i} Eh"
        if energy_shape == "first_float_kcal":
            return f"   {rel:.6f} kcal/mol"
        if energy_shape == "bare_float":
            return f"{rel:.3f}"
        if energy_shape == "eh_only":
            return f"Energy: -123.4{i} Eh"
        return f"frame {i}"

    def _stub(cmd, *, cwd, stdout_path, stderr_path):
        cwd.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text("fake orca stdout\nORCA TERMINATED NORMALLY\n")
        stderr_path.write_text("")

        if n_conformers > 0:
            frames = []
            for i in range(1, n_conformers + 1):
                comment = _comment_for(i)
                frames.append(
                    f"3\n{comment}\n"
                    f"C {0.01*i:.3f} 0 0\nO 0 0 1.4\nH 0 0 -1.0"
                )
            (Path(cwd) / "orca.finalensemble.xyz").write_text(
                "\n".join(frames) + "\n"
            )

        if write_best:
            (Path(cwd) / "orca.globalminimum.xyz").write_text(
                "3\nbest geom\nC 0 0 0\nO 0 0 1.41\nH 0 0 -1.0\n"
            )

        if write_property_json:
            (Path(cwd) / "orca_property.json").write_text(
                '{"NormalTermination": true}'
            )

        return rc, 0.05

    return _stub


@pytest.fixture
def stub_orca(monkeypatch):
    """Make ``shutil.which('orca')`` succeed and ``run_orca_goat``
    produce fake outputs."""
    import shutil as _shutil

    def _which(name):
        if name == "orca":
            return "/fake/bin/orca"
        return _shutil.which(name)

    monkeypatch.setattr(og.shutil, "which", _which)
    monkeypatch.setattr(og, "run_orca_goat", _stub_run_orca_goat_factory())
    # Wipe SLURM env so resolve_threads is deterministic.
    for k in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "OMP_NUM_THREADS"):
        monkeypatch.delenv(k, raising=False)


def _run_node(tmp_path: Path, *config_tokens: str, ok_pointer: bool = True) -> dict:
    """Invoke OrcaGoat against a freshly-built upstream.

    Returns the parsed manifest dict. Call directory is
    ``tmp_path/calls/orca`` so the upstream tree is left alone.
    """
    up_manifest_path, _ = _make_upstream(tmp_path)
    pointer_text = _pointer_text(up_manifest_path, ok=ok_pointer)

    call_dir = tmp_path / "calls" / "orca"
    call_dir.mkdir(parents=True)

    cwd = os.getcwd()
    os.chdir(call_dir)
    try:
        rc = OrcaGoat().invoke(["orca_goat", pointer_text, *config_tokens])
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
    def test_writes_expected_artifacts(self, tmp_path, stub_orca):
        m = _run_node(tmp_path)
        assert m["ok"] is True
        assert m["step"] == "orca_goat"

        # input_xyz staged + orca_input written.
        files_labels = {f["label"] for f in m["artifacts"]["files"]}
        assert "input_xyz" in files_labels
        assert "orca_input" in files_labels

        # property json captured.
        assert "orca_property_json" in files_labels

        # Whole multi-xyz published.
        ens = m["artifacts"]["xyz_ensemble"]
        assert len(ens) == 1
        assert ens[0]["label"] == "goat_conformers"
        assert Path(ens[0]["path_abs"]).exists()

        # Per-conformer records: 3 frames in the stub, indexed 1..3.
        confs = m["artifacts"]["conformers"]
        assert [c["index"] for c in confs] == [1, 2, 3]
        for c in confs:
            assert Path(c["path_abs"]).exists()
            assert c["format"] == "xyz"

        # Best xyz published (from globalminimum, not conf_0001).
        xyz = m["artifacts"]["xyz"]
        assert len(xyz) == 1 and xyz[0]["label"] == "best"
        # The fake globalminimum has comment "best geom".
        assert "best geom" in Path(xyz[0]["path_abs"]).read_text()

        # Logs.
        log_labels = {l["label"] for l in m["artifacts"]["logs"]}
        assert log_labels == {"orca_stdout", "orca_stderr"}

        # Operations: single record.
        ops = m["artifacts"]["operations"]
        assert len(ops) == 1
        assert ops[0]["op"] == "orca_goat"
        assert ops[0]["returncode"] == 0
        assert ops[0]["cmd"][0] == "/fake/bin/orca"

    def test_rel_energy_kcal_attached(self, tmp_path, stub_orca):
        m = _run_node(tmp_path)
        confs = m["artifacts"]["conformers"]
        # The stub emits Erel: 0.0, 0.5, 1.0.
        rels = [c.get("rel_energy_kcal") for c in confs]
        assert rels == [0.0, 0.5, 1.0]

    def test_first_float_kcal_shape(self, tmp_path, monkeypatch):
        """Comment with no Erel token, just a leading float + kcal unit."""
        import shutil as _shutil

        def _which(name):
            if name == "orca":
                return "/fake/bin/orca"
            return _shutil.which(name)

        monkeypatch.setattr(og.shutil, "which", _which)
        monkeypatch.setattr(
            og,
            "run_orca_goat",
            _stub_run_orca_goat_factory(energy_shape="first_float_kcal"),
        )
        for k in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "OMP_NUM_THREADS"):
            monkeypatch.delenv(k, raising=False)

        m = _run_node(tmp_path)
        rels = [c.get("rel_energy_kcal") for c in m["artifacts"]["conformers"]]
        assert rels == [0.0, 0.5, 1.0]

    def test_eh_only_yields_no_rel_energy(self, tmp_path, monkeypatch):
        """Comment only has Hartree absolute energy: rel_energy_kcal absent."""
        import shutil as _shutil

        def _which(name):
            if name == "orca":
                return "/fake/bin/orca"
            return _shutil.which(name)

        monkeypatch.setattr(og.shutil, "which", _which)
        monkeypatch.setattr(
            og,
            "run_orca_goat",
            _stub_run_orca_goat_factory(energy_shape="eh_only"),
        )
        for k in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "OMP_NUM_THREADS"):
            monkeypatch.delenv(k, raising=False)

        m = _run_node(tmp_path)
        confs = m["artifacts"]["conformers"]
        # parse_goat_ensemble_energies returns None for Eh-only -> no
        # rel_energy_kcal field on the record.
        for c in confs:
            assert "rel_energy_kcal" not in c

    def test_inputs_block_is_typed(self, tmp_path, stub_orca):
        m = _run_node(
            tmp_path,
            "theory=r2scan-3c",
            "solvent=ch2cl2",
            "charge=-1",
            "unpaired_electrons=1",
            "mode=quick",
            "ewin_kcal=5.5",
            "max_conformers=10",
            "threads=4",
            "maxcore_mb=4096",
        )
        ins = m["inputs"]
        assert ins["theory"] == "r2SCAN-3c"
        assert ins["solvent"] == "ch2cl2"
        assert ins["charge"] == -1
        assert ins["unpaired_electrons"] == 1
        # multiplicity = 2*1 + 1 = 3.
        assert ins["multiplicity"] == 3
        assert ins["mode"] == "quick"
        assert ins["ewin_kcal"] == 5.5
        assert ins["max_conformers"] == 10
        assert ins["threads_requested"] == 4
        assert ins["maxcore_mb"] == 4096

    def test_max_conformers_truncates(self, tmp_path, monkeypatch):
        """Stub produces 5 frames; max_conformers=2 should keep 2."""
        import shutil as _shutil

        def _which(name):
            if name == "orca":
                return "/fake/bin/orca"
            return _shutil.which(name)

        monkeypatch.setattr(og.shutil, "which", _which)
        monkeypatch.setattr(
            og, "run_orca_goat", _stub_run_orca_goat_factory(n_conformers=5)
        )
        for k in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "OMP_NUM_THREADS"):
            monkeypatch.delenv(k, raising=False)

        m = _run_node(tmp_path, "max_conformers=2")
        confs = m["artifacts"]["conformers"]
        assert [c["index"] for c in confs] == [1, 2]

    def test_best_falls_back_to_conf_0001(self, tmp_path, monkeypatch):
        """When globalminimum is missing, ``best`` should be conf_0001 copy."""
        import shutil as _shutil

        def _which(name):
            if name == "orca":
                return "/fake/bin/orca"
            return _shutil.which(name)

        monkeypatch.setattr(og.shutil, "which", _which)
        monkeypatch.setattr(
            og,
            "run_orca_goat",
            _stub_run_orca_goat_factory(n_conformers=2, write_best=False),
        )
        for k in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "OMP_NUM_THREADS"):
            monkeypatch.delenv(k, raising=False)

        m = _run_node(tmp_path)
        xyz = m["artifacts"]["xyz"]
        assert len(xyz) == 1 and xyz[0]["label"] == "best"
        # conf_0001 in the stub has Erel comment for index 1.
        assert "Erel: 0.000" in Path(xyz[0]["path_abs"]).read_text()

    def test_environment_records_orca_path(self, tmp_path, stub_orca):
        m = _run_node(tmp_path)
        env = m["environment"]
        assert env["orca"] == "/fake/bin/orca"
        # SLURM env scrubbed -> auto fallback.
        assert env["threads"] == 1

    def test_upstream_block_filled(self, tmp_path, stub_orca):
        m = _run_node(tmp_path)
        up = m["upstream"]
        assert up["pointer_schema"] == "wf.pointer.v1"
        assert up["ok"] is True
        assert up["manifest_path"].endswith("manifest.json")

    def test_input_file_contents(self, tmp_path, stub_orca):
        """The orca_input artifact should contain the GOAT directives."""
        m = _run_node(tmp_path, "theory=xtb", "ewin_kcal=4.0")
        inp_artifact = next(
            f for f in m["artifacts"]["files"] if f["label"] == "orca_input"
        )
        text = Path(inp_artifact["path_abs"]).read_text()
        assert "! GOAT XTB" in text
        assert "%goat" in text
        assert "maxEn 4" in text
        assert "* xyzfile 0 1 input.xyz" in text


# --------------------------------------------------------------------
# Failure paths
# --------------------------------------------------------------------


class TestFailures:
    def test_orca_not_on_path(self, tmp_path, monkeypatch):
        monkeypatch.setattr(og.shutil, "which", lambda name: None)
        monkeypatch.setattr(og, "run_orca_goat", _stub_run_orca_goat_factory())

        m = _run_node(tmp_path)
        assert m["ok"] is False
        assert any(
            "orca_not_found_on_PATH" in f["error"] for f in m["failures"]
        )
        # We never invoked run_orca_goat, so no operations record.
        assert m["artifacts"].get("operations") in (None, [])

    def test_no_upstream_xyz(self, tmp_path, stub_orca):
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
            rc = OrcaGoat().invoke(["orca_goat", _pointer_text(m_up_path)])
        finally:
            os.chdir(cwd)
        assert rc == 0
        m = json.loads((call_dir / "outputs" / "manifest.json").read_text())
        assert m["ok"] is False
        assert any("upstream_xyz_error" in f["error"] for f in m["failures"])

    def test_orca_returns_nonzero(self, tmp_path, monkeypatch):
        import shutil as _shutil

        def _which(name):
            if name == "orca":
                return "/fake/bin/orca"
            return _shutil.which(name)

        monkeypatch.setattr(og.shutil, "which", _which)
        # Even with rc=1 we still emit the ensemble file (so the test
        # exercises both the failure record AND the partial-collection
        # path).
        monkeypatch.setattr(
            og,
            "run_orca_goat",
            _stub_run_orca_goat_factory(rc=1, n_conformers=2),
        )
        for k in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "OMP_NUM_THREADS"):
            monkeypatch.delenv(k, raising=False)

        m = _run_node(tmp_path)
        assert m["ok"] is False
        assert any(
            "orca_failed_returncode_1" in f["error"] for f in m["failures"]
        )
        ops = m["artifacts"]["operations"]
        assert len(ops) == 1 and ops[0]["returncode"] == 1
        # We still collected what ORCA wrote.
        assert len(m["artifacts"]["conformers"]) == 2

    def test_no_ensemble_produced(self, tmp_path, monkeypatch):
        # rc=0 but no finalensemble.xyz — must mark ok=false.
        import shutil as _shutil

        def _which(name):
            if name == "orca":
                return "/fake/bin/orca"
            return _shutil.which(name)

        monkeypatch.setattr(og.shutil, "which", _which)
        monkeypatch.setattr(
            og,
            "run_orca_goat",
            _stub_run_orca_goat_factory(
                n_conformers=0,
                write_best=False,
                write_property_json=False,
            ),
        )
        for k in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "OMP_NUM_THREADS"):
            monkeypatch.delenv(k, raising=False)

        m = _run_node(tmp_path)
        assert m["ok"] is False
        assert any(
            "no_best_xyz_artifact_produced" in f["error"]
            for f in m["failures"]
        )
        assert m["artifacts"].get("conformers", []) == []
        assert m["artifacts"].get("xyz_ensemble", []) == []

    def test_bad_pointer_is_soft_fail(self, tmp_path, stub_orca):
        call_dir = tmp_path / "call"
        call_dir.mkdir()
        cwd = os.getcwd()
        os.chdir(call_dir)
        try:
            rc = OrcaGoat().invoke(["orca_goat", "this-is-not-json"])
        finally:
            os.chdir(cwd)
        assert rc == 0
        m = json.loads((call_dir / "outputs" / "manifest.json").read_text())
        assert m["ok"] is False

    def test_unknown_mode_is_argv_parse_failed(self, tmp_path, stub_orca):
        m = _run_node(tmp_path, "mode=turbo")
        assert m["ok"] is False
        assert any("argv_parse_failed" in f["error"] for f in m["failures"])

    def test_unknown_solvent_is_argv_parse_failed(self, tmp_path, stub_orca):
        m = _run_node(tmp_path, "solvent=supercritical_co2")
        assert m["ok"] is False
        assert any("argv_parse_failed" in f["error"] for f in m["failures"])

    def test_theory_with_metacharacter_is_argv_parse_failed(
        self, tmp_path, stub_orca
    ):
        # Shell-injection guard: a `!` in the theory string would corrupt
        # the ORCA simple-input line.
        m = _run_node(tmp_path, "theory=B3LYP!evil")
        assert m["ok"] is False
        assert any("argv_parse_failed" in f["error"] for f in m["failures"])

    def test_zero_ewin_is_argv_parse_failed(self, tmp_path, stub_orca):
        m = _run_node(tmp_path, "ewin_kcal=0")
        assert m["ok"] is False
        assert any("argv_parse_failed" in f["error"] for f in m["failures"])

    def test_negative_ewin_is_argv_parse_failed(self, tmp_path, stub_orca):
        m = _run_node(tmp_path, "ewin_kcal=-1.0")
        assert m["ok"] is False
        assert any("argv_parse_failed" in f["error"] for f in m["failures"])

    def test_negative_max_conformers_is_argv_parse_failed(
        self, tmp_path, stub_orca
    ):
        m = _run_node(tmp_path, "max_conformers=-3")
        assert m["ok"] is False
        assert any("argv_parse_failed" in f["error"] for f in m["failures"])

    def test_low_maxcore_is_argv_parse_failed(self, tmp_path, stub_orca):
        # ORCA needs at least ~100 MB to do anything meaningful.
        m = _run_node(tmp_path, "maxcore_mb=50")
        assert m["ok"] is False
        assert any("argv_parse_failed" in f["error"] for f in m["failures"])

    def test_hard_fail_policy_returns_one(self, tmp_path, monkeypatch):
        # No orca on PATH -> ok=false. With fail_policy=hard, exit 1.
        monkeypatch.setattr(og.shutil, "which", lambda name: None)
        monkeypatch.setattr(og, "run_orca_goat", _stub_run_orca_goat_factory())

        up_manifest_path, _ = _make_upstream(tmp_path)
        call_dir = tmp_path / "call"
        call_dir.mkdir()
        cwd = os.getcwd()
        os.chdir(call_dir)
        try:
            rc = OrcaGoat().invoke(
                [
                    "orca_goat",
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
        assert OrcaGoat.step == "orca_goat"
        assert OrcaGoat.accepts_upstream is True
        assert OrcaGoat.requires_upstream is True

    def test_main_factory_returns_callable(self):
        # ``main`` is what pyproject's ``wf-orca-goat`` console script targets.
        assert callable(og.main)

    def test_alias_table_targets_are_stable(self):
        """Every canonical target in the alias table should round-trip
        through normalize_theory. Catches typos like an alias target
        that itself fails the alias lookup."""
        for canonical in set(og._THEORY_ALIASES.values()):
            # If the canonical's lowercased form is also a key (which
            # is the case for e.g. "XTB" -> "xtb" -> "XTB"), the alias
            # path returns it as-is. Otherwise pass-through preserves it.
            assert normalize_theory(canonical) == canonical
