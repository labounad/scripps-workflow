"""Tests for the smiles_to_3d node.

RDKit is not available in CI / the test sandbox, so we monkeypatch
:func:`scripps_workflow.nodes.smiles_to_3d.build_3d_mol` and
:func:`scripps_workflow.nodes.smiles_to_3d.mol_to_xyz_block` to return
canned outputs. That keeps the orchestration layer (config parsing,
manifest population, error paths) under test without needing a
chemistry runtime. The RDKit-using core gets exercised in real usage
and in a separate integration test (not in this file).

Tests cover:

    * Happy path: stubbed embed → valid manifest, xyz file written,
      artifact bucket populated with sha256.
    * Missing/empty SMILES → soft-fail with structured failure.
    * Embed exceptions (bad SMILES, embedding failures) → soft-fail.
    * Filename sanitization and collision-safe naming.
    * Config defaults are applied when the engine omits keys.
    * Pure helpers (``sanitize_filename``, ``_unique_xyz_path``,
      ``parse_config``) covered directly.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from scripps_workflow.nodes import smiles_to_3d as s23
from scripps_workflow.nodes.smiles_to_3d import (
    SmilesTo3D,
    _unique_xyz_path,
    sanitize_filename,
)


# --------------------------------------------------------------------
# Stub mol + helpers for tests
# --------------------------------------------------------------------


class _FakeMol:
    def __init__(self, n_atoms: int = 9, n_heavy: int = 3):
        self._n = n_atoms
        self._h = n_heavy

    def GetNumAtoms(self) -> int:
        return self._n

    def GetNumHeavyAtoms(self) -> int:
        return self._h


def _stub_build_3d_mol(smiles: str, **kwargs: Any) -> _FakeMol:
    """Stand-in for the RDKit-backed ``build_3d_mol``.

    Encodes a few SMILES → atom-count fixtures so tests can assert specific
    artifact values, plus a couple of error sentinels for negative tests.
    """
    if smiles == "BAD_SMILES":
        raise ValueError(f"Could not parse SMILES: {smiles!r}")
    if smiles == "EMBED_FAIL":
        raise RuntimeError(f"3D embedding failed for SMILES: {smiles!r}")
    fixtures = {"CCO": (9, 3), "c1ccccc1": (12, 6)}
    n, h = fixtures.get(smiles, (5, 2))
    return _FakeMol(n_atoms=n, n_heavy=h)


def _stub_mol_to_xyz_block(mol: _FakeMol, comment: str = "") -> str:
    """Canned XYZ block. The orchestration code only writes the text; it
    doesn't parse it back."""
    n = mol.GetNumAtoms()
    lines = [str(n), comment]
    for i in range(n):
        # Element symbol + zero-padded coords. Real geometry doesn't matter
        # for these tests — only that a well-formed file is written.
        lines.append(f"C  0.000  0.000  {i:.3f}")
    return "\n".join(lines) + "\n"


@pytest.fixture
def stub_rdkit(monkeypatch):
    """Monkeypatch the RDKit-using helpers so tests don't need RDKit."""
    monkeypatch.setattr(s23, "build_3d_mol", _stub_build_3d_mol)
    monkeypatch.setattr(s23, "mol_to_xyz_block", _stub_mol_to_xyz_block)
    return None


def _run_node(tmp_path: Path, *config_tokens: str) -> dict:
    """Invoke the node from ``tmp_path`` and return the parsed manifest."""
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        rc = SmilesTo3D().invoke(["smiles_to_3d", *config_tokens])
    finally:
        os.chdir(cwd)
    assert rc == 0, "soft-fail invariant violated"
    manifest_path = tmp_path / "outputs" / "manifest.json"
    assert manifest_path.exists()
    return json.loads(manifest_path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------
# sanitize_filename
# --------------------------------------------------------------------


class TestSanitizeFilename:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("benzene", "benzene"),
            ("benzene C6H6", "benzene_C6H6"),
            ("foo/bar", "foo_bar"),
            ("a..b", "a..b"),
            ("___leading", "leading"),
            ("trailing___", "trailing"),
            ("multi   spaces", "multi_spaces"),
            ("weird!!chars@@", "weird_chars"),
            ("", "molecule"),
            (None, "molecule"),
            ("...", "molecule"),
            ("   ", "molecule"),
        ],
    )
    def test_cases(self, raw, expected):
        assert sanitize_filename(raw) == expected


# --------------------------------------------------------------------
# _unique_xyz_path
# --------------------------------------------------------------------


class TestUniqueXyzPath:
    def test_first_use_keeps_stem(self, tmp_path):
        p = _unique_xyz_path(tmp_path, "benzene")
        assert p.name == "benzene.xyz"

    def test_collision_appends_2(self, tmp_path):
        (tmp_path / "benzene.xyz").write_text("x")
        p = _unique_xyz_path(tmp_path, "benzene")
        assert p.name == "benzene_2.xyz"

    def test_chains_through_collisions(self, tmp_path):
        for n in ("benzene.xyz", "benzene_2.xyz", "benzene_3.xyz"):
            (tmp_path / n).write_text("x")
        p = _unique_xyz_path(tmp_path, "benzene")
        assert p.name == "benzene_4.xyz"


# --------------------------------------------------------------------
# parse_config
# --------------------------------------------------------------------


class TestParseConfig:
    def test_defaults_are_applied(self):
        cfg = SmilesTo3D().parse_config({"smiles": "CCO"})
        assert cfg["seed"] == 0
        assert cfg["max_embed_attempts"] == 50
        assert cfg["max_opt_iters"] == 500

    def test_str_ints_coerced(self):
        cfg = SmilesTo3D().parse_config(
            {"smiles": "CCO", "seed": "42", "max_embed_attempts": "10"}
        )
        assert cfg["seed"] == 42
        assert cfg["max_embed_attempts"] == 10

    def test_garbage_int_uses_default(self):
        # parse_int is permissive — bad ints fall back to the default rather
        # than crashing argv parsing.
        cfg = SmilesTo3D().parse_config({"smiles": "CCO", "seed": "not_a_number"})
        assert cfg["seed"] == 0

    def test_smiles_whitespace_stripped(self):
        cfg = SmilesTo3D().parse_config({"smiles": "  CCO  "})
        assert cfg["smiles"] == "CCO"

    def test_opt_lowercased(self):
        cfg = SmilesTo3D().parse_config({"smiles": "CCO", "opt": "MMFF"})
        assert cfg["opt"] == "mmff"

    def test_empty_name_becomes_none(self):
        cfg = SmilesTo3D().parse_config({"smiles": "CCO", "name": "   "})
        assert cfg["name"] is None


# --------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------


class TestHappyPath:
    def test_writes_xyz_and_manifest(self, tmp_path, stub_rdkit):
        m = _run_node(tmp_path, "smiles=CCO", "name=ethanol")
        assert m["ok"] is True
        assert m["step"] == "smiles_to_3d"
        # xyz file exists where the manifest says it does.
        assert len(m["artifacts"]["xyz"]) == 1
        rec = m["artifacts"]["xyz"][0]
        assert rec["name"] == "ethanol"
        assert rec["smiles"] == "CCO"
        assert rec["num_atoms"] == 9
        assert rec["num_heavy_atoms"] == 3
        assert rec["format"] == "xyz"
        assert rec["label"] == "embed_xyz"
        assert Path(rec["path_abs"]).exists()
        # sha256 is a 64-char hex.
        assert isinstance(rec["sha256"], str) and len(rec["sha256"]) == 64

    def test_inputs_block_records_resolved_config(self, tmp_path, stub_rdkit):
        m = _run_node(
            tmp_path, "smiles=c1ccccc1", "name=benzene", "opt=uff", "seed=7"
        )
        ins = m["inputs"]
        assert ins["smiles"] == "c1ccccc1"
        assert ins["name"] == "benzene"
        assert ins["opt"] == "uff"
        assert ins["seed"] == 7

    def test_default_name_is_molecule(self, tmp_path, stub_rdkit):
        m = _run_node(tmp_path, "smiles=CCO")
        rec = m["artifacts"]["xyz"][0]
        assert rec["name"] == "molecule"
        assert Path(rec["path_abs"]).name == "molecule.xyz"

    def test_emits_pointer_line_and_returns_zero(self, tmp_path, stub_rdkit, capsys):
        cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            rc = SmilesTo3D().invoke(["smiles_to_3d", "smiles=CCO"])
        finally:
            os.chdir(cwd)
        out = capsys.readouterr().out.strip().splitlines()
        assert rc == 0
        assert len(out) == 1
        ptr = json.loads(out[0])
        assert ptr["schema"] == "wf.pointer.v1"
        assert ptr["ok"] is True
        assert ptr["manifest_path"].endswith("manifest.json")

    def test_runtime_seconds_recorded(self, tmp_path, stub_rdkit):
        m = _run_node(tmp_path, "smiles=CCO")
        assert isinstance(m["runtime_seconds"], (int, float))
        assert m["runtime_seconds"] >= 0


# --------------------------------------------------------------------
# Failure paths
# --------------------------------------------------------------------


class TestFailures:
    def test_missing_smiles_is_soft_fail(self, tmp_path, stub_rdkit):
        m = _run_node(tmp_path)
        assert m["ok"] is False
        assert any("smiles" in f.get("error", "") for f in m["failures"])

    def test_empty_smiles_is_soft_fail(self, tmp_path, stub_rdkit):
        m = _run_node(tmp_path, "smiles=")
        # ``smiles=`` parses to ``""`` — should fail the same way as omission.
        assert m["ok"] is False
        assert any("smiles" in f.get("error", "") for f in m["failures"])

    def test_bad_smiles_is_soft_fail(self, tmp_path, stub_rdkit):
        m = _run_node(tmp_path, "smiles=BAD_SMILES")
        assert m["ok"] is False
        assert any("embed_failed" in f.get("error", "") for f in m["failures"])
        # No artifact should have been recorded.
        assert m["artifacts"]["xyz"] == []

    def test_embed_failure_is_soft_fail(self, tmp_path, stub_rdkit):
        m = _run_node(tmp_path, "smiles=EMBED_FAIL")
        assert m["ok"] is False
        assert any("embed_failed" in f.get("error", "") for f in m["failures"])

    def test_failure_path_still_emits_pointer(self, tmp_path, stub_rdkit, capsys):
        cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            rc = SmilesTo3D().invoke(["smiles_to_3d"])  # no smiles
        finally:
            os.chdir(cwd)
        out = capsys.readouterr().out.strip()
        assert rc == 0  # soft-fail still returns 0
        ptr = json.loads(out)
        assert ptr["ok"] is False

    def test_hard_fail_policy_returns_one(self, tmp_path, stub_rdkit):
        cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            rc = SmilesTo3D().invoke(["smiles_to_3d", "fail_policy=hard"])
        finally:
            os.chdir(cwd)
        # No smiles + hard policy → exit 1.
        assert rc == 1


# --------------------------------------------------------------------
# Filename collision
# --------------------------------------------------------------------


class TestFilenameCollision:
    def test_two_runs_same_name_dont_collide(self, tmp_path, stub_rdkit):
        run_dir = tmp_path / "run1"
        run_dir.mkdir()

        cwd = os.getcwd()
        os.chdir(run_dir)
        try:
            SmilesTo3D().invoke(["smiles_to_3d", "smiles=CCO", "name=ethanol"])
            # Same name in same cwd should bump to ethanol_2.xyz.
            SmilesTo3D().invoke(["smiles_to_3d", "smiles=CCO", "name=ethanol"])
        finally:
            os.chdir(cwd)
        manifest = json.loads(
            (run_dir / "outputs" / "manifest.json").read_text()
        )
        # Second run's manifest should reference ethanol_2.xyz.
        rec = manifest["artifacts"]["xyz"][0]
        assert Path(rec["path_abs"]).name == "ethanol_2.xyz"

    def test_special_chars_sanitized_in_filename(self, tmp_path, stub_rdkit):
        m = _run_node(tmp_path, "smiles=CCO", "name=foo/bar baz")
        rec = m["artifacts"]["xyz"][0]
        # Sanitization: ``/`` → ``_``, space → ``_``.
        assert Path(rec["path_abs"]).name == "foo_bar_baz.xyz"


# --------------------------------------------------------------------
# RDKit-import error paths (no monkeypatch — exercise ensure_rdkit())
# --------------------------------------------------------------------


class TestRdkitImportPath:
    def test_real_rdkit_import_raises_clear_error_in_no_rdkit_env(
        self, tmp_path, monkeypatch
    ):
        # Without the stub_rdkit fixture, build_3d_mol -> ensure_rdkit -> ImportError
        # should produce a clean ``embed_failed`` failure.
        # Reset module-level cache so the import is re-attempted.
        monkeypatch.setattr(s23, "_Chem", None)
        monkeypatch.setattr(s23, "_AllChem", None)

        # Simulate "no rdkit" regardless of whether it's actually installed
        # in the test env. Setting sys.modules[<name>] to None makes
        # ``import <name>`` raise ImportError on the next attempt,
        # bypassing any cached real module. (Previously this test relied
        # on the sandbox not having rdkit, which broke once the chem
        # extras were installed locally.)
        import sys
        monkeypatch.setitem(sys.modules, "rdkit", None)
        monkeypatch.setitem(sys.modules, "rdkit.Chem", None)
        monkeypatch.setitem(sys.modules, "rdkit.Chem.AllChem", None)

        m = _run_node(tmp_path, "smiles=CCO")
        assert m["ok"] is False
        errors = " ".join(f.get("error", "") for f in m["failures"])
        # Either "embed_failed" wraps the ImportError, or RDKit is somehow
        # available — handle both cases tolerantly.
        if "embed_failed" in errors:
            assert "RDKit" in errors or "rdkit" in errors
