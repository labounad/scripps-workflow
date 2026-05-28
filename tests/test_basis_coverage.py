"""Tests for :mod:`scripps_workflow.basis_coverage`.

Three concerns:

1. Coverage decisions across the realistic basis × element matrix
   (organic basis + organic atoms = no-op; organic basis + Br = Tier 1
   supplementation; organic basis + Pd = Tier 2 escalation; relativistic
   basis + Pd = no-op).
2. ``%basis newgto`` block formatting + fingerprint encoding.
3. Robustness: unknown basis names → warning + skip; empty inputs; xyz
   scan helper.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripps_workflow import basis_coverage as bc


# --------------------------------------------------------------------
# Coverage decisions
# --------------------------------------------------------------------


class TestCoverageDecision:
    """Element × basis matrix — make sure each cell behaves correctly."""

    def test_organic_atoms_organic_basis_is_noop(self):
        """Bread-and-butter case: C/H/N/O/F in 6-31G(d,p) — nothing to do."""
        d = bc.compute_coverage_decision(
            {"H", "C", "N", "O", "F"}, basis="6-31G(d,p)",
        )
        assert not d.has_supplementation
        assert not d.has_tier2
        assert d.extra_blocks == []
        assert d.fingerprint_suffix == ""
        assert d.warnings == []

    def test_bromine_in_pople_dp_is_covered(self):
        """Br (Z=35) IS in the 6-31G(d,p) coverage table (Pople H–Kr)."""
        d = bc.compute_coverage_decision(
            {"H", "C", "Br"}, basis="6-31G(d,p)",
        )
        assert not d.has_supplementation
        assert not d.has_tier2

    def test_bromine_in_pcj2_needs_supplementation(self):
        """pcJ-2 is organic-main-group only — Br must get supplemented."""
        d = bc.compute_coverage_decision(
            {"H", "C", "Br"}, basis="pcJ-2",
        )
        assert d.has_supplementation
        assert d.supplemented_elements == ["Br"]
        assert not d.has_tier2
        assert "Br" in d.extra_blocks[0]
        assert "def2-TZVPP" in d.extra_blocks[0]
        assert d.fingerprint_suffix == "+def2-TZVPP/heavy"

    def test_iodine_in_pople_diffuse_needs_supplementation(self):
        """6-311++G(2d,p) diffuse stops at Ar — I must get supplemented."""
        d = bc.compute_coverage_decision(
            {"H", "C", "I"}, basis="6-311++G(2d,p)",
        )
        assert d.supplemented_elements == ["I"]
        assert d.extra_blocks[0].count("newgto") == 1

    def test_multiple_heavies_get_one_block(self):
        """One %basis block with multiple newgto lines."""
        d = bc.compute_coverage_decision(
            {"H", "C", "Br", "I", "Se"}, basis="pcJ-2",
        )
        assert d.supplemented_elements == ["Br", "I", "Se"]
        assert len(d.extra_blocks) == 1
        block = d.extra_blocks[0]
        assert block.count("newgto") == 3
        for elem in ("Br", "I", "Se"):
            assert f'newgto {elem} "def2-TZVPP" end' in block

    def test_pd_in_organic_basis_escalates_to_tier2(self):
        """Pd is Tier 2 — NOT supplemented, surfaced via tier2_elements."""
        d = bc.compute_coverage_decision(
            {"H", "C", "Pd"}, basis="6-31G(d,p)",
        )
        assert not d.has_supplementation
        assert d.has_tier2
        assert d.tier2_elements == ["Pd"]
        assert d.fingerprint_suffix == ""

    def test_pd_with_zora_basis_is_noop(self):
        """def2-ZORA-TZVPP covers Pd natively — no-op."""
        d = bc.compute_coverage_decision(
            {"H", "C", "Pd"}, basis="def2-ZORA-TZVPP",
        )
        assert not d.has_supplementation
        assert not d.has_tier2

    def test_pd_in_def2_tzvpp_is_noop(self):
        """def2-TZVPP also covers Pd via ECPs — no supplementation needed."""
        d = bc.compute_coverage_decision(
            {"H", "C", "Pd"}, basis="def2-TZVPP",
        )
        assert not d.has_supplementation
        assert not d.has_tier2

    def test_mixed_tier1_and_tier2(self):
        """Br + Pd in 6-31G(d,p): Br is OK (Pople covers Z=35), Pd is Tier 2."""
        d = bc.compute_coverage_decision(
            {"H", "C", "Br", "Pd"}, basis="6-31G(d,p)",
        )
        assert d.supplemented_elements == []
        assert d.tier2_elements == ["Pd"]

    def test_tier1_and_tier2_in_pcj2(self):
        """In pcJ-2: Br needs supplement, Pd needs profile escalation."""
        d = bc.compute_coverage_decision(
            {"H", "C", "Br", "Pd"}, basis="pcJ-2",
        )
        assert d.supplemented_elements == ["Br"]
        assert d.tier2_elements == ["Pd"]
        # Block should NOT contain Pd — only Tier 1 elements.
        assert "Pd" not in d.extra_blocks[0]

    def test_case_insensitive_basis_lookup(self):
        """Lookups are case-insensitive — 6-31G(d,p) and 6-31g(d,p) match."""
        d1 = bc.compute_coverage_decision({"Br"}, basis="6-31G(D,P)")
        d2 = bc.compute_coverage_decision({"Br"}, basis="6-31g(d,p)")
        assert d1.supplemented_elements == d2.supplemented_elements

    def test_unknown_basis_emits_warning_only(self):
        """Unknown basis → no supplementation but a warning surfaces."""
        d = bc.compute_coverage_decision(
            {"H", "C", "Br"}, basis="my-custom-basis-2026",
        )
        assert not d.has_supplementation
        assert not d.has_tier2
        assert any("BASIS_ELEMENT_COVERAGE" in w for w in d.warnings)


# --------------------------------------------------------------------
# Block formatting + fingerprint encoding
# --------------------------------------------------------------------


class TestBlockFormatting:

    def test_build_newgto_block_single_element(self):
        block = bc.build_newgto_block(["Br"], "def2-TZVPP")
        assert block == (
            "%basis\n"
            '  newgto Br "def2-TZVPP" end\n'
            "end"
        )

    def test_build_newgto_block_multiple_elements(self):
        block = bc.build_newgto_block(["Br", "I"], "def2-TZVPP")
        lines = block.splitlines()
        assert lines[0] == "%basis"
        assert lines[-1] == "end"
        assert 'newgto Br "def2-TZVPP" end' in block
        assert 'newgto I "def2-TZVPP" end' in block

    def test_build_newgto_block_empty_is_empty_string(self):
        assert bc.build_newgto_block([], "def2-TZVPP") == ""

    def test_build_newgto_block_uses_supplied_basis_name(self):
        block = bc.build_newgto_block(["Br"], "SARC-ZORA-TZVPP")
        assert 'newgto Br "SARC-ZORA-TZVPP" end' in block

    def test_format_basis_fingerprint_with_supplementation(self):
        d = bc.compute_coverage_decision({"Br"}, basis="pcJ-2")
        fp = bc.format_basis_fingerprint("pcJ-2", d)
        assert fp == "pcJ-2+def2-TZVPP/heavy"

    def test_format_basis_fingerprint_no_supplementation(self):
        d = bc.compute_coverage_decision({"H", "C"}, basis="pcJ-2")
        fp = bc.format_basis_fingerprint("pcJ-2", d)
        assert fp == "pcJ-2"


# --------------------------------------------------------------------
# Robustness — empty inputs, unknown elements, xyz scan helper
# --------------------------------------------------------------------


class TestRobustness:

    def test_empty_elements_is_noop(self):
        d = bc.compute_coverage_decision(set(), basis="6-31G(d,p)")
        assert not d.has_supplementation
        assert not d.has_tier2

    def test_supplement_basis_uncovered_emits_warning(self):
        """If user picks a supplement basis that doesn't cover the
        heavies they're being routed to, warn loudly."""
        # pcJ-2 doesn't cover Br, so using it as a supplement basis
        # for Br would fail at ORCA read time.
        d = bc.compute_coverage_decision(
            {"H", "C", "Br"},
            basis="pcS-2",          # base also doesn't cover Br
            supplement_basis="pcJ-2",  # supplement also doesn't cover Br
        )
        assert d.supplemented_elements == ["Br"]
        assert any("does not cover" in w for w in d.warnings)


class TestXyzScan:

    def test_scan_basic_xyz(self, tmp_path: Path):
        xyz = tmp_path / "mol.xyz"
        xyz.write_text(
            "3\n"
            "title comment line\n"
            "C  0.0  0.0  0.0\n"
            "H  1.0  0.0  0.0\n"
            "Br 0.0  1.5  0.0\n"
        )
        elements = bc.scan_elements_from_xyz_paths([xyz])
        assert elements == {"C", "H", "Br"}

    def test_scan_skips_header_and_numerics(self, tmp_path: Path):
        """Atom-count header line and numeric tokens shouldn't be
        misread as element symbols."""
        xyz = tmp_path / "mol.xyz"
        xyz.write_text(
            "2\n"
            "comment\n"
            "C  0.0  0.0  0.0\n"
            "H  1.0  0.0  0.0\n"
        )
        elements = bc.scan_elements_from_xyz_paths([xyz])
        # No bogus "2" or numeric atoms picked up.
        assert elements == {"C", "H"}

    def test_scan_handles_missing_file(self, tmp_path: Path):
        elements = bc.scan_elements_from_xyz_paths([tmp_path / "missing.xyz"])
        assert elements == set()

    def test_scan_caps_at_max_files(self, tmp_path: Path):
        """Scanning a huge ensemble should not read every file —
        element composition is invariant across conformers."""
        for i in range(10):
            xyz = tmp_path / f"conf_{i}.xyz"
            xyz.write_text(
                f"1\nconf {i}\n"
                + ("C" if i < 3 else "Br")
                + "  0.0  0.0  0.0\n"
            )
        paths = sorted(tmp_path.glob("*.xyz"))
        elements = bc.scan_elements_from_xyz_paths(paths, max_files=3)
        # Only first 3 files scanned — they all have C, never Br.
        assert elements == {"C"}


# --------------------------------------------------------------------
# Parametrized realism — every basis we ship + every interesting element
# --------------------------------------------------------------------


@pytest.mark.parametrize(
    "basis,elements,expects_supplement,expects_tier2",
    [
        # Organic baseline — no-ops.
        ("6-31G(d,p)",     {"H", "C", "N", "O"},      False, False),
        ("6-311++G(2d,p)", {"H", "C", "N", "O", "F"}, False, False),
        ("pcJ-2",          {"H", "C", "F", "P"},      False, False),
        # Tier 1 supplementation cases — the original bug Lucas hit.
        ("pcJ-2",          {"H", "C", "Br"},          True,  False),
        ("pcJ-2",          {"H", "C", "I"},           True,  False),
        ("6-311++G(2d,p)", {"H", "C", "Br"},          True,  False),
        ("6-311++G(2d,p)", {"H", "C", "Se"},          True,  False),
        # Tier 2 cases — must be flagged for profile escalation.
        ("6-31G(d,p)",     {"H", "C", "Pd"},          False, True),
        ("pcJ-2",          {"H", "C", "Pt"},          False, True),
        ("pcJ-2",          {"H", "C", "Rh"},          False, True),
        # Relativistic basis — covers everything we care about.
        ("def2-ZORA-TZVPP", {"H", "C", "Pd"},         False, False),
        ("def2-ZORA-TZVPP", {"H", "C", "Pt", "Br"},   False, False),
        ("def2-TZVPP",      {"H", "C", "Pd"},         False, False),
    ],
)
def test_coverage_matrix(basis, elements, expects_supplement, expects_tier2):
    d = bc.compute_coverage_decision(elements, basis=basis)
    assert d.has_supplementation == expects_supplement
    assert d.has_tier2 == expects_tier2
