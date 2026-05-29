"""Tests for :mod:`scripps_workflow.basis_coverage`.

Three concerns:

1. Coverage decisions across the realistic element × basis matrix
   (organic atoms = no-op; light-heavy + organic basis = Tier 1
   supplementation; HALA-relevant element = full ZORA swap; mixed
   case = ZORA wins).
2. ``%basis newgto`` block formatting + fingerprint encoding.
3. Robustness: unknown basis names, empty inputs, xyz scan helper.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripps_workflow import basis_coverage as bc


# --------------------------------------------------------------------
# Coverage decisions
# --------------------------------------------------------------------


class TestCoverageDecision:
    """Element × basis matrix — every cell behaves correctly."""

    def test_organic_atoms_organic_basis_is_noop(self):
        d = bc.compute_coverage_decision(
            {"H", "C", "N", "O", "F"}, base_basis="6-31G(d,p)",
        )
        assert d.effective_basis == "6-31G(d,p)"
        assert d.fingerprint_basis == "6-31G(d,p)"
        assert d.nmr_keywords_prefix == ""
        assert not d.has_supplementation
        assert not d.has_relativistic_treatment
        assert d.extra_blocks == []

    def test_bromine_in_pople_dp_is_covered(self):
        """Br (Z=35) IS in 6-31G(d,p) (Pople H–Kr)."""
        d = bc.compute_coverage_decision(
            {"H", "C", "Br"}, base_basis="6-31G(d,p)",
        )
        assert not d.has_supplementation
        assert not d.has_relativistic_treatment

    def test_bromine_in_pcj2_needs_supplementation(self):
        d = bc.compute_coverage_decision(
            {"H", "C", "Br"}, base_basis="pcJ-2",
        )
        assert d.has_supplementation
        assert d.supplemented_elements == ["Br"]
        assert not d.has_relativistic_treatment
        assert d.effective_basis == "pcJ-2"   # operator basis stays
        assert d.nmr_keywords_prefix == ""
        assert "Br" in d.extra_blocks[0]
        assert "def2-TZVPP" in d.extra_blocks[0]
        assert d.fingerprint_basis == "pcJ-2+def2-TZVPP/heavy"

    def test_iodine_in_diffuse_pople_needs_supplementation(self):
        d = bc.compute_coverage_decision(
            {"H", "C", "I"}, base_basis="6-311++G(2d,p)",
        )
        assert d.supplemented_elements == ["I"]
        assert d.extra_blocks[0].count("newgto") == 1

    def test_multiple_tier1_heavies_one_block(self):
        d = bc.compute_coverage_decision(
            {"H", "C", "Br", "I", "Se"}, base_basis="pcJ-2",
        )
        assert d.supplemented_elements == ["Br", "I", "Se"]
        assert len(d.extra_blocks) == 1
        block = d.extra_blocks[0]
        for elem in ("Br", "I", "Se"):
            assert f'newgto {elem} "def2-TZVPP" end' in block

    def test_pd_triggers_full_zora_swap(self):
        """Pd → effective_basis becomes the relativistic basis, the
        operator's base_basis is discarded for this job."""
        d = bc.compute_coverage_decision(
            {"H", "C", "Pd"}, base_basis="6-31G(d,p)",
        )
        assert d.has_relativistic_treatment
        assert d.tier2_elements == ["Pd"]
        assert d.effective_basis == "def2-ZORA-TZVPP"
        assert d.fingerprint_basis == "def2-ZORA-TZVPP"
        assert d.nmr_keywords_prefix == "ZORA"
        assert d.extra_blocks == []   # no per-atom block in the swap path
        assert not d.has_supplementation

    def test_pt_also_triggers_zora(self):
        d = bc.compute_coverage_decision(
            {"H", "C", "Pt"}, base_basis="pcJ-2",
        )
        assert d.has_relativistic_treatment
        assert d.nmr_keywords_prefix == "ZORA"
        assert d.effective_basis == "def2-ZORA-TZVPP"

    def test_lanthanide_triggers_zora(self):
        """Lanthanides also need a relativistic Hamiltonian."""
        d = bc.compute_coverage_decision(
            {"H", "C", "Eu"}, base_basis="6-31G(d,p)",
        )
        assert d.has_relativistic_treatment

    def test_first_row_tm_does_NOT_trigger_zora(self):
        """3d transition metals (Sc–Zn) are deliberately outside the
        relativistic set — ECPs / no-ECP organic basis handle them."""
        d = bc.compute_coverage_decision(
            {"H", "C", "Fe"}, base_basis="def2-TZVPP",
        )
        assert not d.has_relativistic_treatment

    def test_tier2_subsumes_tier1(self):
        """When Tier 1 (Br) and Tier 2 (Pd) both present: full ZORA
        swap covers Br via def2-ZORA-TZVPP, no per-atom block."""
        d = bc.compute_coverage_decision(
            {"H", "C", "Br", "Pd"}, base_basis="pcJ-2",
        )
        assert d.has_relativistic_treatment
        assert not d.has_supplementation
        assert d.extra_blocks == []
        assert d.effective_basis == "def2-ZORA-TZVPP"

    def test_custom_relativistic_basis_threads_through(self):
        d = bc.compute_coverage_decision(
            {"H", "C", "Pd"},
            base_basis="pcJ-2",
            relativistic_basis="SARC-ZORA-TZVPP",
        )
        assert d.effective_basis == "SARC-ZORA-TZVPP"
        assert d.fingerprint_basis == "SARC-ZORA-TZVPP"

    def test_case_insensitive_basis_lookup(self):
        d1 = bc.compute_coverage_decision({"Br"}, base_basis="6-31G(D,P)")
        d2 = bc.compute_coverage_decision({"Br"}, base_basis="6-31g(d,p)")
        assert d1.has_supplementation == d2.has_supplementation

    def test_unknown_basis_emits_warning_only(self):
        d = bc.compute_coverage_decision(
            {"H", "C", "Br"}, base_basis="my-custom-basis-2026",
        )
        assert not d.has_supplementation
        assert not d.has_relativistic_treatment
        assert any("BASIS_ELEMENT_COVERAGE" in w for w in d.warnings)


class TestNeedsRelativisticTreatment:
    """The standalone predicate matches what compute_coverage_decision
    uses internally."""

    def test_organic_false(self):
        assert not bc.needs_relativistic_treatment({"H", "C", "N", "O"})

    def test_first_row_tm_false(self):
        assert not bc.needs_relativistic_treatment({"H", "C", "Fe"})

    def test_pd_true(self):
        assert bc.needs_relativistic_treatment({"H", "C", "Pd"})

    def test_lanthanide_true(self):
        assert bc.needs_relativistic_treatment({"H", "C", "Eu"})

    def test_actinide_true(self):
        assert bc.needs_relativistic_treatment({"H", "C", "U"})

    def test_empty_false(self):
        assert not bc.needs_relativistic_treatment(set())


# --------------------------------------------------------------------
# Block formatting + fingerprint extraction
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


class TestExtractBaseBasis:
    """Calibration lookup uses this to strip a supplemented basis back
    to its calibrated base form."""

    def test_supplemented_strips_to_base(self):
        assert bc.extract_base_basis(
            "6-31G(d,p)+def2-TZVPP/heavy"
        ) == "6-31G(d,p)"

    def test_unchanged_passes_through(self):
        assert bc.extract_base_basis("6-31G(d,p)") == "6-31G(d,p)"

    def test_zora_full_swap_passes_through(self):
        """The Tier-2 full-swap basis IS the identity — no suffix to
        strip. The calibration lookup will then either find a separately
        lab-fit ZORA row or return None."""
        assert bc.extract_base_basis("def2-ZORA-TZVPP") == "def2-ZORA-TZVPP"


# --------------------------------------------------------------------
# Robustness — empty inputs, unknown elements, xyz scan helper
# --------------------------------------------------------------------


class TestRobustness:

    def test_empty_elements_is_noop(self):
        d = bc.compute_coverage_decision(set(), base_basis="6-31G(d,p)")
        assert not d.has_supplementation
        assert not d.has_relativistic_treatment

    def test_supplement_basis_uncovered_emits_warning(self):
        d = bc.compute_coverage_decision(
            {"H", "C", "Br"},
            base_basis="pcS-2",
            heavy_atom_basis="pcJ-2",   # also doesn't cover Br
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
        xyz = tmp_path / "mol.xyz"
        xyz.write_text(
            "2\n"
            "comment\n"
            "C  0.0  0.0  0.0\n"
            "H  1.0  0.0  0.0\n"
        )
        elements = bc.scan_elements_from_xyz_paths([xyz])
        assert elements == {"C", "H"}

    def test_scan_handles_missing_file(self, tmp_path: Path):
        elements = bc.scan_elements_from_xyz_paths([tmp_path / "missing.xyz"])
        assert elements == set()

    def test_scan_caps_at_max_files(self, tmp_path: Path):
        for i in range(10):
            xyz = tmp_path / f"conf_{i}.xyz"
            xyz.write_text(
                f"1\nconf {i}\n"
                + ("C" if i < 3 else "Br")
                + "  0.0  0.0  0.0\n"
            )
        paths = sorted(tmp_path.glob("*.xyz"))
        elements = bc.scan_elements_from_xyz_paths(paths, max_files=3)
        assert elements == {"C"}


# --------------------------------------------------------------------
# Parametrized realism — every basis we ship + every interesting element
# --------------------------------------------------------------------


@pytest.mark.parametrize(
    "basis,elements,expect_supp,expect_zora",
    [
        # Organic baseline — no-ops.
        ("6-31G(d,p)",     {"H", "C", "N", "O"},      False, False),
        ("6-311++G(2d,p)", {"H", "C", "N", "O", "F"}, False, False),
        ("pcJ-2",          {"H", "C", "F", "P"},      False, False),
        # Tier 1 supplementation cases.
        ("pcJ-2",          {"H", "C", "Br"},          True,  False),
        ("pcJ-2",          {"H", "C", "I"},           True,  False),
        ("6-311++G(2d,p)", {"H", "C", "Br"},          True,  False),
        ("6-311++G(2d,p)", {"H", "C", "Se"},          True,  False),
        # Tier 2 — full swap.
        ("6-31G(d,p)",     {"H", "C", "Pd"},          False, True),
        ("pcJ-2",          {"H", "C", "Pt"},          False, True),
        ("pcJ-2",          {"H", "C", "Rh"},          False, True),
        ("6-31G(d,p)",     {"H", "C", "Eu"},          False, True),
        # Tier 2 subsumes Tier 1.
        ("pcJ-2",          {"H", "C", "Br", "Pd"},    False, True),
        # Relativistic basis already → no-op.
        ("def2-ZORA-TZVPP", {"H", "C", "Pd"},         False, True),
        ("def2-TZVPP",      {"H", "C", "Pd"},         False, True),
    ],
)
def test_coverage_matrix(basis, elements, expect_supp, expect_zora):
    d = bc.compute_coverage_decision(elements, base_basis=basis)
    assert d.has_supplementation == expect_supp, d
    assert d.has_relativistic_treatment == expect_zora, d
