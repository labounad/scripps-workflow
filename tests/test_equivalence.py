"""Tests for ``scripps_workflow.equivalence``.

Three concerns, each in its own class:

1. Pure helpers that don't need RDKit (Excel-style label generation,
   the magnetic-equivalence J-vector test, the tier classifier).
   Parameterized cases for breadth without bulk.

2. RDKit-backed topological-class detection. We test three pinned
   molecules:

   * Methane (single class of 4 equivalent H's)
   * Ethanol (CH₃ class of 3, CH₂ class of 2, OH class of 1)
   * 1,2-dichlorobenzene (two classes of 2 — the C₂-symmetric AA'BB'
     pattern)
   * 2-bromobutane with explicit (S) stereo (diastereotopic CH₂ —
     two methylene H's get distinct ranks)

3. The end-to-end orchestrator on synthetic J/shift data, exercising
   the HARD / SOFT / NONE dispatch and verifying averaged shifts +
   pairwise J's land in the EquivalenceGroup objects.

RDKit-using tests are guarded by ``pytest.importorskip("rdkit")`` so the
suite still runs cleanly in stripped environments.
"""

from __future__ import annotations

from typing import Optional

import pytest

from scripps_workflow.equivalence import (
    EquivalenceGroup,
    Tier,
    assign_group_labels,
    classify_class_tier,
    compute_equivalence_groups,
    equivalence_classes_by_substitution,
    magnetic_equivalence_test,
    mol_from_smiles_or_xyz,
    topological_classes,
)


# --------------------------------------------------------------------
# Excel-style group labels (no rdkit needed)
# --------------------------------------------------------------------


class TestAssignGroupLabels:
    @pytest.mark.parametrize(
        "n, expected_tail",
        [
            (1, ["A"]),
            (3, ["A", "B", "C"]),
            (26, ["X", "Y", "Z"]),
            (27, ["Y", "Z", "AA"]),
            (28, ["Z", "AA", "AB"]),
            (52, ["AY", "AZ"]),
            (53, ["AY", "AZ", "BA"]),
            (702, ["ZX", "ZY", "ZZ"]),
            (703, ["ZY", "ZZ", "AAA"]),
        ],
    )
    def test_known_endings(self, n, expected_tail):
        labels = assign_group_labels(n)
        assert len(labels) == n
        # Head is monotonically A, B, C, ...
        assert labels[0] == "A"
        # Tail matches the Excel-style spec.
        assert labels[-len(expected_tail):] == expected_tail

    def test_zero_returns_empty(self):
        assert assign_group_labels(0) == []

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            assign_group_labels(-1)

    def test_uniqueness(self):
        # Sanity: no collisions across the AA-AZ boundary.
        labels = assign_group_labels(100)
        assert len(set(labels)) == 100


# --------------------------------------------------------------------
# Magnetic-equivalence test (pure, no rdkit)
# --------------------------------------------------------------------


class TestMagneticEquivalenceTest:
    def test_size_one_is_trivially_equivalent(self):
        assert magnetic_equivalence_test(
            class_atoms=[5],
            other_atoms=[1, 2, 3],
            j_matrix={},
            tol_hz=0.5,
        ) is True

    def test_empty_class_is_trivially_equivalent(self):
        assert magnetic_equivalence_test(
            class_atoms=[],
            other_atoms=[1, 2],
            j_matrix={},
            tol_hz=0.5,
        ) is True

    def test_methyl_pattern_is_equivalent(self):
        # 3 methyl H's (atoms 0, 1, 2) all couple equally to a neighbor (atom 3).
        j_matrix = {
            (0, 3): 7.1,
            (1, 3): 7.1,
            (2, 3): 7.1,
        }
        assert magnetic_equivalence_test(
            class_atoms=[0, 1, 2],
            other_atoms=[3],
            j_matrix=j_matrix,
            tol_hz=0.5,
        ) is True

    def test_aa_xx_pattern_is_inequivalent(self):
        # Atoms 0, 1 are topologically equivalent. Other atoms 2, 3 are
        # topologically equivalent. But J(0,2) ≠ J(0,3): the AA'BB'
        # pattern. Atom 0 sees [J(0,2), J(0,3)] = [8, 1]; atom 1 sees
        # [J(1,2), J(1,3)] = [1, 8]. Vectors don't match → inequivalent.
        j_matrix = {
            (0, 2): 8.0, (0, 3): 1.0,
            (1, 2): 1.0, (1, 3): 8.0,
        }
        assert magnetic_equivalence_test(
            class_atoms=[0, 1],
            other_atoms=[2, 3],
            j_matrix=j_matrix,
            tol_hz=0.5,
        ) is False

    def test_within_tolerance_still_equivalent(self):
        # DFT-noise level (0.3 Hz) within the default 0.5 Hz tolerance.
        j_matrix = {
            (0, 3): 7.0,
            (1, 3): 7.3,
            (2, 3): 7.1,
        }
        assert magnetic_equivalence_test(
            class_atoms=[0, 1, 2],
            other_atoms=[3],
            j_matrix=j_matrix,
            tol_hz=0.5,
        ) is True

    def test_above_tolerance_breaks_equivalence(self):
        j_matrix = {
            (0, 3): 7.0,
            (1, 3): 7.6,  # 0.6 Hz off → exceeds 0.5 Hz tolerance
        }
        assert magnetic_equivalence_test(
            class_atoms=[0, 1],
            other_atoms=[3],
            j_matrix=j_matrix,
            tol_hz=0.5,
        ) is False

    def test_missing_vs_present_J_breaks_equivalence(self):
        # Atom 0 has J to 3, atom 1 doesn't. Treating "missing" as
        # "implicit zero" risks calling AA'XX' patterns equivalent
        # when the parser dropped a long-range row; keep them strict.
        j_matrix = {(0, 3): 5.0}
        assert magnetic_equivalence_test(
            class_atoms=[0, 1],
            other_atoms=[3],
            j_matrix=j_matrix,
            tol_hz=0.5,
        ) is False

    def test_both_missing_is_match(self):
        # Two atoms, both have no J to a distant atom — match.
        assert magnetic_equivalence_test(
            class_atoms=[0, 1],
            other_atoms=[10],
            j_matrix={},
            tol_hz=0.5,
        ) is True


# --------------------------------------------------------------------
# Tier classifier (pure, no rdkit)
# --------------------------------------------------------------------


class TestClassifyClassTier:
    def test_size_one_is_none(self):
        tier = classify_class_tier(
            class_atoms=[5],
            other_atoms=[1, 2],
            j_matrix={(1, 5): 7.0},
            tol_hz=0.5,
        )
        assert tier == Tier.NONE

    def test_methyl_pattern_is_hard(self):
        tier = classify_class_tier(
            class_atoms=[0, 1, 2],
            other_atoms=[3],
            j_matrix={(0, 3): 7.1, (1, 3): 7.1, (2, 3): 7.1},
            tol_hz=0.5,
        )
        assert tier == Tier.HARD

    def test_aa_xx_is_soft(self):
        tier = classify_class_tier(
            class_atoms=[0, 1],
            other_atoms=[2, 3],
            j_matrix={
                (0, 2): 8.0, (0, 3): 1.0,
                (1, 2): 1.0, (1, 3): 8.0,
            },
            tol_hz=0.5,
        )
        assert tier == Tier.SOFT

    def test_empty_class_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            classify_class_tier(
                class_atoms=[],
                other_atoms=[1],
                j_matrix={},
                tol_hz=0.5,
            )


# --------------------------------------------------------------------
# Topological classes via RDKit (rdkit-required)
# --------------------------------------------------------------------


class TestTopologicalClasses:
    """Pinned-molecule cases. RDKit's CanonicalRankAtoms behavior is
    well-defined for these small molecules; if it changes upstream
    these tests catch it.
    """

    @pytest.fixture(scope="class")
    def Chem(self):
        rdkit = pytest.importorskip("rdkit")
        from rdkit import Chem  # noqa: WPS433
        return Chem

    def _build(self, Chem, smiles: str):
        return Chem.AddHs(Chem.MolFromSmiles(smiles))

    def test_methane_one_class(self, Chem):
        # CH4: 4 H's all topologically equivalent.
        mol = self._build(Chem, "C")
        h_classes = topological_classes(mol, element="H")
        assert len(h_classes) == 1
        assert len(h_classes[0]) == 4

    def test_ethanol_three_h_classes(self, Chem):
        # CH3-CH2-OH: 3+2+1 split.
        mol = self._build(Chem, "CCO")
        h_classes = topological_classes(mol, element="H")
        sizes = sorted(len(c) for c in h_classes)
        assert sizes == [1, 2, 3]

    def test_o_dichlorobenzene_two_h_classes_of_two(self, Chem):
        # 1,2-dichlorobenzene aromatic protons: by C2 symmetry,
        # H3↔H6 and H4↔H5. Two classes of 2.
        mol = self._build(Chem, "c1ccc(Cl)c(Cl)c1")
        h_classes = topological_classes(mol, element="H")
        sizes = sorted(len(c) for c in h_classes)
        assert sizes == [2, 2]

    def test_bromobutane_chirality_does_not_split_prochiral_ch2(self, Chem):
        # 2-bromobutane with explicit (S) stereo. The CH₂ on C3 is
        # chemically diastereotopic (one H is pro-R, one is pro-S in the
        # context of the chiral C2), but RDKit's CanonicalRankAtoms
        # with includeChirality=True does NOT split prochiral H's:
        # the chirality flag only applies to atom-level CIP @ tags,
        # which exist on C2 but not on its neighbors' H's.
        #
        # The orchestrator (compute_equivalence_groups) compensates
        # via a data-aware refinement step that splits classes with
        # DFT-shift spread > tol_shift_ppm. This test pins RDKit's
        # observed behavior at the topology layer; the matching test
        # in TestComputeEquivalenceGroups exercises the data-aware
        # split.
        mol = self._build(Chem, "C[C@H](Br)CC")
        h_classes = topological_classes(mol, element="H")
        sizes = sorted(len(c) for c in h_classes)
        # 4 classes: methine (1), diastereotopic CH₂ pair as one class
        # (2), and the two non-equivalent methyls (3, 3).
        assert sizes == [1, 2, 3, 3]
        assert sum(sizes) == 9  # 2-bromobutane is C4H9Br


# --------------------------------------------------------------------
# Substitution-test chemical classes (rdkit-required)
# --------------------------------------------------------------------


class TestSubstitutionEquivalenceClasses:
    @pytest.fixture(scope="class")
    def Chem(self):
        rdkit = pytest.importorskip("rdkit")
        from rdkit import Chem  # noqa: WPS433
        return Chem

    def _build(self, Chem, smiles: str):
        return Chem.AddHs(Chem.MolFromSmiles(smiles))

    def test_terminal_vinyl_ch2_hydrogens_are_distinct(self, Chem):
        # Propene is the minimal R-CH=CH2 case. Replacing one terminal
        # vinyl H gives the E isotopomer; replacing the other gives the Z
        # isotopomer. They are diastereotopic and must be NONE singletons
        # downstream, not a chemically-equivalent 2H class.
        mol = self._build(Chem, "C=CC")
        h_classes = equivalence_classes_by_substitution(mol, element="H")
        sizes = sorted(len(c) for c in h_classes)
        assert sizes == [1, 1, 1, 3]

        terminal_vinyl_hs = sorted(
            h.GetIdx()
            for h in mol.GetAtoms()
            if h.GetSymbol() == "H"
            and h.GetNeighbors()[0].GetSymbol() == "C"
            and sum(
                1 for n in h.GetNeighbors()[0].GetNeighbors()
                if n.GetSymbol() == "H"
            ) == 2
            and any(
                bond.GetBondType() == Chem.BondType.DOUBLE
                for bond in h.GetNeighbors()[0].GetBonds()
            )
        )
        assert len(terminal_vinyl_hs) == 2
        assert all([h] in h_classes for h in terminal_vinyl_hs)

    def test_ethene_and_symmetric_11_disubstituted_vinyl_ch2_merge(self, Chem):
        # Ethene has no differentiating substituent, and isobutylene's
        # two alkene substituents are identical. In both cases the =CH2
        # hydrogens remain chemically equivalent.
        ethene = self._build(Chem, "C=C")
        ethene_h = equivalence_classes_by_substitution(ethene, element="H")
        assert sorted(len(c) for c in ethene_h) == [4]

        isobutylene = self._build(Chem, "C=C(C)C")
        h_classes = equivalence_classes_by_substitution(
            isobutylene,
            element="H",
        )
        sizes = sorted(len(c) for c in h_classes)
        assert sizes == [2, 6]

    def test_oxetane_methylene_hydrogens_are_aa_bb_not_four_equivalent(self, Chem):
        smi = "O=C(C1(COC1)C=C)NCC2=CC=CC=N2"
        mol = self._build(Chem, smi)
        h_classes = equivalence_classes_by_substitution(mol, element="H")

        oxetane_hs = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != "C":
                continue
            if not atom.IsInRingSize(4):
                continue
            h_neighbors = sorted(
                n.GetIdx() for n in atom.GetNeighbors() if n.GetSymbol() == "H"
            )
            if len(h_neighbors) == 2:
                oxetane_hs.extend(h_neighbors)

        assert len(oxetane_hs) == 4
        oxetane_classes = [
            sorted(set(cls) & set(oxetane_hs))
            for cls in h_classes
            if set(cls) & set(oxetane_hs)
        ]
        assert sorted(len(c) for c in oxetane_classes) == [2, 2]

        # Each class spans the two symmetry-related CH2 carbons; the two
        # classes correspond to the two diastereotopic ring faces.
        for cls in oxetane_classes:
            parents = sorted(
                mol.GetAtomWithIdx(h).GetNeighbors()[0].GetIdx()
                for h in cls
            )
            assert len(set(parents)) == 2

    def test_oxetane_ch2_carbons_remain_equivalent_for_13c(self, Chem):
        smi = "O=C(C1(COC1)C=C)NCC2=CC=CC=N2"
        mol = self._build(Chem, smi)
        c_classes = equivalence_classes_by_substitution(mol, element="C")
        oxetane_ch2_carbons = sorted(
            atom.GetIdx()
            for atom in mol.GetAtoms()
            if atom.GetSymbol() == "C"
            and atom.IsInRingSize(4)
            and sum(1 for n in atom.GetNeighbors() if n.GetSymbol() == "H") == 2
        )
        assert len(oxetane_ch2_carbons) == 2
        assert any(set(cls) == set(oxetane_ch2_carbons) for cls in c_classes)


# --------------------------------------------------------------------
# Mol construction (rdkit-required)
# --------------------------------------------------------------------


class TestMolFromSmilesOrXyz:
    @pytest.fixture(scope="class")
    def rdkit(self):
        return pytest.importorskip("rdkit")

    def test_smiles_path(self, rdkit):
        mol = mol_from_smiles_or_xyz(smiles="CCO")
        assert mol is not None
        assert mol.GetNumAtoms() == 9  # 3 heavy + 6 H

    def test_xyz_fallback_path(self, rdkit):
        # Methane xyz block. Coordinates are tetrahedral-ish; RDKit's
        # DetermineBonds should perceive 4 C-H single bonds.
        xyz = (
            "5\n"
            "methane\n"
            "C  0.0000  0.0000  0.0000\n"
            "H  0.6300  0.6300  0.6300\n"
            "H -0.6300 -0.6300  0.6300\n"
            "H -0.6300  0.6300 -0.6300\n"
            "H  0.6300 -0.6300 -0.6300\n"
        )
        mol = mol_from_smiles_or_xyz(xyz_text=xyz, charge=0)
        assert mol is not None
        assert mol.GetNumAtoms() == 5

    def test_smiles_preferred_over_xyz_when_both_given(self, rdkit):
        # Even with garbage xyz, a clean SMILES should win.
        mol = mol_from_smiles_or_xyz(
            smiles="C", xyz_text="garbage\n"
        )
        assert mol is not None
        assert mol.GetNumAtoms() == 5  # CH4

    def test_neither_returns_none(self):
        assert mol_from_smiles_or_xyz() is None

    def test_invalid_smiles_returns_none(self, rdkit):
        # Bare SMILES failure with no xyz fallback → None.
        assert mol_from_smiles_or_xyz(smiles="not_a_smiles_$$") is None


# --------------------------------------------------------------------
# Orchestrator (rdkit-required)
# --------------------------------------------------------------------


class TestComputeEquivalenceGroups:
    @pytest.fixture(scope="class")
    def rdkit(self):
        return pytest.importorskip("rdkit")

    def _h_indices(self, mol):
        return [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "H"]

    def test_ethanol_methyl_hard_methylene_soft(self, rdkit):
        # CCO: methyl 3H (HARD), methylene 2H (SOFT), OH (NONE).
        # The methylene protons share one averaged shift but remain as
        # separate spin groups so any geminal / AA′ topology can be
        # preserved downstream.
        from rdkit import Chem  # noqa: WPS433
        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        h_atoms = self._h_indices(mol)
        # Identify which H's are methyl / methylene / hydroxyl by
        # their parent heavy atom's connectivity.
        methyl_hs = sorted(
            h for h in h_atoms
            if mol.GetAtomWithIdx(h).GetNeighbors()[0].GetDegree() == 4
            and mol.GetAtomWithIdx(h).GetNeighbors()[0].GetSymbol() == "C"
            and len([
                n for n in mol.GetAtomWithIdx(h).GetNeighbors()[0].GetNeighbors()
                if n.GetSymbol() == "H"
            ]) == 3
        )
        methylene_hs = sorted(
            h for h in h_atoms
            if mol.GetAtomWithIdx(h).GetNeighbors()[0].GetSymbol() == "C"
            and len([
                n for n in mol.GetAtomWithIdx(h).GetNeighbors()[0].GetNeighbors()
                if n.GetSymbol() == "H"
            ]) == 2
        )
        hydroxyl_hs = sorted(
            h for h in h_atoms
            if mol.GetAtomWithIdx(h).GetNeighbors()[0].GetSymbol() == "O"
        )
        assert len(methyl_hs) == 3 and len(methylene_hs) == 2 and len(hydroxyl_hs) == 1

        # Synthetic shifts: typical-ish ethanol values.
        shifts = {h: 1.20 for h in methyl_hs}  # CH3 ~ 1.2 ppm
        shifts.update({h: 3.70 for h in methylene_hs})  # CH2 ~ 3.7 ppm
        shifts.update({h: 2.50 for h in hydroxyl_hs})  # OH (varies)

        # Symmetric J's between methyl and methylene (3J ~ 7 Hz).
        # All methyl H's couple equally to all methylene H's, so the
        # methyl class is HARD-equivalent.
        j_matrix: dict[tuple[int, int], float] = {}
        for m in methyl_hs:
            for ml in methylene_hs:
                j_matrix[(min(m, ml), max(m, ml))] = 7.0

        groups = compute_equivalence_groups(
            mol=mol,
            element="H",
            shifts_by_atom=shifts,
            j_matrix=j_matrix,
            tol_jcoupling_hz=0.5,
        )
        # 4 groups: A (methyl, number=3, 1.2 ppm), B/C (methylene
        # SOFT groups, same averaged 3.7 ppm), D (OH, number=1, 2.5 ppm).
        assert len(groups) == 4
        ch3 = next(g for g in groups if g.shift_avg_ppm == pytest.approx(1.20))
        ch2_groups = [g for g in groups if g.shift_avg_ppm == pytest.approx(3.70)]
        oh = next(g for g in groups if g.shift_avg_ppm == pytest.approx(2.50))
        assert ch3.tier == Tier.HARD and ch3.number == 3
        assert len(ch2_groups) == 2
        assert all(g.tier == Tier.SOFT and g.number == 1 for g in ch2_groups)
        assert oh.tier == Tier.NONE and oh.number == 1
        # Methyl-to-each-methylene-soft-group J should be 7.0, averaged
        # over the three methyl H's for each destination proton.
        for ch2 in ch2_groups:
            assert ch3.j_couplings[ch2.name] == pytest.approx(7.0)
        # No J from methyl-or-methylene to OH (we didn't put any in).
        assert oh.name not in ch3.j_couplings
        for ch2 in ch2_groups:
            assert oh.name not in ch2.j_couplings

    def test_aa_xx_aromatic_emits_soft_groups(self, rdkit):
        # 1,2-dichlorobenzene. Pre-stage synthetic J's that mimic the
        # ortho/meta/para coupling pattern: J(H3,H4)=8 Hz (ortho),
        # J(H3,H5)=1 Hz (meta), J(H4,H6)=1 Hz (meta), J(H5,H6)=8 Hz
        # (ortho). Cross-class J's must NOT all match → SOFT dispatch.
        from rdkit import Chem  # noqa: WPS433
        mol = Chem.AddHs(Chem.MolFromSmiles("c1ccc(Cl)c(Cl)c1"))
        h_classes = topological_classes(mol, element="H")
        # Two classes of 2.
        assert sorted(len(c) for c in h_classes) == [2, 2]
        cls_a = h_classes[0]  # contains lowest-index H
        cls_b = h_classes[1]
        # Pin shifts within each class so we can identify them by value.
        shifts = {a: 7.40 for a in cls_a}
        shifts.update({a: 7.20 for a in cls_b})
        # Cross-class J's: by C2 symmetry, J(a0, b0) == J(a1, b1) and
        # J(a0, b1) == J(a1, b0), but those two values differ.
        a0, a1 = cls_a
        b0, b1 = cls_b
        j_matrix = {
            (min(a0, b0), max(a0, b0)): 8.0,
            (min(a0, b1), max(a0, b1)): 1.0,
            (min(a1, b0), max(a1, b0)): 1.0,
            (min(a1, b1), max(a1, b1)): 8.0,
        }

        groups = compute_equivalence_groups(
            mol=mol,
            element="H",
            shifts_by_atom=shifts,
            j_matrix=j_matrix,
            tol_jcoupling_hz=0.5,
        )
        # 4 groups (each topological class fans out into 2 SOFT groups).
        assert len(groups) == 4
        # All four groups are SOFT — the class structure is what makes
        # this AA'BB' rather than separate singletons.
        assert all(g.tier == Tier.SOFT for g in groups)
        # All number=1 (SOFT never collapses).
        assert all(g.number == 1 for g in groups)
        # Two groups with shift 7.40 (cls_a), two with 7.20 (cls_b).
        shifts_seen = sorted(g.shift_avg_ppm for g in groups)
        assert shifts_seen == pytest.approx([7.20, 7.20, 7.40, 7.40])

    def test_diastereotopic_ch2_splits_via_shift_spread(self, rdkit):
        # 2-bromobutane (S). RDKit's topology gives the CH₂ pair as
        # one class of 2 (chirality-of-attached-atoms doesn't propagate
        # through CanonicalRankAtoms). The orchestrator's data-aware
        # refinement step splits the class because real DFT-computed
        # shifts for diastereotopic CH₂ in a chiral environment spread
        # well above the tol_shift_ppm threshold.
        #
        # We feed realistic-ish shifts (≈0.3 ppm spread for the
        # diastereotopic pair, uniform within each genuinely-equivalent
        # class) and verify the post-refinement structure: 2 HARD
        # methyls (one per non-equivalent methyl group) + 3 NONE
        # singletons (methine + diastereotopic pair).
        from rdkit import Chem  # noqa: WPS433
        mol = Chem.AddHs(Chem.MolFromSmiles("C[C@H](Br)CC"))

        # Identify atoms by carbon environment:
        methylene_pair: list[int] = []
        methine_h: list[int] = []
        methyl_groups: list[list[int]] = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != "C":
                continue
            h_neighbors = sorted(
                n.GetIdx() for n in atom.GetNeighbors() if n.GetSymbol() == "H"
            )
            if len(h_neighbors) == 3:
                methyl_groups.append(h_neighbors)
            elif len(h_neighbors) == 2:
                methylene_pair = h_neighbors
            elif len(h_neighbors) == 1:
                methine_h = h_neighbors
        assert len(methylene_pair) == 2
        assert len(methine_h) == 1
        assert len(methyl_groups) == 2

        # Synthetic shifts:
        #   methyls: uniform within each group (homotopic),
        #   methine: 4.00 ppm,
        #   diastereotopic CH₂: 1.40 ppm vs 1.75 ppm (0.35 ppm spread).
        shifts: dict[int, float] = {}
        shifts[methyl_groups[0][0]] = 1.05
        shifts[methyl_groups[0][1]] = 1.05
        shifts[methyl_groups[0][2]] = 1.05
        shifts[methyl_groups[1][0]] = 0.95
        shifts[methyl_groups[1][1]] = 0.95
        shifts[methyl_groups[1][2]] = 0.95
        shifts[methine_h[0]] = 4.00
        shifts[methylene_pair[0]] = 1.40
        shifts[methylene_pair[1]] = 1.75

        groups = compute_equivalence_groups(
            mol=mol,
            element="H",
            shifts_by_atom=shifts,
            j_matrix={},
            tol_jcoupling_hz=0.5,
            tol_shift_ppm=0.05,
        )
        hard = [g for g in groups if g.tier == Tier.HARD]
        none_groups = [g for g in groups if g.tier == Tier.NONE]
        # 2 methyl groups (HARD, number=3 each), 3 NONE singletons:
        # methine + 2 split diastereotopic H's.
        assert len(hard) == 2
        assert all(g.number == 3 for g in hard)
        assert len(none_groups) == 3
        assert all(g.number == 1 for g in none_groups)
        # The diastereotopic pair lands as TWO distinct NONE groups,
        # one at 1.40 and one at 1.75 — not averaged.
        none_shifts = sorted(g.shift_avg_ppm for g in none_groups)
        assert any(abs(s - 1.40) < 1e-6 for s in none_shifts)
        assert any(abs(s - 1.75) < 1e-6 for s in none_shifts)
        assert any(abs(s - 4.00) < 1e-6 for s in none_shifts)

    def test_diastereotopic_ch2_splits_even_when_shifts_close(self, rdkit):
        # Same molecule but with shifts within tol_shift_ppm: chemical
        # inequivalence is now decided by the substitution test, not DFT
        # shift spread. The diastereotopic CH2 protons are therefore NONE
        # singletons even when their predicted shifts happen to be close.
        from rdkit import Chem  # noqa: WPS433
        mol = Chem.AddHs(Chem.MolFromSmiles("C[C@H](Br)CC"))
        h_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "H"]
        methylene_pair: list[int] = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != "C":
                continue
            h_neighbors = sorted(
                n.GetIdx() for n in atom.GetNeighbors() if n.GetSymbol() == "H"
            )
            if len(h_neighbors) == 2:
                methylene_pair = h_neighbors
        assert len(methylene_pair) == 2

        # Uniform shifts across the methylene pair (well within tol).
        shifts = {h: 1.0 for h in h_atoms}
        groups = compute_equivalence_groups(
            mol=mol,
            element="H",
            shifts_by_atom=shifts,
            j_matrix={},
            tol_jcoupling_hz=0.5,
            tol_shift_ppm=0.05,
        )
        assert sum(g.number for g in groups) == 9
        methylene_groups = [
            g for g in groups if set(g.atom_indices).issubset(set(methylene_pair))
        ]
        assert len(methylene_groups) == 2
        assert all(g.tier == Tier.NONE and g.number == 1 for g in methylene_groups)

    def test_force_hard_methyl_despite_asymmetric_js(self, rdkit):
        # Regression for the live ethanol run: real DFT on 2 conformers
        # gave methyl H–H J's that spread 5–15 Hz across atoms (no
        # rotation averaging), so the magnetic-equivalence test would
        # flag the class as SOFT. Force-HARD shortcut: size ≥ 3 +
        # same parent C → always HARD regardless of J asymmetry,
        # because methyl C₃ rotation is universal in solution NMR.
        from rdkit import Chem  # noqa: WPS433
        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        h_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "H"]
        # Identify the 3 methyl H's (parent C with 3 H neighbors).
        methyl_hs: list[int] = []
        for h in h_atoms:
            parent = mol.GetAtomWithIdx(h).GetNeighbors()[0]
            if parent.GetSymbol() == "C" and sum(
                1 for n in parent.GetNeighbors() if n.GetSymbol() == "H"
            ) == 3:
                methyl_hs.append(h)
        assert len(methyl_hs) == 3
        shifts = {h: 1.16 for h in h_atoms}
        # Wildly asymmetric J's between methyl H's — pre-fix this
        # would fail the mag-equiv test (any tol up to 10 Hz) → SOFT.
        # Post-fix: same-parent shortcut forces HARD.
        m1, m2, m3 = methyl_hs
        non_methyl = [h for h in h_atoms if h not in methyl_hs]
        j_matrix: dict[tuple[int, int], float] = {}
        # Give methyl H's WIDELY different J's to the methylene H's.
        if len(non_methyl) >= 2:
            j_matrix[(min(m1, non_methyl[0]), max(m1, non_methyl[0]))] = 5.0
            j_matrix[(min(m1, non_methyl[1]), max(m1, non_methyl[1]))] = 15.0
            j_matrix[(min(m2, non_methyl[0]), max(m2, non_methyl[0]))] = 15.0
            j_matrix[(min(m2, non_methyl[1]), max(m2, non_methyl[1]))] = 5.0
            j_matrix[(min(m3, non_methyl[0]), max(m3, non_methyl[0]))] = 2.0
            j_matrix[(min(m3, non_methyl[1]), max(m3, non_methyl[1]))] = 2.0

        groups = compute_equivalence_groups(
            mol=mol, element="H",
            shifts_by_atom=shifts, j_matrix=j_matrix,
            tol_jcoupling_hz=0.5, tol_shift_ppm=0.05,
        )
        methyl_group = next(
            (g for g in groups if set(g.atom_indices) == set(methyl_hs)),
            None,
        )
        assert methyl_group is not None
        assert methyl_group.tier == Tier.HARD
        assert methyl_group.number == 3

    def test_achiral_molecule_no_false_split_even_when_shifts_spread(
        self, rdkit
    ):
        # Regression for the live-run bug on ethanol (workflow run 102271):
        # 2-conformer Boltzmann ensemble didn't sample methyl C₃ rotation
        # uniformly, so DFT gave the 3 methyl H's σ values spread by
        # ~0.14 ppm (> tol_shift_ppm=0.05). Pre-fix, the data-aware
        # refinement split the methyl class into 3 NONE singletons,
        # producing 6 H groups (A/B/C/D/E/F). Post-fix, refinement is
        # gated on stereo presence; ethanol has no chiral center →
        # refinement skipped → topology rules preserve methyl HARD and
        # methylene SOFT equivalence without raw singleton splitting.
        from rdkit import Chem  # noqa: WPS433
        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        h_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "H"]
        # Identify by parent: methyl C has 3 H neighbors, methylene C
        # has 2, hydroxyl O has 1.
        methyl_hs: list[int] = []
        methylene_hs: list[int] = []
        hydroxyl_hs: list[int] = []
        for h in h_atoms:
            parent = mol.GetAtomWithIdx(h).GetNeighbors()[0]
            if parent.GetSymbol() == "O":
                hydroxyl_hs.append(h)
                continue
            n_h = sum(
                1 for n in parent.GetNeighbors() if n.GetSymbol() == "H"
            )
            if n_h == 3:
                methyl_hs.append(h)
            elif n_h == 2:
                methylene_hs.append(h)
        assert len(methyl_hs) == 3 and len(methylene_hs) == 2

        # Synthetic shifts that mimic the ethanol bug — methyl spread
        # 0.13 ppm, methylene spread 0.25 ppm — both exceed
        # tol_shift_ppm=0.05 but should NOT trigger refinement
        # because the molecule is achiral.
        shifts = {}
        shifts[methyl_hs[0]] = 1.20
        shifts[methyl_hs[1]] = 1.09
        shifts[methyl_hs[2]] = 1.16
        shifts[methylene_hs[0]] = 3.65
        shifts[methylene_hs[1]] = 3.90
        shifts[hydroxyl_hs[0]] = 2.50

        groups = compute_equivalence_groups(
            mol=mol,
            element="H",
            shifts_by_atom=shifts,
            j_matrix={},
            tol_jcoupling_hz=0.5,
            tol_shift_ppm=0.05,
        )
        # 4 groups: methyl (HARD, number=3), methylene as two SOFT
        # groups with the same averaged shift, OH (NONE, number=1).
        # NOT 6 raw singletons from shift-spread false splitting.
        assert len(groups) == 4
        hard = [g for g in groups if g.tier == Tier.HARD]
        soft = [g for g in groups if g.tier == Tier.SOFT]
        none = [g for g in groups if g.tier == Tier.NONE]
        assert len(hard) == 1 and hard[0].number == 3
        assert len(soft) == 2 and all(g.number == 1 for g in soft)
        assert len(none) == 1
        assert all(g.shift_avg_ppm == pytest.approx(3.775) for g in soft)
        # Sizes match the emitted spin groups.
        assert sorted(g.number for g in groups) == [1, 1, 1, 3]
    def test_tbutyl_nine_protons_force_hard_despite_shift_spread(self, rdkit):
        # tert-butanol: all nine tBu protons are one RDKit class spanning
        # three complete methyl groups. Even wildly asymmetric predicted
        # shifts from a non-rotationally averaged geometry should collapse
        # to one HARD 9H group.
        from rdkit import Chem  # noqa: WPS433
        mol = Chem.AddHs(Chem.MolFromSmiles("CC(C)(C)O"))
        h_classes = topological_classes(mol, element="H")
        tbu_class = next(c for c in h_classes if len(c) == 9)
        oh_class = next(c for c in h_classes if len(c) == 1)
        shifts = {h: 0.5 + 0.1 * i for i, h in enumerate(tbu_class)}
        shifts[oh_class[0]] = 2.0

        groups = compute_equivalence_groups(
            mol=mol,
            element="H",
            shifts_by_atom=shifts,
            j_matrix={},
            tol_jcoupling_hz=0.5,
            tol_shift_ppm=0.05,
        )
        tbu = next(g for g in groups if set(g.atom_indices) == set(tbu_class))
        assert tbu.tier == Tier.HARD
        assert tbu.number == 9
        assert tbu.shift_avg_ppm == pytest.approx(sum(shifts[h] for h in tbu_class) / 9)

    def test_patchouli_methyl_classes_survive_stereo_shift_refinement(self, rdkit):
        # Regression for patchouli alcohol: the molecule is stereochemically
        # rich, so the old broad shift-spread refinement split methyl H's
        # into A/B/C singletons. The corrected logic protects complete
        # methyl classes, while the substitution test is allowed to split
        # diastereotopic methyl substituents that the old topology-only
        # logic incorrectly folded into one gem-dimethyl 6H class.
        from rdkit import Chem  # noqa: WPS433
        smi = "C[C@H]1CC[C@@]2([C@@]3([C@H]1C[C@H](C2(C)C)CC3)C)O"
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        h_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "H"]
        # Deliberately give large within-methyl spreads; HARD rotational
        # classes should still average them rather than split.
        shifts = {h: 0.2 * (i % 7) for i, h in enumerate(h_atoms)}

        groups = compute_equivalence_groups(
            mol=mol,
            element="H",
            shifts_by_atom=shifts,
            j_matrix={},
            tol_jcoupling_hz=0.5,
            tol_shift_ppm=0.05,
        )
        hard_sizes = sorted(g.number for g in groups if g.tier == Tier.HARD)
        assert hard_sizes == [3, 3, 3, 3]
        assert all(g.number != 1 for g in groups if g.tier == Tier.HARD)

