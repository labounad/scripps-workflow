"""Tests for ``scripps_workflow.molecule_diagram``.

Two renderers, two test classes:

1. :class:`TestRenderShiftSvg` — RDKit-backed, so guarded by
   ``pytest.importorskip("rdkit")``. Verifies the SVG is well-formed
   XML, contains the expected atom annotations, and survives the
   "no groups" edge case.

2. :class:`TestRenderShiftHtml` — pure string assembly, no RDKit
   dependency. Verifies the embedded xyz data, the labels JSON
   (including centroid math for HARD groups and per-atom labels for
   NONE/SOFT groups), title escaping, and the 3Dmol.js script tag.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import pytest

from scripps_workflow.equivalence import EquivalenceGroup, Tier
from scripps_workflow.molecule_diagram import (
    render_shift_html,
    render_shift_svg,
)


# --------------------------------------------------------------------
# Fixtures (no RDKit needed)
# --------------------------------------------------------------------


def _hard_methyl(name: str = "A", shift: float = 1.26) -> EquivalenceGroup:
    """A 3-atom HARD methyl group at atom indices (1, 2, 3)."""
    return EquivalenceGroup(
        name=name, element="H", atom_indices=(1, 2, 3),
        shift_avg_ppm=shift, tier=Tier.HARD, j_couplings={},
    )


def _none_h(name: str, atom_idx: int, shift: float) -> EquivalenceGroup:
    return EquivalenceGroup(
        name=name, element="H", atom_indices=(atom_idx,),
        shift_avg_ppm=shift, tier=Tier.NONE, j_couplings={},
    )


# Tetrahedral xyz: 4 H atoms at corners of a tetrahedron centered on
# origin. Centroid of {0,1,2,3} is (0, 0, 0). Centroid of any singleton
# is its own position.
_TETRA_XYZ = (
    "4\n"
    "tetra\n"
    "H  1.0  1.0  1.0\n"
    "H -1.0 -1.0  1.0\n"
    "H -1.0  1.0 -1.0\n"
    "H  1.0 -1.0 -1.0\n"
)


# --------------------------------------------------------------------
# render_shift_svg (RDKit-required)
# --------------------------------------------------------------------


class TestRenderShiftSvg:
    @pytest.fixture
    def rdkit(self):
        return pytest.importorskip("rdkit")

    def test_methane_svg_well_formed(self, rdkit):
        from rdkit import Chem
        mol = Chem.AddHs(Chem.MolFromSmiles("C"))
        groups = [_hard_methyl()]
        svg = render_shift_svg(mol=mol, groups=groups)
        # Starts with the XML declaration RDKit emits.
        assert svg.lstrip().startswith("<?xml")
        # Round-trips through ElementTree.
        ET.fromstring(svg)

    def test_atom_note_includes_shift_and_group_name(self, rdkit):
        from rdkit import Chem
        mol = Chem.AddHs(Chem.MolFromSmiles("C"))
        groups = [_hard_methyl(name="A", shift=1.26)]
        svg = render_shift_svg(mol=mol, groups=groups)
        # The atom note is rendered into the SVG as a <text> element.
        # We don't pin the exact glyph layout — just check the substrings
        # that should appear in the rendered text.
        assert "1.26" in svg
        # The "(A)" annotation should appear somewhere in the SVG.
        # RDKit may break it across separate <text> spans, so check loosely.
        assert "A" in svg

    def test_hard_group_carries_multiplicity_factor(self, rdkit):
        from rdkit import Chem
        mol = Chem.AddHs(Chem.MolFromSmiles("C"))
        groups = [_hard_methyl(name="A", shift=1.26)]
        svg = render_shift_svg(mol=mol, groups=groups)
        # ×4 should appear (methane has 4 H atoms in the HARD group).
        # Wait — methyl_group has 3 atoms in atom_indices. Adjust:
        # a methane all-H group would have atom_indices=(1,2,3,4) and number=4,
        # but our _hard_methyl fixture uses 3 atoms (a CH₃ within a larger mol).
        # Either way, the multiplicity factor should match number.
        assert f"×{groups[0].number}" in svg or f"&#215;{groups[0].number}" in svg or f"×{groups[0].number}" in svg

    def test_empty_groups_still_renders_a_molecule(self, rdkit):
        # No annotations = bare molecule diagram.
        from rdkit import Chem
        mol = Chem.AddHs(Chem.MolFromSmiles("C"))
        svg = render_shift_svg(mol=mol, groups=[])
        assert svg.lstrip().startswith("<?xml")
        ET.fromstring(svg)

    def test_caller_mol_not_mutated(self, rdkit):
        # Confirm the renderer makes a copy — calling it shouldn't
        # leave atomNote properties on the caller's mol.
        from rdkit import Chem
        mol = Chem.AddHs(Chem.MolFromSmiles("C"))
        atom = mol.GetAtomWithIdx(1)
        # No atomNote initially.
        assert not atom.HasProp("atomNote")
        render_shift_svg(mol=mol, groups=[_hard_methyl()])
        # Still no atomNote on the original.
        assert not atom.HasProp("atomNote")


# --------------------------------------------------------------------
# render_shift_html (no RDKit needed — pure string assembly)
# --------------------------------------------------------------------


class TestRenderShiftHtml:
    def test_well_formed_html(self):
        groups = [_hard_methyl()]
        # Add the atom indices that the xyz actually has (0..3).
        groups = [
            EquivalenceGroup(
                name="A", element="H", atom_indices=(0, 1, 2, 3),
                shift_avg_ppm=1.26, tier=Tier.HARD, j_couplings={},
            )
        ]
        html_text = render_shift_html(
            groups=groups, xyz_text=_TETRA_XYZ, title="test",
        )
        assert html_text.startswith("<!DOCTYPE html>")
        assert "<title>test</title>" in html_text
        assert html_text.rstrip().endswith("</html>")

    def test_loads_3dmol_from_cdn(self):
        groups = []
        html_text = render_shift_html(
            groups=groups, xyz_text=_TETRA_XYZ, title="t",
        )
        assert "3Dmol-min.js" in html_text
        # The JS global ($3Dmol) is preserved through Template's
        # ``$$`` escape — not turned into ``$3Dmol`` literal in the
        # template substitution step.
        assert "$3Dmol.createViewer" in html_text

    def test_xyz_data_embedded_inline(self):
        groups = []
        html_text = render_shift_html(
            groups=groups, xyz_text=_TETRA_XYZ, title="t",
        )
        # JSON-encoded xyz (escapes newlines as \n in the string literal).
        assert json.dumps(_TETRA_XYZ) in html_text

    def test_hard_group_centroid_label(self):
        # 4 atoms at tetrahedral corners → centroid at origin.
        groups = [
            EquivalenceGroup(
                name="A", element="H", atom_indices=(0, 1, 2, 3),
                shift_avg_ppm=1.26, tier=Tier.HARD, j_couplings={},
            ),
        ]
        html_text = render_shift_html(
            groups=groups, xyz_text=_TETRA_XYZ, title="t",
        )
        # Find the embedded labels JSON. Each entry has position + text.
        # Centroid of (1,1,1), (-1,-1,1), (-1,1,-1), (1,-1,-1) is (0,0,0).
        assert '"x": 0.0' in html_text
        assert '"y": 0.0' in html_text
        assert '"z": 0.0' in html_text
        # Label text carries the multiplicity factor (JSON-escaped × → ×).
        assert "1.26 (A) \\u00d74" in html_text

    def test_singleton_group_atom_position_label(self):
        # NONE-tier group → label at the single atom's position
        # (no multiplicity factor).
        groups = [_none_h(name="B", atom_idx=2, shift=3.50)]
        html_text = render_shift_html(
            groups=groups, xyz_text=_TETRA_XYZ, title="t",
        )
        # Atom 2 in _TETRA_XYZ is at (-1, 1, -1).
        assert '"x": -1.0' in html_text
        assert '"text": "3.50 (B)"' in html_text
        # No multiplicity factor on a singleton.
        assert "\\u00d7" not in html_text.split('"text": "3.50 (B)"')[0].split(
            '"x": -1.0'
        )[1]

    def test_title_html_escaped(self):
        # XML/HTML special chars in the title shouldn't break the page.
        html_text = render_shift_html(
            groups=[], xyz_text=_TETRA_XYZ,
            title="Methane <test> & <H1>",
        )
        # The literal title text escaped:
        assert "&lt;test&gt;" in html_text
        assert "&amp;" in html_text
        # And NOT present in raw form.
        assert "<test>" not in html_text

    def test_out_of_range_atom_skipped(self):
        # Group references atom 99, but xyz only has 4 atoms — the
        # group should be skipped silently rather than crashing.
        groups = [
            EquivalenceGroup(
                name="X", element="H", atom_indices=(99,),
                shift_avg_ppm=2.0, tier=Tier.NONE, j_couplings={},
            ),
        ]
        html_text = render_shift_html(
            groups=groups, xyz_text=_TETRA_XYZ, title="t",
        )
        # No "(X)" label in the labels JSON.
        assert '"text": "2.00 (X)"' not in html_text

    def test_dimensions_propagate(self):
        html_text = render_shift_html(
            groups=[], xyz_text=_TETRA_XYZ, title="t",
            width=1200, height=800,
        )
        assert "width: 1200px" in html_text
        assert "height: 800px" in html_text
