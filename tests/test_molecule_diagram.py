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

    def test_annotations_increase_svg_content(self, rdkit):
        # RDKit's MolDraw2DSVG renders text as glyph paths (a sequence
        # of <path> elements with curve/line commands), not <text>
        # nodes. So we can't substring-search for the literal "1.26".
        # The functional invariant we CAN check: passing groups
        # produces measurably more drawing content than passing none,
        # because the atom-note glyphs add path elements.
        from rdkit import Chem
        mol = Chem.AddHs(Chem.MolFromSmiles("C"))
        svg_bare = render_shift_svg(mol=mol, groups=[])
        svg_annotated = render_shift_svg(
            mol=mol, groups=[_hard_methyl(name="A", shift=1.26)]
        )
        # Annotated SVG must have noticeably more content (multiple
        # extra path elements for the digits and parens of "1.26 (A) ×3").
        assert len(svg_annotated) > len(svg_bare) + 100
        # And both must round-trip through ElementTree.
        ET.fromstring(svg_bare)
        ET.fromstring(svg_annotated)

    def test_annotated_atom_carries_atomnote_property(self, rdkit):
        # Functional check on the OTHER side of the renderer: when we
        # call render_shift_svg, the renderer's internal mol copy gets
        # an atomNote property set on the anchor atom. We test this by
        # patching the SVG-emitting step (rdMolDraw2D) to capture the
        # mol it sees.
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D
        mol = Chem.AddHs(Chem.MolFromSmiles("C"))
        groups = [_hard_methyl(name="A", shift=1.26)]

        captured: dict = {}
        original = rdMolDraw2D.MolDraw2DSVG

        class _Spy:
            def __init__(self, w, h):
                self._inner = original(w, h)

            def drawOptions(self):
                return self._inner.drawOptions()

            def DrawMolecule(self, mol_arg):
                captured["mol"] = mol_arg
                return self._inner.DrawMolecule(mol_arg)

            def FinishDrawing(self):
                return self._inner.FinishDrawing()

            def GetDrawingText(self):
                return self._inner.GetDrawingText()

        rdMolDraw2D.MolDraw2DSVG = _Spy
        try:
            render_shift_svg(mol=mol, groups=groups)
        finally:
            rdMolDraw2D.MolDraw2DSVG = original

        # Anchor atom (first in atom_indices) carries the formatted
        # atomNote in the new "A: 1.16 (3)" style.
        anchor = captured["mol"].GetAtomWithIdx(groups[0].atom_indices[0])
        assert anchor.HasProp("atomNote")
        note = anchor.GetProp("atomNote")
        assert note == f"A: 1.26 ({groups[0].number})"

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
        # Centroid of (1,1,1), (-1,-1,1), (-1,1,-1), (1,-1,-1) is (0,0,0).
        assert '"x": 0.0' in html_text
        assert '"y": 0.0' in html_text
        assert '"z": 0.0' in html_text
        # TWO labels per group: bold letter + regular-weight shift.
        # Arial replaces Helvetica Neue Bold/Light so the bold-vs-regular
        # distinction renders cross-platform (Helvetica Neue worked only
        # on macOS). Colon between the two halves is dropped — the
        # anchor-alignment split already separates them visually.
        assert '"text": "A"' in html_text
        assert "Arial Bold" in html_text
        assert '"text": " 1.26 (4)"' in html_text
        assert '"font": "Arial"' in html_text

    def test_singleton_group_atom_position_label(self):
        # NONE-tier group → labels at the single atom's position. Same
        # two-label format; multiplicity is (1) for singletons.
        groups = [_none_h(name="B", atom_idx=2, shift=3.50)]
        html_text = render_shift_html(
            groups=groups, xyz_text=_TETRA_XYZ, title="t",
        )
        # Atom 2 in _TETRA_XYZ is at (-1, 1, -1).
        assert '"x": -1.0' in html_text
        # Bold letter + regular shift+multiplicity pair, no colon.
        assert '"text": "B"' in html_text
        assert '"text": " 3.50 (1)"' in html_text

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
        # No "X" group label in the labels JSON.
        assert '"text": "X"' not in html_text
        assert '"text": ": 2.00 (1)"' not in html_text

    def test_dimensions_propagate(self):
        html_text = render_shift_html(
            groups=[], xyz_text=_TETRA_XYZ, title="t",
            width=1200, height=800,
        )
        assert "width: 1200px" in html_text
        assert "height: 800px" in html_text
