"""Molecule-with-shift-label diagrams for NMR predictions.

Two renderers, each producing a self-contained file:

* :func:`render_shift_svg` — RDKit-driven 2D SVG using ``rdMolDraw2D``.
  Atoms labeled with their predicted shift + group name. Embeddable in
  reports and notebooks; viewable in any browser, PDF viewer with SVG
  support, or vector-graphics editor.

* :func:`render_shift_html` — Standalone HTML page with an embedded
  `3Dmol.js`_ viewer. The molecule's optimized 3D geometry is included
  inline as XYZ text, and per-group labels are placed at the centroid
  of each equivalence class. The user can drag to rotate, scroll to
  zoom, and see all the atom labels in 3D. Loads ``3Dmol-min.js`` from
  CDN — the page needs an internet connection on first open, but the
  molecule data itself is fully embedded so it's portable across
  machines.

Both renderers operate on the same ``EquivalenceGroup`` list that the
mnova-spinsim XML emitter consumes, so the visualization is consistent
with whatever spectrum mnova ends up rendering. The labels include the
group name (``A``, ``B``, ...) so the diagram acts as a legend for
reading the simulated spectrum: peak in spectrum at 1.26 ppm with
``name="A"`` corresponds to the atoms labeled "1.26 (A)" on the diagram.

Label-placement convention:

* For HARD groups (size ≥ 2 — methyls, symmetric methylenes), one label
  at the centroid of the contributing atoms with text
  ``"{shift:.2f} ({name}) ×{N}"``. Avoids cluttering the diagram with
  N identical labels.
* For SOFT and NONE groups (size 1 in either case — diastereotopic
  CH₂, isolated CH, AA'BB' members), one label at the single atom
  with text ``"{shift:.2f} ({name})"``.

.. _3Dmol.js: https://3dmol.csb.pitt.edu
"""

from __future__ import annotations

import html
import json
from string import Template
from typing import Any, Optional

from .equivalence import EquivalenceGroup, Tier


# --------------------------------------------------------------------
# 2D SVG renderer
# --------------------------------------------------------------------


def render_shift_svg(
    *,
    mol: Any,
    groups: list[EquivalenceGroup],
    width: int = 900,
    height: int = 600,
) -> str:
    """Render the molecule as 2D SVG with each group's atoms annotated.

    Computes a fresh 2D layout via ``AllChem.Compute2DCoords`` (so the
    diagram looks like a typical chemistry drawing rather than a
    flattened 3D projection). Annotates one atom per group with the
    predicted shift + group name; other atoms in a HARD group remain
    unlabeled to keep the diagram readable.

    The returned string is a complete ``<?xml ... ?>`` SVG document
    suitable for writing to disk or embedding in HTML.

    RDKit is required.
    """
    from rdkit import Chem  # type: ignore[import-not-found]
    from rdkit.Chem import AllChem  # type: ignore[import-not-found]
    from rdkit.Chem.Draw import rdMolDraw2D  # type: ignore[import-not-found]

    # Work on a copy so we don't mutate the caller's mol (the aggregator
    # builds the mol once and reuses it for both ¹H and ¹³C diagrams).
    mol_copy = Chem.Mol(mol)
    AllChem.Compute2DCoords(mol_copy)

    # Annotate one atom per group. For HARD (collapsed) groups the
    # annotated atom is the first one in atom_indices; the others stay
    # bare so the visual stays clean.
    for group in groups:
        if not group.atom_indices:
            continue
        anchor_idx = group.atom_indices[0]
        atom = mol_copy.GetAtomWithIdx(anchor_idx)
        if group.number > 1 and group.tier == Tier.HARD:
            note = f"{group.shift_avg_ppm:.2f} ({group.name}) ×{group.number}"
        else:
            note = f"{group.shift_avg_ppm:.2f} ({group.name})"
        atom.SetProp("atomNote", note)

    drawer = rdMolDraw2D.MolDraw2DSVG(int(width), int(height))
    opts = drawer.drawOptions()
    opts.addAtomIndices = False
    # Make atom-note text smaller than the default so 3-character labels
    # like "1.26 (A) ×3" don't crowd the bonds.
    opts.annotationFontScale = 0.7
    drawer.DrawMolecule(mol_copy)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


# --------------------------------------------------------------------
# 3D HTML renderer (3Dmol.js viewer)
# --------------------------------------------------------------------


def _parse_xyz_positions(xyz_text: str) -> list[tuple[float, float, float]]:
    """Parse a multi-line xyz string and return a list of (x, y, z) tuples.

    Standard xyz format: line 1 = atom count, line 2 = comment, then
    one ``element x y z`` line per atom. We tolerate extra whitespace
    and bail gracefully when lines are short — a partially-parsed list
    is more useful than a hard error since the caller only labels a
    subset of atoms anyway.
    """
    positions: list[tuple[float, float, float]] = []
    lines = xyz_text.splitlines()
    if len(lines) < 2:
        return positions
    try:
        n = int(lines[0].strip().split()[0])
    except (ValueError, IndexError):
        return positions
    for i in range(2, min(2 + n, len(lines))):
        parts = lines[i].split()
        if len(parts) < 4:
            continue
        try:
            positions.append(
                (float(parts[1]), float(parts[2]), float(parts[3]))
            )
        except ValueError:
            continue
    return positions


def _label_for_group(
    group: EquivalenceGroup,
    positions: list[tuple[float, float, float]],
) -> Optional[dict[str, Any]]:
    """Compute one label entry per group (centroid + text), or None.

    HARD groups get a centroid label including the multiplicity factor
    (``×N``); other groups get a single-atom label. Returns ``None`` if
    the group's atoms are out-of-range for the supplied positions list
    (defensive — shouldn't happen if positions and mol come from the
    same xyz, but guards against caller errors).
    """
    valid = [
        positions[idx]
        for idx in group.atom_indices
        if 0 <= idx < len(positions)
    ]
    if not valid:
        return None
    cx = sum(p[0] for p in valid) / len(valid)
    cy = sum(p[1] for p in valid) / len(valid)
    cz = sum(p[2] for p in valid) / len(valid)
    if group.number > 1 and group.tier == Tier.HARD:
        text = f"{group.shift_avg_ppm:.2f} ({group.name}) ×{group.number}"
    else:
        text = f"{group.shift_avg_ppm:.2f} ({group.name})"
    return {"position": {"x": cx, "y": cy, "z": cz}, "text": text}


# Standalone HTML template. Uses string.Template with $-substitutions
# so the JavaScript braces don't collide with the more common .format
# pattern. ``$$3Dmol`` escapes to literal ``$3Dmol`` (the JS global
# from the 3Dmol.js library).
_HTML_TPL = Template("""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>$title</title>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; margin: 24px; color: #1a1a1a; }
  h1 { font-size: 18px; font-weight: 600; margin: 0 0 12px 0; }
  #viewer { width: ${width}px; height: ${height}px; position: relative; border: 1px solid #d0d0d0; border-radius: 4px; background: #fafafa; }
  .legend { font-size: 12px; color: #555; margin-top: 10px; max-width: ${width}px; line-height: 1.5; }
  code { background: #f0f0f0; padding: 1px 4px; border-radius: 2px; font-size: 11px; }
</style>
</head>
<body>
<h1>$title</h1>
<div id="viewer"></div>
<p class="legend">
  Each label shows <code>predicted δ (group)</code>. HARD-equivalent groups (methyls, symmetric methylenes) show a multiplicity factor like <code>×3</code>. Drag to rotate, scroll to zoom. Powered by <a href="https://3dmol.csb.pitt.edu">3Dmol.js</a>.
</p>
<script>
const xyzData = $xyz_data;
const labels = $labels_json;
const viewer = $$3Dmol.createViewer("viewer", {backgroundColor: "white"});
viewer.addModel(xyzData, "xyz");
viewer.setStyle({}, {stick: {radius: 0.15}, sphere: {radius: 0.3, scale: 0.25}});
for (const lbl of labels) {
  viewer.addLabel(lbl.text, {
    position: lbl.position,
    backgroundColor: "white",
    backgroundOpacity: 0.85,
    fontColor: "black",
    fontSize: 11,
    borderColor: "#999",
    borderThickness: 0.5,
    inFront: true,
  });
}
viewer.zoomTo();
viewer.render();
</script>
</body>
</html>
""")


def render_shift_html(
    *,
    groups: list[EquivalenceGroup],
    xyz_text: str,
    title: str = "Predicted NMR shifts",
    width: int = 900,
    height: int = 600,
) -> str:
    """Render the molecule as a standalone HTML page with a 3Dmol.js viewer.

    The molecule's 3D geometry is embedded inline as the supplied
    ``xyz_text`` (typically from any conformer's optimized geometry).
    Per-group labels are placed at the centroid of each equivalence
    class; HARD groups carry a ``×N`` multiplicity factor.

    The output is one HTML file with no external dependencies other
    than the 3Dmol.js library, loaded from CDN. Internet required on
    first open; subsequent opens are cached by the browser.

    Does NOT depend on RDKit — pure string assembly. The atom-index
    range is taken from the parsed xyz, so the caller is responsible
    for ensuring ``groups``' ``atom_indices`` align with the xyz's
    atom ordering. (For the aggregator, both come from the same
    SMILES → AddHs pipeline, so they align by construction.)
    """
    positions = _parse_xyz_positions(xyz_text)
    labels: list[dict[str, Any]] = []
    for group in groups:
        entry = _label_for_group(group, positions)
        if entry is not None:
            labels.append(entry)

    return _HTML_TPL.substitute(
        title=html.escape(title),
        width=int(width),
        height=int(height),
        xyz_data=json.dumps(xyz_text),
        labels_json=json.dumps(labels),
    )


__all__ = [
    "render_shift_html",
    "render_shift_svg",
]
