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

Label format and placement:

* Format ``"{name}: {shift:.2f} ({number})"`` — group letter, colon,
  predicted δ in ppm, multiplicity in parens. Example: ``"A: 1.16 (3)"``
  for a methyl HARD group; ``"D: 2.50 (1)"`` for a hydroxyl singleton.
* HTML viewer renders this as TWO co-located 3Dmol.js labels with
  different fonts: bold "A" right-aligned + light ": 1.16 (3)"
  left-aligned. 3Dmol.js can't mix fonts in a single label, so the
  two-label trick keeps the bold/light distinction Lucas requested.

  Font caveat: 3Dmol.js's :class:`LabelSpec` has only ``font`` (no
  ``fontWeight`` / ``fontStyle``). The string is passed straight to
  canvas font matching. On macOS, ``"Helvetica Neue Bold"`` and
  ``"Helvetica Neue Light"`` are registered as standalone font NAMES
  that resolve to the correct weight variants. On Windows/Linux,
  those names typically aren't matched and the labels fall back to
  the system default font with no weight distinction. The HTML
  still renders correctly — just without the bold/light contrast.
* For HARD groups (size ≥ 2 — methyls, symmetric methylenes), labels
  anchor at the centroid of the contributing atoms (one label-pair
  per group, not per atom).
* For SOFT and NONE groups (size 1 atoms — diastereotopic CH₂,
  isolated CH, AA'BB' members), labels anchor at the single atom.

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
    xyz_text: Optional[str] = None,
    width: int = 900,
    height: int = 600,
) -> str:
    """Render the molecule as a 2D SVG snapshot with shift annotations.

    When ``xyz_text`` is provided, the molecule's 3D coordinates are
    PCA-aligned (largest variance → x, smallest → z) and projected
    onto the x-y plane. RDKit's drawer then renders the actual 3D
    shape rather than a schematic 2D layout — bond angles, ring
    puckers, etc. all reflect the real geometry. This is the "snapshot
    of the 3D structure" view, which matches what shows up in the
    3Dmol.js HTML viewer.

    Without ``xyz_text``, falls back to ``AllChem.Compute2DCoords`` —
    a clean schematic chemistry diagram. Use that mode when no
    geometry is available (e.g., SMILES-only path with no upstream
    conformer).

    Annotates one atom per group with the predicted shift + group
    name; other atoms in a HARD group remain unlabeled to keep the
    visual clean.

    Returns a complete ``<?xml ... ?>`` SVG document. RDKit is
    required; ``xyz_text``-driven snapshots additionally require numpy
    (already in the ``chem`` extra alongside RDKit).
    """
    from rdkit import Chem  # type: ignore[import-not-found]
    from rdkit.Chem import AllChem  # type: ignore[import-not-found]
    from rdkit.Chem.Draw import rdMolDraw2D  # type: ignore[import-not-found]

    # Work on a copy so we don't mutate the caller's mol (the aggregator
    # builds the mol once and reuses it for both ¹H and ¹³C diagrams).
    mol_copy = Chem.Mol(mol)

    # Choose the 2D layout source: 3D snapshot (xyz-driven) or
    # schematic (RDKit's Compute2DCoords).
    if xyz_text:
        used_3d = _set_2d_conformer_from_xyz(mol_copy, xyz_text)
    else:
        used_3d = False
    if not used_3d:
        AllChem.Compute2DCoords(mol_copy)

    # Annotate one atom per group with format "A: 1.16 (3)" — group
    # letter, colon, shift, multiplicity in parens. For HARD groups
    # the annotated atom is the first one in atom_indices; the others
    # stay bare so the visual stays clean.
    for group in groups:
        if not group.atom_indices:
            continue
        anchor_idx = group.atom_indices[0]
        atom = mol_copy.GetAtomWithIdx(anchor_idx)
        atom.SetProp(
            "atomNote",
            f"{group.name}: {group.shift_avg_ppm:.2f} ({group.number})",
        )

    drawer = rdMolDraw2D.MolDraw2DSVG(int(width), int(height))
    opts = drawer.drawOptions()
    opts.addAtomIndices = False
    # Make atom-note text smaller than the default so 3-character labels
    # like "1.26 (A) ×3" don't crowd the bonds.
    opts.annotationFontScale = 0.7
    drawer.DrawMolecule(mol_copy)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def _set_2d_conformer_from_xyz(mol: Any, xyz_text: str) -> bool:
    """Replace ``mol``'s conformer with a 2D projection of ``xyz_text``.

    Pipeline:

    1. Parse atomic coordinates from the xyz string.
    2. Center on the centroid.
    3. Rotate via PCA so the molecule's largest variance aligns with
       the x-axis, second-largest with y, and smallest variance points
       along z. This puts the molecule's "best 2D view" in the x-y
       plane regardless of how RDKit happened to embed it.
    4. Drop the z component and install the result as a fresh 2D
       conformer on ``mol``.

    Returns ``True`` if the projection succeeded, ``False`` if the xyz
    text didn't yield enough atom positions to cover ``mol`` (in which
    case the caller should fall back to ``Compute2DCoords``).

    Requires numpy. RDKit is imported inside via the caller; we keep
    the numpy import local so the module loads cleanly in numpy-less
    environments (only the snapshot mode actually needs it).
    """
    from rdkit import Chem  # type: ignore[import-not-found]

    positions = _parse_xyz_positions(xyz_text)
    n_atoms = mol.GetNumAtoms()
    if len(positions) < n_atoms:
        return False

    try:
        import numpy as np  # type: ignore[import-not-found]
    except ImportError:
        return False

    coords_3d = np.array([positions[i] for i in range(n_atoms)], dtype=float)
    centered = coords_3d - coords_3d.mean(axis=0)
    # PCA: covariance matrix → eigenvectors give the principal axes.
    # eigh returns eigenvalues ascending; we want the largest two as
    # the in-plane axes, so we reverse the column order.
    cov = centered.T @ centered
    _, eigvecs = np.linalg.eigh(cov)
    rotation = eigvecs[:, ::-1]  # columns: largest, middle, smallest variance
    aligned = centered @ rotation

    new_conf = Chem.Conformer(n_atoms)
    for i in range(n_atoms):
        new_conf.SetAtomPosition(i, (float(aligned[i, 0]), float(aligned[i, 1]), 0.0))
    mol.RemoveAllConformers()
    mol.AddConformer(new_conf, assignId=True)
    return True


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


def _labels_for_group(
    group: EquivalenceGroup,
    positions: list[tuple[float, float, float]],
) -> list[dict[str, Any]]:
    """Build the two 3Dmol.js label entries for one equivalence group.

    Each group renders as TWO co-located labels at the centroid (HARD
    collapsed atoms) or single atom position (NONE/SOFT singleton):

    * Bold group letter, ``alignment="centerRight"`` so the text grows
      leftward and ends at the anchor.
    * Light shift + multiplicity (``": 1.16 (3)"``),
      ``alignment="centerLeft"`` so the text grows rightward starting
      at the anchor.

    Together they read as ``"A: 1.16 (3)"`` with mixed Helvetica Neue
    Bold / Light weights — 3Dmol.js can't mix fonts within a single
    label, so the two-label trick is necessary.

    Returns ``[]`` if the group's atom indices are out of range for
    the supplied positions list (defensive — shouldn't happen if
    positions and mol come from the same xyz).
    """
    valid = [
        positions[idx]
        for idx in group.atom_indices
        if 0 <= idx < len(positions)
    ]
    if not valid:
        return []
    cx = sum(p[0] for p in valid) / len(valid)
    cy = sum(p[1] for p in valid) / len(valid)
    cz = sum(p[2] for p in valid) / len(valid)
    pos = {"x": cx, "y": cy, "z": cz}
    return [
        {
            "position": pos,
            "text": group.name,
            "font": "Helvetica Neue Bold",
            "alignment": "centerRight",
        },
        {
            "position": pos,
            "text": f": {group.shift_avg_ppm:.2f} ({group.number})",
            "font": "Helvetica Neue Light",
            "alignment": "centerLeft",
        },
    ]


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
  Each label reads <code>group: shift (multiplicity)</code> — e.g., <code>A: 1.16 (3)</code> for a methyl. The bold letter identifies the spin-system group; the light portion is the predicted &delta; (ppm) and the number of equivalent nuclei. Drag to rotate, scroll to zoom. Powered by <a href="https://3dmol.csb.pitt.edu">3Dmol.js</a>.
</p>
<script>
const xyzData = $xyz_data;
const labels = $labels_json;
const viewer = $$3Dmol.createViewer("viewer", {backgroundColor: "white"});
viewer.addModel(xyzData, "xyz");
// Ball-and-stick: spheres scaled to ~24% of VdW radius (so heavier
// atoms naturally appear larger), sticks at 0.13 Å. Tweak STICK_R
// and SPHERE_SCALE to taste — 0.13 + 0.24 is the lab default;
// 0.20 + 0.35 reads chunkier, 0.10 + 0.18 is more wireframe-y.
const STICK_R = 0.13;
const SPHERE_SCALE = 0.24;
viewer.setStyle({}, {stick: {radius: STICK_R}, sphere: {scale: SPHERE_SCALE}});
for (const lbl of labels) {
  viewer.addLabel(lbl.text, {
    position: lbl.position,
    backgroundColor: "white",
    backgroundOpacity: 0.85,
    fontColor: "black",
    fontSize: 11,
    font: lbl.font,
    alignment: lbl.alignment,
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
        labels.extend(_labels_for_group(group, positions))

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
