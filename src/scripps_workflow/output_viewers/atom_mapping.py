"""Derive 2D-drawing atom maps for standalone molecular viewers.

The browser viewer draws the 2D inset from a SMILES string using RDKit.js.
Those SVG atom indices are SMILES/RDKit atom indices, not necessarily the same
as the atom order in an XYZ file produced by CREST, xTB, ORCA, etc.  Deriving
the map in Python during bundle creation is much more reliable than asking the
browser build of RDKit.js to perceive a molecule directly from raw XYZ.
"""

from __future__ import annotations


def _first_xyz_frame(xyz_text: str) -> str | None:
    lines = xyz_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i >= len(lines):
        return None
    try:
        n_atoms = int(lines[i].strip())
    except ValueError:
        return None
    if n_atoms <= 0:
        return None
    end = i + 2 + n_atoms
    if end > len(lines):
        return None
    return "\n".join(lines[i:end]) + "\n"


def _make_connectivity_query(mol):
    """Return a copy with bond orders/aromaticity relaxed for matching."""

    from rdkit import Chem  # type: ignore[import-not-found]

    out = Chem.Mol(mol)
    for atom in out.GetAtoms():
        atom.SetIsAromatic(False)
    for bond in out.GetBonds():
        bond.SetBondType(Chem.BondType.UNSPECIFIED)
        bond.SetIsAromatic(False)
    return out


def derive_smiles_to_xyz_atom_map(smiles: str | None, xyz_text: str) -> list[int] | None:
    """Map RDKit/SMILES heavy-atom indices to XYZ atom indices.

    Returns ``map[smiles_atom_index] = xyz_atom_index`` for the heavy atoms in
    ``smiles``.  The returned indices are zero-based and can be used directly by
    3Dmol.js because its atom arrays preserve the XYZ order.

    The function intentionally returns ``None`` rather than raising when RDKit
    cannot perceive or match the structures; viewer bundle generation should
    still succeed even without a selectable 2D/3D atom map.
    """

    if not smiles or not smiles.strip():
        return None
    frame = _first_xyz_frame(xyz_text)
    if not frame:
        return None

    try:
        from rdkit import Chem  # type: ignore[import-not-found]
        from rdkit.Chem import rdDetermineBonds  # type: ignore[import-not-found]
    except Exception:
        return None

    try:
        query = Chem.MolFromSmiles(smiles)
        if query is None or query.GetNumAtoms() == 0:
            return None

        target = Chem.MolFromXYZBlock(frame)
        if target is None or target.GetNumAtoms() == 0:
            return None

        try:
            rdDetermineBonds.DetermineBonds(target, charge=0)
        except Exception:
            try:
                rdDetermineBonds.DetermineConnectivity(target)
            except Exception:
                return None

        matches = target.GetSubstructMatches(query, useChirality=False)
        if not matches:
            q_relaxed = _make_connectivity_query(query)
            t_relaxed = _make_connectivity_query(target)
            matches = t_relaxed.GetSubstructMatches(q_relaxed, useChirality=False)
        if not matches:
            return None

        match = tuple(int(i) for i in matches[0])
        if len(match) != query.GetNumAtoms():
            return None
        return list(match)
    except Exception:
        return None
