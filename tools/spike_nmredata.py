"""Spike: emit a sample NMReDATA-tagged .sdf for ethanol.

Generates a single file ``ethanol_nmredata.sdf`` (or a path of your
choosing) containing:

  * A 3D-embedded ethanol molecule (mol block, V2000 SDF format).
  * NMReDATA tags carrying hand-picked typical chemical shifts +
    multiplet structure + atom-to-peak assignments for ¹H and ¹³C.

The file is intentionally tiny so it's quick to load in mnova and
inspect. Outcome we want to verify:

  1. mnova opens the .sdf and displays the ethanol structure.
  2. Loading triggers some kind of NMR-aware view (predicted spectrum,
     atom labels overlaid on shift values, etc.).
  3. Atoms are linked to peaks — i.e., clicking a peak highlights the
     corresponding atom on the structure (or vice versa).

If all three work, we commit to the full NMReDATA integration in
nmr_aggregate (replacing the manual hand-picked numbers below with
DFT-predicted values). If only (1) and (2) work but the atom↔peak
linking is missing or broken, we'll either tweak the format details
or fall back to the sidecar-bundle approach.

NMReDATA reference: https://nmredata.org (IUPAC-recommended format,
~2018-onwards). The format is sdf-based with ``>  <NMREDATA_*>`` data
fields appended after the mol block. Atom indices in the assignment
block are 1-based (sdf convention), NOT 0-based (rdkit convention) —
the conversion happens here.

Usage:

    python tools/spike_nmredata.py [output_path]

Default output: ``ethanol_nmredata.sdf`` in the current directory.

Requires rdkit (in the project's ``chem`` extra).
"""

from __future__ import annotations

import sys
from pathlib import Path


# --------------------------------------------------------------------
# Hand-picked typical NMR data for ethanol in CDCl3.
# --------------------------------------------------------------------
#
# Chemical shifts (in ppm) — typical literature values:
#
#   ¹H    methyl (CH₃-)     1.18 ppm   triplet, ³J = 7.0 Hz
#         methylene (-CH₂-) 3.69 ppm   quartet, ³J = 7.0 Hz
#         hydroxyl (-OH)    2.60 ppm   broad singlet (variable)
#
#   ¹³C   methyl            18.0 ppm
#         methylene         58.0 ppm
#
# These numbers are deliberately approximate — what matters for the
# spike is the FILE STRUCTURE, not the exactness of the values.
# --------------------------------------------------------------------


def _build_mol_block() -> str:
    """3D-embed ethanol via RDKit and return a V2000 mol block.

    Atom ordering in the resulting SDF is whatever ``Chem.AddHs`` and
    ``MolToMolBlock`` produce; we read it back to figure out which
    atoms are which (methyl Hs vs methylene Hs vs OH H, etc.).
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    return Chem.MolToMolBlock(mol), mol


def _classify_atoms(mol):
    """Identify methyl Hs / methylene Hs / OH H / methyl C / methylene C / O.

    Returns a dict whose keys are role labels and whose values are
    *1-based* sdf atom indices. NMReDATA atom indices are 1-based.
    """
    methyl_hs: list[int] = []
    methylene_hs: list[int] = []
    hydroxyl_hs: list[int] = []
    methyl_c: int | None = None
    methylene_c: int | None = None
    oxygen: int | None = None

    for atom in mol.GetAtoms():
        idx_1based = atom.GetIdx() + 1
        if atom.GetSymbol() == "O":
            oxygen = idx_1based
        elif atom.GetSymbol() == "C":
            h_neighbors = [
                n for n in atom.GetNeighbors() if n.GetSymbol() == "H"
            ]
            if len(h_neighbors) == 3:
                methyl_c = idx_1based
            elif len(h_neighbors) == 2:
                methylene_c = idx_1based
        elif atom.GetSymbol() == "H":
            parent = atom.GetNeighbors()[0]
            if parent.GetSymbol() == "O":
                hydroxyl_hs.append(idx_1based)
            elif parent.GetSymbol() == "C":
                # Distinguish methyl-H from methylene-H by parent's H count.
                n_h = sum(
                    1 for nb in parent.GetNeighbors() if nb.GetSymbol() == "H"
                )
                if n_h == 3:
                    methyl_hs.append(idx_1based)
                elif n_h == 2:
                    methylene_hs.append(idx_1based)

    return {
        "methyl_hs": sorted(methyl_hs),
        "methylene_hs": sorted(methylene_hs),
        "hydroxyl_hs": sorted(hydroxyl_hs),
        "methyl_c": methyl_c,
        "methylene_c": methylene_c,
        "oxygen": oxygen,
    }


def _format_atom_list(atoms: list[int]) -> str:
    """NMReDATA assignment: comma-separated atom indices."""
    return ", ".join(str(i) for i in atoms)


def _build_nmredata_block(atoms: dict) -> str:
    """Emit the NMReDATA tag block matching the molecule from
    :func:`_build_mol_block`.

    Format notes (from the NMReDATA specification):

    * Each tag opens with ``>  <NMREDATA_*>`` (two spaces before ``<``).
    * Tag bodies are multi-line; each line ENDS WITH ``\\`` (a literal
      backslash) to mark a continuation. The final line of a tag body
      also gets a trailing ``\\`` followed by a blank line before the
      next tag.
    * The whole record ends with ``$$$$`` on its own line (the SDF
      record separator).
    * NMREDATA_LEVEL=1 means "structure + chemical shifts" (level 2
      adds full coupling constants, level 3 adds 2D experiments).

    The assignment block links group labels (L1, L2, ...) to atom
    indices. The 1D blocks then reference those labels per peak.
    """
    methyl_hs = _format_atom_list(atoms["methyl_hs"])
    methylene_hs = _format_atom_list(atoms["methylene_hs"])
    hydroxyl_hs = _format_atom_list(atoms["hydroxyl_hs"])

    return (
        ">  <NMREDATA_VERSION>\n"
        "1.1\\\n"
        "\n"
        ">  <NMREDATA_LEVEL>\n"
        "2\\\n"
        "\n"
        # Atom assignment: maps labels to sdf atom indices.
        ">  <NMREDATA_ASSIGNMENT>\n"
        f"L1, 1.18, {methyl_hs}\\\n"
        f"L2, 3.69, {methylene_hs}\\\n"
        f"L3, 2.60, {hydroxyl_hs}\\\n"
        f"L4, 18.0, {atoms['methyl_c']}\\\n"
        f"L5, 58.0, {atoms['methylene_c']}\\\n"
        "\n"
        # 1D ¹H: per-multiplet line is `shift, multiplicity, J=..., label`.
        ">  <NMREDATA_1D_1H>\n"
        "Larmor=400.13\\\n"
        "Solvent=CDCl3\\\n"
        "1.18, T, J=7.0, L1\\\n"
        "3.69, Q, J=7.0, L2\\\n"
        "2.60, BS, , L3\\\n"
        "\n"
        # 1D ¹³C: broadband-decoupled, so all lines are S (singlet).
        ">  <NMREDATA_1D_13C>\n"
        "Larmor=100.61\\\n"
        "Solvent=CDCl3\\\n"
        "18.0, S, , L4\\\n"
        "58.0, S, , L5\\\n"
        "\n"
        "$$$$\n"
    )


def render_ethanol_nmredata() -> str:
    """Build the full NMReDATA-tagged .sdf string for ethanol."""
    mol_block, mol = _build_mol_block()
    atoms = _classify_atoms(mol)
    nmredata = _build_nmredata_block(atoms)
    # The mol block from RDKit ends with ``M  END`` followed by a
    # newline. NMReDATA tags follow immediately.
    return mol_block + "\n" + nmredata


def main():
    out_path = Path(
        sys.argv[1] if len(sys.argv) > 1 else "ethanol_nmredata.sdf"
    )
    out_path.write_text(render_ethanol_nmredata(), encoding="utf-8")
    print(f"wrote {out_path}")
    print()
    print("To verify: open this file in mnova and check whether:")
    print("  1. The ethanol structure displays correctly.")
    print("  2. Predicted shifts appear annotated on atoms or in a")
    print("     spectrum view (methyl ~1.18, methylene ~3.69, OH ~2.6).")
    print("  3. Clicking a peak highlights the corresponding atom on")
    print("     the structure (or vice versa).")


if __name__ == "__main__":
    raise SystemExit(main())
