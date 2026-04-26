"""ORCA-specific helpers shared across nodes that drive ORCA.

This module is the home for ORCA primitives that two or more nodes
need: the SMD solvent-name alias map, the simple-input file generator
(``%pal`` / ``%maxcore`` / ``%cpcm`` / ``! keywords`` / ``* xyzfile``
blocks), the ``FINAL SINGLE POINT ENERGY`` regex, and the
``orca.energies`` 3-column writer used by the array nodes.

What's NOT here (deliberate):

    * ``XyzBlock`` / ``split_multixyz`` / ``write_xyz_block`` — those
      are xyz primitives that live in :mod:`scripps_workflow.nodes.crest`
      (where they were first needed). orca-specific code re-exports
      them rather than redefining them.
    * The simple-input keyword strings (``GOAT`` etc.) — the
      orca_goat node uses a different simple-input shape (single-line
      ``! GOAT XTB`` + ``%goat`` block, no ``%cpcm`` SMD block) so it
      builds its own input. This module owns the *DFT/freq* shape with
      SMD solvation, which is what the array nodes need.

The HARTREE_TO_KCAL conversion factor is the CODATA 2018 value used in
ORCA's own thermochemistry section; it's exact enough that the tail
of the ``rel_kcal`` column in ``orca.energies`` matches what an
operator would compute by hand.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


#: 1 Eh in kcal/mol (CODATA 2018 / NIST). ORCA's internal
#: thermochemistry uses this same value, so per-conformer
#: ``rel_energy_kcal`` will match what ``orca.out`` reports to the last
#: digit.
HARTREE_TO_KCAL: float = 627.509474


# --------------------------------------------------------------------
# SMD solvent name normalization
# --------------------------------------------------------------------


#: User-facing solvent token → ORCA SMDsolvent name.
#:
#: ORCA's SMD library is case-insensitive but expects specific spellings
#: (``CH2Cl2`` not ``CH2CL2``, ``CHCl3`` not ``CHCL3``). The mapping
#: below catches the typo cases that would otherwise silently run in
#: vacuum because ORCA refuses to recognize the token. Keys are
#: lowercased; unknown tokens pass through unchanged so an operator
#: can specify any solvent ORCA happens to know about.
_SMD_ALIASES: dict[str, str] = {
    "ch2cl2": "CH2Cl2",
    "dcm": "CH2Cl2",
    "methylene_chloride": "CH2Cl2",
    "dichloromethane": "CH2Cl2",
    "chcl3": "CHCl3",
    "chloroform": "CHCl3",
    "thf": "THF",
    "dmf": "DMF",
    "dmso": "DMSO",
    "meoh": "methanol",
    "methanol": "methanol",
    "etoh": "ethanol",
    "ethanol": "ethanol",
    "h2o": "water",
    "water": "water",
    "toluene": "toluene",
    "benzene": "benzene",
    "acetonitrile": "acetonitrile",
    "mecn": "acetonitrile",
    "acetone": "acetone",
    "hexane": "hexane",
    "n_hexane": "hexane",
    "n-hexane": "hexane",
    "pyridine": "pyridine",
    "diethylether": "diethylether",
    "ether": "diethylether",
    "et2o": "diethylether",
}


def solvent_to_orca_smd(solvent: str) -> str:
    """Map a user-supplied solvent token to ORCA's SMDsolvent name.

    Falls through verbatim if the token isn't in the alias table — the
    caller (a :class:`Node`) is responsible for deciding whether to
    pre-validate against a whitelist; this function is only the
    spelling layer.
    """
    if solvent is None:
        raise ValueError("solvent_to_orca_smd called with None")
    s = str(solvent).strip()
    if not s:
        raise ValueError("solvent_to_orca_smd called with empty string")
    return _SMD_ALIASES.get(s.lower(), s)


# --------------------------------------------------------------------
# ORCA simple-input file (DFT / freq shape)
# --------------------------------------------------------------------


def make_orca_simple_input(
    *,
    keywords: str,
    nprocs: int,
    maxcore: int,
    charge: int,
    multiplicity: int,
    solvent: Optional[str],
    smd_solvent_override: Optional[str] = None,
    xyz_filename: str = "input.xyz",
) -> str:
    """Render the ``orca_*.inp`` text used by the array nodes.

    The output looks like::

        ! r2scan-3c TightSCF TightOpt

        %pal
          nprocs 16
        end

        %maxcore 4000

        %cpcm
          smd true
          SMDsolvent "CH2Cl2"
        end

        * xyzfile 0 1 input.xyz

    The leading ``!`` is added if the caller's ``keywords`` doesn't
    already start with it (operators commonly forget). The ``%cpcm``
    block is omitted when ``solvent is None`` (vacuum). When
    ``smd_solvent_override`` is provided it bypasses the alias table
    and is fed verbatim to SMDsolvent — handy for solvents whose ORCA
    spelling we haven't memorized.

    Args:
        keywords: ORCA simple-input keywords (e.g.
            ``"r2scan-3c TightSCF TightOpt"``). With or without
            leading ``!``.
        nprocs: ``%pal nprocs`` value.
        maxcore: ``%maxcore`` MB-per-process value.
        charge: net charge.
        multiplicity: spin multiplicity (``2S + 1``).
        solvent: SMD solvent token, or ``None`` for vacuum.
        smd_solvent_override: bypass the alias table and pass this
            string verbatim.
        xyz_filename: filename for ``* xyzfile``; relative to the ORCA
            run directory.
    """
    kw = str(keywords).strip()
    if kw.startswith("!"):
        kw = kw[1:].strip()
    if not kw:
        raise ValueError("make_orca_simple_input: keywords must be non-empty")

    lines: list[str] = []
    lines.append(f"! {kw}")
    lines.append("")
    lines.append("%pal")
    lines.append(f"  nprocs {int(nprocs)}")
    lines.append("end")
    lines.append("")
    lines.append(f"%maxcore {int(maxcore)}")
    lines.append("")

    if solvent is not None:
        smd_name = (
            smd_solvent_override
            if smd_solvent_override
            else solvent_to_orca_smd(solvent)
        )
        lines.append("%cpcm")
        lines.append("  smd true")
        lines.append(f'  SMDsolvent "{smd_name}"')
        lines.append("end")
        lines.append("")

    lines.append(f"* xyzfile {int(charge)} {int(multiplicity)} {xyz_filename}")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------
# Output parsing
# --------------------------------------------------------------------


#: Pattern matching ORCA's ``FINAL SINGLE POINT ENERGY  -123.456789``
#: line. We take the *last* match in the file because TightOpt /
#: numerical-Hessian runs print the line repeatedly and only the final
#: copy is the converged value.
FINAL_E_RE: re.Pattern[str] = re.compile(
    r"FINAL\s+SINGLE\s+POINT\s+ENERGY\s+(-?\d+\.\d+(?:[Ee][+-]?\d+)?)"
)


def parse_orca_final_energy(out_path: Path) -> Optional[float]:
    """Read an ``orca.out`` and return the last FINAL SINGLE POINT ENERGY.

    Returns ``None`` if the file is missing, unreadable, or contains no
    matches — callers treat that as "no energy" rather than raising,
    because partial outputs are common when SLURM kills a task.
    """
    try:
        text = Path(out_path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None

    matches = FINAL_E_RE.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except (ValueError, TypeError):
        return None


# --------------------------------------------------------------------
# Aggregation: ensemble.xyz + orca.energies
# --------------------------------------------------------------------


def concat_xyz_files(paths: list[Path], out_path: Path) -> None:
    """Concatenate per-conformer xyz files into a single multi-xyz blob.

    Each input is appended verbatim with a trailing newline if missing.
    Mirrors :func:`scripps_workflow.nodes.prism_screen.concat_xyz_files`
    so the array nodes can use a single import (this module) rather
    than reaching across to a sibling node module.
    """
    chunks: list[str] = []
    for p in paths:
        t = Path(p).read_text(encoding="utf-8", errors="replace")
        if not t.endswith("\n"):
            t += "\n"
        chunks.append(t)
    Path(out_path).write_text("".join(chunks), encoding="utf-8")


def write_energy_file(
    *,
    energies_h: list[Optional[float]],
    out_path: Path,
    rel_kcal_per_h: float = HARTREE_TO_KCAL,
) -> tuple[list[Optional[float]], Optional[float]]:
    """Write a 3-column ``orca.energies`` (index, abs_Eh, rel_kcal).

    Format::

           1   -123.456789012   0.000000
           2   -123.450000000   4.193471
           3      NaN           NaN

    The reference for the relative column is the *minimum* finite
    energy. Missing entries (``None``) are written as ``NaN`` in both
    columns. Returns ``(rel_kcal_list, e_min)`` so the caller can
    attach ``rel_energy_kcal`` to the per-conformer artifact records
    in one pass.

    The format intentionally matches the legacy ``orca_dft_opt_array``
    script byte-for-byte — downstream tooling (the marc /
    thermo_aggregate nodes) already grew a parser for that shape and
    this preserves backward compatibility.
    """
    finite = [e for e in energies_h if isinstance(e, float)]
    e_min = min(finite) if finite else None

    rel_kcal: list[Optional[float]] = []
    for e in energies_h:
        if e_min is None or e is None:
            rel_kcal.append(None)
        else:
            rel_kcal.append((e - e_min) * rel_kcal_per_h)

    lines: list[str] = []
    for i, (e, rk) in enumerate(zip(energies_h, rel_kcal), start=1):
        if e is None:
            lines.append(f"{i:6d}   NaN   NaN")
        elif rk is None:
            lines.append(f"{i:6d}   {e: .12f}   NaN")
        else:
            lines.append(f"{i:6d}   {e: .12f}   {rk: .6f}")

    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rel_kcal, e_min


__all__ = [
    "FINAL_E_RE",
    "HARTREE_TO_KCAL",
    "concat_xyz_files",
    "make_orca_simple_input",
    "parse_orca_final_energy",
    "solvent_to_orca_smd",
    "write_energy_file",
]
