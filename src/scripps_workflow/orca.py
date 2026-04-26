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


def make_orca_compound_input(
    *,
    keywords: str,
    singlepoint_keywords: Optional[str] = None,
    nprocs: int,
    maxcore: int,
    charge: int,
    multiplicity: int,
    solvent: Optional[str],
    smd_solvent_override: Optional[str] = None,
    xyz_filename: str = "input.xyz",
) -> str:
    """Render a single- or two-job ORCA input.

    When ``singlepoint_keywords`` is ``None`` (or empty after a strip)
    the output is identical to :func:`make_orca_simple_input` — a
    single ORCA job. When provided, two jobs are concatenated with a
    ``$new_job`` separator::

        ! r2scan-3c TightSCF Freq
        ...
        * xyzfile 0 1 input.xyz

        $new_job

        ! wB97M-V def2-TZVPP TightSCF
        ...
        * xyzfile 0 1 input.xyz

    ORCA runs the jobs sequentially, sharing the same SCF guess strategy
    but with fresh basis/functional setup for each. The geometry is
    re-specified from the same xyz for both jobs (the freq job doesn't
    optimize, so the coordinates are unchanged anyway — but explicit is
    safer than relying on ORCA's inheritance behavior).

    This is the standard "low-level thermo + high-level single point"
    composite protocol used to compute Gibbs energies at a level the
    Hessian itself isn't affordable at. Downstream tooling (the
    ``thermo_aggregate`` node) reads ``G - E(el)`` from the *first*
    job's freq block and ``FINAL SINGLE POINT ENERGY`` from the last
    occurrence in the file — the SP — so the composite Gibbs is
    naturally::

        G_composite = E_SP_high + (G - E_el)_low

    Note ``parse_orca_final_energy`` returns the LAST FINAL E match by
    design, so callers parsing energies don't need to know whether the
    file is single- or two-job.

    Args:
        keywords: First-job ``!`` line (e.g. the freq calc).
        singlepoint_keywords: Second-job ``!`` line, or ``None`` to
            render a single-job input. Empty/whitespace-only is
            treated as ``None``.
        nprocs / maxcore / charge / multiplicity / solvent /
        smd_solvent_override / xyz_filename:
            Forwarded to both jobs verbatim.
    """
    job1 = make_orca_simple_input(
        keywords=keywords,
        nprocs=nprocs,
        maxcore=maxcore,
        charge=charge,
        multiplicity=multiplicity,
        solvent=solvent,
        smd_solvent_override=smd_solvent_override,
        xyz_filename=xyz_filename,
    )
    sp = (singlepoint_keywords or "").strip()
    if not sp:
        return job1

    job2 = make_orca_simple_input(
        keywords=sp,
        nprocs=nprocs,
        maxcore=maxcore,
        charge=charge,
        multiplicity=multiplicity,
        solvent=solvent,
        smd_solvent_override=smd_solvent_override,
        xyz_filename=xyz_filename,
    )
    # ``make_orca_simple_input`` ends with a trailing newline (the xyz
    # line). Insert a blank line, then ``$new_job``, another blank, and
    # the second job — keeps the file readable when an operator opens
    # it.
    return job1 + "\n$new_job\n\n" + job2


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


#: Pattern for ORCA's "ORCA TERMINATED NORMALLY" footer. ORCA prints
#: this exact phrase only on a clean exit; SCF blowups, geometry
#: failures, and SLURM-killed jobs all leave it absent. Case-insensitive
#: because some legacy builds emit lowercase variants.
ORCA_NORMAL_RE: re.Pattern[str] = re.compile(
    r"ORCA\s+TERMINATED\s+NORMALLY", re.IGNORECASE
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


def orca_terminated_normally(
    out_path: Path, *, tail_bytes: int = 16384
) -> bool:
    """Check whether an ``orca.out`` ends with the normal-termination footer.

    Used by the thermo-array node to detect partial/killed runs that
    nonetheless produced a FINAL E line earlier (SLURM walltime kills
    after the SCF but before the Freq finishes are the typical culprit).

    Reads only the file's tail (default 16 KiB) — ORCA's freq output
    can be tens of MB so a full read isn't worth it for one regex.
    Returns ``False`` for missing / unreadable files, just like
    :func:`parse_orca_final_energy`.
    """
    try:
        with Path(out_path).open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - int(tail_bytes)), 0)
            data = f.read()
    except Exception:
        return False
    text = data.decode("utf-8", errors="replace")
    return ORCA_NORMAL_RE.search(text) is not None


# --------------------------------------------------------------------
# Thermochemistry block parsing
# --------------------------------------------------------------------


#: Pattern matching ORCA's ``G-E(el)            ...    -76.345 Eh`` line in
#: the thermochemistry block. This is the "thermal correction to electronic
#: energy" — the piece that, when added to a high-level SP energy, gives
#: the composite Gibbs::
#:
#:     G_composite = E_SP_high + (G - E_el)_low
G_MINUS_E_EL_RE: re.Pattern[str] = re.compile(
    r"G-E\(el\)\s+\.{3}\s+(-?\d+\.\d+(?:[Ee][+-]?\d+)?)\s+Eh"
)


#: Pattern matching ORCA's ``Final Gibbs free energy   ...   -76.123 Eh``
#: line (low-level total Gibbs). Used as a fallback when the high-level SP
#: energy is missing — the aggregator can still publish *something*.
FINAL_G_RE: re.Pattern[str] = re.compile(
    r"Final\s+Gibbs\s+free\s+energy\s+\.{3}\s+(-?\d+\.\d+(?:[Ee][+-]?\d+)?)\s+Eh",
    re.IGNORECASE,
)


#: Pattern matching ``Total enthalpy   ...   -76.345 Eh``.
TOTAL_H_RE: re.Pattern[str] = re.compile(
    r"Total\s+enthalpy\s+\.{3}\s+(-?\d+\.\d+(?:[Ee][+-]?\d+)?)\s+Eh",
    re.IGNORECASE,
)


#: Pattern matching ``Total entropy correction   ...   -0.034 Eh``. Note
#: ORCA labels this as -T*S so it is already in energy units, not entropy.
S_CORR_RE: re.Pattern[str] = re.compile(
    r"Total\s+entropy\s+correction\s+\.{3}\s+(-?\d+\.\d+(?:[Ee][+-]?\d+)?)\s+Eh",
    re.IGNORECASE,
)


#: Pattern matching ``THERMOCHEMISTRY AT T = 298.15 K``. Used to read back
#: the exact temperature ORCA computed thermo at, so the aggregator can
#: warn if the operator's requested ``temperature_k`` doesn't match.
THERMO_T_RE: re.Pattern[str] = re.compile(
    r"THERMOCHEMISTRY\s+AT\s+T\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*K",
    re.IGNORECASE,
)


def _last_float(regex: re.Pattern[str], text: str) -> Optional[float]:
    """Return the LAST regex match converted to float, or None.

    Used for thermo parsing where compound (freq + SP) outputs may print
    overlapping markers; the last one is the value computed in the final
    block, which is what the aggregator wants.
    """
    matches = regex.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except (ValueError, TypeError):
        return None


def parse_orca_thermochem(out_path: Path) -> dict[str, Optional[float]]:
    """Extract thermochemistry numbers from an ``orca.out``.

    Returns a dict with the following keys (any value may be ``None`` if
    the marker is missing):

        * ``final_sp_energy_eh`` — last ``FINAL SINGLE POINT ENERGY``.
        * ``g_minus_e_el_eh``   — last ``G-E(el)`` thermal correction.
        * ``final_g_eh``        — last ``Final Gibbs free energy``.
        * ``total_h_eh``        — last ``Total enthalpy``.
        * ``total_entropy_corr_eh`` — last ``Total entropy correction``.
        * ``temperature_k``     — last ``THERMOCHEMISTRY AT T = ... K``.

    On read errors all values are returned as ``None``. This is the
    parsing layer used by ``thermo_aggregate``; it is intentionally
    silent on missing markers — the aggregator decides which combinations
    of missing values constitute a hard failure.
    """
    try:
        text = Path(out_path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return {
            "final_sp_energy_eh": None,
            "g_minus_e_el_eh": None,
            "final_g_eh": None,
            "total_h_eh": None,
            "total_entropy_corr_eh": None,
            "temperature_k": None,
        }
    return {
        "final_sp_energy_eh": _last_float(FINAL_E_RE, text),
        "g_minus_e_el_eh": _last_float(G_MINUS_E_EL_RE, text),
        "final_g_eh": _last_float(FINAL_G_RE, text),
        "total_h_eh": _last_float(TOTAL_H_RE, text),
        "total_entropy_corr_eh": _last_float(S_CORR_RE, text),
        "temperature_k": _last_float(THERMO_T_RE, text),
    }


def classify_orca_outfile(out_path: Path) -> tuple[bool, bool]:
    """Inspect an ``orca.out`` and report ``(has_thermo, has_final_e)``.

    ``has_thermo`` is True iff the file contains any of the canonical
    thermochemistry markers (``GIBBS FREE ENERGY``, ``G-E(el)``, or
    ``THERMOCHEMISTRY``). ``has_final_e`` is True iff the file contains a
    ``FINAL SINGLE POINT ENERGY`` line.

    A compound (freq + high-level SP) output has BOTH True; a pure SP
    output has only ``has_final_e``. Used by :func:`pick_orca_outputs` to
    pair the right thermo / SP files when a task dir contains multiple.
    Returns ``(False, False)`` on read errors.
    """
    try:
        text = Path(out_path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return False, False
    has_thermo = (
        "GIBBS FREE ENERGY" in text
        or "G-E(el)" in text
        or "THERMOCHEMISTRY" in text
    )
    has_final_e = "FINAL SINGLE POINT ENERGY" in text
    return has_thermo, has_final_e


def pick_orca_outputs(
    task_dir: Path, *, glob: str = "*.out"
) -> tuple[Optional[Path], Optional[Path]]:
    """Choose the (thermo_out, sp_out) pair for a task directory.

    Selection rules:

        * If a single ``*.out`` exists it is used for both (the compound
          freq + SP layout — :mod:`scripps_workflow.nodes.orca_thermo_array`
          writes exactly this shape).
        * If multiple ``*.out`` exist, prefer a file with thermo markers
          for the thermo slot and a file with FINAL E *and no thermo
          markers* for the SP slot. This matches the legacy
          ``orca_thermo_aggregator`` behavior, which expected separate
          freq + SP files when present.
        * If only a thermo file exists (no separate SP), the thermo file
          is reused as the SP source — its ``FINAL SINGLE POINT ENERGY``
          is the low-level SCF energy and downstream code degrades to
          ``G_composite = G_low``.
        * If only an SP file exists, ``thermo_out`` falls back to the SP
          file so the aggregator can attempt to parse — typically yields
          ``g_minus_e_el_eh = None``, which the aggregator surfaces.
        * Returns ``(None, None)`` when ``task_dir`` has no ``*.out``
          files at all.

    The selection is deterministic: candidates are sorted by name, and
    ties go to the LAST in sorted order (later runs / longer compound
    files win when an operator iterates).
    """
    if not task_dir.is_dir():
        return None, None

    outs = sorted(task_dir.glob(glob))
    if not outs:
        return None, None

    thermo_candidates: list[Path] = []
    sp_pure_candidates: list[Path] = []
    sp_fallback: list[Path] = []
    for p in outs:
        has_thermo, has_final_e = classify_orca_outfile(p)
        if has_thermo:
            thermo_candidates.append(p)
        if has_final_e and not has_thermo:
            sp_pure_candidates.append(p)
        if has_final_e:
            sp_fallback.append(p)

    thermo_out = thermo_candidates[-1] if thermo_candidates else None
    sp_out: Optional[Path]
    if sp_pure_candidates:
        sp_out = sp_pure_candidates[-1]
    elif sp_fallback:
        sp_out = sp_fallback[-1]
    else:
        sp_out = None

    if thermo_out is None and sp_out is not None:
        thermo_out = sp_out
    if sp_out is None and thermo_out is not None:
        sp_out = thermo_out

    # File(s) exist but classify found neither thermo nor FINAL E in any
    # of them — pick the last one as both slots so the aggregator can
    # parse, fail in a structured way, and surface
    # ``missing_thermochem_or_energy_for_G``. Returning ``(None, None)``
    # here would conflate "no files at all" with "files present but
    # unparseable".
    if thermo_out is None and sp_out is None:
        thermo_out = sp_out = outs[-1]

    return thermo_out, sp_out


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
    "FINAL_G_RE",
    "G_MINUS_E_EL_RE",
    "HARTREE_TO_KCAL",
    "ORCA_NORMAL_RE",
    "S_CORR_RE",
    "THERMO_T_RE",
    "TOTAL_H_RE",
    "classify_orca_outfile",
    "concat_xyz_files",
    "make_orca_compound_input",
    "make_orca_simple_input",
    "orca_terminated_normally",
    "parse_orca_final_energy",
    "parse_orca_thermochem",
    "pick_orca_outputs",
    "solvent_to_orca_smd",
    "write_energy_file",
]
