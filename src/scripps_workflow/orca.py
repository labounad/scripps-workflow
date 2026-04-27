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
from typing import Any, Optional


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
    extra_blocks: Optional[list[str]] = None,
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

    # Caller-supplied raw blocks (``%method``, ``%eprnmr`` for NMR
    # shieldings/couplings, custom ``%basis``, etc.). Each block is
    # written verbatim with one trailing blank line so an operator
    # opening the .inp can read it. Empty/None entries are skipped.
    #
    # ``%eprnmr`` blocks must appear AFTER the geometry input because
    # their ``Nuclei = all H``-style selectors are resolved against the
    # parsed atom list at read time — putting ``%eprnmr`` before the
    # geometry trips ORCA with::
    #
    #     Error in [EPRNMR] block: nuclear properties are requested
    #     but no coordinates have been read!
    #
    # Auto-partition extra_blocks: anything starting with ``%eprnmr``
    # is deferred to post-xyz, everything else stays pre-xyz where the
    # rest of ORCA's declarative blocks live. Idempotent and order-
    # preserving within each partition.
    pre_xyz: list[str] = []
    post_xyz: list[str] = []
    for block in extra_blocks or []:
        text = (block or "").strip()
        if not text:
            continue
        if text.lstrip().lower().startswith("%eprnmr"):
            post_xyz.append(text)
        else:
            pre_xyz.append(text)

    for text in pre_xyz:
        lines.append(text)
        lines.append("")

    lines.append(f"* xyzfile {int(charge)} {int(multiplicity)} {xyz_filename}")
    lines.append("")

    for text in post_xyz:
        lines.append(text)
        lines.append("")

    return "\n".join(lines)


#: Composite "3c" methods (r2SCAN-3c, B97-3c, HF-3c, PBEh-3c, B3LYP-3c,
#: ωB97X-3c). These bake D3/D4 dispersion + gCP geometric counterpoise
#: into the SCF, and ORCA propagates those ``%method`` settings into
#: every subsequent ``$new_job`` block. If the next job uses a
#: nonlocal-VV10 functional (``wB97M-V``, ``wB97X-V``, ``B97M-V``, ...)
#: ORCA hard-aborts with::
#:
#:     DFT-NL dispersion correction can not be applied together with D3/D4!
#:
#: To break the leak we inject ``%method DFTDOPT 0; DoGCP false; end``
#: at the top of every post-job when the *first* job is a 3c composite.
#: See :func:`_uses_3c_composite_method` and the
#: ``reset_3c_dispersion_in_post_jobs`` knob below.
_3C_COMPOSITE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:r2scan|b97|hf|pbeh|b3lyp|wb97x|ωb97x)-3c(?![A-Za-z0-9])",
    re.IGNORECASE,
)

_DISPERSION_RESET_BLOCK = (
    "%method\n"
    "  DFTDOPT 0\n"
    "  DoGCP false\n"
    "end"
)


def _uses_3c_composite_method(keywords: str) -> bool:
    """True if ``keywords`` names a composite ``*-3c`` ORCA method.

    Used to decide whether to prepend a dispersion/gCP reset block to
    each post-job in :func:`make_orca_compound_input` so a leaking
    ``%method DFTDOPT/DoGCP`` doesn't collide with VV10-type
    nonlocal-dispersion functionals downstream.
    """
    return bool(_3C_COMPOSITE_PATTERN.search(keywords or ""))


def make_orca_compound_input(
    *,
    keywords: str,
    singlepoint_keywords: Optional[str] = None,
    post_jobs: Optional[list[dict[str, Any]]] = None,
    nprocs: int,
    maxcore: int,
    charge: int,
    multiplicity: int,
    solvent: Optional[str],
    smd_solvent_override: Optional[str] = None,
    xyz_filename: str = "input.xyz",
    reset_3c_dispersion_in_post_jobs: bool = True,
) -> str:
    """Render a chain of ORCA jobs separated by ``$new_job``.

    Three calling shapes, in increasing flexibility:

    1. **Plain freq** — pass only ``keywords``. Output is identical to
       :func:`make_orca_simple_input`.

    2. **Freq + one SP** — pass ``keywords`` (the freq line) and
       ``singlepoint_keywords`` (the high-level SP line). The SP is
       rendered as a single ``$new_job`` block. This is the classic
       "low-level thermo + high-level single point" recipe that
       :mod:`scripps_workflow.nodes.thermo_aggregate` consumes::

            G_composite = E_SP_high + (G - E_el)_low

    3. **Freq + arbitrary chain** — pass ``post_jobs`` instead of (or
       in addition to) ``singlepoint_keywords``. Each entry is a dict
       with at least a ``keywords`` field plus an optional
       ``extra_blocks`` list and ``kind`` label. Each post-job becomes
       its own ``$new_job`` segment. Used by the NMR pipeline to chain
       freq + high-level SP + 1H shieldings + 13C shieldings + J
       couplings inside ONE allocated SLURM job.

    When both ``singlepoint_keywords`` and ``post_jobs`` are set, the
    SP entry is prepended to ``post_jobs`` so the SP comes first
    (matches the legacy thermo aggregator's last-FINAL-E convention).

    Each post-job inherits ``nprocs``/``maxcore``/``charge``/
    ``multiplicity``/``solvent`` from the parent call — different
    methods/basis sets/functionals are conveyed via the per-job
    ``keywords`` and ``extra_blocks`` (e.g. ``%method`` or
    ``%eprnmr`` blocks for custom functionals or NMR settings).

    Args:
        keywords: First-job ``!`` line (typically the freq calc).
        singlepoint_keywords: Optional shorthand for "one SP job after
            the freq". Empty/whitespace/sentinel-string treated as
            ``None``.
        post_jobs: Optional list of post-freq job specs. Each entry:

            * ``keywords`` (required): the ``!`` line for the job.
            * ``extra_blocks`` (optional): list of raw block strings
              appended after ``%cpcm`` and before ``* xyzfile`` (e.g.
              ``"%eprnmr\\n  Nuclei = all H { shift }\\nend"``).
            * ``kind`` (optional): label-only, ignored at render time
              but useful for the manifest's ``inputs`` echo.

        nprocs / maxcore / charge / multiplicity / solvent /
        smd_solvent_override / xyz_filename:
            Forwarded to every job verbatim.
        reset_3c_dispersion_in_post_jobs: When the first job uses a
            composite ``*-3c`` method (which silently enables D3/D4 +
            gCP via ``%method``), prepend a ``%method DFTDOPT 0; DoGCP
            false; end`` block to every post-job's ``extra_blocks``.
            Without this, ORCA aborts the second job with "DFT-NL
            dispersion correction can not be applied together with
            D3/D4" whenever that job uses a nonlocal-VV10 functional
            (``wB97M-V``, ``wB97X-V``, ``B97M-V``, ...). Defaults to
            ``True`` and is a no-op when the first job isn't a 3c
            composite. Set to ``False`` to suppress the injection if
            you're hand-tuning the chain.
    """
    # Normalize the call shape into a single list of jobs.
    jobs: list[dict[str, Any]] = []
    sp = (singlepoint_keywords or "").strip()
    if sp:
        jobs.append({"keywords": sp, "extra_blocks": None, "kind": "energy"})
    if post_jobs:
        for j in post_jobs:
            kw = str(j.get("keywords", "")).strip()
            if not kw:
                raise ValueError(
                    f"post_jobs entry missing 'keywords': {j!r}"
                )
            jobs.append(
                {
                    "keywords": kw,
                    "extra_blocks": j.get("extra_blocks") or None,
                    "kind": j.get("kind"),
                }
            )

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
    if not jobs:
        return job1

    # Decide once: does the first job leak D3/D4 + gCP via %method?
    # If so, every post-job needs the reset block prepended (idempotent —
    # skipped if a caller already supplied an explicit %method override).
    inject_reset = (
        reset_3c_dispersion_in_post_jobs
        and _uses_3c_composite_method(keywords)
    )

    chunks: list[str] = [job1]
    for spec in jobs:
        spec_extra = list(spec.get("extra_blocks") or [])
        if inject_reset and not _has_dispersion_directives(spec_extra):
            spec_extra.insert(0, _DISPERSION_RESET_BLOCK)
        block = make_orca_simple_input(
            keywords=spec["keywords"],
            nprocs=nprocs,
            maxcore=maxcore,
            charge=charge,
            multiplicity=multiplicity,
            solvent=solvent,
            smd_solvent_override=smd_solvent_override,
            xyz_filename=xyz_filename,
            extra_blocks=spec_extra or None,
        )
        # ``make_orca_simple_input`` ends with a trailing newline (the
        # xyz line). One blank line + ``$new_job`` + blank, then the
        # next block — keeps the file readable.
        chunks.append("\n$new_job\n\n" + block)
    return "".join(chunks)


def _has_dispersion_directives(extra_blocks: list[str]) -> bool:
    """True if any ``extra_blocks`` entry already sets DFTDOPT or DoGCP.

    Used to make the 3c-dispersion-reset injection idempotent at the
    *directive* level rather than the block level. Two ``%method``
    blocks in a single ORCA input are legal (settings stack), so the
    earlier "skip if any %method block exists" rule was too coarse —
    it suppressed the reset whenever a caller added an unrelated
    ``%method`` block (e.g. functional re-mixing for WP04). We now
    only skip the auto-injection when the caller has already taken
    explicit ownership of the two specific directives this helper
    sets, which is the correct invariant.
    """
    for raw in extra_blocks:
        text = (raw or "").lower()
        if not text.lstrip().startswith("%method"):
            continue
        if "dftdopt" in text or "dogcp" in text:
            return True
    return False


# --------------------------------------------------------------------
# Functional alias resolution
# --------------------------------------------------------------------


#: ``ScalHFX``/``ScalDFX``/``ScalLDAC``/``ScalGGAC`` mixing parameters
#: for the WP04 functional (Wiitala, Cramer, Hoye, J. Chem. Theory
#: Comput. 2006, 2, 1085). WP04 is the cheshire-recommended functional
#: for ¹H GIAO shieldings; ORCA doesn't ship a built-in keyword for it,
#: so we re-mix B3LYP/G via ``%method``. Numbers verified against the
#: original publication (a₀=0.1161, ax=0.8839, ac(LYP)=0.82,
#: ac(VWN)=0.18).
_WP04_METHOD_BLOCK = (
    "%method\n"
    "  ScalHFX  0.1161\n"
    "  ScalDFX  0.8839\n"
    "  ScalLDAC 0.1800\n"
    "  ScalGGAC 0.8200\n"
    "end"
)


def resolve_functional_alias(method: str) -> tuple[str, list[str]]:
    """Translate a non-native functional name into ``(orca_keyword, extra_blocks)``.

    Some calibration-table functionals — most importantly **WP04** —
    aren't ORCA simple-input keywords. They're parametric tweaks of
    standard hybrids that need a ``%method`` block to define. ORCA
    aborts with::

        UNRECOGNIZED OR DUPLICATED KEYWORD(S) IN SIMPLE INPUT LINE
          WP04

    when fed the bare name. This helper takes the calibration-table
    label (case-insensitive) and returns the actual ORCA keyword to
    drop on the ``!`` line plus any ``%method``/``%basis``/etc. blocks
    that need to accompany it. Returns ``(method, [])`` unchanged when
    the input is already an ORCA-native keyword.

    Currently handled aliases:

    * ``WP04`` → ``B3LYP/G`` + ``%method ScalHFX/ScalDFX/ScalLDAC/
      ScalGGAC ...``. Wiitala–Cramer–Hoye 2006 reparametrization of
      B3LYP/G; standard cheshire ¹H NMR functional.

    * ``wB97X-D`` → ``wB97X-D3``. ORCA 6 dropped the bare ``wB97X-D``
      simple-input keyword (a stand-in for Chai–Head-Gordon 2008,
      which used D2-style dispersion). The closest in-tree analog is
      ``wB97X-D3`` (zero-damping D3). Calibrations fit against the
      original "wB97X-D" are typically transferable since the
      functional form is identical and only the dispersion kernel
      changed; if you need exact-match accuracy, swap to ``wB97X-V``
      and refit the calibration.

    Add new entries here as the lab adopts other custom functionals.
    """
    m = (method or "").strip()
    low = m.lower()
    if low == "wp04":
        return ("B3LYP/G", [_WP04_METHOD_BLOCK])
    if low == "wb97x-d":
        return ("wB97X-D3", [])
    return (m, [])


# --------------------------------------------------------------------
# NMR-specific %eprnmr block helpers
# --------------------------------------------------------------------


def nmr_shielding_block(nuclei: str) -> str:
    """Render a ``%eprnmr`` block requesting GIAO chemical shieldings.

    ``nuclei`` is an ORCA nucleus selector — ``"all H"``, ``"all C"``,
    or a specific atom-index list like ``"1, 4, 7"``. ORCA's ``! NMR``
    keyword automatically selects GIAO; this block just narrows which
    atoms get shieldings printed.

    Example::

        %eprnmr
          Nuclei = all H { shift }
        end
    """
    sel = str(nuclei).strip()
    if not sel:
        raise ValueError("nmr_shielding_block: nuclei selector cannot be empty")
    return (
        "%eprnmr\n"
        f"  Nuclei = {sel} {{ shift }}\n"
        "end"
    )


def nmr_coupling_block(
    nuclei_pairs: list[str],
    *,
    ssall: bool = True,
    spinspin_thresh: Optional[float] = 8.0,
) -> str:
    """Render a ``%eprnmr`` block requesting indirect spin-spin couplings.

    ``ssall=True`` enables the full Ramsey decomposition (FC + SD +
    PSO + DSO) — required for accurate J's, especially 1H-1H couplings
    where FC dominates but SD/PSO contribute meaningfully. The default
    ORCA setup is FC-only, which is wrong for production use.

    ``spinspin_thresh`` (in Å) caps the inter-nucleus distance for
    which couplings are computed. 8 Å is a practical default that
    catches all chemically-relevant couplings while skipping the
    O(N²) work for distant pairs.

    ``nuclei_pairs`` is a list of selectors (e.g.
    ``["all H", "all C"]``). ORCA computes couplings between any pair
    of selected nuclei.

    Example output::

        %eprnmr
          Nuclei = all H { ssall }
          Nuclei = all C { ssall }
          SpinSpinRThresh 8.0
        end
    """
    if not nuclei_pairs:
        raise ValueError("nmr_coupling_block: nuclei_pairs must be non-empty")
    term = "ssall" if ssall else "ssfc"
    lines: list[str] = ["%eprnmr"]
    for sel in nuclei_pairs:
        s = str(sel).strip()
        if not s:
            continue
        lines.append(f"  Nuclei = {s} {{ {term} }}")
    if spinspin_thresh is not None:
        lines.append(f"  SpinSpinRThresh {float(spinspin_thresh)}")
    lines.append("end")
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


# --------------------------------------------------------------------
# NMR output parsing
# --------------------------------------------------------------------


#: Marker that opens an ORCA "CHEMICAL SHIELDING SUMMARY" table. Match
#: case-insensitively because legacy ORCA versions print mixed case.
SHIELDING_HDR_RE: re.Pattern[str] = re.compile(
    r"CHEMICAL\s+SHIELDING\s+SUMMARY", re.IGNORECASE
)


#: Per-row pattern: ``  0  C   175.234   12.345`` — atom index, element
#: symbol, isotropic shielding (ppm), anisotropy (ppm). The element is
#: 1–2 letters; numbers may be negative or in scientific notation.
SHIELDING_ROW_RE: re.Pattern[str] = re.compile(
    r"^\s*(\d+)\s+([A-Z][a-z]?)\s+"
    r"(-?\d+\.\d+(?:[Ee][+-]?\d+)?)\s+"
    r"(-?\d+\.\d+(?:[Ee][+-]?\d+)?)\s*$"
)


#: Per-pair "Nucleus A:" header in modern ORCA 6 output. The detail
#: blocks for spin-spin couplings emit ``Nucleus A:`` and ``Nucleus B:``
#: on TWO consecutive lines, each carrying ``<atom_index> <element>``
#: (one-based or zero-based depending on ORCA build — we capture the
#: integer verbatim and the consumer is responsible for the offset).
#:
#: Example::
#:
#:      Nucleus A: 0 C
#:      Nucleus B: 1 H
#:
#: We also accept the variant where element comes first
#: (``Nucleus A: C 0``) — older builds and some print levels swap order.
COUPLING_NUCLEUS_RE: re.Pattern[str] = re.compile(
    r"\bNucleus\s+([AB])\s*[:=]\s*"
    r"(?:(\d+)\s+([A-Z][a-z]?)|([A-Z][a-z]?)\s+(\d+))",
    re.IGNORECASE,
)


#: Legacy single-line header retained for backward compatibility with
#: pre-ORCA-6 outputs that emitted both nuclei on one comma-separated
#: line. New parser will fall through to :data:`COUPLING_NUCLEUS_RE` when
#: this doesn't match.
COUPLING_PAIR_RE: re.Pattern[str] = re.compile(
    r"NUCLEUS\s+A\s*=\s*(\d+)\s+([A-Z][a-z]?)\s*,\s*"
    r"NUCLEUS\s+B\s*=\s*(\d+)\s+([A-Z][a-z]?)",
    re.IGNORECASE,
)


def parse_orca_shieldings(out_path: Path) -> list[dict[str, Any]]:
    """Parse the ``CHEMICAL SHIELDING SUMMARY`` table from an ORCA out.

    Returns a list of dicts ordered by atom index (ORCA's intrinsic
    order, which is the input-geometry order)::

        [{"atom_index": 0, "element": "C", "sigma_iso_ppm": 175.234,
          "sigma_aniso_ppm": 12.345}, ...]

    If a compound (multi-job) output contains MULTIPLE shielding
    summaries (e.g. one per ``$new_job`` block), the parser collects
    rows from every block and de-duplicates by ``atom_index`` keeping
    the LAST occurrence — so the final job's table wins. This matches
    the convention used by the energy parsers.

    Returns ``[]`` on read errors or when no summary marker is found.
    """
    try:
        text = Path(out_path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []

    # Find every "CHEMICAL SHIELDING SUMMARY" block; iterate through
    # the lines that follow it until we hit a blank or a clearly
    # non-row line (e.g. another header).
    rows_by_idx: dict[int, dict[str, Any]] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if SHIELDING_HDR_RE.search(lines[i]):
            i += 1
            # Walk forward, skipping ORCA's separator/sub-header lines
            # (composed of dashes, blank, or non-numeric column titles).
            saw_data = False
            while i < len(lines):
                m = SHIELDING_ROW_RE.match(lines[i])
                if m:
                    saw_data = True
                    idx = int(m.group(1))
                    rows_by_idx[idx] = {
                        "atom_index": idx,
                        "element": m.group(2),
                        "sigma_iso_ppm": float(m.group(3)),
                        "sigma_aniso_ppm": float(m.group(4)),
                    }
                    i += 1
                    continue
                # Stop the inner loop once we exit the data rows. Allow
                # a few non-matching lines BEFORE the data starts (header
                # separators); once we've started seeing data, the first
                # non-match ends the block.
                if saw_data:
                    break
                # Bail if we wandered into a completely unrelated section.
                if "SUMMARY" in lines[i].upper() and not SHIELDING_HDR_RE.search(lines[i]):
                    break
                i += 1
        else:
            i += 1

    return [rows_by_idx[k] for k in sorted(rows_by_idx)]


#: Patterns for the per-term lines inside a single pair block.
#:
#: ORCA 6 prints spelled-out term names with a ``:`` separator and
#: trailing ``Hz`` unit, e.g.::
#:
#:     Fermi contact contribution                 :  143.456 Hz
#:     Spin-dipolar contribution                  :    0.123 Hz
#:     Paramagnetic contribution                  :    1.234 Hz
#:     Diamagnetic contribution                   :    0.310 Hz
#:     Spin-dipolar/Fermi-contact cross term      :    0.001 Hz
#:     Total                                      :  145.124 Hz
#:
#: Some older builds use bare abbreviations (``FC = ...``); each
#: pattern below matches either spelling. The ``Total`` pattern
#: deliberately requires word-boundary anchors so a "Total" appearing
#: inside another label doesn't match.
_COUPLING_TERM_PATTERNS: dict[str, re.Pattern[str]] = {
    "J_total_hz": re.compile(
        r"(?:^|\s)(?:Total(?:\s+(?:iso(?:tropic)?|J|coupling))?|TOTAL\s+JISO)\b"
        r"[^=:]*[=:]\s*(-?\d+\.\d+(?:[Ee][+-]?\d+)?)",
        re.IGNORECASE,
    ),
    "J_FC_hz": re.compile(
        r"(?:Fermi[-\s]contact(?:\s+contribution)?|\bFC\b)"
        r"[^=:]*[=:]\s*(-?\d+\.\d+(?:[Ee][+-]?\d+)?)",
        re.IGNORECASE,
    ),
    "J_SD_hz": re.compile(
        r"(?:Spin[-\s]dipolar(?:\s+contribution)?|\bSD\b)"
        r"[^=:]*[=:]\s*(-?\d+\.\d+(?:[Ee][+-]?\d+)?)",
        re.IGNORECASE,
    ),
    "J_PSO_hz": re.compile(
        r"(?:Paramagnetic(?:\s+(?:spin[-\s]orbit|contribution))?"
        r"|Para(?:magnetic)?\s+SO|\bPSO\b)"
        r"[^=:]*[=:]\s*(-?\d+\.\d+(?:[Ee][+-]?\d+)?)",
        re.IGNORECASE,
    ),
    "J_DSO_hz": re.compile(
        r"(?:Diamagnetic(?:\s+(?:spin[-\s]orbit|contribution))?"
        r"|Dia(?:magnetic)?\s+SO|\bDSO\b)"
        r"[^=:]*[=:]\s*(-?\d+\.\d+(?:[Ee][+-]?\d+)?)",
        re.IGNORECASE,
    ),
}

#: The cross term (Spin-dipolar/Fermi-contact) is small in magnitude but
#: ORCA reports it; we capture it so synthesized totals (FC+SD+PSO+DSO+
#: cross) match the printed Total exactly. It isn't surfaced in the
#: per-pair record by default — the consumer rarely needs it — but it
#: participates in the synthesized-total fallback below.
_COUPLING_CROSS_RE: re.Pattern[str] = re.compile(
    r"(?:Spin[-\s]dipolar/Fermi[-\s]contact(?:\s+cross\s+term)?|\bSD/FC\b)"
    r"[^=:]*[=:]\s*(-?\d+\.\d+(?:[Ee][+-]?\d+)?)",
    re.IGNORECASE,
)


def _parse_nucleus_line(
    line: str,
) -> Optional[tuple[str, int, str]]:
    """Parse a ``Nucleus A: <idx> <El>`` style line.

    Returns ``(side, atom_index, element)`` where ``side`` is ``"A"``
    or ``"B"``, or ``None`` if the line isn't a nucleus header.
    Accepts both ``<idx> <El>`` and ``<El> <idx>`` orderings.
    """
    m = COUPLING_NUCLEUS_RE.search(line)
    if not m:
        return None
    side = m.group(1).upper()
    if m.group(2) is not None:
        idx = int(m.group(2))
        el = m.group(3)
    else:
        idx = int(m.group(5))
        el = m.group(4)
    return side, idx, el


def parse_orca_couplings(out_path: Path) -> list[dict[str, Any]]:
    """Parse spin-spin coupling per-pair blocks from an ORCA out.

    Returns a list of dicts, one per (i, j) nucleus pair::

        [{"i": 0, "elem_i": "C", "j": 1, "elem_j": "H",
          "J_total_hz": 145.123,
          "J_FC_hz": 143.456, "J_SD_hz": 0.123,
          "J_PSO_hz": 1.234, "J_DSO_hz": 0.310}, ...]

    Pairs are de-duplicated on (min(i,j), max(i,j)). Multiple blocks
    (compound output) merge with last-occurrence-wins. Missing Ramsey
    terms are returned as ``None``.

    Two header formats are supported:

        * **Modern ORCA 6** — the per-pair block opens with two
          consecutive lines::

              Nucleus A: 0 C
              Nucleus B: 1 H

          (or with element/index swapped, or with ``=`` instead of
          ``:``). The parser pairs an ``A`` with the next ``B`` it
          finds within a small look-ahead window.

        * **Legacy** — a single ``NUCLEUS A = idx el, NUCLEUS B = idx
          el`` line. Retained so older runs still parse.

    For each identified pair, the next ~30 lines are scanned for the
    five term lines (Total, Fermi-contact, spin-dipolar,
    paramagnetic, diamagnetic) and the cross term. Spelled-out names
    take precedence; bare abbreviations are accepted as fallbacks.
    If ``Total`` was not printed, it is synthesized as
    ``FC + SD + PSO + DSO + (cross or 0)`` when all four Ramsey terms
    are present.

    Returns ``[]`` on read errors.
    """
    try:
        text = Path(out_path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []

    lines = text.splitlines()
    n = len(lines)
    pairs: dict[tuple[int, int], dict[str, Any]] = {}
    # Look-ahead window for "Nucleus B" after "Nucleus A", and for term
    # lines after a pair header. ORCA 6 puts the totals 8–15 lines below
    # the header; 30 is a generous cap that still bails before drifting
    # into the next pair.
    NUCLEUS_LOOKAHEAD = 6
    TERM_LOOKAHEAD = 30

    def _record_pair(
        a_idx: int, a_el: str, b_idx: int, b_el: str, scan_from: int
    ) -> int:
        """Scan terms starting from ``scan_from``, register the pair,
        and return the line index immediately past the scanned region."""
        key = (min(a_idx, b_idx), max(a_idx, b_idx))
        rec: dict[str, Any] = {
            "i": key[0],
            "j": key[1],
            "elem_i": a_el if a_idx == key[0] else b_el,
            "elem_j": b_el if b_idx == key[1] else a_el,
            "J_total_hz": None,
            "J_FC_hz": None,
            "J_SD_hz": None,
            "J_PSO_hz": None,
            "J_DSO_hz": None,
        }
        cross: Optional[float] = None

        end = min(scan_from + TERM_LOOKAHEAD, n)
        k = scan_from
        while k < end:
            ln = lines[k]
            # Stop early if we've drifted into the next pair header.
            if (
                _parse_nucleus_line(ln) is not None
                or COUPLING_PAIR_RE.search(ln) is not None
            ):
                break
            for term, pat in _COUPLING_TERM_PATTERNS.items():
                if rec[term] is None:
                    mt = pat.search(ln)
                    if mt:
                        try:
                            rec[term] = float(mt.group(1))
                        except (ValueError, TypeError):
                            pass
            if cross is None:
                mc = _COUPLING_CROSS_RE.search(ln)
                if mc:
                    try:
                        cross = float(mc.group(1))
                    except (ValueError, TypeError):
                        cross = None
            k += 1

        # Synthesize Total when all four Ramsey terms are present but
        # Total wasn't printed (some print levels suppress it).
        if rec["J_total_hz"] is None:
            terms = [
                rec[t]
                for t in ("J_FC_hz", "J_SD_hz", "J_PSO_hz", "J_DSO_hz")
            ]
            if all(t is not None for t in terms):
                total = sum(terms)  # type: ignore[arg-type]
                if cross is not None:
                    total += cross
                rec["J_total_hz"] = total

        pairs[key] = rec
        return k

    i = 0
    while i < n:
        line = lines[i]

        # Try the modern multi-line ``Nucleus A: ... / Nucleus B: ...``
        # shape first.
        nuc = _parse_nucleus_line(line)
        if nuc is not None and nuc[0] == "A":
            a_idx, a_el = nuc[1], nuc[2]
            j = i + 1
            j_end = min(j + NUCLEUS_LOOKAHEAD, n)
            b_idx: Optional[int] = None
            b_el: Optional[str] = None
            while j < j_end:
                nb = _parse_nucleus_line(lines[j])
                if nb is not None and nb[0] == "B":
                    b_idx, b_el = nb[1], nb[2]
                    break
                j += 1
            if b_idx is not None and b_el is not None:
                next_i = _record_pair(a_idx, a_el, b_idx, b_el, scan_from=j + 1)
                i = next_i if next_i > i else i + 1
                continue

        # Legacy single-line header.
        m = COUPLING_PAIR_RE.search(line)
        if m is not None:
            next_i = _record_pair(
                int(m.group(1)),
                m.group(2),
                int(m.group(3)),
                m.group(4),
                scan_from=i + 1,
            )
            i = next_i if next_i > i else i + 1
            continue

        i += 1

    return [pairs[k] for k in sorted(pairs)]


__all__ = [
    "COUPLING_NUCLEUS_RE",
    "COUPLING_PAIR_RE",
    "FINAL_E_RE",
    "FINAL_G_RE",
    "G_MINUS_E_EL_RE",
    "HARTREE_TO_KCAL",
    "ORCA_NORMAL_RE",
    "S_CORR_RE",
    "SHIELDING_HDR_RE",
    "SHIELDING_ROW_RE",
    "THERMO_T_RE",
    "TOTAL_H_RE",
    "classify_orca_outfile",
    "concat_xyz_files",
    "make_orca_compound_input",
    "make_orca_simple_input",
    "_uses_3c_composite_method",
    "resolve_functional_alias",
    "nmr_coupling_block",
    "nmr_shielding_block",
    "orca_terminated_normally",
    "parse_orca_couplings",
    "parse_orca_final_energy",
    "parse_orca_shieldings",
    "parse_orca_thermochem",
    "pick_orca_outputs",
    "solvent_to_orca_smd",
    "write_energy_file",
]
