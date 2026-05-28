"""Element coverage of named ORCA basis sets + heavy-atom supplementation.

The NMR jobs in :mod:`scripps_workflow.nodes.orca_thermo_array` default to
basis sets calibrated for routine organic molecules — Pople 6-31G(d,p) /
6-311++G(2d,p) for shieldings, Jensen pcJ-2 for J couplings. None of
these define basis functions for elements much past Cl/Kr, so a molecule
containing Br, I, Se, Pd, Pt, … silently corrupts the input file and
aborts the ORCA job after the SLURM queue time has already been spent.

This module is the coverage / supplementation layer:

    * :data:`BASIS_ELEMENT_COVERAGE` records which elements each named
      basis covers in ORCA 6.0.0.

    * :data:`TIER2_REQUIRES_RELATIVISTIC` lists heavy elements where
      scalar-relativistic ECPs alone are not enough for accurate NMR on
      neighboring light atoms — they need a relativistic Hamiltonian
      (ZORA or DKH) on the whole molecule, not just a per-atom basis
      patch. These elements escalate to a Tier 2 ``system_class``
      profile (e.g. ``organopd``) rather than being supplemented.

    * :func:`compute_coverage_decision` is the single entry point. Given
      the set of elements actually in the molecule + the operator's
      configured basis + the supplement basis, it returns a
      :class:`CoverageDecision` describing what ``%basis newgto`` block
      to inject and which (if any) Tier 2 elements were found.

The design keeps the calibrated light-atom basis intact: only the
uncovered heavy atoms get a per-element basis override via ORCA's
``%basis newgto`` syntax. The cached basis fingerprint records this so
re-running with a different ``heavy_atom_basis`` correctly invalidates
the cache.

Tier 2 elements are explicitly NOT handled by this module — the caller
is expected to either auto-promote ``system_class`` or fail with a
helpful message, depending on operator policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# --------------------------------------------------------------------
# Element coverage tables
# --------------------------------------------------------------------
#
# All keys are stored lower-cased; lookup goes through :func:`_norm`.
# When unsure whether ORCA actually defines a basis for an element, err
# on the side of UNDER-claiming coverage — the worst case is
# over-supplementing (harmless, just emits a redundant ``%basis newgto``
# that ORCA accepts and overrides on top of the base table). The
# expensive failure mode is OMITTING a supplement that was actually
# needed, which is the failure we are trying to eliminate.

# Pople through Kr — 6-31G / 6-311G / their polarization variants are
# defined for H–Kr in ORCA's built-in tables. The polarization
# functions (d, p, 2d, ...) follow the base set's element coverage.
_POPLE_HKR: frozenset[str] = frozenset({
    "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    # First-row transition metals (Sc–Zn) are technically defined in
    # 6-31G* and similar in ORCA, but the parameterization is rough
    # and we never use Pople bases for them — leave them out.
})

# Pople bases with diffuse functions (``+`` / ``++``) are only defined
# through Ar for the diffuse augmentation in standard 6-31+G / 6-311+G.
_POPLE_HAR: frozenset[str] = frozenset({
    "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
})

# Jensen pcS / pcJ series — explicitly parameterized for the
# main-group organic set only. Any element outside this list trips
# ORCA at input-read time.
_JENSEN_PCN: frozenset[str] = frozenset({
    "H", "B", "C", "N", "O", "F", "Al", "Si", "P", "S", "Cl",
})

# Karlsruhe def2 family covers H–Rn (Z=1..86) with built-in Stuttgart
# ECPs for Z >= 37. ORCA picks up the ECP automatically when the
# def2-* basis is requested. Excludes elements beyond Rn (Z > 86).
_DEF2_FULL: frozenset[str] = frozenset({
    # Row 1–2 (Z=1..18)
    "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    # Row 3 (Z=19..36)
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    # Row 4 (Z=37..54)
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    # Row 5 (Z=55..86)
    "Cs", "Ba",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
})

# Same span as def2-* but all-electron with ZORA recontraction (no ECPs).
_DEF2_ZORA: frozenset[str] = _DEF2_FULL


#: Element coverage by named basis set. Keys are lower-cased — lookups
#: go through :func:`_norm`. Add new bases here as they enter the
#: default config. Entries omitted from this table are treated as
#: "unknown coverage" by :func:`compute_coverage_decision`: it skips the
#: supplementation step and emits a warning so the gap is visible.
BASIS_ELEMENT_COVERAGE: dict[str, frozenset[str]] = {
    # Pople — base sets through Kr.
    "6-31g":           _POPLE_HKR,
    "6-31g(d)":        _POPLE_HKR,
    "6-31g(d,p)":      _POPLE_HKR,
    "6-31g*":          _POPLE_HKR,
    "6-31g**":         _POPLE_HKR,
    "6-311g":          _POPLE_HKR,
    "6-311g(d,p)":     _POPLE_HKR,
    "6-311g*":         _POPLE_HKR,
    "6-311g**":        _POPLE_HKR,
    "6-311g(2d,p)":    _POPLE_HKR,
    # Pople — diffuse-augmented sets through Ar.
    "6-31+g(d,p)":     _POPLE_HAR,
    "6-31++g(d,p)":    _POPLE_HAR,
    "6-311+g(d,p)":    _POPLE_HAR,
    "6-311++g(d,p)":   _POPLE_HAR,
    "6-311+g(2d,p)":   _POPLE_HAR,
    "6-311++g(2d,p)":  _POPLE_HAR,
    # Jensen pcS / pcJ — organic main-group only.
    "pcs-0":           _JENSEN_PCN,
    "pcs-1":           _JENSEN_PCN,
    "pcs-2":           _JENSEN_PCN,
    "pcs-3":           _JENSEN_PCN,
    "pcs-4":           _JENSEN_PCN,
    "pcj-0":           _JENSEN_PCN,
    "pcj-1":           _JENSEN_PCN,
    "pcj-2":           _JENSEN_PCN,
    "pcj-3":           _JENSEN_PCN,
    "pcj-4":           _JENSEN_PCN,
    # Karlsruhe def2 — full coverage with ECPs for Z >= 37.
    "def2-svp":        _DEF2_FULL,
    "def2-svpd":       _DEF2_FULL,
    "def2-tzvp":       _DEF2_FULL,
    "def2-tzvp(-f)":   _DEF2_FULL,
    "def2-tzvpp":      _DEF2_FULL,
    "def2-tzvppd":     _DEF2_FULL,
    "def2-qzvp":       _DEF2_FULL,
    "def2-qzvpp":      _DEF2_FULL,
    "def2-mtzvpp":     _DEF2_FULL,        # r2scan-3c default
    "def2-mtzvp":      _DEF2_FULL,
    # def2 ZORA-recontracted.
    "def2-zora-svp":   _DEF2_ZORA,
    "def2-zora-tzvp":  _DEF2_ZORA,
    "def2-zora-tzvpp": _DEF2_ZORA,
    "def2-zora-qzvpp": _DEF2_ZORA,
}


#: Heavy elements safe to supplement via ``%basis newgto`` without
#: switching to a relativistic Hamiltonian. Scalar-relativistic ECPs
#: (which the def2 family carries for Z >= 37) are sufficient for
#: light-atom NMR observables here — the spin-orbit HALA contribution
#: is below typical experimental noise on neighboring ¹H / ¹³C.
TIER1_SUPPLEMENTABLE: frozenset[str] = frozenset({
    # Heavy halogens / chalcogens / pnictogens — common in organic
    # chemistry, contribute via through-bond electron density but not
    # via large spin-orbit coupling to neighboring NMR-active nuclei
    # (at least not enough to demand a full ZORA recontraction).
    "Br", "I",
    "Se", "Te",
    "As", "Sb",
    # Heavy tetrels / group 13 — Sn, Pb show up in organometallic
    # NMR but the literature treats them as supplementable (def2-TZVPP
    # works well enough for J's on adjacent H/C).
    "Ga", "Ge", "In", "Sn", "Tl", "Pb",
    "Bi",
    # Heavy noble gases — rare but covered for completeness.
    "Kr", "Xe",
    # Heavy alkali / alkaline-earth counter-ions.
    "Rb", "Sr", "Cs", "Ba",
})


#: Elements that REQUIRE a relativistic Hamiltonian (ZORA / DKH) for
#: accurate light-atom NMR. Per-atom basis supplementation alone is
#: NOT sufficient — the HALA effect on neighboring ¹H / ¹³C shieldings
#: is large (~0.5–2 ppm on ¹H, ~5–20 ppm on ¹³C for a directly bonded
#: C) and only a relativistic Hamiltonian captures it.
#:
#: Caller policy decides what happens when one of these is detected
#: under a non-relativistic profile: fail loud, warn, or auto-promote
#: ``system_class`` to the matching Tier 2 profile.
TIER2_REQUIRES_RELATIVISTIC: frozenset[str] = frozenset({
    # 4d transition metals.
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    # 5d transition metals.
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    # Lanthanides — rare, but if present they unambiguously need ZORA.
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    # Actinides — even rarer, same story.
    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm",
})


def _norm(name: str) -> str:
    """Lower-case + strip for case-insensitive basis-name lookup."""
    return (name or "").strip().lower()


# --------------------------------------------------------------------
# Decision API
# --------------------------------------------------------------------


@dataclass(frozen=True)
class CoverageDecision:
    """Outcome of a basis-coverage check for one ORCA job.

    Attributes:
        extra_blocks: ORCA block strings (``%basis newgto …``) to
            append to ``make_orca_simple_input``'s ``extra_blocks``.
            Empty when no supplementation is needed.
        supplemented_elements: sorted symbols of elements that received
            a Tier-1 ``%basis newgto`` supplement. Empty when there was
            nothing to supplement.
        tier2_elements: sorted symbols of Tier-2 elements present in
            the molecule and NOT covered by the active basis. The
            caller must decide policy (fail / warn / auto-promote
            ``system_class``).
        fingerprint_suffix: string to append to the basis name when
            building the cache key — e.g. ``"+def2-TZVPP/heavy"``.
            Empty when no supplementation. Encodes the supplement
            choice into the cached basis identity so a re-run with a
            different ``heavy_atom_basis`` correctly invalidates.
        warnings: free-form messages worth logging (e.g. unknown basis,
            supplement basis missing coverage). Non-fatal.
    """
    extra_blocks: list[str] = field(default_factory=list)
    supplemented_elements: list[str] = field(default_factory=list)
    tier2_elements: list[str] = field(default_factory=list)
    fingerprint_suffix: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def has_supplementation(self) -> bool:
        """True when at least one element was Tier-1 supplemented."""
        return bool(self.supplemented_elements)

    @property
    def has_tier2(self) -> bool:
        """True when at least one Tier-2 element was detected."""
        return bool(self.tier2_elements)


def compute_coverage_decision(
    elements: set[str] | frozenset[str],
    *,
    basis: str,
    supplement_basis: str = "def2-TZVPP",
) -> CoverageDecision:
    """Decide what ``%basis newgto`` block (if any) is needed for the job.

    Pure function — no I/O, no logging. The caller drives policy on the
    returned :class:`CoverageDecision` (e.g. apply the
    ``on_uncovered_heavy_metal`` knob against
    :attr:`CoverageDecision.tier2_elements`).

    Args:
        elements: set of element symbols actually present in the
            molecule. Get this from scanning the staged xyz files.
        basis: the operator-configured basis name (e.g.
            ``"6-31G(d,p)"``). Case-insensitive.
        supplement_basis: basis to attach to each uncovered Tier-1
            element via ``%basis newgto``. Defaults to ``def2-TZVPP``
            because the def2 family spans H–Rn with ECPs for Z >= 37
            and is what r2scan-3c / wB97M-V already use elsewhere in
            the pipeline.

    Returns:
        A :class:`CoverageDecision`. When everything is already covered,
        all fields are empty / default and the caller does nothing.
    """
    covered = BASIS_ELEMENT_COVERAGE.get(_norm(basis))
    if covered is None:
        # Unknown basis — assume the operator knows what they're doing
        # but flag it so the gap in our coverage table is visible.
        return CoverageDecision(
            warnings=[
                f"basis {basis!r} not in BASIS_ELEMENT_COVERAGE; "
                "skipping coverage check (extend the table if heavy "
                "atom jobs with this basis are tripping at ORCA "
                "read time)",
            ],
        )

    missing = set(elements) - covered
    if not missing:
        return CoverageDecision()

    tier2 = sorted(missing & TIER2_REQUIRES_RELATIVISTIC)
    tier1 = sorted(missing - TIER2_REQUIRES_RELATIVISTIC)
    warnings: list[str] = []

    # Tier-1 elements that we don't list as supplementable still get a
    # supplement attempt, but warn so unknown-element categories surface
    # quickly. (E.g. an exotic post-transition heavy that we forgot to
    # add to TIER1_SUPPLEMENTABLE still gets a def2-TZVPP override.)
    unknown_tier = [e for e in tier1 if e not in TIER1_SUPPLEMENTABLE]
    if unknown_tier:
        warnings.append(
            f"element(s) {unknown_tier!r} not in TIER1_SUPPLEMENTABLE; "
            "supplementing anyway with the configured heavy_atom_basis "
            "(add to the registry if this is a recurring system class)"
        )

    blocks: list[str] = []
    fingerprint_suffix = ""
    if tier1:
        # Sanity-check: does the supplement basis itself cover the
        # Tier-1 elements we're routing to it? If not, the resulting
        # %basis newgto block will fail at ORCA read time.
        sup_covered = BASIS_ELEMENT_COVERAGE.get(_norm(supplement_basis))
        if sup_covered is not None:
            uncovered_by_sup = sorted(set(tier1) - sup_covered)
            if uncovered_by_sup:
                warnings.append(
                    f"supplement basis {supplement_basis!r} does not "
                    f"cover {uncovered_by_sup!r}; ORCA will reject the "
                    "%basis newgto block at read time. Pick a broader "
                    "heavy_atom_basis or extend the coverage table."
                )
        blocks.append(build_newgto_block(tier1, supplement_basis))
        fingerprint_suffix = f"+{supplement_basis}/heavy"

    return CoverageDecision(
        extra_blocks=blocks,
        supplemented_elements=tier1,
        tier2_elements=tier2,
        fingerprint_suffix=fingerprint_suffix,
        warnings=warnings,
    )


def build_newgto_block(
    elements: list[str] | tuple[str, ...],
    supplement_basis: str,
) -> str:
    """Render an ORCA ``%basis newgto`` block for per-element overrides.

    Produces text shaped like::

        %basis
          newgto Br "def2-TZVPP" end
          newgto I  "def2-TZVPP" end
        end

    The output is suitable for inclusion in
    :func:`scripps_workflow.orca.make_orca_simple_input`'s
    ``extra_blocks`` list. ORCA's input parser places ``%basis`` blocks
    in the pre-xyz section, which the simple-input renderer already
    handles correctly.

    Args:
        elements: list of element symbols to override. Empty list
            returns the empty string.
        supplement_basis: basis name to attach to each element.

    Returns:
        The rendered ``%basis`` block as a single string, or ``""``
        when ``elements`` is empty.
    """
    if not elements:
        return ""
    lines = ["%basis"]
    for elem in elements:
        # Two-space indent to match the rest of the simple-input
        # blocks (%pal, %cpcm, %eprnmr).
        lines.append(f'  newgto {elem} "{supplement_basis}" end')
    lines.append("end")
    return "\n".join(lines)


def format_basis_fingerprint(basis: str, decision: CoverageDecision) -> str:
    """Encode the supplementation choice into the basis identity string.

    Used by :mod:`scripps_workflow.nodes.orca_thermo_array` when
    recording ``shielding_basis_*`` / ``coupling_basis`` in
    ``manifest.inputs`` and when building the ``PredictedRunKey`` cache
    fingerprint. A supplemented run thus sees its basis as e.g.
    ``"6-31G(d,p)+def2-TZVPP/heavy"`` — distinct from the unsupplemented
    ``"6-31G(d,p)"``, so the cache correctly distinguishes the two.

    Idempotent on the no-supplement case: returns ``basis`` unchanged.
    """
    return f"{basis}{decision.fingerprint_suffix}"


def extract_base_basis(basis: str) -> str:
    """Strip the heavy-atom supplement suffix back to the base basis.

    Inverse of :func:`format_basis_fingerprint` for cases where the
    consumer only cares about the calibrated light-atom basis — most
    notably :func:`scripps_workflow.nmr_calibration.lookup_calibration`,
    which is keyed by the calibrated `(functional, basis, solvent,
    nucleus)` tuple. Heavy-atom supplementation doesn't change the
    light-atom shielding values enough to invalidate the calibration
    (the supplemented basis only acts on the heavy atom itself), so
    the supplemented and unsupplemented forms share the same
    calibration row.

    Examples:
        ``"6-31G(d,p)+def2-TZVPP/heavy"`` -> ``"6-31G(d,p)"``
        ``"6-31G(d,p)"``                 -> ``"6-31G(d,p)"``
        ``"def2-ZORA-TZVPP"``            -> ``"def2-ZORA-TZVPP"``
    """
    if not isinstance(basis, str):
        return basis
    # The supplement suffix shape is ``+<basis>/heavy``. Anything else
    # passes through unchanged so operator-typed basis names with a
    # legitimate ``+`` (none of the named bases we support contain
    # ``+``, but be defensive) aren't accidentally truncated.
    idx = basis.find("+")
    if idx > 0 and basis.endswith("/heavy"):
        return basis[:idx]
    return basis


# --------------------------------------------------------------------
# Element scanning from xyz files
# --------------------------------------------------------------------


def scan_elements_from_xyz_paths(
    paths: list[Path] | tuple[Path, ...],
    *,
    max_files: int = 3,
) -> set[str]:
    """Collect the set of element symbols across a list of xyz files.

    Cap defaults to 3 because element composition is invariant across
    conformers of one molecule — a 50-conformer ensemble would scan
    the same elements 50 times. Three is enough to be robust to a
    single corrupted file in the first slot.

    Lines that don't parse as ``Elem  x  y  z`` (atom-count header,
    blank lines, comments) are silently skipped; we keep any leading
    token that looks like an element symbol (alphabetic, length 1–2,
    starts with an uppercase letter).

    Returns:
        A set of element symbols. Empty set when nothing was readable.
    """
    elements: set[str] = set()
    for xyz in list(paths)[:max_files]:
        try:
            text = Path(xyz).read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            tokens = line.split()
            if not tokens:
                continue
            tok = tokens[0]
            # Element symbol heuristic: 1–2 chars, first char A-Z,
            # all alphabetic. Skips atom-count header, x/y/z coords,
            # comment lines, etc.
            if 1 <= len(tok) <= 2 and tok[0].isupper() and tok.isalpha():
                elements.add(tok)
    return elements


__all__ = [
    "BASIS_ELEMENT_COVERAGE",
    "CoverageDecision",
    "TIER1_SUPPLEMENTABLE",
    "TIER2_REQUIRES_RELATIVISTIC",
    "build_newgto_block",
    "compute_coverage_decision",
    "extract_base_basis",
    "format_basis_fingerprint",
    "scan_elements_from_xyz_paths",
]
