"""Element-driven basis selection for the NMR shielding / coupling jobs.

The default shielding and coupling bases in
:mod:`scripps_workflow.nodes.orca_thermo_array` (Pople 6-31G(d,p) /
6-311++G(2d,p) for shieldings, Jensen pcJ-2 for J couplings) are
calibrated for routine organic molecules and don't define basis
functions for elements past Kr / Ar / Cl respectively. A heavy atom
in the molecule corrupts the ORCA input silently and aborts the job
after the SLURM queue time has already been spent.

This module's single entry point :func:`compute_coverage_decision`
takes the set of elements actually in the molecule and the operator's
configured basis, and returns a :class:`CoverageDecision` carrying
everything the caller needs to render a working ORCA input:

* :attr:`CoverageDecision.effective_basis` — the basis to put on the
  ``! NMR <method> <basis>`` keyword line. Equal to the operator's
  basis when no element override is needed; equal to
  ``relativistic_basis`` (def2-ZORA-TZVPP) when any HALA-relevant
  metal is present.
* :attr:`CoverageDecision.nmr_keywords_prefix` — empty string for the
  organic case, ``"ZORA"`` when the relativistic Hamiltonian was
  enabled (driven entirely by which elements were detected, not by a
  profile name).
* :attr:`CoverageDecision.extra_blocks` — ``%basis newgto`` blocks
  for the per-atom supplementation case (Br / I / Se / Sn / ... in a
  Pople-only base basis). Empty when not needed.
* :attr:`CoverageDecision.fingerprint_basis` — the basis identity to
  record into ``manifest.inputs.shielding_basis_*`` and the
  PredictedRunKey cache fingerprint. Distinct strings for the
  unsupplemented / supplemented / ZORA-swapped cases, so the cache
  cleanly distinguishes them.

The classification is per-element, not per-molecule. There's no
``system_class`` profile, no per-metal grouping — only the two
element tags below:

* :data:`BASIS_ELEMENT_COVERAGE` — for each named basis, the set of
  elements ORCA can resolve from its built-in tables. Used to detect
  Tier-1 coverage gaps (light-heavy atoms outside the base basis).
* :data:`ELEMENT_REQUIRES_RELATIVISTIC` — elements where scalar-
  relativistic ECPs alone aren't enough for accurate NMR on neighbors
  (HALA effect dominates). Their presence flips the whole job to a
  relativistic Hamiltonian.

Adding a new element class is a one-line edit in one of those two
sets — no profile dispatch tree to update.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# --------------------------------------------------------------------
# Element coverage of named basis sets
# --------------------------------------------------------------------
#
# All keys are stored lower-cased; lookup goes through :func:`_norm`.
# Conservative: when in doubt, UNDER-claim coverage so the supplementation
# path fires for an element ORCA might actually handle. Over-supplementing
# is harmless (ORCA accepts a redundant %basis newgto override). Under-
# supplementing is the bug we're trying to eliminate.

# Pople through Kr — 6-31G / 6-311G / their polarization variants are
# defined for H–Kr in ORCA's built-in tables.
_POPLE_HKR: frozenset[str] = frozenset({
    "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
})

# Pople bases with diffuse functions (``+`` / ``++``) are only defined
# through Ar for the diffuse augmentation in standard 6-31+G / 6-311+G.
_POPLE_HAR: frozenset[str] = frozenset({
    "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
})

# Jensen pcS / pcJ series — organic main-group only.
_JENSEN_PCN: frozenset[str] = frozenset({
    "H", "B", "C", "N", "O", "F", "Al", "Si", "P", "S", "Cl",
})

# Karlsruhe def2 family covers H–Rn (Z=1..86) with built-in Stuttgart
# ECPs for Z >= 37. ORCA picks up the ECP automatically when the
# def2-* basis is requested.
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

# def2 ZORA-recontracted: same span as def2-*, all-electron, no ECPs.
_DEF2_ZORA: frozenset[str] = _DEF2_FULL


#: Element coverage by named basis set. Keys are lower-cased.
#: Entries omitted from this table are treated as "unknown coverage"
#: by :func:`compute_coverage_decision`: it skips the supplementation
#: step and emits a warning so the gap is visible.
BASIS_ELEMENT_COVERAGE: dict[str, frozenset[str]] = {
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
    "6-31+g(d,p)":     _POPLE_HAR,
    "6-31++g(d,p)":    _POPLE_HAR,
    "6-311+g(d,p)":    _POPLE_HAR,
    "6-311++g(d,p)":   _POPLE_HAR,
    "6-311+g(2d,p)":   _POPLE_HAR,
    "6-311++g(2d,p)":  _POPLE_HAR,
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
    "def2-svp":        _DEF2_FULL,
    "def2-svpd":       _DEF2_FULL,
    "def2-tzvp":       _DEF2_FULL,
    "def2-tzvp(-f)":   _DEF2_FULL,
    "def2-tzvpp":      _DEF2_FULL,
    "def2-tzvppd":     _DEF2_FULL,
    "def2-qzvp":       _DEF2_FULL,
    "def2-qzvpp":      _DEF2_FULL,
    "def2-mtzvpp":     _DEF2_FULL,
    "def2-mtzvp":      _DEF2_FULL,
    "def2-zora-svp":   _DEF2_ZORA,
    "def2-zora-tzvp":  _DEF2_ZORA,
    "def2-zora-tzvpp": _DEF2_ZORA,
    "def2-zora-qzvpp": _DEF2_ZORA,
}


#: Elements where scalar-relativistic ECPs alone are NOT enough for
#: accurate light-atom NMR — spin-orbit-driven HALA contributions on
#: neighboring ¹H / ¹³C are large enough (~0.5–2 ppm on ¹H, ~5–20 ppm
#: on ¹³C for a directly bonded C) that you need a relativistic
#: Hamiltonian (ZORA / DKH) globally, not just per-atom basis
#: supplementation.
#:
#: Their presence in the molecule auto-flips the NMR jobs to the
#: ``relativistic_basis`` (def2-ZORA-TZVPP by default) on every atom
#: + a ``ZORA`` prefix on the ``!`` line. The calibrated light-atom
#: basis is intentionally NOT preserved here — under the relativistic
#: Hamiltonian the calibrated values are nominally invalid anyway,
#: and ZORA basis on heavy atoms with non-ZORA basis on light atoms
#: is technically inconsistent in ORCA. Use a separate calibration
#: tuple for the relativistic basis if you need calibrated shifts on
#: heavy-metal complexes (TODO #90).
#:
#: Coverage: 4d + 5d transition metals, lanthanides, actinides.
#: First-row transition metals (Sc–Zn) are EXCLUDED — scalar
#: relativistic effects there are small enough that ECPs (or no ECP
#: for the def2 family on 3d metals) handle them adequately.
ELEMENT_REQUIRES_RELATIVISTIC: frozenset[str] = frozenset({
    # 4d transition metals.
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    # 5d transition metals.
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    # Lanthanides.
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    # Actinides.
    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm",
})


#: Default supplement basis for Tier-1 light-heavy atoms (Br, I, Se,
#: Sn, ...). ``def2-TZVPP`` spans Z=1..86 with built-in Stuttgart ECPs
#: for Z >= 37 — the safe choice for organic + light-heavy systems.
DEFAULT_HEAVY_ATOM_BASIS: str = "def2-TZVPP"

#: Default relativistic basis used when an element from
#: :data:`ELEMENT_REQUIRES_RELATIVISTIC` is present. Recontracted for
#: the ZORA Hamiltonian; covers Z=1..86 all-electron (no ECPs, ZORA
#: replaces the relativistic-core treatment).
DEFAULT_RELATIVISTIC_BASIS: str = "def2-ZORA-TZVPP"


def _norm(name: str) -> str:
    """Lower-case + strip for case-insensitive basis-name lookup."""
    return (name or "").strip().lower()


def needs_relativistic_treatment(elements: set[str] | frozenset[str]) -> bool:
    """True when any element in the molecule needs a relativistic
    Hamiltonian for accurate light-atom NMR (HALA effect)."""
    return bool(set(elements) & ELEMENT_REQUIRES_RELATIVISTIC)


# --------------------------------------------------------------------
# Decision API
# --------------------------------------------------------------------


@dataclass(frozen=True)
class CoverageDecision:
    """Outcome of an element-driven basis check for one NMR job.

    Attributes:
        effective_basis: basis to put on the ORCA ``! NMR <method>
            <basis>`` keyword line. Equal to the caller's ``base_basis``
            when no Tier-2 element was detected; equal to
            ``relativistic_basis`` (the full swap) when ZORA was
            triggered.
        nmr_keywords_prefix: ``"ZORA"`` when a relativistic Hamiltonian
            was enabled, empty string otherwise. Prepend to the ``!``
            line: ``! {prefix} NMR {method} {basis}``.
        extra_blocks: ``%basis newgto`` blocks for Tier-1 per-atom
            supplementation. Empty when the relativistic full swap
            applies (the swapped basis already covers everything) or
            when nothing needed supplementation.
        fingerprint_basis: basis identity to record into
            ``manifest.inputs.shielding_basis_*`` / cache fingerprint.
            Distinct strings across the {unchanged / Tier-1
            supplemented / Tier-2 full-swapped} cases so the
            PredictedRunKey cleanly distinguishes them.
        supplemented_elements: sorted Tier-1 elements that received a
            ``%basis newgto`` supplement. Empty for the ZORA / no-op
            cases.
        tier2_elements: sorted Tier-2 elements detected in the
            molecule. Non-empty drives the ZORA swap.
        warnings: free-form messages worth logging (unknown basis,
            supplement basis missing coverage).
    """
    effective_basis: str = ""
    nmr_keywords_prefix: str = ""
    extra_blocks: list[str] = field(default_factory=list)
    fingerprint_basis: str = ""
    supplemented_elements: list[str] = field(default_factory=list)
    tier2_elements: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def has_supplementation(self) -> bool:
        """True when at least one element was Tier-1 supplemented."""
        return bool(self.supplemented_elements)

    @property
    def has_relativistic_treatment(self) -> bool:
        """True when the relativistic full-swap was triggered."""
        return bool(self.tier2_elements)


def compute_coverage_decision(
    elements: set[str] | frozenset[str],
    *,
    base_basis: str,
    heavy_atom_basis: str = DEFAULT_HEAVY_ATOM_BASIS,
    relativistic_basis: str = DEFAULT_RELATIVISTIC_BASIS,
) -> CoverageDecision:
    """Resolve one NMR job's basis configuration from element scan.

    Two branches:

    * If any element in :data:`ELEMENT_REQUIRES_RELATIVISTIC` is
      present in ``elements``: swap the whole job to
      ``relativistic_basis`` (the operator's ``base_basis`` is
      discarded for THIS job) and emit a ``"ZORA"`` keyword prefix.
      ORCA's ZORA implementation is strictly only consistent when
      every atom carries a ZORA-recontracted basis — the swap is
      mandatory, not optional. No ``%basis newgto`` blocks needed:
      ``relativistic_basis`` covers everything ORCA recognizes.

    * Otherwise: keep the operator's ``base_basis``. Any elements
      outside its coverage table get a per-atom ``%basis newgto``
      supplement using ``heavy_atom_basis``. The light-atom basis
      stays calibrated.

    Pure function — no I/O, no logging. The caller drives policy +
    logging from :class:`CoverageDecision`.

    Args:
        elements: element symbols present in the molecule.
        base_basis: operator-configured basis (e.g. ``"6-31G(d,p)"``,
            ``"pcJ-2"``). Case-insensitive.
        heavy_atom_basis: basis attached per-element to Tier-1
            light-heavy atoms (Br, I, Se, ...). Defaults to
            :data:`DEFAULT_HEAVY_ATOM_BASIS`.
        relativistic_basis: basis used globally when any
            HALA-relevant element is detected. Defaults to
            :data:`DEFAULT_RELATIVISTIC_BASIS`.
    """
    elements_set = set(elements)
    tier2 = sorted(elements_set & ELEMENT_REQUIRES_RELATIVISTIC)

    if tier2:
        # Full swap branch. The relativistic basis covers everything;
        # no per-atom supplementation needed. ``ZORA`` prefix goes on
        # the ``!`` line. Cache fingerprint = the relativistic basis
        # itself — different identity from any Tier-1 supplemented
        # form, so cache rows don't collide.
        return CoverageDecision(
            effective_basis=relativistic_basis,
            nmr_keywords_prefix="ZORA",
            fingerprint_basis=relativistic_basis,
            tier2_elements=tier2,
        )

    # Tier-1 branch — check per-element coverage of base_basis.
    covered = BASIS_ELEMENT_COVERAGE.get(_norm(base_basis))
    if covered is None:
        return CoverageDecision(
            effective_basis=base_basis,
            fingerprint_basis=base_basis,
            warnings=[
                f"basis {base_basis!r} not in BASIS_ELEMENT_COVERAGE; "
                "skipping coverage check (extend the table if heavy "
                "atom jobs with this basis are tripping at ORCA "
                "read time)",
            ],
        )

    missing = sorted(elements_set - covered)
    if not missing:
        return CoverageDecision(
            effective_basis=base_basis,
            fingerprint_basis=base_basis,
        )

    # Per-element supplementation. Verify the supplement basis itself
    # covers what we're routing to it — if not, ORCA will reject the
    # %basis block at read time.
    warnings: list[str] = []
    sup_covered = BASIS_ELEMENT_COVERAGE.get(_norm(heavy_atom_basis))
    if sup_covered is not None:
        uncovered_by_sup = sorted(set(missing) - sup_covered)
        if uncovered_by_sup:
            warnings.append(
                f"supplement basis {heavy_atom_basis!r} does not cover "
                f"{uncovered_by_sup!r}; ORCA will reject the %basis "
                "newgto block at read time. Pick a broader "
                "heavy_atom_basis or extend the coverage table."
            )

    return CoverageDecision(
        effective_basis=base_basis,
        nmr_keywords_prefix="",
        extra_blocks=[build_newgto_block(missing, heavy_atom_basis)],
        fingerprint_basis=f"{base_basis}+{heavy_atom_basis}/heavy",
        supplemented_elements=missing,
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
    ``extra_blocks`` list. Empty ``elements`` returns the empty string.
    """
    if not elements:
        return ""
    lines = ["%basis"]
    for elem in elements:
        lines.append(f'  newgto {elem} "{supplement_basis}" end')
    lines.append("end")
    return "\n".join(lines)


def extract_base_basis(basis: str) -> str:
    """Strip the heavy-atom supplement suffix back to the base basis.

    Inverse of the Tier-1 fingerprint encoding (``+<supplement>/heavy``).
    Used by
    :func:`scripps_workflow.nmr_calibration.lookup_calibration` so a
    supplemented run finds the same calibration row as its
    unsupplemented sibling.

    Examples:
        ``"6-31G(d,p)+def2-TZVPP/heavy"`` -> ``"6-31G(d,p)"``
        ``"6-31G(d,p)"``                 -> ``"6-31G(d,p)"``
        ``"def2-ZORA-TZVPP"``            -> ``"def2-ZORA-TZVPP"``

    The Tier-2 full-swap case (``"def2-ZORA-TZVPP"``) passes through
    unchanged — the relativistic basis IS the identity, no suffix to
    strip. The calibration lookup will then either find a separately
    lab-fit ZORA calibration row or return ``None`` and the aggregator
    falls back to raw σ.
    """
    if not isinstance(basis, str):
        return basis
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
    conformers of one molecule. Returns an empty set when nothing was
    readable.
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
            # all alphabetic. Skips atom-count header and numeric
            # tokens.
            if 1 <= len(tok) <= 2 and tok[0].isupper() and tok.isalpha():
                elements.add(tok)
    return elements


__all__ = [
    "BASIS_ELEMENT_COVERAGE",
    "CoverageDecision",
    "DEFAULT_HEAVY_ATOM_BASIS",
    "DEFAULT_RELATIVISTIC_BASIS",
    "ELEMENT_REQUIRES_RELATIVISTIC",
    "build_newgto_block",
    "compute_coverage_decision",
    "extract_base_basis",
    "needs_relativistic_treatment",
    "scan_elements_from_xyz_paths",
]
