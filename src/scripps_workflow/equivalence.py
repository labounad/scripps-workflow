"""Chemical-equivalence detection for NMR spin-system grouping.

The mnova-spinsim XML emitter needs to know, given a molecule + per-atom
chemical shifts + per-pair J couplings, how to bucket the atoms into
``<group>`` blocks. The grouping is the difference between getting the
right multiplet pattern in the simulated spectrum (e.g., a methyl as a
single triplet) and getting nonsense (the methyl as three near-coincident
singlets that smear into a broad blob).

We use a **three-tier classification**:

* **HARD** — homotopic AND magnetically equivalent. All class members
  have identical J-vectors to every other atom (within a small Hz
  tolerance, since DFT-noise-level differences are unavoidable). These
  collapse into ONE group with ``number=N``. Their shifts and J's are
  averaged across the class. Example: a methyl group's three protons.

* **SOFT** — homotopic (same chemical shift) but magnetically
  inequivalent (different J-vectors to some other atom). Classic
  AA'BB' / AA'XX' patterns: 1,2-disubstituted aromatics, vinyl
  groups, monosubstituted alkenes. These emit as N separate groups
  with identical shifts but distinct J's preserved.

* **NONE** — topologically distinct. No averaging. Each atom emits as
  its own group. Catches diastereotopic CH₂ in chiral environments,
  isolated CH protons, and so on.

The dispatch is mechanical:

1. Compute topological classes from RDKit's chirality-aware
   ``CanonicalRankAtoms`` — atoms with the same rank are in one class.
   This catches "homotopic" cleanly even when the symmetry operation
   is a non-trivial rotation/reflection.

2. **Data-aware refinement (gated on stereo).** RDKit's
   ``includeChirality=True`` only propagates atom-level CIP flags
   (``@`` tags) — it does NOT split prochiral H's attached to a
   carbon adjacent to a chiral center. Diastereotopic CH₂ pairs in
   chiral environments therefore look topologically identical to
   RDKit even though they're chemically distinct. When the molecule
   has stereo info (chiral centers, stereo bonds), we compensate by
   inspecting the DFT-computed shifts: if class members spread more
   than ``tol_shift_ppm``, the class gets split into singletons.

   When the molecule has NO stereo info, the refinement is skipped
   entirely. Same-rank atoms are then chemically equivalent under
   fast rotation / NMR timescale by symmetry, and any within-class
   shift variation is sampling noise from a finite-conformer
   Boltzmann ensemble (a 2-conformer run on ethanol doesn't sample
   methyl C₃ rotation uniformly, leaving 0.1-0.3 ppm artifactual
   spread on what should be a single methyl peak). Skipping the
   refinement in the achiral case prevents false-splitting that
   would emit 3 separate "A/B/C" groups for one methyl.

3. Within each (refined) class of size ≥ 2, test whether every
   member's J-coupling vector to every other atom matches (within
   ``tol_jcoupling_hz``). If yes → HARD. If no → SOFT. Class size 1
   is trivially NONE.

This module is a pure helper. RDKit is imported lazily inside the
functions that need it so the module can be imported without RDKit
installed (e.g., for code-loading in the node base class). Functions
that operate on already-built RDKit ``Mol`` objects don't need RDKit
imported here at all — only :func:`mol_from_smiles_or_xyz` and
:func:`topological_classes` invoke it.

Atom-index alignment: this module assumes the caller's atom indices
match :class:`rdkit.Chem.Mol` indices after ``AddHs``. ORCA outputs
preserve the input-xyz ordering, and ``smiles_to_3d`` writes its xyz
in ``mol.GetAtoms()`` order after ``AddHs``, so the chain is
self-consistent end-to-end. :func:`mol_from_smiles_or_xyz` re-derives
the mol the same way (``MolFromSmiles`` then ``AddHs``) to keep that
alignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# --------------------------------------------------------------------
# Tier enum + EquivalenceGroup dataclass
# --------------------------------------------------------------------


class Tier(str, Enum):
    """Equivalence-class disposition for spin-system rendering.

    Stringly-typed (inherits ``str``) so the value can be dropped into a
    JSON manifest without a custom encoder.
    """

    #: Homotopic + magnetically equivalent. Collapse into one group with
    #: ``number=N``, average shifts, average J's.
    HARD = "hard"

    #: Homotopic but magnetically inequivalent (AA'BB' etc.). Emit as N
    #: separate groups with identical shifts and distinct J's.
    SOFT = "soft"

    #: Topologically distinct. Emit as individual groups, no averaging.
    NONE = "none"


#: 2I (twice the nuclear spin) for common NMR-active nuclei. ¹H, ¹³C,
#: ¹⁹F, ³¹P are all spin-½ → 2I=1. ²H (D) and ¹⁴N are spin-1 → 2I=2.
#: Used to fill mnova's ``spinByTwo`` group attribute. Anything not in
#: this map defaults to 1 — most consumers care only about spin-½.
_SPIN_BY_TWO_BY_ELEMENT: dict[str, int] = {
    "H": 1,
    "C": 1,
    "N": 2,  # ¹⁴N is the natural-abundance default; ¹⁵N would also be 1
    "O": 1,  # ¹⁷O is 5; rare enough to ignore
    "F": 1,
    "P": 1,
    "D": 2,
}


@dataclass(frozen=True)
class EquivalenceGroup:
    """One mnova ``<group>`` block worth of data.

    ``atom_indices`` lists the source atoms folded into this group:
    multiple for HARD (e.g., methyl), exactly one for SOFT and NONE.
    ``number`` is ``len(atom_indices)`` and is what mnova's
    ``number="..."`` attribute carries.

    ``j_couplings`` is keyed by *other* group's name (the destination's
    ``name`` attribute), value is the J coupling in Hz already averaged
    over whatever class members participated. Self-couplings are NOT
    represented — magnetically equivalent nuclei within a HARD group
    don't show observable splitting from each other and mnova handles
    that implicitly.
    """

    name: str
    element: str
    atom_indices: tuple[int, ...]
    shift_avg_ppm: float
    tier: Tier
    j_couplings: dict[str, float] = field(default_factory=dict)

    @property
    def number(self) -> int:
        return len(self.atom_indices)

    @property
    def spin_by_two(self) -> int:
        return _SPIN_BY_TWO_BY_ELEMENT.get(self.element.upper(), 1)


# --------------------------------------------------------------------
# Excel-style group labels
# --------------------------------------------------------------------


def _index_to_excel_letters(idx: int) -> str:
    """Convert a 0-based index to Excel-style letters: 0→A, 25→Z, 26→AA, …

    Standard "bijective base-26" — every column has a unique letter
    string with no leading-A ambiguity (so A, B, ..., Z, AA, AB, ..., AZ,
    BA, ...). This is what mnova accepts in the ``name`` attribute,
    confirmed empirically — anything past 26 single-letter groups falls
    back to 2-letter labels rather than running out.
    """
    if idx < 0:
        raise ValueError(f"_index_to_excel_letters: idx must be >= 0, got {idx}")
    letters: list[str] = []
    n = idx
    while True:
        letters.append(chr(ord("A") + n % 26))
        n = n // 26 - 1
        if n < 0:
            break
    return "".join(reversed(letters))


def assign_group_labels(n: int) -> list[str]:
    """Generate ``n`` Excel-style group labels: ``["A", "B", ..., "AA", ...]``.

    Returned list is length ``n``. Labels are unique, ordered, and stable
    — the same input produces the same list, suitable for use as mnova
    ``<group name="...">`` attributes. ``n=0`` returns ``[]``.
    """
    if n < 0:
        raise ValueError(f"assign_group_labels: n must be >= 0, got {n}")
    return [_index_to_excel_letters(i) for i in range(n)]


# --------------------------------------------------------------------
# Mol construction (RDKit-backed, lazy import)
# --------------------------------------------------------------------


def mol_from_smiles_or_xyz(
    *,
    smiles: Optional[str] = None,
    xyz_text: Optional[str] = None,
    charge: int = 0,
) -> Optional[Any]:
    """Build an RDKit ``Mol`` from SMILES (preferred) or xyz fallback.

    Precedence:

    1. If ``smiles`` is provided and parses cleanly → ``MolFromSmiles``
       + ``AddHs``. Atom order matches what ``smiles_to_3d`` writes to
       the xyz, so atom indices align with ORCA output indices.

    2. If only ``xyz_text`` is provided → ``MolFromXYZBlock`` +
       ``DetermineBonds`` (uses ``charge`` to disambiguate). Bond
       perception is decent for clean organic molecules but can fail
       on charged species, hypervalent centers, or anything with
       coordinative bonds.

    3. Both absent or both fail → returns ``None``. Callers (typically
       :class:`scripps_workflow.nodes.nmr_aggregate.NmrAggregate`) treat
       this as "skip equivalence detection, emit each atom as its own
       group" or surface a structured ``mnova_xml_skipped`` failure.

    RDKit is imported lazily so this module imports cleanly in test
    environments without RDKit. The no-input fast path returns ``None``
    *before* attempting the import, so a caller probing whether to
    enable XML emission can call this safely on a stripped env. If
    inputs ARE provided and RDKit is missing, the import error
    propagates with the standard RDKit failure message.
    """
    smi = str(smiles or "").strip()
    xyz = str(xyz_text or "").strip()
    if not smi and not xyz:
        # Nothing to parse — don't pay the rdkit import cost.
        return None

    from rdkit import Chem  # type: ignore[import-not-found]

    if smi:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            # AddHs is what aligns indices with smiles_to_3d's xyz.
            # Order: implicit H's get appended in heavy-atom order.
            return Chem.AddHs(mol)

    if xyz:
        try:
            mol = Chem.MolFromXYZBlock(xyz)
        except Exception:
            mol = None
        if mol is not None:
            try:
                from rdkit.Chem import (  # type: ignore[import-not-found]
                    rdDetermineBonds,
                )

                rdDetermineBonds.DetermineBonds(mol, charge=int(charge))
                return mol
            except Exception:
                # Bond perception failed — usable atom positions but
                # no connectivity, which means CanonicalRankAtoms
                # will give every atom a unique rank (no useful
                # equivalence info). Return None so the caller
                # falls back to one-group-per-atom rendering.
                return None

    return None


# --------------------------------------------------------------------
# Topological classes (RDKit canonical rank)
# --------------------------------------------------------------------


def topological_classes(
    mol: Any,
    *,
    element: Optional[str] = None,
) -> list[list[int]]:
    """Group atoms by RDKit's chirality-aware canonical rank.

    Each returned sublist is a topological-equivalence class: atoms with
    the same canonical rank, sorted ascending by atom index. The outer
    list is sorted by the smallest atom index in each class (so the
    result is a stable canonical ordering).

    ``element``, if provided, filters atoms by symbol (``"H"``, ``"C"``,
    …) before ranking — the rank is still computed over the full
    molecule (so chirality propagates correctly), but classes
    containing zero atoms of the requested element are dropped.

    Why ``includeChirality=True``: the diastereotopic CH₂ pair in a
    chiral environment gets distinct ranks only when chirality is
    propagated. Without it, the two methylene H's get the same rank
    and we'd erroneously collapse them into one HARD group.
    """
    from rdkit import Chem  # type: ignore[import-not-found]

    ranks = list(
        Chem.CanonicalRankAtoms(mol, breakTies=False, includeChirality=True)
    )
    by_rank: dict[int, list[int]] = {}
    for atom_idx, rank in enumerate(ranks):
        atom = mol.GetAtomWithIdx(atom_idx)
        if element is not None and atom.GetSymbol() != element:
            continue
        by_rank.setdefault(int(rank), []).append(atom_idx)

    return sorted(
        (sorted(atoms) for atoms in by_rank.values()),
        key=lambda lst: lst[0],
    )


# --------------------------------------------------------------------
# Magnetic-equivalence test
# --------------------------------------------------------------------


def magnetic_equivalence_test(
    *,
    class_atoms: list[int],
    other_atoms: list[int],
    j_matrix: dict[tuple[int, int], float],
    tol_hz: float = 0.5,
) -> bool:
    """True iff every atom in ``class_atoms`` has the same J-vector to
    ``other_atoms`` (within ``tol_hz``).

    The J-vector for atom ``a`` is ``[J(a, b) for b in other_atoms]``,
    walking ``other_atoms`` in caller-provided order. Two J-vectors
    "match" when:

    * Both entries are ``None`` (parser found no J for that pair); or
    * Both entries are floats and differ by at most ``tol_hz``.

    A ``None`` vs a float is a mismatch — treating "missing data" as
    "implicitly zero" risks calling AA'XX' patterns magnetically
    equivalent just because the parser dropped a long-range J row.

    Class size 0 or 1 is trivially equivalent (no pairs to compare).

    ``j_matrix`` is keyed by canonical pair ``(min(i, j), max(i, j))``
    so callers don't have to canonicalize before passing.
    """
    if len(class_atoms) <= 1:
        return True

    def j_at(a: int, b: int) -> Optional[float]:
        return j_matrix.get((min(a, b), max(a, b)))

    ref_atom = class_atoms[0]
    ref_vec = [j_at(ref_atom, b) for b in other_atoms]
    for a in class_atoms[1:]:
        a_vec = [j_at(a, b) for b in other_atoms]
        for ref_v, a_v in zip(ref_vec, a_vec):
            if (ref_v is None) != (a_v is None):
                return False
            if ref_v is not None and a_v is not None:
                if abs(ref_v - a_v) > float(tol_hz):
                    return False
    return True


def classify_class_tier(
    *,
    class_atoms: list[int],
    other_atoms: list[int],
    j_matrix: dict[tuple[int, int], float],
    tol_hz: float = 0.5,
) -> Tier:
    """Map a topological class onto a :class:`Tier`.

    Trivial size-1 class → :attr:`Tier.NONE` (no equivalence question).
    Larger classes hit :func:`magnetic_equivalence_test` and dispatch
    HARD / SOFT.
    """
    if len(class_atoms) == 0:
        raise ValueError("classify_class_tier: class_atoms must be non-empty")
    if len(class_atoms) == 1:
        return Tier.NONE
    if magnetic_equivalence_test(
        class_atoms=class_atoms,
        other_atoms=other_atoms,
        j_matrix=j_matrix,
        tol_hz=tol_hz,
    ):
        return Tier.HARD
    return Tier.SOFT


# --------------------------------------------------------------------
# Orchestrator: classes → labeled groups with averaged shifts + J's
# --------------------------------------------------------------------


def _mol_has_stereo(mol: Any) -> bool:
    """True if ``mol`` carries any stereochemistry that could create
    rank-indistinguishable diastereotopic atoms.

    Checks for atom-level chirality tags (``@``-bearing centers in the
    SMILES) and stereo bonds (E/Z designations). A molecule with
    neither has no diastereotopicity that would be invisible to
    :func:`topological_classes` — same-rank atoms are then chemically
    equivalent, and any shift variation within a class is sampling
    noise that the equivalence detector should not interpret as a
    splitting signal.

    Lazy RDKit import: this function is only called from
    :func:`compute_equivalence_groups` (which already needs RDKit),
    so the import is inside the function body to keep the module
    loadable without RDKit.
    """
    from rdkit import Chem  # type: ignore[import-not-found]

    for atom in mol.GetAtoms():
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            return True
    for bond in mol.GetBonds():
        if bond.GetStereo() != Chem.BondStereo.STEREONONE:
            return True
    return False


def _is_force_hard_class(
    class_atoms: list[int],
    mol: Any,
    stereo_present: bool,
) -> bool:
    """True if a class is structurally guaranteed to be magnetically
    equivalent regardless of J asymmetry from frozen-geometry DFT.

    Two cases:

    * **Methyl-style (size ≥ 3, same parent C, single-bond
      connectivity).** C₃ rotation exchanges all 3 H's on the NMR
      timescale; equivalence holds in any molecule, chiral or
      achiral. (Methyls in 2-bromobutane (S) included.)

    * **Achiral methylene-style (size 2, same parent C, single-bond
      connectivity, no stereo in the molecule).** The two H's
      exchange under C₂ rotation of the parent C, and in an achiral
      surrounding they see the same environment from both positions.
      In a chiral molecule the H's may be diastereotopic — falls
      through to the normal mag-equiv path, where the data-aware
      refinement can still split them via shift spread.

    Note: this is structurally restrictive on purpose. We're NOT
    saying "any same-parent class is HARD" — that would wrongly
    collapse diastereotopic CH₂ in chiral molecules. We're saying
    "same-parent classes that are structurally rotation-symmetric
    are HARD even when DFT data looks asymmetric."
    """
    if len(class_atoms) < 2:
        return False

    # All members must be H atoms attached to the same parent atom.
    parents: set[int] = set()
    for a in class_atoms:
        atom = mol.GetAtomWithIdx(a)
        if atom.GetSymbol() != "H":
            return False
        neighbors = atom.GetNeighbors()
        if len(neighbors) != 1:
            return False
        parents.add(neighbors[0].GetIdx())
    if len(parents) != 1:
        return False

    # Parent must be a carbon with at least one single-bond
    # non-hydrogen connection to the rest of the molecule (so it's
    # genuinely rotatable, not a bare CH4 / CH3 anion).
    from rdkit import Chem  # type: ignore[import-not-found]

    parent_idx = parents.pop()
    parent = mol.GetAtomWithIdx(parent_idx)
    if parent.GetSymbol() != "C":
        return False
    has_single_to_heavy = any(
        b.GetBondType() == Chem.BondType.SINGLE
        and b.GetOtherAtom(parent).GetSymbol() != "H"
        for b in parent.GetBonds()
    )
    if not has_single_to_heavy:
        return False

    # Methyl-style: always force HARD (C₃ rotation universal).
    if len(class_atoms) >= 3:
        return True

    # Methylene-style: force HARD only when the molecule lacks stereo.
    # In chiral environments the two H's may be diastereotopic; let
    # the data-aware refinement decide via shift spread.
    return not stereo_present


def _avg_or_none(vals: list[Optional[float]]) -> Optional[float]:
    """Mean over the non-None entries, or None if every entry is None."""
    finite = [v for v in vals if isinstance(v, (int, float))]
    if not finite:
        return None
    return float(sum(finite)) / len(finite)


def _avg_pairwise_j(
    *,
    atoms_a: tuple[int, ...],
    atoms_b: tuple[int, ...],
    j_matrix: dict[tuple[int, int], float],
) -> Optional[float]:
    """Average J(a, b) over a ∈ atoms_a, b ∈ atoms_b (skipping None entries).

    Used to compute the mnova ``<jCoupling>`` value between two groups
    when at least one is HARD (collapsed). Symmetric in atoms_a vs
    atoms_b. Returns ``None`` when no pair has a parseable J — the
    caller decides whether to omit the ``<jCoupling>`` element entirely
    or render it as ``0.0``.
    """
    vals: list[float] = []
    for a in atoms_a:
        for b in atoms_b:
            if a == b:
                continue
            v = j_matrix.get((min(a, b), max(a, b)))
            if v is not None:
                vals.append(float(v))
    if not vals:
        return None
    return sum(vals) / len(vals)


def compute_equivalence_groups(
    *,
    mol: Any,
    element: str,
    shifts_by_atom: dict[int, float],
    j_matrix: dict[tuple[int, int], float],
    tol_jcoupling_hz: float = 0.5,
    tol_shift_ppm: float = 0.05,
) -> list[EquivalenceGroup]:
    """End-to-end: build equivalence groups for one nucleus type.

    Pipeline:

    1. Compute topological classes for atoms of the requested ``element``
       (via :func:`topological_classes`).
    2. **Data-aware refinement.** RDKit's ``CanonicalRankAtoms``
       propagates chirality only through atom-level CIP flags (the ``@``
       tag) — it does NOT split prochiral H's whose parent C is next to
       a chiral center. Diastereotopic CH₂ pairs in chiral environments
       therefore appear as one topological class of size 2 even though
       they're chemically distinct. We compensate by splitting any
       class whose *DFT-computed* shifts span more than
       ``tol_shift_ppm`` into singletons. The DFT calculation already
       respects 3D geometry, so the shift spread is a reliable signal
       that the class members are NOT truly equivalent. Genuinely
       homotopic / enantiotopic classes (methyls, achiral CH₂) have
       sub-tolerance shift spread after Boltzmann averaging, so this
       refinement leaves them alone.
    3. Within each (refined) class, classify the tier
       (:func:`classify_class_tier`).
    4. Materialize the per-tier groups:

       * **HARD** → one group with ``atom_indices = tuple(class)``,
         ``shift = ⟨σᵢ⟩`` over class members.
       * **SOFT** → ``len(class)`` groups, each with one atom; shift is
         the same average over the class (since DFT-noise level
         differences within an equivalent class are exactly what
         averaging is for).
       * **NONE** → one group with the single atom; shift is its raw
         per-atom value.

    5. Sort groups by their lowest atom index, assign Excel-style names.
    6. Compute pairwise J couplings between distinct groups (averaged
       over the involved atoms via :func:`_avg_pairwise_j`).

    Returned groups are immutable dataclasses ready for the mnova XML
    emitter. Caller is responsible for converting ``shift_avg_ppm``
    from raw σ to predicted δ via the linear-scaling calibration BEFORE
    calling this function — equivalence is a structural / averaging
    operation that doesn't know about cheshire/Bally-Rablen scaling.
    """
    elem = str(element).strip()
    if not elem:
        raise ValueError("compute_equivalence_groups: element must be non-empty")

    classes = topological_classes(mol, element=elem)
    if not classes:
        return []

    # Data-aware refinement (see step 2 in docstring above).
    #
    # The refinement only matters when the molecule has stereo info
    # that could create rank-indistinguishable diastereotopic atoms
    # (chiral centers, stereo bonds). For purely achiral molecules,
    # same-rank atoms ARE chemically equivalent under fast rotation
    # / NMR timescale by symmetry — any shift variation within a
    # class is DFT noise from a finite-conformer Boltzmann ensemble
    # (e.g., 2-conformer methyl groups don't sample C3 rotation
    # uniformly), not a real diastereotopic signal. Skipping the
    # refinement in the achiral case prevents false-splitting of
    # methyls / methylenes when sampling is incomplete.
    if _mol_has_stereo(mol):
        refined: list[list[int]] = []
        for cls in classes:
            if len(cls) <= 1:
                refined.append(cls)
                continue
            vals = [shifts_by_atom.get(a) for a in cls]
            finite = [v for v in vals if isinstance(v, (int, float))]
            if len(finite) < 2 or (max(finite) - min(finite)) <= tol_shift_ppm:
                refined.append(cls)
            else:
                # Class members have meaningfully different DFT shifts —
                # they're really distinct (e.g., diastereotopic CH₂ in a
                # chiral environment). Split into singletons; each will
                # later classify as Tier.NONE.
                for a in cls:
                    refined.append([a])
        classes = refined

    # All atoms of this element across every class. Used as the universe
    # for the magnetic-equivalence test ("J-vector to other atoms").
    all_elem_atoms: list[int] = sorted(
        idx for cls in classes for idx in cls
    )

    # Stage 1: classify each class + collect its atoms.
    #
    # Structural force-HARD shortcut: H atoms sharing a parent C with
    # rotatable single-bond connectivity are equivalent by molecular
    # symmetry (C₃ rotation for methyls, C₂ for symmetric methylenes)
    # regardless of how asymmetric the DFT-computed J's look across
    # frozen conformer geometries. The mag-equiv test fails for these
    # in practice — real ethanol methyl J's vary 5–15 Hz across H
    # atoms in a 2-conformer ensemble because rotation isn't sampled.
    # Bypass the test for the cases where chemistry is unambiguous:
    # methyls always, methylenes when the molecule is achiral.
    stereo_present = _mol_has_stereo(mol)
    classified: list[tuple[Tier, list[int]]] = []
    for cls in classes:
        if _is_force_hard_class(cls, mol, stereo_present):
            classified.append((Tier.HARD, cls))
            continue
        others = [a for a in all_elem_atoms if a not in cls]
        tier = classify_class_tier(
            class_atoms=cls,
            other_atoms=others,
            j_matrix=j_matrix,
            tol_hz=tol_jcoupling_hz,
        )
        classified.append((tier, cls))

    # Stage 2: materialize raw groups (no names yet).
    raw: list[tuple[Tier, list[int], tuple[int, ...]]] = []
    for tier, cls in classified:
        if tier == Tier.HARD:
            raw.append((tier, cls, tuple(cls)))
        elif tier == Tier.SOFT:
            for atom in cls:
                raw.append((tier, cls, (atom,)))
        else:  # NONE
            raw.append((tier, cls, tuple(cls)))

    # Stage 3: sort by min atom index (canonical, stable), then label.
    raw.sort(key=lambda r: min(r[2]))
    labels = assign_group_labels(len(raw))

    # Atom → group_name mapping for the J-coupling fill-in below.
    atom_to_group: dict[int, str] = {}
    for label, (_tier, _cls, atoms) in zip(labels, raw):
        for a in atoms:
            atom_to_group[a] = label

    # Stage 4: build the EquivalenceGroup objects, computing shifts
    # (averaged over the topological CLASS for SOFT/HARD; raw for NONE)
    # and inter-group J's (averaged over atoms-in-each-group).
    groups: list[EquivalenceGroup] = []
    for label, (tier, cls, atoms) in zip(labels, raw):
        # Shift averaging:
        # * HARD: average over class members (== atoms here, since the
        #   whole class collapsed into one group).
        # * SOFT: average over the topological CLASS even though this
        #   group only owns one atom — class members are equal under
        #   the equivalence by definition; averaging removes DFT noise.
        # * NONE: just the single atom's value.
        shift_source = cls if tier in (Tier.HARD, Tier.SOFT) else list(atoms)
        shift_vals = [shifts_by_atom.get(a) for a in shift_source]
        shift_avg = _avg_or_none(shift_vals)
        if shift_avg is None:
            # Defensive: an atom with no parseable shielding shouldn't
            # land here, but if it does, skip the group rather than
            # emitting NaN. Caller will see fewer groups than expected
            # and can flag it.
            continue

        # J-couplings to OTHER groups. For each other group, average
        # J(a, b) over a ∈ this group's atoms, b ∈ other group's atoms.
        # We skip self-couplings (intra-group); mnova handles that
        # implicitly via the spin-equivalence count.
        j_couplings: dict[str, float] = {}
        for other_label, (_o_tier, _o_cls, other_atoms) in zip(labels, raw):
            if other_label == label:
                continue
            j_avg = _avg_pairwise_j(
                atoms_a=atoms,
                atoms_b=other_atoms,
                j_matrix=j_matrix,
            )
            if j_avg is not None:
                j_couplings[other_label] = j_avg

        groups.append(
            EquivalenceGroup(
                name=label,
                element=elem,
                atom_indices=tuple(atoms),
                shift_avg_ppm=float(shift_avg),
                tier=tier,
                j_couplings=j_couplings,
            )
        )

    return groups


__all__ = [
    "EquivalenceGroup",
    "Tier",
    "assign_group_labels",
    "classify_class_tier",
    "compute_equivalence_groups",
    "magnetic_equivalence_test",
    "mol_from_smiles_or_xyz",
    "topological_classes",
]
