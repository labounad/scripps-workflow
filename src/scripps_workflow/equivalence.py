"""Chemical-equivalence detection for NMR spin-system grouping.

The mnova-spinsim XML emitter needs to know, given a molecule + per-atom
chemical shifts + per-pair J couplings, how to bucket the atoms into
``<group>`` blocks. The grouping is the difference between getting the
right multiplet pattern in the simulated spectrum (e.g., a methyl as a
single triplet) and getting nonsense (the methyl as three near-coincident
singlets that smear into a broad blob).

We use a **three-tier classification**:

* **HARD** — chemically and magnetically equivalent for rendering.
  Members collapse into ONE group with ``number=N``; shifts and J's to
  other groups are averaged. Examples: methyl CH₃, equivalent
  gem-dimethyl 6H, tert-butyl 9H, benzene-like fully symmetric Aₙ
  classes, and equivalent 13C intensity groups.

* **SOFT** — chemically equivalent but deliberately kept as separate
  spin groups with one common averaged shift. This preserves magnetic
  inequivalence / second-order topology for cases such as methylene
  H₂, AA′BB′ aromatic patterns, vinyl groups, and mirrored ring
  protons.

* **NONE** — topologically or stereochemically distinct. No averaging.
  Each atom emits as its own group. Catches diastereotopic CH₂ in
  chiral environments, isolated CH protons, and so on.

The dispatch is mechanical:

1. Compute topological classes from RDKit's chirality-aware
   ``CanonicalRankAtoms`` — atoms with the same rank are in one class.
   This catches chemical equivalence cleanly even when the symmetry
   operation is a non-trivial rotation/reflection.

2. **Narrow data-aware refinement.** RDKit's ``includeChirality=True``
   only propagates atom-level CIP flags (``@`` tags); it does not
   reliably split prochiral H's attached to a carbon adjacent to a
   chiral center. We therefore allow DFT shift spread to split only
   same-parent H₂ classes in molecules with stereochemical information.
   Complete methyl-like classes and non-H classes are protected from
   shift-spread splitting because finite conformer ensembles often
   break their symmetry numerically.

3. Classify refined classes structurally rather than trusting noisy
   J-vectors for the collapse decision: rotational methyl-like classes
   and all-of-nucleus symmetric classes become HARD; other multi-proton
   classes become SOFT; singletons become NONE. The parsed/averaged J's
   are still retained in the emitted groups.

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


def _same_parent_hydrogen_class(class_atoms: list[int], mol: Any) -> Optional[int]:
    """Return the shared parent index for a pure H class, else ``None``."""
    if not class_atoms:
        return None

    parents: set[int] = set()
    for a in class_atoms:
        atom = mol.GetAtomWithIdx(a)
        if atom.GetSymbol() != "H":
            return None
        neighbors = atom.GetNeighbors()
        if len(neighbors) != 1:
            return None
        parents.add(neighbors[0].GetIdx())

    if len(parents) != 1:
        return None
    return next(iter(parents))


def _hydrogen_neighbors(parent: Any) -> list[int]:
    """Hydrogen-neighbor indices for ``parent``, sorted for stability."""
    return sorted(
        n.GetIdx() for n in parent.GetNeighbors() if n.GetSymbol() == "H"
    )


def _is_rotationally_hard_hydrogen_class(class_atoms: list[int], mol: Any) -> bool:
    """True for proton classes that should be collapsed structurally.

    This deliberately targets *rapid local rotation / full symmetry*
    cases rather than using noisy DFT J-vectors:

    * one or more complete methyl groups in the same RDKit equivalence
      class: CH₃, gem-dimethyl 6H, tert-butyl 9H, neopentane 12H, …;
    * methane's four equivalent protons.

    The key guard is "complete local set": if a methyl carbon contributes
    to the class, all three of its hydrogens must be in the class. This
    prevents accidental collapsing of partial / prochiral H sets while
    still handling tBu-like multi-methyl classes naturally.
    """
    if len(class_atoms) < 3:
        return False

    class_set = set(class_atoms)
    parent_to_hs: dict[int, list[int]] = {}
    for a in class_atoms:
        atom = mol.GetAtomWithIdx(a)
        if atom.GetSymbol() != "H":
            return False
        neighbors = atom.GetNeighbors()
        if len(neighbors) != 1:
            return False
        parent = neighbors[0]
        if parent.GetSymbol() != "C":
            return False
        parent_to_hs.setdefault(parent.GetIdx(), []).append(a)

    # Methane: one carbon with exactly four hydrogens and no heavy-atom
    # neighbors. It is an uncommon workflow input, but chemically this is
    # the same "all protons equivalent" case and keeps the classifier sane.
    if len(parent_to_hs) == 1:
        parent_idx = next(iter(parent_to_hs))
        parent = mol.GetAtomWithIdx(parent_idx)
        parent_hs = _hydrogen_neighbors(parent)
        heavy_neighbors = [
            n for n in parent.GetNeighbors() if n.GetSymbol() != "H"
        ]
        if (
            len(parent_hs) == 4
            and not heavy_neighbors
            and set(parent_hs) == class_set
        ):
            return True

    # Methyl / multi-methyl classes. Every parent must be a complete CH3
    # unit, and the topological class must contain an integer number of
    # complete methyl units. This covers isolated CH3, equivalent geminal
    # methyls, tert-butyl, neopentane, p-xylene's two methyl groups, etc.
    if len(class_atoms) % 3 != 0:
        return False

    for parent_idx, hs_from_class in parent_to_hs.items():
        parent = mol.GetAtomWithIdx(parent_idx)
        parent_hs = _hydrogen_neighbors(parent)
        if len(parent_hs) != 3:
            return False
        if set(parent_hs) != set(hs_from_class):
            return False

    return True


def _should_split_class_by_shift(
    *,
    cls: list[int],
    mol: Any,
    element: str,
    stereo_present: bool,
    shifts_by_atom: dict[int, float],
    tol_shift_ppm: float,
) -> bool:
    """Whether shift spread should override RDKit's topology.

    The old logic split *any* same-rank class in a stereochemical
    molecule when DFT shifts differed by more than ``tol_shift_ppm``.
    That was too broad: it split methyls in patchouli alcohol and could
    split equivalent 13C atoms just because a finite conformer ensemble
    did not preserve symmetry exactly.

    The robust use case for data-aware splitting is narrow: same-parent
    H₂ classes in molecules with stereochemical information. Those are
    the classic RDKit blind spot for diastereotopic methylene protons.
    Complete methyl-like classes are explicitly protected because methyl
    rotation makes them HARD even in chiral molecules.
    """
    if not stereo_present or len(cls) <= 1:
        return False
    if element != "H":
        return False
    if _is_rotationally_hard_hydrogen_class(cls, mol):
        return False
    if len(cls) != 2:
        return False
    if _same_parent_hydrogen_class(cls, mol) is None:
        return False

    vals = [shifts_by_atom.get(a) for a in cls]
    finite = [v for v in vals if isinstance(v, (int, float))]
    if len(finite) < 2:
        return False
    return (max(finite) - min(finite)) > float(tol_shift_ppm)


def _classify_structural_tier(
    *,
    cls: list[int],
    all_elem_atoms: list[int],
    mol: Any,
    element: str,
) -> Tier:
    """Classify a refined chemical-equivalence class.

    This is intentionally more structural than the older J-vector-only
    approach. DFT J values from one/few frozen conformers are excellent
    for populating couplings, but they are a poor source of truth for
    deciding whether fast-rotating methyl / tBu protons should be
    collapsed. Conversely, non-methyl topological equivalences are safer
    as SOFT groups: they keep one common shift while preserving possible
    AA′BB′ / methylene / ring-coupling topology.

    Policy:

    * singleton → NONE;
    * 13C / other non-proton equivalent atoms → HARD/intensity group;
    * rotational methyl-like H classes → HARD;
    * a class containing every atom of that nucleus in the spin system
      → HARD (benzene, ethene, methane-like fully symmetric cases);
    * all remaining multi-proton classes → SOFT.
    """
    if len(cls) == 0:
        raise ValueError("_classify_structural_tier: cls must be non-empty")
    if len(cls) == 1:
        return Tier.NONE

    elem = str(element).strip()
    if elem != "H":
        return Tier.HARD

    if _is_rotationally_hard_hydrogen_class(cls, mol):
        return Tier.HARD

    cls_set = set(cls)
    all_set = set(all_elem_atoms)
    if cls_set == all_set:
        return Tier.HARD

    return Tier.SOFT


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
    2. **Narrow data-aware refinement.** RDKit's ``CanonicalRankAtoms``
       propagates chirality only through atom-level CIP flags (the ``@``
       tag) and does not reliably split prochiral H₂ classes next to a
       chiral center. If a same-parent H₂ class in a stereochemical
       molecule spans more than ``tol_shift_ppm``, split it into NONE
       singletons. Do not split methyl-like classes or non-H atoms by
       shift spread.
    3. Classify each refined class with the structural HARD/SOFT/NONE
       policy in :func:`_classify_structural_tier`.
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
    # Keep this deliberately narrow. RDKit's practical blind spot here
    # is same-parent H₂ classes in a molecule with stereochemical
    # information (diastereotopic methylene protons). We do NOT split
    # methyl-like classes or non-H atoms by shift spread: those spreads
    # are usually finite-conformer / non-symmetrized-geometry artifacts,
    # and splitting them caused the patchouli alcohol methyl failure.
    stereo_present = _mol_has_stereo(mol)
    refined: list[list[int]] = []
    for cls in classes:
        if _should_split_class_by_shift(
            cls=cls,
            mol=mol,
            element=elem,
            stereo_present=stereo_present,
            shifts_by_atom=shifts_by_atom,
            tol_shift_ppm=tol_shift_ppm,
        ):
            # Same-parent H₂ class in a stereochemical molecule with
            # meaningfully different predicted shifts: treat as genuine
            # diastereotopic / NONE singletons.
            refined.extend([a] for a in cls)
        else:
            refined.append(cls)
    classes = refined

    # All atoms of this element across every class. Used to decide whether
    # a class is the only spin class for that nucleus (e.g., benzene A6).
    all_elem_atoms: list[int] = sorted(
        idx for cls in classes for idx in cls
    )

    # Stage 1: classify each class + collect its atoms.
    #
    # Classification is structural-first. J values are still carried
    # into the final groups, but are no longer allowed to split methyl /
    # tert-butyl classes simply because a finite conformer ensemble did
    # not rotationally average them. Non-methyl multi-proton classes are
    # emitted SOFT by default, which preserves AA′BB′ / ring / methylene
    # coupling topology while giving the class one averaged shift.
    classified: list[tuple[Tier, list[int]]] = []
    for cls in classes:
        tier = _classify_structural_tier(
            cls=cls,
            all_elem_atoms=all_elem_atoms,
            mol=mol,
            element=elem,
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
