"""Statistical-mechanics helpers shared across thermo-aware nodes.

Right now only :mod:`scripps_workflow.nodes.thermo_aggregate` consumes
these; they live in their own module rather than inline so the aggregator
node stays focused on I/O + manifest wiring, and so the marc / prism /
GUI sides can pull them in later without picking up the node-class
machinery.

All helpers are stateless and stdlib-only. ``None`` propagates: when an
input list contains ``None`` (a conformer the aggregator couldn't get a
Gibbs energy for) the corresponding output slot is also ``None`` so the
aggregator can keep records 1:1 with task indices.
"""

from __future__ import annotations

import math
from typing import Optional


#: Ideal-gas constant in kcal mol⁻¹ K⁻¹. Matches the value the legacy
#: ``orca_thermo_aggregator`` script used so a regression test against
#: archival CSVs is exact.
R_KCAL_PER_MOL_K: float = 0.00198720425864083


def boltzmann_weights(
    delta_g_kcal: list[Optional[float]],
    temperature_k: float,
) -> list[Optional[float]]:
    """Compute normalized Boltzmann weights from ΔG values in kcal/mol.

    ``w_i = exp(-ΔG_i / RT) / Σ_j exp(-ΔG_j / RT)`` over the conformers
    with finite ΔG. Conformers with ``None`` ΔG keep ``None`` weight (the
    aggregator surfaces them as separate failure records).

    Returns a list of the same length as ``delta_g_kcal``. If every
    finite weight underflows to zero (extreme ΔG / very low T) the
    function returns ``None`` for every position rather than dividing by
    zero — this is a defensive fallback; in practice well-converged
    conformer ensembles span a few kcal/mol and never hit this path.

    Args:
        delta_g_kcal: per-conformer ΔG values (kcal/mol), with ``None``
            for missing entries.
        temperature_k: temperature in Kelvin. Must be positive.

    Raises:
        ValueError: if ``temperature_k`` is non-positive.
    """
    T = float(temperature_k)
    if T <= 0:
        raise ValueError(f"temperature_k must be positive, got {T!r}")
    beta = 1.0 / (R_KCAL_PER_MOL_K * T)

    finite_exps: list[float] = []
    for d in delta_g_kcal:
        if d is None:
            continue
        try:
            finite_exps.append(math.exp(-float(d) * beta))
        except OverflowError:
            # exp(-very_negative) → +inf; treat as a single dominant
            # weight (1.0). exp(-very_positive) → 0.0 already.
            finite_exps.append(float("inf"))

    denom = sum(finite_exps) if finite_exps else 0.0
    if not math.isfinite(denom) or denom <= 0.0:
        return [None] * len(delta_g_kcal)

    out: list[Optional[float]] = []
    j = 0
    for d in delta_g_kcal:
        if d is None:
            out.append(None)
        else:
            w = finite_exps[j] / denom
            out.append(float(w))
            j += 1
    return out


def rt_ln_24_46_kcal(temperature_k: float) -> float:
    """Standard-state correction RT·ln(24.46), in kcal/mol.

    Used to convert a 1 atm gas-phase standard state to a 1 mol/L
    solution standard state (the Ben-Naim convention). At 298.15 K this
    evaluates to ~+1.894 kcal/mol. Because the correction is a constant
    additive shift across all conformers, it cancels out of relative ΔG
    and Boltzmann weights — but downstream comparisons against
    experimental free energies of solvation need it.
    """
    T = float(temperature_k)
    if T <= 0:
        raise ValueError(f"temperature_k must be positive, got {T!r}")
    return R_KCAL_PER_MOL_K * T * math.log(24.46)


def cumulative_weights_by_dg(
    delta_g_kcal: list[Optional[float]],
    weights: list[Optional[float]],
) -> list[Optional[float]]:
    """Running sum of Boltzmann weights, sorted by ascending ΔG.

    Returns a list aligned with the input order: position ``i`` carries
    the cumulative weight (in [0, 1]) you'd see after enumerating all
    conformers up to and including ``i`` in ΔG-ascending order. ``None``
    entries (missing ΔG OR missing weight) keep ``None`` cumulative.

    Useful for "smallest set of conformers covering 90% of the
    Boltzmann population" thresholding — sort by cumulative ascending
    and stop at the first cumulative ≥ 0.9.
    """
    if len(delta_g_kcal) != len(weights):
        raise ValueError(
            f"length mismatch: {len(delta_g_kcal)} ΔG vs {len(weights)} weights"
        )

    n = len(delta_g_kcal)
    sortable: list[tuple[int, float]] = []
    for i, d in enumerate(delta_g_kcal):
        if d is None or weights[i] is None:
            continue
        sortable.append((i, float(d)))
    sortable.sort(key=lambda t: t[1])

    cum: list[Optional[float]] = [None] * n
    running = 0.0
    for i, _d in sortable:
        w = weights[i]
        if w is None:
            continue
        running += float(w)
        cum[i] = running
    return cum


__all__ = [
    "R_KCAL_PER_MOL_K",
    "boltzmann_weights",
    "cumulative_weights_by_dg",
    "rt_ln_24_46_kcal",
]
