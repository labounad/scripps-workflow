"""Tests for ``scripps_workflow.thermo`` stat-mech helpers.

The functions here are stateless and stdlib-only — these tests pin the
numerical contract (Boltzmann normalization, RT·ln(24.46) at 298.15 K,
cumulative weighting in ΔG-ascending order) so any future refactor that
breaks the math gets caught at unit-test time.
"""

from __future__ import annotations

import math

import pytest

from scripps_workflow.thermo import (
    R_KCAL_PER_MOL_K,
    boltzmann_weights,
    cumulative_weights_by_dg,
    rt_ln_24_46_kcal,
)


# --------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------


class TestRConstant:
    def test_r_value_kcal(self):
        # CODATA-derived R in kcal mol⁻¹ K⁻¹ — must match the legacy
        # aggregator value byte-for-byte so a regression test against
        # archival CSVs stays exact.
        assert R_KCAL_PER_MOL_K == pytest.approx(0.00198720425864083)


# --------------------------------------------------------------------
# boltzmann_weights
# --------------------------------------------------------------------


class TestBoltzmannWeights:
    def test_uniform_dg_gives_uniform_weights(self):
        ws = boltzmann_weights([0.0, 0.0, 0.0, 0.0], 298.15)
        assert ws == [pytest.approx(0.25)] * 4

    def test_weights_normalize_to_one(self):
        ws = boltzmann_weights([0.0, 1.0, 2.0, 5.0], 298.15)
        # All finite — weights sum to 1.
        assert sum(ws) == pytest.approx(1.0)

    def test_lower_dg_gets_higher_weight(self):
        ws = boltzmann_weights([0.0, 1.0, 2.0], 298.15)
        assert ws[0] > ws[1] > ws[2] > 0.0

    def test_dg_zero_dominates_at_low_t(self):
        # At very low T the lowest-ΔG conformer should dominate.
        ws = boltzmann_weights([0.0, 5.0], 50.0)
        assert ws[0] > 0.99
        assert ws[1] < 0.01

    def test_high_t_flattens_distribution(self):
        # At very high T the distribution flattens toward equal weights.
        ws_cold = boltzmann_weights([0.0, 1.0, 2.0], 100.0)
        ws_hot = boltzmann_weights([0.0, 1.0, 2.0], 1e5)
        # Hot weights are closer to uniform (1/3 each).
        assert abs(ws_hot[0] - 1 / 3) < abs(ws_cold[0] - 1 / 3)

    def test_none_propagates(self):
        # None entries keep None weights; finite entries normalize among
        # themselves.
        ws = boltzmann_weights([0.0, None, 1.0], 298.15)
        assert ws[1] is None
        # ws[0] + ws[2] should normalize to 1 across the finite slots.
        finite = [w for w in ws if w is not None]
        assert sum(finite) == pytest.approx(1.0)

    def test_all_none_returns_all_none(self):
        ws = boltzmann_weights([None, None], 298.15)
        assert ws == [None, None]

    def test_empty_list(self):
        ws = boltzmann_weights([], 298.15)
        assert ws == []

    def test_negative_dg_works(self):
        # ΔG < 0 means a conformer below the (badly chosen) reference.
        # The math is the same; the function shouldn't reject it.
        ws = boltzmann_weights([-1.0, 0.0, 1.0], 298.15)
        assert sum(ws) == pytest.approx(1.0)
        assert ws[0] > ws[1] > ws[2]

    def test_zero_temperature_raises(self):
        with pytest.raises(ValueError, match="positive"):
            boltzmann_weights([0.0, 1.0], 0.0)

    def test_negative_temperature_raises(self):
        with pytest.raises(ValueError, match="positive"):
            boltzmann_weights([0.0, 1.0], -100.0)

    def test_extreme_dg_does_not_crash(self):
        # ΔG = +1e5 kcal/mol → exp(-huge) underflows to 0. Function
        # should still return finite weights for the lowest, and 0 (or
        # very small) for the upper.
        ws = boltzmann_weights([0.0, 1e5], 298.15)
        assert ws[0] == pytest.approx(1.0)
        assert ws[1] < 1e-50


# --------------------------------------------------------------------
# rt_ln_24_46_kcal
# --------------------------------------------------------------------


class TestRtLn2446:
    def test_value_at_298_15(self):
        # The textbook 1 atm → 1 M correction at 298.15 K is roughly
        # +1.894 kcal/mol. Lock the value so a constant change here is
        # noticed.
        v = rt_ln_24_46_kcal(298.15)
        assert v == pytest.approx(1.8941975683244532)

    def test_scales_with_t(self):
        # RT·ln(24.46): ratio at T1 / T2 == T1 / T2.
        v_298 = rt_ln_24_46_kcal(298.15)
        v_596 = rt_ln_24_46_kcal(596.30)
        assert v_596 / v_298 == pytest.approx(2.0, rel=1e-6)

    def test_zero_temperature_raises(self):
        with pytest.raises(ValueError, match="positive"):
            rt_ln_24_46_kcal(0.0)

    def test_explicit_formula(self):
        # Bind to the closed form so refactors that swap log bases or
        # constants get noticed.
        T = 310.0
        expected = R_KCAL_PER_MOL_K * T * math.log(24.46)
        assert rt_ln_24_46_kcal(T) == pytest.approx(expected)


# --------------------------------------------------------------------
# cumulative_weights_by_dg
# --------------------------------------------------------------------


class TestCumulativeWeightsByDg:
    def test_ascending_dg_is_monotone(self):
        # When ΔG is already ascending, cumulative is just a running
        # sum from low to high.
        dG = [0.0, 1.0, 2.0]
        ws = boltzmann_weights(dG, 298.15)
        cum = cumulative_weights_by_dg(dG, ws)
        # cum[0] == ws[0]; cum[1] == ws[0] + ws[1]; cum[2] == 1.0.
        assert cum[0] == pytest.approx(ws[0])
        assert cum[1] == pytest.approx(ws[0] + ws[1])
        assert cum[2] == pytest.approx(1.0)

    def test_unsorted_input_sorted_internally(self):
        # Function sorts internally by ΔG, so the lowest-ΔG slot gets
        # its own weight as cumulative regardless of input order.
        dG = [2.0, 0.0, 1.0]
        ws = boltzmann_weights(dG, 298.15)
        cum = cumulative_weights_by_dg(dG, ws)
        # idx=1 has dG=0 → cum[1] == ws[1]
        assert cum[1] == pytest.approx(ws[1])
        # idx=2 has dG=1 → cum[2] == ws[1] + ws[2]
        assert cum[2] == pytest.approx(ws[1] + ws[2])
        # idx=0 has dG=2 (highest) → cum[0] == 1.0
        assert cum[0] == pytest.approx(1.0)

    def test_none_propagates(self):
        # An entry with None ΔG OR None weight gets None cumulative.
        dG = [0.0, None, 1.0]
        ws = boltzmann_weights(dG, 298.15)
        cum = cumulative_weights_by_dg(dG, ws)
        assert cum[1] is None
        # The two finite entries' cumulative should sum to ~1.0 at the
        # highest-ΔG entry.
        assert max(c for c in cum if c is not None) == pytest.approx(1.0)

    def test_all_none(self):
        cum = cumulative_weights_by_dg([None, None], [None, None])
        assert cum == [None, None]

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            cumulative_weights_by_dg([0.0, 1.0], [0.5])

    def test_empty(self):
        assert cumulative_weights_by_dg([], []) == []
