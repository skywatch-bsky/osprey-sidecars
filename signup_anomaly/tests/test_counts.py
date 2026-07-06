# pattern: Functional Core
"""Tests for dispersion-aware count p-value calculation."""

from __future__ import annotations

import pytest
from scipy.stats import nbinom, poisson
import scipy.special

from signup_anomaly.counts import count_p_value


class TestCountPValue:
    """Test dispersion-aware count test: NB with Poisson fallback."""

    def test_ac1_1_nb_hand_computed_example(self) -> None:
        """AC1.1: Hand-computed NB example.

        count_p_value(3, 2.0, 4.0) should give NB(r=2, p=0.5).
        For NB(r=2, p=0.5), P(X=k) = (k+1)·0.25·0.5^k.
        P(X≥3) = 1 - (P(X=0) + P(X=1) + P(X=2))
               = 1 - (0.25 + 0.125 + 0.0625) = 1 - 0.4375 = 0.5625

        Actually computing more carefully:
        r = 2, p = 0.5
        P(X=0) = 0.5^2 = 0.25
        P(X=1) = 2 · 0.5 · 0.5^2 = 0.25
        P(X=2) = 3 · 0.5^2 · 0.5^2 = 0.1875
        P(X≥3) = 1 - (0.25 + 0.25 + 0.1875) = 0.3125
        """
        result = count_p_value(3, 2.0, 4.0)
        assert result == pytest.approx(0.3125, abs=1e-12)

    def test_ac1_2_poisson_fallback_variance_less_than_mean(self) -> None:
        """AC1.2: Poisson fallback when variance <= mean."""
        result = count_p_value(5, 3.0, 2.0)
        expected = float(poisson.sf(4, 3.0))
        assert result == pytest.approx(expected, rel=1e-10)

    def test_ac1_2_poisson_fallback_variance_equals_mean(self) -> None:
        """AC1.2: Poisson fallback when variance == mean."""
        result = count_p_value(5, 3.0, 3.0)
        expected = float(poisson.sf(4, 3.0))
        assert result == pytest.approx(expected, rel=1e-10)

    def test_ac1_2_poisson_fallback_variance_none(self) -> None:
        """AC1.2: Poisson fallback when variance is None."""
        result = count_p_value(5, 3.0, None)
        expected = float(poisson.sf(4, 3.0))
        assert result == pytest.approx(expected, rel=1e-10)

    def test_ac1_3_non_integer_r_regression_guard(self) -> None:
        """AC1.3: Non-integer r produces valid p-value in (0, 1] with scipy#16120 guard.

        mean=3.0, variance=5.0 gives r = 9/2 = 4.5, p = 0.6.
        Result should match the exact incomplete-beta identity:
        P(X >= observed) = betainc(observed, r, 1 - p)
        """
        mean = 3.0
        variance = 5.0
        observed = 7
        r = mean * mean / (variance - mean)
        p = mean / variance

        result = count_p_value(observed, mean, variance)

        # Result should be in valid range
        assert 0.0 < result <= 1.0

        # Should match the exact incomplete-beta identity (cross-check)
        expected = float(scipy.special.betainc(observed, r, 1.0 - p))
        assert result == pytest.approx(expected, abs=1e-10)

        # Sanity: should not match nbdtrc truncation (which would give ~0.05476 instead)
        # nbdtrc truncates r to int(4.5) = 4, which would be wrong
        truncated_r = int(r)
        wrong_via_truncation = float(scipy.special.betainc(observed, truncated_r, 1.0 - p))
        assert result != pytest.approx(wrong_via_truncation, abs=1e-3)

    def test_ac1_4_mean_zero(self) -> None:
        """AC1.4: mean <= 0 returns 1.0."""
        result = count_p_value(5, 0.0, 4.0)
        assert result == 1.0

    def test_ac1_4_mean_negative(self) -> None:
        """AC1.4: negative mean returns 1.0."""
        result = count_p_value(5, -1.0, None)
        assert result == 1.0

    def test_ac1_4_observed_zero_nb(self) -> None:
        """AC1.4: observed = 0 never flags in NB branch (sf(-1, ...) = 1.0)."""
        result = count_p_value(0, 10.0, 20.0)
        assert result == 1.0

    def test_ac1_4_observed_zero_poisson(self) -> None:
        """AC1.4: observed = 0 never flags in Poisson branch."""
        result = count_p_value(0, 10.0, None)
        assert result == 1.0

    def test_sanity_p_decreases_with_observed_increase(self) -> None:
        """Sanity: p-value should decrease as observed count increases."""
        p_low_observed = count_p_value(10, 5.0, 10.0)
        p_high_observed = count_p_value(20, 5.0, 10.0)
        assert p_high_observed < p_low_observed

    def test_sanity_nb_larger_p_than_poisson_for_overdispersed(self) -> None:
        """Sanity: NB (variance > mean) gives larger p than Poisson at same mean.

        This is the entire point of the dispersion-aware change.
        """
        observed = 15
        mean = 5.0
        variance_overdispersed = 15.0

        p_nb = count_p_value(observed, mean, variance_overdispersed)
        p_poisson = float(poisson.sf(observed - 1, mean))

        assert p_nb > p_poisson
