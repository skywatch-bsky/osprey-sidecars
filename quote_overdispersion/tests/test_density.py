# pattern: Functional Core
"""Tests for beta-binomial density p-value calculation."""

from __future__ import annotations

import pytest
from scipy.stats import betabinom, binom

from quote_overdispersion.density import density_p_value


class TestDensityPValue:
    """Test one-sided beta-binomial density test with binomial fallback."""

    def test_ac3_1_beta_binomial_hand_verified_example(self) -> None:
        """AC3.1: Hand-verified beta-binomial example.

        density_p_value(9, 10, 0.5, 0.05):
        mu = 0.5, density_variance = 0.05
        M = 0.5 * (1 - 0.5) / 0.05 - 1 = 0.25/0.05 - 1 = 5 - 1 = 4
        alpha = 0.5 * 4 = 2
        beta = 0.5 * 4 = 2

        For BetaBinom(n=10, alpha=2, beta=2):
        P(X >= 9) = P(X=9) + P(X=10)
        P(X=9) = 10 * B(11, 3) / B(2, 2) = 10 * (10! * 2! * 2! / 12!) / (1! * 1! / 2!)
               = 10 * (10! * 4 / 12!) / 1

        Using the formula: P(X=k) = C(n,k) * B(k+a, n-k+b) / B(a,b)
        This gives P(X >= 9) ≈ 186/1716 ≈ 0.108392
        """
        result = density_p_value(9, 10, 0.5, 0.05)
        expected = 186 / 1716
        assert result == pytest.approx(expected, rel=1e-9)

    def test_ac3_1_betabinom_sf_equivalence(self) -> None:
        """AC3.1: Result matches betabinom.sf with correct alpha/beta wiring."""
        unique_sharers = 9
        total_shares = 10
        expected_density = 0.5
        density_variance = 0.05

        result = density_p_value(unique_sharers, total_shares, expected_density, density_variance)

        # Verify the alpha/beta calculation
        mu = expected_density
        m = mu * (1.0 - mu) / density_variance - 1.0
        alpha = mu * m
        beta = (1.0 - mu) * m

        expected = float(betabinom.sf(unique_sharers - 1, total_shares, alpha, beta))
        assert result == pytest.approx(expected, abs=1e-10)

    def test_ac3_2_binomial_fallback_variance_none(self) -> None:
        """AC3.2: Binomial fallback when variance is None."""
        result = density_p_value(9, 10, 0.5, None)
        expected = float(binom.sf(8, 10, 0.5))
        assert result == pytest.approx(expected, rel=1e-10)

    def test_ac3_2_binomial_fallback_variance_zero(self) -> None:
        """AC3.2: Binomial fallback when variance is zero (M <= 0)."""
        result = density_p_value(9, 10, 0.5, 0.0)
        expected = float(binom.sf(8, 10, 0.5))
        assert result == pytest.approx(expected, rel=1e-10)

    def test_ac3_2_binomial_fallback_variance_too_high(self) -> None:
        """AC3.2: Binomial fallback when variance >= mu(1-mu) (M <= 0)."""
        # For mu=0.5, mu(1-mu)=0.25, so variance >= 0.25 should fallback
        result = density_p_value(9, 10, 0.5, 0.3)
        expected = float(binom.sf(8, 10, 0.5))
        assert result == pytest.approx(expected, rel=1e-10)

    def test_ac3_3_one_sided_at_baseline(self) -> None:
        """AC3.3: One-sided: observed density at baseline returns 1.0."""
        # density = 5/10 = 0.5, expected_density = 0.5 -> at baseline
        result = density_p_value(5, 10, 0.5, 0.05)
        assert result == 1.0

    def test_ac3_3_one_sided_below_baseline(self) -> None:
        """AC3.3: One-sided: observed density below baseline returns 1.0."""
        # density = 3/10 = 0.3, expected_density = 0.5 -> below baseline
        result = density_p_value(3, 10, 0.5, 0.05)
        assert result == 1.0

    def test_ac3_3_one_sided_with_high_expected_density(self) -> None:
        """AC3.3: One-sided behavior with expected_density=1.0."""
        # When expected_density = 1.0, observed <= 1.0 always returns 1.0
        # because we clamp mu to 1.0 and then the condition checks observed/total <= mu
        result = density_p_value(9, 10, 1.0, 0.05)
        assert result == 1.0

    def test_ac3_4_expected_density_zero(self) -> None:
        """AC3.4: expected_density <= 0 returns 1.0."""
        result = density_p_value(9, 10, 0.0, 0.05)
        assert result == 1.0

    def test_ac3_4_expected_density_negative(self) -> None:
        """AC3.4: negative expected_density returns 1.0."""
        result = density_p_value(9, 10, -0.1, None)
        assert result == 1.0

    def test_ac3_4_total_shares_zero(self) -> None:
        """AC3.4: total_shares == 0 returns 1.0."""
        result = density_p_value(0, 0, 0.5, 0.05)
        assert result == 1.0

    def test_fat_tail_sanity_beta_binomial_vs_binomial(self) -> None:
        """Sanity: Beta-binomial p-value exceeds binomial for same inputs.

        This is the point of using beta-binomial: overdispersion in density
        (across days) makes extreme density observations less surprising.
        """
        unique_sharers = 9
        total_shares = 10
        expected_density = 0.5
        density_variance = 0.05

        p_beta_binomial = density_p_value(unique_sharers, total_shares, expected_density, density_variance)
        p_binomial = float(binom.sf(unique_sharers - 1, total_shares, expected_density))

        assert p_beta_binomial > p_binomial

    def test_density_monotonicity(self) -> None:
        """Sanity: p-value decreases as observed density increases above baseline."""
        base_density = 0.5
        total_shares = 100
        variance = 0.02

        # Higher unique sharers -> higher density -> lower p-value
        p_lower_sharers = density_p_value(60, total_shares, base_density, variance)
        p_higher_sharers = density_p_value(70, total_shares, base_density, variance)

        assert p_higher_sharers < p_lower_sharers

    def test_edge_case_mu_clamping(self) -> None:
        """Edge: expected_density > 1.0 gets clamped to 1.0."""
        # When expected_density = 1.5, it should be clamped to mu = 1.0
        # Then unique_sharers/total_shares <= 1.0 should always be true
        # So this should return 1.0
        result = density_p_value(9, 10, 1.5, 0.05)
        assert result == 1.0

    def test_edge_case_density_variance_negative(self) -> None:
        """Edge: Negative variance should fallback to binomial (condition: density_variance > 0)."""
        result = density_p_value(9, 10, 0.5, -0.05)
        expected = float(binom.sf(8, 10, 0.5))
        assert result == pytest.approx(expected, rel=1e-10)

    def test_very_small_variance_betabinom(self) -> None:
        """Test: Very small positive variance uses beta-binomial."""
        result = density_p_value(9, 10, 0.5, 0.001)
        # This should use betabinom because 0.001 > 0 and 0.001 < 0.25
        # m = 0.5 * 0.5 / 0.001 - 1.0 = 250 - 1 = 249
        expected = float(betabinom.sf(8, 10, 0.5 * 249, 0.5 * 249))
        assert result == pytest.approx(expected, abs=1e-10)
