# pattern: Functional Core
"""One-sided sharer-density test: beta-binomial with binomial fallback."""

from scipy.stats import betabinom, binom


def density_p_value(
    unique_sharers: int,
    total_shares: int,
    expected_density: float,
    density_variance: float | None,
) -> float:
    """P(U >= unique_sharers) — one-sided, high-density direction.

    Beta-binomial with alpha/beta by method of moments from
    (expected_density, density_variance) when the fit is valid;
    plain binomial(total_shares, expected_density) otherwise.
    expected_density <= 0 or total_shares == 0 -> 1.0.
    """
    if expected_density <= 0 or total_shares == 0:
        return 1.0
    if unique_sharers / total_shares <= expected_density:
        return 1.0
    mu = min(expected_density, 1.0)
    if density_variance is not None and density_variance > 0 and mu < 1.0:
        m = mu * (1.0 - mu) / density_variance - 1.0
        if m > 0:
            return float(betabinom.sf(unique_sharers - 1, total_shares, mu * m, (1.0 - mu) * m))
    return float(binom.sf(unique_sharers - 1, total_shares, mu))
