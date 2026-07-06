# pattern: Functional Core
"""Dispersion-aware count tests: negative binomial with Poisson fallback."""

from scipy.stats import nbinom, poisson


def count_p_value(observed: int, mean: float, variance: float | None) -> float:
    """P(X >= observed) under NB when variance > mean, else Poisson(mean).

    NB via method of moments: r = mean**2 / (variance - mean), p = mean / variance.
    mean <= 0 -> 1.0. Non-integer r supported (scipy nbinom).
    """
    if mean <= 0:
        return 1.0
    if variance is not None and variance > mean:
        r = mean * mean / (variance - mean)
        p = mean / variance
        return float(nbinom.sf(observed - 1, r, p))
    return float(poisson.sf(observed - 1, mean))
