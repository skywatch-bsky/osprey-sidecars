# pattern: Functional Core
"""Benjamini-Hochberg false-discovery-rate adjustment. Pure Python, no dependencies."""


def bh_adjust(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg q-values: step-up with cumulative-min monotonicity.

    Input order preserved. Empty list -> empty list. q-values capped at 1.0.
    """
    n = len(p_values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: p_values[i])
    q_values = [0.0] * n
    running_min = 1.0
    for position in range(n - 1, -1, -1):
        idx = order[position]
        rank = position + 1
        running_min = min(running_min, min(1.0, p_values[idx] * n / rank))
        q_values[idx] = running_min
    return q_values
