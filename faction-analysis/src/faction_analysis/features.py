# pattern: Functional Core
"""Per-account features: within-faction degree z-score and the participation coefficient.

Both depend on the partition, so they are comparable within one run and across runs only with the lineage in
hand. An undefined value is ``NaN`` and is stored as NULL.
"""

import numpy as np

MIN_ACCOUNTS_FOR_Z = 3


def within_faction_z(
    degree: np.ndarray, membership: np.ndarray, *, min_accounts: int = MIN_ACCOUNTS_FOR_Z
) -> np.ndarray:
    """z-score of ``log(1 + degree)`` inside the faction.

    ``NaN`` for an unplaced account, for a faction with fewer than ``min_accounts`` accounts and for a faction
    with zero variance.
    """
    weights = np.asarray(degree, dtype=np.float64)
    labels = np.asarray(membership, dtype=np.int64)
    placed = labels >= 0
    result = np.full(len(labels), np.nan, dtype=np.float64)
    if not placed.any():
        return result
    faction = labels[placed]
    communities = int(faction.max()) + 1
    count = np.bincount(faction, minlength=communities)
    x = np.log1p(weights[placed])
    total = np.bincount(faction, weights=x, minlength=communities)
    mean = total / np.maximum(count, 1)
    deviation = x - mean[faction]
    variance = np.bincount(faction, weights=deviation**2, minlength=communities) / np.maximum(count, 1)
    # A faction has no variance when its smallest and its largest value are equal. A computed variance can
    # differ from zero by rounding alone, and would then turn every z-score into exactly 1 or -1.
    order = np.argsort(faction, kind='stable')
    starts = _boundaries(faction[order])
    sorted_x = x[order]
    flat = np.zeros(communities, dtype=bool)
    flat[faction[order][starts]] = np.maximum.reduceat(sorted_x, starts) == np.minimum.reduceat(sorted_x, starts)
    defined = (count >= min_accounts) & ~flat
    with np.errstate(divide='ignore', invalid='ignore'):
        result[placed] = deviation / np.sqrt(variance[faction])
    undefined = np.ones(len(labels), dtype=bool)
    undefined[placed] = ~defined[faction]
    result[undefined] = np.nan
    return result


def participation_coefficient(
    membership: np.ndarray, u: np.ndarray, v: np.ndarray, weight: np.ndarray, degree: np.ndarray
) -> np.ndarray:
    """Guimerà-Amaral participation coefficient: ``1 - sum_s (k_is / k_i)^2``.

    ``s`` runs over the factions of the account's neighbours. The value is 0 for an account whose neighbours
    are all in its own faction and approaches 1 for an account whose weight spreads evenly over many
    factions. ``NaN`` for an unplaced account and for an account without an edge.
    """
    labels = np.asarray(membership, dtype=np.int64)
    n = int(len(labels))
    result = np.full(n, np.nan, dtype=np.float64)
    placed = labels >= 0
    valid = placed[u] & placed[v]
    if not valid.any():
        return result
    factions = int(labels[placed].max()) + 1
    side = np.concatenate([u[valid], v[valid]]).astype(np.int64)
    neighbour_faction = np.concatenate([labels[v[valid]], labels[u[valid]]])
    weights = np.concatenate([np.asarray(weight, dtype=np.float64)[valid]] * 2)

    # Sum the weight from each account into each neighbouring faction, then square the shares.
    key = side * factions + neighbour_faction
    order = np.argsort(key, kind='stable')
    key, weights = key[order], weights[order]
    del order
    starts = _boundaries(key)
    shared = np.add.reduceat(weights, starts)
    account = key[starts] // factions
    del key, weights
    strength = np.asarray(degree, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        share = shared / strength[account]
    squared = np.where(np.isfinite(share), share**2, 0.0)
    touched = np.bincount(account, minlength=n)
    result[placed] = 1.0 - np.bincount(account, weights=squared, minlength=n)[placed]
    # An account without an edge to a placed account has no participation to speak of.
    result[touched == 0] = np.nan
    result[~placed] = np.nan
    result[strength <= 0] = np.nan
    return np.clip(result, 0.0, 1.0)


def _boundaries(sorted_keys: np.ndarray) -> np.ndarray:
    change = np.empty(len(sorted_keys), dtype=bool)
    change[0] = True
    change[1:] = sorted_keys[1:] != sorted_keys[:-1]
    return np.flatnonzero(change).astype(np.int64)
