# pattern: Functional Core
"""Affinity graph construction. Pure functions over NumPy arrays.

Accounts travel as 64-bit hashes. The two passes are the raw directed weight
``follow + repost_alpha * log(1 + reposts)`` and the damping of edges into large accounts. Leiden runs on the
undirected graph whose edge weight is the sum of both directions, rescaled so the mean edge weight is 1.
"""

from dataclasses import dataclass

import numpy as np

from faction_analysis.config import GraphConfig


@dataclass(frozen=True)
class DeletedRecords:
    """Latest delete time per record URI hash."""

    uri_hash: np.ndarray
    deleted_ms: np.ndarray


@dataclass(frozen=True)
class WindowEdges:
    """Directed account pairs aggregated over the window. ``src_index`` and ``dst_index`` index ``account_hash``."""

    account_hash: np.ndarray
    src_index: np.ndarray
    dst_index: np.ndarray
    follows: np.ndarray
    reposts: np.ndarray


@dataclass(frozen=True)
class AffinityGraph:
    """Undirected graph for Leiden. ``u`` is below ``v`` on every edge."""

    account_hash: np.ndarray
    u: np.ndarray
    v: np.ndarray
    weight: np.ndarray
    mutual_follow: np.ndarray
    in_strength: np.ndarray
    median_in_strength: float
    rescale: float

    @property
    def vertices(self) -> int:
        return int(len(self.account_hash))

    @property
    def edges(self) -> int:
        return int(len(self.u))

    def degree(self) -> np.ndarray:
        """Weighted degree per vertex."""
        n = self.vertices
        if self.edges == 0:
            return np.zeros(n, dtype=np.float64)
        return np.bincount(self.u, weights=self.weight, minlength=n) + np.bincount(
            self.v, weights=self.weight, minlength=n
        )


def _boundaries(sorted_keys: np.ndarray) -> np.ndarray:
    if len(sorted_keys) == 0:
        return np.empty(0, dtype=np.int64)
    change = np.empty(len(sorted_keys), dtype=bool)
    change[0] = True
    change[1:] = sorted_keys[1:] != sorted_keys[:-1]
    return np.flatnonzero(change).astype(np.int64)


def _as_hash_array(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.uint64)


def fold_deletes(uri_hash: np.ndarray, event_ms: np.ndarray) -> DeletedRecords:
    """Keep the latest delete per URI hash."""
    hashes = _as_hash_array(uri_hash)
    times = np.asarray(event_ms, dtype=np.int64)
    if len(hashes) == 0:
        return DeletedRecords(np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.int64))
    order = np.argsort(hashes, kind='stable')
    hashes, times = hashes[order], times[order]
    starts = _boundaries(hashes)
    return DeletedRecords(hashes[starts], np.maximum.reduceat(times, starts))


def superseded_mask(uri_hash: np.ndarray, event_ms: np.ndarray, deleted: DeletedRecords) -> np.ndarray:
    """True for each create that a later delete of the same record removed.

    A delete at or before the create leaves the create: the record was created again after the delete.
    """
    hashes, times = _as_hash_array(uri_hash), np.asarray(event_ms, dtype=np.int64)
    if len(deleted.uri_hash) == 0 or len(hashes) == 0:
        return np.zeros(len(hashes), dtype=bool)
    position = np.searchsorted(deleted.uri_hash, hashes)
    clipped = np.minimum(position, len(deleted.uri_hash) - 1)
    found = deleted.uri_hash[clipped] == hashes
    return found & (deleted.deleted_ms[clipped] > times)


def drop_deleted_creates(
    src: np.ndarray,
    dst: np.ndarray,
    uri_hash: np.ndarray,
    event_ms: np.ndarray,
    deleted: DeletedRecords,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Remove follow or block creates that a later delete of the same record removed."""
    src, dst = _as_hash_array(src), _as_hash_array(dst)
    superseded = superseded_mask(uri_hash, event_ms, deleted)
    return src[~superseded], dst[~superseded], int(superseded.sum())


def index_vertices(*hash_arrays: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Sorted distinct hashes and, per input array, the index of each element into that array.

    Each array is reduced on its own first, so the large sort runs on distinct accounts only.
    """
    if not hash_arrays:
        return np.empty(0, dtype=np.uint64), []
    converted = [_as_hash_array(a) for a in hash_arrays]
    accounts = np.empty(0, dtype=np.uint64)
    for array in converted:
        accounts = np.union1d(accounts, np.unique(array))
    return accounts, [np.searchsorted(accounts, array) for array in converted]


def aggregate_window_edges(
    follow_src: np.ndarray,
    follow_dst: np.ndarray,
    repost_src: np.ndarray,
    repost_dst: np.ndarray,
) -> WindowEdges:
    """Fold live follow creates and repost creates into distinct directed pairs.

    ``follows`` is 1 when the pair has at least one live follow create. ``reposts`` is the repost count.
    """
    accounts, (fs, fd, rs, rd) = index_vertices(follow_src, follow_dst, repost_src, repost_dst)
    n = len(accounts)
    empty = np.empty(0, dtype=np.int64)
    if n == 0:
        return WindowEdges(accounts, empty, empty.copy(), np.empty(0, dtype=np.uint8), empty.copy())
    follow_count = int(len(fs))
    total = follow_count + len(rs)
    follows = np.zeros(total, dtype=np.uint8)
    follows[:follow_count] = 1
    reposts = np.zeros(total, dtype=np.int32)
    reposts[follow_count:] = 1
    keys = np.concatenate([fs * n + fd, rs * n + rd])
    del fs, fd, rs, rd
    order = np.argsort(keys, kind='stable')
    keys = keys[order]
    follows, reposts = follows[order], reposts[order]
    del order
    starts = _boundaries(keys)
    distinct = keys[starts]
    del keys
    return WindowEdges(
        account_hash=accounts,
        src_index=(distinct // n).astype(np.int64),
        dst_index=(distinct % n).astype(np.int64),
        follows=np.maximum.reduceat(follows, starts).astype(np.uint8),
        reposts=np.add.reduceat(reposts, starts),
    )


def _empty_graph(account_hash: np.ndarray) -> AffinityGraph:
    n = int(len(account_hash))
    return AffinityGraph(
        account_hash=account_hash,
        u=np.empty(0, dtype=np.int64),
        v=np.empty(0, dtype=np.int64),
        weight=np.empty(0, dtype=np.float64),
        mutual_follow=np.empty(0, dtype=bool),
        in_strength=np.zeros(n, dtype=np.float64),
        median_in_strength=0.0,
        rescale=1.0,
    )


def build_graph(edges: WindowEdges, config: GraphConfig) -> AffinityGraph:
    """Apply both weighting passes and fold the directed pairs into undirected edges."""
    n = int(len(edges.account_hash))
    if n == 0 or len(edges.src_index) == 0:
        return _empty_graph(edges.account_hash)
    src, dst = edges.src_index.astype(np.int64), edges.dst_index.astype(np.int64)
    keep = src != dst
    src, dst, follows, reposts = src[keep], dst[keep], edges.follows[keep], edges.reposts[keep]
    if len(src) == 0:
        return _empty_graph(edges.account_hash)

    # The reverse direction is found by binary search, which needs sorted and distinct pairs.
    key = src * n + dst
    if len(key) > 1 and bool(np.any(key[1:] <= key[:-1])):
        raise ValueError('WindowEdges must be sorted by (src_index, dst_index) with distinct pairs')
    reverse_key = dst * n + src
    position = np.searchsorted(key, reverse_key)
    clipped = np.minimum(position, len(key) - 1)
    has_reverse = key[clipped] == reverse_key
    mutual_follow = (follows == 1) & has_reverse & (follows[clipped] == 1)
    follow_weight = np.where(follows == 0, 0.0, np.where(mutual_follow, config.mutual_follow_weight, 1.0))
    raw = follow_weight + config.repost_alpha * np.log1p(reposts.astype(np.float64))

    in_strength = np.bincount(dst, weights=raw, minlength=n)
    positive = in_strength[in_strength > 0]
    median_in = float(np.median(positive)) if len(positive) else 0.0
    if config.damp_large_accounts and median_in > 0:
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.log1p(median_in) / np.log1p(in_strength)
        damp = np.where(in_strength > 0, np.minimum(1.0, ratio), 1.0)
    else:
        damp = np.ones(n, dtype=np.float64)
    directed_weight = raw * damp[dst]

    low = np.minimum(src, dst)
    high = np.maximum(src, dst)
    undirected_key = low * n + high
    order = np.argsort(undirected_key, kind='stable')
    undirected_key = undirected_key[order]
    directed_weight = directed_weight[order]
    mutual_follow = mutual_follow[order].astype(np.uint8)
    starts = _boundaries(undirected_key)
    weight = np.add.reduceat(directed_weight, starts)
    rescale = float(weight.mean()) if len(weight) else 1.0
    if not np.isfinite(rescale) or rescale <= 0:
        rescale = 1.0
    return AffinityGraph(
        account_hash=edges.account_hash,
        u=undirected_key[starts] // n,
        v=undirected_key[starts] % n,
        weight=weight / rescale,
        mutual_follow=np.maximum.reduceat(mutual_follow, starts).astype(bool),
        in_strength=in_strength,
        median_in_strength=median_in,
        rescale=rescale,
    )
