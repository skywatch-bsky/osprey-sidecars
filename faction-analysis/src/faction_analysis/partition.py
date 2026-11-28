# pattern: Functional Core
"""Leiden partition and min-size absorption.

Wraps ``igraph`` and ``leidenalg``. Nothing here reads or writes anything.
"""

from dataclasses import dataclass

import igraph
import leidenalg
import numpy as np

from faction_analysis.config import Objective

UNPLACED = -1
# Chunked so that only this many Python integers and tuples exist at once while the graph is built.
_EDGE_CHUNK = 2_000_000


class NoFactionError(ValueError):
    """No community reached the minimum faction size, so the run has nothing to publish."""


@dataclass(frozen=True)
class PartitionResult:
    """The membership of one Leiden run and the value of the objective it optimised."""

    membership: np.ndarray
    quality: float


@dataclass(frozen=True)
class Absorption:
    """Membership after small communities joined an established neighbour. ``UNPLACED`` marks an unplaced account."""

    membership: np.ndarray
    established: np.ndarray
    absorbed_communities: int
    unplaced_accounts: int


def _edge_tuples(u: np.ndarray, v: np.ndarray):
    for start in range(0, len(u), _EDGE_CHUNK):
        stop = min(start + _EDGE_CHUNK, len(u))
        yield from zip(u[start:stop].tolist(), v[start:stop].tolist(), strict=True)


def build_igraph(vertices: int, u: np.ndarray, v: np.ndarray, weight: np.ndarray) -> igraph.Graph:
    """Undirected weighted graph. The edge order of ``u`` and ``v`` is kept, which keeps Leiden deterministic."""
    graph = igraph.Graph(n=int(vertices), edges=_edge_tuples(u, v), directed=False)
    graph.es['weight'] = weight.tolist()
    return graph


def run_partition(
    graph: igraph.Graph,
    *,
    objective: Objective,
    resolution: float,
    seed: int,
    n_iterations: int,
    initial_membership: list[int] | None = None,
) -> PartitionResult:
    """One Leiden run. The same graph, parameters and seed always give the same membership.

    The graph loses its ``weight`` attribute. The partition copies the weights on creation, and the attribute
    is a Python list of floats that costs 32 bytes per edge for as long as it lives, which is several GiB on
    the production graph. Keep the weight array if you need it afterwards.
    """
    partition_type = leidenalg.CPMVertexPartition if objective == 'cpm' else leidenalg.RBConfigurationVertexPartition
    partition = partition_type(
        graph,
        initial_membership=initial_membership,
        weights='weight',
        resolution_parameter=resolution,
    )
    del graph.es['weight']
    optimiser = leidenalg.Optimiser()
    optimiser.set_rng_seed(seed)
    optimiser.optimise_partition(partition, n_iterations)
    return PartitionResult(
        membership=np.asarray(partition.membership, dtype=np.int32),
        quality=float(partition.quality()),
    )


def weighted_modularity(membership: np.ndarray, u: np.ndarray, v: np.ndarray, weight: np.ndarray) -> float:
    """Newman's weighted modularity of a membership on an undirected graph.

    ``sum_c e_c / m - sum_c (a_c / 2m)^2`` with ``e_c`` the internal weight of community ``c`` and ``a_c``
    the weight of its incident edges. The same value ``igraph`` gives for ``Graph.modularity`` with weights.
    """
    labels = np.asarray(membership, dtype=np.int64)
    weights = np.asarray(weight, dtype=np.float64)
    total = float(weights.sum())
    if total <= 0:
        return 0.0
    communities = int(labels.max()) + 1
    internal = labels[u] == labels[v]
    e_c = np.bincount(labels[u[internal]], weights=weights[internal], minlength=communities)
    a_c = np.bincount(labels[u], weights=weights, minlength=communities) + np.bincount(
        labels[v], weights=weights, minlength=communities
    )
    return float((e_c / total).sum() - ((a_c / (2.0 * total)) ** 2).sum())


def warm_start_membership(previous: np.ndarray | None) -> list[int] | None:
    """Initial membership for a warm start, or ``None`` for a cold start.

    ``previous`` holds the previous faction index per current vertex. A vertex without a previous faction
    (``UNPLACED``) starts in a fresh singleton community. A vertex of the previous image that is gone from the
    current graph is simply absent from ``previous``.
    """
    if previous is None:
        return None
    placed = np.asarray(previous) >= 0
    if not placed.any():
        return None
    initial = np.asarray(previous, dtype=np.int64).copy()
    fresh = int(initial.max()) + 1
    new_vertices = int((~placed).sum())
    initial[~placed] = np.arange(fresh, fresh + new_vertices, dtype=np.int64)
    return initial.tolist()


def _boundary_starts(sorted_keys: np.ndarray) -> np.ndarray:
    change = np.empty(len(sorted_keys), dtype=bool)
    change[0] = True
    change[1:] = sorted_keys[1:] != sorted_keys[:-1]
    return np.flatnonzero(change).astype(np.int64)


def _absorption_targets(
    labels: np.ndarray, u: np.ndarray, v: np.ndarray, weight: np.ndarray, established: np.ndarray
) -> np.ndarray:
    """Established neighbour each small community joins, or ``-1`` when it has none. Indexed by community."""
    community_count = int(len(established))
    targets = np.full(community_count, -1, dtype=np.int64)
    side_u, side_v = labels[u], labels[v]
    small_u = ~established[side_u]
    small_v = ~established[side_v]
    crossing = small_u != small_v
    if not crossing.any():
        return targets
    small_side = np.where(small_u, side_u, side_v)[crossing]
    large_side = np.where(small_u, side_v, side_u)[crossing]
    cross_weight = np.asarray(weight, dtype=np.float64)[crossing]

    keys = small_side * community_count + large_side
    order = np.argsort(keys, kind='stable')
    keys, cross_weight = keys[order], cross_weight[order]
    starts = _boundary_starts(keys)
    sums = np.add.reduceat(cross_weight, starts)
    small_group = keys[starts] // community_count
    large_group = keys[starts] % community_count

    # ``lexsort`` keeps the ascending order of ``large_group`` inside a group, so the first row of each small
    # community is the largest weight and, on a tie, the smaller community index.
    ranked = np.lexsort((-sums, small_group))
    small_group, large_group = small_group[ranked], large_group[ranked]
    first = np.ones(len(small_group), dtype=bool)
    first[1:] = small_group[1:] != small_group[:-1]
    targets[small_group[first]] = large_group[first]
    return targets


def absorb_small_communities(
    membership: np.ndarray, u: np.ndarray, v: np.ndarray, weight: np.ndarray, min_size: int
) -> Absorption:
    """Every community below ``min_size`` joins its established neighbour with the largest connecting weight.

    A community is established when it has at least ``min_size`` accounts. Every community is decided once
    against the original partition, so the step always terminates. A small community without an established
    neighbour leaves its accounts unplaced. Ties go to the smaller community index.
    """
    labels = np.asarray(membership, dtype=np.int64)
    if len(labels) == 0:
        return Absorption(labels.astype(np.int32), np.empty(0, dtype=bool), 0, 0)
    established = np.bincount(labels, minlength=int(labels.max()) + 1) >= min_size
    if not established.any():
        raise NoFactionError('no community reached the minimum faction size')
    targets = _absorption_targets(labels, u, v, weight, established)
    target_of_vertex = targets[labels]
    absorbed = np.where(target_of_vertex >= 0, target_of_vertex, labels)
    unplaced = (target_of_vertex < 0) & ~established[labels]
    absorbed[unplaced] = UNPLACED
    return Absorption(
        membership=absorbed.astype(np.int32),
        established=established,
        absorbed_communities=int((targets >= 0).sum()),
        unplaced_accounts=int(unplaced.sum()),
    )
