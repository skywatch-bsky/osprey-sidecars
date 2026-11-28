"""Runs small graphs through the real ``igraph`` and ``leidenalg``, so library behaviour is tested, not assumed."""

import igraph
import numpy as np
import pytest

from faction_analysis.config import Objective
from faction_analysis.partition import (
    UNPLACED,
    NoFactionError,
    PartitionResult,
    absorb_small_communities,
    build_igraph,
    run_partition,
    warm_start_membership,
    weighted_modularity,
)


def two_cliques(
    *, isolated: bool = False, bridge_weight: float = 0.05
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Two cliques of six vertices joined by one weak edge, plus an optional isolated vertex."""
    u: list[int] = []
    v: list[int] = []
    w: list[float] = []
    for offset in (0, 6):
        for a in range(offset, offset + 6):
            for b in range(a + 1, offset + 6):
                u.append(a)
                v.append(b)
                w.append(1.0)
    u.append(0)
    v.append(6)
    w.append(bridge_weight)
    vertices = 13 if isolated else 12
    return vertices, np.asarray(u), np.asarray(v), np.asarray(w)


def partition(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    vertices: int,
    *,
    objective: Objective,
    resolution: float,
    seed: int,
    n_iterations: int,
    initial_membership: list[int] | None = None,
) -> PartitionResult:
    return run_partition(
        build_igraph(vertices, u, v, w),
        objective=objective,
        resolution=resolution,
        seed=seed,
        n_iterations=n_iterations,
        initial_membership=initial_membership,
    )


@pytest.mark.parametrize('objective', ['cpm', 'modularity'])
def test_same_graph_same_seed_same_membership(objective: Objective) -> None:
    vertices, u, v, w = two_cliques()
    first = partition(u, v, w, vertices, objective=objective, resolution=0.1, seed=7, n_iterations=-1)
    second = partition(u, v, w, vertices, objective=objective, resolution=0.1, seed=7, n_iterations=-1)
    assert first.membership.tolist() == second.membership.tolist()
    assert first.quality == pytest.approx(second.quality)
    assert len(first.membership) == vertices


def test_graph_keeps_the_edge_order_and_weights() -> None:
    vertices, u, v, w = two_cliques()
    graph = build_igraph(vertices, u, v, w)
    assert graph.vcount() == vertices and not graph.is_directed()
    assert graph.ecount() == len(u)
    assert graph.get_edgelist() == list(zip(u.tolist(), v.tolist(), strict=True))
    assert graph.es['weight'] == w.tolist()


def test_run_partition_releases_the_weight_attribute_and_keeps_its_result() -> None:
    vertices, u, v, w = two_cliques(isolated=True)
    graph = build_igraph(vertices, u, v, w)
    result = run_partition(graph, objective='cpm', resolution=0.1, seed=1, n_iterations=-1)
    # The Python list of floats is the largest avoidable cost on the production graph.
    assert 'weight' not in graph.es.attributes()
    assert len(set(result.membership.tolist())) == 3
    assert weighted_modularity(result.membership, u, v, w) == pytest.approx(0.4983, abs=1e-3)


@pytest.mark.parametrize('objective', ['cpm', 'modularity'])
def test_cold_start_separates_the_two_clques(objective: Objective) -> None:
    vertices, u, v, w = two_cliques(isolated=True)
    result = partition(u, v, w, vertices, objective=objective, resolution=0.1, seed=1, n_iterations=-1)
    left, right = result.membership[0], result.membership[6]
    assert left != right
    assert set(result.membership[:6].tolist()) == {left}
    assert set(result.membership[6:12].tolist()) == {right}
    # Two equal cliques give Q = 1 - 1/2 with the weighted bridge. The isolated vertex changes nothing.
    assert weighted_modularity(result.membership, u, v, w) == pytest.approx(0.4983, abs=1e-3)


def test_warm_start_accepts_a_previous_membership_that_lacks_and_keeps_vertices() -> None:
    vertices, u, v, w = two_cliques(isolated=True)
    # Vertex 2 was not placed before and vertex 12 is new. The previous labels are arbitrary.
    previous = np.asarray([5, 5, UNPLACED, 5, 5, 5, 9, 9, 9, 9, 9, 9, UNPLACED], dtype=np.int32)
    initial = warm_start_membership(previous)
    assert initial is not None and len(initial) == vertices
    warm = partition(
        u, v, w, vertices, objective='modularity', resolution=0.5, seed=3, n_iterations=-1, initial_membership=initial
    )
    cold = partition(u, v, w, vertices, objective='modularity', resolution=0.5, seed=3, n_iterations=-1)
    assert warm.membership[0] == warm.membership[1] == warm.membership[3]
    assert warm.membership[0] != warm.membership[6]
    assert cold.membership[0] != cold.membership[6]
    assert len(warm.membership) == len(cold.membership) == vertices


def test_weighted_modularity_matches_igraph() -> None:
    rng = np.random.default_rng(3)
    for _ in range(10):
        u, v = np.unique(np.sort(rng.integers(0, 40, size=(200, 2)), axis=1), axis=0).T
        keep = u != v
        u, v = u[keep], v[keep]
        w = rng.random(len(u)) + 0.1
        graph = build_igraph(40, u, v, w)
        labels = rng.integers(0, 5, size=40)
        assert weighted_modularity(labels, u, v, w) == pytest.approx(graph.modularity(labels, weights='weight'))
    assert weighted_modularity(np.array([0, 0]), np.array([0]), np.array([1]), np.array([0.0])) == 0.0


def test_isolated_vertex_is_unplaced() -> None:
    vertices, u, v, w = two_cliques(isolated=True)
    result = partition(u, v, w, vertices, objective='cpm', resolution=0.1, seed=1, n_iterations=-1)
    absorbed = absorb_small_communities(result.membership, u, v, w, min_size=3)
    assert absorbed.membership[12] == UNPLACED
    assert len(set(absorbed.membership[:6].tolist())) == 1
    assert len(set(absorbed.membership[6:12].tolist())) == 1
    assert absorbed.membership[0] != absorbed.membership[6]
    assert absorbed.unplaced_accounts == 1


def test_small_component_without_large_neighbour_is_unplaced() -> None:
    # A two-vertex component apart from the cliques has no established neighbour.
    vertices, u, v, w = two_cliques()
    u = np.append(u, [12])
    v = np.append(v, [13])
    w = np.append(w, [1.0])
    result = partition(u, v, w, 14, objective='cpm', resolution=0.1, seed=1, n_iterations=-1)
    absorbed = absorb_small_communities(result.membership, u, v, w, min_size=3)
    assert absorbed.membership[12:].tolist() == [UNPLACED, UNPLACED]
    assert absorbed.unplaced_accounts == 2


def test_all_small_graph_is_refused() -> None:
    ring_u = np.arange(10)
    ring_v = (ring_u + 1) % 10
    result = partition(ring_u, ring_v, np.ones(10), 10, objective='modularity', resolution=1.0, seed=1, n_iterations=-1)
    with pytest.raises(NoFactionError):
        absorb_small_communities(result.membership, ring_u, ring_v, np.ones(10), min_size=11)


def test_absorption_is_deterministic() -> None:
    vertices, u, v, w = two_cliques(isolated=True)
    first = partition(u, v, w, vertices, objective='cpm', resolution=0.1, seed=11, n_iterations=-1)
    second = partition(u, v, w, vertices, objective='cpm', resolution=0.1, seed=11, n_iterations=-1)
    a = absorb_small_communities(first.membership, u, v, w, min_size=3)
    b = absorb_small_communities(second.membership, u, v, w, min_size=3)
    assert a.membership.tolist() == b.membership.tolist()
    assert a.established.tolist() == b.established.tolist()
    assert (a.absorbed_communities, a.unplaced_accounts) == (b.absorbed_communities, b.unplaced_accounts)


def test_igraph_is_the_real_library() -> None:
    vertices, u, v, w = two_cliques()
    assert isinstance(build_igraph(vertices, u, v, w), igraph.Graph)
