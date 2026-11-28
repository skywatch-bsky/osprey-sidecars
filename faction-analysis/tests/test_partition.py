import numpy as np
import pytest

from faction_analysis.partition import UNPLACED, NoFactionError, absorb_small_communities, warm_start_membership


def absorb(membership: list[int], u: list[int], v: list[int], weight: list[float], min_size: int):
    return absorb_small_communities(
        np.asarray(membership, dtype=np.int32),
        np.asarray(u, dtype=np.int64),
        np.asarray(v, dtype=np.int64),
        np.asarray(weight, dtype=np.float64),
        min_size,
    )


def test_established_communities_are_left_alone() -> None:
    result = absorb([0, 0, 0, 1, 1, 1], [0, 3], [1, 4], [1.0, 1.0], min_size=3)
    assert result.membership.tolist() == [0, 0, 0, 1, 1, 1]
    assert result.established.tolist() == [True, True]
    assert (result.absorbed_communities, result.unplaced_accounts) == (0, 0)


def test_small_community_joins_the_established_neighbour_with_the_largest_weight() -> None:
    # Community 1 (vertices 3, 4) touches community 0 with weight 1 and community 2 (vertices 5, 6, 7)
    # with weight 50.
    membership = [0, 0, 0, 1, 1, 2, 2, 2]
    result = absorb(membership, [3, 4, 3, 0, 5], [0, 6, 4, 1, 6], [1.0, 50.0, 30.0, 1.0, 1.0], min_size=3)
    assert result.membership.tolist() == [0, 0, 0, 2, 2, 2, 2, 2]
    assert result.absorbed_communities == 1
    assert result.unplaced_accounts == 0


def test_tie_goes_to_the_smaller_community_index() -> None:
    membership = [0, 0, 0, 1, 1, 2, 2, 2]
    u, v, weight = [3, 4, 0, 5], [0, 6, 1, 6], [7.0, 7.0, 1.0, 1.0]
    result = absorb(membership, u, v, weight, min_size=3)
    assert result.membership[3:5].tolist() == [0, 0]
    # Relabel so that the equally heavy neighbour has the smaller index.
    swapped = [2, 2, 2, 1, 1, 0, 0, 0]
    result = absorb(swapped, u, v, weight, min_size=3)
    assert result.membership[3:5].tolist() == [0, 0]


def test_internal_edges_of_a_small_community_do_not_count() -> None:
    # Community 1 has one heavy internal edge and one light edge to the established community 0.
    membership = [0, 0, 0, 1, 1]
    result = absorb(membership, [0, 3, 3], [1, 4, 0], [1.0, 100.0, 0.5], min_size=3)
    assert result.membership.tolist() == [0, 0, 0, 0, 0]


def test_isolated_vertex_is_unplaced() -> None:
    result = absorb([0, 0, 0, 1], [0, 1], [1, 2], [1.0, 1.0], min_size=3)
    assert result.membership.tolist() == [0, 0, 0, UNPLACED]
    assert result.unplaced_accounts == 1
    assert result.absorbed_communities == 0


def test_small_component_without_large_neighbour_is_unplaced() -> None:
    result = absorb([0, 0, 0, 1, 1, 2], [0, 1, 3, 5], [1, 2, 4, 5], [1.0, 1.0, 1.0, 1.0], min_size=3)
    assert result.membership.tolist() == [0, 0, 0, UNPLACED, UNPLACED, UNPLACED]
    assert result.unplaced_accounts == 3


def test_every_community_is_decided_once() -> None:
    # Community 1 touches only community 2, which is itself small. Neither has an established
    # neighbour, so both stay unplaced. Nothing cascades through a small community.
    membership = [0, 0, 0, 1, 1, 2, 2]
    result = absorb(membership, [0, 3, 5], [1, 5, 6], [1.0, 100.0, 1.0], min_size=3)
    assert result.membership.tolist() == [0, 0, 0, UNPLACED, UNPLACED, UNPLACED, UNPLACED]
    assert result.unplaced_accounts == 4


def test_all_small_graph_is_refused() -> None:
    with pytest.raises(NoFactionError, match='minimum faction size'):
        absorb([0, 0, 1, 1], [0, 2], [1, 3], [1.0, 1.0], min_size=3)


def test_absorption_is_deterministic() -> None:
    # Community 1 ties between established communities 0 and 2. Vertex 8 is isolated.
    membership = [0, 0, 0, 1, 1, 2, 2, 2, 3]
    u, v, weight = [3, 4, 0, 5], [0, 6, 1, 6], [2.0, 2.0, 1.0, 1.0]
    first = absorb(membership, u, v, weight, min_size=3)
    second = absorb(membership, u, v, weight, min_size=3)
    assert first.membership.tolist() == second.membership.tolist()
    assert first.established.tolist() == second.established.tolist()
    assert (first.absorbed_communities, first.unplaced_accounts) == (
        second.absorbed_communities,
        second.unplaced_accounts,
    )
    assert first.membership.tolist() == [0, 0, 0, 0, 0, 2, 2, 2, UNPLACED]
    assert first.unplaced_accounts == 1


def test_empty_membership_is_absorbed_to_nothing() -> None:
    result = absorb([], [], [], [], min_size=3)
    assert result.membership.size == 0
    assert result.unplaced_accounts == 0


def test_warm_start_aligns_vertices() -> None:
    # Vertex order changed since the previous image: vertex 1 is new, vertex 2 keeps faction 1.
    initial = warm_start_membership(np.asarray([0, UNPLACED, 1, 0], dtype=np.int32))
    assert initial == [0, 2, 1, 0]


def test_warm_start_is_absent_without_a_previous_image() -> None:
    assert warm_start_membership(None) is None
    assert warm_start_membership(np.asarray([UNPLACED, UNPLACED])) is None
