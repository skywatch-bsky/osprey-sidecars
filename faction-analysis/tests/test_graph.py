import math
import statistics

import numpy as np
import pytest

from faction_analysis.config import GraphConfig
from faction_analysis.graph import (
    WindowEdges,
    aggregate_window_edges,
    build_graph,
    drop_deleted_creates,
    fold_deletes,
    index_vertices,
)

A, B, C, D, E = 10, 20, 30, 40, 50
EMPTY = np.empty(0, dtype=np.uint64)


def hashes(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.uint64)


def test_hand_computed_weights() -> None:
    """Both passes, by hand.

    A and B follow each other and A reposts B three times. C reposts D twice without a follow.
    """
    config = GraphConfig(window_days=60, mutual_follow_weight=2.0, repost_alpha=0.5, damp_large_accounts=True)
    edges = aggregate_window_edges(hashes(A, B), hashes(B, A), hashes(A, A, A, C, C), hashes(B, B, B, D, D))

    # Pass one: the raw directed weight follow + alpha * log(1 + reposts).
    raw_ab = 2.0 + 0.5 * math.log1p(3)
    raw_ba = 2.0 + 0.5 * math.log1p(0)
    raw_cd = 0.0 + 0.5 * math.log1p(2)

    # The median in-strength over accounts with an incoming edge, and the damping of each target.
    median_in = statistics.median([raw_ba, raw_ab, raw_cd])
    damp_a = min(1.0, math.log1p(median_in) / math.log1p(raw_ba))
    damp_b = min(1.0, math.log1p(median_in) / math.log1p(raw_ab))
    damp_d = min(1.0, math.log1p(median_in) / math.log1p(raw_cd))

    # Pass two: raw weight times the damping of the target.
    w_ab, w_ba, w_cd = raw_ab * damp_b, raw_ba * damp_a, raw_cd * damp_d

    # Undirected weight is the sum of both directions, rescaled so the mean edge weight is 1.
    edge_ab, edge_cd = w_ab + w_ba, w_cd
    rescale = (edge_ab + edge_cd) / 2

    graph = build_graph(edges, config)

    assert graph.account_hash.tolist() == [A, B, C, D]
    assert graph.u.tolist() == [0, 2]
    assert graph.v.tolist() == [1, 3]
    assert graph.weight.tolist() == pytest.approx([edge_ab / rescale, edge_cd / rescale])
    assert graph.weight.mean() == pytest.approx(1.0)
    assert graph.median_in_strength == pytest.approx(median_in)
    assert graph.rescale == pytest.approx(rescale)
    assert graph.in_strength.tolist() == pytest.approx([raw_ba, raw_ab, 0.0, raw_cd])
    assert graph.mutual_follow.tolist() == [True, False]
    # Each vertex sees the whole undirected edge it belongs to.
    assert graph.degree().tolist() == pytest.approx(
        [edge_ab / rescale, edge_ab / rescale, edge_cd / rescale, edge_cd / rescale]
    )


def test_weights_without_damping() -> None:
    config = GraphConfig(window_days=60, mutual_follow_weight=2.0, repost_alpha=0.5, damp_large_accounts=False)
    edges = aggregate_window_edges(hashes(A, B), hashes(B, A), hashes(A, A, A, C, C), hashes(B, B, B, D, D))
    raw_ab = 2.0 + 0.5 * math.log1p(3)
    raw_ba = 2.0
    raw_cd = 0.5 * math.log1p(2)
    rescale = (raw_ab + raw_ba + raw_cd) / 2
    graph = build_graph(edges, config)
    assert graph.weight.tolist() == pytest.approx([(raw_ab + raw_ba) / rescale, raw_cd / rescale])


def test_damping_keeps_the_median_account_at_one_and_never_exceeds_one() -> None:
    counts = [1, 5, 20, 50, 100]
    targets = [100, 200, 300, 400, 500]
    # Five targets are followed by 1, 5, 20, 50 and 100 distinct accounts.
    src: list[int] = []
    dst: list[int] = []
    next_source = 1000
    for target, count in zip(targets, counts, strict=True):
        for _ in range(count):
            src.append(next_source)
            dst.append(target)
            next_source += 1
    graph = build_graph(
        aggregate_window_edges(hashes(*src), hashes(*dst), EMPTY, EMPTY), GraphConfig(60, 1.0, 1.0, True)
    )

    target_index = {int(account): i for i, account in enumerate(graph.account_hash)}
    in_strength = {t: float(graph.in_strength[target_index[t]]) for t in targets}
    assert [round(in_strength[t]) for t in targets] == counts
    median = in_strength[300]
    assert graph.median_in_strength == pytest.approx(median)
    # Targets hash below their followers, so each target is the ``u`` end of its edges.
    assert set(graph.u.tolist()) == {target_index[t] for t in targets}
    weight = dict(zip(graph.u.tolist(), graph.weight.tolist(), strict=True))
    # Damping is log1p(median) / log1p(in_strength), capped at 1. Rescaling cancels in every ratio.
    assert weight[target_index[100]] == pytest.approx(weight[target_index[300]])
    assert weight[target_index[200]] == pytest.approx(weight[target_index[300]])
    assert weight[target_index[400]] == pytest.approx(
        weight[target_index[300]] * math.log1p(median) / math.log1p(in_strength[400])
    )
    assert weight[target_index[500]] == pytest.approx(
        weight[target_index[300]] * math.log1p(median) / math.log1p(in_strength[500])
    )
    assert weight[target_index[500]] < weight[target_index[400]] < weight[target_index[300]]


def test_mutual_follow_needs_a_follow_in_each_direction() -> None:
    edges = aggregate_window_edges(hashes(A, B, C), hashes(B, A, D), hashes(A, C), hashes(B, D))
    graph = build_graph(edges, GraphConfig(60, 2.0, 0.0, False))
    mutual = {
        (min(int(u), int(v)), max(int(u), int(v))): bool(flag)
        for u, v, flag in zip(graph.u, graph.v, graph.mutual_follow, strict=True)
    }
    assert mutual == {(0, 1): True, (2, 3): False}


def test_self_pairs_are_dropped() -> None:
    edges = aggregate_window_edges(hashes(A, A, B, C), hashes(B, A, A, A), EMPTY, EMPTY)
    graph = build_graph(edges, GraphConfig(60, 1.0, 0.0, False))
    assert len(edges.src_index) == 4
    assert graph.edges == 2
    assert not np.any(graph.u == graph.v)
    # A and B follow each other and C follows A, so the edges are 2 and 1 around a mean of 1.5.
    assert graph.weight.tolist() == pytest.approx([2.0 / 1.5, 1.0 / 1.5])


def test_unsorted_pairs_are_refused() -> None:
    edges = WindowEdges(hashes(A, B), np.array([1, 0]), np.array([0, 1]), hashes(1, 1).astype(np.uint8), hashes(1, 1))
    with pytest.raises(ValueError, match='sorted'):
        build_graph(edges, GraphConfig(60, 1.0, 0.0, False))


def test_empty_inputs_give_an_empty_graph() -> None:
    edges = aggregate_window_edges(EMPTY, EMPTY, EMPTY, EMPTY)
    graph = build_graph(edges, GraphConfig(60, 1.0, 0.0, False))
    assert graph.vertices == 0
    assert graph.edges == 0
    assert graph.degree().size == 0


def test_aggregate_window_edges_counts_reposts_and_folds_follows() -> None:
    edges = aggregate_window_edges(hashes(A, A, B, A), hashes(B, B, A, C), hashes(A, A, A, D), hashes(B, B, B, D))
    assert edges.account_hash.tolist() == [A, B, C, D]
    assert [(int(s), int(d)) for s, d in zip(edges.src_index, edges.dst_index, strict=True)] == [
        (0, 1),
        (0, 2),
        (1, 0),
        (3, 3),
    ]
    assert edges.follows.tolist() == [1, 1, 1, 0]
    assert edges.reposts.tolist() == [3, 0, 0, 1]


def test_fold_deletes_keeps_the_latest_delete_per_uri() -> None:
    deleted = fold_deletes(hashes(9, 4, 9), np.array([100, 5, 300], dtype=np.int64))
    assert deleted.uri_hash.tolist() == [4, 9]
    assert deleted.deleted_ms.tolist() == [5, 300]


def test_drop_deleted_creates_removes_only_later_deletes() -> None:
    deleted = fold_deletes(hashes(7, 8), np.array([500, 900], dtype=np.int64))
    src, dst, removed = drop_deleted_creates(
        hashes(A, A, A, A),
        hashes(B, C, D, E),
        hashes(7, 7, 8, 9),
        np.array([400, 600, 900, 1000], dtype=np.int64),
        deleted,
    )
    # URI 7 at 400 was deleted at 500. URI 7 at 600 came after its delete. URI 8 was deleted at the
    # same millisecond it was created. URI 9 was never deleted.
    assert removed == 1
    assert src.tolist() == [A, A, A]
    assert dst.tolist() == [C, D, E]


def test_drop_deleted_creates_without_deletes_changes_nothing() -> None:
    deleted = fold_deletes(EMPTY, np.empty(0, dtype=np.int64))
    src, dst, removed = drop_deleted_creates(hashes(A), hashes(B), hashes(1), np.array([1]), deleted)
    assert (removed, src.tolist(), dst.tolist()) == (0, [A], [B])


def test_index_vertices_returns_sorted_hashes_and_indices() -> None:
    accounts, (left, right) = index_vertices(hashes(30, 10), hashes(20, 30))
    assert accounts.tolist() == [10, 20, 30]
    assert left.tolist() == [2, 0]
    assert right.tolist() == [1, 2]
