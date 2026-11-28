import math

import numpy as np
import pytest

from faction_analysis.features import participation_coefficient, within_faction_z


def test_within_faction_z_is_the_z_score_of_log_degree() -> None:
    degree = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    membership = np.array([0, 0, 0, 1, 1, 1])
    z = within_faction_z(degree, membership)

    def hand(values: list[float]) -> list[float]:
        xs = [math.log1p(value) for value in values]
        mean = sum(xs) / len(xs)
        std = math.sqrt(sum((x - mean) ** 2 for x in xs) / len(xs))
        return [(x - mean) / std for x in xs]

    assert z.tolist() == pytest.approx(hand([1.0, 2.0, 3.0]) + hand([10.0, 20.0, 30.0]))
    assert z[:3].sum() == pytest.approx(0.0, abs=1e-12)
    assert (z[:3] ** 2).sum() == pytest.approx(3.0)


def test_within_faction_z_null_cases() -> None:
    z = within_faction_z(np.array([5.0, 5.0, 5.0, 7.0, 7.0, 9.0]), np.array([0, 0, 0, 1, 1, -1]))
    assert np.isnan(z[:3]).all()  # zero variance
    assert np.isnan(z[3:5]).all()  # two accounts only
    assert np.isnan(z[5])  # unplaced
    assert not np.isnan(within_faction_z(np.array([1.0, 2.0, 4.0]), np.array([0, 0, 0]))).any()


def star(weights: list[float], factions: list[int]):
    """Vertex 0 in the middle, one leaf per weight. Returns (u, v, weight, degree, membership)."""
    leaves = np.arange(1, len(weights) + 1)
    u = np.zeros(len(weights), dtype=np.int64)
    v = leaves
    weight = np.asarray(weights, dtype=np.float64)
    degree = np.concatenate([[weight.sum()], weight])
    membership = np.asarray([0] + factions, dtype=np.int64)
    return u, v, weight, degree, membership


def test_participation_is_zero_when_every_neighbour_is_in_the_own_faction() -> None:
    u, v, weight, degree, membership = star([1.0, 1.0, 1.0, 1.0], [0, 0, 0, 0])
    p = participation_coefficient(membership, u, v, weight, degree)
    assert p[0] == pytest.approx(0.0)
    assert p[1:].tolist() == pytest.approx([0.0] * 4)


def test_participation_is_two_thirds_for_three_equally_weighted_factions() -> None:
    u, v, weight, degree, membership = star([1.0, 1.0, 1.0], [0, 1, 2])
    p = participation_coefficient(membership, u, v, weight, degree)
    assert p[0] == pytest.approx(1 - 3 * (1 / 3) ** 2)


def test_participation_uses_edge_weights() -> None:
    u, v, weight, degree, membership = star([3.0, 1.0], [0, 1])
    p = participation_coefficient(membership, u, v, weight, degree)
    assert p[0] == pytest.approx(1 - (3 / 4) ** 2 - (1 / 4) ** 2)


def test_participation_null_cases() -> None:
    u = np.array([0, 1], dtype=np.int64)
    v = np.array([1, 2], dtype=np.int64)
    weight = np.ones(2)
    degree = np.array([1.0, 2.0, 1.0])
    # Vertex 1 has one edge to a placed account and one to an unplaced account. Its degree covers both, so
    # a quarter of its weight lies outside any faction.
    membership = np.array([0, 0, -1])
    p = participation_coefficient(membership, u, v, weight, degree)
    assert p[0] == pytest.approx(0.0)
    assert p[1] == pytest.approx(0.75)
    assert np.isnan(p[2])
    # Every edge of vertex 1 leads to an unplaced account.
    p = participation_coefficient(
        np.array([0, 0, -1]), np.array([1]), np.array([2]), np.ones(1), np.array([0.0, 1.0, 1.0])
    )
    assert np.isnan(p[1])
    # A vertex with no edge at all.
    p = participation_coefficient(
        np.array([0, 0]), np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([]), np.array([0.0, 1.0])
    )
    assert np.isnan(p).all()


def test_participation_is_bounded() -> None:
    rng = np.random.default_rng(1)
    u = rng.integers(0, 50, size=400)
    v = rng.integers(0, 50, size=400)
    weight = rng.random(400) + 0.1
    degree = np.bincount(u, weights=weight, minlength=50) + np.bincount(v, weights=weight, minlength=50)
    membership = rng.integers(0, 4, size=50)
    p = participation_coefficient(membership, u, v, weight, degree)
    finite = p[np.isfinite(p)]
    assert ((finite >= 0) & (finite <= 1)).all()
