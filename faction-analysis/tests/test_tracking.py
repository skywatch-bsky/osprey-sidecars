import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from faction_analysis.tracking import (
    LineageEvent,
    adjusted_rand_index,
    diff_moves,
    seed_confidence,
    stable_hash,
    track,
)


def track_previous(previous: list[int], current: list[int], *, next_id: int = 10, split_share: float = 0.1, known=None):
    return track(
        np.asarray(previous, dtype=np.int64),
        np.asarray(current, dtype=np.int64),
        np.ones(len(current)),
        next_faction_id=next_id,
        split_share=split_share,
        previous_faction_ids=known,
    )


def test_continuation_keeps_ids() -> None:
    tracked = track_previous([5, 5, 5, 9, 9, 9], [0, 0, 0, 1, 1, 1])
    assert tracked.faction.tolist() == [5, 5, 5, 9, 9, 9]
    assert tracked.next_faction_id == 10
    assert tracked.lineage == ()
    assert tracked.adjusted_rand == pytest.approx(1.0)


def test_community_labels_do_not_matter_only_overlap_does() -> None:
    # The labels of the new communities are permuted. Each keeps the ID of the accounts it holds.
    tracked = track_previous([5, 5, 5, 9, 9, 9], [7, 7, 7, 3, 3, 3])
    assert tracked.faction.tolist() == [5, 5, 5, 9, 9, 9]
    assert tracked.lineage == ()


def test_split_largest_part_keeps_id() -> None:
    tracked = track_previous([5] * 6, [0, 0, 0, 0, 1, 1])
    assert tracked.faction.tolist() == [5, 5, 5, 5, 10, 10]
    assert tracked.next_faction_id == 11
    assert tracked.lineage == (LineageEvent(10, 'split', 5),)


def test_part_below_the_split_share_is_a_birth_not_a_split() -> None:
    tracked = track_previous([5] * 10, [0] * 9 + [1], split_share=0.2)
    assert tracked.faction.tolist() == [5] * 9 + [10]
    assert tracked.lineage == (LineageEvent(10, 'birth', None),)


def test_split_at_exactly_the_share_counts() -> None:
    tracked = track_previous([5] * 10, [0] * 9 + [1], split_share=0.1)
    assert tracked.lineage == (LineageEvent(10, 'split', 5),)


def test_largest_part_keeps_the_id_by_weighted_degree() -> None:
    # Four accounts in community 1 carry degree 1 each; three in community 0 carry degree 2 each.
    tracked = track(
        np.asarray([5] * 7),
        np.asarray([0, 0, 0, 1, 1, 1, 1]),
        np.asarray([2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0]),
        next_faction_id=10,
        split_share=0.1,
    )
    assert tracked.faction.tolist() == [5, 5, 5, 10, 10, 10, 10]


def test_merge_largest_predecessor_wins() -> None:
    tracked = track_previous([5, 5, 5, 5, 9, 9], [0] * 6)
    assert tracked.faction.tolist() == [5] * 6
    assert tracked.next_faction_id == 10
    assert tracked.lineage == (LineageEvent(9, 'merge', 5),)


def test_merge_tie_goes_to_the_smaller_faction_id() -> None:
    tracked = track_previous([9, 9, 5, 5], [0] * 4)
    assert tracked.faction.tolist() == [5, 5, 5, 5]
    assert tracked.lineage == (LineageEvent(9, 'merge', 5),)


def test_pointing_tie_goes_to_the_smaller_community_index() -> None:
    # Faction 5 has equal degree in communities 0 and 1. It points at 0, so community 0 keeps ID 5 and
    # community 1 is a split with parent 5.
    tracked = track_previous([5, 5], [1, 0])
    assert tracked.faction.tolist() == [10, 5]
    assert tracked.lineage == (LineageEvent(10, 'split', 5),)


def test_a_new_community_with_no_predecessor_is_a_birth() -> None:
    tracked = track_previous([5, 5, 5, -1], [0, 0, 0, 1], known=[5])
    assert tracked.faction.tolist() == [5, 5, 5, 10]
    assert tracked.lineage == (LineageEvent(10, 'birth', None),)


def test_retired_faction_never_reused() -> None:
    first = track_previous([5, 5, 9, 9], [0, 0, 0, 0])
    assert first.lineage == (LineageEvent(9, 'merge', 5),)
    # Faction 9 no longer exists. Later IDs continue after the highest ID ever issued.
    second = track_previous([5, 5, 5, 5, -1], [0, 0, 0, 0, 1], next_id=first.next_faction_id, known=[5])
    assert second.faction.tolist() == [5, 5, 5, 5, 10]
    assert 9 not in second.faction.tolist()


def test_faction_whose_accounts_all_left_is_retired() -> None:
    tracked = track_previous([5, 5, -1], [0, 0, 1], known=[5, 9])
    assert tracked.faction.tolist() == [5, 5, 10]
    assert set(tracked.lineage) == {LineageEvent(10, 'birth', None), LineageEvent(9, 'death', None)}


def test_faction_whose_accounts_are_all_unplaced_now_is_retired() -> None:
    tracked = track_previous([5, 5, 9, 9], [0, 0, -1, -1], known=[5, 9])
    assert tracked.faction.tolist() == [5, 5, -1, -1]
    assert tracked.lineage == (LineageEvent(9, 'death', None),)


def test_unplaced_accounts_get_no_faction() -> None:
    tracked = track_previous([5, 5, 5, 5], [0, 0, -1, 0])
    assert tracked.faction.tolist() == [5, 5, -1, 5]
    assert tracked.next_faction_id == 10


def test_first_run_is_all_births() -> None:
    tracked = track(None, np.asarray([1, 1, 0, 0, -1]), np.ones(5), next_faction_id=1, split_share=0.1)
    assert tracked.faction.tolist() == [2, 2, 1, 1, -1]
    assert tracked.next_faction_id == 3
    assert set(tracked.lineage) == {LineageEvent(1, 'birth', None), LineageEvent(2, 'birth', None)}
    assert tracked.adjusted_rand is None


def test_adjusted_rand_index_matches_the_reference_implementation() -> None:
    rng = np.random.default_rng(7)
    for _ in range(20):
        first = rng.integers(0, 5, size=60)
        second = rng.integers(0, 4, size=60)
        assert adjusted_rand_index(first, second) == pytest.approx(adjusted_rand_score(first, second))
    assert adjusted_rand_index(np.array([1, 1, 1]), np.array([2, 2, 2])) == 1.0
    with pytest.raises(ValueError):
        adjusted_rand_index(np.array([1]), np.array([1, 2]))


def test_adjusted_rand_index_of_one_item_is_one() -> None:
    # One item makes no pair, so the index has no denominator. scikit-learn defines the result as 1.0.
    assert adjusted_rand_index(np.array([4]), np.array([9])) == 1.0
    assert adjusted_rand_score([4], [9]) == 1.0


def test_moves_log_holds_only_changes() -> None:
    diff = diff_moves(
        np.asarray(['a', 'b', 'c', 'd']),
        np.asarray([1, 1, 2, 2]),
        np.asarray(['r0', 'r0', 'r0', 'r0']),
        np.asarray(['a', 'b', 'd', 'e']),
        np.asarray([1, 2, 2, 2]),
        'r1',
    )
    moves = {
        did: (kind, int(src), int(dst))
        for did, kind, src, dst in zip(
            diff.moves.did.tolist(), diff.moves.kind.tolist(), diff.moves.from_faction_id, diff.moves.to_faction_id
        )
    }
    assert moves == {'b': (1, 1, 2), 'c': (2, 2, -1), 'e': (0, -1, 2)}


def test_tenure_carries_forward() -> None:
    diff = diff_moves(
        np.asarray(['a', 'b', 'd']),
        np.asarray([1, 1, 2]),
        np.asarray(['r0', 'r0', 'r0']),
        np.asarray(['a', 'b', 'd', 'e']),
        np.asarray([1, 2, 2, 2]),
        'r1',
    )
    assert diff.since_run_id.tolist() == ['r0', 'r1', 'r0', 'r1']
    assert diff.prev_faction_id.tolist() == [1, 1, 2, -1]


def test_no_changes_means_no_moves() -> None:
    diff = diff_moves(
        np.asarray(['a', 'b']),
        np.asarray([1, 2]),
        np.asarray(['r0', 'r0']),
        np.asarray(['b', 'a']),
        np.asarray([2, 1]),
        'r1',
    )
    assert len(diff.moves) == 0
    assert diff.since_run_id.tolist() == ['r0', 'r0']


def test_first_image_assigns_everyone() -> None:
    diff = diff_moves(
        np.asarray([], dtype=object),
        np.asarray([], dtype=np.int64),
        np.asarray([], dtype=object),
        np.asarray(['a']),
        np.asarray([3]),
        'r1',
    )
    assert len(diff.moves) == 1
    assert diff.since_run_id.tolist() == ['r1']


def test_stable_hash_matches_the_fnv1a_reference_vectors() -> None:
    # http://www.isthe.com/chongo/tech/comp/fnv/
    assert stable_hash(['']).tolist() == [0xCBF29CE484222325]
    assert stable_hash(['a']).tolist() == [0xAF63DC4C8601EC8C]
    assert stable_hash(['foobar']).tolist() == [0x85944171F73967E8]
    hashes = stable_hash(['did:plc:a', 'did:plc:b', 'did:plc:a', ''])
    assert hashes[0] == hashes[2]
    assert len({int(hashes[0]), int(hashes[1]), int(hashes[3])}) == 3


def test_seed_confidence_is_the_share_of_agreeing_runs() -> None:
    main = np.asarray([0, 0, 0, 1, 1, -1])
    same = np.asarray([4, 4, 4, 2, 2, -1])
    one_moved = np.asarray([4, 4, 2, 2, 2, -1])
    confidence = seed_confidence(main, [same, one_moved], np.ones(6), split_share=0.1)
    assert confidence[:5].tolist() == pytest.approx([1.0, 1.0, 0.5, 1.0, 1.0])
    assert np.isnan(confidence[5])


def test_seed_confidence_is_nan_without_extra_runs() -> None:
    confidence = seed_confidence(np.asarray([0, 1]), [], np.ones(2), split_share=0.1)
    assert np.isnan(confidence).all()
