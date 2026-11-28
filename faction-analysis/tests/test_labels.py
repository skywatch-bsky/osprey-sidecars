import numpy as np
import pytest

from faction_analysis.config import LabelConfig
from faction_analysis.labels import (
    BLOCK,
    CALIBRATION,
    JEV,
    RANDOM,
    TEST,
    TRAIN,
    LabelInput,
    block_positive_mask,
    build_labels,
    choose_jev_sample,
    elapsed_horizon_mask,
    label_policy_hash,
    negative_mask,
    split_codes,
)

MINUTE = 60_000
DAY = 86_400_000
A, B, C = 11, 22, 33


def inputs(count: int, *, start_ms: int = 0) -> LabelInput:
    return LabelInput(
        uri_hash=np.arange(1, count + 1, dtype=np.uint64) * np.uint64(7),
        kind=np.zeros(count, dtype=np.int8),
        event_ms=np.full(count, start_ms, dtype=np.int64),
        author=np.full(count, A, dtype=np.uint64),
        target=np.full(count, B, dtype=np.uint64),
        block_ms=np.empty(0, dtype=np.int64),
        block_blocker=np.empty(0, dtype=np.uint64),
        block_blocked=np.empty(0, dtype=np.uint64),
    )


def with_blocks(base: LabelInput, ms: list[int], blocker: list[int], blocked: list[int]) -> LabelInput:
    return LabelInput(
        **{
            **base.__dict__,
            'block_ms': np.asarray(ms, dtype=np.int64),
            'block_blocker': np.asarray(blocker, dtype=np.uint64),
            'block_blocked': np.asarray(blocked, dtype=np.uint64),
        }
    )


def test_block_positive_needs_the_target_to_block_the_author_within_the_delay() -> None:
    base = inputs(1)
    positive = block_positive_mask(with_blocks(base, [59 * MINUTE], [B], [A]), positive_minutes=60)
    assert positive.tolist() == [True]
    late = block_positive_mask(with_blocks(base, [60 * MINUTE + 1], [B], [A]), positive_minutes=60)
    assert late.tolist() == [False]
    exact = block_positive_mask(with_blocks(base, [60 * MINUTE], [B], [A]), positive_minutes=60)
    assert exact.tolist() == [True]


def test_a_block_before_the_interaction_or_in_the_other_direction_does_not_count() -> None:
    base = inputs(1, start_ms=DAY)
    before = block_positive_mask(with_blocks(base, [DAY - 1], [B], [A]), positive_minutes=60)
    assert before.tolist() == [False]
    other_way = block_positive_mask(with_blocks(base, [DAY + 1], [A], [B]), positive_minutes=60)
    assert other_way.tolist() == [False]


def test_the_first_later_block_decides() -> None:
    # The first block after the interaction is too late, so the interaction is not a positive even though
    # a second block falls inside the delay.
    base = inputs(1)
    ms = [2 * 60 * MINUTE, 3 * 60 * MINUTE]
    assert block_positive_mask(with_blocks(base, ms, [B, B], [A, A]), positive_minutes=60).tolist() == [False]


def test_many_interactions_share_few_blocks() -> None:
    count = 1000
    base = inputs(count, start_ms=DAY)
    blocks = with_blocks(base, [DAY + 1000, DAY + 2000], [B, B], [A, A])
    positive = block_positive_mask(blocks, positive_minutes=60)
    assert positive.sum() == count


def test_negative_horizon_boundary() -> None:
    event = np.array([1_000_000], dtype=np.int64)
    assert elapsed_horizon_mask(event, now_ms=1_000_000 + 7 * DAY, horizon_days=7)[0]
    assert not elapsed_horizon_mask(event, now_ms=1_000_000 + 7 * DAY - 1, horizon_days=7)[0]


def test_a_block_inside_the_horizon_leaves_no_negative() -> None:
    """A block after the positive delay but inside the horizon makes an interaction neither label.

    Its target is not 0: the author was blocked for it, only later than the positive delay.
    """
    config = LabelConfig.from_mapping({})

    def is_negative(block_ms: int) -> bool:
        blocked = with_blocks(inputs(1), [block_ms], [B], [A])
        positives = block_positive_mask(blocked, positive_minutes=60)
        elapsed = elapsed_horizon_mask(blocked.event_ms, now_ms=7 * DAY, horizon_days=7)
        return bool(negative_mask(blocked, positives=positives, elapsed=elapsed, horizon_days=7)[0])

    assert is_negative(30 * DAY)
    assert not is_negative(59 * MINUTE)
    assert not is_negative(61 * MINUTE)
    assert not is_negative(7 * DAY)
    assert is_negative(7 * DAY + 1)

    blocked = with_blocks(inputs(1), [61 * MINUTE], [B], [A])
    positives = block_positive_mask(blocked, positive_minutes=60)
    elapsed = elapsed_horizon_mask(blocked.event_ms, now_ms=7 * DAY, horizon_days=7)
    negatives = negative_mask(blocked, positives=positives, elapsed=elapsed, horizon_days=7)
    rows = build_labels(
        blocked,
        np.full(1, np.nan, dtype=np.float32),
        positives=positives,
        negatives=negatives,
        config=config,
    )
    assert len(rows) == 0


def test_splits_follow_the_uri_hash_in_70_15_15_shares() -> None:
    rng = np.random.default_rng(5)
    hashes = rng.integers(1, 2**63, size=100_000).astype(np.uint64)
    codes = split_codes(hashes)
    share = np.bincount(codes, minlength=3) / len(codes)
    assert share.tolist() == pytest.approx([0.70, 0.15, 0.15], abs=0.01)
    assert (codes == split_codes(hashes)).all()


def test_stratum_quotas_are_deterministic() -> None:
    config = LabelConfig.from_mapping(
        {'FACTION_ANALYSIS_JEV_STRATUM_POSITIVES': '100', 'FACTION_ANALYSIS_JEV_STRATUM_RANDOM': '50'}
    )
    base = inputs(1000)
    rng = np.random.default_rng(9)
    positives = rng.random(1000) < 0.5
    elapsed = np.ones(1000, dtype=bool)
    labelled = np.zeros(1000, dtype=bool)
    first = choose_jev_sample(base, positives=positives, elapsed=elapsed, labelled=labelled, config=config)
    second = choose_jev_sample(base, positives=positives, elapsed=elapsed, labelled=labelled, config=config)
    assert first.tolist() == second.tolist()
    positives_sample = first & positives
    assert positives_sample.sum() == 100
    # The random stratum draws from everything with an elapsed horizon and may overlap the positives.
    assert first.sum() <= 150


def test_labelled_interactions_are_not_sent_again() -> None:
    config = LabelConfig.from_mapping({'FACTION_ANALYSIS_JEV_STRATUM_RANDOM': '10'})
    base = inputs(20)
    elapsed = np.ones(20, dtype=bool)
    positives = np.zeros(20, dtype=bool)
    labelled = np.zeros(20, dtype=bool)
    labelled[:15] = True
    sample = choose_jev_sample(base, positives=positives, elapsed=elapsed, labelled=labelled, config=config)
    assert not (sample & labelled).any()
    assert sample.sum() == 5


def test_labels_follow_the_precedence_and_the_quotas() -> None:
    config = LabelConfig.from_mapping(
        {
            'FACTION_ANALYSIS_MAX_WEAK_POSITIVES': '60',
            'FACTION_ANALYSIS_NEGATIVE_SAMPLE': '100',
            'FACTION_ANALYSIS_WEAK_POSITIVE_WEIGHT': '0.5',
            'FACTION_ANALYSIS_NEGATIVE_WEIGHT': '0.25',
        }
    )
    count = 1000
    base = inputs(count)
    rng = np.random.default_rng(3)
    positives = rng.random(count) < 0.3
    elapsed = rng.random(count) < 0.8
    jev_h = np.full(count, np.nan, dtype=np.float32)
    labelled = rng.random(count) < 0.1
    jev_h[labelled] = rng.random(labelled.sum())
    negatives = elapsed & ~positives
    rows = build_labels(base, jev_h, positives=positives, negatives=negatives, config=config)

    # A Jev-labelled row always takes the Jev rule, whatever the block says.
    was_labelled = labelled[rows.uri_hash // 7 - 1]
    assert (rows.label_source[was_labelled] == JEV).all()
    is_jev = rows.label_source == JEV
    assert np.array_equal(rows.target[is_jev], (np.nan_to_num(rows.label_h[is_jev]) >= 0.5).astype(np.uint8))
    assert (rows.weight[is_jev] == 1.0).all()
    assert (rows.split[is_jev] == split_codes(rows.uri_hash[is_jev])).all()

    weak = rows.label_source == BLOCK
    assert weak.sum() == min(60, (positives & ~labelled).sum())
    assert (rows.target[weak] == 1).all() and (rows.weight[weak] == 0.5).all()
    negative = rows.label_source == RANDOM
    assert negative.sum() == min(100, (negatives & ~labelled).sum())
    assert (rows.target[negative] == 0).all() and (rows.weight[negative] == 0.25).all()


def test_calibration_and_test_pairs_never_train() -> None:
    config = LabelConfig.from_mapping({})
    count = 2000
    base = inputs(count)
    rng = np.random.default_rng(4)
    positives = rng.random(count) < 0.2
    elapsed = np.ones(count, dtype=bool)
    jev_h = np.full(count, np.nan, dtype=np.float32)
    some = rng.random(count) < 0.5
    jev_h[some] = rng.random(int(some.sum()))
    negatives = elapsed & ~positives
    rows = build_labels(base, jev_h, positives=positives, negatives=negatives, config=config)
    held_out = (rows.split == CALIBRATION) | (rows.split == TEST)
    assert held_out.any()
    assert (rows.label_source[held_out] == JEV).all()
    training = rows.split == TRAIN
    assert set(rows.label_source[~training].tolist()) == {JEV}
    assert (rows.split[~held_out] == TRAIN).all()


def test_label_policy_hash_changes_with_the_policy() -> None:
    first = label_policy_hash(LabelConfig.from_mapping({}))
    assert first == label_policy_hash(LabelConfig.from_mapping({}))
    assert first != label_policy_hash(LabelConfig.from_mapping({'FACTION_ANALYSIS_BLOCK_POSITIVE_MINUTES': '30'}))
    assert first != label_policy_hash(LabelConfig.from_mapping({'FACTION_ANALYSIS_JEV_MODEL': 'jev-1.15.0'}))
