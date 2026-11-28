import math

import numpy as np
import pytest

from faction_analysis.conflict import (
    BLOCK,
    CELL_COLUMNS,
    QUOTE,
    REPLY,
    Events,
    cell_counts,
    cells_frame,
    hostile_counts,
    swarm_flags,
)

MINUTE = 60_000
K = 20.0
MIN_TEXT = 2
FACTIONS = (10, 20)


def make(rows: list[tuple[int, int, int, int, int, int, int, float]], accounts: int = 5) -> Events:
    """Rows are (event_ms, kind, author, target, from_faction, to_faction, thread_key, h)."""
    columns = np.asarray(rows, dtype=np.float64).T
    h = columns[7].copy()
    return Events(
        event_ms=columns[0].astype(np.int64),
        kind=columns[1].astype(np.int8),
        author=columns[2].astype(np.int64),
        target=columns[3].astype(np.int64),
        from_faction=columns[4].astype(np.int64),
        to_faction=columns[5].astype(np.int64),
        thread_key=columns[6].astype(np.uint64),
        h=h.astype(np.float32),
        accounts=accounts,
        factions=len(FACTIONS),
        faction_ids=np.asarray(FACTIONS, dtype=np.int64),
    )


# Accounts 0 and 1 are in faction 10; accounts 2, 3 and 4 are in faction 20.
A0, A1, B0, B1, B2 = 0, 1, 2, 3, 4
SCENARIO = [
    # Two replies from faction 10 on one thread in one bucket: a pile-on, both counted.
    (0, REPLY, A0, B0, 0, 1, 101, 0.2),
    (1 * MINUTE, REPLY, A1, B0, 0, 1, 101, 0.4),
    # Two more in a later bucket, also a pile-on.
    (60 * MINUTE, REPLY, A0, B1, 0, 1, 102, math.nan),
    (60 * MINUTE, REPLY, A1, B1, 0, 1, 102, 0.9),
    # One reply back: not a pile-on.
    (10, REPLY, B0, A0, 1, 0, 103, 0.1),
    # Two quotes of one post in one bucket: a swarm, both counted.
    (20, QUOTE, B1, A1, 1, 0, 201, 0.6),
    (30, QUOTE, B2, A1, 1, 0, 201, math.nan),
    # Two blocks.
    (40, BLOCK, A0, B1, 0, 1, 0, math.nan),
    (50, BLOCK, A1, B1, 0, 1, 0, math.nan),
]


def frames(rows=SCENARIO, *, has_model: bool = True, min_repliers: int = 2, min_quoters: int = 2):
    events = make(rows)
    pileon, swarm = swarm_flags(
        events,
        min_repliers=min_repliers,
        reply_window_minutes=60,
        min_quoters=min_quoters,
        quote_window_minutes=60,
    )
    counts = cell_counts(events, pileon, swarm)
    frame = cells_frame(counts, prior_strength=K, min_text_scored=MIN_TEXT, has_model=has_model)
    return frame.set_index(['from_faction_id', 'to_faction_id']), pileon, swarm


def test_hand_computed_cells() -> None:
    frame, pileon, swarm = frames()
    assert list(frame.reset_index().columns[:2]) == ['from_faction_id', 'to_faction_id']
    assert tuple(frame.reset_index().columns[2:]) == tuple(CELL_COLUMNS[2:])

    # Observed counts, by hand.
    forward = frame.loc[(10, 20)]
    backward = frame.loc[(20, 10)]
    assert (forward['contact'], forward['pileon_replies'], forward['swarm_quotes'], forward['blocks']) == (4, 4, 0, 2)
    assert (backward['contact'], backward['pileon_replies'], backward['swarm_quotes'], backward['blocks']) == (
        3,
        0,
        2,
        0,
    )
    assert (forward['text_scored'], forward['text_hostility']) == (3, pytest.approx(1.5))
    assert (backward['text_scored'], backward['text_hostility']) == (2, pytest.approx(0.7))
    assert (forward['distinct_sources'], forward['distinct_targets']) == (2, 2)
    assert (backward['distinct_sources'], backward['distinct_targets']) == (3, 2)

    # Contact: expected from the row and column totals, then the lift.
    total = 7.0
    expected_forward = 4 * 4 / total
    expected_backward = 3 * 3 / total
    assert forward['contact_expected'] == pytest.approx(expected_forward)
    assert forward['contact_lift'] == pytest.approx(4 / expected_forward)
    assert backward['contact_lift'] == pytest.approx(3 / expected_backward)

    # Each signal: the global rate g, the shrunk rate with k pseudo-events, divided by g.
    g_pileon, g_swarm, g_blocks = 4 / total, 2 / total, 2 / total
    assert forward['pileon_lift'] == pytest.approx(((4 + K * g_pileon) / (4 + K)) / g_pileon)
    assert backward['pileon_lift'] == pytest.approx(((0 + K * g_pileon) / (3 + K)) / g_pileon)
    assert backward['swarm_lift'] == pytest.approx(((2 + K * g_swarm) / (3 + K)) / g_swarm)
    assert forward['swarm_lift'] == pytest.approx(((0 + K * g_swarm) / (4 + K)) / g_swarm)
    assert forward['blocks_lift'] == pytest.approx(((2 + K * g_blocks) / (4 + K)) / g_blocks)

    # Text: the mean hostility over all scored texts is the baseline.
    g_text = 2.2 / 5
    assert forward['text_lift'] == pytest.approx(((1.5 + K * g_text) / (3 + K)) / g_text)
    assert backward['text_lift'] == pytest.approx(((0.7 + K * g_text) / (2 + K)) / g_text)

    # Direction: (forward - backward) / (forward + backward).
    assert forward['direction_pileon'] == pytest.approx(1.0) and backward['direction_pileon'] == pytest.approx(-1.0)
    assert forward['direction_blocks'] == pytest.approx(1.0) and backward['direction_blocks'] == pytest.approx(-1.0)
    assert forward['direction_swarm'] == pytest.approx(-1.0) and backward['direction_swarm'] == pytest.approx(1.0)
    assert forward['direction_text'] == pytest.approx((1.5 - 0.7) / 2.2)
    assert backward['direction_text'] == pytest.approx(-forward['direction_text'])

    # The score is the smaller of the contact lift and the largest signal lift.
    assert forward['conflict_score'] == pytest.approx(min(forward['contact_lift'], forward['blocks_lift']))
    assert backward['conflict_score'] == pytest.approx(min(backward['contact_lift'], backward['swarm_lift']))

    assert pileon.tolist() == [True, True, True, True, False, False, False, False, False]
    assert swarm.tolist() == [False, False, False, False, False, True, True, False, False]


def test_diagonal_and_asymmetry() -> None:
    rows = SCENARIO + [
        (5, REPLY, A0, A1, 0, 0, 301, 0.5),
        (6, REPLY, A1, A0, 0, 0, 302, math.nan),
        (7, BLOCK, A0, A1, 0, 0, 0, math.nan),
    ]
    frame, _, _ = frames(rows)
    assert set(frame.index) == {(10, 20), (20, 10), (10, 10)}
    inside = frame.loc[(10, 10)]
    # On the diagonal the two directions are the same events, so every direction is 0.
    assert inside['direction_blocks'] == pytest.approx(0.0)
    assert inside['direction_text'] == pytest.approx(0.0)
    assert inside['blocks'] == 1 and inside['contact'] == 2
    assert frame.loc[(10, 20)]['blocks'] != frame.loc[(20, 10)]['blocks']


def test_blocks_without_contact() -> None:
    block_only = [(0, BLOCK, A0, B0, 0, 1, 0, math.nan)]
    inside = [(10, REPLY, A0, A1, 0, 0, 9, 0.1), (11, REPLY, B0, B1, 1, 1, 8, 0.1)]
    # No contact reaches faction 20, so the cell has no expectation and no lift.
    frame, _, _ = frames(block_only + inside[:1])
    only_blocks = frame.loc[(10, 20)]
    assert only_blocks['contact'] == 0 and only_blocks['blocks'] == 1
    assert only_blocks['contact_lift'] is None and only_blocks['contact_expected'] == 0.0
    # The blocks lift is kept: it is blocks per unit of contact, pulled toward the global rate by the prior.
    assert only_blocks['blocks_lift'] == pytest.approx((1 + K * 1.0) / K)
    assert only_blocks['conflict_score'] is None
    # The direction of the blocks is still defined.
    assert only_blocks['direction_blocks'] == pytest.approx(1.0)

    # With contact inside each faction the cell expects half a contact and its contact lift is zero. A
    # zero lift must not read as "no conflict": the score stays NULL.
    frame, _, _ = frames(block_only + inside)
    only_blocks = frame.loc[(10, 20)]
    assert only_blocks['contact'] == 0 and only_blocks['contact_expected'] == pytest.approx(0.5)
    assert only_blocks['contact_lift'] == pytest.approx(0.0)
    assert only_blocks['conflict_score'] is None
    assert only_blocks['blocks_lift'] == pytest.approx((1 + K * 0.5) / K / 0.5)


def test_blocks_only_window() -> None:
    frame, _, _ = frames([(0, BLOCK, A0, B0, 0, 1, 0, math.nan)])
    assert list(frame.index) == [(10, 20)]
    for column in ('contact_lift', 'pileon_lift', 'swarm_lift', 'blocks_lift', 'text_lift', 'conflict_score'):
        assert frame.loc[(10, 20)][column] is None, column
    assert frame.loc[(10, 20)]['contact_expected'] == 0.0


def test_no_signals() -> None:
    frame, _, _ = frames([(0, REPLY, A0, B0, 0, 1, 1, math.nan), (10, REPLY, B0, A0, 1, 0, 2, math.nan)])
    for column in ('pileon_lift', 'swarm_lift', 'blocks_lift', 'text_lift', 'conflict_score'):
        assert frame.loc[(10, 20)][column] is None, column
    # Contact is still there, so its lift is defined.
    assert frame.loc[(10, 20)]['contact_lift'] == pytest.approx(2.0)
    assert frame.loc[(10, 20)]['direction_pileon'] is None


def test_zero_text_baseline() -> None:
    rows = [
        (ms, kind, a, b, f, t, thread, 0.0 if not math.isnan(h) else h) for ms, kind, a, b, f, t, thread, h in SCENARIO
    ]
    frame, _, _ = frames(rows)
    assert frame.loc[(10, 20)]['text_lift'] is None
    assert frame.loc[(10, 20)]['text_scored'] == 3


def test_text_lift_needs_a_model_and_enough_scored_texts() -> None:
    frame, _, _ = frames(has_model=False)
    assert frame.loc[(10, 20)]['text_lift'] is None
    assert frame.loc[(10, 20)]['text_scored'] == 3
    frame, _, _ = frames()
    assert frame.loc[(10, 20)]['text_lift'] is not None
    frame, _, _ = frames([(0, REPLY, A0, B0, 0, 1, 1, 0.5), (10, REPLY, A1, B0, 0, 1, 2, 0.5)])
    # Two scored texts reach the minimum of two.
    assert frame.loc[(10, 20)]['text_scored'] == 2
    assert frame.loc[(10, 20)]['text_lift'] is not None
    frame, _, _ = frames([(0, REPLY, A0, B0, 0, 1, 1, 0.5)])
    # One scored text is below the minimum.
    assert frame.loc[(10, 20)]['text_lift'] is None


def test_threshold_crossed_mid_bucket() -> None:
    # Three replies by two authors in one bucket are not a pile-on. A fourth, by a third author, makes all
    # four a pile-on: the threshold counts distinct authors and includes the earlier replies.
    base = [
        (0, REPLY, A0, B0, 0, 1, 1, math.nan),
        (1, REPLY, A0, B0, 0, 1, 1, math.nan),
        (2, REPLY, A1, B0, 0, 1, 1, math.nan),
    ]
    _, pileon, _ = frames(base, min_repliers=3)
    assert not pileon[:3].any()
    _, pileon, _ = frames(base + [(3, REPLY, B2, B0, 0, 1, 1, math.nan)], min_repliers=3)
    assert pileon[:4].tolist() == [True, True, True, True]


def test_bucket_boundaries_are_aligned_to_the_epoch() -> None:
    # The third author replies one millisecond into the next bucket, so the first three never reach three
    # authors in one bucket. In the next bucket it is alone.
    rows = [
        (0, REPLY, A0, B0, 0, 1, 1, math.nan),
        (1, REPLY, A1, B0, 0, 1, 1, math.nan),
        (60 * MINUTE, REPLY, B2, B0, 0, 1, 1, math.nan),
    ]
    events = make(rows)
    pileon, _ = swarm_flags(events, min_repliers=3, reply_window_minutes=60, min_quoters=3, quote_window_minutes=60)
    assert not pileon.any()
    rows[2] = (60 * MINUTE - 1, REPLY, B2, B0, 0, 1, 1, math.nan)
    pileon, _ = swarm_flags(make(rows), min_repliers=3, reply_window_minutes=60, min_quoters=3, quote_window_minutes=60)
    assert pileon.tolist() == [True, True, True]


def test_a_different_thread_or_target_starts_a_new_group() -> None:
    rows = [
        (0, REPLY, A0, B0, 0, 1, 1, math.nan),
        (1, REPLY, A1, B0, 0, 1, 2, math.nan),  # other thread
        (2, REPLY, A0, B1, 0, 1, 1, math.nan),  # other target
        (3, REPLY, B2, B0, 1, 1, 1, math.nan),  # other author faction
    ]
    pileon, _ = swarm_flags(make(rows), min_repliers=2, reply_window_minutes=60, min_quoters=2, quote_window_minutes=60)
    assert not pileon.any()


def test_hostile_counts_per_account() -> None:
    events = make(SCENARIO)
    pileon, swarm = swarm_flags(events, min_repliers=2, reply_window_minutes=60, min_quoters=2, quote_window_minutes=60)
    counts = hostile_counts(events, pileon, swarm)
    assert counts.pileon_sent.tolist() == [2, 2, 0, 0, 0]
    assert counts.pileon_received.tolist() == [0, 0, 2, 2, 0]
    assert counts.swarm_sent.tolist() == [0, 0, 0, 1, 1]
    assert counts.swarm_received.tolist() == [0, 2, 0, 0, 0]
    assert counts.blocks_sent.tolist() == [1, 1, 0, 0, 0]
    assert counts.blocks_received.tolist() == [0, 0, 0, 2, 0]
    assert counts.text_sent.tolist() == pytest.approx([0.2, 1.3, 0.1, 0.6, 0.0])
    assert counts.text_received.tolist() == pytest.approx([0.1, 0.6, 0.6, 0.9, 0.0])


def test_window_selects_events_by_time() -> None:
    events = make(SCENARIO)
    early = events.in_window(0, 60 * MINUTE)
    # The window is half-open: the two replies that start the second bucket are not in it.
    assert len(early) == 7
    assert set(early.kind.tolist()) == {REPLY, QUOTE, BLOCK}
