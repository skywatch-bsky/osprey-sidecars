import json
from dataclasses import replace
from datetime import timedelta

import pytest

from faction_analysis.promotion import (
    IntentRecord,
    IntentUnreadable,
    intent_document,
    parse_intent,
    recover,
    run_id_ms,
    startup_plan,
)
from faction_analysis.runstate import RunRow, RunState, RunStateName

# A run ID starts with epoch milliseconds. These are spread over two days in November 2025.
NOW_MS = 1_764_400_000_000
STALE_RUN = '1764100000000.00000001'
OTHER_STALE_RUN = '1764110000001.00000003'
FRESH_RUN = '1764399999999.00000002'
PROMOTED_RUN = '1764241600000.00000004'


def promoting(run: RunRow) -> RunRow:
    """The complete row of ``run`` as it is when the intent is recorded."""
    return replace(run, next_faction_id=7, status='promoting', recorded_at=run.started_at + timedelta(minutes=30))


def state(run_id: str, name: RunStateName) -> RunState:
    return RunState(
        run_id=run_id,
        promoted=name == 'promoted',
        failed=name == 'failed',
        promoting=name == 'unresolved',
        running=name == 'running',
    )


def test_intent_document_round_trips(run_row: RunRow) -> None:
    restored = parse_intent(intent_document(promoting(run_row)))
    assert restored.run == promoting(run_row)


def test_unreadable_intent_is_never_treated_as_absent(run_row: RunRow) -> None:
    row = promoting(run_row)
    document = intent_document(row)
    mismatched = json.loads(document)
    mismatched['run_id'] = '1764241600001.0123abcd'
    broken = [
        'not json',
        document.replace('"version": 1', '"version": 2'),
        '{"version": 1}',
        '{"version": 1, "run_id": "x", "row": []}',
        # The row must name the same run as the document.
        json.dumps(mismatched),
        # The row must be complete: a time column that is not a string.
        document.replace(row.recorded_at.isoformat(), '7'),
    ]
    for text in broken:
        with pytest.raises(IntentUnreadable):
            parse_intent(text)


def test_intent_holds_only_a_promoting_row(run_row: RunRow) -> None:
    with pytest.raises(IntentUnreadable, match='promoting'):
        parse_intent(intent_document(run_row))


def test_the_intent_document_is_json_with_the_run_and_its_row(run_row: RunRow) -> None:
    from faction_analysis.schema import RUNS_COLUMNS

    document = json.loads(intent_document(promoting(run_row)))
    assert set(document) == {'version', 'run_id', 'row'}
    assert document['version'] == 1
    assert document['run_id'] == run_row.run_id
    assert len(document['row']) == len(RUNS_COLUMNS)
    assert document['row'][RUNS_COLUMNS.index('next_faction_id')] == 7


def test_run_that_the_intent_names_is_completed_forward(run_row: RunRow) -> None:
    record = IntentRecord(promoting(run_row))
    # Whatever a re-read shows, the run is completed and nothing of it is removed.
    for name in ('running', 'unresolved', 'failed', 'promoted'):
        decision = recover(run_row, record, state(run_row.run_id, name))
        assert decision.resend == ('promoting', 'promoted')
        assert not decision.cleanup and not decision.mark_failed and not decision.stop


def test_an_intent_for_another_run_does_not_protect_this_one(run_row: RunRow) -> None:
    other = replace(promoting(run_row), run_id='1764241600001.0123abcd')
    decision = recover(run_row, IntentRecord(other), state(run_row.run_id, 'running'))
    assert decision.mark_failed and decision.cleanup


def test_failed_re_read_stops_without_touching_anything(run_row: RunRow) -> None:
    decision = recover(run_row, None, None)
    assert decision.stop
    assert decision.resend == () and not decision.cleanup and not decision.mark_failed


def test_a_promoted_or_failed_run_needs_nothing(run_row: RunRow) -> None:
    for name in ('promoted', 'failed'):
        decision = recover(run_row, None, state(run_row.run_id, name))
        assert decision.resend == ()
        assert not decision.cleanup and not decision.mark_failed and not decision.stop


def test_an_unresolved_run_is_completed(run_row: RunRow) -> None:
    decision = recover(run_row, None, state(run_row.run_id, 'unresolved'))
    assert decision.resend == ('promoted',)
    assert not decision.cleanup and not decision.stop


def test_a_running_run_is_marked_failed_and_cleaned_up(run_row: RunRow) -> None:
    decision = recover(run_row, None, state(run_row.run_id, 'running'))
    assert decision.mark_failed and decision.cleanup and decision.resend == ()


def test_startup_completes_the_run_the_intent_names() -> None:
    plan = startup_plan(STALE_RUN, {STALE_RUN: state(STALE_RUN, 'unresolved')}, NOW_MS)
    assert plan.complete_run_id == STALE_RUN
    assert plan.stale_run_ids == ()


def test_startup_completes_an_unresolved_run_without_an_intent() -> None:
    plan = startup_plan(None, {STALE_RUN: state(STALE_RUN, 'unresolved')}, NOW_MS)
    assert plan.complete_run_id == STALE_RUN
    assert plan.stale_run_ids == ()


def test_startup_cleans_up_only_stale_runs_that_nobody_completes() -> None:
    states = {
        STALE_RUN: state(STALE_RUN, 'running'),
        OTHER_STALE_RUN: state(OTHER_STALE_RUN, 'failed'),
        FRESH_RUN: state(FRESH_RUN, 'running'),
        PROMOTED_RUN: state(PROMOTED_RUN, 'promoted'),
    }
    plan = startup_plan(None, states, NOW_MS)
    assert plan.complete_run_id is None
    assert plan.stale_run_ids == (STALE_RUN, OTHER_STALE_RUN)


def test_startup_leaves_a_fresh_failed_run_alone() -> None:
    plan = startup_plan(None, {FRESH_RUN: state(FRESH_RUN, 'failed')}, NOW_MS)
    assert plan.stale_run_ids == ()


def test_run_id_milliseconds() -> None:
    assert run_id_ms(STALE_RUN) == 1_764_100_000_000
    assert run_id_ms(STALE_RUN) < NOW_MS - 86_400_000
    assert run_id_ms(FRESH_RUN) > NOW_MS - 86_400_000
    with pytest.raises(ValueError):
        run_id_ms('run-1')
