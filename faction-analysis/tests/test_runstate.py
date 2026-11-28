from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from faction_analysis.runstate import (
    RunRow,
    new_run_id,
    resolve_run_state,
    run_row_from_columns,
    run_row_to_columns,
)


def test_two_attempts_in_the_same_millisecond_get_distinct_run_ids() -> None:
    first = new_run_id(1_764_240_000_000, '0123abcd')
    second = new_run_id(1_764_240_000_000, 'ffffffff')
    assert first != second
    assert first == '1764240000000.0123abcd'
    assert len(first.split('.')[0]) == 13


def test_run_ids_sort_by_start() -> None:
    ids = [new_run_id(1_764_240_000_009, 'aaaaaaaa'), new_run_id(1_764_240_000_000, 'ffffffff')]
    assert sorted(ids) == ids[::-1]


def test_run_id_rejects_bad_inputs() -> None:
    for now in (-1, 10**13, True, '123'):
        with pytest.raises(ValueError):
            new_run_id(now, '0123abcd')  # ty: ignore[invalid-argument-type]
    for suffix in ('1234abc', '0123ABCD', '0123abcg', ''):
        with pytest.raises(ValueError):
            new_run_id(1_764_240_000_000, suffix)


def test_state_resolution_prefers_promoted_then_failed_then_unresolved() -> None:
    assert resolve_run_state(True, True, True, True) == 'promoted'
    assert resolve_run_state(False, True, True, True) == 'failed'
    assert resolve_run_state(False, False, True, True) == 'unresolved'
    assert resolve_run_state(False, False, False, True) == 'running'


def test_row_roundtrips_through_the_column_order(run_row: RunRow) -> None:
    restored = run_row_from_columns(run_row_to_columns(run_row))
    assert restored == run_row


def test_row_changes_status_without_losing_the_rest(run_row: RunRow) -> None:
    at = datetime(2026, 11, 27, 13, tzinfo=UTC)
    promoted = run_row.with_status('promoted', at)
    failed = run_row.with_failure('partition', 'NoFactionError', at)
    assert promoted.status == 'promoted' and promoted.recorded_at == at
    assert promoted.started_at == run_row.started_at
    assert (failed.status, failed.failed_stage, failed.error_class) == ('failed', 'partition', 'NoFactionError')


def test_row_refuses_an_invalid_run_id(run_row: RunRow) -> None:
    with pytest.raises(ValueError, match='run_id'):
        replace(run_row, run_id='run-1')


def test_row_reader_names_the_bad_value() -> None:
    row = run_row_to_columns(make_row())
    row[1] = 'paused'
    with pytest.raises(ValueError, match='paused'):
        run_row_from_columns(row)
    with pytest.raises(ValueError):
        run_row_from_columns(row[:-1])


def make_row() -> RunRow:
    started = datetime(2026, 11, 1, tzinfo=UTC)
    return RunRow(
        run_id='1764241600000.00000001',
        status='promoted',
        kind='full',
        recorded_at=started,
        started_at=started,
        window_start=started - timedelta(days=60),
        window_end=started,
        window_days=60,
        objective='cpm',
        resolution=0.01,
        seed=42,
        next_faction_id=300,
        accounts_placed=4_000_000,
        accounts_unplaced=10,
        hash_collisions=0,
        factions=299,
        moves=0,
        births=299,
        merges=0,
        splits=0,
        deaths=0,
        agreement_prev=None,
        modularity=0.71,
        source_gap_days=1,
        text_model_version='',
        text_coverage=0.0,
        stage_seconds={'extract_graph': 12.5},
        failed_stage='',
        error_class='',
    )
