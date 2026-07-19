# pattern: Functional Core
from __future__ import annotations

from datetime import date, datetime
from typing import Sequence

import pytest

from quote_cosharing.analyzer import EvolutionEvent, PairRow, TimestampedCluster
from quote_cosharing.config import AnalysisConfig, AppConfig, ClickHouseConfig
from quote_cosharing.db import MembershipRow, MemberTimestamp
from quote_cosharing.main import run_cycle


class FakeDb:
    """In-memory stub for database layer, captures inserted results."""

    def __init__(self) -> None:
        self.pairs: list[PairRow] = []
        self.membership_rows: list[MembershipRow] = []
        self.timestamp_rows: list[MemberTimestamp] = []
        self.captured_clusters: list[tuple[date, TimestampedCluster, EvolutionEvent]] = []
        self.captured_membership: list[tuple[date, str, str]] = []
        self.deleted_run_dates: list[tuple[str, date]] = []

    def fetch_pairs(self, query: str) -> list[PairRow]:
        """Returns pre-configured pairs."""
        return self.pairs

    def fetch_historical_membership(self, query: str) -> list[MembershipRow]:
        """Returns pre-configured membership rows."""
        return self.membership_rows

    def fetch_member_timestamps(self, query: str) -> list[MemberTimestamp]:
        """Returns pre-configured timestamp rows."""
        return self.timestamp_rows

    def insert_clusters(self, table: str, clusters: Sequence[tuple[date, TimestampedCluster, EvolutionEvent]]) -> None:
        """Captures cluster results for later assertion."""
        self.captured_clusters.extend(clusters)

    def insert_membership(self, table: str, membership: Sequence[tuple[date, str, str]]) -> None:
        """Captures membership results for later assertion."""
        self.captured_membership.extend(membership)

    def delete_run_date(self, table: str, run_date: date) -> None:
        """Records deletion calls for later assertion."""
        self.deleted_run_dates.append((table, run_date))

    def close(self) -> None:
        """No-op."""
        pass


@pytest.fixture
def base_config() -> AnalysisConfig:
    return AnalysisConfig(
        interval_seconds=3600,
        resolution=0.05,
        min_edge_weight=2,
        min_cluster_size=3,
        min_cosharers=3,
        jaccard_threshold=0.5,
        evolution_window_days=7,
        pairs_table='quote_cosharing_pairs',
        clusters_table='quote_cosharing_clusters',
        membership_table='quote_cosharing_membership',
        source_table='osprey_execution_results',
    )


@pytest.fixture
def app_config(base_config: AnalysisConfig) -> AppConfig:
    return AppConfig(
        clickhouse=ClickHouseConfig(
            host='localhost',
            port=8123,
            user='default',
            password='clickhouse',
            database='default',
        ),
        analysis=base_config,
    )


class TestSanitizeDid:
    """Tests for _sanitize_did defence-in-depth SQL sanitization."""

    def test_valid_did_passes_unchanged(self) -> None:
        """Valid did:plc:abc123 passes through unchanged."""
        from quote_cosharing.main import _sanitize_did

        result = _sanitize_did('did:plc:abc123')
        assert result == 'did:plc:abc123'

    def test_injection_attempt_stripped(self) -> None:
        """Injection attempt like did:plc:abc'; DROP TABLE-- is stripped to safe characters."""
        from quote_cosharing.main import _sanitize_did

        result = _sanitize_did("did:plc:abc'; DROP TABLE--")
        assert result == 'did:plc:abc'

    def test_empty_string_returns_empty(self) -> None:
        """Empty string returns empty string."""
        from quote_cosharing.main import _sanitize_did

        result = _sanitize_did('')
        assert result == ''

    def test_uppercase_characters_stripped(self) -> None:
        """Uppercase characters are stripped (regex only allows lowercase)."""
        from quote_cosharing.main import _sanitize_did

        result = _sanitize_did('did:plc:ABC123xyz')
        assert result == 'did:plc:123xyz'

    def test_special_characters_removed(self) -> None:
        """Special characters are removed, keeping only [a-z0-9:.]."""
        from quote_cosharing.main import _sanitize_did

        result = _sanitize_did('did:plc:test!@#$%^&*()')
        assert result == 'did:plc:test'

    def test_period_preserved(self) -> None:
        """Period character is preserved (allowed in regex)."""
        from quote_cosharing.main import _sanitize_did

        result = _sanitize_did('did:plc:test.value')
        assert result == 'did:plc:test.value'

    def test_multiple_colons_preserved(self) -> None:
        """Multiple colons are preserved (allowed in regex)."""
        from quote_cosharing.main import _sanitize_did

        result = _sanitize_did('did:plc:test:value:more')
        assert result == 'did:plc:test:value:more'

    def test_digits_preserved(self) -> None:
        """Digits are preserved (allowed in regex)."""
        from quote_cosharing.main import _sanitize_did

        result = _sanitize_did('did:plc:123456789')
        assert result == 'did:plc:123456789'

    def test_complex_injection_attempt(self) -> None:
        """Complex injection with mixed valid/invalid characters is sanitized."""
        from quote_cosharing.main import _sanitize_did

        result = _sanitize_did("did:plc:abc' OR '1'='1")
        assert result == 'did:plc:abc11'


class TestLatestPreviousMembership:
    """Tests for _latest_previous_membership snapshot selection."""

    def test_uses_only_latest_snapshot_per_cluster(self) -> None:
        """Departed members from older snapshots must not accumulate."""
        from quote_cosharing.main import _latest_previous_membership

        rows = [
            MembershipRow(run_date=date(2026, 7, 4), cluster_id='c1', did='did:plc:a'),
            MembershipRow(run_date=date(2026, 7, 4), cluster_id='c1', did='did:plc:b'),
            MembershipRow(run_date=date(2026, 7, 4), cluster_id='c1', did='did:plc:c'),
            MembershipRow(run_date=date(2026, 7, 6), cluster_id='c1', did='did:plc:b'),
            MembershipRow(run_date=date(2026, 7, 6), cluster_id='c1', did='did:plc:c'),
            MembershipRow(run_date=date(2026, 7, 6), cluster_id='c1', did='did:plc:d'),
        ]

        result = _latest_previous_membership(rows)

        assert result == {'c1': frozenset({'did:plc:b', 'did:plc:c', 'did:plc:d'})}

    def test_each_cluster_uses_own_latest_date(self) -> None:
        """Each cluster resolves its own latest snapshot date."""
        from quote_cosharing.main import _latest_previous_membership

        rows = [
            MembershipRow(run_date=date(2026, 7, 6), cluster_id='c1', did='did:plc:a'),
            MembershipRow(run_date=date(2026, 7, 3), cluster_id='c2', did='did:plc:x'),
            MembershipRow(run_date=date(2026, 7, 5), cluster_id='c2', did='did:plc:y'),
        ]

        result = _latest_previous_membership(rows)

        assert result == {
            'c1': frozenset({'did:plc:a'}),
            'c2': frozenset({'did:plc:y'}),
        }

    def test_result_independent_of_row_order(self) -> None:
        """Result does not depend on row ordering from the query."""
        from quote_cosharing.main import _latest_previous_membership

        rows = [
            MembershipRow(run_date=date(2026, 7, 3), cluster_id='c1', did='did:plc:old'),
            MembershipRow(run_date=date(2026, 7, 6), cluster_id='c1', did='did:plc:new'),
        ]

        assert _latest_previous_membership(rows) == _latest_previous_membership(rows[::-1])
        assert _latest_previous_membership(rows) == {'c1': frozenset({'did:plc:new'})}

    def test_empty_history(self) -> None:
        from quote_cosharing.main import _latest_previous_membership

        assert _latest_previous_membership([]) == {}


class TestRunCycle:
    def test_run_cycle_processes_pairs_and_writes_clusters(self, app_config: AppConfig) -> None:
        """Test that run_cycle processes pairs and writes clusters."""
        fake_db = FakeDb()

        fake_db.pairs = [
            PairRow(
                date=date(2026, 3, 22),
                account_a='did:plc:user1',
                account_b='did:plc:user2',
                weight=5,
                newman_weight=1.0,
                shared_uris=['at://did:plc:user1/app.bsky.feed.post/abc123'],
            ),
            PairRow(
                date=date(2026, 3, 22),
                account_a='did:plc:user1',
                account_b='did:plc:user3',
                weight=4,
                newman_weight=1.0,
                shared_uris=['at://did:plc:user1/app.bsky.feed.post/abc123'],
            ),
            PairRow(
                date=date(2026, 3, 22),
                account_a='did:plc:user2',
                account_b='did:plc:user3',
                weight=3,
                newman_weight=1.0,
                shared_uris=['at://did:plc:user1/app.bsky.feed.post/abc123'],
            ),
        ]

        fake_db.timestamp_rows = [
            MemberTimestamp(did='did:plc:user1', ts=datetime(2026, 3, 21, 12, 0, 0)),
            MemberTimestamp(did='did:plc:user2', ts=datetime(2026, 3, 21, 13, 0, 0)),
            MemberTimestamp(did='did:plc:user3', ts=datetime(2026, 3, 21, 14, 0, 0)),
        ]

        run_cycle(fake_db, app_config)

        assert len(fake_db.captured_clusters) > 0
        assert len(fake_db.captured_membership) > 0

    def test_empty_pairs_skips_processing(self, app_config: AppConfig) -> None:
        """Test that empty pairs skip processing."""
        fake_db = FakeDb()
        fake_db.pairs = []

        run_cycle(fake_db, app_config)

        assert len(fake_db.captured_clusters) == 0
        assert len(fake_db.captured_membership) == 0

    def test_run_cycle_exception_propagates(self, app_config: AppConfig) -> None:
        """Test that exceptions from FakeDb propagate."""
        fake_db = FakeDb()

        def raise_error(query):
            raise RuntimeError('ClickHouse unavailable')

        fake_db.fetch_pairs = raise_error

        with pytest.raises(RuntimeError, match='ClickHouse unavailable'):
            run_cycle(fake_db, app_config)

    def test_run_cycle_creates_birth_clusters(self, app_config: AppConfig) -> None:
        """Test that clusters with no historical match are marked as birth."""
        fake_db = FakeDb()

        fake_db.pairs = [
            PairRow(
                date=date(2026, 3, 22),
                account_a='did:plc:user1',
                account_b='did:plc:user2',
                weight=5,
                newman_weight=1.0,
                shared_uris=['at://did:plc:user1/app.bsky.feed.post/abc123'],
            ),
            PairRow(
                date=date(2026, 3, 22),
                account_a='did:plc:user1',
                account_b='did:plc:user3',
                weight=4,
                newman_weight=1.0,
                shared_uris=['at://did:plc:user1/app.bsky.feed.post/abc123'],
            ),
            PairRow(
                date=date(2026, 3, 22),
                account_a='did:plc:user2',
                account_b='did:plc:user3',
                weight=3,
                newman_weight=1.0,
                shared_uris=['at://did:plc:user1/app.bsky.feed.post/abc123'],
            ),
        ]

        fake_db.timestamp_rows = [
            MemberTimestamp(did='did:plc:user1', ts=datetime(2026, 3, 21, 12, 0, 0)),
            MemberTimestamp(did='did:plc:user2', ts=datetime(2026, 3, 21, 13, 0, 0)),
            MemberTimestamp(did='did:plc:user3', ts=datetime(2026, 3, 21, 14, 0, 0)),
        ]

        run_cycle(fake_db, app_config)

        assert len(fake_db.captured_clusters) > 0
        for run_date, cluster, event in fake_db.captured_clusters:
            assert event.evolution_type == 'birth'
            assert event.jaccard_score == 0.0

    def test_run_cycle_groups_membership_by_cluster(self, app_config: AppConfig) -> None:
        """Test that membership rows are created for each cluster member."""
        fake_db = FakeDb()

        fake_db.pairs = [
            PairRow(
                date=date(2026, 3, 22),
                account_a='did:plc:user1',
                account_b='did:plc:user2',
                weight=5,
                newman_weight=1.0,
                shared_uris=['at://did:plc:user1/app.bsky.feed.post/abc123'],
            ),
            PairRow(
                date=date(2026, 3, 22),
                account_a='did:plc:user1',
                account_b='did:plc:user3',
                weight=4,
                newman_weight=1.0,
                shared_uris=['at://did:plc:user1/app.bsky.feed.post/abc123'],
            ),
            PairRow(
                date=date(2026, 3, 22),
                account_a='did:plc:user2',
                account_b='did:plc:user3',
                weight=3,
                newman_weight=1.0,
                shared_uris=['at://did:plc:user1/app.bsky.feed.post/abc123'],
            ),
        ]

        fake_db.timestamp_rows = [
            MemberTimestamp(did='did:plc:user1', ts=datetime(2026, 3, 21, 12, 0, 0)),
            MemberTimestamp(did='did:plc:user2', ts=datetime(2026, 3, 21, 13, 0, 0)),
            MemberTimestamp(did='did:plc:user3', ts=datetime(2026, 3, 21, 14, 0, 0)),
        ]

        run_cycle(fake_db, app_config)

        membership = fake_db.captured_membership
        assert len(membership) >= 3

        for run_date, cluster_id, did in membership:
            assert isinstance(run_date, date)
            assert isinstance(cluster_id, str)
            assert isinstance(did, str)

    def test_run_cycle_deletes_before_insert(self, app_config: AppConfig) -> None:
        """Test that stale data for today is deleted before inserting new results."""
        fake_db = FakeDb()

        fake_db.pairs = [
            PairRow(
                date=date(2026, 3, 22),
                account_a='did:plc:user1',
                account_b='did:plc:user2',
                weight=5,
                newman_weight=1.0,
                shared_uris=['at://did:plc:user1/app.bsky.feed.post/abc123'],
            ),
            PairRow(
                date=date(2026, 3, 22),
                account_a='did:plc:user1',
                account_b='did:plc:user3',
                weight=4,
                newman_weight=1.0,
                shared_uris=['at://did:plc:user1/app.bsky.feed.post/abc123'],
            ),
            PairRow(
                date=date(2026, 3, 22),
                account_a='did:plc:user2',
                account_b='did:plc:user3',
                weight=3,
                newman_weight=1.0,
                shared_uris=['at://did:plc:user1/app.bsky.feed.post/abc123'],
            ),
        ]

        fake_db.timestamp_rows = [
            MemberTimestamp(did='did:plc:user1', ts=datetime(2026, 3, 21, 12, 0, 0)),
            MemberTimestamp(did='did:plc:user2', ts=datetime(2026, 3, 21, 13, 0, 0)),
            MemberTimestamp(did='did:plc:user3', ts=datetime(2026, 3, 21, 14, 0, 0)),
        ]

        run_cycle(fake_db, app_config)

        assert len(fake_db.deleted_run_dates) == 2
        tables_deleted = [table for table, _ in fake_db.deleted_run_dates]
        assert 'quote_cosharing_clusters' in tables_deleted
        assert 'quote_cosharing_membership' in tables_deleted

    def test_empty_pairs_skips_delete(self, app_config: AppConfig) -> None:
        """Test that no deletion happens when there are no pairs to process."""
        fake_db = FakeDb()
        fake_db.pairs = []

        run_cycle(fake_db, app_config)

        assert len(fake_db.deleted_run_dates) == 0

    def test_run_cycle_with_temporal_timestamps(self, app_config: AppConfig) -> None:
        """Test that temporal metrics are computed correctly."""
        fake_db = FakeDb()

        fake_db.pairs = [
            PairRow(
                date=date(2026, 3, 22),
                account_a='did:plc:user1',
                account_b='did:plc:user2',
                weight=5,
                newman_weight=1.0,
                shared_uris=['at://did:plc:user1/app.bsky.feed.post/abc123'],
            ),
            PairRow(
                date=date(2026, 3, 22),
                account_a='did:plc:user1',
                account_b='did:plc:user3',
                weight=4,
                newman_weight=1.0,
                shared_uris=['at://did:plc:user1/app.bsky.feed.post/abc123'],
            ),
            PairRow(
                date=date(2026, 3, 22),
                account_a='did:plc:user2',
                account_b='did:plc:user3',
                weight=3,
                newman_weight=1.0,
                shared_uris=['at://did:plc:user1/app.bsky.feed.post/abc123'],
            ),
        ]

        fake_db.timestamp_rows = [
            MemberTimestamp(did='did:plc:user1', ts=datetime(2026, 3, 21, 10, 0, 0)),
            MemberTimestamp(did='did:plc:user1', ts=datetime(2026, 3, 21, 12, 0, 0)),
            MemberTimestamp(did='did:plc:user2', ts=datetime(2026, 3, 21, 13, 0, 0)),
            MemberTimestamp(did='did:plc:user3', ts=datetime(2026, 3, 21, 14, 0, 0)),
        ]

        run_cycle(fake_db, app_config)

        assert len(fake_db.captured_clusters) > 0
        for run_date, cluster, event in fake_db.captured_clusters:
            assert cluster.temporal_spread_hours >= 0.0
            assert cluster.mean_posting_interval_seconds >= 0.0


class RecordingCounter:
    def __init__(self) -> None:
        self.calls: list[tuple[int, dict[str, object] | None]] = []

    def add(self, value: int, attributes: dict[str, object] | None = None) -> None:
        self.calls.append((value, attributes))


class RecordingHistogram:
    def __init__(self) -> None:
        self.calls: list[tuple[int | float, dict[str, object] | None]] = []

    def record(self, value: int | float, attributes: dict[str, object] | None = None) -> None:
        self.calls.append((value, attributes))


@pytest.fixture
def telemetry_handles():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from quote_cosharing.telemetry import TelemetryHandles

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    handles = TelemetryHandles(
        tracer=provider.get_tracer('quote_cosharing.tests'),
        meter=object(),  # type: ignore[arg-type]
        runs_total=RecordingCounter(),  # type: ignore[arg-type]
        runs_failed_total=RecordingCounter(),  # type: ignore[arg-type]
        run_duration_seconds=RecordingHistogram(),  # type: ignore[arg-type]
        stage_duration_seconds=RecordingHistogram(),  # type: ignore[arg-type]
        pairs_fetched=RecordingHistogram(),  # type: ignore[arg-type]
        graph_nodes=RecordingHistogram(),  # type: ignore[arg-type]
        graph_edges=RecordingHistogram(),  # type: ignore[arg-type]
        cluster_count=RecordingHistogram(),  # type: ignore[arg-type]
        membership_rows=RecordingHistogram(),  # type: ignore[arg-type]
        shutdown_callbacks=(provider.shutdown,),
    )
    return handles, exporter


def _three_pair_cluster() -> list[PairRow]:
    return [
        PairRow(
            date=date(2026, 3, 22),
            account_a='did:plc:user1',
            account_b='did:plc:user2',
            weight=5,
            newman_weight=1.0,
            shared_uris=['at://did:plc:secret/app.bsky.feed.post/abc123'],
        ),
        PairRow(
            date=date(2026, 3, 22),
            account_a='did:plc:user1',
            account_b='did:plc:user3',
            weight=4,
            newman_weight=1.0,
            shared_uris=['at://did:plc:secret/app.bsky.feed.post/abc123'],
        ),
        PairRow(
            date=date(2026, 3, 22),
            account_a='did:plc:user2',
            account_b='did:plc:user3',
            weight=3,
            newman_weight=1.0,
            shared_uris=['at://did:plc:secret/app.bsky.feed.post/abc123'],
        ),
    ]


def test_run_cycle_records_quote_cosharing_stage_spans_and_metrics(app_config: AppConfig, telemetry_handles) -> None:
    handles, exporter = telemetry_handles
    fake_db = FakeDb()
    fake_db.pairs = _three_pair_cluster()
    fake_db.timestamp_rows = [
        MemberTimestamp(did='did:plc:user1', ts=datetime(2026, 3, 21, 12, 0, 0)),
        MemberTimestamp(did='did:plc:user2', ts=datetime(2026, 3, 21, 13, 0, 0)),
        MemberTimestamp(did='did:plc:user3', ts=datetime(2026, 3, 21, 14, 0, 0)),
    ]

    run_cycle(fake_db, app_config, handles)

    span_names = {span.name for span in exporter.get_finished_spans()}
    expected = {
        'quote_cosharing.run_cycle',
        'quote_cosharing.fetch_pairs',
        'quote_cosharing.build_graph',
        'quote_cosharing.cluster_graph',
        'quote_cosharing.fetch_member_timestamps',
        'quote_cosharing.compute_temporal_metrics',
        'quote_cosharing.fetch_historical_membership',
        'quote_cosharing.compute_evolution',
        'quote_cosharing.delete_stale_run_date',
        'quote_cosharing.persist_clusters',
        'quote_cosharing.persist_membership',
    }
    assert expected <= span_names
    assert len(handles.runs_total.calls) == 1
    assert len(handles.stage_duration_seconds.calls) == 10
    rendered = (
        repr(exporter.get_finished_spans())
        + repr(handles.runs_total.calls)
        + repr(handles.stage_duration_seconds.calls)
    )
    assert 'did:plc:secret' not in rendered
    assert 'at://did:plc:secret' not in rendered


def test_run_cycle_empty_pairs_records_success_without_delete_or_failure(
    app_config: AppConfig, telemetry_handles
) -> None:
    handles, exporter = telemetry_handles
    fake_db = FakeDb()

    run_cycle(fake_db, app_config, handles)

    assert handles.runs_total.calls == [(1, {'had_pairs': False})]
    assert handles.runs_failed_total.calls == []
    assert fake_db.deleted_run_dates == []
    assert fake_db.captured_clusters == []
    span_names = {span.name for span in exporter.get_finished_spans()}
    assert 'quote_cosharing.fetch_pairs' in span_names
    assert 'quote_cosharing.delete_stale_run_date' not in span_names


def test_run_cycle_failure_records_error_type_and_reraises(app_config: AppConfig, telemetry_handles) -> None:
    handles, _ = telemetry_handles
    fake_db = FakeDb()

    def fail(_query: str):
        raise RuntimeError('private failure message')

    fake_db.fetch_pairs = fail
    with pytest.raises(RuntimeError, match='private failure message'):
        run_cycle(fake_db, app_config, handles)

    assert handles.runs_failed_total.calls == [(1, {'stage': 'run_cycle', 'error.type': 'RuntimeError'})]
    assert 'private failure message' not in repr(handles.runs_failed_total.calls)
