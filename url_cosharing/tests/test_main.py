from __future__ import annotations

from datetime import date, datetime
from typing import Sequence

import pytest

from url_cosharing.analyzer import EvolutionEvent, PairRow, TimestampedCluster
from url_cosharing.config import AnalysisConfig, AppConfig, ClickHouseConfig
from url_cosharing.db import MembershipRow, MemberTimestamp
from url_cosharing.main import run_cycle


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
        pairs_table='url_cosharing_pairs',
        clusters_table='url_cosharing_clusters',
        membership_table='url_cosharing_membership',
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
        from url_cosharing.main import _sanitize_did

        result = _sanitize_did('did:plc:abc123')
        assert result == 'did:plc:abc123'

    def test_injection_attempt_stripped(self) -> None:
        """Injection attempt like did:plc:abc'; DROP TABLE-- is stripped to safe characters."""
        from url_cosharing.main import _sanitize_did

        result = _sanitize_did("did:plc:abc'; DROP TABLE--")
        assert result == 'did:plc:abc'

    def test_empty_string_returns_empty(self) -> None:
        """Empty string returns empty string."""
        from url_cosharing.main import _sanitize_did

        result = _sanitize_did('')
        assert result == ''

    def test_uppercase_characters_stripped(self) -> None:
        """Uppercase characters are stripped (regex only allows lowercase)."""
        from url_cosharing.main import _sanitize_did

        result = _sanitize_did('did:plc:ABC123xyz')
        assert result == 'did:plc:123xyz'

    def test_special_characters_removed(self) -> None:
        """Special characters are removed, keeping only [a-z0-9:.]."""
        from url_cosharing.main import _sanitize_did

        result = _sanitize_did('did:plc:test!@#$%^&*()')
        assert result == 'did:plc:test'

    def test_period_preserved(self) -> None:
        """Period character is preserved (allowed in regex)."""
        from url_cosharing.main import _sanitize_did

        result = _sanitize_did('did:plc:test.value')
        assert result == 'did:plc:test.value'

    def test_multiple_colons_preserved(self) -> None:
        """Multiple colons are preserved (allowed in regex)."""
        from url_cosharing.main import _sanitize_did

        result = _sanitize_did('did:plc:test:value:more')
        assert result == 'did:plc:test:value:more'

    def test_digits_preserved(self) -> None:
        """Digits are preserved (allowed in regex)."""
        from url_cosharing.main import _sanitize_did

        result = _sanitize_did('did:plc:123456789')
        assert result == 'did:plc:123456789'

    def test_complex_injection_attempt(self) -> None:
        """Complex injection with mixed valid/invalid characters is sanitized."""
        from url_cosharing.main import _sanitize_did

        result = _sanitize_did("did:plc:abc' OR '1'='1")
        assert result == 'did:plc:abc11'


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
                newman_weight=5 / 2,
                shared_urls=['https://example.com'],
            ),
            PairRow(
                date=date(2026, 3, 22),
                account_a='did:plc:user1',
                account_b='did:plc:user3',
                weight=4,
                newman_weight=4 / 2,
                shared_urls=['https://example.com'],
            ),
            PairRow(
                date=date(2026, 3, 22),
                account_a='did:plc:user2',
                account_b='did:plc:user3',
                weight=3,
                newman_weight=3 / 2,
                shared_urls=['https://example.com'],
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
                newman_weight=5 / 2,
                shared_urls=['https://example.com'],
            ),
            PairRow(
                date=date(2026, 3, 22),
                account_a='did:plc:user1',
                account_b='did:plc:user3',
                weight=4,
                newman_weight=4 / 2,
                shared_urls=['https://example.com'],
            ),
            PairRow(
                date=date(2026, 3, 22),
                account_a='did:plc:user2',
                account_b='did:plc:user3',
                weight=3,
                newman_weight=3 / 2,
                shared_urls=['https://example.com'],
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
                newman_weight=5 / 2,
                shared_urls=['https://example.com'],
            ),
            PairRow(
                date=date(2026, 3, 22),
                account_a='did:plc:user1',
                account_b='did:plc:user3',
                weight=4,
                newman_weight=4 / 2,
                shared_urls=['https://example.com'],
            ),
            PairRow(
                date=date(2026, 3, 22),
                account_a='did:plc:user2',
                account_b='did:plc:user3',
                weight=3,
                newman_weight=3 / 2,
                shared_urls=['https://example.com'],
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
                newman_weight=5 / 2,
                shared_urls=['https://example.com'],
            ),
            PairRow(
                date=date(2026, 3, 22),
                account_a='did:plc:user1',
                account_b='did:plc:user3',
                weight=4,
                newman_weight=4 / 2,
                shared_urls=['https://example.com'],
            ),
            PairRow(
                date=date(2026, 3, 22),
                account_a='did:plc:user2',
                account_b='did:plc:user3',
                weight=3,
                newman_weight=3 / 2,
                shared_urls=['https://example.com'],
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
        assert 'url_cosharing_clusters' in tables_deleted
        assert 'url_cosharing_membership' in tables_deleted

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
                newman_weight=5 / 2,
                shared_urls=['https://example.com'],
            ),
            PairRow(
                date=date(2026, 3, 22),
                account_a='did:plc:user1',
                account_b='did:plc:user3',
                weight=4,
                newman_weight=4 / 2,
                shared_urls=['https://example.com'],
            ),
            PairRow(
                date=date(2026, 3, 22),
                account_a='did:plc:user2',
                account_b='did:plc:user3',
                weight=3,
                newman_weight=3 / 2,
                shared_urls=['https://example.com'],
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
