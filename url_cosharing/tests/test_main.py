from __future__ import annotations

from datetime import date, datetime
from typing import Sequence

import pytest

from url_cosharing.analyzer import EvolutionEvent, TimestampedCluster
from url_cosharing.config import AnalysisConfig, AppConfig, ClickHouseConfig
from url_cosharing.db import MembershipRow, MemberTimestamp, RunMetadata
from url_cosharing.main import run_cycle
from url_cosharing.similarity import UrlShareRow


class FakeDb:
    """In-memory stub for database layer, captures inserted results."""

    def __init__(self) -> None:
        self.url_shares: list[UrlShareRow] = []
        self.raw_account_count: int = 0
        self.membership_rows: list[MembershipRow] = []
        self.timestamp_rows: list[MemberTimestamp] = []
        self.captured_clusters: list[tuple[date, TimestampedCluster, EvolutionEvent]] = []
        self.captured_membership: list[tuple[date, str, str]] = []
        self.captured_runs: list[tuple[str, RunMetadata]] = []
        self.deleted_run_dates: list[tuple[str, date]] = []

    def fetch_url_shares(self, query: str) -> list[UrlShareRow]:
        """Returns pre-configured URL shares."""
        return self.url_shares

    def fetch_raw_account_count(self, query: str) -> int:
        """Returns the pre-configured raw window account count."""
        return self.raw_account_count

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

    def insert_run(self, table: str, run: RunMetadata) -> None:
        """Captures run metadata for later assertion."""
        self.captured_runs.append((table, run))

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
        resolution=0.5,
        min_cluster_size=3,
        jaccard_threshold=0.5,
        evolution_window_days=7,
        window_days=7,
        min_unique_urls=2,
        min_url_sharers=2,
        max_url_df_fraction=0.99,
        edge_epsilon=0.01,
        edge_quantile_grid=(0.5, 0.9),
        centrality_quantile_grid=(0.5, 0.9),
        density_floor=0.5,
        max_flagged_fraction=0.9,
        runs_table='url_cosharing_runs',
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


class TestLatestPreviousMembership:
    """Tests for _latest_previous_membership snapshot selection."""

    def test_uses_only_latest_snapshot_per_cluster(self) -> None:
        """Departed members from older snapshots must not accumulate."""
        from url_cosharing.main import _latest_previous_membership

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

    def test_clusters_keep_independent_latest_dates(self) -> None:
        """Each cluster resolves its own latest snapshot date."""
        from url_cosharing.main import _latest_previous_membership

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

    def test_order_independent(self) -> None:
        """Result does not depend on row ordering from the query."""
        from url_cosharing.main import _latest_previous_membership

        rows = [
            MembershipRow(run_date=date(2026, 7, 3), cluster_id='c1', did='did:plc:old'),
            MembershipRow(run_date=date(2026, 7, 6), cluster_id='c1', did='did:plc:new'),
        ]

        assert _latest_previous_membership(rows) == _latest_previous_membership(rows[::-1])
        assert _latest_previous_membership(rows) == {'c1': frozenset({'did:plc:new'})}

    def test_empty_history(self) -> None:
        from url_cosharing.main import _latest_previous_membership

        assert _latest_previous_membership([]) == {}


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
    def test_full_cycle_with_coordinated_accounts(self, app_config: AppConfig) -> None:
        """Full cycle: coordinated 4 accounts share same 4 URLs, land in one cluster."""
        fake_db = FakeDb()

        # 4 coordinated accounts, each sharing the same 4 URLs
        coordinated = ['did:plc:c1', 'did:plc:c2', 'did:plc:c3', 'did:plc:c4']
        urls = ['https://example.com/1', 'https://example.com/2', 'https://example.com/3', 'https://example.com/4']
        for account in coordinated:
            for url in urls:
                fake_db.url_shares.append(UrlShareRow(did=account, url=url, share_count=1))

        # 6 background accounts, each with 2 distinct URLs + 1 shared with a coordinated URL
        for i in range(6):
            account = f'did:plc:bg{i}'
            # Distinct URLs
            fake_db.url_shares.append(UrlShareRow(did=account, url=f'https://bg.com/{i}', share_count=1))
            fake_db.url_shares.append(UrlShareRow(did=account, url=f'https://bg.com/{i}_alt', share_count=1))
            # One shared with a coordinated URL (keeps df < N)
            fake_db.url_shares.append(UrlShareRow(did=account, url=urls[i % 4], share_count=1))

        # Timestamps for all members
        for account in coordinated:
            for hour in [10, 12, 14]:
                fake_db.timestamp_rows.append(MemberTimestamp(did=account, ts=datetime(2026, 3, 21, hour, 0, 0)))

        # Raw window population exceeds the 10 accounts surviving SQL filters
        fake_db.raw_account_count = 250

        run_cycle(fake_db, app_config)

        # Should delete three tables
        assert len(fake_db.deleted_run_dates) == 3
        tables_deleted = {table for table, _ in fake_db.deleted_run_dates}
        assert tables_deleted == {'url_cosharing_clusters', 'url_cosharing_membership', 'url_cosharing_runs'}

        # Should have one cluster with 4 members
        assert len(fake_db.captured_clusters) >= 1
        assert any(len(event.members) == 4 for _, _, event in fake_db.captured_clusters)

        # Membership rows for all cluster members
        assert len(fake_db.captured_membership) >= 4

        # Run metadata written exactly once
        assert len(fake_db.captured_runs) == 1
        table, run = fake_db.captured_runs[0]
        assert table == 'url_cosharing_runs'
        assert run.knee_found is True
        assert run.cluster_count >= 1
        assert run.flagged_accounts >= 4

        # accounts_raw is the pre-filter window population from its own query,
        # not the SQL-final row population the network is built from
        assert run.accounts_raw == 250
        assert run.accounts_eligible == 10

    def test_empty_input_writes_run_metadata(self, app_config: AppConfig) -> None:
        """Empty input: no clusters, run row written with zero counts."""
        fake_db = FakeDb()
        fake_db.url_shares = []

        run_cycle(fake_db, app_config)

        # Should still delete three tables (idempotency)
        assert len(fake_db.deleted_run_dates) == 3

        # No clusters or membership
        assert len(fake_db.captured_clusters) == 0
        assert len(fake_db.captured_membership) == 0

        # Run metadata still written with zero counts
        assert len(fake_db.captured_runs) == 1
        table, run = fake_db.captured_runs[0]
        assert table == 'url_cosharing_runs'
        assert run.accounts_raw == 0
        assert run.accounts_eligible == 0
        assert run.urls_eligible == 0
        assert run.cluster_count == 0
        assert run.knee_found is False

    def test_no_knee_forces_empty_clusters(self, app_config: AppConfig, base_config: AnalysisConfig) -> None:
        """No knee via density_floor=1.0: zero clusters, run records knee_found=False and edge_quantile==0.0."""
        fake_db = FakeDb()

        # Add some URL shares that would normally form a cluster
        for account in ['did:plc:c1', 'did:plc:c2', 'did:plc:c3']:
            for url in ['https://example.com/1', 'https://example.com/2']:
                fake_db.url_shares.append(UrlShareRow(did=account, url=url, share_count=1))

        # Create new config with density_floor=1.0
        no_knee_config = AnalysisConfig(
            interval_seconds=base_config.interval_seconds,
            resolution=base_config.resolution,
            min_cluster_size=base_config.min_cluster_size,
            jaccard_threshold=base_config.jaccard_threshold,
            evolution_window_days=base_config.evolution_window_days,
            window_days=base_config.window_days,
            min_unique_urls=base_config.min_unique_urls,
            min_url_sharers=base_config.min_url_sharers,
            max_url_df_fraction=base_config.max_url_df_fraction,
            edge_epsilon=base_config.edge_epsilon,
            edge_quantile_grid=base_config.edge_quantile_grid,
            centrality_quantile_grid=base_config.centrality_quantile_grid,
            density_floor=1.0,
            max_flagged_fraction=base_config.max_flagged_fraction,
            runs_table=base_config.runs_table,
            clusters_table=base_config.clusters_table,
            membership_table=base_config.membership_table,
            source_table=base_config.source_table,
        )
        test_config = AppConfig(clickhouse=app_config.clickhouse, analysis=no_knee_config)

        run_cycle(fake_db, test_config)

        # No clusters should be written
        assert len(fake_db.captured_clusters) == 0

        # Run metadata shows no knee and edge_quantile == 0.0 when no knee selected
        assert len(fake_db.captured_runs) == 1
        _, run = fake_db.captured_runs[0]
        assert run.knee_found is False
        assert run.edge_quantile == 0.0, "DDL documents edge_quantile=0 when no knee selected"
        assert run.cluster_count == 0

    def test_idempotency_deletes_all_three_tables(self, app_config: AppConfig) -> None:
        """Idempotency: delete_run_date called for clusters, membership, AND runs."""
        fake_db = FakeDb()

        fake_db.url_shares.append(UrlShareRow(did='did:plc:a1', url='https://example.com', share_count=1))
        fake_db.url_shares.append(UrlShareRow(did='did:plc:a2', url='https://example.com', share_count=1))

        run_cycle(fake_db, app_config)

        # Verify all three tables were deleted
        assert len(fake_db.deleted_run_dates) == 3
        tables = {table for table, _ in fake_db.deleted_run_dates}
        assert tables == {'url_cosharing_clusters', 'url_cosharing_membership', 'url_cosharing_runs'}

    def test_death_events_skipped_in_writes(self, app_config: AppConfig) -> None:
        """Verify death events are structurally prevented from being written.

        Plant a prior membership row for a cluster that dies (its members absent today),
        and a surviving cluster today. The test confirms two structural guarantees:

        (1) Death events cannot pair with written rows: main.py:132 zips timestamped_clusters
            (len == #current clusters) with events (where death events are appended after
            per-cluster events), so zip truncation structurally prevents death events from
            reaching the insertion layer.

        (2) The written behaviour is correct: death cluster ID is absent from cluster inserts,
            surviving cluster is present with non-empty captures, and surviving cluster membership
            is written for all members. Death cluster members do not appear in membership inserts.

        The death-skip filter (`if event.evolution_type != 'death'` at main.py:133, 140) is
        functionally present but structurally redundant due to zip truncation. This test
        positively pins the end behaviour without relying on the filter.
        """
        fake_db = FakeDb()

        # Plant a prior membership row for a cluster that will die
        # Use a date within the evolution_window_days (7 days)
        prior_cluster_id = '2026-07-06-0001'
        prior_member = 'did:plc:old_user'
        fake_db.membership_rows.append(
            MembershipRow(
                run_date=date(2026, 7, 6),
                cluster_id=prior_cluster_id,
                did=prior_member,
            )
        )

        # Coordinated group (surviving): 4 accounts share same 4 URLs
        # min_cluster_size=3, so this will form a cluster
        coordinated = ['did:plc:c1', 'did:plc:c2', 'did:plc:c3', 'did:plc:c4']
        urls = [
            'https://example.com/1', 'https://example.com/2',
            'https://example.com/3', 'https://example.com/4'
        ]
        for account in coordinated:
            for url in urls:
                fake_db.url_shares.append(UrlShareRow(did=account, url=url, share_count=1))

        # Background accounts to inflate df and keep coordinated URLs viable
        for i in range(6):
            account = f'did:plc:bg{i}'
            fake_db.url_shares.append(UrlShareRow(did=account, url=f'https://bg.com/{i}', share_count=1))
            fake_db.url_shares.append(UrlShareRow(did=account, url=f'https://bg.com/{i}_alt', share_count=1))
            fake_db.url_shares.append(UrlShareRow(did=account, url=urls[i % 4], share_count=1))

        # Timestamps for coordinated members
        for account in coordinated:
            for hour in [10, 12, 14]:
                fake_db.timestamp_rows.append(MemberTimestamp(did=account, ts=datetime(2026, 7, 7, hour, 0, 0)))

        run_cycle(fake_db, app_config)

        # Assert positively: surviving cluster IS present
        assert len(fake_db.captured_clusters) >= 1, "Should have at least one surviving cluster"
        surviving_cluster_ids = {event.cluster_id for _, _, event in fake_db.captured_clusters}

        # UNCONDITIONAL ASSERTION: Death cluster_id must be ABSENT from cluster inserts.
        # Structurally guaranteed: zip(timestamped_clusters, events) truncates at the length
        # of timestamped_clusters (current clusters only), so death events appended after
        # per-cluster events can never pair with written rows.
        assert prior_cluster_id not in surviving_cluster_ids, (
            f"Death cluster {prior_cluster_id} must not appear in cluster inserts. "
            f"Got cluster IDs: {surviving_cluster_ids}. "
            f"This indicates death events are not being properly filtered."
        )

        # UNCONDITIONAL ASSERTION: Surviving cluster membership MUST be present
        # At least 4 rows (one for each coordinated member)
        assert len(fake_db.captured_membership) >= 4, (
            f"Should have at least 4 membership rows for surviving cluster members; "
            f"got {len(fake_db.captured_membership)}"
        )

        # UNCONDITIONAL ASSERTION: Death cluster member must not appear in membership inserts
        captured_membership_dids = {did for _, _, did in fake_db.captured_membership}
        assert prior_member not in captured_membership_dids, (
            f"Death cluster member {prior_member} must not appear in membership inserts"
        )

        # UNCONDITIONAL ASSERTION: All captured cluster events must be non-death types
        for _, _, event in fake_db.captured_clusters:
            assert event.evolution_type != 'death', (
                f"Death events must be skipped in cluster writes; got evolution_type={event.evolution_type}"
            )
