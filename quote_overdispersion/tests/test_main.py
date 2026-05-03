# pattern: Functional Core
from __future__ import annotations

from datetime import datetime
from typing import Sequence

import pytest

from quote_overdispersion.config import AnalysisConfig, AppConfig, ClickHouseConfig
from quote_overdispersion.db import AggregatedRow, ScoredResult
from quote_overdispersion.main import run_cycle


class FakeDb:
    """In-memory stub for database layer, captures inserted results."""

    def __init__(self) -> None:
        self.daily_rows: list[AggregatedRow] = []
        self.hourly_rows: list[AggregatedRow] = []
        self.captured_results: list[ScoredResult] = []

    def fetch_aggregated_rows(self, query: str) -> list[AggregatedRow]:
        """Returns pre-configured rows based on query content."""
        if 'toDate' in query:
            return self.daily_rows
        else:
            return self.hourly_rows

    def insert_results(self, table: str, results: Sequence[ScoredResult]) -> None:
        """Captures results for later assertion."""
        self.captured_results.extend(results)

    def close(self) -> None:
        """No-op."""
        pass


@pytest.fixture
def base_config() -> AnalysisConfig:
    return AnalysisConfig(
        interval_seconds=900,
        volume_p_threshold=0.01,
        density_p_threshold=0.01,
        baseline_days=14,
        cold_start_min_days=3,
        min_sharers=3,
        source_table='osprey_execution_results',
        output_table='quote_overdispersion_results',
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


class TestRunCycle:
    def test_ac1_6_writes_quoted_author_did(
        self,
        app_config: AppConfig,
    ) -> None:
        """AC1.6: Verify quoted_author_did is extracted and stored in results."""
        fake_db = FakeDb()
        bucket = datetime(2026, 3, 20, 0, 0, 0)

        row = AggregatedRow(
            quoted_uri='at://did:plc:test123/app.bsky.feed.post/abc',
            bucket_start=bucket,
            total_shares=100,
            unique_sharers=20,
            sharer_density=0.2,
            rolling_volume_mean=50.0,
            rolling_density_mean=0.1,
            baseline_days_available=7,
            sample_dids=['did1', 'did2', 'did3'],
            population_volume_median=None,
            population_density_median=None,
        )

        fake_db.daily_rows = [row]
        fake_db.hourly_rows = []

        run_cycle(fake_db, app_config)

        # Verify quoted_author_did extracted from AT-URI
        assert len(fake_db.captured_results) == 1
        result = fake_db.captured_results[0]
        assert result.quoted_author_did == 'did:plc:test123'

    def test_ac1_7_malformed_uri_empty_quoted_author_did(
        self,
        app_config: AppConfig,
    ) -> None:
        """AC1.7: Malformed AT-URI produces empty string for quoted_author_did."""
        fake_db = FakeDb()
        bucket = datetime(2026, 3, 20, 0, 0, 0)

        row = AggregatedRow(
            quoted_uri='malformed-uri',
            bucket_start=bucket,
            total_shares=100,
            unique_sharers=20,
            sharer_density=0.2,
            rolling_volume_mean=50.0,
            rolling_density_mean=0.1,
            baseline_days_available=7,
            sample_dids=['did1'],
            population_volume_median=None,
            population_density_median=None,
        )

        fake_db.daily_rows = [row]
        fake_db.hourly_rows = []

        run_cycle(fake_db, app_config)

        # Verify empty quoted_author_did for malformed URI
        assert len(fake_db.captured_results) == 1
        result = fake_db.captured_results[0]
        assert result.quoted_author_did == ''

    def test_ac3_1_both_granularities_processed(
        self,
        app_config: AppConfig,
    ) -> None:
        """Verify both daily and hourly granularities are processed."""
        fake_db = FakeDb()
        bucket_daily = datetime(2026, 3, 20, 0, 0, 0)
        bucket_hourly = datetime(2026, 3, 20, 12, 0, 0)

        daily_row = AggregatedRow(
            quoted_uri='at://did:plc:daily/app.bsky.feed.post/post1',
            bucket_start=bucket_daily,
            total_shares=50,
            unique_sharers=10,
            sharer_density=0.2,
            rolling_volume_mean=25.0,
            rolling_density_mean=0.15,
            baseline_days_available=7,
            sample_dids=['did1'],
            population_volume_median=None,
            population_density_median=None,
        )

        hourly_row = AggregatedRow(
            quoted_uri='at://did:plc:hourly/app.bsky.feed.post/post2',
            bucket_start=bucket_hourly,
            total_shares=5,
            unique_sharers=3,
            sharer_density=0.6,
            rolling_volume_mean=2.0,
            rolling_density_mean=0.5,
            baseline_days_available=7,
            sample_dids=['did2'],
            population_volume_median=None,
            population_density_median=None,
        )

        fake_db.daily_rows = [daily_row]
        fake_db.hourly_rows = [hourly_row]

        run_cycle(fake_db, app_config)

        # Verify both granularities captured
        assert len(fake_db.captured_results) == 2
        daily_results = [r for r in fake_db.captured_results if r.granularity == 'daily']
        hourly_results = [r for r in fake_db.captured_results if r.granularity == 'hourly']
        assert len(daily_results) == 1
        assert len(hourly_results) == 1
        assert daily_results[0].quoted_author_did == 'did:plc:daily'
        assert hourly_results[0].quoted_author_did == 'did:plc:hourly'

    def test_ac3_1_empty_rows_skipped(
        self,
        app_config: AppConfig,
    ) -> None:
        """Verify empty row sets are skipped without writing."""
        fake_db = FakeDb()
        bucket = datetime(2026, 3, 20, 0, 0, 0)

        daily_row = AggregatedRow(
            quoted_uri='at://did:plc:daily/app.bsky.feed.post/post1',
            bucket_start=bucket,
            total_shares=50,
            unique_sharers=10,
            sharer_density=0.2,
            rolling_volume_mean=25.0,
            rolling_density_mean=0.15,
            baseline_days_available=7,
            sample_dids=['did1'],
            population_volume_median=None,
            population_density_median=None,
        )

        fake_db.daily_rows = [daily_row]
        fake_db.hourly_rows = []

        run_cycle(fake_db, app_config)

        # Only daily results, hourly was empty so skipped
        assert len(fake_db.captured_results) == 1
        assert fake_db.captured_results[0].granularity == 'daily'

    def test_all_results_written_both_normal_and_anomalous(
        self,
        app_config: AppConfig,
    ) -> None:
        """Verify both normal and anomalous results are written."""
        fake_db = FakeDb()
        bucket = datetime(2026, 3, 20, 0, 0, 0)

        # High shares -> likely anomaly
        anomalous_row = AggregatedRow(
            quoted_uri='at://did:plc:anom/app.bsky.feed.post/post1',
            bucket_start=bucket,
            total_shares=200,
            unique_sharers=20,
            sharer_density=0.1,
            rolling_volume_mean=50.0,
            rolling_density_mean=0.05,
            baseline_days_available=7,
            sample_dids=['did1'],
            population_volume_median=None,
            population_density_median=None,
        )

        # Low shares -> normal
        normal_row = AggregatedRow(
            quoted_uri='at://did:plc:normal/app.bsky.feed.post/post2',
            bucket_start=bucket,
            total_shares=52,
            unique_sharers=10,
            sharer_density=0.19,
            rolling_volume_mean=50.0,
            rolling_density_mean=0.05,
            baseline_days_available=7,
            sample_dids=['did2'],
            population_volume_median=None,
            population_density_median=None,
        )

        fake_db.daily_rows = [anomalous_row, normal_row]
        fake_db.hourly_rows = []

        run_cycle(fake_db, app_config)

        # Both should be captured
        assert len(fake_db.captured_results) == 2

        anomalous = next(r for r in fake_db.captured_results if r.quoted_author_did == 'did:plc:anom')
        normal = next(r for r in fake_db.captured_results if r.quoted_author_did == 'did:plc:normal')

        assert anomalous.is_anomaly == 1
        assert normal.is_anomaly == 0

    def test_run_cycle_uses_real_analyzer(
        self,
        app_config: AppConfig,
    ) -> None:
        """Verify run_cycle uses the real analyzer scoring logic."""
        fake_db = FakeDb()
        bucket = datetime(2026, 3, 20, 0, 0, 0)

        test_row = AggregatedRow(
            quoted_uri='at://did:plc:test/app.bsky.feed.post/post1',
            bucket_start=bucket,
            total_shares=200,
            unique_sharers=20,
            sharer_density=0.1,
            rolling_volume_mean=50.0,
            rolling_density_mean=0.05,
            baseline_days_available=7,
            sample_dids=['did1'],
            population_volume_median=None,
            population_density_median=None,
        )

        fake_db.daily_rows = [test_row]
        fake_db.hourly_rows = []

        run_cycle(fake_db, app_config)

        result = fake_db.captured_results[0]

        # Verify scoring applied
        assert result.expected_volume_lambda == 50.0
        assert result.expected_density_lambda == 0.05
        assert result.baseline_source == 'entity'
        # High observed (200) vs expected (50) -> anomaly
        assert result.is_anomaly == 1

    def test_ac3_4_cycle_exception_propagates(
        self,
        app_config: AppConfig,
    ) -> None:
        """Verify run_cycle propagates exceptions (try/except is in main loop)."""

        class FailingDb:
            def fetch_aggregated_rows(self, query: str) -> list[AggregatedRow]:
                raise RuntimeError('ClickHouse unavailable')

            def insert_results(self, table: str, results: Sequence[ScoredResult]) -> None:
                pass

            def close(self) -> None:
                pass

        fake_db = FailingDb()

        with pytest.raises(RuntimeError, match='ClickHouse unavailable'):
            run_cycle(fake_db, app_config)
