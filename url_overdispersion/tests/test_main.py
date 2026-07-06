from __future__ import annotations

from datetime import datetime
from typing import Sequence

import pytest

from url_overdispersion.config import AnalysisConfig, AppConfig, ClickHouseConfig
from url_overdispersion.db import AggregatedRow, ScoredResult
from url_overdispersion.main import run_cycle


def create_test_row(
    domain: str,
    bucket_start: datetime,
    total_shares: int,
    unique_sharers: int,
    sharer_density: float,
    rolling_volume_mean: float,
    rolling_density_mean: float,
    baseline_days_available: int,
    sample_dids: list[str],
    sample_urls: list[str],
) -> AggregatedRow:
    """Helper to create AggregatedRow with new fields defaulting to None."""
    return AggregatedRow(
        domain=domain,
        bucket_start=bucket_start,
        total_shares=total_shares,
        unique_sharers=unique_sharers,
        sharer_density=sharer_density,
        rolling_volume_median=rolling_volume_mean,
        rolling_volume_mean=rolling_volume_mean,
        rolling_volume_variance=rolling_volume_mean,
        rolling_density_mean=rolling_density_mean,
        rolling_density_variance=rolling_density_mean * 0.1,
        baseline_days_available=baseline_days_available,
        sample_dids=sample_dids,
        sample_urls=sample_urls,
        population_volume_median=None,
        population_volume_dispersion=None,
        population_density_median=None,
        population_density_variance=None,
    )


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
        watchlist_domains=(),
        source_table='osprey_execution_results',
        output_table='url_overdispersion_results',
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
    def test_ac1_8_writes_results_with_sample_evidence(
        self,
        app_config: AppConfig,
    ) -> None:
        """Verify sample_dids and sample_urls are captured in results (AC1.8)."""
        fake_db = FakeDb()
        bucket = datetime(2026, 3, 20, 0, 0, 0)

        row = create_test_row(
            domain='example.com',
            bucket_start=bucket,
            total_shares=100,
            unique_sharers=20,
            sharer_density=0.2,
            rolling_volume_mean=50.0,
            rolling_density_mean=0.1,
            baseline_days_available=7,
            sample_dids=['did1', 'did2', 'did3'],
            sample_urls=['https://example.com/1', 'https://example.com/2'],
        )

        fake_db.daily_rows = [row]
        fake_db.hourly_rows = []

        run_cycle(fake_db, app_config)

        # Verify results captured with non-empty samples
        assert len(fake_db.captured_results) == 1
        result = fake_db.captured_results[0]
        assert result.sample_dids == ['did1', 'did2', 'did3']
        assert result.sample_urls == ['https://example.com/1', 'https://example.com/2']

    def test_ac1_9_watchlist_enrichment_on_watchlist(
        self,
        base_config: AnalysisConfig,
        app_config: AppConfig,
    ) -> None:
        """Verify on_watchlist=1 when domain matches watchlist (AC1.9)."""
        fake_db = FakeDb()
        bucket = datetime(2026, 3, 20, 0, 0, 0)

        # Create config with watchlist
        config_with_watchlist = AppConfig(
            clickhouse=app_config.clickhouse,
            analysis=AnalysisConfig(
                interval_seconds=base_config.interval_seconds,
                volume_p_threshold=base_config.volume_p_threshold,
                density_p_threshold=base_config.density_p_threshold,
                baseline_days=base_config.baseline_days,
                cold_start_min_days=base_config.cold_start_min_days,
                min_sharers=base_config.min_sharers,
                watchlist_domains=('evil.com',),
                source_table=base_config.source_table,
                output_table=base_config.output_table,
            ),
        )

        row = create_test_row(
            domain='evil.com',
            bucket_start=bucket,
            total_shares=10,
            unique_sharers=5,
            sharer_density=0.5,
            rolling_volume_mean=1.0,
            rolling_density_mean=0.5,
            baseline_days_available=7,
            sample_dids=['did1'],
            sample_urls=['https://evil.com/post'],
        )

        fake_db.daily_rows = [row]
        fake_db.hourly_rows = []

        run_cycle(fake_db, config_with_watchlist)

        result = fake_db.captured_results[0]
        assert result.on_watchlist == 1

    def test_ac1_9_non_watchlist_domain(
        self,
        app_config: AppConfig,
    ) -> None:
        """Verify on_watchlist=0 when domain not in watchlist (AC1.9)."""
        fake_db = FakeDb()
        bucket = datetime(2026, 3, 20, 0, 0, 0)

        row = create_test_row(
            domain='normal.com',
            bucket_start=bucket,
            total_shares=10,
            unique_sharers=5,
            sharer_density=0.5,
            rolling_volume_mean=1.0,
            rolling_density_mean=0.5,
            baseline_days_available=7,
            sample_dids=['did1'],
            sample_urls=['https://normal.com/post'],
        )

        fake_db.daily_rows = [row]
        fake_db.hourly_rows = []

        run_cycle(fake_db, app_config)

        result = fake_db.captured_results[0]
        assert result.on_watchlist == 0

    def test_ac3_1_both_granularities_processed(
        self,
        app_config: AppConfig,
    ) -> None:
        """Verify both daily and hourly granularities are processed (AC3.1)."""
        fake_db = FakeDb()
        bucket_daily = datetime(2026, 3, 20, 0, 0, 0)
        bucket_hourly = datetime(2026, 3, 20, 12, 0, 0)

        daily_row = create_test_row(
            domain='daily.com',
            bucket_start=bucket_daily,
            total_shares=50,
            unique_sharers=10,
            sharer_density=0.2,
            rolling_volume_mean=25.0,
            rolling_density_mean=0.15,
            baseline_days_available=7,
            sample_dids=['did1'],
            sample_urls=['https://daily.com/post'],
        )

        hourly_row = create_test_row(
            domain='hourly.com',
            bucket_start=bucket_hourly,
            total_shares=5,
            unique_sharers=3,
            sharer_density=0.6,
            rolling_volume_mean=2.0,
            rolling_density_mean=0.5,
            baseline_days_available=7,
            sample_dids=['did2'],
            sample_urls=['https://hourly.com/post'],
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
        assert daily_results[0].domain == 'daily.com'
        assert hourly_results[0].domain == 'hourly.com'

    def test_ac3_1_empty_rows_skipped(
        self,
        app_config: AppConfig,
    ) -> None:
        """Verify empty row sets are skipped without writing (AC3.1)."""
        fake_db = FakeDb()
        bucket = datetime(2026, 3, 20, 0, 0, 0)

        daily_row = create_test_row(
            domain='daily.com',
            bucket_start=bucket,
            total_shares=50,
            unique_sharers=10,
            sharer_density=0.2,
            rolling_volume_mean=25.0,
            rolling_density_mean=0.15,
            baseline_days_available=7,
            sample_dids=['did1'],
            sample_urls=['https://daily.com/post'],
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
        anomalous_row = create_test_row(
            domain='anomalous.com',
            bucket_start=bucket,
            total_shares=200,
            unique_sharers=20,
            sharer_density=0.1,
            rolling_volume_mean=50.0,
            rolling_density_mean=0.05,
            baseline_days_available=7,
            sample_dids=['did1'],
            sample_urls=['https://anomalous.com/post'],
        )

        # Low shares -> normal
        normal_row = create_test_row(
            domain='normal.com',
            bucket_start=bucket,
            total_shares=52,
            unique_sharers=10,
            sharer_density=0.19,
            rolling_volume_mean=50.0,
            rolling_density_mean=0.05,
            baseline_days_available=7,
            sample_dids=['did2'],
            sample_urls=['https://normal.com/post'],
        )

        fake_db.daily_rows = [anomalous_row, normal_row]
        fake_db.hourly_rows = []

        run_cycle(fake_db, app_config)

        # Both should be captured
        assert len(fake_db.captured_results) == 2

        anomalous = next(r for r in fake_db.captured_results if r.domain == 'anomalous.com')
        normal = next(r for r in fake_db.captured_results if r.domain == 'normal.com')

        assert anomalous.is_anomaly == 1
        assert normal.is_anomaly == 0

    def test_run_cycle_uses_real_analyzer(
        self,
        app_config: AppConfig,
    ) -> None:
        """Verify run_cycle uses the real analyzer scoring logic."""
        fake_db = FakeDb()
        bucket = datetime(2026, 3, 20, 0, 0, 0)

        test_row = create_test_row(
            domain='test.com',
            bucket_start=bucket,
            total_shares=200,
            unique_sharers=20,
            sharer_density=0.1,
            rolling_volume_mean=50.0,
            rolling_density_mean=0.05,
            baseline_days_available=7,
            sample_dids=['did1'],
            sample_urls=['https://test.com/post'],
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
