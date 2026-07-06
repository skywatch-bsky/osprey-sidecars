from __future__ import annotations

from typing import Sequence

import pytest

from signup_anomaly.config import AnalysisConfig, AppConfig, ClickHouseConfig
from signup_anomaly.db import AggregatedRow, ScoredResult
from signup_anomaly.main import run_cycle


class FakeDb:
    """In-memory stub for database layer, captures inserted results."""

    def __init__(self) -> None:
        self.daily_rows: list[AggregatedRow] = []
        self.hourly_rows: list[AggregatedRow] = []
        self.captured_results: list[ScoredResult] = []

    def fetch_aggregated_rows(self, query: str) -> list[AggregatedRow]:
        """Returns pre-configured rows based on query content."""
        if 'raw_counts' in query and 'toDate' in query:
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
        interval_seconds=3600,
        daily_p_value_threshold=0.01,
        hourly_p_value_threshold=0.05,
        baseline_days=7,
        cold_start_min_days=3,
        excluded_hosts=('bsky.network', 'bridgy-fed.appspot.com', 'mostr.pub'),
        source_table='osprey_execution_results',
        output_table='pds_signup_anomalies',
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
    def test_ac4_1_writes_all_results_with_columns_populated(
        self,
        app_config: AppConfig,
    ) -> None:
        """Verify end-to-end pipeline writes queryable results with all columns populated."""
        fake_db = FakeDb()

        # Configure with one obvious anomaly and one normal PDS
        anomaly_row = AggregatedRow(
            pds_host='anomalous.com',
            observed_count=200,
            distinct_accounts=200,
            rolling_mean=50.0,
            baseline_days_available=7,
            sample_dids=['did1', 'did2'],
            population_median_lambda=None,
            rolling_variance=None,
            dispersion_index=None,
            population_dispersion_index=None,
        )
        normal_row = AggregatedRow(
            pds_host='normal.com',
            observed_count=52,
            distinct_accounts=52,
            rolling_mean=50.0,
            baseline_days_available=7,
            sample_dids=['did3', 'did4'],
            population_median_lambda=None,
            rolling_variance=None,
            dispersion_index=None,
            population_dispersion_index=None,
        )

        fake_db.daily_rows = [anomaly_row, normal_row]
        fake_db.hourly_rows = []

        run_cycle(fake_db, app_config)

        # Verify both rows captured
        assert len(fake_db.captured_results) == 2

        # Verify all columns populated, no None values
        for result in fake_db.captured_results:
            assert result.run_timestamp is not None
            assert result.granularity is not None
            assert result.pds_host is not None
            assert result.observed_count is not None
            assert result.expected_lambda is not None
            assert result.p_value is not None
            assert result.is_anomaly is not None
            assert result.baseline_source is not None
            assert result.baseline_days_available is not None
            assert result.sample_dids is not None

        # Verify anomalous PDS has is_anomaly=1 and p_value < 0.01
        anomalous_result = next(r for r in fake_db.captured_results if r.pds_host == 'anomalous.com')
        assert anomalous_result.is_anomaly == 1
        assert anomalous_result.p_value < 0.01

    def test_ac5_1_daily_threshold_correctly_applied(
        self,
        app_config: AppConfig,
    ) -> None:
        """Verify daily anomaly detection uses configured daily p-value threshold (0.01)."""
        fake_db = FakeDb()

        # Create borderline case: p-value between 0.01 and 0.05
        # With observed=65, rolling_mean=50, p-value ≈ 0.0236
        borderline_row = AggregatedRow(
            pds_host='borderline.com',
            observed_count=65,
            distinct_accounts=65,
            rolling_mean=50.0,
            baseline_days_available=7,
            sample_dids=['did1'],
            population_median_lambda=None,
            rolling_variance=None,
            dispersion_index=None,
            population_dispersion_index=None,
        )

        fake_db.daily_rows = [borderline_row]
        fake_db.hourly_rows = []

        run_cycle(fake_db, app_config)

        # Find daily result (from daily_rows)
        daily_result = next(r for r in fake_db.captured_results if r.granularity == 'daily')

        # p-value should be between thresholds
        assert 0.01 < daily_result.p_value < 0.05

        # Should NOT be flagged as anomaly since p > 0.01 (daily threshold)
        assert daily_result.is_anomaly == 0

    def test_ac5_2_hourly_threshold_correctly_applied(
        self,
        app_config: AppConfig,
    ) -> None:
        """Verify hourly anomaly detection uses configured hourly p-value threshold (0.05)."""
        fake_db = FakeDb()

        # Same borderline case for hourly
        # With observed=65, rolling_mean=50, p-value ≈ 0.0236
        borderline_row = AggregatedRow(
            pds_host='borderline.com',
            observed_count=65,
            distinct_accounts=65,
            rolling_mean=50.0,
            baseline_days_available=7,
            sample_dids=['did1'],
            population_median_lambda=None,
            rolling_variance=None,
            dispersion_index=None,
            population_dispersion_index=None,
        )

        fake_db.daily_rows = []
        fake_db.hourly_rows = [borderline_row]

        run_cycle(fake_db, app_config)

        # Find hourly result
        hourly_result = next(r for r in fake_db.captured_results if r.granularity == 'hourly')

        # p-value should be between thresholds
        assert 0.01 < hourly_result.p_value < 0.05

        # SHOULD be flagged as anomaly since p < 0.05 (hourly threshold)
        assert hourly_result.is_anomaly == 1

    def test_ac5_3_granularity_enum_in_output(
        self,
        app_config: AppConfig,
    ) -> None:
        """Verify both daily and hourly results appear with correct granularity enum."""
        fake_db = FakeDb()

        daily_row = AggregatedRow(
            pds_host='daily.com',
            observed_count=50,
            distinct_accounts=50,
            rolling_mean=50.0,
            baseline_days_available=7,
            sample_dids=['did1'],
            population_median_lambda=None,
            rolling_variance=None,
            dispersion_index=None,
            population_dispersion_index=None,
        )
        hourly_row = AggregatedRow(
            pds_host='hourly.com',
            observed_count=50,
            distinct_accounts=50,
            rolling_mean=50.0,
            baseline_days_available=7,
            sample_dids=['did2'],
            population_median_lambda=None,
            rolling_variance=None,
            dispersion_index=None,
            population_dispersion_index=None,
        )

        fake_db.daily_rows = [daily_row]
        fake_db.hourly_rows = [hourly_row]

        run_cycle(fake_db, app_config)

        # Verify both granularities present
        daily_results = [r for r in fake_db.captured_results if r.granularity == 'daily']
        hourly_results = [r for r in fake_db.captured_results if r.granularity == 'hourly']

        assert len(daily_results) == 1
        assert len(hourly_results) == 1
        assert daily_results[0].pds_host == 'daily.com'
        assert hourly_results[0].pds_host == 'hourly.com'

    def test_empty_result_set_skips_writing(
        self,
        app_config: AppConfig,
    ) -> None:
        """Verify that empty result sets are skipped without writing."""
        fake_db = FakeDb()

        # Only daily rows, empty hourly
        daily_row = AggregatedRow(
            pds_host='daily.com',
            observed_count=50,
            distinct_accounts=50,
            rolling_mean=50.0,
            baseline_days_available=7,
            sample_dids=['did1'],
            population_median_lambda=None,
            rolling_variance=None,
            dispersion_index=None,
            population_dispersion_index=None,
        )

        fake_db.daily_rows = [daily_row]
        fake_db.hourly_rows = []

        run_cycle(fake_db, app_config)

        # Only daily results captured
        assert len(fake_db.captured_results) == 1
        assert fake_db.captured_results[0].granularity == 'daily'

    def test_all_results_written_both_normal_and_anomalous(
        self,
        app_config: AppConfig,
    ) -> None:
        """Verify all results (normal and anomalous) are written to output."""
        fake_db = FakeDb()

        anomalous_row = AggregatedRow(
            pds_host='anomalous.com',
            observed_count=200,
            distinct_accounts=200,
            rolling_mean=50.0,
            baseline_days_available=7,
            sample_dids=['did1'],
            population_median_lambda=None,
            rolling_variance=None,
            dispersion_index=None,
            population_dispersion_index=None,
        )
        normal_row = AggregatedRow(
            pds_host='normal.com',
            observed_count=52,
            distinct_accounts=52,
            rolling_mean=50.0,
            baseline_days_available=7,
            sample_dids=['did2'],
            population_median_lambda=None,
            rolling_variance=None,
            dispersion_index=None,
            population_dispersion_index=None,
        )

        fake_db.daily_rows = [anomalous_row, normal_row]
        fake_db.hourly_rows = []

        run_cycle(fake_db, app_config)

        # Both normal and anomalous should be captured
        assert len(fake_db.captured_results) == 2

        anomalous = next(r for r in fake_db.captured_results if r.pds_host == 'anomalous.com')
        normal = next(r for r in fake_db.captured_results if r.pds_host == 'normal.com')

        assert anomalous.is_anomaly == 1
        assert normal.is_anomaly == 0

    def test_run_cycle_uses_real_analyzer(
        self,
        app_config: AppConfig,
    ) -> None:
        """Verify run_cycle uses the real analyzer scoring logic."""
        fake_db = FakeDb()

        # Create test row
        test_row = AggregatedRow(
            pds_host='test.com',
            observed_count=200,
            distinct_accounts=200,
            rolling_mean=50.0,
            baseline_days_available=7,
            sample_dids=['did1'],
            population_median_lambda=None,
            rolling_variance=None,
            dispersion_index=None,
            population_dispersion_index=None,
        )

        fake_db.daily_rows = [test_row]
        fake_db.hourly_rows = []

        run_cycle(fake_db, app_config)

        result = fake_db.captured_results[0]

        # Verify scoring applied correctly
        assert result.expected_lambda == 50.0
        assert result.baseline_source == 'entity'
        assert result.is_anomaly == 1  # p_value < 0.01 for Poisson(50) observed 200

    def test_dispersion_values_flow_through_pipeline(
        self,
        app_config: AppConfig,
    ) -> None:
        """Verify dispersion values flow through pipeline: entity dispersion preferred over population."""
        fake_db = FakeDb()

        # Configure with entity dispersion values and sufficient history
        dispersion_row = AggregatedRow(
            pds_host='dispersion.com',
            observed_count=100,
            distinct_accounts=100,
            rolling_mean=50.0,
            baseline_days_available=7,
            sample_dids=['did1', 'did2'],
            population_median_lambda=None,
            rolling_variance=12.5,
            dispersion_index=2.5,
            population_dispersion_index=1.8,
        )

        fake_db.daily_rows = [dispersion_row]
        fake_db.hourly_rows = []

        run_cycle(fake_db, app_config)

        # Verify both rolling_variance and dispersion_index are captured
        assert len(fake_db.captured_results) == 1
        result = fake_db.captured_results[0]
        assert result.rolling_variance == 12.5
        assert result.dispersion_index == 2.5

    def test_cold_start_uses_population_dispersion(
        self,
        app_config: AppConfig,
    ) -> None:
        """Verify cold-start scenario: population dispersion used when entity has insufficient history."""
        fake_db = FakeDb()

        # Configure with insufficient history (below cold_start_min_days=3)
        cold_start_row = AggregatedRow(
            pds_host='coldstart.com',
            observed_count=100,
            distinct_accounts=100,
            rolling_mean=50.0,
            baseline_days_available=1,
            sample_dids=['did1'],
            population_median_lambda=None,
            rolling_variance=None,
            dispersion_index=None,
            population_dispersion_index=1.5,
        )

        fake_db.daily_rows = [cold_start_row]
        fake_db.hourly_rows = []

        run_cycle(fake_db, app_config)

        # Verify population dispersion is used as fallback
        assert len(fake_db.captured_results) == 1
        result = fake_db.captured_results[0]
        assert result.dispersion_index == 1.5

    def test_mean_guard_produces_none_dispersion(
        self,
        app_config: AppConfig,
    ) -> None:
        """Verify mean guard scenario: None dispersion when rolling_mean < 1.0 (SQL mean guard)."""
        fake_db = FakeDb()

        # Configure with sufficient history but low rolling_mean
        # This simulates the SQL mean guard scenario where dispersion_index is set to NULL
        mean_guard_row = AggregatedRow(
            pds_host='meanguard.com',
            observed_count=50,
            distinct_accounts=50,
            rolling_mean=0.3,
            baseline_days_available=7,
            sample_dids=['did1'],
            population_median_lambda=None,
            rolling_variance=0.3,
            dispersion_index=None,  # SQL mean guard set this to NULL
            population_dispersion_index=None,
        )

        fake_db.daily_rows = [mean_guard_row]
        fake_db.hourly_rows = []

        run_cycle(fake_db, app_config)

        # Verify rolling_variance is preserved, but dispersion_index is None
        assert len(fake_db.captured_results) == 1
        result = fake_db.captured_results[0]
        assert result.rolling_variance == 0.3
        assert result.dispersion_index is None

    def test_all_dispersion_none(
        self,
        app_config: AppConfig,
    ) -> None:
        """Verify scenario with all dispersion fields None."""
        fake_db = FakeDb()

        # Configure with all dispersion values as None
        no_dispersion_row = AggregatedRow(
            pds_host='nodispersion.com',
            observed_count=50,
            distinct_accounts=50,
            rolling_mean=50.0,
            baseline_days_available=7,
            sample_dids=['did1'],
            population_median_lambda=None,
            rolling_variance=None,
            dispersion_index=None,
            population_dispersion_index=None,
        )

        fake_db.daily_rows = [no_dispersion_row]
        fake_db.hourly_rows = []

        run_cycle(fake_db, app_config)

        # Verify both rolling_variance and dispersion_index are None
        assert len(fake_db.captured_results) == 1
        result = fake_db.captured_results[0]
        assert result.rolling_variance is None
        assert result.dispersion_index is None

    def test_ac5_1_existing_scoring_unaffected_by_dispersion_fields(
        self,
        app_config: AppConfig,
    ) -> None:
        """Verify existing scoring (is_anomaly, p_value, baseline) unchanged with dispersion fields."""
        fake_db = FakeDb()

        # Use same row setup as test_ac4_1 but with dispersion fields
        anomaly_row = AggregatedRow(
            pds_host='anomalous.com',
            observed_count=200,
            distinct_accounts=200,
            rolling_mean=50.0,
            baseline_days_available=7,
            sample_dids=['did1', 'did2'],
            population_median_lambda=None,
            rolling_variance=5.0,
            dispersion_index=2.2,
            population_dispersion_index=None,
        )
        normal_row = AggregatedRow(
            pds_host='normal.com',
            observed_count=52,
            distinct_accounts=52,
            rolling_mean=50.0,
            baseline_days_available=7,
            sample_dids=['did3', 'did4'],
            population_median_lambda=None,
            rolling_variance=1.0,
            dispersion_index=1.1,
            population_dispersion_index=None,
        )

        fake_db.daily_rows = [anomaly_row, normal_row]
        fake_db.hourly_rows = []

        run_cycle(fake_db, app_config)

        # Verify both results captured
        assert len(fake_db.captured_results) == 2

        # Verify anomalous result has unchanged scoring
        anomalous = next(r for r in fake_db.captured_results if r.pds_host == 'anomalous.com')
        assert anomalous.is_anomaly == 1
        assert anomalous.p_value < 0.01
        assert anomalous.expected_lambda == 50.0
        assert anomalous.baseline_source == 'entity'

        # Verify normal result has unchanged scoring
        normal = next(r for r in fake_db.captured_results if r.pds_host == 'normal.com')
        assert normal.is_anomaly == 0
        assert normal.p_value > 0.05
        assert normal.expected_lambda == 50.0
        assert normal.baseline_source == 'entity'
