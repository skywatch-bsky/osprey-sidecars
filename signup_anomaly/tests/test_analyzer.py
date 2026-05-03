# pattern: Functional Core
from __future__ import annotations

from datetime import datetime

import pytest

from signup_anomaly.analyzer import (
    compute_p_value,
    determine_baseline,
    determine_dispersion,
    score_row,
    score_rows,
)
from signup_anomaly.config import AnalysisConfig
from signup_anomaly.db import AggregatedRow, ScoredResult


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
def run_timestamp() -> datetime:
    return datetime(2024, 3, 20, 12, 0, 0)


class TestComputePValue:
    def test_returns_1_when_lambda_is_zero(self) -> None:
        result = compute_p_value(100, 0)
        assert result == 1.0

    def test_returns_1_when_lambda_is_negative(self) -> None:
        result = compute_p_value(100, -5.0)
        assert result == 1.0

    def test_returns_float_type(self) -> None:
        result = compute_p_value(10, 5.0)
        assert isinstance(result, float)

    def test_high_observed_count_with_low_lambda_gives_low_p_value(self) -> None:
        # Poisson(50) observed 200 should have very low p-value (high tail probability)
        result = compute_p_value(200, 50.0)
        assert result < 0.01

    def test_observed_count_near_mean_gives_high_p_value(self) -> None:
        # Poisson(50) observed 52 should have high p-value (near the bulk)
        result = compute_p_value(52, 50.0)
        assert result > 0.05

    def test_observed_count_zero_with_positive_lambda(self) -> None:
        # P(X >= 0) = 1.0 for any lambda > 0
        result = compute_p_value(0, 50.0)
        assert result == 1.0

    def test_observed_count_one_with_positive_lambda(self) -> None:
        # P(X >= 1) should be approximately 1.0 for high lambda
        result = compute_p_value(1, 50.0)
        assert result > 0.99


class TestDetermineDispersion:
    def test_uses_entity_dispersion_when_sufficient_history(self) -> None:
        row = AggregatedRow(
            pds_host='example.com',
            observed_count=100,
            distinct_accounts=95,
            rolling_mean=30.0,
            baseline_days_available=5,
            sample_dids=['did1', 'did2'],
            population_median_lambda=10.0,
            rolling_variance=0.5,
            dispersion_index=1.5,
            population_dispersion_index=1.2,
        )
        result = determine_dispersion(row, cold_start_min_days=3)
        assert result == 1.5

    def test_uses_population_dispersion_when_insufficient_entity_history(self) -> None:
        row = AggregatedRow(
            pds_host='example.com',
            observed_count=100,
            distinct_accounts=95,
            rolling_mean=30.0,
            baseline_days_available=2,
            sample_dids=['did1', 'did2'],
            population_median_lambda=10.0,
            rolling_variance=0.5,
            dispersion_index=1.5,
            population_dispersion_index=1.2,
        )
        result = determine_dispersion(row, cold_start_min_days=3)
        assert result == 1.2

    def test_uses_population_dispersion_when_entity_dispersion_is_none(self) -> None:
        row = AggregatedRow(
            pds_host='example.com',
            observed_count=100,
            distinct_accounts=95,
            rolling_mean=30.0,
            baseline_days_available=5,
            sample_dids=['did1', 'did2'],
            population_median_lambda=10.0,
            rolling_variance=0.5,
            dispersion_index=None,
            population_dispersion_index=1.2,
        )
        result = determine_dispersion(row, cold_start_min_days=3)
        assert result == 1.2

    def test_returns_none_when_both_dispersion_values_are_none(self) -> None:
        row = AggregatedRow(
            pds_host='example.com',
            observed_count=100,
            distinct_accounts=95,
            rolling_mean=30.0,
            baseline_days_available=5,
            sample_dids=['did1', 'did2'],
            population_median_lambda=10.0,
            rolling_variance=0.5,
            dispersion_index=None,
            population_dispersion_index=None,
        )
        result = determine_dispersion(row, cold_start_min_days=3)
        assert result is None

    def test_returns_none_when_entity_insufficient_and_population_none(self) -> None:
        row = AggregatedRow(
            pds_host='example.com',
            observed_count=100,
            distinct_accounts=95,
            rolling_mean=30.0,
            baseline_days_available=2,
            sample_dids=['did1', 'did2'],
            population_median_lambda=10.0,
            rolling_variance=0.5,
            dispersion_index=1.5,
            population_dispersion_index=None,
        )
        result = determine_dispersion(row, cold_start_min_days=3)
        assert result is None

    def test_edge_case_exactly_at_cold_start_threshold(self) -> None:
        row = AggregatedRow(
            pds_host='example.com',
            observed_count=100,
            distinct_accounts=95,
            rolling_mean=30.0,
            baseline_days_available=3,
            sample_dids=['did1', 'did2'],
            population_median_lambda=10.0,
            rolling_variance=0.5,
            dispersion_index=1.5,
            population_dispersion_index=1.2,
        )
        result = determine_dispersion(row, cold_start_min_days=3)
        assert result == 1.5


class TestDetermineBaseline:
    def test_uses_entity_baseline_when_sufficient_history(self) -> None:
        row = AggregatedRow(
            pds_host='example.com',
            observed_count=100,
            distinct_accounts=95,
            rolling_mean=30.0,
            baseline_days_available=5,
            sample_dids=['did1', 'did2'],
            population_median_lambda=10.0,
            rolling_variance=None,
            dispersion_index=None,
            population_dispersion_index=None,
        )
        lambda_val, source = determine_baseline(row, cold_start_min_days=3)
        assert lambda_val == 30.0
        assert source == 'entity'

    def test_uses_population_baseline_when_insufficient_entity_history(self) -> None:
        row = AggregatedRow(
            pds_host='example.com',
            observed_count=100,
            distinct_accounts=95,
            rolling_mean=30.0,
            baseline_days_available=2,
            sample_dids=['did1', 'did2'],
            population_median_lambda=10.0,
            rolling_variance=None,
            dispersion_index=None,
            population_dispersion_index=None,
        )
        lambda_val, source = determine_baseline(row, cold_start_min_days=3)
        assert lambda_val == 10.0
        assert source == 'population'

    def test_uses_population_baseline_when_rolling_mean_is_none(self) -> None:
        row = AggregatedRow(
            pds_host='example.com',
            observed_count=100,
            distinct_accounts=100,
            rolling_mean=None,
            baseline_days_available=5,
            sample_dids=['did1', 'did2'],
            population_median_lambda=10.0,
            rolling_variance=None,
            dispersion_index=None,
            population_dispersion_index=None,
        )
        lambda_val, source = determine_baseline(row, cold_start_min_days=3)
        assert lambda_val == 10.0
        assert source == 'population'

    def test_returns_zero_when_no_baselines_available(self) -> None:
        row = AggregatedRow(
            pds_host='example.com',
            observed_count=100,
            distinct_accounts=95,
            rolling_mean=None,
            baseline_days_available=2,
            sample_dids=['did1', 'did2'],
            population_median_lambda=None,
            rolling_variance=None,
            dispersion_index=None,
            population_dispersion_index=None,
        )
        lambda_val, source = determine_baseline(row, cold_start_min_days=3)
        assert lambda_val == 0.0
        assert source == 'population'

    def test_returns_zero_when_population_lambda_is_zero(self) -> None:
        row = AggregatedRow(
            pds_host='example.com',
            observed_count=100,
            distinct_accounts=100,
            rolling_mean=None,
            baseline_days_available=2,
            sample_dids=['did1', 'did2'],
            population_median_lambda=0.0,
            rolling_variance=None,
            dispersion_index=None,
            population_dispersion_index=None,
        )
        lambda_val, source = determine_baseline(row, cold_start_min_days=3)
        assert lambda_val == 0.0
        assert source == 'population'

    def test_edge_case_exactly_at_cold_start_threshold(self) -> None:
        row = AggregatedRow(
            pds_host='example.com',
            observed_count=100,
            distinct_accounts=95,
            rolling_mean=30.0,
            baseline_days_available=3,
            sample_dids=['did1', 'did2'],
            population_median_lambda=10.0,
            rolling_variance=None,
            dispersion_index=None,
            population_dispersion_index=None,
        )
        lambda_val, source = determine_baseline(row, cold_start_min_days=3)
        assert lambda_val == 30.0
        assert source == 'entity'


class TestScoreRow:
    # AC1.1: High observed count with low baseline produces low p-value and anomaly flag
    def test_ac1_1_high_observed_count_flagged_as_anomaly(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        row = AggregatedRow(
            pds_host='example.com',
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
        result = score_row(row, base_config, 'daily', run_timestamp)
        assert result.p_value < 0.01
        assert result.is_anomaly == 1

    # AC1.2: Observed count near mean produces high p-value and no anomaly
    def test_ac1_2_count_near_mean_not_flagged(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        row = AggregatedRow(
            pds_host='example.com',
            observed_count=52,
            distinct_accounts=52,
            rolling_mean=50.0,
            baseline_days_available=7,
            sample_dids=['did1', 'did2'],
            population_median_lambda=None,
            rolling_variance=None,
            dispersion_index=None,
            population_dispersion_index=None,
        )
        result = score_row(row, base_config, 'daily', run_timestamp)
        assert result.p_value > 0.05
        assert result.is_anomaly == 0

    # AC1.3: Lambda of 0 does not crash and produces p_value=1.0
    def test_ac1_3_lambda_zero_no_crash(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        row = AggregatedRow(
            pds_host='example.com',
            observed_count=100,
            distinct_accounts=95,
            rolling_mean=None,
            baseline_days_available=2,
            sample_dids=['did1', 'did2'],
            population_median_lambda=None,
            rolling_variance=None,
            dispersion_index=None,
            population_dispersion_index=None,
        )
        result = score_row(row, base_config, 'daily', run_timestamp)
        assert result.p_value == 1.0
        assert result.is_anomaly == 0

    # AC1.4: Observed count of 0 never flagged as anomalous
    def test_ac1_4_zero_count_never_anomalous(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        row = AggregatedRow(
            pds_host='example.com',
            observed_count=0,
            distinct_accounts=0,
            rolling_mean=50.0,
            baseline_days_available=7,
            sample_dids=['did1', 'did2'],
            population_median_lambda=None,
            rolling_variance=None,
            dispersion_index=None,
            population_dispersion_index=None,
        )
        result = score_row(row, base_config, 'daily', run_timestamp)
        assert result.is_anomaly == 0

    # AC2.1: Cold start uses population baseline
    def test_ac2_1_cold_start_uses_population_baseline(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        row = AggregatedRow(
            pds_host='example.com',
            observed_count=100,
            distinct_accounts=100,
            rolling_mean=None,
            baseline_days_available=2,
            sample_dids=['did1', 'did2'],
            population_median_lambda=10.0,
            rolling_variance=None,
            dispersion_index=None,
            population_dispersion_index=None,
        )
        result = score_row(row, base_config, 'daily', run_timestamp)
        assert result.baseline_source == 'population'
        assert result.expected_lambda == 10.0

    # AC2.2: Sufficient history uses entity baseline
    def test_ac2_2_sufficient_history_uses_entity_baseline(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        row = AggregatedRow(
            pds_host='example.com',
            observed_count=100,
            distinct_accounts=95,
            rolling_mean=30.0,
            baseline_days_available=5,
            sample_dids=['did1', 'did2'],
            population_median_lambda=10.0,
            rolling_variance=None,
            dispersion_index=None,
            population_dispersion_index=None,
        )
        result = score_row(row, base_config, 'daily', run_timestamp)
        assert result.baseline_source == 'entity'
        assert result.expected_lambda == 30.0

    def test_preserves_all_row_fields(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        row = AggregatedRow(
            pds_host='test.com',
            observed_count=50,
            distinct_accounts=50,
            rolling_mean=40.0,
            baseline_days_available=7,
            sample_dids=['did1', 'did2', 'did3'],
            population_median_lambda=20.0,
            rolling_variance=None,
            dispersion_index=None,
            population_dispersion_index=None,
        )
        result = score_row(row, base_config, 'daily', run_timestamp)
        assert result.pds_host == 'test.com'
        assert result.observed_count == 50
        assert result.baseline_days_available == 7
        assert result.sample_dids == ['did1', 'did2', 'did3']
        assert result.granularity == 'daily'
        assert result.run_timestamp == run_timestamp

    def test_cold_start_uses_daily_threshold_even_for_hourly_granularity(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        # Create a cold start scenario (insufficient entity history)
        # with borderline p-value between hourly and daily thresholds
        row = AggregatedRow(
            pds_host='example.com',
            observed_count=100,
            distinct_accounts=95,
            rolling_mean=None,
            baseline_days_available=2,
            sample_dids=['did1'],
            population_median_lambda=50.0,
            rolling_variance=None,
            dispersion_index=None,
            population_dispersion_index=None,
        )
        # With hourly threshold (0.05), this might be marginally different
        # But with population baseline, we use daily threshold (0.01)
        result = score_row(row, base_config, 'hourly', run_timestamp)
        assert result.baseline_source == 'population'
        # Verify daily threshold was applied
        p_val = compute_p_value(100, 50.0)
        expected_anomaly = 1 if (p_val < base_config.daily_p_value_threshold and 100 > 0) else 0
        assert result.is_anomaly == expected_anomaly

    def test_daily_granularity_uses_daily_threshold(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        row = AggregatedRow(
            pds_host='example.com',
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
        result = score_row(row, base_config, 'daily', run_timestamp)
        p_val = compute_p_value(200, 50.0)
        expected_anomaly = 1 if (p_val < base_config.daily_p_value_threshold and 200 > 0) else 0
        assert result.is_anomaly == expected_anomaly

    def test_hourly_granularity_uses_hourly_threshold_with_entity_baseline(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        row = AggregatedRow(
            pds_host='example.com',
            observed_count=100,
            distinct_accounts=100,
            rolling_mean=50.0,
            baseline_days_available=7,
            sample_dids=['did1'],
            population_median_lambda=None,
            rolling_variance=None,
            dispersion_index=None,
            population_dispersion_index=None,
        )
        result = score_row(row, base_config, 'hourly', run_timestamp)
        p_val = compute_p_value(100, 50.0)
        expected_anomaly = 1 if (p_val < base_config.hourly_p_value_threshold and 100 > 0) else 0
        assert result.is_anomaly == expected_anomaly

    def test_ac4_1_compute_p_value_unchanged_with_dispersion_fields(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        # Verify that compute_p_value is not affected by dispersion fields
        row_with_dispersion = AggregatedRow(
            pds_host='example.com',
            observed_count=100,
            distinct_accounts=100,
            rolling_mean=50.0,
            baseline_days_available=7,
            sample_dids=['did1'],
            population_median_lambda=None,
            rolling_variance=0.5,
            dispersion_index=1.5,
            population_dispersion_index=None,
        )
        result = score_row(row_with_dispersion, base_config, 'daily', run_timestamp)
        expected_p_val = compute_p_value(100, 50.0)
        assert result.p_value == expected_p_val

    def test_ac4_2_is_anomaly_unchanged_with_dispersion_fields(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        # Verify that is_anomaly determination is unaffected by dispersion values
        row_with_dispersion = AggregatedRow(
            pds_host='example.com',
            observed_count=200,
            distinct_accounts=200,
            rolling_mean=50.0,
            baseline_days_available=7,
            sample_dids=['did1'],
            population_median_lambda=None,
            rolling_variance=0.5,
            dispersion_index=1.5,
            population_dispersion_index=None,
        )
        result = score_row(row_with_dispersion, base_config, 'daily', run_timestamp)
        assert result.is_anomaly == 1  # Same logic: p_value < 0.01 for Poisson(50) observed 200

    def test_score_row_passes_through_rolling_variance(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        row = AggregatedRow(
            pds_host='example.com',
            observed_count=100,
            distinct_accounts=100,
            rolling_mean=50.0,
            baseline_days_available=7,
            sample_dids=['did1'],
            population_median_lambda=None,
            rolling_variance=0.75,
            dispersion_index=1.5,
            population_dispersion_index=None,
        )
        result = score_row(row, base_config, 'daily', run_timestamp)
        assert result.rolling_variance == 0.75

    def test_score_row_resolves_dispersion_index_entity_preference(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        row = AggregatedRow(
            pds_host='example.com',
            observed_count=100,
            distinct_accounts=100,
            rolling_mean=50.0,
            baseline_days_available=7,
            sample_dids=['did1'],
            population_median_lambda=None,
            rolling_variance=0.5,
            dispersion_index=1.5,
            population_dispersion_index=1.2,
        )
        result = score_row(row, base_config, 'daily', run_timestamp)
        assert result.dispersion_index == 1.5

    def test_score_row_resolves_dispersion_index_population_fallback(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        row = AggregatedRow(
            pds_host='example.com',
            observed_count=100,
            distinct_accounts=100,
            rolling_mean=50.0,
            baseline_days_available=2,
            sample_dids=['did1'],
            population_median_lambda=None,
            rolling_variance=0.5,
            dispersion_index=1.5,
            population_dispersion_index=1.2,
        )
        result = score_row(row, base_config, 'daily', run_timestamp)
        assert result.dispersion_index == 1.2


class TestScoreRows:
    def test_returns_correct_number_of_results(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        rows = [
            AggregatedRow(
                pds_host='host1.com',
                observed_count=100,
                distinct_accounts=100,
                rolling_mean=50.0,
                baseline_days_available=7,
                sample_dids=['did1'],
                population_median_lambda=None,
                rolling_variance=None,
                dispersion_index=None,
                population_dispersion_index=None,
            ),
            AggregatedRow(
                pds_host='host2.com',
                observed_count=50,
                distinct_accounts=50,
                rolling_mean=40.0,
                baseline_days_available=7,
                sample_dids=['did2'],
                population_median_lambda=None,
                rolling_variance=None,
                dispersion_index=None,
                population_dispersion_index=None,
            ),
            AggregatedRow(
                pds_host='host3.com',
                observed_count=200,
                distinct_accounts=200,
                rolling_mean=30.0,
                baseline_days_available=7,
                sample_dids=['did3'],
                population_median_lambda=None,
                rolling_variance=None,
                dispersion_index=None,
                population_dispersion_index=None,
            ),
        ]
        results = score_rows(rows, base_config, 'daily', run_timestamp)
        assert len(results) == 3

    def test_returns_all_scored_results(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        rows = [
            AggregatedRow(
                pds_host='host1.com',
                observed_count=100,
                distinct_accounts=100,
                rolling_mean=50.0,
                baseline_days_available=7,
                sample_dids=['did1'],
                population_median_lambda=None,
                rolling_variance=None,
                dispersion_index=None,
                population_dispersion_index=None,
            ),
            AggregatedRow(
                pds_host='host2.com',
                observed_count=50,
                distinct_accounts=50,
                rolling_mean=40.0,
                baseline_days_available=7,
                sample_dids=['did2'],
                population_median_lambda=None,
                rolling_variance=None,
                dispersion_index=None,
                population_dispersion_index=None,
            ),
        ]
        results = score_rows(rows, base_config, 'daily', run_timestamp)
        assert all(isinstance(r, ScoredResult) for r in results)
        assert results[0].pds_host == 'host1.com'
        assert results[1].pds_host == 'host2.com'

    def test_empty_list_returns_empty_results(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        results = score_rows([], base_config, 'daily', run_timestamp)
        assert results == []

    def test_score_rows_respects_granularity(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        rows = [
            AggregatedRow(
                pds_host='host1.com',
                observed_count=100,
                distinct_accounts=100,
                rolling_mean=50.0,
                baseline_days_available=7,
                sample_dids=['did1'],
                population_median_lambda=None,
                rolling_variance=None,
                dispersion_index=None,
                population_dispersion_index=None,
            ),
        ]
        daily_results = score_rows(rows, base_config, 'daily', run_timestamp)
        hourly_results = score_rows(rows, base_config, 'hourly', run_timestamp)
        assert daily_results[0].granularity == 'daily'
        assert hourly_results[0].granularity == 'hourly'
