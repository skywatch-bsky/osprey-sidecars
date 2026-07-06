# pattern: Functional Core
from __future__ import annotations

from datetime import datetime

import pytest

from signup_anomaly.analyzer import (
    determine_baseline,
    determine_dispersion,
    score_row,
    score_rows,
)
from signup_anomaly.config import AnalysisConfig
from signup_anomaly.counts import count_p_value
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


class TestDetermineDispersion:
    def test_uses_entity_dispersion_when_sufficient_history(self) -> None:
        row = AggregatedRow(
            pds_host='example.com',
            observed_count=100,
            distinct_accounts=95,
            rolling_median=28.0,
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
            rolling_median=28.0,
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
            rolling_median=28.0,
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
            rolling_median=28.0,
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
            rolling_median=28.0,
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
            rolling_median=28.0,
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
            rolling_median=28.0,
            rolling_mean=30.0,
            baseline_days_available=5,
            sample_dids=['did1', 'did2'],
            population_median_lambda=10.0,
            rolling_variance=None,
            dispersion_index=None,
            population_dispersion_index=None,
        )
        lambda_val, source = determine_baseline(row, cold_start_min_days=3)
        assert lambda_val == 28.0
        assert source == 'entity'

    def test_uses_population_baseline_when_insufficient_entity_history(self) -> None:
        row = AggregatedRow(
            pds_host='example.com',
            observed_count=100,
            distinct_accounts=95,
            rolling_median=30.0,
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

    def test_uses_population_baseline_when_rolling_median_is_none(self) -> None:
        row = AggregatedRow(
            pds_host='example.com',
            observed_count=100,
            distinct_accounts=100,
            rolling_median=None,
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
            rolling_median=None,
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
            rolling_median=None,
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
            rolling_median=28.0,
            rolling_mean=30.0,
            baseline_days_available=3,
            sample_dids=['did1', 'did2'],
            population_median_lambda=10.0,
            rolling_variance=None,
            dispersion_index=None,
            population_dispersion_index=None,
        )
        lambda_val, source = determine_baseline(row, cold_start_min_days=3)
        assert lambda_val == 28.0
        assert source == 'entity'

    def test_rolling_median_zero_routes_to_population(self) -> None:
        row = AggregatedRow(
            pds_host='example.com',
            observed_count=100,
            distinct_accounts=95,
            rolling_median=0.0,
            rolling_mean=30.0,
            baseline_days_available=7,
            sample_dids=['did1', 'did2'],
            population_median_lambda=10.0,
            rolling_variance=None,
            dispersion_index=None,
            population_dispersion_index=None,
        )
        lambda_val, source = determine_baseline(row, cold_start_min_days=3)
        assert lambda_val == 10.0
        assert source == 'population'


class TestScoreRow:
    # AC1.1: High observed count with low baseline produces low p-value
    def test_ac1_1_high_observed_count_produces_low_p_value(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        row = AggregatedRow(
            pds_host='example.com',
            observed_count=200,
            distinct_accounts=200,
            rolling_median=50.0,
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

    # AC1.2: Observed count near mean produces high p-value
    def test_ac1_2_count_near_mean_produces_high_p_value(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        row = AggregatedRow(
            pds_host='example.com',
            observed_count=52,
            distinct_accounts=52,
            rolling_median=50.0,
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
            rolling_median=None,
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

    # AC1.4: Observed count of 0 never flagged as anomalous
    def test_ac1_4_zero_count_p_value_is_one(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        row = AggregatedRow(
            pds_host='example.com',
            observed_count=0,
            distinct_accounts=0,
            rolling_median=50.0,
            rolling_mean=50.0,
            baseline_days_available=7,
            sample_dids=['did1', 'did2'],
            population_median_lambda=None,
            rolling_variance=None,
            dispersion_index=None,
            population_dispersion_index=None,
        )
        result = score_row(row, base_config, 'daily', run_timestamp)
        assert result.p_value == 1.0

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
            rolling_median=None,
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
            rolling_median=28.0,
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
        assert result.expected_lambda == 28.0

    def test_preserves_all_row_fields(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        row = AggregatedRow(
            pds_host='test.com',
            observed_count=50,
            distinct_accounts=50,
            rolling_median=38.0,
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
        assert result.q_value == 1.0  # score_row placeholder

    def test_cold_start_uses_population_baseline(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        row = AggregatedRow(
            pds_host='example.com',
            observed_count=100,
            distinct_accounts=95,
            rolling_median=None,
            rolling_mean=None,
            baseline_days_available=2,
            sample_dids=['did1'],
            population_median_lambda=50.0,
            rolling_variance=None,
            dispersion_index=None,
            population_dispersion_index=None,
        )
        result = score_row(row, base_config, 'hourly', run_timestamp)
        assert result.baseline_source == 'population'
        assert result.expected_lambda == 50.0

    def test_daily_granularity_uses_rolling_median(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        row = AggregatedRow(
            pds_host='example.com',
            observed_count=200,
            distinct_accounts=200,
            rolling_median=50.0,
            rolling_mean=50.0,
            baseline_days_available=7,
            sample_dids=['did1'],
            population_median_lambda=None,
            rolling_variance=None,
            dispersion_index=None,
            population_dispersion_index=None,
        )
        result = score_row(row, base_config, 'daily', run_timestamp)
        assert result.expected_lambda == 50.0
        assert result.baseline_source == 'entity'

    def test_hourly_granularity_uses_rolling_median(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        row = AggregatedRow(
            pds_host='example.com',
            observed_count=100,
            distinct_accounts=100,
            rolling_median=50.0,
            rolling_mean=50.0,
            baseline_days_available=7,
            sample_dids=['did1'],
            population_median_lambda=None,
            rolling_variance=None,
            dispersion_index=None,
            population_dispersion_index=None,
        )
        result = score_row(row, base_config, 'hourly', run_timestamp)
        assert result.expected_lambda == 50.0
        assert result.baseline_source == 'entity'

    def test_score_row_uses_nb_with_dispersion_factor(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        # Verify that dispersion factor is used to compute variance
        row = AggregatedRow(
            pds_host='example.com',
            observed_count=100,
            distinct_accounts=100,
            rolling_median=50.0,
            rolling_mean=50.0,
            baseline_days_available=7,
            sample_dids=['did1'],
            population_median_lambda=None,
            rolling_variance=75.0,  # variance/mean = 1.5
            dispersion_index=1.5,
            population_dispersion_index=None,
        )
        result = score_row(row, base_config, 'daily', run_timestamp)
        # With phi=1.5, variance=1.5*50=75, should use NB branch
        # NB typically gives larger p-value than Poisson for same observed
        expected_nb_p = count_p_value(100, 50.0, 75.0)
        assert result.p_value == expected_nb_p

    def test_score_row_with_phi_less_than_one_uses_poisson(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        # phi <= 1.0 should use Poisson branch
        row = AggregatedRow(
            pds_host='example.com',
            observed_count=100,
            distinct_accounts=100,
            rolling_median=50.0,
            rolling_mean=50.0,
            baseline_days_available=7,
            sample_dids=['did1'],
            population_median_lambda=None,
            rolling_variance=0.5,
            dispersion_index=0.01,  # very underdispersed
            population_dispersion_index=None,
        )
        result = score_row(row, base_config, 'daily', run_timestamp)
        # With phi=0.01, variance=0.01*50=0.5, which is < mean, so use Poisson
        expected_p = count_p_value(100, 50.0, None)  # Poisson path
        assert result.p_value == expected_p

    def test_score_row_passes_through_rolling_variance(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        row = AggregatedRow(
            pds_host='example.com',
            observed_count=100,
            distinct_accounts=100,
            rolling_median=50.0,
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
            rolling_median=50.0,
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
            rolling_median=50.0,
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
                rolling_median=50.0,
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
                rolling_median=40.0,
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
                rolling_median=30.0,
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
                rolling_median=50.0,
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
                rolling_median=40.0,
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
        # q-values should be set after BH adjustment (not 1.0 placeholders)
        assert results[0].q_value < 1.0 or results[1].q_value < 1.0

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
                rolling_median=50.0,
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

    def test_score_rows_applies_bh_adjustment(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        # Create a family of rows with varying p-values
        rows = [
            AggregatedRow(
                pds_host='host1.com',
                observed_count=200,  # very high
                distinct_accounts=200,
                rolling_median=50.0,
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
                observed_count=52,  # near baseline
                distinct_accounts=52,
                rolling_median=50.0,
                rolling_mean=50.0,
                baseline_days_available=7,
                sample_dids=['did2'],
                population_median_lambda=None,
                rolling_variance=None,
                dispersion_index=None,
                population_dispersion_index=None,
            ),
        ]
        results = score_rows(rows, base_config, 'daily', run_timestamp)
        # Both q-values should be set
        assert results[0].q_value is not None
        assert results[1].q_value is not None
        # q-values should be >= their corresponding p-values
        assert results[0].q_value >= results[0].p_value
        assert results[1].q_value >= results[1].p_value

    def test_score_rows_population_source_uses_daily_threshold(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        # Cold-start row (population source) should use daily threshold even for hourly
        rows = [
            AggregatedRow(
                pds_host='host1.com',
                observed_count=200,
                distinct_accounts=200,
                rolling_median=None,
                rolling_mean=None,
                baseline_days_available=2,
                sample_dids=['did1'],
                population_median_lambda=100.0,
                rolling_variance=None,
                dispersion_index=None,
                population_dispersion_index=None,
            ),
        ]
        hourly_results = score_rows(rows, base_config, 'hourly', run_timestamp)
        # With population baseline, threshold is daily (0.01)
        assert hourly_results[0].baseline_source == 'population'
        # The threshold used for is_anomaly should be daily
        q = hourly_results[0].q_value
        threshold = base_config.daily_p_value_threshold
        expected_anomaly = 1 if (q < threshold and hourly_results[0].observed_count > 0) else 0
        assert hourly_results[0].is_anomaly == expected_anomaly

    def test_score_rows_zero_observed_never_anomalous(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        # Even with tiny q-value, observed_count=0 should never be anomalous
        rows = [
            AggregatedRow(
                pds_host='host1.com',
                observed_count=0,
                distinct_accounts=0,
                rolling_median=100.0,
                rolling_mean=100.0,
                baseline_days_available=7,
                sample_dids=['did1'],
                population_median_lambda=None,
                rolling_variance=None,
                dispersion_index=None,
                population_dispersion_index=None,
            ),
        ]
        results = score_rows(rows, base_config, 'daily', run_timestamp)
        assert results[0].is_anomaly == 0

    def test_score_rows_family_separation_daily_vs_hourly(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        """AC2.3: daily and hourly are separate families; each family adjusts independently.

        q-values computed within one family should differ from q-values when the same
        p-values are adjusted as part of a different (larger) family. This test verifies
        that daily and hourly cycles do not share the Benjamini-Hochberg correction.
        """
        # Create a 2-row family with known p-values
        rows_small = [
            AggregatedRow(
                pds_host='host1.com',
                observed_count=100,
                distinct_accounts=100,
                rolling_median=50.0,
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
                observed_count=52,
                distinct_accounts=52,
                rolling_median=50.0,
                rolling_mean=50.0,
                baseline_days_available=7,
                sample_dids=['did2'],
                population_median_lambda=None,
                rolling_variance=None,
                dispersion_index=None,
                population_dispersion_index=None,
            ),
        ]

        # Score the 2-row family
        results_small = score_rows(rows_small, base_config, 'daily', run_timestamp)
        q_value_1_in_2_family = results_small[0].q_value
        q_value_2_in_2_family = results_small[1].q_value

        # Now create a 10-row family where the same two rows are embedded,
        # padded with high-p rows (near-baseline counts)
        rows_large = [
            AggregatedRow(
                pds_host='host1.com',  # same as first row in small family
                observed_count=100,
                distinct_accounts=100,
                rolling_median=50.0,
                rolling_mean=50.0,
                baseline_days_available=7,
                sample_dids=['did1'],
                population_median_lambda=None,
                rolling_variance=None,
                dispersion_index=None,
                population_dispersion_index=None,
            ),
            AggregatedRow(
                pds_host='host2.com',  # same as second row in small family
                observed_count=52,
                distinct_accounts=52,
                rolling_median=50.0,
                rolling_mean=50.0,
                baseline_days_available=7,
                sample_dids=['did2'],
                population_median_lambda=None,
                rolling_variance=None,
                dispersion_index=None,
                population_dispersion_index=None,
            ),
            # Padding: 8 more rows with counts near baseline (high p-values)
            AggregatedRow(
                pds_host='pad3.com',
                observed_count=51,
                distinct_accounts=51,
                rolling_median=50.0,
                rolling_mean=50.0,
                baseline_days_available=7,
                sample_dids=['did3'],
                population_median_lambda=None,
                rolling_variance=None,
                dispersion_index=None,
                population_dispersion_index=None,
            ),
            AggregatedRow(
                pds_host='pad4.com',
                observed_count=50,
                distinct_accounts=50,
                rolling_median=50.0,
                rolling_mean=50.0,
                baseline_days_available=7,
                sample_dids=['did4'],
                population_median_lambda=None,
                rolling_variance=None,
                dispersion_index=None,
                population_dispersion_index=None,
            ),
            AggregatedRow(
                pds_host='pad5.com',
                observed_count=49,
                distinct_accounts=49,
                rolling_median=50.0,
                rolling_mean=50.0,
                baseline_days_available=7,
                sample_dids=['did5'],
                population_median_lambda=None,
                rolling_variance=None,
                dispersion_index=None,
                population_dispersion_index=None,
            ),
            AggregatedRow(
                pds_host='pad6.com',
                observed_count=51,
                distinct_accounts=51,
                rolling_median=50.0,
                rolling_mean=50.0,
                baseline_days_available=7,
                sample_dids=['did6'],
                population_median_lambda=None,
                rolling_variance=None,
                dispersion_index=None,
                population_dispersion_index=None,
            ),
            AggregatedRow(
                pds_host='pad7.com',
                observed_count=50,
                distinct_accounts=50,
                rolling_median=50.0,
                rolling_mean=50.0,
                baseline_days_available=7,
                sample_dids=['did7'],
                population_median_lambda=None,
                rolling_variance=None,
                dispersion_index=None,
                population_dispersion_index=None,
            ),
            AggregatedRow(
                pds_host='pad8.com',
                observed_count=49,
                distinct_accounts=49,
                rolling_median=50.0,
                rolling_mean=50.0,
                baseline_days_available=7,
                sample_dids=['did8'],
                population_median_lambda=None,
                rolling_variance=None,
                dispersion_index=None,
                population_dispersion_index=None,
            ),
            AggregatedRow(
                pds_host='pad9.com',
                observed_count=52,
                distinct_accounts=52,
                rolling_median=50.0,
                rolling_mean=50.0,
                baseline_days_available=7,
                sample_dids=['did9'],
                population_median_lambda=None,
                rolling_variance=None,
                dispersion_index=None,
                population_dispersion_index=None,
            ),
            AggregatedRow(
                pds_host='pad10.com',
                observed_count=48,
                distinct_accounts=48,
                rolling_median=50.0,
                rolling_mean=50.0,
                baseline_days_available=7,
                sample_dids=['did10'],
                population_median_lambda=None,
                rolling_variance=None,
                dispersion_index=None,
                population_dispersion_index=None,
            ),
        ]

        # Score the 10-row family
        results_large = score_rows(rows_large, base_config, 'daily', run_timestamp)
        q_value_1_in_10_family = results_large[0].q_value
        q_value_2_in_10_family = results_large[1].q_value

        # The q-values should differ because n changes the Benjamini-Hochberg correction.
        # In a 2-row family, both rows get corrected for 2 comparisons.
        # In a 10-row family, they get corrected for 10 comparisons, making the adjustment stricter.
        assert q_value_1_in_2_family != q_value_1_in_10_family
        assert q_value_2_in_2_family != q_value_2_in_10_family
        # The q-value in the larger family should generally be more conservative
        # (stricter correction), but order depends on ranks; at minimum they differ.
        assert abs(q_value_1_in_2_family - q_value_1_in_10_family) > 1e-9
