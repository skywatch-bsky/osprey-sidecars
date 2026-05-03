# pattern: Functional Core
from __future__ import annotations

from datetime import datetime

import pytest

from url_overdispersion.analyzer import (
    compute_density_p_value,
    compute_p_value,
    determine_baseline,
    score_row,
    score_rows,
)
from url_overdispersion.config import AnalysisConfig
from url_overdispersion.db import AggregatedRow, ScoredResult


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
def run_timestamp() -> datetime:
    return datetime(2024, 3, 20, 12, 0, 0)


class TestComputePValue:
    def test_ac1_7_zero_lambda_returns_one(self) -> None:
        result = compute_p_value(100, 0)
        assert result == 1.0

    def test_negative_lambda_returns_one(self) -> None:
        result = compute_p_value(100, -1)
        assert result == 1.0

    def test_high_observed_low_lambda_gives_low_pvalue(self) -> None:
        result = compute_p_value(50, 5)
        assert result < 0.001

    def test_near_mean_gives_high_pvalue(self) -> None:
        result = compute_p_value(5, 5)
        assert result > 0.3

    def test_zero_observed_gives_one(self) -> None:
        result = compute_p_value(0, 10)
        assert result > 0.99


class TestComputeDensityPValue:
    def test_zero_expected_returns_one(self) -> None:
        result = compute_density_p_value(0.9, 0, 100)
        assert result == 1.0

    def test_insufficient_observations_returns_one(self) -> None:
        result = compute_density_p_value(0.9, 0.5, 1)
        assert result == 1.0

    def test_high_density_vs_low_expected_gives_low_pvalue(self) -> None:
        result = compute_density_p_value(0.9, 0.1, 100)
        assert result < 0.01

    def test_density_below_expected_returns_one(self) -> None:
        result = compute_density_p_value(0.1, 0.5, 100)
        assert result == 1.0


class TestDetermineBaseline:
    def test_ac1_3_entity_baseline_used_when_sufficient(self) -> None:
        row = AggregatedRow(
            domain='example.com',
            bucket_start=datetime(2024, 3, 20),
            total_shares=100,
            unique_sharers=20,
            sharer_density=0.2,
            rolling_volume_mean=30.0,
            rolling_density_mean=0.25,
            baseline_days_available=5,
            sample_dids=['did1', 'did2'],
            sample_urls=['url1', 'url2'],
            population_volume_median=10.0,
            population_density_median=0.1,
        )
        volume_lambda, density_lambda, source = determine_baseline(row, cold_start_min_days=3)
        assert volume_lambda == 30.0
        assert density_lambda == 0.25
        assert source == 'entity'

    def test_ac1_4_population_fallback_when_insufficient(self) -> None:
        row = AggregatedRow(
            domain='example.com',
            bucket_start=datetime(2024, 3, 20),
            total_shares=100,
            unique_sharers=20,
            sharer_density=0.2,
            rolling_volume_mean=30.0,
            rolling_density_mean=0.25,
            baseline_days_available=2,
            sample_dids=['did1', 'did2'],
            sample_urls=['url1', 'url2'],
            population_volume_median=10.0,
            population_density_median=0.15,
        )
        volume_lambda, density_lambda, source = determine_baseline(row, cold_start_min_days=3)
        assert volume_lambda == 10.0
        assert density_lambda == 0.15
        assert source == 'population'

    def test_ac1_4_population_fallback_when_rolling_none(self) -> None:
        row = AggregatedRow(
            domain='example.com',
            bucket_start=datetime(2024, 3, 20),
            total_shares=100,
            unique_sharers=20,
            sharer_density=0.2,
            rolling_volume_mean=None,
            rolling_density_mean=0.25,
            baseline_days_available=5,
            sample_dids=['did1', 'did2'],
            sample_urls=['url1', 'url2'],
            population_volume_median=10.0,
            population_density_median=0.15,
        )
        volume_lambda, density_lambda, source = determine_baseline(row, cold_start_min_days=3)
        assert volume_lambda == 10.0
        assert density_lambda == 0.15
        assert source == 'population'

    def test_ac1_7_zero_fallback_when_no_data(self) -> None:
        row = AggregatedRow(
            domain='example.com',
            bucket_start=datetime(2024, 3, 20),
            total_shares=100,
            unique_sharers=20,
            sharer_density=0.2,
            rolling_volume_mean=None,
            rolling_density_mean=None,
            baseline_days_available=2,
            sample_dids=['did1', 'did2'],
            sample_urls=['url1', 'url2'],
            population_volume_median=None,
            population_density_median=None,
        )
        volume_lambda, density_lambda, source = determine_baseline(row, cold_start_min_days=3)
        assert volume_lambda == 0.0
        assert density_lambda == 0.0
        assert source == 'population'

    def test_exactly_at_threshold(self) -> None:
        row = AggregatedRow(
            domain='example.com',
            bucket_start=datetime(2024, 3, 20),
            total_shares=100,
            unique_sharers=20,
            sharer_density=0.2,
            rolling_volume_mean=30.0,
            rolling_density_mean=0.25,
            baseline_days_available=3,
            sample_dids=['did1', 'did2'],
            sample_urls=['url1', 'url2'],
            population_volume_median=10.0,
            population_density_median=0.15,
        )
        volume_lambda, density_lambda, source = determine_baseline(row, cold_start_min_days=3)
        assert volume_lambda == 30.0
        assert density_lambda == 0.25
        assert source == 'entity'


class TestScoreRow:
    def test_volume_anomaly_sets_flag(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        row = AggregatedRow(
            domain='example.com',
            bucket_start=datetime(2024, 3, 20),
            total_shares=50,
            unique_sharers=20,
            sharer_density=0.4,
            rolling_volume_mean=5.0,
            rolling_density_mean=0.1,
            baseline_days_available=7,
            sample_dids=['did1', 'did2'],
            sample_urls=['url1', 'url2'],
            population_volume_median=None,
            population_density_median=None,
        )
        result = score_row(row, base_config, 'daily', run_timestamp, on_watchlist=0)
        assert result.volume_p_value < base_config.volume_p_threshold
        assert result.is_anomaly == 1

    def test_density_anomaly_sets_flag(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        row = AggregatedRow(
            domain='example.com',
            bucket_start=datetime(2024, 3, 20),
            total_shares=10,
            unique_sharers=10,
            sharer_density=0.9,
            rolling_volume_mean=15.0,
            rolling_density_mean=0.1,
            baseline_days_available=7,
            sample_dids=['did1', 'did2'],
            sample_urls=['url1', 'url2'],
            population_volume_median=None,
            population_density_median=None,
        )
        result = score_row(row, base_config, 'daily', run_timestamp, on_watchlist=0)
        assert result.density_p_value < base_config.density_p_threshold
        assert result.is_anomaly == 1

    def test_both_normal_no_flag(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        row = AggregatedRow(
            domain='example.com',
            bucket_start=datetime(2024, 3, 20),
            total_shares=10,
            unique_sharers=10,
            sharer_density=0.1,
            rolling_volume_mean=10.0,
            rolling_density_mean=0.1,
            baseline_days_available=7,
            sample_dids=['did1', 'did2'],
            sample_urls=['url1', 'url2'],
            population_volume_median=None,
            population_density_median=None,
        )
        result = score_row(row, base_config, 'daily', run_timestamp, on_watchlist=0)
        assert result.volume_p_value >= base_config.volume_p_threshold
        assert result.density_p_value >= base_config.density_p_threshold
        assert result.is_anomaly == 0

    def test_ac1_7_zero_baseline_no_anomaly(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        row = AggregatedRow(
            domain='example.com',
            bucket_start=datetime(2024, 3, 20),
            total_shares=100,
            unique_sharers=20,
            sharer_density=0.9,
            rolling_volume_mean=None,
            rolling_density_mean=None,
            baseline_days_available=2,
            sample_dids=['did1', 'did2'],
            sample_urls=['url1', 'url2'],
            population_volume_median=None,
            population_density_median=None,
        )
        result = score_row(row, base_config, 'daily', run_timestamp, on_watchlist=0)
        assert result.volume_p_value == 1.0
        assert result.density_p_value == 1.0
        assert result.is_anomaly == 0

    def test_watchlist_enrichment(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        row = AggregatedRow(
            domain='example.com',
            bucket_start=datetime(2024, 3, 20),
            total_shares=10,
            unique_sharers=10,
            sharer_density=0.1,
            rolling_volume_mean=10.0,
            rolling_density_mean=0.1,
            baseline_days_available=7,
            sample_dids=['did1'],
            sample_urls=['url1'],
            population_volume_median=None,
            population_density_median=None,
        )
        result = score_row(row, base_config, 'daily', run_timestamp, on_watchlist=1)
        assert result.on_watchlist == 1


class TestScoreRows:
    def test_maps_all_rows(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        rows = [
            AggregatedRow(
                domain='example1.com',
                bucket_start=datetime(2024, 3, 20),
                total_shares=10,
                unique_sharers=10,
                sharer_density=0.1,
                rolling_volume_mean=10.0,
                rolling_density_mean=0.1,
                baseline_days_available=7,
                sample_dids=['did1'],
                sample_urls=['url1'],
                population_volume_median=None,
                population_density_median=None,
            ),
            AggregatedRow(
                domain='example2.com',
                bucket_start=datetime(2024, 3, 20),
                total_shares=20,
                unique_sharers=15,
                sharer_density=0.15,
                rolling_volume_mean=20.0,
                rolling_density_mean=0.15,
                baseline_days_available=7,
                sample_dids=['did2'],
                sample_urls=['url2'],
                population_volume_median=None,
                population_density_median=None,
            ),
        ]
        results = score_rows(rows, base_config, 'daily', run_timestamp, ())
        assert len(results) == 2
        assert all(isinstance(r, ScoredResult) for r in results)

    def test_watchlist_matching(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        rows = [
            AggregatedRow(
                domain='watchlisted.com',
                bucket_start=datetime(2024, 3, 20),
                total_shares=10,
                unique_sharers=10,
                sharer_density=0.1,
                rolling_volume_mean=10.0,
                rolling_density_mean=0.1,
                baseline_days_available=7,
                sample_dids=['did1'],
                sample_urls=['url1'],
                population_volume_median=None,
                population_density_median=None,
            ),
            AggregatedRow(
                domain='normal.com',
                bucket_start=datetime(2024, 3, 20),
                total_shares=10,
                unique_sharers=10,
                sharer_density=0.1,
                rolling_volume_mean=10.0,
                rolling_density_mean=0.1,
                baseline_days_available=7,
                sample_dids=['did2'],
                sample_urls=['url2'],
                population_volume_median=None,
                population_density_median=None,
            ),
        ]
        watchlist = ('watchlisted.com',)
        results = score_rows(rows, base_config, 'daily', run_timestamp, watchlist)
        assert results[0].on_watchlist == 1
        assert results[1].on_watchlist == 0
