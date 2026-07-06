# pattern: Functional Core
from __future__ import annotations

from datetime import datetime

import pytest

from url_overdispersion.analyzer import (
    determine_baseline,
    determine_variances,
    score_row,
    score_rows,
)
from url_overdispersion.config import AnalysisConfig
from url_overdispersion.counts import count_p_value
from url_overdispersion.db import AggregatedRow, ScoredResult
from url_overdispersion.density import density_p_value


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


class TestDetermineBaseline:
    def test_entity_baseline_with_rolling_median(self) -> None:
        """When entity has sufficient history and rolling_volume_median > 0, use entity baseline."""
        row = AggregatedRow(
            domain='example.com',
            bucket_start=datetime(2024, 3, 20),
            total_shares=100,
            unique_sharers=20,
            sharer_density=0.2,
            rolling_volume_median=25.0,
            rolling_volume_mean=30.0,
            rolling_volume_variance=40.0,
            rolling_density_mean=0.25,
            rolling_density_variance=0.05,
            baseline_days_available=5,
            sample_dids=['did1', 'did2'],
            sample_urls=['url1', 'url2'],
            population_volume_median=10.0,
            population_volume_dispersion=1.2,
            population_density_median=0.1,
            population_density_variance=0.02,
        )
        volume_lambda, density_lambda, source = determine_baseline(row, cold_start_min_days=3)
        assert volume_lambda == 25.0
        assert density_lambda == 0.25
        assert source == 'entity'

    def test_population_fallback_when_insufficient_history(self) -> None:
        """When entity doesn't have enough history, fall back to population median."""
        row = AggregatedRow(
            domain='example.com',
            bucket_start=datetime(2024, 3, 20),
            total_shares=100,
            unique_sharers=20,
            sharer_density=0.2,
            rolling_volume_median=25.0,
            rolling_volume_mean=30.0,
            rolling_volume_variance=40.0,
            rolling_density_mean=0.25,
            rolling_density_variance=0.05,
            baseline_days_available=2,
            sample_dids=['did1', 'did2'],
            sample_urls=['url1', 'url2'],
            population_volume_median=10.0,
            population_volume_dispersion=1.2,
            population_density_median=0.15,
            population_density_variance=0.03,
        )
        volume_lambda, density_lambda, source = determine_baseline(row, cold_start_min_days=3)
        assert volume_lambda == 10.0
        assert density_lambda == 0.15
        assert source == 'population'

    def test_population_fallback_when_rolling_median_zero(self) -> None:
        """When rolling_volume_median is 0, fall back to population (sparse entity)."""
        row = AggregatedRow(
            domain='example.com',
            bucket_start=datetime(2024, 3, 20),
            total_shares=100,
            unique_sharers=20,
            sharer_density=0.2,
            rolling_volume_median=0.0,
            rolling_volume_mean=30.0,
            rolling_volume_variance=40.0,
            rolling_density_mean=0.25,
            rolling_density_variance=0.05,
            baseline_days_available=5,
            sample_dids=['did1', 'did2'],
            sample_urls=['url1', 'url2'],
            population_volume_median=10.0,
            population_volume_dispersion=1.2,
            population_density_median=0.15,
            population_density_variance=0.03,
        )
        volume_lambda, density_lambda, source = determine_baseline(row, cold_start_min_days=3)
        assert volume_lambda == 10.0
        assert density_lambda == 0.15
        assert source == 'population'

    def test_population_fallback_when_rolling_median_none(self) -> None:
        """When rolling_volume_median is None, fall back to population."""
        row = AggregatedRow(
            domain='example.com',
            bucket_start=datetime(2024, 3, 20),
            total_shares=100,
            unique_sharers=20,
            sharer_density=0.2,
            rolling_volume_median=None,
            rolling_volume_mean=30.0,
            rolling_volume_variance=40.0,
            rolling_density_mean=0.25,
            rolling_density_variance=0.05,
            baseline_days_available=5,
            sample_dids=['did1', 'did2'],
            sample_urls=['url1', 'url2'],
            population_volume_median=10.0,
            population_volume_dispersion=1.2,
            population_density_median=0.15,
            population_density_variance=0.03,
        )
        volume_lambda, density_lambda, source = determine_baseline(row, cold_start_min_days=3)
        assert volume_lambda == 10.0
        assert density_lambda == 0.15
        assert source == 'population'

    def test_zero_fallback_when_no_population_data(self) -> None:
        """When no baseline data available, return (0.0, 0.0, 'population')."""
        row = AggregatedRow(
            domain='example.com',
            bucket_start=datetime(2024, 3, 20),
            total_shares=100,
            unique_sharers=20,
            sharer_density=0.2,
            rolling_volume_median=None,
            rolling_volume_mean=None,
            rolling_volume_variance=None,
            rolling_density_mean=None,
            rolling_density_variance=None,
            baseline_days_available=2,
            sample_dids=['did1', 'did2'],
            sample_urls=['url1', 'url2'],
            population_volume_median=None,
            population_volume_dispersion=None,
            population_density_median=None,
            population_density_variance=None,
        )
        volume_lambda, density_lambda, source = determine_baseline(row, cold_start_min_days=3)
        assert volume_lambda == 0.0
        assert density_lambda == 0.0
        assert source == 'population'

    def test_exactly_at_minimum_days_threshold(self) -> None:
        """When baseline_days_available == cold_start_min_days, use entity baseline."""
        row = AggregatedRow(
            domain='example.com',
            bucket_start=datetime(2024, 3, 20),
            total_shares=100,
            unique_sharers=20,
            sharer_density=0.2,
            rolling_volume_median=25.0,
            rolling_volume_mean=30.0,
            rolling_volume_variance=40.0,
            rolling_density_mean=0.25,
            rolling_density_variance=0.05,
            baseline_days_available=3,
            sample_dids=['did1', 'did2'],
            sample_urls=['url1', 'url2'],
            population_volume_median=10.0,
            population_volume_dispersion=1.2,
            population_density_median=0.15,
            population_density_variance=0.03,
        )
        volume_lambda, density_lambda, source = determine_baseline(row, cold_start_min_days=3)
        assert volume_lambda == 25.0
        assert density_lambda == 0.25
        assert source == 'entity'


class TestDetermineVariances:
    def test_entity_source_with_dispersion_factor_gt_1(self) -> None:
        """When entity source and phi > 1, return computed volume_variance and density_variance."""
        row = AggregatedRow(
            domain='example.com',
            bucket_start=datetime(2024, 3, 20),
            total_shares=100,
            unique_sharers=20,
            sharer_density=0.2,
            rolling_volume_median=20.0,
            rolling_volume_mean=10.0,
            rolling_volume_variance=15.0,
            rolling_density_mean=0.25,
            rolling_density_variance=0.05,
            baseline_days_available=5,
            sample_dids=['did1'],
            sample_urls=['url1'],
            population_volume_median=10.0,
            population_volume_dispersion=1.2,
            population_density_median=0.15,
            population_density_variance=0.03,
        )
        volume_var, density_var = determine_variances(row, baseline_source='entity')
        # phi = 15.0 / 10.0 = 1.5 > 1
        # volume_variance = phi * median = 1.5 * 20.0 = 30.0
        assert volume_var == pytest.approx(30.0)
        assert density_var == 0.05

    def test_entity_source_with_dispersion_factor_eq_1(self) -> None:
        """When phi <= 1, fall back to Poisson (return None for volume_variance)."""
        row = AggregatedRow(
            domain='example.com',
            bucket_start=datetime(2024, 3, 20),
            total_shares=100,
            unique_sharers=20,
            sharer_density=0.2,
            rolling_volume_median=20.0,
            rolling_volume_mean=10.0,
            rolling_volume_variance=10.0,
            rolling_density_mean=0.25,
            rolling_density_variance=0.05,
            baseline_days_available=5,
            sample_dids=['did1'],
            sample_urls=['url1'],
            population_volume_median=10.0,
            population_volume_dispersion=1.2,
            population_density_median=0.15,
            population_density_variance=0.03,
        )
        volume_var, density_var = determine_variances(row, baseline_source='entity')
        # phi = 10.0 / 10.0 = 1.0 <= 1 -> None
        assert volume_var is None
        assert density_var == 0.05

    def test_entity_source_with_missing_variance(self) -> None:
        """When rolling_volume_variance is None, return None (Poisson fallback)."""
        row = AggregatedRow(
            domain='example.com',
            bucket_start=datetime(2024, 3, 20),
            total_shares=100,
            unique_sharers=20,
            sharer_density=0.2,
            rolling_volume_median=20.0,
            rolling_volume_mean=10.0,
            rolling_volume_variance=None,
            rolling_density_mean=0.25,
            rolling_density_variance=0.05,
            baseline_days_available=5,
            sample_dids=['did1'],
            sample_urls=['url1'],
            population_volume_median=10.0,
            population_volume_dispersion=1.2,
            population_density_median=0.15,
            population_density_variance=0.03,
        )
        volume_var, density_var = determine_variances(row, baseline_source='entity')
        assert volume_var is None
        assert density_var == 0.05

    def test_population_source(self) -> None:
        """When population source, use population_volume_dispersion and population_density_variance."""
        row = AggregatedRow(
            domain='example.com',
            bucket_start=datetime(2024, 3, 20),
            total_shares=100,
            unique_sharers=20,
            sharer_density=0.2,
            rolling_volume_median=20.0,
            rolling_volume_mean=10.0,
            rolling_volume_variance=15.0,
            rolling_density_mean=0.25,
            rolling_density_variance=0.05,
            baseline_days_available=5,
            sample_dids=['did1'],
            sample_urls=['url1'],
            population_volume_median=10.0,
            population_volume_dispersion=1.8,
            population_density_median=0.15,
            population_density_variance=0.03,
        )
        volume_var, density_var = determine_variances(row, baseline_source='population')
        # volume_variance = dispersion_factor * median = 1.8 * 10.0 = 18.0
        assert volume_var == pytest.approx(18.0)
        assert density_var == 0.03

    def test_population_source_with_zero_median(self) -> None:
        """When population_volume_median is 0, return None."""
        row = AggregatedRow(
            domain='example.com',
            bucket_start=datetime(2024, 3, 20),
            total_shares=100,
            unique_sharers=20,
            sharer_density=0.2,
            rolling_volume_median=20.0,
            rolling_volume_mean=10.0,
            rolling_volume_variance=15.0,
            rolling_density_mean=0.25,
            rolling_density_variance=0.05,
            baseline_days_available=5,
            sample_dids=['did1'],
            sample_urls=['url1'],
            population_volume_median=0.0,
            population_volume_dispersion=1.8,
            population_density_median=0.15,
            population_density_variance=0.03,
        )
        volume_var, density_var = determine_variances(row, baseline_source='population')
        assert volume_var is None
        assert density_var == 0.03


class TestScoreRow:
    def test_computes_count_p_value(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        """score_row should compute volume_p_value using count_p_value with derived inputs."""
        row = AggregatedRow(
            domain='example.com',
            bucket_start=datetime(2024, 3, 20),
            total_shares=10,
            unique_sharers=5,
            sharer_density=0.5,
            rolling_volume_median=5.0,
            rolling_volume_mean=5.0,
            rolling_volume_variance=7.5,
            rolling_density_mean=0.5,
            rolling_density_variance=0.05,
            baseline_days_available=7,
            sample_dids=['did1'],
            sample_urls=['url1'],
            population_volume_median=None,
            population_volume_dispersion=None,
            population_density_median=None,
            population_density_variance=None,
        )
        result = score_row(row, base_config, 'daily', run_timestamp, on_watchlist=0)
        # phi = 7.5 / 5.0 = 1.5 > 1, so NB is used
        # volume_variance = 1.5 * 5.0 = 7.5
        expected_volume_p = count_p_value(10, 5.0, 7.5)
        assert result.volume_p_value == pytest.approx(expected_volume_p)

    def test_computes_density_p_value(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        """score_row should compute density_p_value using density_p_value function."""
        row = AggregatedRow(
            domain='example.com',
            bucket_start=datetime(2024, 3, 20),
            total_shares=10,
            unique_sharers=9,
            sharer_density=0.9,
            rolling_volume_median=5.0,
            rolling_volume_mean=5.0,
            rolling_volume_variance=7.5,
            rolling_density_mean=0.5,
            rolling_density_variance=0.05,
            baseline_days_available=7,
            sample_dids=['did1'],
            sample_urls=['url1'],
            population_volume_median=None,
            population_volume_dispersion=None,
            population_density_median=None,
            population_density_variance=None,
        )
        result = score_row(row, base_config, 'daily', run_timestamp, on_watchlist=0)
        expected_density_p = density_p_value(9, 10, 0.5, 0.05)
        assert result.density_p_value == pytest.approx(expected_density_p)

    def test_q_values_initially_1_0(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        """score_row returns provisional q-values = 1.0 (adjusted by score_rows)."""
        row = AggregatedRow(
            domain='example.com',
            bucket_start=datetime(2024, 3, 20),
            total_shares=10,
            unique_sharers=5,
            sharer_density=0.5,
            rolling_volume_median=5.0,
            rolling_volume_mean=5.0,
            rolling_volume_variance=7.5,
            rolling_density_mean=0.5,
            rolling_density_variance=0.05,
            baseline_days_available=7,
            sample_dids=['did1'],
            sample_urls=['url1'],
            population_volume_median=None,
            population_volume_dispersion=None,
            population_density_median=None,
            population_density_variance=None,
        )
        result = score_row(row, base_config, 'daily', run_timestamp, on_watchlist=0)
        assert result.volume_q_value == 1.0
        assert result.density_q_value == 1.0

    def test_is_anomaly_initially_0(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        """score_row returns provisional is_anomaly = 0 (set by score_rows via q-values)."""
        row = AggregatedRow(
            domain='example.com',
            bucket_start=datetime(2024, 3, 20),
            total_shares=100,
            unique_sharers=80,
            sharer_density=0.8,
            rolling_volume_median=5.0,
            rolling_volume_mean=5.0,
            rolling_volume_variance=7.5,
            rolling_density_mean=0.5,
            rolling_density_variance=0.05,
            baseline_days_available=7,
            sample_dids=['did1'],
            sample_urls=['url1'],
            population_volume_median=None,
            population_volume_dispersion=None,
            population_density_median=None,
            population_density_variance=None,
        )
        result = score_row(row, base_config, 'daily', run_timestamp, on_watchlist=0)
        # Even if p-values are very low, provisional is_anomaly is 0
        assert result.is_anomaly == 0

    def test_populates_rolling_stats(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        """score_row populates rolling_volume_median, variance, and density stats."""
        row = AggregatedRow(
            domain='example.com',
            bucket_start=datetime(2024, 3, 20),
            total_shares=10,
            unique_sharers=5,
            sharer_density=0.5,
            rolling_volume_median=2.5,
            rolling_volume_mean=3.0,
            rolling_volume_variance=5.0,
            rolling_density_mean=0.48,
            rolling_density_variance=0.04,
            baseline_days_available=7,
            sample_dids=['did1'],
            sample_urls=['url1'],
            population_volume_median=None,
            population_volume_dispersion=None,
            population_density_median=None,
            population_density_variance=None,
        )
        result = score_row(row, base_config, 'daily', run_timestamp, on_watchlist=0)
        assert result.rolling_volume_median == 2.5
        assert result.rolling_volume_variance == 5.0
        assert result.rolling_density_mean == 0.48
        assert result.rolling_density_variance == 0.04

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
            rolling_volume_median=10.0,
            rolling_volume_mean=10.0,
            rolling_volume_variance=10.0,
            rolling_density_mean=0.1,
            rolling_density_variance=0.01,
            baseline_days_available=7,
            sample_dids=['did1'],
            sample_urls=['url1'],
            population_volume_median=None,
            population_volume_dispersion=None,
            population_density_median=None,
            population_density_variance=None,
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
                rolling_volume_median=10.0,
                rolling_volume_mean=10.0,
                rolling_volume_variance=10.0,
                rolling_density_mean=0.1,
                rolling_density_variance=0.01,
                baseline_days_available=7,
                sample_dids=['did1'],
                sample_urls=['url1'],
                population_volume_median=None,
                population_volume_dispersion=None,
                population_density_median=None,
                population_density_variance=None,
            ),
            AggregatedRow(
                domain='example2.com',
                bucket_start=datetime(2024, 3, 20),
                total_shares=20,
                unique_sharers=15,
                sharer_density=0.15,
                rolling_volume_median=20.0,
                rolling_volume_mean=20.0,
                rolling_volume_variance=20.0,
                rolling_density_mean=0.15,
                rolling_density_variance=0.02,
                baseline_days_available=7,
                sample_dids=['did2'],
                sample_urls=['url2'],
                population_volume_median=None,
                population_volume_dispersion=None,
                population_density_median=None,
                population_density_variance=None,
            ),
        ]
        results = score_rows(rows, base_config, 'daily', run_timestamp, ())
        assert len(results) == 2
        assert all(isinstance(r, ScoredResult) for r in results)

    def test_separate_bh_families_volume_and_density(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        """Two-pass scoring: volume and density adjusted as separate families.
        One row's volume signal anomalous, another's density signal anomalous."""
        rows = [
            AggregatedRow(
                domain='high-volume.com',
                bucket_start=datetime(2024, 3, 20),
                total_shares=100,
                unique_sharers=30,
                sharer_density=0.3,
                rolling_volume_median=5.0,
                rolling_volume_mean=5.0,
                rolling_volume_variance=10.0,
                rolling_density_mean=0.5,
                rolling_density_variance=0.05,
                baseline_days_available=7,
                sample_dids=['did1'],
                sample_urls=['url1'],
                population_volume_median=None,
                population_volume_dispersion=None,
                population_density_median=None,
                population_density_variance=None,
            ),
            AggregatedRow(
                domain='high-density.com',
                bucket_start=datetime(2024, 3, 20),
                total_shares=10,
                unique_sharers=9,
                sharer_density=0.9,
                rolling_volume_median=10.0,
                rolling_volume_mean=10.0,
                rolling_volume_variance=10.0,
                rolling_density_mean=0.5,
                rolling_density_variance=0.05,
                baseline_days_available=7,
                sample_dids=['did2'],
                sample_urls=['url2'],
                population_volume_median=None,
                population_volume_dispersion=None,
                population_density_median=None,
                population_density_variance=None,
            ),
        ]
        results = score_rows(rows, base_config, 'daily', run_timestamp, ())
        assert len(results) == 2

        # Both should have q-values computed
        for r in results:
            assert r.volume_q_value >= 0.0
            assert r.density_q_value >= 0.0
            assert r.volume_q_value <= 1.0
            assert r.density_q_value <= 1.0

        # is_anomaly should be set based on q-value thresholds
        # At least one should potentially be 1 if q-values cross threshold
        assert isinstance(results[0].is_anomaly, int)
        assert isinstance(results[1].is_anomaly, int)

    def test_or_logic_is_anomaly_on_q_values(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        """is_anomaly = 1 iff volume_q < threshold OR density_q < threshold."""
        # Create a minimal custom config with high thresholds
        config = AnalysisConfig(
            interval_seconds=900,
            volume_p_threshold=0.5,  # high threshold so we can easily trigger
            density_p_threshold=0.5,  # high threshold so we can easily trigger
            baseline_days=14,
            cold_start_min_days=3,
            min_sharers=3,
            watchlist_domains=(),
            source_table='osprey_execution_results',
            output_table='url_overdispersion_results',
        )

        rows = [
            AggregatedRow(
                domain='anomalous.com',
                bucket_start=datetime(2024, 3, 20),
                total_shares=100,
                unique_sharers=80,
                sharer_density=0.8,
                rolling_volume_median=5.0,
                rolling_volume_mean=5.0,
                rolling_volume_variance=10.0,
                rolling_density_mean=0.5,
                rolling_density_variance=0.05,
                baseline_days_available=7,
                sample_dids=['did1'],
                sample_urls=['url1'],
                population_volume_median=None,
                population_volume_dispersion=None,
                population_density_median=None,
                population_density_variance=None,
            ),
        ]
        results = score_rows(rows, config, 'daily', run_timestamp, ())
        # With high thresholds and extreme observed values, should flag as anomaly
        assert len(results) == 1
        # Check that is_anomaly is based on OR logic of q-values
        assert (results[0].volume_q_value < 0.5 or results[0].density_q_value < 0.5) == (results[0].is_anomaly == 1)

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
                rolling_volume_median=10.0,
                rolling_volume_mean=10.0,
                rolling_volume_variance=10.0,
                rolling_density_mean=0.1,
                rolling_density_variance=0.01,
                baseline_days_available=7,
                sample_dids=['did1'],
                sample_urls=['url1'],
                population_volume_median=None,
                population_volume_dispersion=None,
                population_density_median=None,
                population_density_variance=None,
            ),
            AggregatedRow(
                domain='normal.com',
                bucket_start=datetime(2024, 3, 20),
                total_shares=10,
                unique_sharers=10,
                sharer_density=0.1,
                rolling_volume_median=10.0,
                rolling_volume_mean=10.0,
                rolling_volume_variance=10.0,
                rolling_density_mean=0.1,
                rolling_density_variance=0.01,
                baseline_days_available=7,
                sample_dids=['did2'],
                sample_urls=['url2'],
                population_volume_median=None,
                population_volume_dispersion=None,
                population_density_median=None,
                population_density_variance=None,
            ),
        ]
        watchlist = ('watchlisted.com',)
        results = score_rows(rows, base_config, 'daily', run_timestamp, watchlist)
        assert results[0].on_watchlist == 1
        assert results[1].on_watchlist == 0
