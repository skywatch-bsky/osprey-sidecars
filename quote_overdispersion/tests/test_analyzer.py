# pattern: Functional Core
from __future__ import annotations

from datetime import datetime

import pytest

from quote_overdispersion.analyzer import (
    compute_density_p_value,
    compute_p_value,
    determine_baseline,
    extract_quoted_author_did,
    score_row,
    score_rows,
)
from quote_overdispersion.config import AnalysisConfig
from quote_overdispersion.db import AggregatedRow, ScoredResult


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
def run_timestamp() -> datetime:
    return datetime(2024, 3, 20, 12, 0, 0)


class TestExtractQuotedAuthorDid:
    def test_valid_at_uri_plc_format(self) -> None:
        uri = 'at://did:plc:abc123/app.bsky.feed.post/xyz'
        result = extract_quoted_author_did(uri)
        assert result == 'did:plc:abc123'

    def test_valid_at_uri_web_format(self) -> None:
        uri = 'at://did:web:example.com/app.bsky.feed.post/xyz'
        result = extract_quoted_author_did(uri)
        assert result == 'did:web:example.com'

    def test_malformed_uri_missing_prefix(self) -> None:
        uri = 'https://example.com/app.bsky.feed.post/xyz'
        result = extract_quoted_author_did(uri)
        assert result == ''

    def test_uri_with_insufficient_segments(self) -> None:
        uri = 'at://did:plc:abc123'
        result = extract_quoted_author_did(uri)
        assert result == 'did:plc:abc123'

    def test_empty_string_input(self) -> None:
        result = extract_quoted_author_did('')
        assert result == ''

    def test_non_at_uri_string(self) -> None:
        uri = 'https://example.com'
        result = extract_quoted_author_did(uri)
        assert result == ''


class TestComputePValue:
    def test_ac1_5_zero_lambda_returns_one(self) -> None:
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
            quoted_uri='at://did:plc:abc/app.bsky.feed.post/xyz',
            bucket_start=datetime(2024, 3, 20),
            total_shares=100,
            unique_sharers=20,
            sharer_density=0.2,
            rolling_volume_mean=30.0,
            rolling_density_mean=0.25,
            baseline_days_available=5,
            sample_dids=['did1', 'did2'],
            population_volume_median=10.0,
            population_density_median=0.1,
        )
        volume_lambda, density_lambda, source = determine_baseline(row, cold_start_min_days=3)
        assert volume_lambda == 30.0
        assert density_lambda == 0.25
        assert source == 'entity'

    def test_ac1_4_population_fallback_when_insufficient(self) -> None:
        row = AggregatedRow(
            quoted_uri='at://did:plc:abc/app.bsky.feed.post/xyz',
            bucket_start=datetime(2024, 3, 20),
            total_shares=100,
            unique_sharers=20,
            sharer_density=0.2,
            rolling_volume_mean=30.0,
            rolling_density_mean=0.25,
            baseline_days_available=2,
            sample_dids=['did1', 'did2'],
            population_volume_median=10.0,
            population_density_median=0.15,
        )
        volume_lambda, density_lambda, source = determine_baseline(row, cold_start_min_days=3)
        assert volume_lambda == 10.0
        assert density_lambda == 0.15
        assert source == 'population'

    def test_ac1_4_population_fallback_when_rolling_none(self) -> None:
        row = AggregatedRow(
            quoted_uri='at://did:plc:abc/app.bsky.feed.post/xyz',
            bucket_start=datetime(2024, 3, 20),
            total_shares=100,
            unique_sharers=20,
            sharer_density=0.2,
            rolling_volume_mean=None,
            rolling_density_mean=0.25,
            baseline_days_available=5,
            sample_dids=['did1', 'did2'],
            population_volume_median=10.0,
            population_density_median=0.15,
        )
        volume_lambda, density_lambda, source = determine_baseline(row, cold_start_min_days=3)
        assert volume_lambda == 10.0
        assert density_lambda == 0.15
        assert source == 'population'

    def test_ac1_5_zero_fallback_when_no_data(self) -> None:
        row = AggregatedRow(
            quoted_uri='at://did:plc:abc/app.bsky.feed.post/xyz',
            bucket_start=datetime(2024, 3, 20),
            total_shares=100,
            unique_sharers=20,
            sharer_density=0.2,
            rolling_volume_mean=None,
            rolling_density_mean=None,
            baseline_days_available=2,
            sample_dids=['did1', 'did2'],
            population_volume_median=None,
            population_density_median=None,
        )
        volume_lambda, density_lambda, source = determine_baseline(row, cold_start_min_days=3)
        assert volume_lambda == 0.0
        assert density_lambda == 0.0
        assert source == 'population'

    def test_exactly_at_threshold(self) -> None:
        row = AggregatedRow(
            quoted_uri='at://did:plc:abc/app.bsky.feed.post/xyz',
            bucket_start=datetime(2024, 3, 20),
            total_shares=100,
            unique_sharers=20,
            sharer_density=0.2,
            rolling_volume_mean=30.0,
            rolling_density_mean=0.25,
            baseline_days_available=3,
            sample_dids=['did1', 'did2'],
            population_volume_median=10.0,
            population_density_median=0.15,
        )
        volume_lambda, density_lambda, source = determine_baseline(row, cold_start_min_days=3)
        assert volume_lambda == 30.0
        assert density_lambda == 0.25
        assert source == 'entity'


class TestScoreRow:
    def test_ac1_1_volume_anomaly_sets_flag(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        row = AggregatedRow(
            quoted_uri='at://did:plc:abc/app.bsky.feed.post/xyz',
            bucket_start=datetime(2024, 3, 20),
            total_shares=50,
            unique_sharers=20,
            sharer_density=0.4,
            rolling_volume_mean=5.0,
            rolling_density_mean=0.1,
            baseline_days_available=7,
            sample_dids=['did1', 'did2'],
            population_volume_median=None,
            population_density_median=None,
        )
        result = score_row(row, base_config, 'daily', run_timestamp)
        assert result.volume_p_value < base_config.volume_p_threshold
        assert result.is_anomaly == 1

    def test_ac1_2_density_anomaly_sets_flag(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        row = AggregatedRow(
            quoted_uri='at://did:plc:abc/app.bsky.feed.post/xyz',
            bucket_start=datetime(2024, 3, 20),
            total_shares=10,
            unique_sharers=10,
            sharer_density=0.9,
            rolling_volume_mean=15.0,
            rolling_density_mean=0.1,
            baseline_days_available=7,
            sample_dids=['did1', 'did2'],
            population_volume_median=None,
            population_density_median=None,
        )
        result = score_row(row, base_config, 'daily', run_timestamp)
        assert result.density_p_value < base_config.density_p_threshold
        assert result.is_anomaly == 1

    def test_both_normal_no_flag(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        row = AggregatedRow(
            quoted_uri='at://did:plc:abc/app.bsky.feed.post/xyz',
            bucket_start=datetime(2024, 3, 20),
            total_shares=10,
            unique_sharers=10,
            sharer_density=0.1,
            rolling_volume_mean=10.0,
            rolling_density_mean=0.1,
            baseline_days_available=7,
            sample_dids=['did1', 'did2'],
            population_volume_median=None,
            population_density_median=None,
        )
        result = score_row(row, base_config, 'daily', run_timestamp)
        assert result.volume_p_value >= base_config.volume_p_threshold
        assert result.density_p_value >= base_config.density_p_threshold
        assert result.is_anomaly == 0

    def test_ac1_5_zero_baseline_no_anomaly(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        row = AggregatedRow(
            quoted_uri='at://did:plc:abc/app.bsky.feed.post/xyz',
            bucket_start=datetime(2024, 3, 20),
            total_shares=100,
            unique_sharers=20,
            sharer_density=0.9,
            rolling_volume_mean=None,
            rolling_density_mean=None,
            baseline_days_available=2,
            sample_dids=['did1', 'did2'],
            population_volume_median=None,
            population_density_median=None,
        )
        result = score_row(row, base_config, 'daily', run_timestamp)
        assert result.volume_p_value == 1.0
        assert result.density_p_value == 1.0
        assert result.is_anomaly == 0

    def test_extracts_quoted_author_did(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        row = AggregatedRow(
            quoted_uri='at://did:plc:testdid123/app.bsky.feed.post/xyz',
            bucket_start=datetime(2024, 3, 20),
            total_shares=10,
            unique_sharers=10,
            sharer_density=0.1,
            rolling_volume_mean=10.0,
            rolling_density_mean=0.1,
            baseline_days_available=7,
            sample_dids=['did1'],
            population_volume_median=None,
            population_density_median=None,
        )
        result = score_row(row, base_config, 'daily', run_timestamp)
        assert result.quoted_author_did == 'did:plc:testdid123'


class TestScoreRows:
    def test_maps_all_rows(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        rows = [
            AggregatedRow(
                quoted_uri='at://did:plc:abc1/app.bsky.feed.post/xyz1',
                bucket_start=datetime(2024, 3, 20),
                total_shares=10,
                unique_sharers=10,
                sharer_density=0.1,
                rolling_volume_mean=10.0,
                rolling_density_mean=0.1,
                baseline_days_available=7,
                sample_dids=['did1'],
                population_volume_median=None,
                population_density_median=None,
            ),
            AggregatedRow(
                quoted_uri='at://did:plc:abc2/app.bsky.feed.post/xyz2',
                bucket_start=datetime(2024, 3, 20),
                total_shares=20,
                unique_sharers=15,
                sharer_density=0.15,
                rolling_volume_mean=20.0,
                rolling_density_mean=0.15,
                baseline_days_available=7,
                sample_dids=['did2'],
                population_volume_median=None,
                population_density_median=None,
            ),
        ]
        results = score_rows(rows, base_config, 'daily', run_timestamp)
        assert len(results) == 2
        assert all(isinstance(r, ScoredResult) for r in results)
        assert results[0].quoted_author_did == 'did:plc:abc1'
        assert results[1].quoted_author_did == 'did:plc:abc2'

    def test_processes_empty_list(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        rows = []
        results = score_rows(rows, base_config, 'daily', run_timestamp)
        assert len(results) == 0
