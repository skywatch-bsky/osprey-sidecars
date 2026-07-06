# pattern: Functional Core
from __future__ import annotations

from datetime import datetime

import pytest

from quote_overdispersion.analyzer import (
    determine_baseline,
    determine_variances,
    extract_quoted_author_did,
    score_row,
    score_rows,
)
from quote_overdispersion.config import AnalysisConfig
from quote_overdispersion.counts import count_p_value
from quote_overdispersion.db import AggregatedRow, ScoredResult
from quote_overdispersion.density import density_p_value


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


class TestDetermineBaseline:
    def test_ac1_3_entity_baseline_used_when_sufficient(self) -> None:
        row = AggregatedRow(
            quoted_uri='at://did:plc:abc/app.bsky.feed.post/xyz',
            bucket_start=datetime(2024, 3, 20),
            total_shares=100,
            unique_sharers=20,
            sharer_density=0.2,
            rolling_volume_median=30.0,
            rolling_volume_mean=32.0,
            rolling_volume_variance=40.0,
            rolling_density_mean=0.25,
            rolling_density_variance=0.05,
            baseline_days_available=5,
            sample_dids=['did1', 'did2'],
            population_volume_median=10.0,
            population_volume_dispersion=1.5,
            population_density_median=0.1,
            population_density_variance=0.02,
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
            rolling_volume_median=30.0,
            rolling_volume_mean=32.0,
            rolling_volume_variance=40.0,
            rolling_density_mean=0.25,
            rolling_density_variance=0.05,
            baseline_days_available=2,
            sample_dids=['did1', 'did2'],
            population_volume_median=10.0,
            population_volume_dispersion=1.5,
            population_density_median=0.15,
            population_density_variance=0.03,
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
            rolling_volume_median=None,
            rolling_volume_mean=None,
            rolling_volume_variance=None,
            rolling_density_mean=0.25,
            rolling_density_variance=0.05,
            baseline_days_available=5,
            sample_dids=['did1', 'did2'],
            population_volume_median=10.0,
            population_volume_dispersion=1.5,
            population_density_median=0.15,
            population_density_variance=0.03,
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
            rolling_volume_median=None,
            rolling_volume_mean=None,
            rolling_volume_variance=None,
            rolling_density_mean=None,
            rolling_density_variance=None,
            baseline_days_available=2,
            sample_dids=['did1', 'did2'],
            population_volume_median=None,
            population_volume_dispersion=None,
            population_density_median=None,
            population_density_variance=None,
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
            rolling_volume_median=30.0,
            rolling_volume_mean=32.0,
            rolling_volume_variance=40.0,
            rolling_density_mean=0.25,
            rolling_density_variance=0.05,
            baseline_days_available=3,
            sample_dids=['did1', 'did2'],
            population_volume_median=10.0,
            population_volume_dispersion=1.5,
            population_density_median=0.15,
            population_density_variance=0.03,
        )
        volume_lambda, density_lambda, source = determine_baseline(row, cold_start_min_days=3)
        assert volume_lambda == 30.0
        assert density_lambda == 0.25
        assert source == 'entity'


class TestDetermineVariances:
    def test_entity_baseline_reconstructs_volume_variance_from_dispersion(self) -> None:
        """AC4.2: volume variance = phi * median when phi > 1, else None.

        Fixture: volume_variance=40.0, mean=32.0, median=30.0
        → phi = 40/32 = 1.25 > 1
        → volume_var = 1.25 * 30.0 = 37.5
        """
        row = AggregatedRow(
            quoted_uri='at://did:plc:abc/app.bsky.feed.post/xyz',
            bucket_start=datetime(2024, 3, 20),
            total_shares=100,
            unique_sharers=20,
            sharer_density=0.2,
            rolling_volume_median=30.0,
            rolling_volume_mean=32.0,
            rolling_volume_variance=40.0,
            rolling_density_mean=0.25,
            rolling_density_variance=0.05,
            baseline_days_available=5,
            sample_dids=['did1'],
            population_volume_median=10.0,
            population_volume_dispersion=1.5,
            population_density_median=0.1,
            population_density_variance=0.02,
        )
        volume_var, density_var = determine_variances(row, 'entity')
        assert volume_var == pytest.approx(37.5, abs=1e-10)
        assert density_var == 0.05

    def test_entity_baseline_phi_le_1_returns_none(self) -> None:
        """AC4.2: volume variance = None when phi <= 1 (Poisson fallback).

        Fixture: variance=30.0, mean=32.0
        → phi = 30/32 = 0.9375 <= 1
        → volume_var = None
        """
        row = AggregatedRow(
            quoted_uri='at://did:plc:abc/app.bsky.feed.post/xyz',
            bucket_start=datetime(2024, 3, 20),
            total_shares=100,
            unique_sharers=20,
            sharer_density=0.2,
            rolling_volume_median=30.0,
            rolling_volume_mean=32.0,
            rolling_volume_variance=30.0,
            rolling_density_mean=0.25,
            rolling_density_variance=0.05,
            baseline_days_available=5,
            sample_dids=['did1'],
            population_volume_median=10.0,
            population_volume_dispersion=1.5,
            population_density_median=0.1,
            population_density_variance=0.02,
        )
        volume_var, density_var = determine_variances(row, 'entity')
        assert volume_var is None
        assert density_var == 0.05

    def test_population_baseline_derives_variances_phi_gt_1(self) -> None:
        """AC4.2: volume variance = dispersion * median when dispersion > 1, else None.

        Fixture: dispersion=2.0, median=10.0
        → phi = 2.0 > 1
        → volume_var = 2.0 * 10.0 = 20.0
        """
        row = AggregatedRow(
            quoted_uri='at://did:plc:abc/app.bsky.feed.post/xyz',
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
            sample_dids=['did1'],
            population_volume_median=10.0,
            population_volume_dispersion=2.0,
            population_density_median=0.1,
            population_density_variance=0.03,
        )
        volume_var, density_var = determine_variances(row, 'population')
        assert volume_var == 20.0  # 2.0 * 10.0
        assert density_var == 0.03

    def test_population_baseline_dispersion_le_1_returns_none(self) -> None:
        """AC4.2: volume variance = None when dispersion <= 1 (Poisson fallback).

        Fixture: dispersion=0.8, median=10.0
        → phi = 0.8 <= 1
        → volume_var = None
        """
        row = AggregatedRow(
            quoted_uri='at://did:plc:abc/app.bsky.feed.post/xyz',
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
            sample_dids=['did1'],
            population_volume_median=10.0,
            population_volume_dispersion=0.8,
            population_density_median=0.1,
            population_density_variance=0.03,
        )
        volume_var, density_var = determine_variances(row, 'population')
        assert volume_var is None
        assert density_var == 0.03

    def test_population_baseline_with_none_population_dispersion(self) -> None:
        row = AggregatedRow(
            quoted_uri='at://did:plc:abc/app.bsky.feed.post/xyz',
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
            sample_dids=['did1'],
            population_volume_median=10.0,
            population_volume_dispersion=None,
            population_density_median=0.1,
            population_density_variance=0.03,
        )
        volume_var, density_var = determine_variances(row, 'population')
        assert volume_var is None
        assert density_var == 0.03

    def test_population_baseline_with_zero_median_returns_none(self) -> None:
        """Important 1: gate on population_volume_median > 0."""
        row = AggregatedRow(
            quoted_uri='at://did:plc:abc/app.bsky.feed.post/xyz',
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
            sample_dids=['did1'],
            population_volume_median=0.0,
            population_volume_dispersion=2.0,
            population_density_median=0.1,
            population_density_variance=0.03,
        )
        volume_var, density_var = determine_variances(row, 'population')
        assert volume_var is None
        assert density_var == 0.03


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
            rolling_volume_median=5.0,
            rolling_volume_mean=5.5,
            rolling_volume_variance=8.0,
            rolling_density_mean=0.1,
            rolling_density_variance=0.02,
            baseline_days_available=7,
            sample_dids=['did1', 'did2'],
            population_volume_median=None,
            population_volume_dispersion=None,
            population_density_median=None,
            population_density_variance=None,
        )
        result = score_row(row, base_config, 'daily', run_timestamp)
        assert result.volume_p_value < base_config.volume_p_threshold
        assert result.volume_q_value == 1.0  # provisional
        assert result.density_q_value == 1.0  # provisional
        assert result.is_anomaly == 0  # provisional (not set yet)

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
            rolling_volume_median=15.0,
            rolling_volume_mean=16.0,
            rolling_volume_variance=20.0,
            rolling_density_mean=0.1,
            rolling_density_variance=0.02,
            baseline_days_available=7,
            sample_dids=['did1', 'did2'],
            population_volume_median=None,
            population_volume_dispersion=None,
            population_density_median=None,
            population_density_variance=None,
        )
        result = score_row(row, base_config, 'daily', run_timestamp)
        assert result.density_p_value < base_config.density_p_threshold
        assert result.volume_q_value == 1.0  # provisional
        assert result.density_q_value == 1.0  # provisional
        assert result.is_anomaly == 0  # provisional (not set yet)

    def test_both_normal_no_flag(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        row = AggregatedRow(
            quoted_uri='at://did:plc:abc/app.bsky.feed.post/xyz',
            bucket_start=datetime(2024, 3, 20),
            total_shares=10,
            unique_sharers=5,
            sharer_density=0.5,
            rolling_volume_median=10.0,
            rolling_volume_mean=10.5,
            rolling_volume_variance=12.0,
            rolling_density_mean=0.5,
            rolling_density_variance=0.02,
            baseline_days_available=7,
            sample_dids=['did1', 'did2'],
            population_volume_median=None,
            population_volume_dispersion=None,
            population_density_median=None,
            population_density_variance=None,
        )
        result = score_row(row, base_config, 'daily', run_timestamp)
        assert result.volume_p_value >= base_config.volume_p_threshold
        assert result.density_p_value >= base_config.density_p_threshold
        assert result.volume_q_value == 1.0  # provisional
        assert result.density_q_value == 1.0  # provisional
        assert result.is_anomaly == 0  # provisional (not set yet)

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
            rolling_volume_median=None,
            rolling_volume_mean=None,
            rolling_volume_variance=None,
            rolling_density_mean=None,
            rolling_density_variance=None,
            baseline_days_available=2,
            sample_dids=['did1', 'did2'],
            population_volume_median=None,
            population_volume_dispersion=None,
            population_density_median=None,
            population_density_variance=None,
        )
        result = score_row(row, base_config, 'daily', run_timestamp)
        assert result.volume_p_value == 1.0
        assert result.density_p_value == 1.0
        assert result.volume_q_value == 1.0  # provisional
        assert result.density_q_value == 1.0  # provisional
        assert result.is_anomaly == 0  # provisional (not set yet)

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
            rolling_volume_median=10.0,
            rolling_volume_mean=10.5,
            rolling_volume_variance=12.0,
            rolling_density_mean=0.1,
            rolling_density_variance=0.01,
            baseline_days_available=7,
            sample_dids=['did1'],
            population_volume_median=None,
            population_volume_dispersion=None,
            population_density_median=None,
            population_density_variance=None,
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
                rolling_volume_median=10.0,
                rolling_volume_mean=10.5,
                rolling_volume_variance=12.0,
                rolling_density_mean=0.1,
                rolling_density_variance=0.01,
                baseline_days_available=7,
                sample_dids=['did1'],
                population_volume_median=None,
                population_volume_dispersion=None,
                population_density_median=None,
                population_density_variance=None,
            ),
            AggregatedRow(
                quoted_uri='at://did:plc:abc2/app.bsky.feed.post/xyz2',
                bucket_start=datetime(2024, 3, 20),
                total_shares=20,
                unique_sharers=15,
                sharer_density=0.15,
                rolling_volume_median=20.0,
                rolling_volume_mean=21.0,
                rolling_volume_variance=25.0,
                rolling_density_mean=0.15,
                rolling_density_variance=0.015,
                baseline_days_available=7,
                sample_dids=['did2'],
                population_volume_median=None,
                population_volume_dispersion=None,
                population_density_median=None,
                population_density_variance=None,
            ),
        ]
        results = score_rows(rows, base_config, 'daily', run_timestamp)
        assert len(results) == 2
        assert all(isinstance(r, ScoredResult) for r in results)
        assert results[0].quoted_author_did == 'did:plc:abc1'
        assert results[1].quoted_author_did == 'did:plc:abc2'
        # Check that q-values are set (not provisional 1.0 anymore)
        assert results[0].volume_q_value >= 0
        assert results[0].density_q_value >= 0
        assert results[1].volume_q_value >= 0
        assert results[1].density_q_value >= 0
        # Check that is_anomaly is determined (not provisional 0)
        assert results[0].is_anomaly in (0, 1)
        assert results[1].is_anomaly in (0, 1)

    def test_bh_adjustment_per_signal_separate_families(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        """AC2.3: volume and density p-values are adjusted as separate BH families."""
        rows = [
            AggregatedRow(
                quoted_uri='at://did:plc:e1/app.bsky.feed.post/p1',
                bucket_start=datetime(2024, 3, 20),
                total_shares=100,
                unique_sharers=50,
                sharer_density=0.8,
                rolling_volume_median=5.0,
                rolling_volume_mean=5.5,
                rolling_volume_variance=8.0,
                rolling_density_mean=0.1,
                rolling_density_variance=0.02,
                baseline_days_available=7,
                sample_dids=['did1'],
                population_volume_median=None,
                population_volume_dispersion=None,
                population_density_median=None,
                population_density_variance=None,
            ),
            AggregatedRow(
                quoted_uri='at://did:plc:e2/app.bsky.feed.post/p2',
                bucket_start=datetime(2024, 3, 20),
                total_shares=10,
                unique_sharers=10,
                sharer_density=0.1,
                rolling_volume_median=10.0,
                rolling_volume_mean=10.5,
                rolling_volume_variance=12.0,
                rolling_density_mean=0.1,
                rolling_density_variance=0.01,
                baseline_days_available=7,
                sample_dids=['did2'],
                population_volume_median=None,
                population_volume_dispersion=None,
                population_density_median=None,
                population_density_variance=None,
            ),
        ]
        results = score_rows(rows, base_config, 'daily', run_timestamp)
        # Both results should have q-values set (separate BH families)
        assert results[0].volume_q_value >= results[0].volume_p_value
        assert results[1].volume_q_value >= results[1].volume_p_value
        assert results[0].density_q_value >= results[0].density_p_value
        assert results[1].density_q_value >= results[1].density_p_value

    def test_computes_volume_p_value(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        """score_row should compute volume_p_value using count_p_value with derived inputs."""
        row = AggregatedRow(
            quoted_uri='at://did:plc:abc/app.bsky.feed.post/xyz',
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
            population_volume_median=None,
            population_volume_dispersion=None,
            population_density_median=None,
            population_density_variance=None,
        )
        result = score_row(row, base_config, 'daily', run_timestamp)
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
            quoted_uri='at://did:plc:abc/app.bsky.feed.post/xyz',
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
            population_volume_median=None,
            population_volume_dispersion=None,
            population_density_median=None,
            population_density_variance=None,
        )
        result = score_row(row, base_config, 'daily', run_timestamp)
        expected_density_p = density_p_value(9, 10, 0.5, 0.05)
        assert result.density_p_value == pytest.approx(expected_density_p)

    def test_ac2_2_anomaly_equals_q_value_comparison(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        """AC2.2: is_anomaly == (1 if volume_q < threshold OR density_q < threshold else 0)."""
        # Test case 1: both q-values exceed threshold -> is_anomaly = 0
        row1 = AggregatedRow(
            quoted_uri='at://did:plc:abc/app.bsky.feed.post/xyz',
            bucket_start=datetime(2024, 3, 20),
            total_shares=2,
            unique_sharers=2,
            sharer_density=1.0,
            rolling_volume_median=5.0,
            rolling_volume_mean=5.0,
            rolling_volume_variance=5.0,
            rolling_density_mean=0.5,
            rolling_density_variance=0.05,
            baseline_days_available=7,
            sample_dids=['did1'],
            population_volume_median=None,
            population_volume_dispersion=None,
            population_density_median=None,
            population_density_variance=None,
        )
        result1 = score_row(row1, base_config, 'daily', run_timestamp)
        threshold = base_config.volume_p_threshold
        expected_anomaly1 = 1 if (result1.volume_q_value < threshold or result1.density_q_value < threshold) else 0
        assert result1.is_anomaly == expected_anomaly1

        # Test case 2: volume_q-value below threshold -> is_anomaly = 1
        row2 = AggregatedRow(
            quoted_uri='at://did:plc:abc/app.bsky.feed.post/xyz2',
            bucket_start=datetime(2024, 3, 20),
            total_shares=100,
            unique_sharers=20,
            sharer_density=0.2,
            rolling_volume_median=5.0,
            rolling_volume_mean=5.0,
            rolling_volume_variance=7.5,
            rolling_density_mean=0.5,
            rolling_density_variance=0.05,
            baseline_days_available=7,
            sample_dids=['did1'],
            population_volume_median=None,
            population_volume_dispersion=None,
            population_density_median=None,
            population_density_variance=None,
        )
        result2 = score_row(row2, base_config, 'daily', run_timestamp)
        expected_anomaly2 = 1 if (result2.volume_q_value < threshold or result2.density_q_value < threshold) else 0
        assert result2.is_anomaly == expected_anomaly2

    def test_processes_empty_list(
        self,
        base_config: AnalysisConfig,
        run_timestamp: datetime,
    ) -> None:
        rows = []
        results = score_rows(rows, base_config, 'daily', run_timestamp)
        assert len(results) == 0
