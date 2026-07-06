# pattern: Functional Core
from datetime import datetime

import pytest

from signup_anomaly.db import AggregatedRow, ScoredResult


class TestAggregatedRow:
    def test_create_with_all_fields(self) -> None:
        row = AggregatedRow(
            pds_host='example.com',
            observed_count=42,
            distinct_accounts=40,
            rolling_median=3.2,
            rolling_mean=3.5,
            baseline_days_available=7,
            sample_dids=['did:plc:abc1', 'did:plc:abc2'],
            population_median_lambda=2.1,
            rolling_variance=None,
            dispersion_index=None,
            population_dispersion_index=None,
        )
        assert row.pds_host == 'example.com'
        assert row.observed_count == 42
        assert row.rolling_median == 3.2
        assert row.rolling_mean == 3.5
        assert row.baseline_days_available == 7
        assert row.sample_dids == ['did:plc:abc1', 'did:plc:abc2']
        assert row.population_median_lambda == 2.1

    def test_create_with_none_rolling_mean(self) -> None:
        row = AggregatedRow(
            pds_host='example.com',
            observed_count=1,
            distinct_accounts=1,
            rolling_median=None,
            rolling_mean=None,
            baseline_days_available=0,
            sample_dids=[],
            population_median_lambda=None,
            rolling_variance=None,
            dispersion_index=None,
            population_dispersion_index=None,
        )
        assert row.rolling_median is None
        assert row.rolling_mean is None
        assert row.population_median_lambda is None

    def test_is_frozen(self) -> None:
        row = AggregatedRow(
            pds_host='example.com',
            observed_count=42,
            distinct_accounts=42,
            rolling_median=3.2,
            rolling_mean=3.5,
            baseline_days_available=7,
            sample_dids=['did:plc:abc1'],
            population_median_lambda=2.1,
            rolling_variance=None,
            dispersion_index=None,
            population_dispersion_index=None,
        )
        with pytest.raises(AttributeError):
            row.observed_count = 100

    def test_empty_sample_dids(self) -> None:
        row = AggregatedRow(
            pds_host='example.com',
            observed_count=0,
            distinct_accounts=0,
            rolling_median=None,
            rolling_mean=None,
            baseline_days_available=0,
            sample_dids=[],
            population_median_lambda=None,
            rolling_variance=None,
            dispersion_index=None,
            population_dispersion_index=None,
        )
        assert row.sample_dids == []

    def test_create_aggregated_row_with_dispersion_fields(self) -> None:
        row = AggregatedRow(
            pds_host='example.com',
            observed_count=42,
            distinct_accounts=42,
            rolling_median=3.2,
            rolling_mean=3.5,
            baseline_days_available=7,
            sample_dids=['did:plc:abc1'],
            population_median_lambda=2.1,
            rolling_variance=0.75,
            dispersion_index=1.5,
            population_dispersion_index=1.2,
        )
        assert row.rolling_median == 3.2
        assert row.rolling_variance == 0.75
        assert row.dispersion_index == 1.5
        assert row.population_dispersion_index == 1.2


class TestScoredResult:
    def test_create_with_entity_baseline(self) -> None:
        now = datetime(2026, 3, 20, 12, 0, 0)
        result = ScoredResult(
            run_timestamp=now,
            granularity='daily',
            pds_host='example.com',
            observed_count=42,
            distinct_accounts=40,
            expected_lambda=3.5,
            p_value=0.001,
            q_value=0.005,
            is_anomaly=1,
            baseline_source='entity',
            baseline_days_available=7,
            sample_dids=['did:plc:abc1', 'did:plc:abc2'],
            rolling_mean=None,
            rolling_variance=None,
            dispersion_index=None,
        )
        assert result.run_timestamp == now
        assert result.granularity == 'daily'
        assert result.pds_host == 'example.com'
        assert result.observed_count == 42
        assert result.expected_lambda == 3.5
        assert result.p_value == 0.001
        assert result.q_value == 0.005
        assert result.is_anomaly == 1
        assert result.baseline_source == 'entity'
        assert result.baseline_days_available == 7
        assert result.sample_dids == ['did:plc:abc1', 'did:plc:abc2']

    def test_create_with_population_baseline(self) -> None:
        now = datetime(2026, 3, 20, 12, 0, 0)
        result = ScoredResult(
            run_timestamp=now,
            granularity='hourly',
            pds_host='example.com',
            observed_count=10,
            distinct_accounts=10,
            expected_lambda=2.0,
            p_value=0.05,
            q_value=0.1,
            is_anomaly=0,
            baseline_source='population',
            baseline_days_available=3,
            sample_dids=['did:plc:abc1'],
            rolling_mean=None,
            rolling_variance=None,
            dispersion_index=None,
        )
        assert result.baseline_source == 'population'
        assert result.granularity == 'hourly'
        assert result.q_value == 0.1

    def test_is_frozen(self) -> None:
        now = datetime(2026, 3, 20, 12, 0, 0)
        result = ScoredResult(
            run_timestamp=now,
            granularity='daily',
            pds_host='example.com',
            observed_count=42,
            distinct_accounts=40,
            expected_lambda=3.5,
            p_value=0.001,
            q_value=0.005,
            is_anomaly=1,
            baseline_source='entity',
            baseline_days_available=7,
            sample_dids=['did:plc:abc1'],
            rolling_mean=None,
            rolling_variance=None,
            dispersion_index=None,
        )
        with pytest.raises(AttributeError):
            result.observed_count = 100

    def test_create_with_empty_sample_dids(self) -> None:
        now = datetime(2026, 3, 20, 12, 0, 0)
        result = ScoredResult(
            run_timestamp=now,
            granularity='daily',
            pds_host='example.com',
            observed_count=0,
            distinct_accounts=0,
            expected_lambda=1.0,
            p_value=0.9,
            q_value=0.9,
            is_anomaly=0,
            baseline_source='entity',
            baseline_days_available=0,
            sample_dids=[],
            rolling_mean=None,
            rolling_variance=None,
            dispersion_index=None,
        )
        assert result.sample_dids == []

    def test_is_anomaly_values(self) -> None:
        now = datetime(2026, 3, 20, 12, 0, 0)
        result_anomaly = ScoredResult(
            run_timestamp=now,
            granularity='daily',
            pds_host='example.com',
            observed_count=100,
            distinct_accounts=95,
            expected_lambda=2.0,
            p_value=0.001,
            q_value=0.005,
            is_anomaly=1,
            baseline_source='entity',
            baseline_days_available=7,
            sample_dids=[],
            rolling_mean=None,
            rolling_variance=None,
            dispersion_index=None,
        )
        assert result_anomaly.is_anomaly == 1

        result_normal = ScoredResult(
            run_timestamp=now,
            granularity='daily',
            pds_host='example.com',
            observed_count=2,
            distinct_accounts=2,
            expected_lambda=2.0,
            p_value=0.9,
            q_value=0.9,
            is_anomaly=0,
            baseline_source='entity',
            baseline_days_available=7,
            sample_dids=[],
            rolling_mean=None,
            rolling_variance=None,
            dispersion_index=None,
        )
        assert result_normal.is_anomaly == 0

    def test_create_scored_result_with_dispersion_fields(self) -> None:
        now = datetime(2026, 3, 20, 12, 0, 0)
        result = ScoredResult(
            run_timestamp=now,
            granularity='daily',
            pds_host='example.com',
            observed_count=42,
            distinct_accounts=40,
            expected_lambda=3.5,
            p_value=0.001,
            q_value=0.005,
            is_anomaly=1,
            baseline_source='entity',
            baseline_days_available=7,
            sample_dids=['did:plc:abc1'],
            rolling_mean=3.5,
            rolling_variance=0.75,
            dispersion_index=1.5,
        )
        assert result.rolling_mean == 3.5
        assert result.rolling_variance == 0.75
        assert result.dispersion_index == 1.5
        assert result.q_value == 0.005
