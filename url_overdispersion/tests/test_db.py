# pattern: Functional Core
from datetime import date, datetime
from unittest.mock import MagicMock

import pytest

from url_overdispersion.db import AggregatedRow, ScoredResult, UrlOverdispersionDb


class TestAggregatedRow:
    def test_create_with_all_fields(self) -> None:
        now = datetime(2026, 3, 20, 12, 0, 0)
        row = AggregatedRow(
            domain='example.com',
            bucket_start=now,
            total_shares=42,
            unique_sharers=15,
            sharer_density=0.85,
            rolling_volume_mean=3.5,
            rolling_density_mean=0.78,
            baseline_days_available=7,
            sample_dids=['did:plc:abc1', 'did:plc:abc2'],
            sample_urls=['https://example.com/post1', 'https://example.com/post2'],
            population_volume_median=2.1,
            population_density_median=0.65,
        )
        assert row.domain == 'example.com'
        assert row.bucket_start == now
        assert row.total_shares == 42
        assert row.unique_sharers == 15
        assert row.sharer_density == 0.85
        assert row.rolling_volume_mean == 3.5
        assert row.rolling_density_mean == 0.78
        assert row.baseline_days_available == 7
        assert row.sample_dids == ['did:plc:abc1', 'did:plc:abc2']
        assert row.sample_urls == ['https://example.com/post1', 'https://example.com/post2']
        assert row.population_volume_median == 2.1
        assert row.population_density_median == 0.65

    def test_create_with_none_optional_fields(self) -> None:
        now = datetime(2026, 3, 20, 12, 0, 0)
        row = AggregatedRow(
            domain='example.com',
            bucket_start=now,
            total_shares=1,
            unique_sharers=1,
            sharer_density=1.0,
            rolling_volume_mean=None,
            rolling_density_mean=None,
            baseline_days_available=0,
            sample_dids=[],
            sample_urls=[],
            population_volume_median=None,
            population_density_median=None,
        )
        assert row.rolling_volume_mean is None
        assert row.rolling_density_mean is None
        assert row.population_volume_median is None
        assert row.population_density_median is None

    def test_is_frozen(self) -> None:
        now = datetime(2026, 3, 20, 12, 0, 0)
        row = AggregatedRow(
            domain='example.com',
            bucket_start=now,
            total_shares=42,
            unique_sharers=15,
            sharer_density=0.85,
            rolling_volume_mean=3.5,
            rolling_density_mean=0.78,
            baseline_days_available=7,
            sample_dids=['did:plc:abc1'],
            sample_urls=['https://example.com/post1'],
            population_volume_median=2.1,
            population_density_median=0.65,
        )
        with pytest.raises(AttributeError):
            row.total_shares = 100

    def test_empty_samples(self) -> None:
        now = datetime(2026, 3, 20, 12, 0, 0)
        row = AggregatedRow(
            domain='example.com',
            bucket_start=now,
            total_shares=0,
            unique_sharers=0,
            sharer_density=0.0,
            rolling_volume_mean=None,
            rolling_density_mean=None,
            baseline_days_available=0,
            sample_dids=[],
            sample_urls=[],
            population_volume_median=None,
            population_density_median=None,
        )
        assert row.sample_dids == []
        assert row.sample_urls == []


class TestScoredResult:
    def test_create_with_all_fields(self) -> None:
        now = datetime(2026, 3, 20, 12, 0, 0)
        bucket_start = datetime(2026, 3, 20, 0, 0, 0)
        result = ScoredResult(
            run_timestamp=now,
            granularity='daily',
            domain='example.com',
            bucket_start=bucket_start,
            total_shares=42,
            unique_sharers=15,
            sharer_density=0.85,
            expected_volume_lambda=3.5,
            expected_density_lambda=0.78,
            volume_p_value=0.001,
            density_p_value=0.005,
            is_anomaly=1,
            baseline_source='entity',
            baseline_days_available=7,
            sample_dids=['did:plc:abc1', 'did:plc:abc2'],
            sample_urls=['https://example.com/post1', 'https://example.com/post2'],
            on_watchlist=1,
        )
        assert result.run_timestamp == now
        assert result.granularity == 'daily'
        assert result.domain == 'example.com'
        assert result.bucket_start == bucket_start
        assert result.total_shares == 42
        assert result.unique_sharers == 15
        assert result.sharer_density == 0.85
        assert result.expected_volume_lambda == 3.5
        assert result.expected_density_lambda == 0.78
        assert result.volume_p_value == 0.001
        assert result.density_p_value == 0.005
        assert result.is_anomaly == 1
        assert result.baseline_source == 'entity'
        assert result.baseline_days_available == 7
        assert result.sample_dids == ['did:plc:abc1', 'did:plc:abc2']
        assert result.sample_urls == ['https://example.com/post1', 'https://example.com/post2']
        assert result.on_watchlist == 1

    def test_create_with_population_baseline(self) -> None:
        now = datetime(2026, 3, 20, 12, 0, 0)
        bucket_start = datetime(2026, 3, 20, 11, 0, 0)
        result = ScoredResult(
            run_timestamp=now,
            granularity='hourly',
            domain='example.com',
            bucket_start=bucket_start,
            total_shares=10,
            unique_sharers=5,
            sharer_density=0.5,
            expected_volume_lambda=2.0,
            expected_density_lambda=0.4,
            volume_p_value=0.05,
            density_p_value=0.10,
            is_anomaly=0,
            baseline_source='population',
            baseline_days_available=3,
            sample_dids=['did:plc:abc1'],
            sample_urls=['https://example.com/post1'],
            on_watchlist=0,
        )
        assert result.baseline_source == 'population'
        assert result.granularity == 'hourly'
        assert result.on_watchlist == 0

    def test_is_frozen(self) -> None:
        now = datetime(2026, 3, 20, 12, 0, 0)
        bucket_start = datetime(2026, 3, 20, 0, 0, 0)
        result = ScoredResult(
            run_timestamp=now,
            granularity='daily',
            domain='example.com',
            bucket_start=bucket_start,
            total_shares=42,
            unique_sharers=15,
            sharer_density=0.85,
            expected_volume_lambda=3.5,
            expected_density_lambda=0.78,
            volume_p_value=0.001,
            density_p_value=0.005,
            is_anomaly=1,
            baseline_source='entity',
            baseline_days_available=7,
            sample_dids=['did:plc:abc1'],
            sample_urls=['https://example.com/post1'],
            on_watchlist=1,
        )
        with pytest.raises(AttributeError):
            result.total_shares = 100

    def test_create_with_empty_samples(self) -> None:
        now = datetime(2026, 3, 20, 12, 0, 0)
        bucket_start = datetime(2026, 3, 20, 0, 0, 0)
        result = ScoredResult(
            run_timestamp=now,
            granularity='daily',
            domain='example.com',
            bucket_start=bucket_start,
            total_shares=0,
            unique_sharers=0,
            sharer_density=0.0,
            expected_volume_lambda=1.0,
            expected_density_lambda=0.5,
            volume_p_value=0.9,
            density_p_value=0.9,
            is_anomaly=0,
            baseline_source='entity',
            baseline_days_available=0,
            sample_dids=[],
            sample_urls=[],
            on_watchlist=0,
        )
        assert result.sample_dids == []
        assert result.sample_urls == []

    def test_anomaly_and_non_anomaly_values(self) -> None:
        now = datetime(2026, 3, 20, 12, 0, 0)
        bucket_start = datetime(2026, 3, 20, 0, 0, 0)

        result_anomaly = ScoredResult(
            run_timestamp=now,
            granularity='daily',
            domain='example.com',
            bucket_start=bucket_start,
            total_shares=100,
            unique_sharers=50,
            sharer_density=0.95,
            expected_volume_lambda=2.0,
            expected_density_lambda=0.4,
            volume_p_value=0.001,
            density_p_value=0.005,
            is_anomaly=1,
            baseline_source='entity',
            baseline_days_available=7,
            sample_dids=[],
            sample_urls=[],
            on_watchlist=0,
        )
        assert result_anomaly.is_anomaly == 1

        result_normal = ScoredResult(
            run_timestamp=now,
            granularity='daily',
            domain='example.com',
            bucket_start=bucket_start,
            total_shares=2,
            unique_sharers=2,
            sharer_density=1.0,
            expected_volume_lambda=2.0,
            expected_density_lambda=1.0,
            volume_p_value=0.9,
            density_p_value=0.9,
            is_anomaly=0,
            baseline_source='entity',
            baseline_days_available=7,
            sample_dids=[],
            sample_urls=[],
            on_watchlist=0,
        )
        assert result_normal.is_anomaly == 0

    def test_watchlist_enrolled_and_not_enrolled(self) -> None:
        now = datetime(2026, 3, 20, 12, 0, 0)
        bucket_start = datetime(2026, 3, 20, 0, 0, 0)

        result_on_list = ScoredResult(
            run_timestamp=now,
            granularity='daily',
            domain='evil.com',
            bucket_start=bucket_start,
            total_shares=10,
            unique_sharers=5,
            sharer_density=0.5,
            expected_volume_lambda=1.0,
            expected_density_lambda=0.5,
            volume_p_value=0.5,
            density_p_value=0.5,
            is_anomaly=0,
            baseline_source='entity',
            baseline_days_available=7,
            sample_dids=['did:plc:xyz'],
            sample_urls=['https://evil.com/post'],
            on_watchlist=1,
        )
        assert result_on_list.on_watchlist == 1

        result_not_on_list = ScoredResult(
            run_timestamp=now,
            granularity='daily',
            domain='normal.com',
            bucket_start=bucket_start,
            total_shares=10,
            unique_sharers=5,
            sharer_density=0.5,
            expected_volume_lambda=1.0,
            expected_density_lambda=0.5,
            volume_p_value=0.5,
            density_p_value=0.5,
            is_anomaly=0,
            baseline_source='entity',
            baseline_days_available=7,
            sample_dids=['did:plc:xyz'],
            sample_urls=['https://normal.com/post'],
            on_watchlist=0,
        )
        assert result_not_on_list.on_watchlist == 0


class TestUrlOverdispersionDb:
    def test_fetch_coerces_date_to_datetime(self) -> None:
        """ClickHouse toDate() returns datetime.date; bucket_start must be datetime."""
        mock_client = MagicMock()
        mock_client.query.return_value.result_rows = [
            (
                'example.com',
                date(2026, 3, 20),  # toDate() returns date, not datetime
                42,
                15,
                0.85,
                3.5,
                0.78,
                7,
                ['did:plc:abc1'],
                ['https://example.com/post1'],
                2.1,
                0.65,
            ),
        ]

        db = UrlOverdispersionDb.__new__(UrlOverdispersionDb)
        db._client = mock_client

        rows = db.fetch_aggregated_rows('SELECT 1')
        assert len(rows) == 1
        assert isinstance(rows[0].bucket_start, datetime)
        assert rows[0].bucket_start == datetime(2026, 3, 20)
