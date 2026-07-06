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
            rolling_volume_median=2.5,
            rolling_volume_mean=3.5,
            rolling_volume_variance=5.2,
            rolling_density_mean=0.78,
            rolling_density_variance=0.05,
            baseline_days_available=7,
            sample_dids=['did:plc:abc1', 'did:plc:abc2'],
            sample_urls=['https://example.com/post1', 'https://example.com/post2'],
            population_volume_median=2.1,
            population_volume_dispersion=1.8,
            population_density_median=0.65,
            population_density_variance=0.04,
        )
        assert row.domain == 'example.com'
        assert row.bucket_start == now
        assert row.total_shares == 42
        assert row.unique_sharers == 15
        assert row.sharer_density == 0.85
        assert row.rolling_volume_median == 2.5
        assert row.rolling_volume_mean == 3.5
        assert row.rolling_volume_variance == 5.2
        assert row.rolling_density_mean == 0.78
        assert row.rolling_density_variance == 0.05
        assert row.baseline_days_available == 7
        assert row.sample_dids == ['did:plc:abc1', 'did:plc:abc2']
        assert row.sample_urls == ['https://example.com/post1', 'https://example.com/post2']
        assert row.population_volume_median == 2.1
        assert row.population_volume_dispersion == 1.8
        assert row.population_density_median == 0.65
        assert row.population_density_variance == 0.04

    def test_create_with_none_optional_fields(self) -> None:
        now = datetime(2026, 3, 20, 12, 0, 0)
        row = AggregatedRow(
            domain='example.com',
            bucket_start=now,
            total_shares=1,
            unique_sharers=1,
            sharer_density=1.0,
            rolling_volume_median=None,
            rolling_volume_mean=None,
            rolling_volume_variance=None,
            rolling_density_mean=None,
            rolling_density_variance=None,
            baseline_days_available=0,
            sample_dids=[],
            sample_urls=[],
            population_volume_median=None,
            population_volume_dispersion=None,
            population_density_median=None,
            population_density_variance=None,
        )
        assert row.rolling_volume_median is None
        assert row.rolling_volume_mean is None
        assert row.rolling_volume_variance is None
        assert row.rolling_density_mean is None
        assert row.rolling_density_variance is None
        assert row.population_volume_median is None
        assert row.population_volume_dispersion is None
        assert row.population_density_median is None
        assert row.population_density_variance is None

    def test_is_frozen(self) -> None:
        now = datetime(2026, 3, 20, 12, 0, 0)
        row = AggregatedRow(
            domain='example.com',
            bucket_start=now,
            total_shares=42,
            unique_sharers=15,
            sharer_density=0.85,
            rolling_volume_median=2.5,
            rolling_volume_mean=3.5,
            rolling_volume_variance=5.2,
            rolling_density_mean=0.78,
            rolling_density_variance=0.05,
            baseline_days_available=7,
            sample_dids=['did:plc:abc1'],
            sample_urls=['https://example.com/post1'],
            population_volume_median=2.1,
            population_volume_dispersion=1.8,
            population_density_median=0.65,
            population_density_variance=0.04,
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
            rolling_volume_median=None,
            rolling_volume_mean=None,
            rolling_volume_variance=None,
            rolling_density_mean=None,
            rolling_density_variance=None,
            baseline_days_available=0,
            sample_dids=[],
            sample_urls=[],
            population_volume_median=None,
            population_volume_dispersion=None,
            population_density_median=None,
            population_density_variance=None,
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
            rolling_volume_median=2.5,
            rolling_volume_variance=5.2,
            rolling_density_mean=0.78,
            rolling_density_variance=0.05,
            volume_p_value=0.001,
            volume_q_value=0.002,
            density_p_value=0.005,
            density_q_value=0.008,
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
        assert result.rolling_volume_median == 2.5
        assert result.rolling_volume_variance == 5.2
        assert result.rolling_density_mean == 0.78
        assert result.rolling_density_variance == 0.05
        assert result.volume_p_value == 0.001
        assert result.volume_q_value == 0.002
        assert result.density_p_value == 0.005
        assert result.density_q_value == 0.008
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
            rolling_volume_median=1.5,
            rolling_volume_variance=3.0,
            rolling_density_mean=0.4,
            rolling_density_variance=0.03,
            volume_p_value=0.05,
            volume_q_value=0.06,
            density_p_value=0.10,
            density_q_value=0.12,
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
        assert result.volume_q_value == 0.06
        assert result.density_q_value == 0.12

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
            rolling_volume_median=2.5,
            rolling_volume_variance=5.2,
            rolling_density_mean=0.78,
            rolling_density_variance=0.05,
            volume_p_value=0.001,
            volume_q_value=0.002,
            density_p_value=0.005,
            density_q_value=0.008,
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
            rolling_volume_median=None,
            rolling_volume_variance=None,
            rolling_density_mean=None,
            rolling_density_variance=None,
            volume_p_value=0.9,
            volume_q_value=0.95,
            density_p_value=0.9,
            density_q_value=0.95,
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
            rolling_volume_median=1.5,
            rolling_volume_variance=3.0,
            rolling_density_mean=0.4,
            rolling_density_variance=0.03,
            volume_p_value=0.001,
            volume_q_value=0.002,
            density_p_value=0.005,
            density_q_value=0.008,
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
            rolling_volume_median=2.0,
            rolling_volume_variance=2.0,
            rolling_density_mean=1.0,
            rolling_density_variance=0.0,
            volume_p_value=0.9,
            volume_q_value=0.92,
            density_p_value=0.9,
            density_q_value=0.92,
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
            rolling_volume_median=1.0,
            rolling_volume_variance=1.0,
            rolling_density_mean=0.5,
            rolling_density_variance=0.05,
            volume_p_value=0.5,
            volume_q_value=0.55,
            density_p_value=0.5,
            density_q_value=0.55,
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
            rolling_volume_median=1.0,
            rolling_volume_variance=1.0,
            rolling_density_mean=0.5,
            rolling_density_variance=0.05,
            volume_p_value=0.5,
            volume_q_value=0.55,
            density_p_value=0.5,
            density_q_value=0.55,
            is_anomaly=0,
            baseline_source='entity',
            baseline_days_available=7,
            sample_dids=['did:plc:xyz'],
            sample_urls=['https://normal.com/post'],
            on_watchlist=0,
        )
        assert result_not_on_list.on_watchlist == 0


class TestInsertColumnList:
    def test_insert_column_list_with_q_values(self) -> None:
        """Verify the insert column list contains exactly 23 names with q-values in correct positions."""
        mock_client = MagicMock()
        db = UrlOverdispersionDb.__new__(UrlOverdispersionDb)
        db._client = mock_client

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
            rolling_volume_median=2.5,
            rolling_volume_variance=5.2,
            rolling_density_mean=0.78,
            rolling_density_variance=0.05,
            volume_p_value=0.001,
            volume_q_value=0.002,
            density_p_value=0.005,
            density_q_value=0.008,
            is_anomaly=1,
            baseline_source='entity',
            baseline_days_available=7,
            sample_dids=['did1'],
            sample_urls=['url1'],
            on_watchlist=1,
        )

        db.insert_results('test_table', [result])

        # Verify insert was called
        assert mock_client.insert.called
        call_args = mock_client.insert.call_args

        # Check column names (23 total)
        column_names = call_args[1]['column_names']
        assert len(column_names) == 23
        assert column_names == [
            'run_timestamp',
            'granularity',
            'domain',
            'bucket_start',
            'total_shares',
            'unique_sharers',
            'sharer_density',
            'expected_volume_lambda',
            'expected_density_lambda',
            'rolling_volume_median',
            'rolling_volume_variance',
            'rolling_density_mean',
            'rolling_density_variance',
            'volume_p_value',
            'volume_q_value',
            'density_p_value',
            'density_q_value',
            'is_anomaly',
            'baseline_source',
            'baseline_days_available',
            'sample_dids',
            'sample_urls',
            'on_watchlist',
        ]

        # Verify data matches column order
        data = call_args[1]['data']
        assert len(data) == 1
        row_data = data[0]
        assert row_data[0] == now  # run_timestamp
        assert row_data[1] == 'daily'  # granularity
        assert row_data[2] == 'example.com'  # domain
        assert row_data[13] == 0.001  # volume_p_value
        assert row_data[14] == 0.002  # volume_q_value
        assert row_data[15] == 0.005  # density_p_value
        assert row_data[16] == 0.008  # density_q_value


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
                2.5,
                3.5,
                5.2,
                0.78,
                0.05,
                7,
                ['did:plc:abc1'],
                ['https://example.com/post1'],
                2.1,
                1.8,
                0.65,
                0.04,
            ),
        ]

        db = UrlOverdispersionDb.__new__(UrlOverdispersionDb)
        db._client = mock_client

        rows = db.fetch_aggregated_rows('SELECT 1')
        assert len(rows) == 1
        assert isinstance(rows[0].bucket_start, datetime)
        assert rows[0].bucket_start == datetime(2026, 3, 20)

    def test_fetch_mapping_order_positional_tuple(self) -> None:
        """Fetch mapping must match exact SELECT order: domain, bucket_start, total_shares, unique_sharers,
        sharer_density, rolling_volume_median, rolling_volume_mean, rolling_volume_variance,
        rolling_density_mean, rolling_density_variance, baseline_days_available, sample_dids,
        sample_urls, population_volume_median, population_volume_dispersion, population_density_median,
        population_density_variance."""
        mock_client = MagicMock()
        mock_client.query.return_value.result_rows = [
            (
                'test.com',
                date(2026, 3, 21),
                100,
                50,
                0.5,
                3.0,
                3.5,
                4.2,
                0.5,
                0.06,
                10,
                ['did1', 'did2'],
                ['url1', 'url2'],
                2.8,
                1.9,
                0.48,
                0.05,
            ),
        ]

        db = UrlOverdispersionDb.__new__(UrlOverdispersionDb)
        db._client = mock_client

        rows = db.fetch_aggregated_rows('SELECT 1')
        assert len(rows) == 1
        r = rows[0]
        assert r.domain == 'test.com'
        assert r.bucket_start == datetime(2026, 3, 21)
        assert r.total_shares == 100
        assert r.unique_sharers == 50
        assert r.sharer_density == 0.5
        assert r.rolling_volume_median == 3.0
        assert r.rolling_volume_mean == 3.5
        assert r.rolling_volume_variance == 4.2
        assert r.rolling_density_mean == 0.5
        assert r.rolling_density_variance == 0.06
        assert r.baseline_days_available == 10
        assert r.sample_dids == ['did1', 'did2']
        assert r.sample_urls == ['url1', 'url2']
        assert r.population_volume_median == 2.8
        assert r.population_volume_dispersion == 1.9
        assert r.population_density_median == 0.48
        assert r.population_density_variance == 0.05
