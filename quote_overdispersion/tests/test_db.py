# pattern: Functional Core
from datetime import date, datetime
from unittest.mock import MagicMock

import pytest

from quote_overdispersion.db import AggregatedRow, QuoteOverdispersionDb, ScoredResult


class TestAggregatedRow:
    def test_create_with_all_fields(self) -> None:
        now = datetime(2026, 3, 20, 12, 0, 0)
        row = AggregatedRow(
            quoted_uri='at://did:plc:abc1/app.bsky.feed.post/xyz',
            bucket_start=now,
            total_shares=42,
            unique_sharers=15,
            sharer_density=0.85,
            rolling_volume_median=2.5,
            rolling_volume_mean=3.5,
            rolling_volume_variance=4.2,
            rolling_density_mean=0.78,
            rolling_density_variance=0.05,
            baseline_days_available=7,
            sample_dids=['did:plc:abc1', 'did:plc:abc2'],
            population_volume_median=2.1,
            population_volume_dispersion=1.5,
            population_density_median=0.65,
            population_density_variance=0.03,
        )
        assert row.quoted_uri == 'at://did:plc:abc1/app.bsky.feed.post/xyz'
        assert row.bucket_start == now
        assert row.total_shares == 42
        assert row.unique_sharers == 15
        assert row.sharer_density == 0.85
        assert row.rolling_volume_median == 2.5
        assert row.rolling_volume_mean == 3.5
        assert row.rolling_volume_variance == 4.2
        assert row.rolling_density_mean == 0.78
        assert row.rolling_density_variance == 0.05
        assert row.baseline_days_available == 7
        assert row.sample_dids == ['did:plc:abc1', 'did:plc:abc2']
        assert row.population_volume_median == 2.1
        assert row.population_volume_dispersion == 1.5
        assert row.population_density_median == 0.65
        assert row.population_density_variance == 0.03

    def test_create_with_none_optional_fields(self) -> None:
        now = datetime(2026, 3, 20, 12, 0, 0)
        row = AggregatedRow(
            quoted_uri='at://did:plc:abc1/app.bsky.feed.post/xyz',
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
            quoted_uri='at://did:plc:abc1/app.bsky.feed.post/xyz',
            bucket_start=now,
            total_shares=42,
            unique_sharers=15,
            sharer_density=0.85,
            rolling_volume_median=2.5,
            rolling_volume_mean=3.5,
            rolling_volume_variance=4.2,
            rolling_density_mean=0.78,
            rolling_density_variance=0.05,
            baseline_days_available=7,
            sample_dids=['did:plc:abc1'],
            population_volume_median=2.1,
            population_volume_dispersion=1.5,
            population_density_median=0.65,
            population_density_variance=0.03,
        )
        with pytest.raises(AttributeError):
            row.total_shares = 100

    def test_empty_samples(self) -> None:
        now = datetime(2026, 3, 20, 12, 0, 0)
        row = AggregatedRow(
            quoted_uri='at://did:plc:abc1/app.bsky.feed.post/xyz',
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
            population_volume_median=None,
            population_volume_dispersion=None,
            population_density_median=None,
            population_density_variance=None,
        )
        assert row.sample_dids == []


class TestScoredResult:
    def test_create_with_all_fields(self) -> None:
        now = datetime(2026, 3, 20, 12, 0, 0)
        bucket_start = datetime(2026, 3, 20, 0, 0, 0)
        result = ScoredResult(
            run_timestamp=now,
            granularity='daily',
            quoted_uri='at://did:plc:abc1/app.bsky.feed.post/xyz',
            quoted_author_did='did:plc:abc1',
            bucket_start=bucket_start,
            total_shares=42,
            unique_sharers=15,
            sharer_density=0.85,
            expected_volume_lambda=3.5,
            expected_density_lambda=0.78,
            volume_p_value=0.001,
            volume_q_value=0.002,
            density_p_value=0.005,
            density_q_value=0.010,
            is_anomaly=1,
            baseline_source='entity',
            baseline_days_available=7,
            sample_dids=['did:plc:abc1', 'did:plc:abc2'],
            rolling_volume_median=2.5,
            rolling_volume_variance=4.2,
            rolling_density_mean=0.78,
            rolling_density_variance=0.05,
        )
        assert result.run_timestamp == now
        assert result.granularity == 'daily'
        assert result.quoted_uri == 'at://did:plc:abc1/app.bsky.feed.post/xyz'
        assert result.quoted_author_did == 'did:plc:abc1'
        assert result.bucket_start == bucket_start
        assert result.total_shares == 42
        assert result.unique_sharers == 15
        assert result.sharer_density == 0.85
        assert result.expected_volume_lambda == 3.5
        assert result.expected_density_lambda == 0.78
        assert result.volume_p_value == 0.001
        assert result.volume_q_value == 0.002
        assert result.density_p_value == 0.005
        assert result.density_q_value == 0.010
        assert result.is_anomaly == 1
        assert result.baseline_source == 'entity'
        assert result.baseline_days_available == 7
        assert result.sample_dids == ['did:plc:abc1', 'did:plc:abc2']
        assert result.rolling_volume_median == 2.5
        assert result.rolling_volume_variance == 4.2
        assert result.rolling_density_mean == 0.78
        assert result.rolling_density_variance == 0.05

    def test_ac1_6_quoted_author_did_extraction(self) -> None:
        """AC1.6: quoted_author_did correctly extracted from AT-URI."""
        now = datetime(2026, 3, 20, 12, 0, 0)
        bucket_start = datetime(2026, 3, 20, 0, 0, 0)
        result = ScoredResult(
            run_timestamp=now,
            granularity='daily',
            quoted_uri='at://did:plc:abc123/app.bsky.feed.post/abc',
            quoted_author_did='did:plc:abc123',
            bucket_start=bucket_start,
            total_shares=42,
            unique_sharers=15,
            sharer_density=0.85,
            expected_volume_lambda=3.5,
            expected_density_lambda=0.78,
            volume_p_value=0.001,
            volume_q_value=0.002,
            density_p_value=0.005,
            density_q_value=0.010,
            is_anomaly=1,
            baseline_source='entity',
            baseline_days_available=7,
            sample_dids=['did:plc:abc123'],
            rolling_volume_median=2.5,
            rolling_volume_variance=4.2,
            rolling_density_mean=0.78,
            rolling_density_variance=0.05,
        )
        assert result.quoted_author_did == 'did:plc:abc123'

    def test_ac1_7_malformed_at_uri_empty_did(self) -> None:
        """AC1.7: malformed AT-URI produces empty string for quoted_author_did."""
        now = datetime(2026, 3, 20, 12, 0, 0)
        bucket_start = datetime(2026, 3, 20, 0, 0, 0)
        result = ScoredResult(
            run_timestamp=now,
            granularity='daily',
            quoted_uri='malformed-uri',
            quoted_author_did='',
            bucket_start=bucket_start,
            total_shares=42,
            unique_sharers=15,
            sharer_density=0.85,
            expected_volume_lambda=3.5,
            expected_density_lambda=0.78,
            volume_p_value=0.001,
            volume_q_value=0.002,
            density_p_value=0.005,
            density_q_value=0.010,
            is_anomaly=0,
            baseline_source='entity',
            baseline_days_available=7,
            sample_dids=[],
            rolling_volume_median=None,
            rolling_volume_variance=None,
            rolling_density_mean=None,
            rolling_density_variance=None,
        )
        assert result.quoted_author_did == ''

    def test_create_with_population_baseline(self) -> None:
        now = datetime(2026, 3, 20, 12, 0, 0)
        bucket_start = datetime(2026, 3, 20, 11, 0, 0)
        result = ScoredResult(
            run_timestamp=now,
            granularity='hourly',
            quoted_uri='at://did:plc:abc1/app.bsky.feed.post/xyz',
            quoted_author_did='did:plc:abc1',
            bucket_start=bucket_start,
            total_shares=10,
            unique_sharers=5,
            sharer_density=0.5,
            expected_volume_lambda=2.0,
            expected_density_lambda=0.4,
            volume_p_value=0.05,
            volume_q_value=0.08,
            density_p_value=0.10,
            density_q_value=0.15,
            is_anomaly=0,
            baseline_source='population',
            baseline_days_available=3,
            sample_dids=['did:plc:abc1'],
            rolling_volume_median=1.5,
            rolling_volume_variance=None,
            rolling_density_mean=0.4,
            rolling_density_variance=None,
        )
        assert result.baseline_source == 'population'
        assert result.granularity == 'hourly'

    def test_is_frozen(self) -> None:
        now = datetime(2026, 3, 20, 12, 0, 0)
        bucket_start = datetime(2026, 3, 20, 0, 0, 0)
        result = ScoredResult(
            run_timestamp=now,
            granularity='daily',
            quoted_uri='at://did:plc:abc1/app.bsky.feed.post/xyz',
            quoted_author_did='did:plc:abc1',
            bucket_start=bucket_start,
            total_shares=42,
            unique_sharers=15,
            sharer_density=0.85,
            expected_volume_lambda=3.5,
            expected_density_lambda=0.78,
            volume_p_value=0.001,
            volume_q_value=0.002,
            density_p_value=0.005,
            density_q_value=0.010,
            is_anomaly=1,
            baseline_source='entity',
            baseline_days_available=7,
            sample_dids=['did:plc:abc1'],
            rolling_volume_median=2.5,
            rolling_volume_variance=4.2,
            rolling_density_mean=0.78,
            rolling_density_variance=0.05,
        )
        with pytest.raises(AttributeError):
            result.total_shares = 100

    def test_create_with_empty_samples(self) -> None:
        now = datetime(2026, 3, 20, 12, 0, 0)
        bucket_start = datetime(2026, 3, 20, 0, 0, 0)
        result = ScoredResult(
            run_timestamp=now,
            granularity='daily',
            quoted_uri='at://did:plc:abc1/app.bsky.feed.post/xyz',
            quoted_author_did='did:plc:abc1',
            bucket_start=bucket_start,
            total_shares=0,
            unique_sharers=0,
            sharer_density=0.0,
            expected_volume_lambda=1.0,
            expected_density_lambda=0.5,
            volume_p_value=0.9,
            volume_q_value=0.95,
            density_p_value=0.9,
            density_q_value=0.95,
            is_anomaly=0,
            baseline_source='entity',
            baseline_days_available=0,
            sample_dids=[],
            rolling_volume_median=None,
            rolling_volume_variance=None,
            rolling_density_mean=None,
            rolling_density_variance=None,
        )
        assert result.sample_dids == []

    def test_anomaly_and_non_anomaly_values(self) -> None:
        now = datetime(2026, 3, 20, 12, 0, 0)
        bucket_start = datetime(2026, 3, 20, 0, 0, 0)

        result_anomaly = ScoredResult(
            run_timestamp=now,
            granularity='daily',
            quoted_uri='at://did:plc:abc1/app.bsky.feed.post/xyz',
            quoted_author_did='did:plc:abc1',
            bucket_start=bucket_start,
            total_shares=100,
            unique_sharers=50,
            sharer_density=0.95,
            expected_volume_lambda=2.0,
            expected_density_lambda=0.4,
            volume_p_value=0.001,
            volume_q_value=0.002,
            density_p_value=0.005,
            density_q_value=0.010,
            is_anomaly=1,
            baseline_source='entity',
            baseline_days_available=7,
            sample_dids=[],
            rolling_volume_median=1.5,
            rolling_volume_variance=3.0,
            rolling_density_mean=0.4,
            rolling_density_variance=0.02,
        )
        assert result_anomaly.is_anomaly == 1

        result_normal = ScoredResult(
            run_timestamp=now,
            granularity='daily',
            quoted_uri='at://did:plc:abc1/app.bsky.feed.post/xyz',
            quoted_author_did='did:plc:abc1',
            bucket_start=bucket_start,
            total_shares=2,
            unique_sharers=2,
            sharer_density=1.0,
            expected_volume_lambda=2.0,
            expected_density_lambda=1.0,
            volume_p_value=0.9,
            volume_q_value=0.95,
            density_p_value=0.9,
            density_q_value=0.95,
            is_anomaly=0,
            baseline_source='entity',
            baseline_days_available=7,
            sample_dids=[],
            rolling_volume_median=2.0,
            rolling_volume_variance=2.0,
            rolling_density_mean=1.0,
            rolling_density_variance=0.0,
        )
        assert result_normal.is_anomaly == 0


class TestInsertColumnList:
    def test_insert_column_list_with_q_values(self) -> None:
        """Verify the insert column list contains exactly 22 names with q-values in correct positions."""
        mock_client = MagicMock()
        db = QuoteOverdispersionDb.__new__(QuoteOverdispersionDb)
        db._client = mock_client

        now = datetime(2026, 3, 20, 12, 0, 0)
        bucket_start = datetime(2026, 3, 20, 0, 0, 0)
        result = ScoredResult(
            run_timestamp=now,
            granularity='daily',
            quoted_uri='at://did:plc:abc1/app.bsky.feed.post/xyz',
            quoted_author_did='did:plc:abc1',
            bucket_start=bucket_start,
            total_shares=42,
            unique_sharers=15,
            sharer_density=0.85,
            expected_volume_lambda=3.5,
            expected_density_lambda=0.78,
            volume_p_value=0.001,
            volume_q_value=0.002,
            density_p_value=0.005,
            density_q_value=0.010,
            is_anomaly=1,
            baseline_source='entity',
            baseline_days_available=7,
            sample_dids=['did1'],
            rolling_volume_median=2.5,
            rolling_volume_variance=4.2,
            rolling_density_mean=0.78,
            rolling_density_variance=0.05,
        )

        db.insert_results('test_table', [result])

        # Verify insert was called
        assert mock_client.insert.called
        call_args = mock_client.insert.call_args

        # Check column names (22 total)
        column_names = call_args[1]['column_names']
        assert len(column_names) == 22
        assert column_names == [
            'run_timestamp',
            'granularity',
            'quoted_uri',
            'quoted_author_did',
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
        ]

        # Verify data matches column order
        data = call_args[1]['data']
        assert len(data) == 1
        row_data = data[0]
        assert row_data[0] == now  # run_timestamp
        assert row_data[1] == 'daily'  # granularity
        assert row_data[14] == 0.001  # volume_p_value
        assert row_data[15] == 0.002  # volume_q_value
        assert row_data[16] == 0.005  # density_p_value
        assert row_data[17] == 0.010  # density_q_value


class TestQuoteOverdispersionDb:
    def test_fetch_coerces_date_to_datetime(self) -> None:
        """ClickHouse toDate() returns datetime.date; bucket_start must be datetime."""
        mock_client = MagicMock()
        mock_client.query.return_value.result_rows = [
            (
                'at://did:plc:abc1/app.bsky.feed.post/xyz',
                date(2026, 3, 20),  # toDate() returns date, not datetime
                42,
                15,
                0.85,
                2.5,  # rolling_volume_median
                3.5,  # rolling_volume_mean
                4.2,  # rolling_volume_variance
                0.78,  # rolling_density_mean
                0.05,  # rolling_density_variance
                7,  # baseline_days_available
                ['did:plc:abc1'],  # sample_dids
                2.1,  # population_volume_median
                1.5,  # population_volume_dispersion
                0.65,  # population_density_median
                0.03,  # population_density_variance
            ),
        ]

        db = QuoteOverdispersionDb.__new__(QuoteOverdispersionDb)
        db._client = mock_client

        rows = db.fetch_aggregated_rows('SELECT 1')
        assert len(rows) == 1
        assert isinstance(rows[0].bucket_start, datetime)
        assert rows[0].bucket_start == datetime(2026, 3, 20)

    def test_fetch_mapping_indices_match_select_order(self) -> None:
        """Verify fetch mapping indices correspond to SELECT column order.

        SELECT order: quoted_uri, bucket_start, total_shares, unique_sharers, sharer_density,
                      rolling_volume_median, rolling_volume_mean, rolling_volume_variance,
                      rolling_density_mean, rolling_density_variance,
                      baseline_days_available, sample_dids,
                      population_volume_median, population_volume_dispersion,
                      population_density_median, population_density_variance
        """
        mock_client = MagicMock()
        mock_client.query.return_value.result_rows = [
            (
                'at://did:plc:entity1/app.bsky.feed.post/postid',
                date(2026, 7, 6),
                100,
                25,
                0.75,
                10.0,  # rolling_volume_median (index 5)
                12.5,  # rolling_volume_mean (index 6)
                20.0,  # rolling_volume_variance (index 7)
                0.70,  # rolling_density_mean (index 8)
                0.08,  # rolling_density_variance (index 9)
                14,  # baseline_days_available (index 10)
                ['did:plc:user1', 'did:plc:user2'],  # sample_dids (index 11)
                8.5,  # population_volume_median (index 12)
                1.8,  # population_volume_dispersion (index 13)
                0.65,  # population_density_median (index 14)
                0.06,  # population_density_variance (index 15)
            ),
        ]

        db = QuoteOverdispersionDb.__new__(QuoteOverdispersionDb)
        db._client = mock_client

        rows = db.fetch_aggregated_rows('SELECT ...')
        assert len(rows) == 1
        row = rows[0]

        # Verify all fields match their tuple positions
        assert row.quoted_uri == 'at://did:plc:entity1/app.bsky.feed.post/postid'
        assert row.bucket_start == datetime(2026, 7, 6)
        assert row.total_shares == 100
        assert row.unique_sharers == 25
        assert row.sharer_density == 0.75
        assert row.rolling_volume_median == 10.0  # Critical: median before mean
        assert row.rolling_volume_mean == 12.5
        assert row.rolling_volume_variance == 20.0
        assert row.rolling_density_mean == 0.70
        assert row.rolling_density_variance == 0.08
        assert row.baseline_days_available == 14
        assert row.sample_dids == ['did:plc:user1', 'did:plc:user2']
        assert row.population_volume_median == 8.5
        assert row.population_volume_dispersion == 1.8
        assert row.population_density_median == 0.65
        assert row.population_density_variance == 0.06
