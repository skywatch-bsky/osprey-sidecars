# pattern: Functional Core
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from signup_anomaly.db import AggregatedRow, ScoredResult, SignupAnomalyDb


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


class TestInsertColumnList:
    def test_insert_column_list_with_q_values(self) -> None:
        """Verify the insert column list contains exactly 15 names with q_value immediately after p_value."""
        mock_client = MagicMock()
        db = SignupAnomalyDb.__new__(SignupAnomalyDb)
        db._client = mock_client

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
            sample_dids=['did1'],
            rolling_mean=3.5,
            rolling_variance=0.75,
            dispersion_index=1.5,
        )

        db.insert_results('test_table', [result])

        # Verify insert was called
        assert mock_client.insert.called
        call_args = mock_client.insert.call_args

        # Check column names (15 total)
        column_names = call_args[1]['column_names']
        assert len(column_names) == 15
        assert column_names == [
            'run_timestamp',
            'granularity',
            'pds_host',
            'observed_count',
            'distinct_accounts',
            'expected_lambda',
            'p_value',
            'q_value',
            'is_anomaly',
            'baseline_source',
            'baseline_days_available',
            'sample_dids',
            'rolling_mean',
            'rolling_variance',
            'dispersion_index',
        ]

        # Verify data matches column order
        data = call_args[1]['data']
        assert len(data) == 1
        row_data = data[0]
        assert row_data[0] == now  # run_timestamp
        assert row_data[1] == 'daily'  # granularity
        assert row_data[2] == 'example.com'  # pds_host
        assert row_data[6] == 0.001  # p_value
        assert row_data[7] == 0.005  # q_value


class TestFetchAggregatedRows:
    def test_fetch_mapping_maps_columns_in_correct_order(self) -> None:
        """Verify that fetch_aggregated_rows maps tuple columns in exact SELECT order.

        SELECT order from queries.py daily/hourly: pds_host, observed_count, distinct_accounts,
        rolling_median, rolling_mean, rolling_variance, dispersion_index, baseline_days_available,
        sample_dids, population_median_lambda, population_dispersion_index.
        """
        db = SignupAnomalyDb.__new__(SignupAnomalyDb)
        db._client = MagicMock()

        stub_row = (
            'example.com',  # [0] pds_host
            42,  # [1] observed_count
            40,  # [2] distinct_accounts
            3.2,  # [3] rolling_median
            3.5,  # [4] rolling_mean
            0.75,  # [5] rolling_variance
            1.5,  # [6] dispersion_index
            7,  # [7] baseline_days_available
            ['did:plc:abc1', 'did:plc:abc2'],  # [8] sample_dids
            2.1,  # [9] population_median_lambda
            1.2,  # [10] population_dispersion_index
        )

        mock_result = MagicMock()
        mock_result.result_rows = [stub_row]
        db._client.query.return_value = mock_result

        rows = db.fetch_aggregated_rows('SELECT ...')

        assert len(rows) == 1
        row = rows[0]
        assert row.pds_host == 'example.com'
        assert row.observed_count == 42
        assert row.distinct_accounts == 40
        assert row.rolling_median == 3.2
        assert row.rolling_mean == 3.5
        assert row.rolling_variance == 0.75
        assert row.dispersion_index == 1.5
        assert row.baseline_days_available == 7
        assert row.sample_dids == ['did:plc:abc1', 'did:plc:abc2']
        assert row.population_median_lambda == 2.1
        assert row.population_dispersion_index == 1.2

    def test_fetch_mapping_handles_none_values(self) -> None:
        """Verify that fetch correctly converts None values to None (not crashes on int(None))."""
        db = SignupAnomalyDb.__new__(SignupAnomalyDb)
        db._client = MagicMock()

        stub_row = (
            'example.com',  # pds_host
            10,  # observed_count
            10,  # distinct_accounts
            None,  # rolling_median (None)
            None,  # rolling_mean (None)
            None,  # rolling_variance (None)
            None,  # dispersion_index (None)
            5,  # baseline_days_available
            [],  # sample_dids (empty)
            None,  # population_median_lambda (None)
            None,  # population_dispersion_index (None)
        )

        mock_result = MagicMock()
        mock_result.result_rows = [stub_row]
        db._client.query.return_value = mock_result

        rows = db.fetch_aggregated_rows('SELECT ...')

        assert len(rows) == 1
        row = rows[0]
        assert row.rolling_median is None
        assert row.rolling_mean is None
        assert row.rolling_variance is None
        assert row.dispersion_index is None
        assert row.population_median_lambda is None
        assert row.population_dispersion_index is None
        assert row.sample_dids == []
        assert row.baseline_days_available == 5
