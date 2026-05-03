# pattern: Functional Core
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from account_entropy.config import ClickHouseConfig
from account_entropy.db import AccountActivityRow, AccountEntropyDb, ScoredResult


class TestAccountActivityRow:
    def test_create_with_all_fields(self) -> None:
        row = AccountActivityRow(
            user_id='did:plc:abc123',
            post_count=42,
            hourly_bins=[1, 2, 3, 4, 5],
            ordered_timestamps=[1000, 2000, 3000, 4000, 5000],
            sample_rkeys=['rkey1', 'rkey2'],
        )
        assert row.user_id == 'did:plc:abc123'
        assert row.post_count == 42
        assert row.hourly_bins == [1, 2, 3, 4, 5]
        assert row.ordered_timestamps == [1000, 2000, 3000, 4000, 5000]
        assert row.sample_rkeys == ['rkey1', 'rkey2']

    def test_create_with_empty_arrays(self) -> None:
        row = AccountActivityRow(
            user_id='did:plc:xyz789',
            post_count=0,
            hourly_bins=[],
            ordered_timestamps=[],
            sample_rkeys=[],
        )
        assert row.hourly_bins == []
        assert row.ordered_timestamps == []
        assert row.sample_rkeys == []

    def test_is_frozen(self) -> None:
        row = AccountActivityRow(
            user_id='did:plc:abc123',
            post_count=42,
            hourly_bins=[1, 2],
            ordered_timestamps=[1000, 2000],
            sample_rkeys=['rkey1'],
        )
        with pytest.raises(AttributeError):
            row.post_count = 100


class TestScoredResult:
    def test_create_with_all_fields(self) -> None:
        now = datetime(2026, 3, 20, 12, 0, 0)
        window_start = datetime(2026, 3, 13, 12, 0, 0)
        window_end = datetime(2026, 3, 20, 12, 0, 0)
        result = ScoredResult(
            run_timestamp=now,
            user_id='did:plc:abc123',
            window_start=window_start,
            window_end=window_end,
            post_count=42,
            hourly_entropy=3.5,
            interval_entropy=2.1,
            mean_interval_seconds=3600.0,
            stddev_interval_seconds=1200.0,
            is_bot_like=1,
            hourly_flag=1,
            interval_flag=0,
            sample_rkeys=['rkey1', 'rkey2'],
        )
        assert result.run_timestamp == now
        assert result.user_id == 'did:plc:abc123'
        assert result.window_start == window_start
        assert result.window_end == window_end
        assert result.post_count == 42
        assert result.hourly_entropy == 3.5
        assert result.interval_entropy == 2.1
        assert result.mean_interval_seconds == 3600.0
        assert result.stddev_interval_seconds == 1200.0
        assert result.is_bot_like == 1
        assert result.hourly_flag == 1
        assert result.interval_flag == 0
        assert result.sample_rkeys == ['rkey1', 'rkey2']

    def test_includes_mean_and_stddev_interval_seconds(self) -> None:
        now = datetime(2026, 3, 20, 12, 0, 0)
        window_start = datetime(2026, 3, 13, 12, 0, 0)
        window_end = datetime(2026, 3, 20, 12, 0, 0)
        result = ScoredResult(
            run_timestamp=now,
            user_id='did:plc:test',
            window_start=window_start,
            window_end=window_end,
            post_count=10,
            hourly_entropy=1.5,
            interval_entropy=0.8,
            mean_interval_seconds=7200.0,
            stddev_interval_seconds=3600.0,
            is_bot_like=0,
            hourly_flag=0,
            interval_flag=0,
            sample_rkeys=[],
        )
        assert result.mean_interval_seconds == 7200.0
        assert result.stddev_interval_seconds == 3600.0

    def test_includes_independent_flags(self) -> None:
        now = datetime(2026, 3, 20, 12, 0, 0)
        window_start = datetime(2026, 3, 13, 12, 0, 0)
        window_end = datetime(2026, 3, 20, 12, 0, 0)
        result = ScoredResult(
            run_timestamp=now,
            user_id='did:plc:test',
            window_start=window_start,
            window_end=window_end,
            post_count=5,
            hourly_entropy=1.0,
            interval_entropy=0.5,
            mean_interval_seconds=3600.0,
            stddev_interval_seconds=0.0,
            is_bot_like=0,
            hourly_flag=1,
            interval_flag=1,
            sample_rkeys=['rkey1'],
        )
        assert result.hourly_flag == 1
        assert result.interval_flag == 1

    def test_is_frozen(self) -> None:
        now = datetime(2026, 3, 20, 12, 0, 0)
        window_start = datetime(2026, 3, 13, 12, 0, 0)
        window_end = datetime(2026, 3, 20, 12, 0, 0)
        result = ScoredResult(
            run_timestamp=now,
            user_id='did:plc:abc123',
            window_start=window_start,
            window_end=window_end,
            post_count=42,
            hourly_entropy=3.5,
            interval_entropy=2.1,
            mean_interval_seconds=3600.0,
            stddev_interval_seconds=1200.0,
            is_bot_like=1,
            hourly_flag=1,
            interval_flag=0,
            sample_rkeys=['rkey1'],
        )
        with pytest.raises(AttributeError):
            result.post_count = 100


class TestAccountEntropyDb:
    @patch('account_entropy.db.clickhouse_connect.get_client')
    def test_fetch_account_rows_maps_columns_correctly(self, mock_get_client) -> None:
        config = ClickHouseConfig(
            host='localhost',
            port=8123,
            user='default',
            password='clickhouse',
            database='default',
        )
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        db = AccountEntropyDb(config)

        mock_result = Mock()
        mock_result.result_rows = [
            ('did:plc:user1', 42, [1, 2, 3], [1000, 2000, 3000], ['rkey1', 'rkey2']),
            ('did:plc:user2', 5, [], [], []),
            ('did:plc:user3', 10, [0, 5], [5000, 10000], ['rkey3']),
        ]
        mock_client.query.return_value = mock_result

        rows = db.fetch_account_rows('SELECT * FROM table')

        assert len(rows) == 3
        assert rows[0].user_id == 'did:plc:user1'
        assert rows[0].post_count == 42
        assert rows[0].hourly_bins == [1, 2, 3]
        assert rows[0].ordered_timestamps == [1000, 2000, 3000]
        assert rows[0].sample_rkeys == ['rkey1', 'rkey2']

        assert rows[1].user_id == 'did:plc:user2'
        assert rows[1].post_count == 5
        assert rows[1].hourly_bins == []
        assert rows[1].ordered_timestamps == []
        assert rows[1].sample_rkeys == []

        assert rows[2].user_id == 'did:plc:user3'
        assert rows[2].post_count == 10
        assert rows[2].hourly_bins == [0, 5]

    @patch('account_entropy.db.clickhouse_connect.get_client')
    def test_fetch_coerces_int_rkeys_to_strings(self, mock_get_client) -> None:
        """ClickHouse __action_id is numeric; sample_rkeys must be list[str]."""
        config = ClickHouseConfig(
            host='localhost', port=8123, user='default', password='clickhouse', database='default',
        )
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        db = AccountEntropyDb(config)

        mock_result = Mock()
        mock_result.result_rows = [
            ('did:plc:user1', 20, [1, 2, 3], [1000, 2000, 3000], [12345, 67890]),
        ]
        mock_client.query.return_value = mock_result

        rows = db.fetch_account_rows('SELECT 1')
        assert rows[0].sample_rkeys == ['12345', '67890']
        assert all(isinstance(k, str) for k in rows[0].sample_rkeys)

    @patch('account_entropy.db.clickhouse_connect.get_client')
    def test_fetch_account_rows_sets_max_execution_time(self, mock_get_client) -> None:
        config = ClickHouseConfig(
            host='localhost',
            port=8123,
            user='default',
            password='clickhouse',
            database='default',
        )
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        db = AccountEntropyDb(config)

        mock_result = Mock()
        mock_result.result_rows = []
        mock_client.query.return_value = mock_result

        db.fetch_account_rows('SELECT * FROM table')

        mock_client.query.assert_called_once()
        call_kwargs = mock_client.query.call_args[1]
        assert call_kwargs.get('settings') == {'max_execution_time': 120}

    @patch('account_entropy.db.clickhouse_connect.get_client')
    def test_insert_results_includes_all_columns(self, mock_get_client) -> None:
        config = ClickHouseConfig(
            host='localhost',
            port=8123,
            user='default',
            password='clickhouse',
            database='default',
        )
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        db = AccountEntropyDb(config)

        now = datetime(2026, 3, 20, 12, 0, 0)
        window_start = datetime(2026, 3, 13, 12, 0, 0)
        window_end = datetime(2026, 3, 20, 12, 0, 0)

        result = ScoredResult(
            run_timestamp=now,
            user_id='did:plc:user1',
            window_start=window_start,
            window_end=window_end,
            post_count=42,
            hourly_entropy=3.5,
            interval_entropy=2.1,
            mean_interval_seconds=3600.0,
            stddev_interval_seconds=1200.0,
            is_bot_like=1,
            hourly_flag=1,
            interval_flag=0,
            sample_rkeys=['rkey1', 'rkey2'],
        )

        db.insert_results('account_entropy_results', [result])

        mock_client.insert.assert_called_once()
        call_args = mock_client.insert.call_args
        assert call_args[1]['table'] == 'account_entropy_results'

        expected_columns = [
            'run_timestamp', 'user_id', 'window_start', 'window_end',
            'post_count', 'hourly_entropy', 'interval_entropy',
            'mean_interval_seconds', 'stddev_interval_seconds',
            'is_bot_like', 'hourly_flag', 'interval_flag',
            'sample_rkeys',
        ]
        assert call_args[1]['column_names'] == expected_columns

    @patch('account_entropy.db.clickhouse_connect.get_client')
    def test_insert_results_includes_flags_and_samples(self, mock_get_client) -> None:
        config = ClickHouseConfig(
            host='localhost',
            port=8123,
            user='default',
            password='clickhouse',
            database='default',
        )
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        db = AccountEntropyDb(config)

        now = datetime(2026, 3, 20, 12, 0, 0)
        window_start = datetime(2026, 3, 13, 12, 0, 0)
        window_end = datetime(2026, 3, 20, 12, 0, 0)

        result = ScoredResult(
            run_timestamp=now,
            user_id='did:plc:user1',
            window_start=window_start,
            window_end=window_end,
            post_count=42,
            hourly_entropy=3.5,
            interval_entropy=2.1,
            mean_interval_seconds=3600.0,
            stddev_interval_seconds=1200.0,
            is_bot_like=1,
            hourly_flag=1,
            interval_flag=0,
            sample_rkeys=['rkey1', 'rkey2'],
        )

        db.insert_results('account_entropy_results', [result])

        call_args = mock_client.insert.call_args
        data = call_args[1]['data']
        assert len(data) == 1

        row_data = data[0]
        assert row_data[10] == 1  # hourly_flag (position 10)
        assert row_data[11] == 0  # interval_flag (position 11)
        assert row_data[12] == ['rkey1', 'rkey2']  # sample_rkeys (position 12)

    @patch('account_entropy.db.clickhouse_connect.get_client')
    def test_insert_results_includes_interval_statistics(self, mock_get_client) -> None:
        config = ClickHouseConfig(
            host='localhost',
            port=8123,
            user='default',
            password='clickhouse',
            database='default',
        )
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        db = AccountEntropyDb(config)

        now = datetime(2026, 3, 20, 12, 0, 0)
        window_start = datetime(2026, 3, 13, 12, 0, 0)
        window_end = datetime(2026, 3, 20, 12, 0, 0)

        result = ScoredResult(
            run_timestamp=now,
            user_id='did:plc:user1',
            window_start=window_start,
            window_end=window_end,
            post_count=42,
            hourly_entropy=3.5,
            interval_entropy=2.1,
            mean_interval_seconds=7200.0,
            stddev_interval_seconds=3600.0,
            is_bot_like=1,
            hourly_flag=1,
            interval_flag=0,
            sample_rkeys=['rkey1'],
        )

        db.insert_results('account_entropy_results', [result])

        call_args = mock_client.insert.call_args
        data = call_args[1]['data']
        row_data = data[0]
        assert row_data[7] == 7200.0  # mean_interval_seconds (position 7)
        assert row_data[8] == 3600.0  # stddev_interval_seconds (position 8)
