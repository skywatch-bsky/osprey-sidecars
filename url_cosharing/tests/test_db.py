# pattern: Functional Core
from datetime import date, datetime
from unittest.mock import Mock, patch

import pytest

from url_cosharing.analyzer import EvolutionEvent, PairRow, TimestampedCluster
from url_cosharing.config import ClickHouseConfig
from url_cosharing.db import CosharingDb, MembershipRow, MemberTimestamp
from url_cosharing.similarity import UrlShareRow


class TestPairRow:
    def test_create_with_all_fields(self) -> None:
        row = PairRow(
            date=date(2026, 3, 22),
            account_a='did:plc:user1',
            account_b='did:plc:user2',
            weight=5,
            newman_weight=2.5,
            shared_urls=['https://example.com', 'https://test.com'],
        )
        assert row.date == date(2026, 3, 22)
        assert row.account_a == 'did:plc:user1'
        assert row.account_b == 'did:plc:user2'
        assert row.weight == 5
        assert row.newman_weight == 2.5
        assert row.shared_urls == ['https://example.com', 'https://test.com']

    def test_create_with_empty_urls(self) -> None:
        row = PairRow(
            date=date(2026, 3, 22),
            account_a='did:plc:user1',
            account_b='did:plc:user2',
            weight=1,
            newman_weight=0.5,
            shared_urls=[],
        )
        assert row.shared_urls == []

    def test_is_frozen(self) -> None:
        row = PairRow(
            date=date(2026, 3, 22),
            account_a='did:plc:user1',
            account_b='did:plc:user2',
            weight=5,
            newman_weight=2.5,
            shared_urls=['https://example.com'],
        )
        with pytest.raises(AttributeError):
            row.weight = 10


class TestMembershipRow:
    def test_create_with_all_fields(self) -> None:
        row = MembershipRow(
            run_date=date(2026, 3, 22),
            cluster_id='2026-03-22-0001',
            did='did:plc:user1',
        )
        assert row.run_date == date(2026, 3, 22)
        assert row.cluster_id == '2026-03-22-0001'
        assert row.did == 'did:plc:user1'

    def test_is_frozen(self) -> None:
        row = MembershipRow(
            run_date=date(2026, 3, 22),
            cluster_id='2026-03-22-0001',
            did='did:plc:user1',
        )
        with pytest.raises(AttributeError):
            row.did = 'did:plc:user2'


class TestMemberTimestamp:
    def test_create_with_all_fields(self) -> None:
        ts = datetime(2026, 3, 22, 12, 30, 45)
        row = MemberTimestamp(
            did='did:plc:user1',
            ts=ts,
        )
        assert row.did == 'did:plc:user1'
        assert row.ts == ts

    def test_is_frozen(self) -> None:
        ts = datetime(2026, 3, 22, 12, 30, 45)
        row = MemberTimestamp(
            did='did:plc:user1',
            ts=ts,
        )
        with pytest.raises(AttributeError):
            row.did = 'did:plc:user2'


class TestCosharingDb:
    @patch('url_cosharing.db.clickhouse_connect.get_client')
    def test_fetch_pairs_maps_columns_correctly(self, mock_get_client) -> None:
        config = ClickHouseConfig(
            host='localhost',
            port=8123,
            user='default',
            password='clickhouse',
            database='default',
        )
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        db = CosharingDb(config)

        mock_result = Mock()
        mock_result.result_rows = [
            (date(2026, 3, 22), 'did:plc:user1', 'did:plc:user2', 5, 2.5, ['https://example.com', 'https://test.com']),
            (date(2026, 3, 22), 'did:plc:user3', 'did:plc:user4', 2, 1.0, []),
        ]
        mock_client.query.return_value = mock_result

        rows = db.fetch_pairs('SELECT * FROM pairs')

        assert len(rows) == 2
        assert rows[0].date == date(2026, 3, 22)
        assert rows[0].account_a == 'did:plc:user1'
        assert rows[0].account_b == 'did:plc:user2'
        assert rows[0].weight == 5
        assert rows[0].newman_weight == 2.5
        assert rows[0].shared_urls == ['https://example.com', 'https://test.com']

        assert rows[1].account_a == 'did:plc:user3'
        assert rows[1].weight == 2
        assert rows[1].newman_weight == 1.0
        assert rows[1].shared_urls == []

    @patch('url_cosharing.db.clickhouse_connect.get_client')
    def test_fetch_pairs_sets_max_execution_time(self, mock_get_client) -> None:
        config = ClickHouseConfig(
            host='localhost',
            port=8123,
            user='default',
            password='clickhouse',
            database='default',
        )
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        db = CosharingDb(config)

        mock_result = Mock()
        mock_result.result_rows = []
        mock_client.query.return_value = mock_result

        db.fetch_pairs('SELECT * FROM pairs')

        mock_client.query.assert_called_once()
        call_kwargs = mock_client.query.call_args[1]
        assert call_kwargs.get('settings') == {'max_execution_time': 120}

    @patch('url_cosharing.db.clickhouse_connect.get_client')
    def test_fetch_historical_membership_maps_columns_correctly(self, mock_get_client) -> None:
        config = ClickHouseConfig(
            host='localhost',
            port=8123,
            user='default',
            password='clickhouse',
            database='default',
        )
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        db = CosharingDb(config)

        mock_result = Mock()
        mock_result.result_rows = [
            (date(2026, 3, 21), '2026-03-21-0001', 'did:plc:user1'),
            (date(2026, 3, 21), '2026-03-21-0001', 'did:plc:user2'),
        ]
        mock_client.query.return_value = mock_result

        rows = db.fetch_historical_membership('SELECT * FROM membership')

        assert len(rows) == 2
        assert rows[0].run_date == date(2026, 3, 21)
        assert rows[0].cluster_id == '2026-03-21-0001'
        assert rows[0].did == 'did:plc:user1'

    @patch('url_cosharing.db.clickhouse_connect.get_client')
    def test_fetch_member_timestamps_maps_columns_correctly(self, mock_get_client) -> None:
        config = ClickHouseConfig(
            host='localhost',
            port=8123,
            user='default',
            password='clickhouse',
            database='default',
        )
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        db = CosharingDb(config)

        ts1 = datetime(2026, 3, 22, 12, 0, 0)
        ts2 = datetime(2026, 3, 22, 13, 30, 45)
        mock_result = Mock()
        mock_result.result_rows = [
            ('did:plc:user1', ts1),
            ('did:plc:user1', ts2),
        ]
        mock_client.query.return_value = mock_result

        rows = db.fetch_member_timestamps('SELECT * FROM timestamps')

        assert len(rows) == 2
        assert rows[0].did == 'did:plc:user1'
        assert rows[0].ts == ts1
        assert rows[1].ts == ts2

    @patch('url_cosharing.db.clickhouse_connect.get_client')
    def test_delete_run_date_issues_alter_table_delete(self, mock_get_client) -> None:
        config = ClickHouseConfig(
            host='localhost',
            port=8123,
            user='default',
            password='clickhouse',
            database='default',
        )
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        db = CosharingDb(config)
        db.delete_run_date('url_cosharing_clusters', date(2026, 3, 22))

        mock_client.command.assert_called_once()
        call_args = mock_client.command.call_args
        assert 'ALTER TABLE url_cosharing_clusters DELETE' in call_args[0][0]
        assert call_args[1]['parameters']['rd'] == date(2026, 3, 22)

    @patch('url_cosharing.db.clickhouse_connect.get_client')
    def test_insert_clusters_sends_correct_columns(self, mock_get_client) -> None:
        config = ClickHouseConfig(
            host='localhost',
            port=8123,
            user='default',
            password='clickhouse',
            database='default',
        )
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        db = CosharingDb(config)

        cluster = TimestampedCluster(
            cluster_id='2026-03-22-0001',
            members=frozenset(['did:plc:user1', 'did:plc:user2']),
            member_count=2,
            total_edges=1,
            total_weight=5,
            unique_urls=3,
            sample_dids=['did:plc:user1', 'did:plc:user2'],
            sample_urls=['https://example.com', 'https://test.com', 'https://other.com'],
            resolution_parameter=0.05,
            temporal_spread_hours=24.5,
            mean_posting_interval_seconds=3600.0,
        )

        event = EvolutionEvent(
            cluster_id='2026-03-22-0001',
            members=frozenset(['did:plc:user1', 'did:plc:user2']),
            evolution_type='birth',
            predecessor_cluster_ids=(),
            jaccard_score=0.0,
        )

        db.insert_clusters('url_cosharing_clusters', [(date(2026, 3, 22), cluster, event)])

        mock_client.insert.assert_called_once()
        call_args = mock_client.insert.call_args

        expected_columns = [
            'run_date',
            'cluster_id',
            'member_count',
            'total_edges',
            'total_weight',
            'unique_urls',
            'temporal_spread_hours',
            'mean_posting_interval_seconds',
            'sample_dids',
            'sample_urls',
            'resolution_parameter',
            'evolution_type',
            'predecessor_cluster_ids',
            'jaccard_score',
        ]
        assert call_args[1]['column_names'] == expected_columns

    @patch('url_cosharing.db.clickhouse_connect.get_client')
    def test_insert_membership_sends_correct_columns(self, mock_get_client) -> None:
        config = ClickHouseConfig(
            host='localhost',
            port=8123,
            user='default',
            password='clickhouse',
            database='default',
        )
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        db = CosharingDb(config)

        membership_rows = [
            (date(2026, 3, 22), '2026-03-22-0001', 'did:plc:user1'),
            (date(2026, 3, 22), '2026-03-22-0001', 'did:plc:user2'),
        ]

        db.insert_membership('url_cosharing_membership', membership_rows)

        mock_client.insert.assert_called_once()
        call_args = mock_client.insert.call_args

        expected_columns = ['run_date', 'cluster_id', 'did']
        assert call_args[1]['column_names'] == expected_columns

    @patch('url_cosharing.db.clickhouse_connect.get_client')
    def test_insert_clusters_includes_data_rows(self, mock_get_client) -> None:
        config = ClickHouseConfig(
            host='localhost',
            port=8123,
            user='default',
            password='clickhouse',
            database='default',
        )
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        db = CosharingDb(config)

        cluster = TimestampedCluster(
            cluster_id='2026-03-22-0001',
            members=frozenset(['did:plc:user1']),
            member_count=1,
            total_edges=0,
            total_weight=0,
            unique_urls=0,
            sample_dids=['did:plc:user1'],
            sample_urls=[],
            resolution_parameter=0.05,
            temporal_spread_hours=0.0,
            mean_posting_interval_seconds=0.0,
        )

        event = EvolutionEvent(
            cluster_id='2026-03-22-0001',
            members=frozenset(['did:plc:user1']),
            evolution_type='birth',
            predecessor_cluster_ids=(),
            jaccard_score=0.0,
        )

        db.insert_clusters('url_cosharing_clusters', [(date(2026, 3, 22), cluster, event)])

        call_args = mock_client.insert.call_args
        data = call_args[1]['data']
        assert len(data) == 1
        row_data = data[0]
        assert row_data[0] == date(2026, 3, 22)  # run_date
        assert row_data[1] == '2026-03-22-0001'  # cluster_id
        assert row_data[2] == 1  # member_count
        assert row_data[11] == 'birth'  # evolution_type
        assert row_data[12] == ()  # predecessor_cluster_ids
        assert row_data[13] == 0.0  # jaccard_score


class TestUrlShareRow:
    def test_create_with_all_fields(self) -> None:
        row = UrlShareRow(
            did='did:plc:user1',
            url='https://example.com',
            share_count=5,
        )
        assert row.did == 'did:plc:user1'
        assert row.url == 'https://example.com'
        assert row.share_count == 5

    def test_is_frozen(self) -> None:
        row = UrlShareRow(
            did='did:plc:user1',
            url='https://example.com',
            share_count=5,
        )
        with pytest.raises(AttributeError):
            row.share_count = 10


class TestFetchUrlShares:
    @patch('url_cosharing.db.clickhouse_connect.get_client')
    def test_maps_columns_correctly(self, mock_get_client) -> None:
        config = ClickHouseConfig(
            host='localhost',
            port=8123,
            user='default',
            password='clickhouse',
            database='default',
        )
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        db = CosharingDb(config)

        mock_result = Mock()
        mock_result.result_rows = [
            ('did:plc:a', 'https://x.test/1', 3),
            ('did:plc:b', 'https://x.test/2', 2),
        ]
        mock_client.query.return_value = mock_result

        rows = db.fetch_url_shares('SELECT * FROM url_shares')

        assert len(rows) == 2
        assert rows[0].did == 'did:plc:a'
        assert rows[0].url == 'https://x.test/1'
        assert rows[0].share_count == 3
        assert isinstance(rows[0].share_count, int)

        assert rows[1].did == 'did:plc:b'
        assert rows[1].url == 'https://x.test/2'
        assert rows[1].share_count == 2

    @patch('url_cosharing.db.clickhouse_connect.get_client')
    def test_sets_max_execution_time_300(self, mock_get_client) -> None:
        config = ClickHouseConfig(
            host='localhost',
            port=8123,
            user='default',
            password='clickhouse',
            database='default',
        )
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        db = CosharingDb(config)

        mock_result = Mock()
        mock_result.result_rows = []
        mock_client.query.return_value = mock_result

        db.fetch_url_shares('SELECT * FROM url_shares')

        mock_client.query.assert_called_once()
        call_kwargs = mock_client.query.call_args[1]
        assert call_kwargs.get('settings') == {'max_execution_time': 300}

    @patch('url_cosharing.db.clickhouse_connect.get_client')
    def test_handles_empty_result_rows(self, mock_get_client) -> None:
        config = ClickHouseConfig(
            host='localhost',
            port=8123,
            user='default',
            password='clickhouse',
            database='default',
        )
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        db = CosharingDb(config)

        mock_result = Mock()
        mock_result.result_rows = []
        mock_client.query.return_value = mock_result

        rows = db.fetch_url_shares('SELECT * FROM url_shares')

        assert rows == []
