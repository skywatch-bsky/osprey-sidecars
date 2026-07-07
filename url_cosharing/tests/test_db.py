# pattern: Functional Core
from datetime import date, datetime
from unittest.mock import Mock, patch

import pytest

from url_cosharing.analyzer import EvolutionEvent, TimestampedCluster
from url_cosharing.config import ClickHouseConfig
from url_cosharing.db import CosharingDb, MembershipRow, MemberTimestamp
from url_cosharing.similarity import UrlShareRow


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
            mean_edge_similarity=0.75,
            subgraph_density=0.5,
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
            'mean_edge_similarity',
            'subgraph_density',
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
            mean_edge_similarity=0.0,
            subgraph_density=0.0,
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
        assert row_data[11] == 0.0  # mean_edge_similarity
        assert row_data[12] == 0.0  # subgraph_density
        assert row_data[13] == 'birth'  # evolution_type
        assert row_data[14] == ()  # predecessor_cluster_ids
        assert row_data[15] == 0.0  # jaccard_score


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


class TestRunMetadata:
    def test_create_with_all_fields(self) -> None:
        from url_cosharing.db import RunMetadata

        run = RunMetadata(
            run_date=date(2026, 3, 22),
            window_days=7,
            accounts_raw=100,
            accounts_eligible=50,
            urls_eligible=200,
            graph_edges=150,
            edge_quantile=0.75,
            centrality_quantile=0.80,
            min_component_density=0.5,
            knee_found=True,
            guardrail_triggered=False,
            flagged_accounts=10,
            cluster_count=3,
        )
        assert run.run_date == date(2026, 3, 22)
        assert run.window_days == 7
        assert run.accounts_raw == 100
        assert run.accounts_eligible == 50
        assert run.urls_eligible == 200
        assert run.graph_edges == 150
        assert run.edge_quantile == 0.75
        assert run.centrality_quantile == 0.80
        assert run.min_component_density == 0.5
        assert run.knee_found is True
        assert run.guardrail_triggered is False
        assert run.flagged_accounts == 10
        assert run.cluster_count == 3

    def test_is_frozen(self) -> None:
        from url_cosharing.db import RunMetadata

        run = RunMetadata(
            run_date=date(2026, 3, 22),
            window_days=7,
            accounts_raw=100,
            accounts_eligible=50,
            urls_eligible=200,
            graph_edges=150,
            edge_quantile=0.75,
            centrality_quantile=0.80,
            min_component_density=0.5,
            knee_found=True,
            guardrail_triggered=False,
            flagged_accounts=10,
            cluster_count=3,
        )
        with pytest.raises(AttributeError):
            run.cluster_count = 5


class TestInsertRun:
    @patch('url_cosharing.db.clickhouse_connect.get_client')
    def test_insert_run_sends_correct_columns(self, mock_get_client) -> None:
        from url_cosharing.db import RunMetadata

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

        run = RunMetadata(
            run_date=date(2026, 3, 22),
            window_days=7,
            accounts_raw=100,
            accounts_eligible=50,
            urls_eligible=200,
            graph_edges=150,
            edge_quantile=0.75,
            centrality_quantile=0.80,
            min_component_density=0.5,
            knee_found=True,
            guardrail_triggered=False,
            flagged_accounts=10,
            cluster_count=3,
        )

        db.insert_run('url_cosharing_runs', run)

        mock_client.insert.assert_called_once()
        call_args = mock_client.insert.call_args

        expected_columns = [
            'run_date',
            'window_days',
            'accounts_raw',
            'accounts_eligible',
            'urls_eligible',
            'graph_edges',
            'edge_quantile',
            'centrality_quantile',
            'min_component_density',
            'knee_found',
            'guardrail_triggered',
            'flagged_accounts',
            'cluster_count',
        ]
        assert call_args[1]['column_names'] == expected_columns

    @patch('url_cosharing.db.clickhouse_connect.get_client')
    def test_insert_run_includes_data_row(self, mock_get_client) -> None:
        from url_cosharing.db import RunMetadata

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

        run = RunMetadata(
            run_date=date(2026, 3, 22),
            window_days=7,
            accounts_raw=100,
            accounts_eligible=50,
            urls_eligible=200,
            graph_edges=150,
            edge_quantile=0.75,
            centrality_quantile=0.80,
            min_component_density=0.5,
            knee_found=True,
            guardrail_triggered=False,
            flagged_accounts=10,
            cluster_count=3,
        )

        db.insert_run('url_cosharing_runs', run)

        call_args = mock_client.insert.call_args
        data = call_args[1]['data']
        assert len(data) == 1
        row_data = data[0]
        assert row_data[0] == date(2026, 3, 22)  # run_date
        assert row_data[1] == 7  # window_days
        assert row_data[2] == 100  # accounts_raw
        assert row_data[3] == 50  # accounts_eligible
        assert row_data[4] == 200  # urls_eligible
        assert row_data[5] == 150  # graph_edges
        assert row_data[6] == 0.75  # edge_quantile
        assert row_data[7] == 0.80  # centrality_quantile
        assert row_data[8] == 0.5  # min_component_density
        assert row_data[9] is True  # knee_found
        assert row_data[10] is False  # guardrail_triggered
        assert row_data[11] == 10  # flagged_accounts
        assert row_data[12] == 3  # cluster_count


class TestFetchRawAccountCount:
    @patch('url_cosharing.db.clickhouse_connect.get_client')
    def test_returns_scalar_count_as_int(self, mock_get_client) -> None:
        config = ClickHouseConfig(
            host='localhost',
            port=8123,
            user='default',
            password='clickhouse',
            database='default',
        )
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        mock_result = Mock()
        mock_result.result_rows = [(225000,)]
        mock_client.query.return_value = mock_result

        db = CosharingDb(config)
        count = db.fetch_raw_account_count('SELECT uniqExact(UserId) FROM t')

        assert count == 225000
        assert isinstance(count, int)
        call_kwargs = mock_client.query.call_args.kwargs
        assert call_kwargs.get('settings') == {'max_execution_time': 300}

    @patch('url_cosharing.db.clickhouse_connect.get_client')
    def test_empty_result_returns_zero(self, mock_get_client) -> None:
        config = ClickHouseConfig(
            host='localhost',
            port=8123,
            user='default',
            password='clickhouse',
            database='default',
        )
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        mock_result = Mock()
        mock_result.result_rows = []
        mock_client.query.return_value = mock_result

        db = CosharingDb(config)

        assert db.fetch_raw_account_count('SELECT uniqExact(UserId) FROM t') == 0
