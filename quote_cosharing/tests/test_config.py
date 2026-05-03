# pattern: Functional Core
import pytest

from quote_cosharing.config import AnalysisConfig, AppConfig, ClickHouseConfig


@pytest.fixture
def base_analysis_config() -> AnalysisConfig:
    return AnalysisConfig(
        interval_seconds=3600,
        resolution=0.05,
        min_edge_weight=2,
        min_cluster_size=3,
        min_cosharers=3,
        jaccard_threshold=0.5,
        evolution_window_days=7,
        pairs_table='quote_cosharing_pairs',
        clusters_table='quote_cosharing_clusters',
        membership_table='quote_cosharing_membership',
        source_table='osprey_execution_results',
    )


class TestClickHouseConfig:
    def test_from_env_defaults(self, monkeypatch) -> None:
        monkeypatch.delenv('OSPREY_CLICKHOUSE_HOST', raising=False)
        monkeypatch.delenv('OSPREY_CLICKHOUSE_PORT', raising=False)
        monkeypatch.delenv('OSPREY_CLICKHOUSE_USER', raising=False)
        monkeypatch.delenv('OSPREY_CLICKHOUSE_PASSWORD', raising=False)
        monkeypatch.delenv('OSPREY_CLICKHOUSE_DB', raising=False)

        config = ClickHouseConfig.from_env()

        assert config.host == 'localhost'
        assert config.port == 8123
        assert config.user == 'default'
        assert config.password == 'clickhouse'
        assert config.database == 'default'

    def test_from_env_overrides(self, monkeypatch) -> None:
        monkeypatch.setenv('OSPREY_CLICKHOUSE_HOST', 'clickhouse.example.com')
        monkeypatch.setenv('OSPREY_CLICKHOUSE_PORT', '9000')
        monkeypatch.setenv('OSPREY_CLICKHOUSE_USER', 'admin')
        monkeypatch.setenv('OSPREY_CLICKHOUSE_PASSWORD', 'secret')
        monkeypatch.setenv('OSPREY_CLICKHOUSE_DB', 'mydb')

        config = ClickHouseConfig.from_env()

        assert config.host == 'clickhouse.example.com'
        assert config.port == 9000
        assert config.user == 'admin'
        assert config.password == 'secret'
        assert config.database == 'mydb'

    def test_is_frozen(self) -> None:
        config = ClickHouseConfig(
            host='localhost',
            port=8123,
            user='default',
            password='clickhouse',
            database='default',
        )
        with pytest.raises(Exception):
            config.host = 'newhost'  # type: ignore


class TestAnalysisConfig:
    def test_from_env_defaults(self, monkeypatch) -> None:
        monkeypatch.delenv('QUOTE_COSHARING_INTERVAL_SECONDS', raising=False)
        monkeypatch.delenv('QUOTE_COSHARING_RESOLUTION', raising=False)
        monkeypatch.delenv('QUOTE_COSHARING_MIN_EDGE_WEIGHT', raising=False)
        monkeypatch.delenv('QUOTE_COSHARING_MIN_CLUSTER_SIZE', raising=False)
        monkeypatch.delenv('QUOTE_COSHARING_MIN_COSHARERS', raising=False)
        monkeypatch.delenv('QUOTE_COSHARING_JACCARD_THRESHOLD', raising=False)
        monkeypatch.delenv('QUOTE_COSHARING_EVOLUTION_WINDOW_DAYS', raising=False)
        monkeypatch.delenv('QUOTE_COSHARING_PAIRS_TABLE', raising=False)
        monkeypatch.delenv('QUOTE_COSHARING_CLUSTERS_TABLE', raising=False)
        monkeypatch.delenv('QUOTE_COSHARING_MEMBERSHIP_TABLE', raising=False)
        monkeypatch.delenv('QUOTE_COSHARING_SOURCE_TABLE', raising=False)

        config = AnalysisConfig.from_env()

        assert config.interval_seconds == 3600
        assert config.resolution == 0.05
        assert config.min_edge_weight == 2
        assert config.min_cluster_size == 3
        assert config.min_cosharers == 3
        assert config.jaccard_threshold == 0.5
        assert config.evolution_window_days == 7
        assert config.pairs_table == 'quote_cosharing_pairs'
        assert config.clusters_table == 'quote_cosharing_clusters'
        assert config.membership_table == 'quote_cosharing_membership'
        assert config.source_table == 'osprey_execution_results'

    def test_from_env_overrides(self, monkeypatch) -> None:
        monkeypatch.setenv('QUOTE_COSHARING_INTERVAL_SECONDS', '1800')
        monkeypatch.setenv('QUOTE_COSHARING_RESOLUTION', '0.10')
        monkeypatch.setenv('QUOTE_COSHARING_MIN_EDGE_WEIGHT', '3')
        monkeypatch.setenv('QUOTE_COSHARING_MIN_CLUSTER_SIZE', '5')
        monkeypatch.setenv('QUOTE_COSHARING_MIN_COSHARERS', '5')
        monkeypatch.setenv('QUOTE_COSHARING_JACCARD_THRESHOLD', '0.6')
        monkeypatch.setenv('QUOTE_COSHARING_EVOLUTION_WINDOW_DAYS', '14')
        monkeypatch.setenv('QUOTE_COSHARING_PAIRS_TABLE', 'custom_pairs')
        monkeypatch.setenv('QUOTE_COSHARING_CLUSTERS_TABLE', 'custom_clusters')
        monkeypatch.setenv('QUOTE_COSHARING_MEMBERSHIP_TABLE', 'custom_membership')
        monkeypatch.setenv('QUOTE_COSHARING_SOURCE_TABLE', 'custom_source')

        config = AnalysisConfig.from_env()

        assert config.interval_seconds == 1800
        assert config.resolution == 0.10
        assert config.min_edge_weight == 3
        assert config.min_cluster_size == 5
        assert config.min_cosharers == 5
        assert config.jaccard_threshold == 0.6
        assert config.evolution_window_days == 14
        assert config.pairs_table == 'custom_pairs'
        assert config.clusters_table == 'custom_clusters'
        assert config.membership_table == 'custom_membership'
        assert config.source_table == 'custom_source'

    def test_interval_seconds_parsed_as_int(self, monkeypatch) -> None:
        monkeypatch.setenv('QUOTE_COSHARING_INTERVAL_SECONDS', '1200')

        config = AnalysisConfig.from_env()

        assert config.interval_seconds == 1200

    def test_resolution_parsed_as_float(self, monkeypatch) -> None:
        monkeypatch.setenv('QUOTE_COSHARING_RESOLUTION', '0.15')

        config = AnalysisConfig.from_env()

        assert config.resolution == 0.15

    def test_pairs_table_validates_name(self, monkeypatch) -> None:
        monkeypatch.setenv('QUOTE_COSHARING_PAIRS_TABLE', 'invalid@table')

        with pytest.raises(ValueError, match='invalid table name'):
            AnalysisConfig.from_env()

    def test_clusters_table_validates_name(self, monkeypatch) -> None:
        monkeypatch.setenv('QUOTE_COSHARING_CLUSTERS_TABLE', 'invalid@table')

        with pytest.raises(ValueError, match='invalid table name'):
            AnalysisConfig.from_env()

    def test_membership_table_validates_name(self, monkeypatch) -> None:
        monkeypatch.setenv('QUOTE_COSHARING_MEMBERSHIP_TABLE', 'invalid@table')

        with pytest.raises(ValueError, match='invalid table name'):
            AnalysisConfig.from_env()

    def test_source_table_validates_name(self, monkeypatch) -> None:
        monkeypatch.setenv('QUOTE_COSHARING_SOURCE_TABLE', 'invalid@table')

        with pytest.raises(ValueError, match='invalid table name'):
            AnalysisConfig.from_env()

    def test_valid_table_names_with_dots_and_underscores(self, monkeypatch) -> None:
        monkeypatch.setenv('QUOTE_COSHARING_PAIRS_TABLE', 'db.table_name')
        monkeypatch.setenv('QUOTE_COSHARING_CLUSTERS_TABLE', 'another_db.clusters_v2')
        monkeypatch.setenv('QUOTE_COSHARING_MEMBERSHIP_TABLE', 'schema.members_123')
        monkeypatch.setenv('QUOTE_COSHARING_SOURCE_TABLE', 'src.results')

        config = AnalysisConfig.from_env()

        assert config.pairs_table == 'db.table_name'
        assert config.clusters_table == 'another_db.clusters_v2'
        assert config.membership_table == 'schema.members_123'
        assert config.source_table == 'src.results'

    def test_is_frozen(self, base_analysis_config: AnalysisConfig) -> None:
        with pytest.raises(Exception):
            base_analysis_config.interval_seconds = 1200  # type: ignore

    def test_construct_directly(self) -> None:
        config = AnalysisConfig(
            interval_seconds=3600,
            resolution=0.05,
            min_edge_weight=2,
            min_cluster_size=3,
            min_cosharers=3,
            jaccard_threshold=0.5,
            evolution_window_days=7,
            pairs_table='quote_cosharing_pairs',
            clusters_table='quote_cosharing_clusters',
            membership_table='quote_cosharing_membership',
            source_table='osprey_execution_results',
        )

        assert config.interval_seconds == 3600
        assert config.resolution == 0.05
        assert config.min_edge_weight == 2


class TestAppConfig:
    def test_from_env_composes_both_configs(self, monkeypatch) -> None:
        monkeypatch.setenv('OSPREY_CLICKHOUSE_HOST', 'ch.example.com')
        monkeypatch.setenv('QUOTE_COSHARING_INTERVAL_SECONDS', '1800')

        config = AppConfig.from_env()

        assert config.clickhouse.host == 'ch.example.com'
        assert config.analysis.interval_seconds == 1800

    def test_is_frozen(self, monkeypatch) -> None:
        monkeypatch.setenv('OSPREY_CLICKHOUSE_HOST', 'ch.example.com')
        config = AppConfig.from_env()

        with pytest.raises(Exception):
            config.analysis = AnalysisConfig.from_env()  # type: ignore

    def test_construct_directly(self) -> None:
        ch_config = ClickHouseConfig(
            host='localhost',
            port=8123,
            user='default',
            password='clickhouse',
            database='default',
        )
        analysis_config = AnalysisConfig(
            interval_seconds=3600,
            resolution=0.05,
            min_edge_weight=2,
            min_cluster_size=3,
            min_cosharers=3,
            jaccard_threshold=0.5,
            evolution_window_days=7,
            pairs_table='quote_cosharing_pairs',
            clusters_table='quote_cosharing_clusters',
            membership_table='quote_cosharing_membership',
            source_table='osprey_execution_results',
        )
        config = AppConfig(
            clickhouse=ch_config,
            analysis=analysis_config,
        )

        assert config.clickhouse.host == 'localhost'
        assert config.analysis.min_edge_weight == 2
