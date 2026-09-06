# pattern: Functional Core
import pytest

from url_cosharing.config import (
    AnalysisConfig,
    AppConfig,
    ClickHouseConfig,
    Exclusions,
    TelemetryConfig,
    load_exclusions,
    load_exclusions_with_fallback,
    parse_exclusions,
)


@pytest.fixture
def base_analysis_config() -> AnalysisConfig:
    return AnalysisConfig(
        interval_seconds=3600,
        resolution=0.05,
        min_cluster_size=3,
        jaccard_threshold=0.5,
        evolution_window_days=7,
        window_days=7,
        min_unique_urls=10,
        min_url_sharers=5,
        max_url_df_fraction=0.90,
        edge_epsilon=0.05,
        edge_quantile_grid=(0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99),
        centrality_quantile_grid=(0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99),
        density_floor=0.5,
        max_flagged_fraction=0.02,
        runs_table='url_cosharing_runs',
        clusters_table='url_cosharing_clusters',
        membership_table='url_cosharing_membership',
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
        monkeypatch.delenv('URL_COSHARING_INTERVAL_SECONDS', raising=False)
        monkeypatch.delenv('URL_COSHARING_RESOLUTION', raising=False)
        monkeypatch.delenv('URL_COSHARING_MIN_CLUSTER_SIZE', raising=False)
        monkeypatch.delenv('URL_COSHARING_JACCARD_THRESHOLD', raising=False)
        monkeypatch.delenv('URL_COSHARING_EVOLUTION_WINDOW_DAYS', raising=False)
        monkeypatch.delenv('URL_COSHARING_CLUSTERS_TABLE', raising=False)
        monkeypatch.delenv('URL_COSHARING_MEMBERSHIP_TABLE', raising=False)
        monkeypatch.delenv('URL_COSHARING_SOURCE_TABLE', raising=False)
        monkeypatch.delenv('URL_COSHARING_WINDOW_DAYS', raising=False)
        monkeypatch.delenv('URL_COSHARING_MIN_UNIQUE_URLS', raising=False)
        monkeypatch.delenv('URL_COSHARING_MIN_URL_SHARERS', raising=False)
        monkeypatch.delenv('URL_COSHARING_MAX_URL_DF_FRACTION', raising=False)
        monkeypatch.delenv('URL_COSHARING_EDGE_EPSILON', raising=False)
        monkeypatch.delenv('URL_COSHARING_EDGE_QUANTILE_GRID', raising=False)
        monkeypatch.delenv('URL_COSHARING_CENTRALITY_QUANTILE_GRID', raising=False)
        monkeypatch.delenv('URL_COSHARING_DENSITY_FLOOR', raising=False)
        monkeypatch.delenv('URL_COSHARING_MAX_FLAGGED_FRACTION', raising=False)
        monkeypatch.delenv('URL_COSHARING_MAX_FLAGGED_ACCOUNTS', raising=False)
        monkeypatch.delenv('URL_COSHARING_RUNS_TABLE', raising=False)

        config = AnalysisConfig.from_env()

        assert config.interval_seconds == 3600
        assert config.resolution == 0.05
        assert config.min_cluster_size == 3
        assert config.jaccard_threshold == 0.5
        assert config.evolution_window_days == 7
        assert config.clusters_table == 'url_cosharing_clusters'
        assert config.membership_table == 'url_cosharing_membership'
        assert config.source_table == 'osprey_execution_results'
        assert config.window_days == 7
        assert config.min_unique_urls == 10
        assert config.min_url_sharers == 5
        assert config.max_url_df_fraction == 0.90
        assert config.edge_epsilon == 0.05
        assert config.edge_quantile_grid == (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99)
        assert config.centrality_quantile_grid == (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99)
        assert config.density_floor == 0.5
        assert config.max_flagged_fraction == 0.05
        assert config.max_flagged_accounts == 750
        assert config.runs_table == 'url_cosharing_runs'

    def test_from_env_overrides(self, monkeypatch) -> None:
        monkeypatch.setenv('URL_COSHARING_INTERVAL_SECONDS', '1800')
        monkeypatch.setenv('URL_COSHARING_RESOLUTION', '0.10')
        monkeypatch.setenv('URL_COSHARING_MIN_CLUSTER_SIZE', '5')
        monkeypatch.setenv('URL_COSHARING_JACCARD_THRESHOLD', '0.6')
        monkeypatch.setenv('URL_COSHARING_EVOLUTION_WINDOW_DAYS', '14')
        monkeypatch.setenv('URL_COSHARING_CLUSTERS_TABLE', 'custom_clusters')
        monkeypatch.setenv('URL_COSHARING_MEMBERSHIP_TABLE', 'custom_membership')
        monkeypatch.setenv('URL_COSHARING_SOURCE_TABLE', 'custom_source')
        monkeypatch.setenv('URL_COSHARING_WINDOW_DAYS', '14')
        monkeypatch.setenv('URL_COSHARING_MIN_UNIQUE_URLS', '20')
        monkeypatch.setenv('URL_COSHARING_MIN_URL_SHARERS', '10')
        monkeypatch.setenv('URL_COSHARING_MAX_URL_DF_FRACTION', '0.75')
        monkeypatch.setenv('URL_COSHARING_EDGE_EPSILON', '0.10')
        monkeypatch.setenv('URL_COSHARING_EDGE_QUANTILE_GRID', '0.6,0.8')
        monkeypatch.setenv('URL_COSHARING_CENTRALITY_QUANTILE_GRID', '0.5,0.75,0.95')
        monkeypatch.setenv('URL_COSHARING_DENSITY_FLOOR', '0.3')
        monkeypatch.setenv('URL_COSHARING_MAX_FLAGGED_FRACTION', '0.10')
        monkeypatch.setenv('URL_COSHARING_MAX_FLAGGED_ACCOUNTS', '400')
        monkeypatch.setenv('URL_COSHARING_RUNS_TABLE', 'custom_runs')

        config = AnalysisConfig.from_env()

        assert config.interval_seconds == 1800
        assert config.resolution == 0.10
        assert config.min_cluster_size == 5
        assert config.jaccard_threshold == 0.6
        assert config.evolution_window_days == 14
        assert config.clusters_table == 'custom_clusters'
        assert config.membership_table == 'custom_membership'
        assert config.source_table == 'custom_source'
        assert config.window_days == 14
        assert config.min_unique_urls == 20
        assert config.min_url_sharers == 10
        assert config.max_url_df_fraction == 0.75
        assert config.edge_epsilon == 0.10
        assert config.edge_quantile_grid == (0.6, 0.8)
        assert config.centrality_quantile_grid == (0.5, 0.75, 0.95)
        assert config.density_floor == 0.3
        assert config.max_flagged_fraction == 0.10
        assert config.max_flagged_accounts == 400
        assert config.runs_table == 'custom_runs'

    def test_interval_seconds_parsed_as_int(self, monkeypatch) -> None:
        monkeypatch.setenv('URL_COSHARING_INTERVAL_SECONDS', '1200')

        config = AnalysisConfig.from_env()

        assert config.interval_seconds == 1200

    def test_resolution_parsed_as_float(self, monkeypatch) -> None:
        monkeypatch.setenv('URL_COSHARING_RESOLUTION', '0.15')

        config = AnalysisConfig.from_env()

        assert config.resolution == 0.15

    def test_clusters_table_validates_name(self, monkeypatch) -> None:
        monkeypatch.setenv('URL_COSHARING_CLUSTERS_TABLE', 'invalid@table')

        with pytest.raises(ValueError, match='invalid table name'):
            AnalysisConfig.from_env()

    def test_membership_table_validates_name(self, monkeypatch) -> None:
        monkeypatch.setenv('URL_COSHARING_MEMBERSHIP_TABLE', 'invalid@table')

        with pytest.raises(ValueError, match='invalid table name'):
            AnalysisConfig.from_env()

    def test_source_table_validates_name(self, monkeypatch) -> None:
        monkeypatch.setenv('URL_COSHARING_SOURCE_TABLE', 'invalid@table')

        with pytest.raises(ValueError, match='invalid table name'):
            AnalysisConfig.from_env()

    def test_valid_table_names_with_dots_and_underscores(self, monkeypatch) -> None:
        monkeypatch.setenv('URL_COSHARING_CLUSTERS_TABLE', 'another_db.clusters_v2')
        monkeypatch.setenv('URL_COSHARING_MEMBERSHIP_TABLE', 'schema.members_123')
        monkeypatch.setenv('URL_COSHARING_SOURCE_TABLE', 'src.results')

        config = AnalysisConfig.from_env()

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
            min_cluster_size=3,
            jaccard_threshold=0.5,
            evolution_window_days=7,
            window_days=7,
            min_unique_urls=10,
            min_url_sharers=5,
            max_url_df_fraction=0.90,
            edge_epsilon=0.05,
            edge_quantile_grid=(0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99),
            centrality_quantile_grid=(0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99),
            density_floor=0.5,
            max_flagged_fraction=0.02,
            runs_table='url_cosharing_runs',
            clusters_table='url_cosharing_clusters',
            membership_table='url_cosharing_membership',
            source_table='osprey_execution_results',
        )

        assert config.interval_seconds == 3600
        assert config.resolution == 0.05
        assert config.window_days == 7
        assert config.edge_quantile_grid == (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99)

    def test_edge_quantile_grid_with_whitespace(self, monkeypatch) -> None:
        monkeypatch.setenv('URL_COSHARING_EDGE_QUANTILE_GRID', ' 0.5, 0.9 ')

        config = AnalysisConfig.from_env()

        assert config.edge_quantile_grid == (0.5, 0.9)

    def test_edge_quantile_grid_empty_raises_error(self, monkeypatch) -> None:
        monkeypatch.setenv('URL_COSHARING_EDGE_QUANTILE_GRID', '')

        with pytest.raises(ValueError, match='quantile grid must contain at least one value'):
            AnalysisConfig.from_env()

    def test_edge_quantile_grid_non_numeric_raises_error(self, monkeypatch) -> None:
        monkeypatch.setenv('URL_COSHARING_EDGE_QUANTILE_GRID', '0.5,abc')

        with pytest.raises(ValueError):
            AnalysisConfig.from_env()

    def test_edge_quantile_grid_value_equals_one_raises_error(self, monkeypatch) -> None:
        monkeypatch.setenv('URL_COSHARING_EDGE_QUANTILE_GRID', '0.5,1.0')

        with pytest.raises(ValueError, match='quantile grid values must be in'):
            AnalysisConfig.from_env()

    def test_edge_quantile_grid_value_less_than_zero_raises_error(self, monkeypatch) -> None:
        monkeypatch.setenv('URL_COSHARING_EDGE_QUANTILE_GRID', '-0.1,0.5')

        with pytest.raises(ValueError, match='quantile grid values must be in'):
            AnalysisConfig.from_env()

    def test_edge_quantile_grid_non_increasing_raises_error(self, monkeypatch) -> None:
        monkeypatch.setenv('URL_COSHARING_EDGE_QUANTILE_GRID', '0.9,0.5')

        with pytest.raises(ValueError, match='quantile grid must be strictly increasing'):
            AnalysisConfig.from_env()

    def test_edge_quantile_grid_duplicate_raises_error(self, monkeypatch) -> None:
        monkeypatch.setenv('URL_COSHARING_EDGE_QUANTILE_GRID', '0.5,0.5')

        with pytest.raises(ValueError, match='quantile grid must be strictly increasing'):
            AnalysisConfig.from_env()

    def test_centrality_quantile_grid_with_whitespace(self, monkeypatch) -> None:
        monkeypatch.setenv('URL_COSHARING_CENTRALITY_QUANTILE_GRID', ' 0.5, 0.9 ')

        config = AnalysisConfig.from_env()

        assert config.centrality_quantile_grid == (0.5, 0.9)

    def test_max_url_df_fraction_too_high_raises_error(self, monkeypatch) -> None:
        monkeypatch.setenv('URL_COSHARING_MAX_URL_DF_FRACTION', '1.5')

        with pytest.raises(ValueError, match='must be in'):
            AnalysisConfig.from_env()

    def test_max_url_df_fraction_negative_raises_error(self, monkeypatch) -> None:
        monkeypatch.setenv('URL_COSHARING_MAX_URL_DF_FRACTION', '-0.1')

        with pytest.raises(ValueError, match='must be in'):
            AnalysisConfig.from_env()

    def test_renamed_env_var_rejected_loudly(self, monkeypatch) -> None:
        """The old percentile env var must fail fast, not be silently ignored."""
        monkeypatch.setenv('URL_COSHARING_MAX_URL_DF_PCTL', '0.90')

        with pytest.raises(ValueError, match='renamed to URL_COSHARING_MAX_URL_DF_FRACTION'):
            AnalysisConfig.from_env()

    def test_edge_epsilon_too_high_raises_error(self, monkeypatch) -> None:
        monkeypatch.setenv('URL_COSHARING_EDGE_EPSILON', '1.5')

        with pytest.raises(ValueError, match='must be in'):
            AnalysisConfig.from_env()

    def test_edge_epsilon_negative_raises_error(self, monkeypatch) -> None:
        monkeypatch.setenv('URL_COSHARING_EDGE_EPSILON', '-0.1')

        with pytest.raises(ValueError, match='must be in'):
            AnalysisConfig.from_env()

    def test_density_floor_too_high_raises_error(self, monkeypatch) -> None:
        monkeypatch.setenv('URL_COSHARING_DENSITY_FLOOR', '1.5')

        with pytest.raises(ValueError, match='must be in'):
            AnalysisConfig.from_env()

    def test_density_floor_negative_raises_error(self, monkeypatch) -> None:
        monkeypatch.setenv('URL_COSHARING_DENSITY_FLOOR', '-0.1')

        with pytest.raises(ValueError, match='must be in'):
            AnalysisConfig.from_env()

    def test_max_flagged_fraction_too_high_raises_error(self, monkeypatch) -> None:
        monkeypatch.setenv('URL_COSHARING_MAX_FLAGGED_FRACTION', '1.5')

        with pytest.raises(ValueError, match='must be in'):
            AnalysisConfig.from_env()

    def test_max_flagged_fraction_negative_raises_error(self, monkeypatch) -> None:
        monkeypatch.setenv('URL_COSHARING_MAX_FLAGGED_FRACTION', '-0.1')

        with pytest.raises(ValueError, match='must be in'):
            AnalysisConfig.from_env()

    def test_max_flagged_accounts_zero_raises_error(self, monkeypatch) -> None:
        monkeypatch.setenv('URL_COSHARING_MAX_FLAGGED_ACCOUNTS', '0')

        with pytest.raises(ValueError, match='must be >= 1'):
            AnalysisConfig.from_env()

    def test_runs_table_validates_name(self, monkeypatch) -> None:
        monkeypatch.setenv('URL_COSHARING_RUNS_TABLE', 'invalid@table')

        with pytest.raises(ValueError, match='invalid table name'):
            AnalysisConfig.from_env()


class TestTelemetryConfig:
    def test_disabled_defaults(self, monkeypatch) -> None:
        for env_var in TELEMETRY_ENV_VARS:
            monkeypatch.delenv(env_var, raising=False)

        config = TelemetryConfig.from_env()

        assert config.enabled is False
        assert config.service_name == 'url-cosharing'
        assert config.service_version == '0.1.0'
        assert config.environment == 'local'
        assert config.otlp_endpoint is None
        assert config.traces_enabled is False
        assert config.metrics_enabled is False

    @pytest.mark.parametrize('value', ['1', 'true', 'TRUE', 'yes', 'On'])
    def test_bool_true_values(self, monkeypatch, value: str) -> None:
        monkeypatch.setenv('URL_COSHARING_OTEL_ENABLED', value)

        config = TelemetryConfig.from_env()

        assert config.enabled is True
        assert config.traces_enabled is True
        assert config.metrics_enabled is True

    @pytest.mark.parametrize('value', ['0', 'false', 'FALSE', 'no', 'Off'])
    def test_bool_false_values(self, monkeypatch, value: str) -> None:
        monkeypatch.setenv('URL_COSHARING_OTEL_ENABLED', 'true')
        monkeypatch.setenv('URL_COSHARING_OTEL_TRACES_ENABLED', value)
        monkeypatch.setenv('URL_COSHARING_OTEL_METRICS_ENABLED', value)

        config = TelemetryConfig.from_env()

        assert config.enabled is True
        assert config.traces_enabled is False
        assert config.metrics_enabled is False

    def test_from_env_overrides(self, monkeypatch) -> None:
        monkeypatch.setenv('URL_COSHARING_OTEL_ENABLED', 'true')
        monkeypatch.setenv('URL_COSHARING_OTEL_SERVICE_NAME', 'custom-service')
        monkeypatch.setenv('URL_COSHARING_OTEL_SERVICE_VERSION', '1.2.3')
        monkeypatch.setenv('URL_COSHARING_OTEL_ENVIRONMENT', 'prod')
        monkeypatch.setenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'http://collector:4317')
        monkeypatch.setenv('URL_COSHARING_OTEL_TRACES_ENABLED', 'false')
        monkeypatch.setenv('URL_COSHARING_OTEL_METRICS_ENABLED', 'true')

        config = TelemetryConfig.from_env()

        assert config.enabled is True
        assert config.service_name == 'custom-service'
        assert config.service_version == '1.2.3'
        assert config.environment == 'prod'
        assert config.otlp_endpoint == 'http://collector:4317'
        assert config.traces_enabled is False
        assert config.metrics_enabled is True

    def test_invalid_bool_names_env_var(self, monkeypatch) -> None:
        monkeypatch.setenv('URL_COSHARING_OTEL_ENABLED', 'maybe')

        with pytest.raises(ValueError, match='URL_COSHARING_OTEL_ENABLED'):
            TelemetryConfig.from_env()


TELEMETRY_ENV_VARS = (
    'URL_COSHARING_OTEL_ENABLED',
    'URL_COSHARING_OTEL_SERVICE_NAME',
    'URL_COSHARING_OTEL_SERVICE_VERSION',
    'URL_COSHARING_OTEL_ENVIRONMENT',
    'OTEL_EXPORTER_OTLP_ENDPOINT',
    'URL_COSHARING_OTEL_TRACES_ENABLED',
    'URL_COSHARING_OTEL_METRICS_ENABLED',
)


class TestAppConfig:
    def test_from_env_composes_both_configs(self, monkeypatch) -> None:
        monkeypatch.setenv('OSPREY_CLICKHOUSE_HOST', 'ch.example.com')
        monkeypatch.setenv('URL_COSHARING_INTERVAL_SECONDS', '1800')

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
            min_cluster_size=3,
            jaccard_threshold=0.5,
            evolution_window_days=7,
            window_days=7,
            min_unique_urls=10,
            min_url_sharers=5,
            max_url_df_fraction=0.90,
            edge_epsilon=0.05,
            edge_quantile_grid=(0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99),
            centrality_quantile_grid=(0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99),
            density_floor=0.5,
            max_flagged_fraction=0.02,
            runs_table='url_cosharing_runs',
            clusters_table='url_cosharing_clusters',
            membership_table='url_cosharing_membership',
            source_table='osprey_execution_results',
        )
        config = AppConfig(
            clickhouse=ch_config,
            analysis=analysis_config,
        )

        assert config.clickhouse.host == 'localhost'
        assert config.analysis.resolution == 0.05
        assert config.telemetry.enabled is False


class TestExclusions:
    def test_empty_revision(self) -> None:
        exclusions = Exclusions.empty()
        assert exclusions.excluded_domains == ()
        assert exclusions.excluded_dids == ()
        assert exclusions.is_empty()

    def test_empty_hash_matches_zero_entry_revision(self) -> None:
        """empty() must be indistinguishable from a file with empty lists."""
        from_file = Exclusions.from_mapping({'excluded_domains': [], 'excluded_dids': []})
        assert from_file == Exclusions.empty()
        assert Exclusions.from_mapping({}) == Exclusions.empty()

    def test_parse_exclusions_valid(self) -> None:
        exclusions = parse_exclusions(
            'excluded_domains:\n  - static.klipy.com\nexcluded_dids:\n  - did:plc:aaaaaaaaaaaaaaaaaaaaaaaa\n'
        )
        assert exclusions.excluded_domains == ('static.klipy.com',)
        assert exclusions.excluded_dids == ('did:plc:aaaaaaaaaaaaaaaaaaaaaaaa',)
        assert not exclusions.is_empty()

    def test_parse_exclusions_normalizes_domains(self) -> None:
        """Domains are stripped, lowercased, deduped, and sorted."""
        exclusions = parse_exclusions(
            'excluded_domains:\n  - " Static.Klipy.com "\n  - static.klipy.com\n  - a.example.com\n'
        )
        assert exclusions.excluded_domains == ('a.example.com', 'static.klipy.com')

    def test_parse_exclusions_empty_text(self) -> None:
        assert parse_exclusions('') == Exclusions.empty()
        assert parse_exclusions('   \n') == Exclusions.empty()
        assert parse_exclusions(None) == Exclusions.empty()

    def test_parse_exclusions_comments_only(self) -> None:
        """A comments-only file is the operator's 'disable by commenting out'
        workflow: a null YAML document, not an error."""
        assert parse_exclusions('# excluded_domains:\n#   - static.klipy.com\n') == Exclusions.empty()

    def test_parse_exclusions_rejects_non_mapping(self) -> None:
        with pytest.raises(ValueError, match='must be a YAML mapping'):
            parse_exclusions('- static.klipy.com\n')

    def test_parse_exclusions_rejects_unknown_keys(self) -> None:
        """Typo protection: `excluded_domain:` must fail loudly, not silently
        exclude nothing."""
        with pytest.raises(ValueError, match='unknown exclusions keys'):
            parse_exclusions('excluded_domain:\n  - static.klipy.com\n')

    def test_parse_exclusions_rejects_non_list(self) -> None:
        with pytest.raises(ValueError, match='must be YAML lists'):
            parse_exclusions('excluded_domains: static.klipy.com\n')

    def test_parse_exclusions_rejects_non_string_domain(self) -> None:
        with pytest.raises(ValueError, match='must be strings'):
            parse_exclusions('excluded_domains:\n  - 123\n')

    def test_parse_exclusions_rejects_invalid_domain(self) -> None:
        """Junk hostnames would produce silently-never-matching predicates."""
        for bad in ['..', '-', '.klipy.com', 'klipy', 'http://klipy.com', 'klipy.com/path', '-klipy.com']:
            with pytest.raises(ValueError, match='invalid hostname'):
                parse_exclusions(f'excluded_domains:\n  - "{bad}"\n')

    def test_parse_exclusions_rejects_invalid_yaml(self) -> None:
        with pytest.raises(ValueError, match='invalid exclusions YAML'):
            parse_exclusions('excluded_domains: [unclosed\n')

    def test_parse_exclusions_validates_dids(self) -> None:
        """DIDs must be full did:plc identifiers: 'did:plc:' + 24 chars of
        [a-z2-7]. Shorter/looser shapes would produce silently-never-matching
        predicates."""
        bad = [
            'did:web:example.com',
            'not-a-did',
            "did:plc:abc'; DROP TABLE--",
            'DID:PLC:ABC',
            'did:plc:',
            'did:plc:abcdefghij',  # too short
            'did:plc:aaaaaaaaaaaaaaaaaaaaaaaaa',  # 25 chars, too long
            'did:plc:aaaaaaa0aaaaaaa1aaaaaaaa',  # 0/1 not in base32 alphabet
        ]
        for entry in bad:
            with pytest.raises(ValueError, match='invalid DID'):
                parse_exclusions(f'excluded_dids:\n  - "{entry}"\n')

    def test_parse_exclusions_sql_metacharacters_rejected_by_did_pattern(self) -> None:
        """Injection defence: quote/backslash payloads cannot survive validation,
        so the generated SQL cannot be broken out of."""
        with pytest.raises(ValueError, match='invalid DID'):
            parse_exclusions("excluded_dids:\n  - \"did:plc:abc','did:plc:evil\"\n")

    def test_content_hash_stable_and_content_sensitive(self) -> None:
        """Same entries in any order/duplication hash identically; any content
        change changes the hash."""
        base = parse_exclusions('excluded_domains:\n  - a.example.com\n  - b.example.com\n')
        reordered = parse_exclusions('excluded_domains:\n  - b.example.com\n  - a.example.com\n  - a.example.com\n')
        assert base.content_hash == reordered.content_hash
        assert base.excluded_domains == reordered.excluded_domains

        extra = parse_exclusions('excluded_domains:\n  - a.example.com\n  - b.example.com\n  - c.example.com\n')
        assert extra.content_hash != base.content_hash

        with_did = parse_exclusions('excluded_dids:\n  - did:plc:aaaaaaaaaaaaaaaaaaaaaaaa\n')
        assert with_did.content_hash != base.content_hash
        assert with_did.content_hash != Exclusions.empty().content_hash

    def test_load_exclusions_reads_file(self, tmp_path) -> None:
        path = tmp_path / 'exclusions.yaml'
        path.write_text('excluded_domains:\n  - static.klipy.com\n', encoding='utf-8')
        exclusions = load_exclusions(str(path))
        assert exclusions.excluded_domains == ('static.klipy.com',)

    def test_load_exclusions_missing_file_raises_oserror(self, tmp_path) -> None:
        with pytest.raises(OSError):
            load_exclusions(str(tmp_path / 'missing.yaml'))

    def test_load_exclusions_invalid_file_raises_valueerror(self, tmp_path) -> None:
        path = tmp_path / 'exclusions.yaml'
        path.write_text('excluded_domain:\n  - static.klipy.com\n', encoding='utf-8')
        with pytest.raises(ValueError):
            load_exclusions(str(path))

    def test_load_exclusions_with_fallback_none_path(self) -> None:
        """The file is not required when the env var is unset."""
        exclusions, error = load_exclusions_with_fallback(None, Exclusions.empty())
        assert exclusions == Exclusions.empty()
        assert error is None

    def test_load_exclusions_with_fallback_success(self, tmp_path) -> None:
        path = tmp_path / 'exclusions.yaml'
        path.write_text('excluded_domains:\n  - static.klipy.com\n', encoding='utf-8')
        exclusions, error = load_exclusions_with_fallback(str(path), Exclusions.empty())
        assert exclusions.excluded_domains == ('static.klipy.com',)
        assert error is None

    def test_load_exclusions_with_fallback_missing_file_retains_previous(self, tmp_path) -> None:
        """Mid-flight file removal keeps last-known-good for the cycle."""
        previous = parse_exclusions('excluded_domains:\n  - klipy.com\n')
        exclusions, error = load_exclusions_with_fallback(str(tmp_path / 'missing.yaml'), previous)
        assert exclusions is previous
        assert error is not None and 'missing.yaml' in error

    def test_load_exclusions_with_fallback_invalid_file_retains_previous(self, tmp_path) -> None:
        previous = parse_exclusions('excluded_domains:\n  - klipy.com\n')
        path = tmp_path / 'exclusions.yaml'
        path.write_text('not: [valid\n', encoding='utf-8')
        exclusions, error = load_exclusions_with_fallback(str(path), previous)
        assert exclusions is previous
        assert error is not None

    def test_exclusions_file_env_var(self, monkeypatch) -> None:
        monkeypatch.delenv('URL_COSHARING_EXCLUSIONS_FILE', raising=False)
        assert AnalysisConfig.from_env().exclusions_file is None

        monkeypatch.setenv('URL_COSHARING_EXCLUSIONS_FILE', '')
        assert AnalysisConfig.from_env().exclusions_file is None

        monkeypatch.setenv('URL_COSHARING_EXCLUSIONS_FILE', '/config/exclusions.yaml')
        assert AnalysisConfig.from_env().exclusions_file == '/config/exclusions.yaml'

    def test_exclusions_is_frozen(self) -> None:
        exclusions = Exclusions.empty()
        with pytest.raises(AttributeError):
            exclusions.excluded_domains = ('a.example.com',)  # type: ignore[misc]
