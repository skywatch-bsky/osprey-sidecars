# pattern: Functional Core
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

_TABLE_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9_.]+$')


def _validate_table_name(table: str) -> str:
    if not _TABLE_NAME_PATTERN.match(table):
        raise ValueError(f'invalid table name: {table!r}')
    return table


def _parse_quantile_grid(raw: str) -> tuple[float, ...]:
    parts = [part.strip() for part in raw.split(',') if part.strip()]
    if not parts:
        raise ValueError(f'quantile grid must contain at least one value: {raw!r}')
    values = tuple(float(part) for part in parts)
    for value in values:
        if not 0.0 < value < 1.0:
            raise ValueError(f'quantile grid values must be in (0, 1): {raw!r}')
    if list(values) != sorted(set(values)):
        raise ValueError(f'quantile grid must be strictly increasing: {raw!r}')
    return values


def _validate_unit_interval(name: str, value: float) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f'{name} must be in [0, 1]: {value!r}')
    return value


def _validate_positive_int(name: str, value: int) -> int:
    if value < 1:
        raise ValueError(f'{name} must be >= 1: {value!r}')
    return value


def _parse_bool(env_var: str, value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {'1', 'true', 'yes', 'on'}:
        return True
    if normalized in {'0', 'false', 'no', 'off'}:
        return False
    raise ValueError(f'{env_var} must be a boolean (1/0, true/false, yes/no, on/off): {value!r}')


@dataclass(frozen=True)
class ClickHouseConfig:
    host: str
    port: int
    user: str
    password: str
    database: str

    @classmethod
    def from_env(cls) -> ClickHouseConfig:
        return cls(
            host=os.environ.get('OSPREY_CLICKHOUSE_HOST', 'localhost'),
            port=int(os.environ.get('OSPREY_CLICKHOUSE_PORT', '8123')),
            user=os.environ.get('OSPREY_CLICKHOUSE_USER', 'default'),
            password=os.environ.get('OSPREY_CLICKHOUSE_PASSWORD', 'clickhouse'),
            database=os.environ.get('OSPREY_CLICKHOUSE_DB', 'default'),
        )


@dataclass(frozen=True)
class AnalysisConfig:
    interval_seconds: int
    resolution: float
    min_cluster_size: int
    jaccard_threshold: float
    evolution_window_days: int
    window_days: int
    min_unique_urls: int
    min_url_sharers: int
    max_url_df_fraction: float
    edge_epsilon: float
    edge_quantile_grid: tuple[float, ...]
    centrality_quantile_grid: tuple[float, ...]
    density_floor: float
    max_flagged_fraction: float
    runs_table: str
    clusters_table: str
    membership_table: str
    source_table: str
    # Absolute survivor cap paired with max_flagged_fraction; the effective
    # guardrail is min(fraction * eligible, accounts). Observed coordinated
    # cores are roughly constant in absolute size (Cinus et al.: 25-764),
    # so the fraction alone would make sensitivity track daily eligibility.
    max_flagged_accounts: int = 750

    @classmethod
    def from_env(cls) -> AnalysisConfig:
        if 'URL_COSHARING_MAX_URL_DF_PCTL' in os.environ:
            raise ValueError(
                'URL_COSHARING_MAX_URL_DF_PCTL was renamed to '
                'URL_COSHARING_MAX_URL_DF_FRACTION with new semantics: the value is '
                'now a fraction of distinct accounts (sklearn max_df), not a '
                'percentile of the URL df distribution. Update the environment.'
            )
        return cls(
            interval_seconds=int(os.environ.get('URL_COSHARING_INTERVAL_SECONDS', '3600')),
            resolution=float(os.environ.get('URL_COSHARING_RESOLUTION', '0.05')),
            min_cluster_size=int(os.environ.get('URL_COSHARING_MIN_CLUSTER_SIZE', '3')),
            jaccard_threshold=float(os.environ.get('URL_COSHARING_JACCARD_THRESHOLD', '0.5')),
            evolution_window_days=int(os.environ.get('URL_COSHARING_EVOLUTION_WINDOW_DAYS', '7')),
            window_days=int(os.environ.get('URL_COSHARING_WINDOW_DAYS', '7')),
            min_unique_urls=int(os.environ.get('URL_COSHARING_MIN_UNIQUE_URLS', '10')),
            min_url_sharers=int(os.environ.get('URL_COSHARING_MIN_URL_SHARERS', '5')),
            max_url_df_fraction=_validate_unit_interval(
                'URL_COSHARING_MAX_URL_DF_FRACTION',
                float(os.environ.get('URL_COSHARING_MAX_URL_DF_FRACTION', '0.90')),
            ),
            edge_epsilon=_validate_unit_interval(
                'URL_COSHARING_EDGE_EPSILON',
                float(os.environ.get('URL_COSHARING_EDGE_EPSILON', '0.05')),
            ),
            edge_quantile_grid=_parse_quantile_grid(
                os.environ.get('URL_COSHARING_EDGE_QUANTILE_GRID', '0.50,0.60,0.70,0.80,0.90,0.95,0.99')
            ),
            centrality_quantile_grid=_parse_quantile_grid(
                os.environ.get('URL_COSHARING_CENTRALITY_QUANTILE_GRID', '0.50,0.60,0.70,0.80,0.90,0.95,0.99')
            ),
            density_floor=_validate_unit_interval(
                'URL_COSHARING_DENSITY_FLOOR',
                float(os.environ.get('URL_COSHARING_DENSITY_FLOOR', '0.5')),
            ),
            max_flagged_fraction=_validate_unit_interval(
                'URL_COSHARING_MAX_FLAGGED_FRACTION',
                float(os.environ.get('URL_COSHARING_MAX_FLAGGED_FRACTION', '0.05')),
            ),
            max_flagged_accounts=_validate_positive_int(
                'URL_COSHARING_MAX_FLAGGED_ACCOUNTS',
                int(os.environ.get('URL_COSHARING_MAX_FLAGGED_ACCOUNTS', '750')),
            ),
            runs_table=_validate_table_name(os.environ.get('URL_COSHARING_RUNS_TABLE', 'url_cosharing_runs')),
            clusters_table=_validate_table_name(
                os.environ.get('URL_COSHARING_CLUSTERS_TABLE', 'url_cosharing_clusters')
            ),
            membership_table=_validate_table_name(
                os.environ.get('URL_COSHARING_MEMBERSHIP_TABLE', 'url_cosharing_membership')
            ),
            source_table=_validate_table_name(os.environ.get('URL_COSHARING_SOURCE_TABLE', 'osprey_execution_results')),
        )


@dataclass(frozen=True)
class TelemetryConfig:
    enabled: bool
    service_name: str
    service_version: str
    environment: str
    otlp_endpoint: str | None
    traces_enabled: bool
    metrics_enabled: bool

    @classmethod
    def disabled(cls) -> TelemetryConfig:
        return cls(
            enabled=False,
            service_name='url-cosharing',
            service_version='0.1.0',
            environment='local',
            otlp_endpoint=None,
            traces_enabled=False,
            metrics_enabled=False,
        )

    @classmethod
    def from_env(cls) -> TelemetryConfig:
        enabled = _parse_bool('URL_COSHARING_OTEL_ENABLED', os.environ.get('URL_COSHARING_OTEL_ENABLED', 'false'))
        traces_enabled = _parse_bool(
            'URL_COSHARING_OTEL_TRACES_ENABLED',
            os.environ.get('URL_COSHARING_OTEL_TRACES_ENABLED', 'true' if enabled else 'false'),
        )
        metrics_enabled = _parse_bool(
            'URL_COSHARING_OTEL_METRICS_ENABLED',
            os.environ.get('URL_COSHARING_OTEL_METRICS_ENABLED', 'true' if enabled else 'false'),
        )
        endpoint = os.environ.get('OTEL_EXPORTER_OTLP_ENDPOINT')
        if endpoint == '':
            endpoint = None
        return cls(
            enabled=enabled,
            service_name=os.environ.get('URL_COSHARING_OTEL_SERVICE_NAME', 'url-cosharing'),
            service_version=os.environ.get('URL_COSHARING_OTEL_SERVICE_VERSION', '0.1.0'),
            environment=os.environ.get('URL_COSHARING_OTEL_ENVIRONMENT', 'local'),
            otlp_endpoint=endpoint,
            traces_enabled=traces_enabled,
            metrics_enabled=metrics_enabled,
        )


@dataclass(frozen=True)
class AppConfig:
    clickhouse: ClickHouseConfig
    analysis: AnalysisConfig
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig.disabled)

    @classmethod
    def from_env(cls) -> AppConfig:
        return cls(
            clickhouse=ClickHouseConfig.from_env(),
            analysis=AnalysisConfig.from_env(),
            telemetry=TelemetryConfig.from_env(),
        )
