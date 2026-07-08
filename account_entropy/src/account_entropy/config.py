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
    window_days: int
    min_posts: int
    hourly_entropy_norm_threshold: float
    interval_entropy_norm_threshold: float
    cv_threshold: float
    interval_bin_edges: tuple[int, ...]
    source_table: str
    output_table: str

    @classmethod
    def from_env(cls) -> AnalysisConfig:
        bin_edges_raw = os.environ.get(
            'ACCOUNT_ENTROPY_INTERVAL_BIN_EDGES',
            '60,300,900,3600,14400,86400',
        )
        return cls(
            interval_seconds=int(os.environ.get('ACCOUNT_ENTROPY_INTERVAL_SECONDS', '3600')),
            window_days=int(os.environ.get('ACCOUNT_ENTROPY_WINDOW_DAYS', '7')),
            min_posts=int(os.environ.get('ACCOUNT_ENTROPY_MIN_POSTS', '10')),
            hourly_entropy_norm_threshold=float(os.environ.get('ACCOUNT_ENTROPY_HOURLY_NORM_THRESHOLD', '0.85')),
            interval_entropy_norm_threshold=float(os.environ.get('ACCOUNT_ENTROPY_INTERVAL_NORM_THRESHOLD', '0.53')),
            cv_threshold=float(os.environ.get('ACCOUNT_ENTROPY_CV_THRESHOLD', '0.5')),
            interval_bin_edges=tuple(int(edge.strip()) for edge in bin_edges_raw.split(',') if edge.strip()),
            source_table=_validate_table_name(
                os.environ.get('ACCOUNT_ENTROPY_SOURCE_TABLE', 'osprey_execution_results')
            ),
            output_table=_validate_table_name(
                os.environ.get('ACCOUNT_ENTROPY_OUTPUT_TABLE', 'account_entropy_results')
            ),
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
            service_name='account-entropy',
            service_version='0.1.0',
            environment='local',
            otlp_endpoint=None,
            traces_enabled=False,
            metrics_enabled=False,
        )

    @classmethod
    def from_env(cls) -> TelemetryConfig:
        enabled = _parse_bool('ACCOUNT_ENTROPY_OTEL_ENABLED', os.environ.get('ACCOUNT_ENTROPY_OTEL_ENABLED', 'false'))
        traces_enabled = _parse_bool(
            'ACCOUNT_ENTROPY_OTEL_TRACES_ENABLED',
            os.environ.get('ACCOUNT_ENTROPY_OTEL_TRACES_ENABLED', 'true' if enabled else 'false'),
        )
        metrics_enabled = _parse_bool(
            'ACCOUNT_ENTROPY_OTEL_METRICS_ENABLED',
            os.environ.get('ACCOUNT_ENTROPY_OTEL_METRICS_ENABLED', 'true' if enabled else 'false'),
        )
        endpoint = os.environ.get('OTEL_EXPORTER_OTLP_ENDPOINT')
        if endpoint == '':
            endpoint = None
        return cls(
            enabled=enabled,
            service_name=os.environ.get('ACCOUNT_ENTROPY_OTEL_SERVICE_NAME', 'account-entropy'),
            service_version=os.environ.get('ACCOUNT_ENTROPY_OTEL_SERVICE_VERSION', '0.1.0'),
            environment=os.environ.get('ACCOUNT_ENTROPY_OTEL_ENVIRONMENT', 'local'),
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
