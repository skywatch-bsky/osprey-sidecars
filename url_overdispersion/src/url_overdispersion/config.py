# pattern: Functional Core
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

_HOSTNAME_PATTERN = re.compile(r'^[a-zA-Z0-9._-]+$')
_TABLE_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9_.]+$')


def _validate_hostname(host: str) -> str:
    if not _HOSTNAME_PATTERN.match(host):
        raise ValueError(f'invalid hostname in watchlist_domains: {host!r}')
    return host


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
    volume_p_threshold: float
    density_p_threshold: float
    baseline_days: int
    cold_start_min_days: int
    min_sharers: int
    watchlist_domains: tuple[str, ...]
    source_table: str
    output_table: str

    @classmethod
    def from_env(cls) -> AnalysisConfig:
        watchlist_raw = os.environ.get('URL_OVERDISPERSION_WATCHLIST_DOMAINS', '')
        return cls(
            interval_seconds=int(os.environ.get('URL_OVERDISPERSION_INTERVAL_SECONDS', '900')),
            volume_p_threshold=float(os.environ.get('URL_OVERDISPERSION_VOLUME_P_THRESHOLD', '0.01')),
            density_p_threshold=float(os.environ.get('URL_OVERDISPERSION_DENSITY_P_THRESHOLD', '0.01')),
            baseline_days=int(os.environ.get('URL_OVERDISPERSION_BASELINE_DAYS', '14')),
            cold_start_min_days=int(os.environ.get('URL_OVERDISPERSION_COLD_START_MIN_DAYS', '3')),
            min_sharers=int(os.environ.get('URL_OVERDISPERSION_MIN_SHARERS', '3')),
            watchlist_domains=tuple(_validate_hostname(h.strip()) for h in watchlist_raw.split(',') if h.strip()),
            source_table=_validate_table_name(
                os.environ.get('URL_OVERDISPERSION_SOURCE_TABLE', 'osprey_execution_results')
            ),
            output_table=_validate_table_name(
                os.environ.get('URL_OVERDISPERSION_OUTPUT_TABLE', 'url_overdispersion_results')
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
            service_name='url-overdispersion',
            service_version='0.1.0',
            environment='local',
            otlp_endpoint=None,
            traces_enabled=False,
            metrics_enabled=False,
        )

    @classmethod
    def from_env(cls) -> TelemetryConfig:
        enabled = _parse_bool(
            'URL_OVERDISPERSION_OTEL_ENABLED', os.environ.get('URL_OVERDISPERSION_OTEL_ENABLED', 'false')
        )
        traces_enabled = _parse_bool(
            'URL_OVERDISPERSION_OTEL_TRACES_ENABLED',
            os.environ.get('URL_OVERDISPERSION_OTEL_TRACES_ENABLED', 'true' if enabled else 'false'),
        )
        metrics_enabled = _parse_bool(
            'URL_OVERDISPERSION_OTEL_METRICS_ENABLED',
            os.environ.get('URL_OVERDISPERSION_OTEL_METRICS_ENABLED', 'true' if enabled else 'false'),
        )
        endpoint = os.environ.get('OTEL_EXPORTER_OTLP_ENDPOINT')
        if endpoint == '':
            endpoint = None
        return cls(
            enabled=enabled,
            service_name=os.environ.get('URL_OVERDISPERSION_OTEL_SERVICE_NAME', 'url-overdispersion'),
            service_version=os.environ.get('URL_OVERDISPERSION_OTEL_SERVICE_VERSION', '0.1.0'),
            environment=os.environ.get('URL_OVERDISPERSION_OTEL_ENVIRONMENT', 'local'),
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
