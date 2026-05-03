# pattern: Functional Core
from __future__ import annotations

import os
import re
from dataclasses import dataclass

_HOSTNAME_PATTERN = re.compile(r'^[a-zA-Z0-9._-]+$')
_TABLE_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9_.]+$')


def _validate_hostname(host: str) -> str:
    if not _HOSTNAME_PATTERN.match(host):
        raise ValueError(f'invalid hostname in excluded_hosts: {host!r}')
    return host


def _validate_table_name(table: str) -> str:
    if not _TABLE_NAME_PATTERN.match(table):
        raise ValueError(f'invalid table name: {table!r}')
    return table


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
    daily_p_value_threshold: float
    hourly_p_value_threshold: float
    baseline_days: int
    cold_start_min_days: int
    excluded_hosts: tuple[str, ...]
    source_table: str
    output_table: str

    @classmethod
    def from_env(cls) -> AnalysisConfig:
        excluded_raw = os.environ.get(
            'SIGNUP_ANOMALY_EXCLUDED_HOSTS',
            'bsky.network,bridgy-fed.appspot.com,mostr.pub',
        )
        return cls(
            interval_seconds=int(os.environ.get('SIGNUP_ANOMALY_INTERVAL_SECONDS', '3600')),
            daily_p_value_threshold=float(os.environ.get('SIGNUP_ANOMALY_DAILY_P_THRESHOLD', '0.01')),
            hourly_p_value_threshold=float(os.environ.get('SIGNUP_ANOMALY_HOURLY_P_THRESHOLD', '0.05')),
            baseline_days=int(os.environ.get('SIGNUP_ANOMALY_BASELINE_DAYS', '7')),
            cold_start_min_days=int(os.environ.get('SIGNUP_ANOMALY_COLD_START_MIN_DAYS', '3')),
            excluded_hosts=tuple(_validate_hostname(h.strip()) for h in excluded_raw.split(',') if h.strip()),
            source_table=_validate_table_name(
                os.environ.get('SIGNUP_ANOMALY_SOURCE_TABLE', 'osprey_execution_results')
            ),
            output_table=_validate_table_name(os.environ.get('SIGNUP_ANOMALY_OUTPUT_TABLE', 'pds_signup_anomalies')),
        )


@dataclass(frozen=True)
class AppConfig:
    clickhouse: ClickHouseConfig
    analysis: AnalysisConfig

    @classmethod
    def from_env(cls) -> AppConfig:
        return cls(
            clickhouse=ClickHouseConfig.from_env(),
            analysis=AnalysisConfig.from_env(),
        )
