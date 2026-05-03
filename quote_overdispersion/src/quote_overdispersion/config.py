# pattern: Functional Core
from __future__ import annotations

import os
import re
from dataclasses import dataclass

_TABLE_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9_.]+$')


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
    volume_p_threshold: float
    density_p_threshold: float
    baseline_days: int
    cold_start_min_days: int
    min_sharers: int
    source_table: str
    output_table: str

    @classmethod
    def from_env(cls) -> AnalysisConfig:
        return cls(
            interval_seconds=int(os.environ.get('QUOTE_OVERDISPERSION_INTERVAL_SECONDS', '900')),
            volume_p_threshold=float(os.environ.get('QUOTE_OVERDISPERSION_VOLUME_P_THRESHOLD', '0.01')),
            density_p_threshold=float(os.environ.get('QUOTE_OVERDISPERSION_DENSITY_P_THRESHOLD', '0.01')),
            baseline_days=int(os.environ.get('QUOTE_OVERDISPERSION_BASELINE_DAYS', '14')),
            cold_start_min_days=int(os.environ.get('QUOTE_OVERDISPERSION_COLD_START_MIN_DAYS', '3')),
            min_sharers=int(os.environ.get('QUOTE_OVERDISPERSION_MIN_SHARERS', '3')),
            source_table=_validate_table_name(
                os.environ.get('QUOTE_OVERDISPERSION_SOURCE_TABLE', 'osprey_execution_results'),
            ),
            output_table=_validate_table_name(
                os.environ.get('QUOTE_OVERDISPERSION_OUTPUT_TABLE', 'quote_overdispersion_results'),
            ),
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
