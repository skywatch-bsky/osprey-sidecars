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
    window_days: int
    min_posts: int
    hourly_entropy_threshold: float
    interval_entropy_threshold: float
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
            hourly_entropy_threshold=float(os.environ.get('ACCOUNT_ENTROPY_HOURLY_ENTROPY_THRESHOLD', '3.9')),
            interval_entropy_threshold=float(
                os.environ.get('ACCOUNT_ENTROPY_INTERVAL_ENTROPY_THRESHOLD', '1.5')
            ),
            interval_bin_edges=tuple(
                int(edge.strip()) for edge in bin_edges_raw.split(',') if edge.strip()
            ),
            source_table=_validate_table_name(
                os.environ.get('ACCOUNT_ENTROPY_SOURCE_TABLE', 'osprey_execution_results')
            ),
            output_table=_validate_table_name(
                os.environ.get('ACCOUNT_ENTROPY_OUTPUT_TABLE', 'account_entropy_results')
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
