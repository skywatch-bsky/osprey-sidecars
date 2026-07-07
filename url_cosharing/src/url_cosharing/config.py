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
    max_url_df_pctl: float
    edge_epsilon: float
    edge_quantile_grid: tuple[float, ...]
    centrality_quantile_grid: tuple[float, ...]
    density_floor: float
    max_flagged_fraction: float
    runs_table: str
    clusters_table: str
    membership_table: str
    source_table: str

    @classmethod
    def from_env(cls) -> AnalysisConfig:
        return cls(
            interval_seconds=int(os.environ.get('URL_COSHARING_INTERVAL_SECONDS', '3600')),
            resolution=float(os.environ.get('URL_COSHARING_RESOLUTION', '0.05')),
            min_cluster_size=int(os.environ.get('URL_COSHARING_MIN_CLUSTER_SIZE', '3')),
            jaccard_threshold=float(os.environ.get('URL_COSHARING_JACCARD_THRESHOLD', '0.5')),
            evolution_window_days=int(os.environ.get('URL_COSHARING_EVOLUTION_WINDOW_DAYS', '7')),
            window_days=int(os.environ.get('URL_COSHARING_WINDOW_DAYS', '7')),
            min_unique_urls=int(os.environ.get('URL_COSHARING_MIN_UNIQUE_URLS', '10')),
            min_url_sharers=int(os.environ.get('URL_COSHARING_MIN_URL_SHARERS', '5')),
            max_url_df_pctl=_validate_unit_interval(
                'URL_COSHARING_MAX_URL_DF_PCTL',
                float(os.environ.get('URL_COSHARING_MAX_URL_DF_PCTL', '0.90')),
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
                float(os.environ.get('URL_COSHARING_MAX_FLAGGED_FRACTION', '0.02')),
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
class AppConfig:
    clickhouse: ClickHouseConfig
    analysis: AnalysisConfig

    @classmethod
    def from_env(cls) -> AppConfig:
        return cls(
            clickhouse=ClickHouseConfig.from_env(),
            analysis=AnalysisConfig.from_env(),
        )
