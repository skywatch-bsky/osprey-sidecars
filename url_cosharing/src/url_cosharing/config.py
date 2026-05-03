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
    resolution: float
    min_edge_weight: int
    min_cluster_size: int
    min_cosharers: int
    jaccard_threshold: float
    evolution_window_days: int
    pairs_table: str
    clusters_table: str
    membership_table: str
    source_table: str

    @classmethod
    def from_env(cls) -> AnalysisConfig:
        return cls(
            interval_seconds=int(os.environ.get('URL_COSHARING_INTERVAL_SECONDS', '3600')),
            resolution=float(os.environ.get('URL_COSHARING_RESOLUTION', '0.05')),
            min_edge_weight=int(os.environ.get('URL_COSHARING_MIN_EDGE_WEIGHT', '2')),
            min_cluster_size=int(os.environ.get('URL_COSHARING_MIN_CLUSTER_SIZE', '3')),
            min_cosharers=int(os.environ.get('URL_COSHARING_MIN_COSHARERS', '3')),
            jaccard_threshold=float(os.environ.get('URL_COSHARING_JACCARD_THRESHOLD', '0.5')),
            evolution_window_days=int(os.environ.get('URL_COSHARING_EVOLUTION_WINDOW_DAYS', '7')),
            pairs_table=_validate_table_name(os.environ.get('URL_COSHARING_PAIRS_TABLE', 'url_cosharing_pairs')),
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
