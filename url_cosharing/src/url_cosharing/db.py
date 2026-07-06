# pattern: Imperative Shell
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Sequence

import clickhouse_connect

from url_cosharing.analyzer import EvolutionEvent, PairRow, TimestampedCluster
from url_cosharing.config import ClickHouseConfig


@dataclass(frozen=True)
class MembershipRow:
    run_date: date
    cluster_id: str
    did: str


@dataclass(frozen=True)
class MemberTimestamp:
    did: str
    ts: datetime


class CosharingDb:
    def __init__(self, config: ClickHouseConfig):
        self._client = clickhouse_connect.get_client(
            host=config.host,
            port=config.port,
            username=config.user,
            password=config.password,
            database=config.database,
        )

    def fetch_pairs(self, query: str) -> list[PairRow]:
        result = self._client.query(
            query,
            settings={'max_execution_time': 120},
        )
        rows = []
        for row in result.result_rows:
            rows.append(
                PairRow(
                    date=row[0],
                    account_a=row[1],
                    account_b=row[2],
                    weight=int(row[3]),
                    newman_weight=float(row[4]),
                    shared_urls=list(row[5]) if row[5] else [],
                )
            )
        return rows

    def fetch_historical_membership(self, query: str) -> list[MembershipRow]:
        result = self._client.query(
            query,
            settings={'max_execution_time': 120},
        )
        rows = []
        for row in result.result_rows:
            rows.append(
                MembershipRow(
                    run_date=row[0],
                    cluster_id=row[1],
                    did=row[2],
                )
            )
        return rows

    def fetch_member_timestamps(self, query: str) -> list[MemberTimestamp]:
        result = self._client.query(
            query,
            settings={'max_execution_time': 120},
        )
        rows = []
        for row in result.result_rows:
            rows.append(
                MemberTimestamp(
                    did=row[0],
                    ts=row[1],
                )
            )
        return rows

    def delete_run_date(self, table: str, run_date: date) -> None:
        """Delete all rows for a given run_date, ensuring idempotent re-runs.

        Table name safety: callers pass table names from AnalysisConfig, which
        validates them against an allowlist pattern at construction time.
        """
        self._client.command(
            f'ALTER TABLE {table} DELETE WHERE run_date = {{rd:Date}}',
            parameters={'rd': run_date},
        )

    def insert_clusters(
        self,
        table: str,
        clusters: Sequence[tuple[date, TimestampedCluster, EvolutionEvent]],
    ) -> None:
        column_names = [
            'run_date',
            'cluster_id',
            'member_count',
            'total_edges',
            'total_weight',
            'unique_urls',
            'temporal_spread_hours',
            'mean_posting_interval_seconds',
            'sample_dids',
            'sample_urls',
            'resolution_parameter',
            'evolution_type',
            'predecessor_cluster_ids',
            'jaccard_score',
        ]
        data = [
            [
                run_date,
                event.cluster_id,
                cluster.member_count,
                cluster.total_edges,
                cluster.total_weight,
                cluster.unique_urls,
                cluster.temporal_spread_hours,
                cluster.mean_posting_interval_seconds,
                cluster.sample_dids,
                cluster.sample_urls,
                cluster.resolution_parameter,
                event.evolution_type,
                event.predecessor_cluster_ids,
                event.jaccard_score,
            ]
            for run_date, cluster, event in clusters
        ]
        self._client.insert(table=table, data=data, column_names=column_names)

    def insert_membership(self, table: str, membership: Sequence[tuple[date, str, str]]) -> None:
        column_names = ['run_date', 'cluster_id', 'did']
        data = [[m[0], m[1], m[2]] for m in membership]
        self._client.insert(table=table, data=data, column_names=column_names)

    def close(self) -> None:
        self._client.close()
