# pattern: Imperative Shell
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Sequence

import clickhouse_connect

from url_cosharing.analyzer import EvolutionEvent, PairRow, TimestampedCluster
from url_cosharing.config import ClickHouseConfig
from url_cosharing.similarity import UrlShareRow


@dataclass(frozen=True)
class MembershipRow:
    run_date: date
    cluster_id: str
    did: str


@dataclass(frozen=True)
class MemberTimestamp:
    did: str
    ts: datetime


@dataclass(frozen=True)
class RunMetadata:
    run_date: date
    window_days: int
    accounts_raw: int
    accounts_eligible: int
    urls_eligible: int
    graph_edges: int
    edge_quantile: float
    centrality_quantile: float
    min_component_density: float
    knee_found: bool
    guardrail_triggered: bool
    flagged_accounts: int
    cluster_count: int


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

    def fetch_url_shares(self, query: str) -> list[UrlShareRow]:
        result = self._client.query(
            query,
            settings={'max_execution_time': 300},
        )
        rows = []
        for row in result.result_rows:
            rows.append(
                UrlShareRow(
                    did=row[0],
                    url=row[1],
                    share_count=int(row[2]),
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
            'mean_edge_similarity',
            'subgraph_density',
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
                cluster.mean_edge_similarity,
                cluster.subgraph_density,
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

    def insert_run(self, table: str, run: RunMetadata) -> None:
        column_names = [
            'run_date',
            'window_days',
            'accounts_raw',
            'accounts_eligible',
            'urls_eligible',
            'graph_edges',
            'edge_quantile',
            'centrality_quantile',
            'min_component_density',
            'knee_found',
            'guardrail_triggered',
            'flagged_accounts',
            'cluster_count',
        ]
        data = [[
            run.run_date,
            run.window_days,
            run.accounts_raw,
            run.accounts_eligible,
            run.urls_eligible,
            run.graph_edges,
            run.edge_quantile,
            run.centrality_quantile,
            run.min_component_density,
            run.knee_found,
            run.guardrail_triggered,
            run.flagged_accounts,
            run.cluster_count,
        ]]
        self._client.insert(table=table, data=data, column_names=column_names)

    def close(self) -> None:
        self._client.close()
