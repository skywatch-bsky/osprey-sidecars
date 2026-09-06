"""Live-ClickHouse verification of exclusion predicate SQL semantics (AC.8).

Gated: SKIPPED unless URL_COSHARING_INTEGRATION_CH_HOST is set. Point it at
any reachable ClickHouse (staging or a local container); connection details
otherwise follow the OSPREY_CLICKHOUSE_* env vars with the integration host
overriding OSPREY_CLICKHOUSE_HOST. Uses clickhouse-connect, an existing
runtime dependency — no test-only deps.

This is the only automated check that exercises actual domain()/endsWith()
behavior; the unit suite asserts on generated SQL strings. It seeds a scratch
table, runs the real generated queries with klipy.com-style exclusions, and
verifies:

- exact-host and dot-anchored subdomain exclusion (klipy.com removes
  klipy.com, static.klipy.com, sub.static.klipy.com)
- suffix safety (evil-klipy.com must survive)
- exact DID exclusion (did:plc:excludeme removed)
- the suppressed-shares audit count matches the removed rows

The scratch table is dropped after the run.
"""
from __future__ import annotations

import os
from datetime import date, datetime

import pytest

pytest.importorskip('clickhouse_connect')


from url_cosharing.config import AnalysisConfig, ClickHouseConfig, Exclusions  # noqa: E402
from url_cosharing.db import CosharingDb  # noqa: E402
from url_cosharing.queries import (  # noqa: E402
    fetch_excluded_shares_count_query,
    fetch_url_shares_query,
)

INTEGRATION_HOST = os.environ.get('URL_COSHARING_INTEGRATION_CH_HOST')
SCRATCH_TABLE = 'url_cosharing_exclusions_it'

pytestmark = pytest.mark.skipif(
    not INTEGRATION_HOST,
    reason='set URL_COSHARING_INTEGRATION_CH_HOST to run the exclusion predicate against live ClickHouse',
)


@pytest.fixture
def db() -> CosharingDb:
    config = ClickHouseConfig.from_env()
    config = ClickHouseConfig(
        host=INTEGRATION_HOST,
        port=config.port,
        user=config.user,
        password=config.password,
        database=config.database,
    )
    client = CosharingDb(config)
    yield client
    client._client.command(f'DROP TABLE IF EXISTS default.{SCRATCH_TABLE}')
    client.close()


def _integration_config() -> AnalysisConfig:
    return AnalysisConfig(
        interval_seconds=3600,
        resolution=0.05,
        min_cluster_size=3,
        jaccard_threshold=0.5,
        evolution_window_days=7,
        window_days=1,  # window = exactly 2026-07-07, matching the seeded rows
        min_unique_urls=1,
        min_url_sharers=1,
        max_url_df_fraction=1.0,
        edge_epsilon=0.05,
        edge_quantile_grid=(0.5,),
        centrality_quantile_grid=(0.5,),
        density_floor=0.5,
        max_flagged_fraction=1.0,
        runs_table='url_cosharing_runs',
        clusters_table='url_cosharing_clusters',
        membership_table='url_cosharing_membership',
        source_table=SCRATCH_TABLE,
    )


def test_exclusion_predicates_against_live_clickhouse(db: CosharingDb) -> None:
    client = db._client
    client.command(f'DROP TABLE IF EXISTS default.{SCRATCH_TABLE}')
    client.command(f"""
        CREATE TABLE default.{SCRATCH_TABLE} (
            __timestamp DateTime,
            UserId String,
            Collection String,
            OperationKind String,
            FacetLinkList Array(String)
        )
        ENGINE = MergeTree()
        ORDER BY __timestamp
    """)

    rows = [
        # did:plc:sharer shares one URL from each host class under test
        (datetime(2026, 7, 7, 10, 0, 0), 'did:plc:sharer', 'app.bsky.feed.post', 'create', ['https://klipy.com/gif.gif']),
        (datetime(2026, 7, 7, 10, 1, 0), 'did:plc:sharer', 'app.bsky.feed.post', 'create', ['https://static.klipy.com/gif.gif']),
        (datetime(2026, 7, 7, 10, 2, 0), 'did:plc:sharer', 'app.bsky.feed.post', 'create', ['https://sub.static.klipy.com/gif.gif']),
        (datetime(2026, 7, 7, 10, 3, 0), 'did:plc:sharer', 'app.bsky.feed.post', 'create', ['https://evil-klipy.com/gif.gif']),
        (datetime(2026, 7, 7, 10, 4, 0), 'did:plc:sharer', 'app.bsky.feed.post', 'create', ['https://example.com/page.gif']),
        # a did-excluded account sharing a benign URL
        (datetime(2026, 7, 7, 10, 5, 0), 'did:plc:excludeme', 'app.bsky.feed.post', 'create', ['https://example.com/mine.gif']),
    ]
    client.insert(
        table=SCRATCH_TABLE,
        data=rows,
        column_names=['__timestamp', 'UserId', 'Collection', 'OperationKind', 'FacetLinkList'],
    )

    config = _integration_config()
    exclusions = Exclusions.from_mapping(
        {
            'excluded_domains': ['klipy.com'],
            'excluded_dids': ['did:plc:excludeme'],
        }
    )

    shares = db.fetch_url_shares(fetch_url_shares_query(config, date(2026, 7, 8), exclusions))
    returned = {(row.did, row.url) for row in shares}

    # Dot-anchored suffix: entry klipy.com removes the host and all subdomains…
    assert ('did:plc:sharer', 'https://klipy.com/gif.gif') not in returned
    assert ('did:plc:sharer', 'https://static.klipy.com/gif.gif') not in returned
    assert ('did:plc:sharer', 'https://sub.static.klipy.com/gif.gif') not in returned
    # …but never a lookalike host with klipy.com as a bare substring.
    assert ('did:plc:sharer', 'https://evil-klipy.com/gif.gif') in returned
    # Benign URL survives.
    assert ('did:plc:sharer', 'https://example.com/page.gif') in returned
    # Exact DID exclusion removes the account entirely.
    assert not any(did == 'did:plc:excludeme' for did, _ in returned)

    suppressed = db.fetch_excluded_shares_count(
        fetch_excluded_shares_count_query(config, date(2026, 7, 8), exclusions)
    )
    # 3 klipy-family rows + 1 did-excluded row
    assert suppressed == 4
