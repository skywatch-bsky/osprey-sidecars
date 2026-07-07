# pattern: Functional Core
from __future__ import annotations

from datetime import date, timedelta

from url_cosharing.config import AnalysisConfig


def _window_bounds(config: AnalysisConfig, as_of: date) -> tuple[date, date]:
    """Detection window for a run_date: the window_days days ending the day
    before as_of. Explicit date literals (rather than yesterday()) keep the
    query anchored to the run being computed, which makes historical backfill
    a matter of passing a past as_of.
    """
    window_end = as_of - timedelta(days=1)
    window_start = window_end - timedelta(days=config.window_days - 1)
    return window_start, window_end


def fetch_url_shares_query(config: AnalysisConfig, as_of: date) -> str:
    window_start, window_end = _window_bounds(config, as_of)
    return f"""
        WITH url_shares AS (
            SELECT
                UserId AS did,
                arrayJoin(FacetLinkList) AS url,
                count() AS share_count
            FROM {config.source_table}
            WHERE Collection = 'app.bsky.feed.post'
                AND OperationKind = 'create'
                AND toDate(__timestamp) >= toDate('{window_start}')
                AND toDate(__timestamp) <= toDate('{window_end}')
                AND length(FacetLinkList) > 0
            GROUP BY did, url
        ),
        url_df AS (
            SELECT url, uniqExact(did) AS df
            FROM url_shares
            GROUP BY url
        ),
        eligible_urls AS (
            SELECT url
            FROM url_df
            WHERE df >= {config.min_url_sharers}
                AND df <= {config.max_url_df_fraction} * (SELECT uniqExact(did) FROM url_shares)
        ),
        active_accounts AS (
            SELECT did
            FROM url_shares
            GROUP BY did
            HAVING uniqExact(url) >= {config.min_unique_urls}
        ),
        eligible_shares AS (
            SELECT
                s.did,
                s.url,
                s.share_count
            FROM url_shares s
            WHERE s.url IN (SELECT url FROM eligible_urls)
                AND s.did IN (SELECT did FROM active_accounts)
        )
        SELECT
            did,
            url,
            share_count
        FROM eligible_shares
        WHERE did IN (
            -- Cosine similarity over a 1-sparse TF-IDF vector is degenerate
            -- (exactly 0 or 1 after normalization), so an account whose
            -- eligible URL set collapsed to a single URL carries no gradable
            -- co-sharing evidence and would form an artificial similarity-1.0
            -- clique with every other sharer of that URL. The bound is the
            -- mathematical minimum for cosine to grade, not a tuning knob.
            SELECT did
            FROM eligible_shares
            GROUP BY did
            HAVING uniqExact(url) >= 2
        )
    """


def fetch_raw_account_count_query(config: AnalysisConfig, as_of: date) -> str:
    """Distinct accounts in the raw rolling window, before eligibility filters.

    Mirrors the url_shares CTE population in fetch_url_shares_query. Run
    metadata reports this alongside accounts_eligible to expose stage-count
    attrition; the main query's final output cannot provide it because its
    rows are already filtered.
    """
    window_start, window_end = _window_bounds(config, as_of)
    return f"""
        SELECT uniqExact(UserId)
        FROM {config.source_table}
        WHERE Collection = 'app.bsky.feed.post'
            AND OperationKind = 'create'
            AND toDate(__timestamp) >= toDate('{window_start}')
            AND toDate(__timestamp) <= toDate('{window_end}')
            AND length(FacetLinkList) > 0
    """


def fetch_historical_membership_query(config: AnalysisConfig, as_of: date) -> str:
    return f"""
        SELECT
            run_date,
            cluster_id,
            did
        FROM {config.membership_table}
        WHERE run_date >= toDate('{as_of}') - {config.evolution_window_days}
            AND run_date < toDate('{as_of}')
        ORDER BY run_date DESC
    """


def fetch_member_timestamps_query(config: AnalysisConfig, dids_placeholder: str, as_of: date) -> str:
    window_start, window_end = _window_bounds(config, as_of)
    return f"""
        SELECT
            UserId AS did,
            __timestamp AS ts
        FROM {config.source_table}
        WHERE Collection = 'app.bsky.feed.post'
            AND OperationKind = 'create'
            AND toDate(__timestamp) >= toDate('{window_start}')
            AND toDate(__timestamp) <= toDate('{window_end}')
            AND UserId IN ({dids_placeholder})
        ORDER BY did, ts
    """


def insert_clusters_query(config: AnalysisConfig) -> str:
    return f"""
        INSERT INTO {config.clusters_table}
        (
            run_date,
            cluster_id,
            member_count,
            total_edges,
            total_weight,
            unique_urls,
            temporal_spread_hours,
            mean_posting_interval_seconds,
            sample_dids,
            sample_urls,
            resolution_parameter,
            mean_edge_similarity,
            subgraph_density,
            evolution_type,
            predecessor_cluster_ids,
            jaccard_score
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """


def insert_membership_query(config: AnalysisConfig) -> str:
    return f"""
        INSERT INTO {config.membership_table}
        (
            run_date,
            cluster_id,
            did
        )
        VALUES (?, ?, ?)
    """
