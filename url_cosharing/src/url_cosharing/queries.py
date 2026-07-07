# pattern: Functional Core
from __future__ import annotations

from url_cosharing.config import AnalysisConfig


def fetch_pairs_query(config: AnalysisConfig) -> str:
    return f"""
        SELECT
            date,
            account_a,
            account_b,
            weight,
            newman_weight,
            shared_urls
        FROM {config.pairs_table}
        WHERE date = yesterday()
    """


def fetch_url_shares_query(config: AnalysisConfig) -> str:
    return f"""
        WITH url_shares AS (
            SELECT
                UserId AS did,
                arrayJoin(FacetLinkList) AS url,
                count() AS share_count
            FROM {config.source_table}
            WHERE Collection = 'app.bsky.feed.post'
                AND OperationKind = 'create'
                AND toDate(__timestamp) >= yesterday() - {config.window_days - 1}
                AND toDate(__timestamp) <= yesterday()
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
                AND df <= (SELECT quantile({config.max_url_df_pctl})(df) FROM url_df)
        ),
        active_accounts AS (
            SELECT did
            FROM url_shares
            GROUP BY did
            HAVING uniqExact(url) >= {config.min_unique_urls}
        )
        SELECT
            s.did,
            s.url,
            s.share_count
        FROM url_shares s
        WHERE s.url IN (SELECT url FROM eligible_urls)
            AND s.did IN (SELECT did FROM active_accounts)
    """


def fetch_historical_membership_query(config: AnalysisConfig) -> str:
    return f"""
        SELECT
            run_date,
            cluster_id,
            did
        FROM {config.membership_table}
        WHERE run_date >= today() - {config.evolution_window_days}
            AND run_date < today()
        ORDER BY run_date DESC
    """


def fetch_member_timestamps_query(config: AnalysisConfig, dids_placeholder: str) -> str:
    return f"""
        SELECT
            UserId AS did,
            __timestamp AS ts
        FROM {config.source_table}
        WHERE Collection = 'app.bsky.feed.post'
            AND OperationKind = 'create'
            AND toDate(__timestamp) = yesterday()
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
            evolution_type,
            predecessor_cluster_ids,
            jaccard_score
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
