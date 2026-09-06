# pattern: Functional Core
from __future__ import annotations

from datetime import date, timedelta

from url_cosharing.config import AnalysisConfig, Exclusions


def _window_bounds(config: AnalysisConfig, as_of: date) -> tuple[date, date]:
    """Detection window for a run_date: the window_days days ending the day
    before as_of. Explicit date literals (rather than yesterday()) keep the
    query anchored to the run being computed, which makes historical backfill
    a matter of passing a past as_of.
    """
    window_end = as_of - timedelta(days=1)
    window_start = window_end - timedelta(days=config.window_days - 1)
    return window_start, window_end


def _exclusion_keep_predicates(exclusions: Exclusions) -> list[str]:
    """Per-entry predicates a share row must satisfy to be kept.

    Domain entries match the exact host OR a subdomain of it: the leading
    dot in '.{domain}' is what anchors the suffix match, so 'klipy.com'
    matches 'static.klipy.com' but can never match 'evil-klipy.com'. Entries
    are validated against _DOMAIN_PATTERN/_DID_PATTERN at parse time, which
    excludes SQL metacharacters and makes direct interpolation safe.
    """
    predicates: list[str] = []
    for domain in exclusions.excluded_domains:
        predicates.append(f"NOT (lower(domain(url)) = '{domain}' OR endsWith(lower(domain(url)), '.{domain}'))")
    if exclusions.excluded_dids:
        dids = ','.join(f"'{did}'" for did in exclusions.excluded_dids)
        predicates.append(f'did NOT IN ({dids})')
    return predicates


def _exclusion_match_clause(exclusions: Exclusions) -> str:
    """Positive predicate matching suppressed rows (any exclusion hits).

    The logical negation of the keep predicates ANDed together — used by the
    audit count query to measure suppressed share rows per run.
    """
    clauses: list[str] = []
    for domain in exclusions.excluded_domains:
        clauses.append(f"(lower(domain(url)) = '{domain}' OR endsWith(lower(domain(url)), '.{domain}'))")
    if exclusions.excluded_dids:
        dids = ','.join(f"'{did}'" for did in exclusions.excluded_dids)
        clauses.append(f'did IN ({dids})')
    return '(' + '\n            OR '.join(clauses) + ')'


def _expanded_shares_cte(config: AnalysisConfig, as_of: date, indent: str) -> str:
    """One row per (account, URL facet) share in the detection window.

    Splitting expansion from aggregation keeps exclusion predicates plain
    column references (no alias-in-WHERE scoping questions) and leaves the
    query byte-identical to the pre-exclusions shape when exclusions are
    empty. `indent` aligns the CTE inside its parent query.
    """
    window_start, window_end = _window_bounds(config, as_of)
    lines = [
        'expanded AS (',
        f'{indent}    SELECT',
        f"{indent}        UserId AS did,",
        f'{indent}        arrayJoin(FacetLinkList) AS url',
        f'{indent}    FROM {config.source_table}',
        f"{indent}    WHERE Collection = 'app.bsky.feed.post'",
        f"{indent}        AND OperationKind = 'create'",
        f"{indent}        AND toDate(__timestamp) >= toDate('{window_start}')",
        f"{indent}        AND toDate(__timestamp) <= toDate('{window_end}')",
        f'{indent}        AND length(FacetLinkList) > 0',
        f'{indent})',
    ]
    return '\n'.join(lines)


def fetch_url_shares_query(config: AnalysisConfig, as_of: date, exclusions: Exclusions = Exclusions.empty()) -> str:
    expanded_cte = _expanded_shares_cte(config, as_of, indent='        ')
    keep_predicates = _exclusion_keep_predicates(exclusions)
    if keep_predicates:
        where_clause = '            WHERE ' + '\n                AND '.join(keep_predicates)
    else:
        where_clause = ''
    return f"""
        WITH {expanded_cte},
        url_shares AS (
            SELECT
                did,
                url,
                count() AS share_count
            FROM expanded
{where_clause}
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


def fetch_excluded_shares_count_query(
    config: AnalysisConfig, as_of: date, exclusions: Exclusions
) -> str:
    """Count share rows suppressed by the applied exclusions in this window.

    Runs only when exclusions are non-empty; one extra windowed aggregate per
    cycle (accepted cost) feeding url_cosharing_runs.excluded_shares_suppressed.
    """
    expanded_cte = _expanded_shares_cte(config, as_of, indent='    ')
    match_clause = _exclusion_match_clause(exclusions)
    return f"""
        WITH {expanded_cte}
        SELECT count()
        FROM expanded
        WHERE {match_clause}
    """


def fetch_raw_account_count_query(config: AnalysisConfig, as_of: date) -> str:
    """Distinct accounts in the raw rolling window, before eligibility filters.

    Deliberately the PRE-EXCLUSION window population (issue #4 decision):
    exclusion predicates apply only inside fetch_url_shares_query, so the
    "mirrors the url_shares CTE" property no longer holds. Exclusion
    suppression intentionally surfaces as extra accounts_raw ->
    accounts_eligible attrition, quantified per run by
    excluded_shares_suppressed in fetch_excluded_shares_count_query. Run
    metadata reports both alongside accounts_eligible to expose stage-count
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
