# pattern: Functional Core
from __future__ import annotations

from quote_overdispersion.config import AnalysisConfig


def daily_aggregation_query(config: AnalysisConfig) -> str:
    return f"""
        WITH
            domain_shares AS (
                SELECT
                    if(PostEmbedRecordUri != '', PostEmbedRecordUri, PostEmbedRecordWithMediaUri) AS quoted_uri,
                    toDate(__timestamp) AS bucket,
                    count() AS total_shares,
                    uniq(UserId) AS unique_sharers,
                    arraySlice(groupArray(DISTINCT UserId), 1, 5) AS sample_dids
                FROM {config.source_table}
                WHERE Collection = 'app.bsky.feed.post'
                    AND OperationKind = 'create'
                    AND (PostEmbedRecordUri != '' OR PostEmbedRecordWithMediaUri != '')
                    AND __timestamp >= now() - INTERVAL {config.baseline_days + 1} DAY
                GROUP BY quoted_uri, bucket
                HAVING unique_sharers >= {config.min_sharers}
            ),
            baseline AS (
                SELECT
                    quoted_uri,
                    bucket,
                    total_shares,
                    unique_sharers,
                    sample_dids,
                    toFloat64(unique_sharers) / total_shares AS sharer_density,
                    avg(total_shares) OVER w AS rolling_volume_mean,
                    avg(toFloat64(unique_sharers) / total_shares) OVER w AS rolling_density_mean,
                    count() OVER w AS baseline_buckets_available
                FROM domain_shares
                WINDOW w AS (
                    PARTITION BY quoted_uri ORDER BY bucket
                    ROWS BETWEEN {config.baseline_days} PRECEDING AND 1 PRECEDING
                )
            ),
            population_stats AS (
                SELECT
                    median(rolling_volume_mean) AS population_volume_median,
                    median(rolling_density_mean) AS population_density_median
                FROM baseline
                WHERE bucket = toDate(now())
                    AND rolling_volume_mean IS NOT NULL
                    AND baseline_buckets_available >= {config.cold_start_min_days}
            )
        SELECT
            b.quoted_uri,
            b.bucket AS bucket_start,
            b.total_shares,
            b.unique_sharers,
            b.sharer_density,
            b.rolling_volume_mean,
            b.rolling_density_mean,
            b.baseline_buckets_available AS baseline_days_available,
            b.sample_dids,
            p.population_volume_median,
            p.population_density_median
        FROM baseline b
        CROSS JOIN population_stats p
        WHERE b.bucket = toDate(now())
    """


def hourly_aggregation_query(config: AnalysisConfig) -> str:
    return f"""
        WITH
            domain_shares AS (
                SELECT
                    if(PostEmbedRecordUri != '', PostEmbedRecordUri, PostEmbedRecordWithMediaUri) AS quoted_uri,
                    toStartOfHour(__timestamp) AS bucket,
                    count() AS total_shares,
                    uniq(UserId) AS unique_sharers,
                    arraySlice(groupArray(DISTINCT UserId), 1, 5) AS sample_dids
                FROM {config.source_table}
                WHERE Collection = 'app.bsky.feed.post'
                    AND OperationKind = 'create'
                    AND (PostEmbedRecordUri != '' OR PostEmbedRecordWithMediaUri != '')
                    AND __timestamp >= now() - INTERVAL {config.baseline_days + 1} DAY
                GROUP BY quoted_uri, bucket
                HAVING unique_sharers >= {config.min_sharers}
            ),
            baseline AS (
                SELECT
                    quoted_uri,
                    bucket,
                    total_shares,
                    unique_sharers,
                    sample_dids,
                    toFloat64(unique_sharers) / total_shares AS sharer_density,
                    avg(total_shares) OVER w AS rolling_volume_mean,
                    avg(toFloat64(unique_sharers) / total_shares) OVER w AS rolling_density_mean,
                    count() OVER w AS baseline_buckets_available
                FROM domain_shares
                WINDOW w AS (
                    PARTITION BY quoted_uri ORDER BY bucket
                    ROWS BETWEEN {config.baseline_days * 24} PRECEDING AND 1 PRECEDING
                )
            ),
            population_stats AS (
                SELECT
                    median(rolling_volume_mean) AS population_volume_median,
                    median(rolling_density_mean) AS population_density_median
                FROM baseline
                WHERE bucket = toStartOfHour(now())
                    AND rolling_volume_mean IS NOT NULL
                    AND baseline_buckets_available >= {config.cold_start_min_days * 24}
            )
        SELECT
            b.quoted_uri,
            b.bucket AS bucket_start,
            b.total_shares,
            b.unique_sharers,
            b.sharer_density,
            b.rolling_volume_mean,
            b.rolling_density_mean,
            toUInt16(intDiv(b.baseline_buckets_available, 24)) AS baseline_days_available,
            b.sample_dids,
            p.population_volume_median,
            p.population_density_median
        FROM baseline b
        CROSS JOIN population_stats p
        WHERE b.bucket = toStartOfHour(now())
    """
