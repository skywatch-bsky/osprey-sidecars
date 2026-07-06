# pattern: Functional Core
from __future__ import annotations

from quote_overdispersion.config import AnalysisConfig


def daily_aggregation_query(config: AnalysisConfig) -> str:
    return f"""
        WITH raw_shares AS (
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
        ),
        scored_entities AS (
            SELECT quoted_uri
            FROM raw_shares
            WHERE bucket = toDate(now()) AND unique_sharers >= {config.min_sharers}
        ),
        entities AS (
            SELECT r.quoted_uri AS quoted_uri, min(r.bucket) AS first_seen
            FROM raw_shares r
            INNER JOIN scored_entities s ON r.quoted_uri = s.quoted_uri
            GROUP BY r.quoted_uri
        ),
        calendar AS (
            SELECT toDate(now()) - number AS bucket FROM numbers({config.baseline_days + 1})
        ),
        dense AS (
            SELECT
                e.quoted_uri AS quoted_uri,
                c.bucket AS bucket,
                coalesce(r.total_shares, 0) AS total_shares,
                coalesce(r.unique_sharers, 0) AS unique_sharers,
                if(coalesce(r.total_shares, 0) > 0, toFloat64(r.unique_sharers) / r.total_shares, NULL) AS sharer_density,
                r.sample_dids AS sample_dids
            FROM entities e
            CROSS JOIN calendar c
            LEFT JOIN raw_shares r ON r.quoted_uri = e.quoted_uri AND r.bucket = c.bucket
            WHERE c.bucket >= e.first_seen
        ),
        baseline AS (
            SELECT
                quoted_uri, bucket, total_shares, unique_sharers, sharer_density, sample_dids,
                medianExact(total_shares) OVER w AS rolling_volume_median,
                avg(total_shares) OVER w AS rolling_volume_mean,
                ifNotFinite(varPop(total_shares) OVER w, NULL) AS rolling_volume_variance,
                avg(sharer_density) OVER w AS rolling_density_mean,
                ifNotFinite(varPop(sharer_density) OVER w, NULL) AS rolling_density_variance,
                count() OVER w AS baseline_buckets_available
            FROM dense
            WINDOW w AS (
                PARTITION BY quoted_uri
                ORDER BY bucket
                ROWS BETWEEN {config.baseline_days} PRECEDING AND 1 PRECEDING
            )
        ),
        population_stats AS (
            SELECT
                median(rolling_volume_median) AS population_volume_median,
                median(if(rolling_volume_mean > 0, rolling_volume_variance / rolling_volume_mean, NULL)) AS population_volume_dispersion,
                median(rolling_density_mean) AS population_density_median,
                median(rolling_density_variance) AS population_density_variance
            FROM baseline
            WHERE bucket = toDate(now())
                AND baseline_buckets_available >= {config.cold_start_min_days}
        )
        SELECT
            b.quoted_uri,
            b.bucket AS bucket_start,
            b.total_shares,
            b.unique_sharers,
            coalesce(b.sharer_density, 0) AS sharer_density,
            b.rolling_volume_median,
            b.rolling_volume_mean,
            b.rolling_volume_variance,
            b.rolling_density_mean,
            b.rolling_density_variance,
            toUInt16(b.baseline_buckets_available) AS baseline_days_available,
            b.sample_dids,
            p.population_volume_median,
            p.population_volume_dispersion,
            p.population_density_median,
            p.population_density_variance
        FROM baseline b
        CROSS JOIN population_stats p
        WHERE b.bucket = toDate(now()) AND b.unique_sharers >= {config.min_sharers}
    """


def hourly_aggregation_query(config: AnalysisConfig) -> str:
    return f"""
        WITH raw_shares AS (
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
        ),
        scored_entities AS (
            SELECT quoted_uri
            FROM raw_shares
            WHERE bucket = toStartOfHour(now()) AND unique_sharers >= {config.min_sharers}
        ),
        entities AS (
            SELECT r.quoted_uri AS quoted_uri, min(r.bucket) AS first_seen
            FROM raw_shares r
            INNER JOIN scored_entities s ON r.quoted_uri = s.quoted_uri
            GROUP BY r.quoted_uri
        ),
        calendar AS (
            SELECT toStartOfHour(now()) - toIntervalHour(number) AS bucket FROM numbers({(config.baseline_days + 1) * 24})
        ),
        dense AS (
            SELECT
                e.quoted_uri AS quoted_uri,
                c.bucket AS bucket,
                coalesce(r.total_shares, 0) AS total_shares,
                coalesce(r.unique_sharers, 0) AS unique_sharers,
                if(coalesce(r.total_shares, 0) > 0, toFloat64(r.unique_sharers) / r.total_shares, NULL) AS sharer_density,
                r.sample_dids AS sample_dids
            FROM entities e
            CROSS JOIN calendar c
            LEFT JOIN raw_shares r ON r.quoted_uri = e.quoted_uri AND r.bucket = c.bucket
            WHERE c.bucket >= e.first_seen
        ),
        baseline AS (
            SELECT
                quoted_uri, bucket, total_shares, unique_sharers, sharer_density, sample_dids,
                medianExact(total_shares) OVER w AS rolling_volume_median,
                avg(total_shares) OVER w AS rolling_volume_mean,
                ifNotFinite(varPop(total_shares) OVER w, NULL) AS rolling_volume_variance,
                avg(sharer_density) OVER w AS rolling_density_mean,
                ifNotFinite(varPop(sharer_density) OVER w, NULL) AS rolling_density_variance,
                count() OVER w AS baseline_buckets_available
            FROM dense
            WINDOW w AS (
                PARTITION BY quoted_uri, toHour(bucket)
                ORDER BY bucket
                ROWS BETWEEN {config.baseline_days} PRECEDING AND 1 PRECEDING
            )
        ),
        population_stats AS (
            SELECT
                median(rolling_volume_median) AS population_volume_median,
                median(if(rolling_volume_mean > 0, rolling_volume_variance / rolling_volume_mean, NULL)) AS population_volume_dispersion,
                median(rolling_density_mean) AS population_density_median,
                median(rolling_density_variance) AS population_density_variance
            FROM baseline
            WHERE bucket = toStartOfHour(now())
                AND baseline_buckets_available >= {config.cold_start_min_days}
        )
        SELECT
            b.quoted_uri,
            b.bucket AS bucket_start,
            b.total_shares,
            b.unique_sharers,
            coalesce(b.sharer_density, 0) AS sharer_density,
            b.rolling_volume_median,
            b.rolling_volume_mean,
            b.rolling_volume_variance,
            b.rolling_density_mean,
            b.rolling_density_variance,
            toUInt16(b.baseline_buckets_available) AS baseline_days_available,
            b.sample_dids,
            p.population_volume_median,
            p.population_volume_dispersion,
            p.population_density_median,
            p.population_density_variance
        FROM baseline b
        CROSS JOIN population_stats p
        WHERE b.bucket = toStartOfHour(now()) AND b.unique_sharers >= {config.min_sharers}
    """
