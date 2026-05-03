# pattern: Functional Core
from __future__ import annotations

from account_entropy.config import AnalysisConfig


def account_activity_query(config: AnalysisConfig) -> str:
    return f"""
        WITH
            active_accounts AS (
                SELECT
                    UserId AS user_id,
                    count() AS post_count
                FROM {config.source_table}
                WHERE Collection = 'app.bsky.feed.post'
                    AND OperationKind = 'create'
                    AND __timestamp >= now() - INTERVAL {config.window_days} DAY
                    AND UserId IS NOT NULL
                GROUP BY UserId
                HAVING post_count >= {config.min_posts}
            ),
            account_data AS (
                SELECT
                    e.UserId AS user_id,
                    a.post_count,
                    groupArray(toUInt8(toHour(e.__timestamp))) AS hourly_bins,
                    arraySort(groupArray(toUnixTimestamp64Milli(e.__timestamp))) AS ordered_timestamps,
                    arraySlice(groupArray(e.__action_id), 1, 5) AS sample_rkeys
                FROM {config.source_table} e
                INNER JOIN active_accounts a ON e.UserId = a.user_id
                WHERE e.Collection = 'app.bsky.feed.post'
                    AND e.OperationKind = 'create'
                    AND e.__timestamp >= now() - INTERVAL {config.window_days} DAY
                GROUP BY e.UserId, a.post_count
            )
        SELECT
            user_id,
            post_count,
            hourly_bins,
            ordered_timestamps,
            sample_rkeys
        FROM account_data
    """
