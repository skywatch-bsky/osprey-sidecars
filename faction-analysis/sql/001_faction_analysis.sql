-- Faction analysis sidecar: table definitions.
--
-- The sidecar never creates its tables. clickhouse-init applies this file. Every table it writes is defined
-- here, and nothing here stores graph edges.
--
-- Two kinds of table. Run-owned tables hold rows that belong to one run. Readers see a run only through a
-- promoted row in faction_analysis_runs, so the rows of a failed run are invisible. The durable caches belong
-- to no run: a cache is keyed by its content and a retried write cannot double-count.
--
-- Cleanup of a failed run needs two privileges on these tables: ALTER DROP PARTITION on the two image tables
-- and ALTER DELETE (lightweight delete) on the other run-owned tables. Without them, failed rows stay,
-- invisible, until their TTL removes them.
--
-- Every TTL wraps its DateTime64 column in toDateTime: ClickHouse 24.1 accepts a TTL only on a DateTime or
-- a Date column.

-- 1. Runs. One row per status a run reaches. A run is promoted if it has a promoted row, else failed if it
--    has a failed row, else unresolved if it has a promoting row, else running. No TTL: the run history is
--    the audit trail and holds next_faction_id, which no later run may reuse.
CREATE TABLE IF NOT EXISTS default.faction_analysis_runs (
    run_id String,
    status Enum8('running' = 1, 'promoting' = 2, 'promoted' = 3, 'failed' = 4),
    kind Enum8('weekly' = 1, 'full' = 2),
    recorded_at DateTime64(3, 'UTC'),
    started_at DateTime64(3, 'UTC'),
    window_start DateTime64(3, 'UTC'),
    window_end DateTime64(3, 'UTC'),
    window_days UInt16,
    objective LowCardinality(String),
    resolution Float64,
    seed Int64,
    next_faction_id UInt32,
    accounts_placed UInt32,
    accounts_unplaced UInt32,
    hash_collisions UInt32,
    factions UInt32,
    moves UInt32,
    births UInt32,
    merges UInt32,
    splits UInt32,
    deaths UInt32,
    agreement_prev Nullable(Float64),
    modularity Nullable(Float64),
    source_gap_days UInt16,
    text_model_version LowCardinality(String),
    text_coverage Float64,
    stage_seconds Map(String, Float64),
    failed_stage LowCardinality(String),
    error_class LowCardinality(String)
) ENGINE = MergeTree()
ORDER BY run_id;

-- 2. The current image. One partition per run, one row per placed account. The sidecar sets retire_at to
--    eight weeks ahead for a weekly image and thirteen months ahead for a monthly image.
CREATE TABLE IF NOT EXISTS default.faction_analysis_members (
    run_id String,
    did String,
    faction_id UInt32,
    since_run_id String,
    prev_faction_id Nullable(UInt32),
    confidence Nullable(Float32),
    retire_at DateTime64(3, 'UTC')
) ENGINE = MergeTree()
PARTITION BY run_id
ORDER BY did
TTL toDateTime(retire_at) DELETE
SETTINGS ttl_only_drop_parts = 1;

-- 3. Per-account features of the same image.
CREATE TABLE IF NOT EXISTS default.faction_analysis_member_features (
    run_id String,
    faction_id UInt32,
    did String,
    degree Float64,
    within_z Nullable(Float32),
    participation Nullable(Float32),
    pileon_replies_sent UInt32,
    pileon_replies_received UInt32,
    swarm_quotes_sent UInt32,
    swarm_quotes_received UInt32,
    blocks_sent UInt32,
    blocks_received UInt32,
    text_hostility_sent Float32,
    text_hostility_received Float32,
    retire_at DateTime64(3, 'UTC')
) ENGINE = MergeTree()
PARTITION BY run_id
ORDER BY (faction_id, did)
TTL toDateTime(retire_at) DELETE
SETTINGS ttl_only_drop_parts = 1;

-- 4. Assignment changes between consecutive promoted runs. Only changes are logged.
CREATE TABLE IF NOT EXISTS default.faction_analysis_moves (
    run_id String,
    did String,
    move_kind Enum8('assign' = 1, 'move' = 2, 'drop' = 3),
    from_faction_id Nullable(UInt32),
    to_faction_id Nullable(UInt32),
    written_at DateTime64(3, 'UTC')
) ENGINE = MergeTree()
ORDER BY (did, run_id)
TTL toDateTime(written_at) + INTERVAL 2 YEAR DELETE;

-- 5. Lineage events. related_faction_id is the parent of a split and the winner of a merge.
CREATE TABLE IF NOT EXISTS default.faction_analysis_lineage (
    run_id String,
    faction_id UInt32,
    event Enum8('birth' = 1, 'split' = 2, 'merge' = 3, 'death' = 4),
    related_faction_id Nullable(UInt32),
    written_at DateTime64(3, 'UTC')
) ENGINE = MergeTree()
ORDER BY (run_id, faction_id)
TTL toDateTime(written_at) + INTERVAL 2 YEAR DELETE;

-- 6. The faction registry. One immutable row per faction per run in which it appears, plus a row when it
--    merges or retires. The current state of a faction is its latest row among promoted runs.
CREATE TABLE IF NOT EXISTS default.faction_analysis_factions (
    faction_id UInt32,
    run_id String,
    status Enum8('active' = 1, 'merged' = 2, 'retired' = 3),
    parent_faction_id Nullable(UInt32),
    born_run_id String,
    member_count UInt32,
    top_tags Array(String),
    top_languages Array(String),
    sample_handles Array(String),
    written_at DateTime64(3, 'UTC')
) ENGINE = MergeTree()
ORDER BY (faction_id, run_id)
TTL toDateTime(written_at) + INTERVAL 2 YEAR DELETE;

-- 7. Per-run faction statistics.
CREATE TABLE IF NOT EXISTS default.faction_analysis_faction_runs (
    run_id String,
    faction_id UInt32,
    member_count UInt32,
    mean_degree Float64,
    hostile_sent UInt32,
    hostile_received UInt32,
    conflict_cells UInt32,
    max_conflict_score Nullable(Float32),
    written_at DateTime64(3, 'UTC')
) ENGINE = MergeTree()
ORDER BY (run_id, faction_id)
TTL toDateTime(written_at) + INTERVAL 2 YEAR DELETE;

-- 8. The conflict matrix. Ordered faction pairs including the diagonal, one set per conflict window.
CREATE TABLE IF NOT EXISTS default.faction_analysis_conflict (
    run_id String,
    window_days UInt16,
    from_faction_id UInt32,
    to_faction_id UInt32,
    contact UInt32,
    contact_expected Float64,
    contact_lift Nullable(Float64),
    pileon_replies UInt32,
    pileon_lift Nullable(Float64),
    direction_pileon Nullable(Float64),
    swarm_quotes UInt32,
    swarm_lift Nullable(Float64),
    direction_swarm Nullable(Float64),
    blocks UInt32,
    blocks_lift Nullable(Float64),
    direction_blocks Nullable(Float64),
    text_hostility Float64,
    text_scored UInt32,
    text_lift Nullable(Float64),
    direction_text Nullable(Float64),
    conflict_score Nullable(Float64),
    distinct_sources UInt32,
    distinct_targets UInt32,
    written_at DateTime64(3, 'UTC')
) ENGINE = MergeTree()
ORDER BY (window_days, run_id, from_faction_id, to_faction_id)
TTL toDateTime(written_at) + INTERVAL 13 MONTH DELETE;

-- 9. Durable cache: every scored interaction, without text. Text is stored for examples only, so that a
--    moderator can see why a cell is flagged. No factions: attribution always comes from the current image.
CREATE TABLE IF NOT EXISTS default.faction_analysis_text_scores (
    uri_hash UInt64,
    h Float32,
    model_version LowCardinality(String),
    kind Enum8('reply' = 1, 'quote' = 2),
    scored_at DateTime64(3, 'UTC'),
    is_example UInt8,
    author_did Nullable(String),
    target_did Nullable(String),
    post_text Nullable(String),
    event_at Nullable(DateTime64(3, 'UTC'))
) ENGINE = MergeTree()
ORDER BY uri_hash
TTL toDateTime(scored_at) + INTERVAL 70 DAY DELETE;

-- 10. Durable cache: paid Jev labels. A pair is never paid for twice.
CREATE TABLE IF NOT EXISTS default.faction_analysis_jev_labels (
    uri_hash UInt64,
    prompt_version LowCardinality(String),
    h Float32,
    p_hostile Float32,
    p_threat Float32,
    model LowCardinality(String),
    input_tokens UInt32,
    labelled_at DateTime64(3, 'UTC')
) ENGINE = MergeTree()
ORDER BY uri_hash
TTL toDateTime(labelled_at) + INTERVAL 400 DAY DELETE;

-- 11. Durable cache: the training set of every model refresh. parent_uri lets a later export join the text
--     of the parent post. No DID or handle is stored: author_did and target_did are hashes of the DIDs.
CREATE TABLE IF NOT EXISTS default.faction_analysis_scorer_dataset (
    dataset_version LowCardinality(String),
    uri_hash UInt64,
    kind Enum8('reply' = 1, 'quote' = 2),
    split Enum8('train' = 1, 'calibration' = 2, 'test' = 3),
    label_source Enum8('jev' = 1, 'block' = 2, 'random' = 3),
    target UInt8,
    weight Float32,
    label_h Nullable(Float32),
    author_did UInt64,
    target_did UInt64,
    text String,
    parent_uri String,
    event_ms Int64,
    languages Array(String),
    text_length UInt32,
    uppercase UInt32,
    exclamations UInt32,
    questions UInt32,
    mentions UInt32,
    written_at DateTime64(3, 'UTC')
) ENGINE = MergeTree()
ORDER BY (dataset_version, uri_hash)
TTL toDateTime(written_at) + INTERVAL 400 DAY DELETE;

-- Grants for cleanup of failed runs (run once, by an administrator):
--   GRANT ALTER DROP PARTITION ON default.faction_analysis_members TO <sidecar_user>;
--   GRANT ALTER DROP PARTITION ON default.faction_analysis_member_features TO <sidecar_user>;
--   GRANT ALTER DELETE ON default.faction_analysis_moves TO <sidecar_user>;
--   GRANT ALTER DELETE ON default.faction_analysis_lineage TO <sidecar_user>;
--   GRANT ALTER DELETE ON default.faction_analysis_factions TO <sidecar_user>;
--   GRANT ALTER DELETE ON default.faction_analysis_faction_runs TO <sidecar_user>;
--   GRANT ALTER DELETE ON default.faction_analysis_conflict TO <sidecar_user>;
--   GRANT ALTER DELETE ON default.faction_analysis_runs TO <sidecar_user>;
