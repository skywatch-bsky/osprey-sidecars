"""Column order of every insert and read.

Types and retention live in ``sql/001_faction_analysis.sql``, which is the authority. ``tests/test_schema.py``
checks that every name here appears there.
"""

RUNS = 'runs'
MEMBERS = 'members'
MEMBER_FEATURES = 'member_features'
MOVES = 'moves'
LINEAGE = 'lineage'
FACTIONS = 'factions'
FACTION_RUNS = 'faction_runs'
CONFLICT = 'conflict'
TEXT_SCORES = 'text_scores'
JEV_LABELS = 'jev_labels'
SCORER_DATASET = 'scorer_dataset'

RUN_OWNED = (RUNS, MEMBERS, MEMBER_FEATURES, MOVES, LINEAGE, FACTIONS, FACTION_RUNS, CONFLICT)
IMAGE_TABLES = (MEMBERS, MEMBER_FEATURES)
CACHES = (TEXT_SCORES, JEV_LABELS, SCORER_DATASET)

RUNS_COLUMNS = (
    'run_id',
    'status',
    'kind',
    'recorded_at',
    'started_at',
    'window_start',
    'window_end',
    'window_days',
    'objective',
    'resolution',
    'seed',
    'next_faction_id',
    'accounts_placed',
    'accounts_unplaced',
    'hash_collisions',
    'factions',
    'moves',
    'births',
    'merges',
    'splits',
    'deaths',
    'agreement_prev',
    'modularity',
    'source_gap_days',
    'text_model_version',
    'text_coverage',
    'stage_seconds',
    'failed_stage',
    'error_class',
)

MEMBERS_COLUMNS = ('run_id', 'did', 'faction_id', 'since_run_id', 'prev_faction_id', 'confidence', 'retire_at')

MEMBER_FEATURES_COLUMNS = (
    'run_id',
    'faction_id',
    'did',
    'degree',
    'within_z',
    'participation',
    'pileon_replies_sent',
    'pileon_replies_received',
    'swarm_quotes_sent',
    'swarm_quotes_received',
    'blocks_sent',
    'blocks_received',
    'text_hostility_sent',
    'text_hostility_received',
    'retire_at',
)

MOVES_COLUMNS = ('run_id', 'did', 'move_kind', 'from_faction_id', 'to_faction_id', 'written_at')

LINEAGE_COLUMNS = ('run_id', 'faction_id', 'event', 'related_faction_id', 'written_at')

FACTIONS_COLUMNS = (
    'faction_id',
    'run_id',
    'status',
    'parent_faction_id',
    'born_run_id',
    'member_count',
    'top_tags',
    'top_languages',
    'sample_handles',
    'written_at',
)

FACTION_RUNS_COLUMNS = (
    'run_id',
    'faction_id',
    'member_count',
    'mean_degree',
    'hostile_sent',
    'hostile_received',
    'conflict_cells',
    'max_conflict_score',
    'written_at',
)

CONFLICT_COLUMNS = (
    'run_id',
    'window_days',
    'from_faction_id',
    'to_faction_id',
    'contact',
    'contact_expected',
    'contact_lift',
    'pileon_replies',
    'pileon_lift',
    'direction_pileon',
    'swarm_quotes',
    'swarm_lift',
    'direction_swarm',
    'blocks',
    'blocks_lift',
    'direction_blocks',
    'text_hostility',
    'text_scored',
    'text_lift',
    'direction_text',
    'conflict_score',
    'distinct_sources',
    'distinct_targets',
    'written_at',
)

TEXT_SCORES_COLUMNS = (
    'uri_hash',
    'h',
    'model_version',
    'kind',
    'scored_at',
    'is_example',
    'author_did',
    'target_did',
    'post_text',
    'event_at',
)

JEV_LABELS_COLUMNS = (
    'uri_hash',
    'prompt_version',
    'h',
    'p_hostile',
    'p_threat',
    'model',
    'input_tokens',
    'labelled_at',
)

SCORER_DATASET_COLUMNS = (
    'dataset_version',
    'uri_hash',
    'kind',
    'split',
    'label_source',
    'target',
    'weight',
    'label_h',
    'author_did',
    'target_did',
    'text',
    'parent_uri',
    'event_ms',
    'languages',
    'text_length',
    'uppercase',
    'exclamations',
    'questions',
    'mentions',
    'written_at',
)

COLUMNS: dict[str, tuple[str, ...]] = {
    RUNS: RUNS_COLUMNS,
    MEMBERS: MEMBERS_COLUMNS,
    MEMBER_FEATURES: MEMBER_FEATURES_COLUMNS,
    MOVES: MOVES_COLUMNS,
    LINEAGE: LINEAGE_COLUMNS,
    FACTIONS: FACTIONS_COLUMNS,
    FACTION_RUNS: FACTION_RUNS_COLUMNS,
    CONFLICT: CONFLICT_COLUMNS,
    TEXT_SCORES: TEXT_SCORES_COLUMNS,
    JEV_LABELS: JEV_LABELS_COLUMNS,
    SCORER_DATASET: SCORER_DATASET_COLUMNS,
}
