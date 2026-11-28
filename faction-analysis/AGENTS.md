# Faction analysis sidecar

Last verified: 2026-11-28

## Purpose

Gives moderators a picture of the factions on Bluesky and of how they relate: which accounts belong
together, which factions direct hostility at which, and who amplifies whom. It exists because the existing
sidecars detect coordination by shared content and say nothing about the standing communities that conflict
plays out in.

Status: implemented and tested. Unit tests and an integration suite against ClickHouse 24.1 pass. The
deployment changes in `skywatch-osprey` (table definitions, grants, compose service) are two GitHub issues
there: [#84, tables and grants](https://github.com/skywatch-bsky/skywatch-osprey/issues/84) and
[#85, Compose service](https://github.com/skywatch-bsky/skywatch-osprey/issues/85). Their bodies are in
`docs/osprey-issues/`. Evidence is in `docs/AGENT_FINDINGS.md`,
measurements in `docs/MEASUREMENT.md`.

## Contracts

- **Exposes:** tables prefixed `faction_analysis_` by default in ClickHouse. `runs` (status history;
  `next_faction_id` lives here), `members` (one image per run), `member_features`, `moves` (changes only),
  `lineage`, `factions` (registry), `faction_runs` (per-run statistics), `conflict` (ordered pairs including
  the diagonal, per window), and the caches `text_scores`, `jev_labels`, `scorer_dataset`.
- **Guarantees:** every account has at most one faction per run. Published faction IDs remain stable while
  the faction exists and are never reused. Read run-owned rows only through distinct `promoted` run IDs:
  status rows can repeat on retry. Intended promotions complete forward without removing their outputs.
  Caches belong to no run and survive failed-run cleanup. Deduplicate scores by URI hash and model version,
  labels by URI hash and prompt version, and datasets by dataset version and URI hash, taking the latest write.
- **Expects:** `osprey_execution_results` with the `OSPREY_CLICKHOUSE_*` settings, the tables of
  `sql/001_faction_analysis.sql` with the cleanup grants, a persistent model directory, and one instance
  per host (the lock refuses a second).

## Dependencies

- **Uses:** ClickHouse (reads the source, writes its own tables), scikit-learn for the hostility model,
  TypeSafe `jev-1.14.0` for a bounded teaching sample, OpenTelemetry (optional).
- **Used by**: moderators can query the tables with SQL. Read-only `skywatch-mcp` tools are planned;
  none exists yet.
- **Boundary:** never writes to `osprey_execution_results` or to another sidecar's tables. No imports from
  `osprey_worker` or from other sidecars. Table definitions are applied by `skywatch-osprey`.

## Key decisions

- **Hostility scores in the source are not used** (operator, 2026-11-27). `HateBertScore` covers 0.35% of
  English posts and `IntoleranceScore` is never populated. Hostility comes from behaviour and from a model
  this sidecar trains.
- **Recent history only, no backfill, window of 60 days** (operator). The source keeps two months. Outputs
  only are stored, never graph edges, and every table except `runs` has a TTL.
- **The affinity graph is organic follows plus windowed reposts.** Starter-pack follows stay out
  (operator allows exclusion). Replies, quotes and blocks feed the conflict layer.
- **Partition: `leidenalg`, modularity at resolution 0.5 in two iterations, seed 42**, warm-started weekly
  and cold-started monthly. The production measurement chose it: 25 factions, 25.2 minutes, 17.7 GiB, and
  two seeds agree at an adjusted Rand index of 0.88 (resolution 1.0: 52 factions, 0.69).
- **One level of factions, 20 to 300** (operator, 2026-11-28), minimum size 500. The operator judged 52
  factions reasonable; `FACTION_ANALYSIS_RESOLUTION=1.0` gives that partition.
- **Faction identity: the contingency-table tracking of the reference paper, with an exact maximum.** The
  paper's weighted vote approximates it. Splits, merges, births and deaths are lineage events.
- **A hostile interaction is counted four ways** (operator, 2026-11-27): block, reply pile-on, quote-post
  swarm, and a text hostility score. Each is its own column, and a cell is written when contact or blocks
  exist.
- **The classifier is a linear model now** (operator): character n-grams and structural features, trained
  on block-derived labels plus a bounded Jev-labelled teaching sample. The build also exports a dataset
  with parent and child text so a transformer can be trained elsewhere. The TypeSafe budget counts every
  attempt: the SDK's own retries are off, a pair is sent at most three times, and paid answers reach the
  cache in batches while the labelling runs.
- **Publication safety is forward only.** All output is written before the intent file is installed. The
  intent file is the source of truth; recovery re-sends its run-status rows. An unreadable intent stops cleanup.
- **The first version is the weekly refresh and the monthly full scan** (operator). The conflict monitor,
  mediator-versus-agitator estimates and cascade summaries come later.

## Invariants

- The functional core does no database, file or network I/O and does not import OpenTelemetry: `config.py`,
  `queries.py`, `graph.py`, `partition.py`, `tracking.py`, `features.py`, `conflict.py`, `labels.py`, `model.py`,
  `runstate.py`, `promotion.py`, `schedule.py`, `schema.py`, `measurement_rules.py`. `config.py` also reads
  the process environment.
- Telemetry carries no DIDs, handles, faction ids, URIs, query text or exception messages. Logs may carry a run ID.
- Every source query filters `__timestamp`. A follow or a block exists from its create until its delete.

## Key files

All module paths are under `src/faction_analysis/`; the core modules are listed above.
- `pipeline.py`, `main.py` — stages, recovery and polling; `--once`, `--kind`, `--dry-run`, `--measure`
- `db.py`, `lock.py`, `model_store.py` — ClickHouse I/O, local lock and durable intent, model persistence
- `train.py`, `jev.py`, `export_dataset.py` — model refresh, bounded teaching requests, text dataset export
- `measure.py`, `telemetry.py`, `__init__.py` — measurement CLI, telemetry, package marker
- `sql/001_faction_analysis.sql` — table definitions embedded in issue #84

## Gotchas

- Source scalar feature columns are `Nullable`; `FacetTagList` and `PostLanguages` are arrays.
  `quote_cosharing` runs Leiden without a seed; `url_cosharing` assigns new IDs on every merge and split.
- **A block exists from its create until its delete, like a follow.** Both the conflict layer and the
  training labels subtract block deletes (`graph.superseded_mask`). A training negative also needs no block
  inside the whole horizon (`labels.negative_mask`).
- **A cell with blocks but no contact has a NULL `conflict_score`**, even where its contact lift is 0.
- **ClickHouse 24.1 rejects a TTL on a `DateTime64` column.** Wrap it: `TTL toDateTime(col) + ...`.
- **An alias equal to a column name is substituted into the other expressions of its SELECT.**
  `min(did) AS did` next to `uniqExact(did)` fails with "Unknown function min". Qualify the columns
  (`a.did`) or use `anyLastIf`. `tests/test_queries.py` guards this.
- **clickhouse-connect streams a block of same-typed columns as a plain 2-D array** and a block of mixed
  types as named fields. Read streamed columns by position (`db._columns`).
- The rows of one run share `recorded_at`, so `run_rows_sql` also orders by status.
- **Cleanup of failed rows uses `DELETE FROM`, the lightweight delete.** `ALTER TABLE ... DELETE` is a mutation.
- The integration tests are gated on `FACTION_ANALYSIS_INTEGRATION_CH_HOST`; `docker/compose.test.yml`
  starts a ClickHouse 24.1 on 18123. A run that skips them proves nothing about publication safety.
