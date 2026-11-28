# Faction analysis sidecar

This sidecar shows moderators which Bluesky accounts form communities and which communities direct hostility at others.
A sidecar is a separate service that works alongside Osprey.
It reads ClickHouse, the database that stores Osprey events, and writes its own analysis tables.
It never writes to the source or another sidecar's tables.

Moderators and analysts can read the tables with SQL. No dedicated moderator tool exists yet.
The sidecar keeps faction IDs stable across weekly and monthly runs.
It measures blocks, reply pile-ons, quote swarms, and text hostility separately.

## Install and run

There is no end-user installation. Production deployment belongs to the separate `skywatch-bsky/skywatch-osprey` repository.
Two GitHub issues request the [table definitions and cleanup grants](https://github.com/skywatch-bsky/skywatch-osprey/issues/84) and the [Compose service](https://github.com/skywatch-bsky/skywatch-osprey/issues/85).
Their bodies and an index are in [docs/osprey-issues/](docs/osprey-issues/).

The deployment needs:

- The tables in [sql/001_faction_analysis.sql](sql/001_faction_analysis.sql). This file also forms the table-definition issue body.
- The `ALTER DROP PARTITION` and `ALTER DELETE` privileges listed at the end of that file.
- A persistent volume for `FACTION_ANALYSIS_MODEL_DIR`, which defaults to `/data/models`.
  This directory also holds the promotion-intent file. Without persistence, a restart can lose that file.
- The `OSPREY_CLICKHOUSE_*` connection settings listed below.
- `TYPESAFE_API_KEY` supplied as a secret for new model labels.
- About 20 GiB of memory per run. The measured partition used 17.7 GiB on the development machine.
- One service instance per host. A local lock refuses a second instance.

### Try the local test server

This procedure needs Python 3.14, uv, Docker Compose, and a checkout of this repository.
It uses synthetic test data, not production data. The tests create their own tables and exercise the analysis and publication paths.

1. From the repository root, enter the sidecar directory.

   ```bash
   cd faction-analysis
   ```

2. Install the locked development dependencies.

   ```bash
   uv sync --locked
   ```

3. Start the isolated test database.

   ```bash
   docker compose -f docker/compose.test.yml up -d --wait
   ```

   Compose waits for a healthy ClickHouse 24.1 server on `127.0.0.1:18123`.
   The server keeps its data in memory.

4. Run the integration tests.

   ```bash
   FACTION_ANALYSIS_INTEGRATION_CH_HOST=127.0.0.1 FACTION_ANALYSIS_INTEGRATION_CH_PORT=18123 uv run pytest -m integration
   ```

   Pytest should report 26 passed integration tests. A failure means the local check did not pass.
   The command exits after the tests.

5. Stop the test server, including after a failed test.

   ```bash
   docker compose -f docker/compose.test.yml down
   ```

### Try a dry run with an existing database

A dry run computes results without publishing analysis rows. It still reads the database and needs the source and output tables.
It creates a local model directory and lock file. It skips the model refresh.
It can consume the same graph memory as a real run.

1. Ask the deployment administrator to supply the connection settings as environment variables.
2. From `faction-analysis/`, run one full analysis with a writable local model directory.

   ```bash
   FACTION_ANALYSIS_MODEL_DIR=./models uv run python -m faction_analysis.main --kind full --dry-run
   ```

   The process logs its outcome and exits. It does not publish a new image.
   A source with no sufficiently large community fails the partition stage instead.
3. To interrupt the command, press Ctrl+C.

## Developer setup

The package lives in `src/faction_analysis/`. `.python-version` selects Python 3.14, and `pyproject.toml` requires Python 3.14 or later.
The project uses uv with `uv.lock`. Development tools are pytest, Ruff, and ty.
Dependencies include scikit-learn 1.7 or later and TypeSafe SDK 0.7, below 0.8.

1. Get a checkout of the `osprey-sidecars` repository from your team.
2. From its root, enter `faction-analysis/`.

   ```bash
   cd faction-analysis
   ```

3. Install the locked dependencies.

   ```bash
   uv sync --locked
   ```

4. Run the tests.

   ```bash
   uv run pytest
   ```

   About 300 unit tests cover the implementation, alongside 26 integration tests. Integration tests skip unless `FACTION_ANALYSIS_INTEGRATION_CH_HOST` is set.
   Use the local-server procedure above to run those tests.

5. Check lint rules.

   ```bash
   uv run ruff check .
   ```

6. Check formatting.

   ```bash
   uv run ruff format --check .
   ```

7. Check types.

   ```bash
   uv run ty check
   ```

Each check should exit successfully. A skipped integration suite does not verify publication safety.
`tests/test_integration_clickhouse.py` tests failure after each write stage and crashes around both promotion markers.
It also tests delayed commits, late commits, and unreadable intent files against ClickHouse 24.1.

### Run commands

These commands run inside `faction-analysis/` with the deployment prerequisites above.
Normal runs write analysis tables. Training can write labels, datasets, and model files, and can incur TypeSafe charges.

| Command | Expected result |
|---|---|
| `uv run python -m faction_analysis.main` | Polls hourly until stopped. Ctrl+C or SIGTERM stops it after the current poll. |
| `uv run python -m faction_analysis.main --once` | Checks the schedule once and exits. It can report that no run is due. |
| `uv run python -m faction_analysis.main --kind weekly` | Forces one warm-start run and exits. |
| `uv run python -m faction_analysis.main --kind full` | Forces one cold-start run and exits. |
| `uv run python -m faction_analysis.main --kind full --dry-run` | Computes one cold-start analysis without publishing rows or refreshing the model. |
| `uv run python -m faction_analysis.train` | Refreshes the hostility model and prints a JSON outcome. A rejected candidate does not replace the current model. |
| `uv run python -m faction_analysis.train --dry-run --no-label` | Evaluates training from stored labels without new Jev calls or dataset/model writes. |
| `uv run python -m faction_analysis.export_dataset --dataset-version X --out-dir DIR` | Exports the selected dataset as JSONL shards and prints a manifest. Replace `X` and `DIR` with your dataset version and destination. |
| `uv run python -m faction_analysis.measure probes` | Runs bounded production probes through the documented SSH path and writes local measurement artifacts. |
| `uv run python -m faction_analysis.main --measure probes` | Dispatches to the same measurement command. |
| `uv run python scripts/make_synthetic_edges.py OUT.npz VERTICES BLOCKS WITHIN_SHARE DEGREE SEED` | Writes a synthetic planted-partition edge file with equally sized blocks. Use it for a local scale check of the measurement command's Leiden child. |

The training command also accepts `--dry-run` and `--no-label` separately.
`--no-label` reuses stored labels. Use both flags to avoid new Jev calls and persisted training outputs.
A `no_model` training outcome exits with code 1. Other training outcomes exit with code 0.

The measurement entry point requires a subcommand. Available subcommands are `probes`, `extract`, `leiden`, `audit`, `compare`, and `report`.
The bare `uv run python -m faction_analysis.measure` command does not run the complete measurement.
Production measurement needs approved SSH access. See [the measurement report](docs/MEASUREMENT.md) before using it.

## Runs and faction identity

The schedule checks hourly. It chooses a cold start when no monthly run exists or the last monthly run is 28 days old.
Otherwise, it chooses a warm start when the latest promoted run is seven days old.
A warm start uses the previous partition. A cold start starts without that partition.

Each run follows this order:

1. Read the previous membership image.
2. Extract the follow and repost graph over 60 days.
3. Partition the graph with Leiden and assign stable faction IDs.
4. Write the new membership image, not yet visible to readers.
5. Refresh the hostility model when due on a monthly run.
6. Extract replies, quotes, and blocks between placed accounts, then score the texts.
7. Compute account features and faction descriptions.
8. Compute conflict cells for the seven-day and 60-day windows.
9. Write the registry, lineage, faction statistics, and moves.
10. Install the promotion intent and publish the run.

The default partition uses modularity, resolution 0.5, seed 42, two iterations, and a minimum faction size of 500.
The [production measurement](docs/MEASUREMENT.md) found 25 factions in 25.2 minutes at 17.7 GiB.
These are measurements, not runtime guarantees.
`FACTION_ANALYSIS_RESOLUTION=1.0` gives the finer partition that produced 52 factions and that the operator also judged reasonable.
The [measurement interpretation](docs/MEASUREMENT_INTERPRETATION.md) explains the stability tradeoff.

Small communities join the established neighbor with the largest connecting edge weight.
Without such a neighbor, their accounts remain unplaced. If no community reaches the minimum size, the run fails without promotion.

Tracking compares old factions with new communities by their overlapping weighted degree.
Each old faction selects its largest overlap. The largest predecessor overlap wins the existing ID for each new community.
Ties use the smaller community index, then the smaller old faction ID, respectively.
Births receive new IDs. Splits, merges, and deaths form lineage events.
Published IDs are never reused.

## Output tables and safe reads

All table names below use the default `faction_analysis_` prefix.
A membership image is the account-to-faction assignment for one run.
Each placed account has at most one faction in that image.

**Read run-owned output only through a promoted run.** Use this filter:

```sql
run_id IN (SELECT run_id FROM faction_analysis_runs WHERE status = 'promoted')
```

For example, this query reads the latest promoted image:

```sql
SELECT did, faction_id, since_run_id, prev_faction_id, confidence
FROM faction_analysis_members
WHERE run_id = (
    SELECT max(run_id)
    FROM faction_analysis_runs
    WHERE status = 'promoted'
)
```

The latest image has the greatest promoted `run_id`.
Its `faction_analysis_runs.next_faction_id` supplies the starting ID for the next run.
The status rows can repeat after a retry. Consumers must select distinct promoted run IDs rather than count status rows.

| Table | Contents and reading rule | Retention |
|---|---|---|
| `faction_analysis_runs` | Status history, counts, gaps, collisions, model coverage, timings, and next faction ID. A promoted row establishes visibility. | No expiry |
| `faction_analysis_members` | Placed accounts, faction IDs, tenure, previous faction, and confidence. Select one promoted image. | Weekly images: eight weeks. Monthly images: thirteen months. |
| `faction_analysis_member_features` | Weighted degree, within-faction z-score, participation, and hostile signals sent and received. Join by run and account. | Same as members |
| `faction_analysis_moves` | Assignment changes only: `assign`, `move`, or `drop`. Read in promoted run order. | Two years |
| `faction_analysis_lineage` | Birth, split, merge, and death events. `related_faction_id` identifies a split parent or merge winner. | Two years |
| `faction_analysis_factions` | Immutable registry rows with status, size, top tags, languages, and sample handles. Use each faction's latest promoted row. | Two years |
| `faction_analysis_faction_runs` | Per-run faction size, mean degree, hostile counts, and conflict summaries. Read with the matching image. | Two years |
| `faction_analysis_conflict` | Ordered faction pairs, including within-faction cells, for each window. Read counts, lifts, coverage, and direction together. | Thirteen months |
| `faction_analysis_text_scores` | Durable scores and selected text examples. Deduplicate by `uri_hash` and `model_version`, taking the latest row. | 70 days |
| `faction_analysis_jev_labels` | Durable Jev labels with prompt version, returned model, and token use. Deduplicate by `uri_hash` and `prompt_version`. | 400 days |
| `faction_analysis_scorer_dataset` | Durable training rows, splits, labels, weights, child text, and parent URI. Deduplicate by `dataset_version` and `uri_hash`. | 400 days |

The three caches belong to no run. Cache reads use `argMax` by their keys to select the latest values.
Cleanup never removes their rows because a run failed.
The scoring cache stores post text only for examples: `h >= FACTION_ANALYSIS_EXAMPLE_MIN_H` or the configured hash sample.
The default sample keeps one text in `FACTION_ANALYSIS_EXAMPLE_SAMPLE_MODULUS` (200).
The training dataset separately retains child text. Export joins parent text for later training elsewhere.

All outputs precede the promotion-intent file. The file records the complete run row before the `promoting` and `promoted` markers.
Recovery completes an intended promotion forward. It never deletes that run's outputs.
An unreadable intent file stops recovery without cleanup.

## Graph and account features

For a directed pair from account i to account j, `f` is zero without a follow and one for a one-way follow.
It is `mutual_follow_weight` when both follows were created within the window.
Starter-pack follows do not enter the graph.

```text
r(i -> j) = f + repost_alpha * log1p(reposts from i of j)
k_in(j) = sum of raw weights r into j
m = median of positive k_in values
damp(j) = min(1, log1p(m) / log1p(k_in(j)))
w(i -> j) = r(i -> j) * damp(j)
```

Damping is one when disabled or when `m` is zero.
The undirected edge sums both directed weights. Rescaling makes the mean edge weight one.
Degree means weighted degree in this undirected graph.

`within_z` compares `log1p(degree)` with the faction mean, divided by its standard deviation.
It is NULL for fewer than three faction members or zero variance.
`participation = 1 - sum_s (k_is / k_i)^2`, where `k_is` is the account's edge weight into faction s.
It is NULL for an account without edges.
Hostile sent and received features cover the long window.

## Conflict formulas

Each cell describes an ordered pair A to B in one window. The reverse cell can differ.
A cell exists when contact or blocks exist. The diagonal describes interactions within one faction.
NULL means the value cannot be computed under these rules. It does not mean zero hostility.

### Contact and expected contact

Contact is replies plus quotes. `out_A` is all contact sent by A, and `in_B` is all contact received by B.
`contact_total` is contact across all factions in the same window.

```text
contact_expected = out_A * in_B / contact_total
contact_lift = contact_AB / contact_expected
```

A contact lift of two means twice the expected contact for those sending and receiving totals.
`contact_lift` is NULL when the expectation or the window's contact total is zero.

### Behavioral signals and shrinkage

A reply pile-on needs five distinct authors from one faction targeting the same account in the same thread-root bucket.
A quote swarm needs five distinct authors from one faction quoting the same post in the same bucket.
The default bucket lasts 60 minutes. Buckets align to the epoch, rather than slide with each event.
Once a group qualifies, all its events count, including earlier events in that bucket. Each event counts once.

Like follows, blocks exist from create until delete, and blocks removed by later deletes do not count.

For each signal s (pile-on replies, swarm quotes, or blocks), compute:

```text
g_s = s_total / contact_total
shrunk_rate_s = (s_AB + k * g_s) / (contact_AB + k)
s_lift = shrunk_rate_s / g_s
```

The default `k` is 20. It adds 20 units of contact at the global rate before calculating the cell's rate.
This pulls small samples toward the global rate instead of trusting a few events.
A lift above one means a rate above the global rate.
`pileon_lift`, `swarm_lift`, and `blocks_lift` are NULL when their signal total is zero or the contact total is zero.
Blocks measure events per unit of contact, not a probability. A block lift can exceed one without implying an impossible probability.

### Text hostility

`text_hostility` sums hostility probabilities h. It does not count texts above a threshold.
`text_scored` counts texts with scores. The global text rate uses only scored texts.

```text
g_t = sum of h across scored texts / count of scored texts
text_lift = ((text_hostility + k * g_t) / (text_scored + k)) / g_t
```

`text_lift` is NULL without a model or with a zero global text rate.
It is also NULL with fewer than 20 scored texts in the cell.
Partial coverage changes `text_scored`, not the denominator to all contact.
Readers must inspect coverage alongside the lift.

### Combined score and direction

```text
conflict_score = min(contact_lift, max(non-NULL hostility lifts))
direction_s = (s_AB - s_BA) / (s_AB + s_BA)
```

The hostility lifts are `pileon_lift`, `swarm_lift`, `blocks_lift`, and `text_lift`.
The combined score requires both elevated contact and at least one elevated hostility rate.
It is NULL when contact lift is NULL or all hostility lifts are NULL.
A block-only cell retains its block count, block lift, and block direction, but has NULL `conflict_score`.
With no contact anywhere in the window, all contact-normalized values are NULL, including block lift.

Direction uses each signal's event count, or the sum of h for text.
Positive direction means more hostility from A to B. Negative direction means more from B to A.
Zero means equal totals. Each direction column is NULL when its denominator is zero.
`distinct_sources` and `distinct_targets` count accounts across replies, quotes, and blocks.
Consumers choose their own minimum account counts before acting on a cell.

## Hostility classifier policy

The scikit-learn linear model uses character n-grams and structural features.
It learns from block-derived labels and a bounded TypeSafe Jev teaching sample. Jev does not score the full interaction stream.

### Labels and precedence

For an interaction from A to B, a block positive means B blocks A within 60 minutes.
A random negative requires the full `FACTION_ANALYSIS_BLOCK_NEGATIVE_HORIZON_DAYS` horizon to elapse, with no block from B to A during it.
The default horizon is seven days.
A block after the 60-minute positive delay but within the horizon supplies neither label and trains nothing without a Jev label.
Precedence is Jev label, then block positive, then random negative.

Jev receives one pair per request: reply and parent, or quote and quoted post.
Its four levels are neutral or friendly, dismissive or mocking, hostile, and threatening.
`h = p(hostile) + p(threatening)`.
The requested model is `jev-1.14.0`. The cache also records the response's model field.

| Training source | Target | Weight |
|---|---|---|
| Jev train split | One when `h >= 0.5`, otherwise zero | 1 |
| Block positive without a Jev label | 1 | `weak_positive_weight`, default 1 |
| Random negative | 0 | `negative_weight`, default 1 |

### Quotas and budgets

Each refresh selects up to 100,000 block positives and 100,000 elapsed-horizon random interactions for Jev.
Deterministic hash sampling reduces larger pools to those quotas.
Jev-labelled rows split 70/15/15 into training, calibration, and test by `uri_hash`.
Calibration and test pairs never enter any training source.
Other block positives supply up to 300,000 weak positives.
Interactions supply up to 500,000 random negatives only after the full horizon elapses without the target blocking the author during it.
The Jev random sample still uses elapsed-horizon interactions regardless of block history, because Jev supplies the label.

The hard limits are 200,000 Jev requests and 60,000,000 input tokens per refresh.
The plan estimates about $2.50 at its recorded input-token price, not a current price guarantee.
`FACTION_ANALYSIS_JEV_MAX_REQUESTS` counts every request sent, including retries. The TypeSafe SDK retries are disabled.
The sidecar allows at most three attempts per pair per refresh.
It retries only connection failures, timeouts, rate limits, server errors, or unreadable answers.
Between attempts, it waits for the server's requested delay or a doubling delay.

Other rejected requests, such as HTTP 400, are not retried.
Failed attempts retain their worst-case token charge because the server may have billed them.
`FACTION_ANALYSIS_JEV_CONCURRENCY` limits requests in flight.

The sidecar saves paid answers to `faction_analysis_jev_labels` in batches of 500 during labeling and again when labeling ends.
If a cache write fails, the sidecar immediately stops labeling and raises the error because label purchases must not outpace storage.
A failure can lose only unstored answers: at most one batch plus the requests already in flight.
A pair with exhausted attempts stays unlabelled and follows the weak-label rules for that refresh.
The next refresh buys labels only for pairs without a stored label for the same URI hash and prompt version.

### Promotion and missing labels

A candidate needs a two-class Jev test split with at least 200 positive test labels.
Its test ROC-AUC must reach 0.70.
A metric on a one-class split is NULL.
Per-language average precision requires at least 50 test positives in that language.
Each refresh records its evaluation metrics.

Without `TYPESAFE_API_KEY`, the trainer skips new Jev labels and reuses stored labels.
It can promote a candidate only if those labels meet the same test requirements.
With insufficient data, it does not replace the current model.
If no model exists, behavioral signals still work, but text lift is NULL.

## Known limits

- Follows are a flow, not a complete relationship snapshot. Older follows are invisible.
  “Mutual” means both follows were created inside the window.
- The linear model sees only the reply or quote text, not its parent.
  A parent join across tens of millions of interactions is too heavy for the production host.
- Blocks do not always indicate hostility. Some blocks target spam.
- Jev's quality per language is unknown. Refresh metrics expose language coverage but do not establish equal quality.
- Warm starts can hide change. Monthly cold starts correct this, but can also change community boundaries because of seed sensitivity.
- `faction_analysis_runs.source_gap_days` counts days with ingestion gaps. Missing events can affect every signal.
- A `cityHash64` collision leaves both accounts unplaced and removes their edges. `hash_collisions` records the count.
- Without cleanup grants, failed-run rows remain invisible until their table retention removes them.
- The source limits the history to 60 days. The sidecar stores outputs, not graph edges, and does not backfill older history.

## Environment variables

### ClickHouse connection

| Variable | Default | Description |
|----------|---------|-------------|
| `OSPREY_CLICKHOUSE_HOST` | `localhost` | ClickHouse HTTP host |
| `OSPREY_CLICKHOUSE_PORT` | `8123` | ClickHouse HTTP port |
| `OSPREY_CLICKHOUSE_USER` | `default` | ClickHouse user |
| `OSPREY_CLICKHOUSE_PASSWORD` | `clickhouse` | ClickHouse password |
| `OSPREY_CLICKHOUSE_DB` | `default` | Database that holds `osprey_execution_results` and the output tables |

### Source queries

| Variable | Default | Description |
|----------|---------|-------------|
| `FACTION_ANALYSIS_SOURCE_TABLE` | `osprey_execution_results` | Source table |
| `FACTION_ANALYSIS_TABLE_PREFIX` | `faction_analysis` | Prefix of every output table |
| `FACTION_ANALYSIS_MAX_THREADS` | `4` | `max_threads` on every query |
| `FACTION_ANALYSIS_MAX_EXECUTION_TIME` | `3600` | `max_execution_time` in seconds on every query |
| `FACTION_ANALYSIS_MAX_MEMORY_USAGE` | `20000000000` | `max_memory_usage` in bytes on every query (20 GB) |

### Affinity graph

| Variable | Default | Description |
|----------|---------|-------------|
| `FACTION_ANALYSIS_WINDOW_DAYS` | `60` | Window length in days. A value above 60 is rejected |
| `FACTION_ANALYSIS_MUTUAL_FOLLOW_WEIGHT` | `2` | Follow weight when both directions were created in the window |
| `FACTION_ANALYSIS_REPOST_ALPHA` | `0.5` | Weight of `log(1 + reposts)` on an edge |
| `FACTION_ANALYSIS_DAMP_LARGE_ACCOUNTS` | `true` | Damp edges into large accounts |

### Partition

| Variable | Default | Description |
|----------|---------|-------------|
| `FACTION_ANALYSIS_OBJECTIVE` | `modularity` | Leiden objective: `cpm` or `modularity`. The default is the one the measurement chose, see `docs/MEASUREMENT.md` |
| `FACTION_ANALYSIS_RESOLUTION` | `0.5` | Leiden resolution parameter |
| `FACTION_ANALYSIS_SEED` | `42` | Leiden random seed |
| `FACTION_ANALYSIS_N_ITERATIONS` | `2` | Leiden iterations. `-1` runs until no move improves the objective |
| `FACTION_ANALYSIS_MIN_FACTION_SIZE` | `500` | Accounts a community needs to be established |
| `FACTION_ANALYSIS_SPLIT_SHARE` | `0.1` | Share of an old faction's degree that makes a new community a split |
| `FACTION_ANALYSIS_CONFIDENCE_SEEDS` | `3` | Extra cold-start runs on a monthly run |

### Conflict matrix

| Variable | Default | Description |
|----------|---------|-------------|
| `FACTION_ANALYSIS_CONFLICT_WINDOWS_DAYS` | `7,60` | Conflict window lengths in days. Each must be at most the window |
| `FACTION_ANALYSIS_PILEON_MIN_REPLIERS` | `5` | Distinct authors that make a reply group a pile-on |
| `FACTION_ANALYSIS_PILEON_WINDOW_MINUTES` | `60` | Length of a pile-on bucket |
| `FACTION_ANALYSIS_SWARM_MIN_QUOTERS` | `5` | Distinct authors that make a quote group a swarm |
| `FACTION_ANALYSIS_SWARM_WINDOW_MINUTES` | `60` | Length of a swarm bucket |
| `FACTION_ANALYSIS_CELL_PRIOR_STRENGTH` | `20` | Pseudo-count `k` in the shrunk lift |
| `FACTION_ANALYSIS_MIN_TEXT_SCORED` | `20` | Scored texts a cell needs before it gets a text lift |

### Labels and model

| Variable | Default | Description |
|----------|---------|-------------|
| `FACTION_ANALYSIS_BLOCK_POSITIVE_MINUTES` | `60` | Delay after an interaction in which a block makes the interaction a block positive |
| `FACTION_ANALYSIS_BLOCK_NEGATIVE_HORIZON_DAYS` | `7` | Days that must elapse without a block before an interaction can be a negative |
| `FACTION_ANALYSIS_JEV_STRATUM_POSITIVES` | `100000` | Block positives sent to Jev per refresh |
| `FACTION_ANALYSIS_JEV_STRATUM_RANDOM` | `100000` | Random interactions sent to Jev per refresh |
| `FACTION_ANALYSIS_MAX_WEAK_POSITIVES` | `300000` | Block positives used as weak labels per refresh |
| `FACTION_ANALYSIS_NEGATIVE_SAMPLE` | `500000` | Random negatives used per refresh |
| `FACTION_ANALYSIS_WEAK_POSITIVE_WEIGHT` | `1` | Training weight of a weak positive |
| `FACTION_ANALYSIS_NEGATIVE_WEIGHT` | `1` | Training weight of a random negative |
| `FACTION_ANALYSIS_JEV_MAX_REQUESTS` | `200000` | Hard limit on Jev requests per refresh |
| `FACTION_ANALYSIS_JEV_MAX_INPUT_TOKENS` | `60000000` | Hard limit on Jev input tokens per refresh |
| `FACTION_ANALYSIS_JEV_CONCURRENCY` | `8` | Jev requests in flight at once |
| `FACTION_ANALYSIS_JEV_MAX_TEXT_CHARS` | `2000` | Characters kept from each text sent to Jev. Sets the per-request token bound |
| `FACTION_ANALYSIS_JEV_MODEL` | `jev-1.14.0` | TypeSafe model ID |
| `FACTION_ANALYSIS_PROMPT_VERSION` | `hostility-v1` | Key that separates Jev label generations |
| `FACTION_ANALYSIS_MIN_TEST_AUC` | `0.70` | Test ROC-AUC a model needs to be promoted |
| `FACTION_ANALYSIS_MIN_TEST_POSITIVES` | `200` | Jev-labelled test positives a model needs to be promoted |
| `FACTION_ANALYSIS_MODEL_MAX_AGE_DAYS` | `28` | Age at which the model is retrained |
| `FACTION_ANALYSIS_EXAMPLE_MIN_H` | `0.5` | Hostility at which a scored text is stored as an example |
| `FACTION_ANALYSIS_EXAMPLE_SAMPLE_MODULUS` | `200` | One text in this many is stored as an example whatever its score |
| `TYPESAFE_API_KEY` | none | TypeSafe API key. Set it in the environment or mount it as the compose secret file `/run/secrets/typesafe_api_key` |

### Schedule and storage

| Variable | Default | Description |
|----------|---------|-------------|
| `FACTION_ANALYSIS_POLL_INTERVAL_SECONDS` | `3600` | Seconds between schedule polls |
| `FACTION_ANALYSIS_WEEKLY_AFTER_DAYS` | `7` | Age of the latest promoted run at which a weekly run starts |
| `FACTION_ANALYSIS_MONTHLY_AFTER_DAYS` | `28` | Age of the latest monthly run at which a monthly run starts |
| `FACTION_ANALYSIS_MODEL_DIR` | `/data/models` | Persistent directory for the model and the promotion-intent file |

### Telemetry

Telemetry is low-cardinality: fixed stage names and counts only. It never carries high-cardinality values
such as a `did`, a `handle`, a `faction_id`, a `uri`, query text or an exception message.

| Variable | Default | Description |
|----------|---------|-------------|
| `FACTION_ANALYSIS_OTEL_ENABLED` | `false` | Enables OpenTelemetry |
| `FACTION_ANALYSIS_OTEL_SERVICE_NAME` | `faction-analysis` | `service.name` resource attribute |
| `FACTION_ANALYSIS_OTEL_SERVICE_VERSION` | `0.1.0` | `service.version` resource attribute |
| `FACTION_ANALYSIS_OTEL_ENVIRONMENT` | `local` | `deployment.environment` resource attribute |
| `FACTION_ANALYSIS_OTEL_TRACES_ENABLED` | `true` when enabled | Exports traces |
| `FACTION_ANALYSIS_OTEL_METRICS_ENABLED` | `true` when enabled | Exports metrics |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | none | OTLP collector endpoint |
