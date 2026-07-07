# TF-IDF Similarity Network + Density-Based Dismantling for URL Co-Sharing Design

## Summary

This design replaces `url_cosharing`'s current detection pipeline — pre-aggregated co-share pairs fed into a Newman-weighted graph, then clustered with Leiden — with a methodology adapted from a peer-reviewed coordinated-inauthentic-behaviour paper (Cinus et al., WWW '25). The new pipeline builds a similarity network directly from raw per-account URL-sharing activity: each account becomes a TF-IDF-weighted vector over the URLs it shared in a rolling 7-day window, and cosine similarity between these vectors defines edge weights in a graph. Because this requires per-account vectors and corpus-wide document frequencies that can't be reconstructed from already-aggregated pairs, the sidecar switches its data source from the `url_cosharing_pairs` materialized view to querying `osprey_execution_results` directly (the pairs MV stays alive for other tooling).

The key new step is "density-based dismantling": rather than clustering the whole similarity graph, the pipeline searches a 2-D grid of edge-similarity and eigenvector-centrality quantile thresholds, computes the minimum density across connected components at each grid cell, and uses knee detection (largest jump in that density surface) to automatically pick thresholds that isolate a small, high-precision "coordinated core" — trading recall for precision, consistent with the source paper's reported >0.9 precision at ~0.1 recall. Only this surviving core is then handed to Leiden clustering, and the existing evolution-tracking and output-writing machinery is extended (new columns, a new run-metadata table) rather than replaced. Guardrails and a "no knee found → flag nothing" fallback keep the automated threshold selection from silently over- or under-flagging on unattended daily runs.

## Definition of Done

The `url_cosharing` sidecar implements the coordination-detection methodology of Cinus, Minici, Luceri & Ferrara, "Exposing Cross-Platform Coordinated Inauthentic Activity in the Run-Up to the 2024 U.S. Election" (WWW '25, DOI 10.2139/ssrn.5018877), adapted to daily unattended operation:

1. The similarity network is built from TF-IDF-weighted account×URL vectors over a rolling window (default 7 days), with the paper's preprocessing filters (minimum unique URLs per account, minimum/maximum URL document frequency).
2. Density-based unsupervised network dismantling (joint edge-similarity and eigenvector-centrality quantile filtering) isolates a high-precision coordinated core, with threshold selection automated via knee detection on the minimum-component-density surface, bounded by configurable guardrails. When no density transition is found, the run flags no accounts rather than guessing.
3. Hybrid clustering: Leiden CPM decomposes the surviving coordinated core into clusters; existing evolution tracking (birth/death/continuation/merge/split) and output tables continue to work, extended with similarity-based metrics.
4. Each run records observability metadata (filter counts, chosen thresholds, minimum density, guardrail outcomes) for calibration and monitoring.
5. All existing and new tests pass via `uv run pytest`.

Out of scope: `quote_cosharing` (follow-up port once calibrated), the paper's text-similarity network, BERTopic topic analysis, AI-generated-content detection, and credibility assessment.

## Acceptance Criteria

### density-dismantling.AC1: Similarity network construction
- **density-dismantling.AC1.1 Success:** A daily run fetches per-account URL share counts over the configured rolling window (default 7 days) from `osprey_execution_results`, not from `url_cosharing_pairs`.
- **density-dismantling.AC1.2 Success:** Accounts sharing fewer than `min_unique_urls` (default 10) unique URLs in the window are excluded before matrix construction.
- **density-dismantling.AC1.3 Success:** URLs shared by fewer than `min_url_sharers` (default 5) accounts, or with document frequency above the `max_url_df_pctl` (default 0.90) percentile, are excluded before TF-IDF.
- **density-dismantling.AC1.4 Success:** Edge weights are cosine similarities between account TF-IDF vectors, in [0, 1]; similarities below `edge_epsilon` are not materialized as edges.
- **density-dismantling.AC1.5 Edge:** An empty or fully-filtered input produces an empty graph; the run completes normally and writes a run-metadata row with zero counts.

### density-dismantling.AC2: Density-based dismantling
- **density-dismantling.AC2.1 Success:** Grid search over the configured edge-similarity × eigenvector-centrality quantile grids produces a minimum-component-density surface (isolates dropped before density is computed).
- **density-dismantling.AC2.2 Success:** Knee detection selects the threshold pair at the largest discrete jump in minimum component density whose resulting density meets `density_floor`; a synthetic graph with a planted dense core plus organic background recovers the planted core.
- **density-dismantling.AC2.3 Failure:** When no cell satisfies the density floor (no transition), zero accounts are flagged and the run row records that no threshold was selected.
- **density-dismantling.AC2.4 Failure:** When the selected thresholds would flag more than `max_flagged_fraction` of eligible accounts, the candidate is rejected and the next-best candidate (or none) is used; the guardrail activation is recorded.
- **density-dismantling.AC2.5 Success:** The chosen quantile pair, resulting minimum density, and account/URL counts at each filter stage are written to the run-metadata table.

### density-dismantling.AC3: Hybrid clustering and outputs
- **density-dismantling.AC3.1 Success:** Leiden CPM (cosine-similarity edge weights) decomposes the surviving core; clusters below `min_cluster_size` are dropped.
- **density-dismantling.AC3.2 Success:** Cluster rows carry the new similarity metrics (`mean_edge_similarity`, `subgraph_density`) alongside existing columns; `total_weight` keeps its co-share-count semantics (Σ over cluster URLs of C(sharing members, 2)).
- **density-dismantling.AC3.3 Success:** Evolution classification against prior membership snapshots (birth, death, continuation, merge, split) behaves as before, verified by existing tests continuing to pass.
- **density-dismantling.AC3.4 Success:** Daily membership snapshots are written for all cluster members.

### density-dismantling.AC4: Cross-cutting
- **density-dismantling.AC4.1:** Every new parameter is an env var with a documented default (`URL_COSHARING_*`), parsed into the frozen `AnalysisConfig`.
- **density-dismantling.AC4.2:** `similarity.py` and `dismantling.py` are pure functional-core modules (no I/O, no ClickHouse imports).
- **density-dismantling.AC4.3:** `cd url_cosharing && uv run pytest` passes with all new and existing tests.

## Glossary

- **TF-IDF (term frequency–inverse document frequency)**: A weighting scheme, borrowed from text retrieval, that scores how distinctive an item is to a given document. Here, "documents" are accounts and "terms" are URLs: a URL an account shares often but that few other accounts share gets a high weight, while a widely-shared URL (e.g., a viral news link) gets down-weighted since it's less informative for detecting coordination.
- **Cosine similarity**: A measure of the angle between two vectors, ranging 0 (orthogonal, no overlap) to 1 (identical direction). Used here to score how similar two accounts' URL-sharing patterns are, independent of how many total URLs each shared.
- **Account×URL matrix / bipartite matrix**: A sparse matrix where rows are accounts, columns are URLs, and cell values are share counts — the raw input structure before TF-IDF weighting and similarity computation.
- **Sparse matrix (scipy CSR)**: A memory-efficient matrix representation that stores only non-zero entries. Necessary here because most accounts share only a tiny fraction of all URLs, so a dense matrix would waste enormous memory.
- **Document frequency (df)**: The number of distinct accounts (documents) that shared a given URL (term). Used both as a floor (URLs shared by too few accounts are noise) and a ceiling (URLs shared by too many accounts, like breaking news, aren't discriminating signal).
- **igraph**: A graph library used to represent the similarity network and run graph algorithms (centrality, community detection) on it.
- **Eigenvector centrality**: A measure of node importance in a graph where a node's score depends on the scores of its neighbours — well-connected nodes linked to other well-connected nodes score highest. Used here as one axis of the threshold grid to help isolate tightly-interconnected coordinated groups.
- **Network dismantling**: Deliberately removing edges/nodes from a graph (via thresholding) to break it down to a smaller, denser residual structure — here, used to strip away loosely-connected "organic" accounts and leave behind a tightly-knit coordinated core.
- **Density (graph/component density)**: The ratio of actual edges to possible edges within a connected component. A dense component indicates a tightly interconnected group of accounts, which is the signature the method is hunting for.
- **Knee detection**: Finding the point of sharpest inflection ("knee") in a curve or surface — here, the largest discrete jump in minimum component density across the threshold grid, used to automatically pick a threshold pair without manual tuning.
- **Quantile grid search**: Trying combinations of threshold values expressed as percentiles/quantiles (rather than raw values) across two parameters (edge similarity, centrality) to find the best-performing combination.
- **Isolates**: Graph nodes left with no edges after filtering — removed before density is computed since a density calculation is meaningless for disconnected single nodes.
- **Leiden (algorithm) / Leiden CPM**: A community-detection algorithm that partitions a graph into clusters of densely-connected nodes. "CPM" (Constant Potts Model) is the specific quality function/resolution scheme it optimizes.
- **Newman-weighted graph**: The prior (pre-this-design) approach to weighting edges in the co-sharing graph, based on Newman's method for weighting projections of bipartite networks — being replaced by cosine-similarity weights.
- **Functional Core / Imperative Shell (FCIS)**: An architectural pattern separating pure, side-effect-free logic (Core: computation, no I/O) from the code that performs I/O and orchestration (Shell: database calls, wiring). Referenced here to explain why `similarity.py` and `dismantling.py` must have no ClickHouse imports.
- **Jaccard threshold / evolution tracking (birth/death/continuation/merge/split)**: Existing machinery that compares today's clusters to prior days' clusters (via Jaccard similarity of membership sets) to classify how clusters change over time. Unchanged by this design but noted as downstream-affected.
- **Sidecar**: In this codebase, a standalone daily-batch analysis service (as opposed to the real-time Osprey rule engine) that reads from ClickHouse, computes derived signals, and writes results back — `url_cosharing` is one such sidecar.
- **osprey_execution_results**: The ClickHouse table holding raw per-event rule-execution output from the Osprey moderation pipeline; the new direct data source for this sidecar instead of the pre-aggregated pairs MV.
- **Materialized view (MV)**: A ClickHouse table that's automatically kept up to date from a query over other tables (here, `url_cosharing_pairs`), avoiding needing to recompute the aggregation on every read.
- **Guardrails (`max_flagged_fraction`, `min_cluster_size`)**: Configured safety bounds that reject an automatically-selected threshold choice if it would flag an implausibly large fraction of accounts, protecting against the automated knee detection over-triggering.
- **CTE (Common Table Expression)**: A named, scoped sub-query (SQL `WITH` clause) — used here to push filtering logic into the ClickHouse query itself so unneeded rows never cross the wire.

## Architecture

The sidecar keeps its Functional Core / Imperative Shell shape and its daily cadence, but replaces the detection path. Today: pre-aggregated pairs → Newman-weighted graph → Leiden. After this design: raw account×URL share vectors → TF-IDF cosine similarity network → density-based dismantling (high-precision coordinated core) → Leiden on the core (cluster decomposition) → existing evolution tracking.

Data flow per daily run:

1. **Fetch** (`db.py` / `queries.py`): one query against `osprey_execution_results` returning `(did, url, share_count)` over the trailing `window_days`, with the activity filter (≥ `min_unique_urls` unique URLs per account) and URL document-frequency filters (≥ `min_url_sharers` accounts, ≤ `max_url_df_pctl` percentile) pushed into SQL CTEs so only qualifying rows cross the wire. The `url_cosharing_pairs` table is no longer read by the sidecar (the MV remains for investigation tooling).
2. **Similarity network** (`similarity.py`, new Core module): build a scipy CSR account×URL count matrix, apply TF-IDF (tf × log(N/df), L2-normalized rows — hand-rolled on scipy.sparse, no sklearn dependency), compute pairwise cosine similarity via sparse matrix product, and emit an igraph graph with `similarity` edge weights, dropping edges below `edge_epsilon`.
3. **Dismantling** (`dismantling.py`, new Core module): compute weighted eigenvector centrality; grid-search the 2-D space of edge-similarity quantile × centrality quantile; for each cell filter edges/nodes below thresholds, drop isolates, and record the minimum density across connected components; select thresholds by knee detection — the largest discrete (forward-difference) jump in minimum density whose resulting density ≥ `density_floor` — subject to guardrails (`max_flagged_fraction`, surviving nodes ≥ `min_cluster_size`). No qualifying knee → empty core, run still completes.
4. **Clustering** (`analyzer.py`): Leiden CPM on the surviving subgraph using cosine-similarity weights (replacing Newman weights). Per-cluster metrics computed from the bipartite matrix: `unique_urls` (URLs shared by ≥ 2 members), `total_weight` (Σ_url C(k_members, 2) — same co-share-count semantics as the previous sum of raw pair weights), `sample_urls` (top URLs by cluster TF-IDF mass, more informative than arbitrary sampling), plus new `mean_edge_similarity` and `subgraph_density`. Temporal metrics and Jaccard evolution tracking are unchanged.
5. **Write** (`db.py`): clusters (extended columns), membership snapshots (unchanged), and one row into a new `url_cosharing_runs` observability table (window, counts after each filter stage, chosen quantiles, minimum density, guardrail/fallback flags).

New configuration on `AnalysisConfig` (env prefix `URL_COSHARING_`): `WINDOW_DAYS` (7), `MIN_UNIQUE_URLS` (10), `MIN_URL_SHARERS` (5), `MAX_URL_DF_PCTL` (0.90), `EDGE_EPSILON`, `EDGE_QUANTILE_GRID` and `CENTRALITY_QUANTILE_GRID` (comma-separated quantiles, defaults spanning 0.50–0.99), `DENSITY_FLOOR`, `MAX_FLAGGED_FRACTION`. `RESOLUTION`, `MIN_CLUSTER_SIZE`, `JACCARD_THRESHOLD`, `EVOLUTION_WINDOW_DAYS` are retained. `MIN_EDGE_WEIGHT` and `MIN_COSHARERS` become obsolete on the detection path and are removed.

### Contract: schema changes (skywatch-osprey `clickhouse-init/05-url-cosharing.sql`)

```sql
ALTER TABLE default.url_cosharing_clusters
    ADD COLUMN mean_edge_similarity Float64,
    ADD COLUMN subgraph_density Float64;

CREATE TABLE IF NOT EXISTS default.url_cosharing_runs (
    run_date Date,
    window_days UInt8,
    accounts_raw UInt64,          -- before activity filter
    accounts_eligible UInt64,     -- after activity + df filters
    urls_eligible UInt64,
    graph_edges UInt64,
    edge_quantile Float64,        -- 0 when no knee selected
    centrality_quantile Float64,
    min_component_density Float64,
    knee_found Bool,
    guardrail_triggered Bool,
    flagged_accounts UInt64,
    cluster_count UInt32
) ENGINE = MergeTree()
ORDER BY run_date;
```

Existing `url_cosharing_clusters` and `url_cosharing_membership` columns keep their meaning; `resolution_parameter` now describes Leiden-on-core. The pairs table/MV are untouched (still consumed by the `cosharing_pairs` MCP investigation tooling).

## Existing Patterns

This design follows the repo's established sidecar patterns:

- **FCIS split** — `config.py`/`queries.py`/`analyzer.py` are Core, `db.py`/`main.py` are Shell (`url_cosharing/CLAUDE.md`). New modules `similarity.py` and `dismantling.py` join the Core.
- **Frozen dataclass config from env vars** with `_validate_table_name` guarding table interpolation (`url_cosharing/src/url_cosharing/config.py`).
- **SQL as pure string builders** in `queries.py`, tested in `tests/test_queries.py`.
- **scipy as a documented analytics dependency** — the count-based sidecars (`url_overdispersion`, `quote_overdispersion`) already depend on scipy per their CLAUDE.md files; `url_cosharing` adds the same dependency.
- **Calibration methodology docs** in `docs/calibration.md` — extended rather than replaced.

Divergence: the sidecar stops consuming its dedicated MV output (`url_cosharing_pairs`) and reads `osprey_execution_results` directly (as `fetch_member_timestamps_query` already does). Justified because TF-IDF cosine similarity requires per-account URL vectors and corpus document frequencies that cannot be reconstructed from pre-aggregated pairs.

## Implementation Phases

<!-- START_PHASE_1 -->
### Phase 1: Configuration and schema
**Goal:** All new knobs parse from env vars; ClickHouse schema supports the new outputs.

**Components:**
- `url_cosharing/src/url_cosharing/config.py` — new `AnalysisConfig` fields (window/filters/grids/guardrails as listed in Architecture), removal of `min_edge_weight`/`min_cosharers`
- `skywatch-osprey/clickhouse-init/05-url-cosharing.sql` — `url_cosharing_runs` table, `ALTER TABLE` for cluster columns, migration comment block (cross-repo change, same pattern as the Newman-weight migration)
- `url_cosharing/tests/test_config.py` — parsing, defaults, validation

**Dependencies:** None.

**Done when:** DDL applies cleanly to a ClickHouse instance; config tests pass (`density-dismantling.AC4.1`).
<!-- END_PHASE_1 -->

<!-- START_PHASE_2 -->
### Phase 2: Bipartite data access
**Goal:** Fetch filtered `(did, url, share_count)` rows for the rolling window.

**Components:**
- `url_cosharing/src/url_cosharing/queries.py` — `fetch_url_shares_query` with CTEs applying the activity filter and df filters (min sharers, max-df percentile computed in SQL); removal of `fetch_pairs_query`
- `url_cosharing/src/url_cosharing/db.py` — fetch method returning typed rows
- `url_cosharing/tests/test_queries.py`, `tests/test_db.py`

**Dependencies:** Phase 1 (config fields referenced by query builder).

**Done when:** Query builder tests verify window bounds and filter clauses (`density-dismantling.AC1.1`–`AC1.3` at the SQL level); db tests pass.
<!-- END_PHASE_2 -->

<!-- START_PHASE_3 -->
### Phase 3: Similarity network core
**Goal:** Account×URL matrix → TF-IDF → cosine similarity graph.

**Components:**
- `url_cosharing/src/url_cosharing/similarity.py` — sparse matrix construction, in-Python re-application of activity/df filters (defense in depth over the SQL prefilter), TF-IDF transform, sparse cosine product, igraph construction with `similarity` weights and `edge_epsilon` cutoff
- `url_cosharing/pyproject.toml` — scipy dependency
- `url_cosharing/tests/test_similarity.py` — known-vector TF-IDF/cosine cases, filter behaviour, empty-input, epsilon cutoff

**Dependencies:** Phase 2 (row shape).

**Done when:** Tests verify `density-dismantling.AC1.2`–`AC1.5`.
<!-- END_PHASE_3 -->

<!-- START_PHASE_4 -->
### Phase 4: Density-based dismantling core
**Goal:** Unsupervised threshold selection isolating the coordinated core.

**Components:**
- `url_cosharing/src/url_cosharing/dismantling.py` — weighted eigenvector centrality, quantile grid search, minimum-component-density surface, knee detection with `density_floor` / `max_flagged_fraction` / minimum-survivor guardrails, structured result (chosen thresholds, surface stats, surviving subgraph)
- `url_cosharing/tests/test_dismantling.py` — synthetic planted-core graphs (dense coordinated component + sparse organic background) recovering the core; no-transition and guardrail-rejection cases

**Dependencies:** Phase 3 (similarity graph as input).

**Done when:** Tests verify `density-dismantling.AC2.1`–`AC2.4`.
<!-- END_PHASE_4 -->

<!-- START_PHASE_5 -->
### Phase 5: Pipeline integration
**Goal:** End-to-end daily run wiring the new detection path into clustering, metrics, evolution, and writes.

**Components:**
- `url_cosharing/src/url_cosharing/analyzer.py` — Leiden on the surviving core with `similarity` weights; cluster metrics from the bipartite matrix (`total_weight`, `unique_urls`, TF-IDF-ranked `sample_urls`, `mean_edge_similarity`, `subgraph_density`); removal of `build_graph`'s pairs-based path; evolution tracking untouched
- `url_cosharing/src/url_cosharing/main.py` — orchestration of fetch → similarity → dismantling → cluster → evolve → write
- `url_cosharing/src/url_cosharing/queries.py` / `db.py` — extended cluster insert, `url_cosharing_runs` insert
- `url_cosharing/tests/test_analyzer.py`, `tests/test_main.py` — integration test on synthetic data through the full core pipeline

**Dependencies:** Phases 1–4.

**Done when:** Tests verify `density-dismantling.AC2.5`, `AC3.1`–`AC3.4`, `AC4.2`; full suite passes (`AC4.3`).
<!-- END_PHASE_5 -->

<!-- START_PHASE_6 -->
### Phase 6: Documentation and calibration
**Goal:** Operators can calibrate grids/guardrails against production data and understand the new methodology.

**Components:**
- `url_cosharing/CLAUDE.md`, `url_cosharing/README.md` — new methodology, contract change (reads `osprey_execution_results`, pairs MV no longer consumed), config reference
- `docs/calibration.md` — density-surface calibration methodology (how to read the grid, choose `density_floor`, validate `max_flagged_fraction` against the paper's observed 0.4–1.5% coordinated-account rates)
- Calibration helper script (location per repo convention) dumping the grid surface from production data for offline inspection

**Dependencies:** Phase 5.

**Done when:** Docs updated with freshness dates; calibration script runs against a ClickHouse instance.
<!-- END_PHASE_6 -->

## Additional Considerations

**Memory/scale:** The account×account cosine product must stay sparse. The df ceiling (p90) bounds per-URL fan-out, and `edge_epsilon` prevents materializing negligible similarities. If post-filter account counts grow beyond low tens of thousands, blockwise sparse multiplication is the escape hatch — noted here so the implementation keeps the product chunked-friendly, not required initially.

**Precision-first semantics:** The paper reports precision > 0.9 at recall ≈ 0.1 against labelled IO campaigns. Downstream consumers should treat cluster membership as a strong signal, not exhaustive coverage. The previous Leiden-over-everything behaviour surfaced more (noisier) clusters; expect flagged-account volume to drop.

**Empty-result behaviour is correct behaviour:** On days with no density phase transition, writing zero clusters plus a `knee_found = false` run row is the designed outcome, not a failure. Monitoring should alert on consecutive `knee_found = false` runs only if paired with rising `accounts_eligible`.

**Window vs. evolution cadence:** Runs stay daily with a 7-day lookback, so consecutive runs share 6/7 of their data; evolution events will skew toward `continuation`. Jaccard threshold may warrant retuning during calibration (Phase 6).

**Threshold automation is an extension of the paper:** The authors selected quantiles by visual inspection and left quantitative selection as future work. The knee-detection rule (largest forward-difference jump subject to a density floor) is our contribution and the run-metadata table exists specifically so its choices can be audited.
