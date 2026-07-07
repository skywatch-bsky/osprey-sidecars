# URL Co-Sharing Graph Sidecar

Last verified: 2026-07-07

## Purpose

Detects coordinated inauthentic behaviour by identifying clusters of accounts that repeatedly share the same URLs. Uses TF-IDF cosine-similarity network over per-account URL-sharing vectors (7-day rolling window) combined with density-based dismantling to isolate a high-precision coordinated core (Cinus, Minici, Luceri & Ferrara, WWW '25), then applies Leiden CPM decomposition to the core. Tracks cluster evolution across days via Jaccard similarity. 

Reads from `osprey_execution_results` (URL share records); `url_cosharing_pairs` is no longer consumed by the sidecar (the materialized view remains for investigation tooling). Writes cluster results to `url_cosharing_clusters`, membership snapshots to `url_cosharing_membership`, and run metadata to `url_cosharing_runs`.

## Architecture

Functional Core / Imperative Shell:
- `config.py` — env var parsing into frozen dataclasses (Core)
- `queries.py` — SQL query generation (Core)
- `similarity.py` — build share matrix from URL share rows, TF-IDF transform, construct cosine-similarity graph (Core, no I/O)
- `dismantling.py` — density-based dismantling (grid search + knee detection + guardrails) to isolate coordinated cores (Core, no I/O)
- `analyzer.py` — Leiden CPM clustering on core with similarity weights, Jaccard evolution tracking (Core)
- `calibrate.py` — density-surface dump for offline tuning (Shell)
- `db.py` — ClickHouse client wrapper (Shell)
- `main.py` — polling loop, signal handling, orchestration (Shell)

## Detection Methodology

**TF-IDF Weighting:** Edge weight w(a, b) = cos(v_a, v_b) where v_a is account a's TF-IDF vector: tf = raw co-share count per URL, idf = ln(N / df), L2-normalized. Edge weights are in [0, 1].

**Density-Based Dismantling:** Grid search over (edge_quantile, centrality_quantile) pairs; the selected cell is the one with the largest jump in minimum-component-density relative to its grid neighbours (the knee), among cells whose density ≥ density_floor that also satisfy the guardrails. Applies knee-detection heuristic to avoid over-flagging (knee_found = false on days with weak phase transitions is expected behaviour). Guardrails: reject candidates if survived nodes exceed max_flagged_fraction of the eligible graph or fewer than min_cluster_size survivors (indicates potential over-flagging).

**Precision-First Semantics:** Pipeline targets precision > 0.9 at recall ≈ 0.1 per the paper — cluster membership is a strong coordination signal, not exhaustive coverage. Empty results (`knee_found = false`) are correct behaviour; investigate only if paired with rising account eligibility.

**Co-share Count Semantics:** Cluster `total_weight` = Σ over cluster URLs of C(sharing members, 2), i.e., the binomial count of unique account pairs sharing each URL in the cluster.

**Cluster Metrics:**
- `mean_edge_similarity` — mean cosine weight on edges in subgraph (0 if no edges)
- `subgraph_density` — (edges) / (possible edges), range [0, 1]

**Run Metadata:** Captures stage counts (accounts_raw/eligible, urls_eligible, graph_edges), chosen quantiles, whether knee was found, flagged account count, and cluster count written.

## Contract

- **Input:** `osprey_execution_results` (account DIDs, URL shares, dates)
- **Output:** `url_cosharing_clusters` (cluster results, metrics + `mean_edge_similarity`, `subgraph_density`, evolution), `url_cosharing_membership` (daily snapshots, no TTL — retained for post-hoc analysis since 2026-07-07), `url_cosharing_runs` (run metadata)
- **Dependencies:** ClickHouse only. `similarity.py` and `dismantling.py` have no I/O or ClickHouse imports (pure functional core).

## Commands

- `cd url_cosharing && uv run pytest` — Run tests
- `uv run python -m url_cosharing.calibrate` — Dump density-surface grid for calibration
- `uv run python -m url_cosharing.backfill START_DATE [END_DATE]` — Recompute historical run_dates (inclusive ISO dates, oldest→newest so evolution chains; overwrites existing rows per run_date; bounded by source-table retention)
- `docker compose up url-cosharing` — Start sidecar
