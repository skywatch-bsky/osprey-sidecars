# URL Co-Sharing Graph Sidecar

Last verified: 2026-07-07

## Purpose

Detects coordinated inauthentic behaviour by identifying clusters of accounts that repeatedly share the same URLs. Uses TF-IDF cosine-similarity network over per-account URL-sharing vectors (7-day rolling window) combined with density-based dismantling to isolate a high-precision coordinated core (Cinus, Minici, Luceri & Ferrara, WWW '25), then applies Leiden CPM decomposition to the core. Tracks cluster evolution across days via Jaccard similarity. 

Reads from `osprey_execution_results` (URL share records); `url_cosharing_pairs` is no longer consumed by the sidecar (the materialized view remains for investigation tooling). Writes cluster results to `url_cosharing_clusters`, membership snapshots to `url_cosharing_membership`, and run metadata to `url_cosharing_runs`.

## Architecture

Functional Core / Imperative Shell:
- `config.py` — env var parsing into frozen dataclasses (Core); `Exclusions` revision type, exclusions YAML parse/validate/hash, last-known-good fallback loader
- `queries.py` — SQL query generation (Core), including exclusion predicate generation (dot-anchored domain suffix + exact DID NOT IN) and the suppressed-shares audit count query
- `similarity.py` — build share matrix from URL share rows, TF-IDF transform, construct cosine-similarity graph (Core, no I/O)
- `dismantling.py` — density-based dismantling (grid search + knee detection + guardrails) to isolate coordinated cores (Core, no I/O)
- `analyzer.py` — Leiden CPM clustering on core with similarity weights, Jaccard evolution tracking (Core)
- `calibrate.py` — density-surface dump for offline tuning (Shell)
- `db.py` — ClickHouse client wrapper (Shell)
- `main.py` — polling loop, signal handling, orchestration (Shell); per-cycle exclusions hot reload
- `telemetry.py` — OpenTelemetry setup, no-op-safe handles, low-cardinality span/metric helpers (Shell)

## Detection Methodology

**TF-IDF Weighting:** Edge weight w(a, b) = cos(v_a, v_b) where v_a is account a's TF-IDF vector: tf = raw co-share count per URL, idf = ln(N / df), L2-normalized. Edge weights are in [0, 1].

**Mono-URL exclusion (SQL):** Accounts whose *eligible* URL set collapses to a single URL after the df floor are excluded in `fetch_url_shares_query`. Cosine over a 1-sparse vector is degenerate (exactly 0 or 1 after normalization), so such accounts form artificial similarity-1.0 cliques around viral one-shot links (e.g. community badge URLs) instead of contributing gradable co-sharing evidence. The ≥2 bound is the mathematical minimum for cosine to grade — not a tuning knob.

**Density-Based Dismantling:** Grid search over (edge_quantile, centrality_quantile) pairs; the selected cell is the one with the largest jump in minimum-component-density relative to its grid neighbours (the knee), among cells whose density ≥ density_floor that also satisfy the guardrails. Applies knee-detection heuristic to avoid over-flagging (knee_found = false on days with weak phase transitions is expected behaviour). Guardrails (ours, not the paper's — the paper prescribes no size cap): reject candidates if survived nodes exceed min(max_flagged_fraction × eligible, max_flagged_accounts) or fall below min_cluster_size. The absolute cap (default 750) governs normal days because observed coordinated cores are roughly constant in absolute size (paper: 25–764 across 6k–178k-account networks); the fraction (default 0.05) guards degenerate low-eligibility days.

**Precision-First Semantics:** Pipeline targets precision > 0.9 at recall ≈ 0.1 per the paper — cluster membership is a strong coordination signal, not exhaustive coverage. Empty results (`knee_found = false`) are correct behaviour; investigate only if paired with rising account eligibility.

**Co-share Count Semantics:** Cluster `total_weight` = Σ over cluster URLs of C(sharing members, 2), i.e., the binomial count of unique account pairs sharing each URL in the cluster.

**Cluster Metrics:**
- `mean_edge_similarity` — mean cosine weight on edges in subgraph (0 if no edges)
- `subgraph_density` — (edges) / (possible edges), range [0, 1]

**Run Metadata:** Captures stage counts (accounts_raw/eligible, urls_eligible, graph_edges), chosen quantiles, whether knee was found, flagged account count, cluster count written, and the exclusion audit fields (applied entry counts, revision hash, suppressed share rows).

**OpenTelemetry:** OTel instrumentation is operational observability only; `url_cosharing_runs` remains the durable domain audit table. OTel spans and metrics are low-cardinality run/stage signals: fixed stage names, run/window counts, knee/guardrail booleans, failure type, and durations. Never add DIDs, URLs/domains, `cluster_id`, sample URLs, sample DIDs, ClickHouse credentials, or exception messages as span attributes or metric labels. Keep OTel SDK setup in `telemetry.py`; functional core modules (`similarity.py`, `dismantling.py`, `analyzer.py`, `queries.py`) must not import OTel.

**Exclusions (gh issue #4):** Declared-benign accounts/URLs come from `exclusions.yaml` (path via `URL_COSHARING_EXCLUSIONS_FILE`) and are excluded at the SQL eligibility layer inside `fetch_url_shares_query`, BEFORE df/eligibility/mono-URL computation — an excluded URL stops counting toward `min_unique_urls` and the ≥2-URL floor, and a fleet whose entire link set is excluded drops out via existing rules. Domain entries match the exact host or a subdomain (dot-anchored suffix: `klipy.com` matches `static.klipy.com`, never `evil-klipy.com`); DIDs match exactly. The daemon hot-reloads the file per cycle (edits apply next cycle); a broken mid-flight edit retains last-known-good and logs an ERROR. `backfill`/`calibrate` load once and fail fast on invalid files. `accounts_raw` remains the PRE-exclusion window population, so exclusion suppression intentionally surfaces as extra `accounts_raw → accounts_eligible` attrition, quantified per run by `excluded_shares_suppressed`. Entry validation (hostname shape, full `did:plc` identifier = 24 chars of `[a-z2-7]`, no unknown keys) is the defence-in-depth that makes direct interpolation into generated SQL safe.

## Contract

- **Input:** `osprey_execution_results` (account DIDs, URL shares, dates); optional `exclusions.yaml` via `URL_COSHARING_EXCLUSIONS_FILE`
- **Output:** `url_cosharing_clusters` (cluster results, metrics + `mean_edge_similarity`, `subgraph_density`, evolution), `url_cosharing_membership` (daily snapshots, no TTL — retained for post-hoc analysis since 2026-07-07), `url_cosharing_runs` (run metadata incl. `excluded_domains_count`, `excluded_dids_count`, `exclusions_hash`, `excluded_shares_suppressed`)
- **Dependencies:** ClickHouse, PyYAML. `similarity.py` and `dismantling.py` have no I/O or ClickHouse imports (pure functional core).

## Commands

- `cd url_cosharing && uv run pytest` — Run tests (`URL_COSHARING_INTEGRATION_CH_HOST` unset skips the live-ClickHouse exclusion test; see `tests/test_integration_clickhouse.py`)
- `uv run python -m url_cosharing.calibrate` — Dump density-surface grid for calibration (honors exclusions)
- `uv run python -m url_cosharing.backfill START_DATE [END_DATE]` — Recompute historical run_dates (inclusive ISO dates, oldest→newest so evolution chains; overwrites existing rows per run_date; bounded by source-table retention; runs under the exclusions revision loaded at startup)
- `docker compose up url-cosharing` — Start sidecar (compose mounts the exclusions file read-only)
