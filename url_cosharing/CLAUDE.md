# URL Co-Sharing Graph Sidecar

Last verified: 2026-07-06

## Purpose

Detects coordinated inauthentic behaviour by identifying clusters of accounts that repeatedly share the same URLs on the same day. Builds a weighted co-sharing graph with Newman-weighted edges to prevent viral URLs from manufacturing clusters, runs Leiden community detection, and tracks cluster evolution across days. Reads from `url_cosharing_pairs` (populated by a ClickHouse scheduled materialized view), writes cluster results to `url_cosharing_clusters` and membership snapshots to `url_cosharing_membership`.

## Architecture

Functional Core / Imperative Shell:
- `config.py` — env var parsing into frozen dataclasses (Core)
- `queries.py` — SQL query generation (Core)
- `analyzer.py` — graph construction, Leiden clustering, per-cluster metrics, Jaccard evolution tracking (Core)
- `db.py` — ClickHouse client wrapper (Shell)
- `main.py` — polling loop, signal handling (Shell)

## Weighting Scheme

Pairs in `url_cosharing_pairs` carry two weights:

1. **Raw weight** (co-share count): The number of URLs shared by both accounts on the same day. Used for:
   - `min_edge_weight` filtering (default 2 — pairs must co-share at least 2 URLs to enter the graph)
   - Investigation context (raw co-share magnitude)

2. **Newman weight** (Σ 1/(k_url − 1)): Per-pair sum of Newman's collaboration coefficients. For each URL the pair co-shares, contributes 1/(k_url − 1) where k_url is the number of accounts sharing that URL. Used for:
   - Leiden clustering to down-weight viral URLs
   - Example: a niche URL shared by 3 accounts contributes 0.5 per pair; a viral URL shared by 500 contributes ~0.002
   - Prevents a single viral URL from manufacturing a heavy edge and triggering spurious cluster merges

**Cluster total_weight** remains the sum of raw weights over cluster edges — semantics: total co-share count inside the cluster.

**Resolution re-tuning:** CPM quality compares community edge-weight density against the resolution parameter. Switching to Newman weights systematically reduces effective edge weights (each shared URL contributes ≤0.5 instead of 1), potentially requiring re-tuning `URL_COSHARING_RESOLUTION` to maintain desired cluster density. See `docs/calibration.md` for calibration methodology.

## Contract

- **Input:** `url_cosharing_pairs` (populated by ClickHouse scheduled materialized view from `osprey_execution_results`, includes `weight` and `newman_weight` columns)
- **Output:** `url_cosharing_clusters` (cluster results with metrics and evolution, `total_weight` is raw sum), `url_cosharing_membership` (daily membership snapshots, TTL 7 days)
- **Dependencies:** ClickHouse only. No imports from osprey_worker or other sidecars.

## Commands

- `cd url_cosharing && uv run pytest` — Run tests
- `docker compose up url-cosharing` — Start sidecar
