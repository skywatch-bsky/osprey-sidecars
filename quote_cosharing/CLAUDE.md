# Quote Post Co-Sharing Graph Sidecar

Last verified: 2026-07-06

## Purpose

Detects coordinated inauthentic behaviour by identifying clusters of accounts that quote-post the same target URIs on the same day. Builds a weighted co-sharing graph with Newman-weighted edges to prevent viral quoted URIs from manufacturing clusters, runs Leiden community detection, and tracks cluster evolution across days. Reads from `quote_cosharing_pairs` (populated by a ClickHouse scheduled materialized view), writes cluster results to `quote_cosharing_clusters` and membership snapshots to `quote_cosharing_membership`.

## Architecture

Functional Core / Imperative Shell:
- `config.py` — env var parsing into frozen dataclasses (Core)
- `queries.py` — SQL query generation (Core)
- `analyzer.py` — graph construction, Leiden clustering, per-cluster metrics, Jaccard evolution tracking (Core)
- `db.py` — ClickHouse client wrapper (Shell)
- `main.py` — polling loop, signal handling (Shell)

## Weighting Scheme

Pairs in `quote_cosharing_pairs` carry two weights:

1. **Raw weight** (co-quote count): The number of quoted URIs co-quoted by both accounts on the same day. Used for:
   - `min_edge_weight` filtering (default 2 — pairs must co-quote at least 2 URIs to enter the graph)
   - Investigation context (raw co-quote magnitude)

2. **Newman weight** (Σ 1/(k_uri − 1)): Per-pair sum of Newman's collaboration coefficients. For each quoted URI the pair co-quotes, contributes 1/(k_uri − 1) where k_uri is the number of accounts quoting that URI. Used for:
   - Leiden clustering to down-weight viral quoted URIs
   - Example: a niche URI quoted by 3 accounts contributes 0.5 per pair; a viral URI quoted by 500 contributes ~0.002
   - Prevents a single viral URI from manufacturing a heavy edge and triggering spurious cluster merges

**Cluster total_weight** remains the sum of raw weights over cluster edges — semantics: total co-quote count inside the cluster.

**Resolution re-tuning:** CPM quality compares community edge-weight density against the resolution parameter. Switching to Newman weights systematically reduces effective edge weights (each shared URI contributes ≤0.5 instead of 1), potentially requiring re-tuning `QUOTE_COSHARING_RESOLUTION` to maintain desired cluster density. See `docs/calibration.md` for calibration methodology.

## Contract

- **Input:** `quote_cosharing_pairs` (populated by ClickHouse scheduled materialized view from `osprey_execution_results`, includes `weight` and `newman_weight` columns)
- **Output:** `quote_cosharing_clusters` (cluster results with metrics and evolution, `total_weight` is raw sum), `quote_cosharing_membership` (daily membership snapshots, TTL 7 days)
- **Dependencies:** ClickHouse only. No imports from osprey_worker or other sidecars.

## Commands

- `cd quote_cosharing && uv run pytest` — Run tests
- `docker compose up quote-cosharing` — Start sidecar
