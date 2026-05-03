# Quote Post Co-Sharing Graph Sidecar

Last verified: 2026-03-27

## Purpose

Detects coordinated behaviour by identifying clusters of accounts that quote-post the same target URIs. Builds a weighted co-sharing graph, runs Leiden community detection, and tracks cluster evolution across days. Reads from `quote_cosharing_pairs` (populated by a ClickHouse scheduled materialized view from `osprey_execution_results`), writes cluster results to `quote_cosharing_clusters` and membership snapshots to `quote_cosharing_membership`.

## Architecture

Functional Core / Imperative Shell:
- `config.py` — env var parsing into frozen dataclasses (Core)
- `queries.py` — SQL query generation (Core)
- `analyzer.py` — graph construction, Leiden clustering, per-cluster metrics, Jaccard evolution tracking (Core)
- `db.py` — ClickHouse client wrapper (Shell)
- `main.py` — polling loop, signal handling (Shell)

## Contract

- **Input:** `quote_cosharing_pairs` (populated by ClickHouse scheduled materialized view from `osprey_execution_results`)
- **Output:** `quote_cosharing_clusters` (cluster results with metrics and evolution), `quote_cosharing_membership` (daily membership snapshots, TTL 7 days)
- **Dependencies:** ClickHouse only. No imports from osprey_worker or other sidecars.

## Commands

- `cd quote_cosharing && uv run pytest` — Run tests
- `docker compose up quote-cosharing` — Start sidecar
