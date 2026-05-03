# URL Co-Sharing Graph Sidecar

Last verified: 2026-03-23

## Purpose

Detects coordinated inauthentic behaviour by identifying clusters of accounts that repeatedly share the same URLs on the same day. Builds a weighted co-sharing graph, runs Leiden community detection, and tracks cluster evolution across days. Reads from `url_cosharing_pairs` (populated by a ClickHouse scheduled query), writes cluster results to `url_cosharing_clusters` and membership snapshots to `url_cosharing_membership`.

## Architecture

Functional Core / Imperative Shell:
- `config.py` — env var parsing into frozen dataclasses (Core)
- `queries.py` — SQL query generation (Core)
- `analyzer.py` — graph construction, Leiden clustering, per-cluster metrics, Jaccard evolution tracking (Core)
- `db.py` — ClickHouse client wrapper (Shell)
- `main.py` — polling loop, signal handling (Shell)

## Contract

- **Input:** `url_cosharing_pairs` (populated by ClickHouse scheduled materialized view from `osprey_execution_results`)
- **Output:** `url_cosharing_clusters` (cluster results with metrics and evolution), `url_cosharing_membership` (daily membership snapshots, TTL 7 days)
- **Dependencies:** ClickHouse only. No imports from osprey_worker or other sidecars.

## Commands

- `cd url_cosharing && uv run pytest` — Run tests
- `docker compose up url-cosharing` — Start sidecar
