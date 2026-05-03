# Quote Post Overdispersion Sidecar

Last verified: 2026-03-27

## Purpose

Detects anomalous quote-post concentration targeting specific posts using Poisson and normal-approximation statistics. Reads from `osprey_execution_results`, writes scored results to `quote_overdispersion_results`.

## Architecture

Functional Core / Imperative Shell:
- `config.py` — env var parsing into frozen dataclasses (Core)
- `queries.py` — SQL query generation (Core)
- `analyzer.py` — Poisson volume and density scoring, baseline selection (Core)
- `db.py` — ClickHouse client wrapper (Shell)
- `main.py` — polling loop, signal handling (Shell)

## Contract

- **Input:** `osprey_execution_results` (posts with embedded quote-post URIs, filtered to `Collection = 'app.bsky.feed.post'` and `OperationKind = 'create'`)
- **Output:** `quote_overdispersion_results` (scored quoted URIs per time bucket)
- **Dependencies:** ClickHouse only. No imports from osprey_worker or other sidecars.

## Commands

- `cd quote_overdispersion && uv run pytest` — Run tests
- `docker compose up quote-overdispersion` — Start sidecar
