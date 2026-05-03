# URL Overdispersion Sidecar

Last verified: 2026-03-21

## Purpose

Detects anomalous domain sharing patterns using Poisson and normal-approximation statistics. Reads from `osprey_execution_results`, writes scored results to `url_overdispersion_results`.

## Architecture

Functional Core / Imperative Shell:
- `config.py` — env var parsing into frozen dataclasses (Core)
- `queries.py` — SQL query generation (Core)
- `analyzer.py` — Poisson volume and density scoring, baseline selection (Core)
- `db.py` — ClickHouse client wrapper (Shell)
- `main.py` — polling loop, signal handling (Shell)

## Contract

- **Input:** `osprey_execution_results` (posts with URLs, filtered to `Collection = 'app.bsky.feed.post'` and `OperationKind = 'create'`)
- **Output:** `url_overdispersion_results` (scored domains per time bucket)
- **Dependencies:** ClickHouse only. No imports from osprey_worker or other sidecars.

## Commands

- `cd url_overdispersion && uv run pytest` — Run tests
- `docker compose up url-overdispersion` — Start sidecar
