# Account Entropy Sidecar

Last verified: 2026-03-21

## Purpose

Computes Shannon entropy over posting behaviour to flag accounts whose timing patterns resemble automated tooling. Reads from `osprey_execution_results`, writes scored results to `account_entropy_results`.

## Architecture

Functional Core / Imperative Shell:
- `config.py` — env var parsing into frozen dataclasses (Core)
- `queries.py` — SQL query generation (Core)
- `analyzer.py` — Shannon entropy computation, bot-likelihood scoring (Core)
- `db.py` — ClickHouse client wrapper (Shell)
- `main.py` — polling loop, signal handling (Shell)

## Contract

- **Input:** `osprey_execution_results` (post timestamps for accounts with N+ posts, filtered to `Collection = 'app.bsky.feed.post'` and `OperationKind = 'create'`)
- **Output:** `account_entropy_results` (entropy scores per account per analysis window)
- **Dependencies:** ClickHouse only. No imports from osprey_worker or other sidecars.

## Commands

- `cd account_entropy && uv run pytest` — Run tests
- `docker compose up account-entropy` — Start sidecar
