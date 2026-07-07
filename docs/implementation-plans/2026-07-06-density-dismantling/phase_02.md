# Density-Dismantling Implementation Plan — Phase 2: Bipartite data access

> **Superseded (2026-07-07, issue #3):** the URL df ceiling described in this document as a percentile of the df distribution (`max_url_df_pctl` / `quantile(max_url_df_pctl)(df)`) was a mis-transcription of Cinus et al.'s published code and is degenerate on production data. The implemented contract is `max_url_df_fraction` (`URL_COSHARING_MAX_URL_DF_FRACTION`): eligible URLs satisfy `df <= max_url_df_fraction * distinct_account_count` (sklearn `max_df` semantics), applied in SQL only. Do not reintroduce percentile/quantile ceiling logic from this document.

**Goal:** Fetch filtered `(did, url, share_count)` rows for the rolling window directly from `osprey_execution_results`, with the activity and document-frequency filters pushed into SQL CTEs.

**Architecture:** `queries.py` gains a pure string-builder `fetch_url_shares_query` (Functional Core); `db.py` gains a typed `fetch_url_shares` method (Imperative Shell). The `UrlShareRow` type is seeded into a new Core module `similarity.py` so that Shell imports Core (same direction as `db.py`'s existing `PairRow` import from `analyzer.py`) — Phase 3 fills in the rest of `similarity.py`.

**Tech Stack:** Python 3 (uv project), pytest 8 + pytest-mock, clickhouse-connect, ClickHouse SQL (CTEs, `quantile`).

**Scope:** Phase 2 of 6 from `docs/design-plans/2026-07-06-density-dismantling.md`.

**Codebase verified:** 2026-07-07 (codebase-investigator; verbatim queries.py/db.py reviewed).

---

## Acceptance Criteria Coverage

This phase implements and tests (at the SQL/builder level; Phase 3 re-verifies AC1.2–AC1.3 in Python, Phase 5 verifies AC1.1 end-to-end):

### density-dismantling.AC1: Similarity network construction
- **density-dismantling.AC1.1 Success:** A daily run fetches per-account URL share counts over the configured rolling window (default 7 days) from `osprey_execution_results`, not from `url_cosharing_pairs`.
- **density-dismantling.AC1.2 Success:** Accounts sharing fewer than `min_unique_urls` (default 10) unique URLs in the window are excluded before matrix construction.
- **density-dismantling.AC1.3 Success:** URLs shared by fewer than `min_url_sharers` (default 5) accounts, or with document frequency above the `max_url_df_pctl` (default 0.90) percentile, are excluded before TF-IDF.

---

## Codebase Context (verified 2026-07-07)

- `url_cosharing/src/url_cosharing/queries.py` (`# pattern: Functional Core`) — five builders today: `fetch_pairs_query`, `fetch_historical_membership_query`, `fetch_member_timestamps_query`, `insert_clusters_query`, `insert_membership_query`. Style: f-string interpolation of `config.*_table` names (validated at config construction) and numeric config values (e.g. `config.evolution_window_days` interpolated directly). `fetch_member_timestamps_query` is the existing template for reading `osprey_execution_results`: filters `Collection = 'app.bsky.feed.post'`, `OperationKind = 'create'`, `toDate(__timestamp)` date bounds, timestamp column `__timestamp`, account column `UserId`.
- The pairs MV DDL (`skywatch-osprey/clickhouse-init/05-url-cosharing.sql`) extracts URLs via `arrayJoin(FacetLinkList)` with `length(FacetLinkList) > 0` — the same extraction applies here.
- `url_cosharing/src/url_cosharing/db.py` (`# pattern: Imperative Shell`) — class `CosharingDb`; fetch methods take a query string, run `self._client.query(query, settings={'max_execution_time': 120})`, and map `result.result_rows` positionally into frozen dataclasses. Row types: `PairRow` (defined in `analyzer.py`), `MembershipRow`/`MemberTimestamp` (defined in `db.py`).
- Tests: `tests/test_queries.py` asserts substrings of the built SQL per behaviour (one test per clause, class-per-builder, `base_config` fixture); `tests/test_db.py` patches `url_cosharing.db.clickhouse_connect.get_client` and feeds `mock_result.result_rows` tuples.
- ClickHouse quantile idiom confirmed in `docs/calibration.md`: `quantile(0.5)(total_weight)`.

**Sequencing adjustment vs. design (intentional):** the design removes `fetch_pairs_query` in this phase, but `main.py` still orchestrates the pairs-based detection path until Phase 5 replaces it. `fetch_pairs_query`, `CosharingDb.fetch_pairs`, and their tests are therefore **removed in Phase 5**, not here. This phase only adds the new data access.

**Filter-order decision (design leaves it open):** both filters are computed over the *raw* window shares in a single pass — URL document frequency is counted before any account is dropped, and account activity (unique-URL count) is counted before any URL is dropped; the final row set keeps rows where the account passes the activity filter AND the URL passes the df filters. This single-pass semantics is simple, order-independent, and is what `similarity.py`'s in-Python re-application (Phase 3) must mirror. Consequence: an account can arrive at matrix construction with fewer than `min_unique_urls` *surviving* URLs — that is by design, and Phase 3's in-Python re-check uses the same raw-count semantics.

---

<!-- START_SUBCOMPONENT_A (tasks 1-3) -->

<!-- START_TASK_1 -->
### Task 1: Seed `similarity.py` with the `UrlShareRow` type

**Verifies:** None directly (type container; enables Task 2/3 typing — density-dismantling.AC4.2 pattern compliance starts here)

**Files:**
- Create: `url_cosharing/src/url_cosharing/similarity.py`

**Implementation:**

Create the module with exactly this content (Phase 3 extends it with matrix/TF-IDF/graph functions):

```python
# pattern: Functional Core
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UrlShareRow:
    did: str
    url: str
    share_count: int
```

`UrlShareRow` lives in the Core (`similarity.py`) rather than `db.py` so that Phase 3's pure code never imports from the Shell; `db.py` importing it mirrors the existing `PairRow`-from-`analyzer` pattern.

**Verification:**
Run: `cd url_cosharing && uv run pytest`
Expected: suite still passes (no behaviour change).

**Commit:** combined with Task 2 (see Task 3).
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: `fetch_url_shares_query` builder

**Verifies:** density-dismantling.AC1.1, density-dismantling.AC1.2, density-dismantling.AC1.3 (SQL level)

**Files:**
- Modify: `url_cosharing/src/url_cosharing/queries.py` (add builder after `fetch_pairs_query`; leave existing builders untouched)
- Test: `url_cosharing/tests/test_queries.py` (unit)

**Implementation:**

```python
def fetch_url_shares_query(config: AnalysisConfig) -> str:
    return f"""
        WITH url_shares AS (
            SELECT
                UserId AS did,
                arrayJoin(FacetLinkList) AS url,
                count() AS share_count
            FROM {config.source_table}
            WHERE Collection = 'app.bsky.feed.post'
                AND OperationKind = 'create'
                AND toDate(__timestamp) >= yesterday() - {config.window_days - 1}
                AND toDate(__timestamp) <= yesterday()
                AND length(FacetLinkList) > 0
            GROUP BY did, url
        ),
        url_df AS (
            SELECT url, uniqExact(did) AS df
            FROM url_shares
            GROUP BY url
        ),
        eligible_urls AS (
            SELECT url
            FROM url_df
            WHERE df >= {config.min_url_sharers}
                AND df <= (SELECT quantile({config.max_url_df_pctl})(df) FROM url_df)
        ),
        active_accounts AS (
            SELECT did
            FROM url_shares
            GROUP BY did
            HAVING uniqExact(url) >= {config.min_unique_urls}
        )
        SELECT
            s.did,
            s.url,
            s.share_count
        FROM url_shares s
        WHERE s.url IN (SELECT url FROM eligible_urls)
            AND s.did IN (SELECT did FROM active_accounts)
    """
```

Notes:
- Window semantics: `window_days` days ending **yesterday** inclusive (`yesterday() - (window_days - 1)` … `yesterday()`), consistent with the daily-batch cadence of the rest of the sidecar (the old MV used `= yesterday()`).
- Numeric config values are interpolated as typed ints/floats from the frozen config, matching the existing `fetch_historical_membership_query` style; table names are `_validate_table_name`-guarded at config construction.
- `uniqExact` (not approximate `uniq`) for both df and the activity filter — these feed hard thresholds.
- The df ceiling uses the repo-standard `quantile(q)(col)` idiom in a scalar subquery over `url_df`.
- Both `eligible_urls` and `active_accounts` are computed over the raw `url_shares` CTE per the filter-order decision above.

**Testing:**

Add `TestFetchUrlSharesQuery` to `tests/test_queries.py`, following the existing one-clause-per-test substring style with the `base_config` fixture (fixture must first gain the Phase 1 config fields if it constructs `AnalysisConfig` directly):

- AC1.1: query reads `FROM {config.source_table}` (assert `'osprey_execution_results'` present and `'url_cosharing_pairs'` absent); window bounds present (`'yesterday() - 6'` for default `window_days=7`, `'<= yesterday()'`); collection/operation filters present (`"Collection = 'app.bsky.feed.post'"`, `"OperationKind = 'create'"`); URL extraction via `'arrayJoin(FacetLinkList)'` and `'length(FacetLinkList) > 0'`; selects `did`, `url`, `share_count`.
- AC1.2: `HAVING uniqExact(url) >= 10` present with default config; changing `min_unique_urls` in config changes the literal.
- AC1.3: `df >= 5` present with default config; `quantile(0.9)(df)` present with default config. Assertion-style caveat: Python's f-string renders the float `0.90` as `0.9`, so assert against the *formatted* value — the robust pattern is `f'quantile({config.max_url_df_pctl})(' in query` rather than a hard-coded string, and the same applies when overriding `min_url_sharers`/`max_url_df_pctl` in the override test.
- Custom `window_days` (e.g. 1) produces `yesterday() - 0`.

**Verification:**
Run: `cd url_cosharing && uv run pytest tests/test_queries.py`
Expected: new tests pass, existing builder tests untouched and passing.

**Commit:** combined with Task 3.
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: `CosharingDb.fetch_url_shares`

**Verifies:** density-dismantling.AC1.1 (typed fetch path)

**Files:**
- Modify: `url_cosharing/src/url_cosharing/db.py`
- Test: `url_cosharing/tests/test_db.py` (unit, mocked client)

**Implementation:**

Add the import (extend the existing import block):

```python
from url_cosharing.similarity import UrlShareRow
```

Add a fetch method to `CosharingDb`, after `fetch_pairs` (which stays until Phase 5):

```python
    def fetch_url_shares(self, query: str) -> list[UrlShareRow]:
        result = self._client.query(
            query,
            settings={'max_execution_time': 300},
        )
        rows = []
        for row in result.result_rows:
            rows.append(
                UrlShareRow(
                    did=row[0],
                    url=row[1],
                    share_count=int(row[2]),
                )
            )
        return rows
```

`max_execution_time: 300` (vs. the 120 used by the single-day fetches) because this query scans a 7-day window of `osprey_execution_results`; calibration (Phase 6) may tune this.

**Testing:**

Add `TestFetchUrlShares` to `tests/test_db.py`, following the existing `@patch('url_cosharing.db.clickhouse_connect.get_client')` pattern:

- Maps columns positionally: `result_rows = [('did:plc:a', 'https://x.test/1', 3), ...]` → `UrlShareRow(did='did:plc:a', url='https://x.test/1', share_count=3)`; `share_count` coerced to `int`.
- Passes `settings={'max_execution_time': 300}` to `client.query`.
- Empty `result_rows` returns `[]`.

**Verification:**
Run: `cd url_cosharing && uv run pytest`
Expected: full suite passes.

**Commit:**
```bash
git add url_cosharing/src/url_cosharing/similarity.py url_cosharing/src/url_cosharing/queries.py url_cosharing/src/url_cosharing/db.py url_cosharing/tests/test_queries.py url_cosharing/tests/test_db.py
git commit -m "feat(url_cosharing): fetch per-account URL shares from osprey_execution_results"
```
<!-- END_TASK_3 -->

<!-- END_SUBCOMPONENT_A -->

---

## Phase completion checklist

- [ ] `cd url_cosharing && uv run pytest` passes.
- [ ] `fetch_url_shares_query` tests pin window bounds, activity filter, df floor, and df-percentile ceiling (density-dismantling.AC1.1–AC1.3 at SQL level).
- [ ] `fetch_pairs_query` / `fetch_pairs` still present (removal deferred to Phase 5).
- [ ] `similarity.py` exists with `# pattern: Functional Core` and only `UrlShareRow` so far.
