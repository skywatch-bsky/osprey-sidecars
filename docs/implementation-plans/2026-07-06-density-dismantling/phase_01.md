# Density-Dismantling Implementation Plan — Phase 1: Configuration and schema

**Goal:** All new detection knobs parse from env vars into the frozen `AnalysisConfig`; ClickHouse schema supports the new outputs (`url_cosharing_runs` table, two new cluster columns).

**Architecture:** The `url_cosharing` sidecar keeps its Functional Core / Imperative Shell split. This phase touches only the Core config module (`config.py`) and the cross-repo ClickHouse DDL in the sibling `skywatch-osprey` repo. No detection-path behaviour changes yet.

**Tech Stack:** Python 3 (uv project), pytest 8 + pytest-mock, frozen dataclasses, ClickHouse SQL (MergeTree DDL).

**Scope:** Phase 1 of 6 from `docs/design-plans/2026-07-06-density-dismantling.md`.

**Codebase verified:** 2026-07-06 (codebase-investigator, both repos at current HEAD; osprey-sidecars branch `density-dismantling`).

---

## Acceptance Criteria Coverage

This phase implements and tests:

### density-dismantling.AC4: Cross-cutting
- **density-dismantling.AC4.1:** Every new parameter is an env var with a documented default (`URL_COSHARING_*`), parsed into the frozen `AnalysisConfig`.

The schema task additionally lays groundwork for `density-dismantling.AC2.5` and `density-dismantling.AC1.5` (run-metadata rows), which are tested in Phase 5.

---

## Codebase Context (verified 2026-07-06)

- `url_cosharing/src/url_cosharing/config.py` — `# pattern: Functional Core`. Contains `_validate_table_name` (regex `^[a-zA-Z0-9_.]+$`), frozen `ClickHouseConfig`, frozen `AnalysisConfig` (11 fields, `from_env()` classmethod reading `URL_COSHARING_*`), frozen `AppConfig` composing both.
- `url_cosharing/tests/test_config.py` — class-based tests (`TestClickHouseConfig`, `TestAnalysisConfig`, `TestAppConfig`) using `monkeypatch.setenv`; covers defaults, overrides, type coercion, table-name validation, frozen-ness, direct construction.
- `/Users/scarndp/dev/skywatch/skywatch-osprey/clickhouse-init/05-url-cosharing.sql` — CREATE TABLE for `url_cosharing_pairs`, `url_cosharing_clusters` (14 columns ending in `jaccard_score Float64`), `url_cosharing_membership`, plus the pairs refreshable MV. No `url_cosharing_runs` table, no `mean_edge_similarity`/`subgraph_density` columns yet.
- Repo DDL migration convention (see `clickhouse-init/03-url-overdispersion.sql:28-33`): CREATE TABLE reflects target state for fresh installs, followed by idempotent `ALTER TABLE … ADD COLUMN IF NOT EXISTS … AFTER <col>` statements for already-deployed tables. No live `DROP` statements in init files.
- There is **no ClickHouse in CI**. DDL is verified statically at implementation time and applied for real at deploy time (see `docs/test-plans/2026-07-06-stats-methodology.md`, Phase 1 and E2E sections). The design's "DDL applies cleanly to a ClickHouse instance" is a deploy-time human verification step and is recorded as such in `test-requirements.md`.

**Sequencing adjustment vs. design (intentional):** the design assigns removal of `min_edge_weight`/`min_cosharers` to Phase 1, but `analyzer.py`, `main.py`, `queries.py`, and their tests still use those fields until the detection path is replaced in Phase 5. Removing them now would break the still-live pairs path and leave Phase 1 red. They are therefore **removed in Phase 5**, together with their call sites. Phase 1 only adds fields. The same applies to `pairs_table` (removed in Phase 5 when the last pairs-path consumer goes away).

**Default values chosen here (design leaves them open; all calibratable in Phase 6):**

| Field | Default | Rationale |
|---|---|---|
| `edge_epsilon` | `0.05` | Prunes negligible cosine similarities so the account×account product stays sparse; well below any plausible grid threshold. |
| `edge_quantile_grid` / `centrality_quantile_grid` | `0.50,0.60,0.70,0.80,0.90,0.95,0.99` | Design: "defaults spanning 0.50–0.99"; 7 points keep the grid 49 cells. |
| `density_floor` | `0.5` | A coordinated core per the source paper is near-clique; requiring ≥ half of possible edges filters weak "knees". |
| `max_flagged_fraction` | `0.02` | Paper's observed coordinated-account rates are 0.4–1.5%; 2% gives headroom while rejecting runaway threshold picks. |

---

<!-- START_SUBCOMPONENT_A (tasks 1-2) -->

<!-- START_TASK_1 -->
### Task 1: Extend `AnalysisConfig` with density-dismantling parameters

**Verifies:** density-dismantling.AC4.1

**Files:**
- Modify: `url_cosharing/src/url_cosharing/config.py`
- Test: `url_cosharing/tests/test_config.py` (unit)

**Implementation:**

Add two module-level helpers to `config.py` after `_validate_table_name` (lines 11–14):

```python
def _parse_quantile_grid(raw: str) -> tuple[float, ...]:
    parts = [part.strip() for part in raw.split(',') if part.strip()]
    if not parts:
        raise ValueError(f'quantile grid must contain at least one value: {raw!r}')
    values = tuple(float(part) for part in parts)
    for value in values:
        if not 0.0 < value < 1.0:
            raise ValueError(f'quantile grid values must be in (0, 1): {raw!r}')
    if list(values) != sorted(set(values)):
        raise ValueError(f'quantile grid must be strictly increasing: {raw!r}')
    return values


def _validate_unit_interval(name: str, value: float) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f'{name} must be in [0, 1]: {value!r}')
    return value
```

Extend `AnalysisConfig` (currently `config.py:36-68`) with ten new fields, inserted after `evolution_window_days` and before the `*_table` fields. Keep the existing 11 fields exactly as they are (including `min_edge_weight`, `min_cosharers`, `pairs_table` — see sequencing note above):

```python
    window_days: int
    min_unique_urls: int
    min_url_sharers: int
    max_url_df_pctl: float
    edge_epsilon: float
    edge_quantile_grid: tuple[float, ...]
    centrality_quantile_grid: tuple[float, ...]
    density_floor: float
    max_flagged_fraction: float
    runs_table: str
```

Extend `from_env()` with the corresponding entries (same insertion point, matching existing style):

```python
            window_days=int(os.environ.get('URL_COSHARING_WINDOW_DAYS', '7')),
            min_unique_urls=int(os.environ.get('URL_COSHARING_MIN_UNIQUE_URLS', '10')),
            min_url_sharers=int(os.environ.get('URL_COSHARING_MIN_URL_SHARERS', '5')),
            max_url_df_pctl=_validate_unit_interval(
                'URL_COSHARING_MAX_URL_DF_PCTL',
                float(os.environ.get('URL_COSHARING_MAX_URL_DF_PCTL', '0.90')),
            ),
            edge_epsilon=_validate_unit_interval(
                'URL_COSHARING_EDGE_EPSILON',
                float(os.environ.get('URL_COSHARING_EDGE_EPSILON', '0.05')),
            ),
            edge_quantile_grid=_parse_quantile_grid(
                os.environ.get('URL_COSHARING_EDGE_QUANTILE_GRID', '0.50,0.60,0.70,0.80,0.90,0.95,0.99')
            ),
            centrality_quantile_grid=_parse_quantile_grid(
                os.environ.get('URL_COSHARING_CENTRALITY_QUANTILE_GRID', '0.50,0.60,0.70,0.80,0.90,0.95,0.99')
            ),
            density_floor=_validate_unit_interval(
                'URL_COSHARING_DENSITY_FLOOR',
                float(os.environ.get('URL_COSHARING_DENSITY_FLOOR', '0.5')),
            ),
            max_flagged_fraction=_validate_unit_interval(
                'URL_COSHARING_MAX_FLAGGED_FRACTION',
                float(os.environ.get('URL_COSHARING_MAX_FLAGGED_FRACTION', '0.02')),
            ),
            runs_table=_validate_table_name(os.environ.get('URL_COSHARING_RUNS_TABLE', 'url_cosharing_runs')),
```

`tuple[float, ...]` (not `list`) keeps the frozen dataclass hashable and truly immutable.

**Testing:**

Write the tests first, watch them fail, then implement. Extend `TestAnalysisConfig` in `url_cosharing/tests/test_config.py` following its existing style (`monkeypatch.setenv`, one behaviour per test). Tests must verify AC4.1 for every new parameter:

- `test_from_env_defaults` (existing test, extend): each new field gets its documented default — `window_days == 7`, `min_unique_urls == 10`, `min_url_sharers == 5`, `max_url_df_pctl == 0.90`, `edge_epsilon == 0.05`, both grids `== (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99)`, `density_floor == 0.5`, `max_flagged_fraction == 0.02`, `runs_table == 'url_cosharing_runs'`.
- `test_from_env_overrides` (existing test, extend): set each new `URL_COSHARING_*` env var to a non-default value and assert it parses (e.g. `URL_COSHARING_EDGE_QUANTILE_GRID='0.6,0.8'` → `(0.6, 0.8)`).
- New tests for grid parsing: whitespace tolerated (`' 0.5, 0.9 '`), non-numeric raises `ValueError`, empty string raises `ValueError`, value ≥ 1.0 or ≤ 0.0 raises `ValueError`, non-increasing (`'0.9,0.5'`) and duplicate (`'0.5,0.5'`) raise `ValueError`.
- New tests for unit-interval validation: `URL_COSHARING_DENSITY_FLOOR='1.5'` and `URL_COSHARING_MAX_FLAGGED_FRACTION='-0.1'` raise `ValueError`.
- `runs_table` goes through table-name validation: `URL_COSHARING_RUNS_TABLE='bad;name'` raises `ValueError`.

Any existing fixture or test that constructs `AnalysisConfig` directly (e.g. `base_analysis_config` in `tests/test_analyzer.py` and `tests/test_main.py`) must gain the ten new constructor arguments — grep for `AnalysisConfig(` across `url_cosharing/` and update every direct construction site.

**Verification:**
Run: `cd url_cosharing && uv run pytest`
Expected: all tests pass (the full pre-existing suite plus the new config tests — zero failures).

**Commit:** `feat(url_cosharing): add density-dismantling config parameters`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: ClickHouse schema — `url_cosharing_runs` table and new cluster columns

**Verifies:** None directly (infrastructure; consumed by Phase 5 tests for density-dismantling.AC2.5/AC1.5; live apply is a deploy-time human verification step in test-requirements.md)

**Files:**
- Modify: `/Users/scarndp/dev/skywatch/skywatch-osprey/clickhouse-init/05-url-cosharing.sql` (sibling repo)

**Step 1: Branch in the sibling repo**

```bash
cd /Users/scarndp/dev/skywatch/skywatch-osprey
git status --porcelain   # must be clean; if not, STOP and surface to the user
git checkout -b density-dismantling
```

**Step 2: Edit the DDL**

Three changes to `clickhouse-init/05-url-cosharing.sql`:

1. In the `CREATE TABLE IF NOT EXISTS default.url_cosharing_clusters` statement, append two columns after `jaccard_score Float64` (target state for fresh installs):

```sql
    jaccard_score Float64,
    mean_edge_similarity Float64,
    subgraph_density Float64
```

2. Immediately after that CREATE TABLE statement, add idempotent migration statements for existing deployments, following the `03-url-overdispersion.sql:28-33` pattern:

```sql
-- Migration for existing deployments (idempotent; fresh installs already get
-- these columns from CREATE TABLE above). Added for the density-dismantling
-- methodology: mean cosine similarity over intra-cluster edges, and the edge
-- density of the cluster's surviving subgraph.
ALTER TABLE default.url_cosharing_clusters ADD COLUMN IF NOT EXISTS mean_edge_similarity Float64 AFTER jaccard_score;
ALTER TABLE default.url_cosharing_clusters ADD COLUMN IF NOT EXISTS subgraph_density Float64 AFTER mean_edge_similarity;
```

3. After the `url_cosharing_membership` CREATE TABLE, add the run-metadata table exactly per the design contract:

```sql
-- Table 4: url_cosharing_runs
-- One row per sidecar run: observability metadata for the density-dismantling
-- pipeline (filter-stage counts, chosen thresholds, guardrail outcomes).
-- Populated by the Python sidecar, no TTL.
CREATE TABLE IF NOT EXISTS default.url_cosharing_runs (
    run_date Date,
    window_days UInt8,
    accounts_raw UInt64,          -- before activity filter
    accounts_eligible UInt64,     -- after activity + df filters
    urls_eligible UInt64,
    graph_edges UInt64,
    edge_quantile Float64,        -- 0 when no knee selected
    centrality_quantile Float64,
    min_component_density Float64,
    knee_found Bool,
    guardrail_triggered Bool,
    flagged_accounts UInt64,
    cluster_count UInt32
) ENGINE = MergeTree()
ORDER BY run_date;
```

Leave `url_cosharing_pairs`, its MV, and `url_cosharing_membership` untouched (the pairs MV stays live for the `cosharing_pairs` MCP investigation tooling).

**Step 3: Verify statically**

```bash
cd /Users/scarndp/dev/skywatch/skywatch-osprey
grep -c 'ADD COLUMN IF NOT EXISTS' clickhouse-init/05-url-cosharing.sql   # expect: 2
grep -v '^--' clickhouse-init/05-url-cosharing.sql | grep -c 'DROP'       # expect: 0 (live statements; the pre-existing MIGRATION comment block mentions DROP but is commented)
grep -c 'CREATE TABLE IF NOT EXISTS default.url_cosharing_runs' clickhouse-init/05-url-cosharing.sql  # expect: 1
```

Confirm by eye: the two new cluster columns appear in BOTH the CREATE TABLE and the ALTER statements, with identical names and `Float64` type, and `url_cosharing_runs` column names/types match the design contract character-for-character (Phase 5's `db.py` insert `column_names` list must line up name-for-name — ClickHouse inserts by name).

**Step 4: Commit (in skywatch-osprey)**

```bash
cd /Users/scarndp/dev/skywatch/skywatch-osprey
git add clickhouse-init/05-url-cosharing.sql
git commit -m "feat: add url_cosharing_runs table and similarity columns for density-dismantling"
```

<!-- END_TASK_2 -->

<!-- END_SUBCOMPONENT_A -->

---

## Phase completion checklist

- [ ] `cd url_cosharing && uv run pytest` passes with the extended config tests (density-dismantling.AC4.1).
- [ ] `skywatch-osprey` branch `density-dismantling` carries the DDL commit; static grep checks pass.
- [ ] `min_edge_weight`, `min_cosharers`, `pairs_table` still present and functional (removal deferred to Phase 5).
