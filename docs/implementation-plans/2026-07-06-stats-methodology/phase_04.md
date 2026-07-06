# Statistical Methodology Fixes — Phase 4: account_entropy Implementation Plan

**Goal:** Replace raw-bit entropy thresholds with bias-corrected, normalized entropy plus a coefficient-of-variation signal, so low-activity accounts can be scored fairly.

**Architecture:** All changes are Functional Core (`analyzer.py`, `config.py`) plus mechanical column additions in the Shell (`db.py`) and the ClickHouse schema. No SQL query changes. The sidecar keeps its standalone-project constraint (no shared imports).

**Tech Stack:** Python 3.11+, pure `math` stdlib (no new dependencies), clickhouse-connect, pytest via `uv run pytest`.

**Scope:** Phase 4 of 7 from `docs/design-plans/2026-07-06-stats-methodology.md` (independent of phases 1–3, 5–6).

**Codebase verified:** 2026-07-06 via codebase-investigator agents.

---

## Acceptance Criteria Coverage

This phase implements and tests:

### stats-methodology.AC5: Entropy
- **stats-methodology.AC5.1 Success:** `normalized_entropy` = 1.0 for uniform over achievable bins, 0.0 for single-bin, always in [0, 1]
- **stats-methodology.AC5.2 Success:** Miller–Madow correction applied and numerically verified
- **stats-methodology.AC5.3 Success:** An account with 10 uniformly-spread posts can cross the hourly threshold (impossible under the old 3.9-bit rule)
- **stats-methodology.AC5.4 Success:** `interval_cv` computed; `cv_flag` when CV ≤ threshold; `is_bot_like = hourly_flag AND (interval_flag OR cv_flag)`

### stats-methodology.AC7: Schemas
- **stats-methodology.AC7.1 Success:** All seven clickhouse-init files updated consistently with sidecar insert column lists (unit-tested per sidecar) — *account_entropy scope: `04-account-entropy.sql`*

### stats-methodology.AC9: Suites
- **stats-methodology.AC9.1 Success:** `uv run pytest` passes in all six sidecars — *account_entropy scope*

---

## Context from codebase verification

Current state (all paths verified 2026-07-06):

- `account_entropy/src/account_entropy/analyzer.py` (183 lines) has `compute_entropy(counts)` (Shannon bits, lines 11–31), `compute_hourly_entropy(hourly_bins)` (24-bin histogram, lines 34–51), `compute_interval_entropy(ordered_timestamps_ms, bin_edges) -> tuple[float, float, float]` returning `(entropy, mean_interval, stddev_interval)` (lines 54–99), `score_account` (lines 102–157), `score_accounts` (lines 160–182).
- `mean_interval_seconds` / `stddev_interval_seconds` are **already computed, stored, and inserted** — the design's "unused mean/stddev" item resolves to deriving `interval_cv` from them.
- Current flags (analyzer.py lines 133–141): `hourly_flag = hourly_entropy >= 3.9` bits, `interval_flag = interval_entropy <= 1.5` bits, `is_bot_like = hourly_flag AND interval_flag`.
- `config.py` `AnalysisConfig` (frozen) has `hourly_entropy_threshold` (env `ACCOUNT_ENTROPY_HOURLY_ENTROPY_THRESHOLD`, default 3.9) and `interval_entropy_threshold` (env `ACCOUNT_ENTROPY_INTERVAL_ENTROPY_THRESHOLD`, default 1.5), plus `interval_bin_edges` default `(60, 300, 900, 3600, 14400, 86400)` — 7 achievable interval bins.
- `db.py` `ScoredResult` has 13 fields; insert column list at lines 68–74.
- Schema: `/Users/scarndp/dev/skywatch/skywatch-osprey/clickhouse-init/04-account-entropy.sql`, table `default.account_entropy_results`, 13 columns, `ORDER BY (run_timestamp, user_id)`.
- Tests: `uv run pytest` from `account_entropy/`; pure-function tests in `tests/test_analyzer.py`, config tests via `monkeypatch`, `FakeDb` stub in `tests/test_main.py`. Ruff: single quotes, 120 cols, py311.

Threshold conversion math (for reference): old hourly 3.9 bits / log2(24) ≈ 4.585 bits = 0.8506 → **0.85**; old interval 1.5 bits / log2(7) ≈ 2.807 bits = 0.534 → **0.53**.

---

<!-- START_SUBCOMPONENT_A (tasks 1-2) -->
<!-- START_TASK_1 -->
### Task 1: `normalized_entropy` with Miller–Madow correction

**Verifies:** stats-methodology.AC5.1, stats-methodology.AC5.2

**Files:**
- Modify: `account_entropy/src/account_entropy/analyzer.py` (add function after `compute_entropy`, i.e. after line 31)
- Test: `account_entropy/tests/test_analyzer.py` (unit)

**Implementation:**

Add to `analyzer.py` (Functional Core — no I/O, deterministic):

```python
def normalized_entropy(counts: list[int], max_bins: int) -> float:
    """Bias-corrected Shannon entropy rescaled to [0, 1].

    Applies the Miller-Madow correction, H_mm = H + (K_occupied - 1) / (2 * N * ln 2)
    bits, where K_occupied is the number of non-zero bins and N the total count,
    then divides by the achievable maximum log2(min(N, max_bins)).

    Returns 0.0 when N < 2 or max_bins < 2 (no meaningful spread is measurable).
    Result is clamped to [0.0, 1.0] because the bias correction can push the
    corrected estimate above the achievable maximum for small N.
    """
    total = sum(counts)
    if total < 2 or max_bins < 2:
        return 0.0
    occupied = sum(1 for c in counts if c > 0)
    corrected = compute_entropy(counts) + (occupied - 1) / (2 * total * math.log(2))
    achievable = math.log2(min(total, max_bins))
    return min(1.0, max(0.0, corrected / achievable))
```

Notes:
- `math` is already imported in `analyzer.py`.
- `(occupied - 1) / (2 * total * math.log(2))` is the Miller–Madow correction expressed in bits (the natural-log form is `(K-1)/(2N)` nats; dividing by `ln 2` converts to bits).
- The `min(total, max_bins)` denominator is the point of the fix: an account with 10 posts can at most achieve `log2(10)` bits over 24 hourly bins, so it is normalized against `log2(10)`, not `log2(24)`.

**Testing:**

Add a `TestNormalizedEntropy` class to `tests/test_analyzer.py`. Tests must verify:

- AC5.1 (uniform over achievable bins → 1.0): `normalized_entropy([10] * 24, 24) == 1.0` (N=240 ≥ 24 bins; corrected entropy exceeds log2(24) and clamps to 1.0) and `normalized_entropy([1] * 10 + [0] * 14, 24) == 1.0` (10 posts in 10 distinct hours: H = log2(10), correction pushes above the achievable max log2(10), clamps to 1.0).
- AC5.1 (single bin → 0.0): `normalized_entropy([10, 0, 0], 24) == 0.0` (H = 0, K_occupied = 1 so the correction term is 0).
- AC5.1 (bounds): for several arbitrary histograms (e.g. `[3, 1, 0, 7]`, `[1, 1]`, `[100, 1]`), result is within `[0.0, 1.0]`.
- AC5.2 (numeric verification of Miller–Madow): `normalized_entropy([5, 5], 24)` must equal `(1.0 + 1 / (20 * math.log(2))) / math.log2(10)` — hand computation: H = 1.0 bit, correction = (2−1)/(2·10·ln 2) ≈ 0.072135 bits, achievable = log2(min(10, 24)) = log2(10) ≈ 3.321928, so expected ≈ **0.322745**. Assert with `pytest.approx(..., rel=1e-9)` against the closed-form expression, not a rounded literal.
- Edge: `normalized_entropy([], 24) == 0.0`, `normalized_entropy([1], 24) == 0.0` (N < 2), `normalized_entropy([2, 3], 1) == 0.0` (max_bins < 2).

Follow the existing test style in `tests/test_analyzer.py` (plain classes, descriptive `test_...` names, no fixtures needed for pure functions).

**Verification:**
Run: `cd account_entropy && uv run pytest tests/test_analyzer.py`
Expected: all tests pass, including pre-existing ones.

**Commit:** `feat: add bias-corrected normalized entropy to account_entropy analyzer`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: `coefficient_of_variation`

**Verifies:** stats-methodology.AC5.4 (CV computation half; flag wiring lands in Task 4)

**Files:**
- Modify: `account_entropy/src/account_entropy/analyzer.py` (add function after `normalized_entropy`)
- Test: `account_entropy/tests/test_analyzer.py` (unit)

**Implementation:**

```python
def coefficient_of_variation(mean: float, stddev: float) -> float:
    """Scale-free variability of inter-post intervals: stddev / mean.

    Returns 0.0 when mean <= 0 (fewer than two posts, or degenerate
    zero-length intervals) — maximally regular by convention.
    """
    if mean <= 0:
        return 0.0
    return stddev / mean
```

The `mean <= 0` convention is intentional: a zero mean interval means all posts share a timestamp, which is the most machine-like cadence possible, so it should not exempt an account from the CV signal. `is_bot_like` still requires the hourly flag (Task 4), so this cannot flag on its own.

**Testing:**

Add a `TestCoefficientOfVariation` class:
- Regular cadence: `coefficient_of_variation(100.0, 5.0) == pytest.approx(0.05)`.
- Irregular cadence: `coefficient_of_variation(100.0, 150.0) == pytest.approx(1.5)`.
- Edge: `coefficient_of_variation(0.0, 0.0) == 0.0` and `coefficient_of_variation(-1.0, 5.0) == 0.0`.

**Verification:**
Run: `cd account_entropy && uv run pytest tests/test_analyzer.py`
Expected: all tests pass.

**Commit:** `feat: add interval coefficient-of-variation helper to account_entropy`
<!-- END_TASK_2 -->
<!-- END_SUBCOMPONENT_A -->

<!-- START_SUBCOMPONENT_B (tasks 3-5) -->
<!-- START_TASK_3 -->
### Task 3: Normalized-scale thresholds in config

**Verifies:** stats-methodology.AC5.4 (threshold plumbing; asserted end-to-end in Task 4)

**Files:**
- Modify: `account_entropy/src/account_entropy/config.py:36-70` (`AnalysisConfig`)
- Test: `account_entropy/tests/test_config.py` (unit)

**Implementation:**

In `AnalysisConfig`, replace the two raw-bit threshold fields with three normalized-scale fields. Field list becomes:

| Field | Env var | Default |
|---|---|---|
| `hourly_entropy_norm_threshold` (float) | `ACCOUNT_ENTROPY_HOURLY_NORM_THRESHOLD` | `0.85` |
| `interval_entropy_norm_threshold` (float) | `ACCOUNT_ENTROPY_INTERVAL_NORM_THRESHOLD` | `0.53` |
| `cv_threshold` (float) | `ACCOUNT_ENTROPY_CV_THRESHOLD` | `0.5` |

Remove `hourly_entropy_threshold` / `interval_entropy_threshold` and their env vars (`ACCOUNT_ENTROPY_HOURLY_ENTROPY_THRESHOLD`, `ACCOUNT_ENTROPY_INTERVAL_ENTROPY_THRESHOLD`). This is a deliberate breaking rename: the old and new values live on different scales, and silently reusing the old env var names would misconfigure deployments (a leftover `...HOURLY_ENTROPY_THRESHOLD=3.9` interpreted on a 0–1 scale would disable the signal entirely).

All other fields (`interval_seconds`, `window_days`, `min_posts`, `interval_bin_edges`, `source_table`, `output_table`) are unchanged. Keep the frozen dataclass + `from_env()` pattern exactly as-is.

**Testing:**

Update `tests/test_config.py`:
- Defaults test asserts the three new fields equal `0.85`, `0.53`, `0.5` when env is clean.
- Override test sets `ACCOUNT_ENTROPY_HOURLY_NORM_THRESHOLD=0.9`, `ACCOUNT_ENTROPY_INTERVAL_NORM_THRESHOLD=0.4`, `ACCOUNT_ENTROPY_CV_THRESHOLD=0.3` via `monkeypatch.setenv` and asserts parsing.
- Delete/replace the tests referencing the removed fields.

**Verification:**
Run: `cd account_entropy && uv run pytest tests/test_config.py`
Expected: all tests pass. (`tests/test_analyzer.py` and `tests/test_main.py` will fail to construct `AnalysisConfig` until Task 4 updates them — run only `test_config.py` here; the full suite gate is Task 7.)

**Commit:** combined with Task 4 (config rename and scoring change are one logical unit; committing the rename alone would leave a broken build, violating bisectability).
<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: Revised `score_account`, row types, and insert columns

**Verifies:** stats-methodology.AC5.3, stats-methodology.AC5.4

**Files:**
- Modify: `account_entropy/src/account_entropy/analyzer.py:54-157` (`compute_interval_entropy` internals, `score_account`)
- Modify: `account_entropy/src/account_entropy/db.py:22-36` (`ScoredResult`), `db.py:68-74` (insert columns)
- Test: `account_entropy/tests/test_analyzer.py`, `account_entropy/tests/test_db.py`, `account_entropy/tests/test_main.py` (unit)

**Implementation:**

1. **Expose histograms.** `normalized_entropy` needs bin counts, but `compute_hourly_entropy` and `compute_interval_entropy` build their histograms internally and discard them. Add one pure helper and reimplement the two existing functions on top of it so their public signatures and behaviour are unchanged:

```python
def interval_histogram(
    ordered_timestamps_ms: list[int],
    bin_edges: tuple[int, ...],
) -> tuple[list[int], float, float]:
    """Histogram of inter-post intervals plus (mean, stddev) of the intervals in seconds.

    Bin structure: [0, edge1), [edge1, edge2), ..., [edgeN, inf).
    Returns ([0] * (len(bin_edges) + 1), 0.0, 0.0) when fewer than 2 timestamps.
    """
```

   Move the interval/mean/stddev/binning body of `compute_interval_entropy` (currently lines 74–98) into `interval_histogram`; `compute_interval_entropy` becomes:

```python
def compute_interval_entropy(
    ordered_timestamps_ms: list[int],
    bin_edges: tuple[int, ...],
) -> tuple[float, float, float]:
    histogram, mean_interval, stddev_interval = interval_histogram(ordered_timestamps_ms, bin_edges)
    return compute_entropy(histogram), mean_interval, stddev_interval
```

   For the hourly side, `score_account` builds the 24-bin histogram inline (`histogram = [0] * 24; for hour in row.hourly_bins: histogram[hour] += 1`) and feeds it to both `compute_entropy` and `normalized_entropy`. Keep `compute_hourly_entropy` as-is for API stability (existing tests cover it).

2. **Rewrite the scoring block of `score_account`** (keep signature and the surrounding structure):

```python
    hourly_hist = [0] * 24
    for hour in row.hourly_bins:
        hourly_hist[hour] += 1
    hourly_entropy = compute_entropy(hourly_hist)
    hourly_entropy_norm = normalized_entropy(hourly_hist, 24)

    interval_hist, mean_interval, stddev_interval = interval_histogram(
        row.ordered_timestamps, config.interval_bin_edges
    )
    interval_entropy = compute_entropy(interval_hist)
    interval_entropy_norm = normalized_entropy(interval_hist, len(config.interval_bin_edges) + 1)
    interval_cv = coefficient_of_variation(mean_interval, stddev_interval)

    hourly_flag = 1 if hourly_entropy_norm >= config.hourly_entropy_norm_threshold else 0
    interval_flag = 1 if interval_entropy_norm <= config.interval_entropy_norm_threshold else 0
    cv_flag = 1 if interval_cv <= config.cv_threshold else 0

    is_bot_like = 1 if (hourly_flag == 1 and (interval_flag == 1 or cv_flag == 1)) else 0
```

   Raw `hourly_entropy` / `interval_entropy` (bits) are retained in the result as context, per the design.

3. **`ScoredResult`** gains four fields (insert after `stddev_interval_seconds`, before the flags, to keep signal fields grouped): `hourly_entropy_norm: float`, `interval_entropy_norm: float`, `interval_cv: float`, and `cv_flag: int` (place `cv_flag` alongside `hourly_flag` / `interval_flag`). Update the insert column list in `db.py` to match — final list:

```python
column_names = [
    'run_timestamp', 'user_id', 'window_start', 'window_end',
    'post_count', 'hourly_entropy', 'interval_entropy',
    'hourly_entropy_norm', 'interval_entropy_norm',
    'mean_interval_seconds', 'stddev_interval_seconds', 'interval_cv',
    'is_bot_like', 'hourly_flag', 'interval_flag', 'cv_flag',
    'sample_rkeys',
]
```

   and update the row-building code so values align with this order.

**Testing:**

Tests must verify each AC listed above:

- AC5.3: an `AccountActivityRow` with 10 posts spread over 10 distinct hours (e.g. hours `0..9`, timestamps 1 hour + jitter apart so intervals land in ≥2 bins) yields `hourly_flag == 1` with the default 0.85 threshold. Include the contrast assertion in a comment or companion test: the same account's raw `hourly_entropy` is `log2(10) ≈ 3.32 < 3.9`, i.e. it could never fire under the old rule.
- AC5.4 (CV path): an account with metronomic intervals (e.g. exactly 3600s apart → stddev 0 → CV 0) but interval entropy concentrated in one bin gets `cv_flag == 1`; an account with wildly varying intervals (CV > 0.5) gets `cv_flag == 0`.
- AC5.4 (conjunction): construct three rows to pin the truth table — (a) `hourly_flag=1, interval_flag=0, cv_flag=1` → `is_bot_like == 1`; (b) `hourly_flag=1, interval_flag=0, cv_flag=0` → `is_bot_like == 0`; (c) `hourly_flag=0, interval_flag=1, cv_flag=1` → `is_bot_like == 0`.
- `compute_interval_entropy` regression: existing tests must still pass unchanged (the refactor through `interval_histogram` is behaviour-preserving).
- `tests/test_db.py`: `ScoredResult` construction with the four new fields; insert test asserts the 17-column list above is passed to the client.
- `tests/test_main.py`: update `FakeDb`-based cycle tests for the new fields; assert new columns flow through `run_cycle` into the insert.
- Update every existing test that constructs `AnalysisConfig` or `ScoredResult` to the new fields/thresholds.

**Verification:**
Run: `cd account_entropy && uv run pytest`
Expected: full suite passes.

**Commit:** `feat: normalized entropy thresholds and CV signal in account_entropy scoring`
<!-- END_TASK_4 -->

<!-- START_TASK_5 -->
### Task 5: ClickHouse schema additions

**Verifies:** stats-methodology.AC7.1 (account_entropy scope)

**Files:**
- Modify: `/Users/scarndp/dev/skywatch/skywatch-osprey/clickhouse-init/04-account-entropy.sql`

**Implementation:**

This file lives in the **skywatch-osprey** repo (sibling checkout at `/Users/scarndp/dev/skywatch/skywatch-osprey`). Two edits:

1. In the `CREATE TABLE IF NOT EXISTS default.account_entropy_results` statement, add four columns so fresh installs match the sidecar:
   - `hourly_entropy_norm Float64` (after `interval_entropy`)
   - `interval_entropy_norm Float64`
   - `interval_cv Float64` (after `stddev_interval_seconds`)
   - `cv_flag UInt8` (after `interval_flag`)

2. Append idempotent migration statements for existing deployments (mirroring the additive-column migration convention from the design's "Migration ordering" note), after the CREATE TABLE:

```sql
ALTER TABLE default.account_entropy_results ADD COLUMN IF NOT EXISTS hourly_entropy_norm Float64 AFTER interval_entropy;
ALTER TABLE default.account_entropy_results ADD COLUMN IF NOT EXISTS interval_entropy_norm Float64 AFTER hourly_entropy_norm;
ALTER TABLE default.account_entropy_results ADD COLUMN IF NOT EXISTS interval_cv Float64 AFTER stddev_interval_seconds;
ALTER TABLE default.account_entropy_results ADD COLUMN IF NOT EXISTS cv_flag UInt8 AFTER interval_flag;
```

Keep the existing 13 columns, engine, and `ORDER BY (run_timestamp, user_id)` untouched. The sidecar and this schema change deploy together (breaking-changes-allowed per the design's Definition of Done; the insert in Task 4 names all 17 columns explicitly, so the sidecar fails fast against an unmigrated table rather than writing misaligned data).

**Verification:**
Run: `grep -c 'ADD COLUMN IF NOT EXISTS' /Users/scarndp/dev/skywatch/skywatch-osprey/clickhouse-init/04-account-entropy.sql`
Expected: `4`. Also visually confirm the CREATE TABLE column order matches the Task 4 insert column list (name-for-name; ClickHouse inserts are by column name so order need not match, but names must).

**Commit** (in skywatch-osprey): `Add normalized entropy and CV columns to account_entropy_results`
(skywatch-osprey uses verb-first commit style without prefixes.)
<!-- END_TASK_5 -->
<!-- END_SUBCOMPONENT_B -->

<!-- START_TASK_6 -->
### Task 6: Documentation updates

**Verifies:** None (documentation; AC8.2 is finalized in Phase 7)

**Files:**
- Modify: `account_entropy/README.md`
- Modify: `account_entropy/CLAUDE.md`

**Implementation:**

Update both files to describe the shipped behaviour:
- Method: Shannon entropy in bits retained as context; scoring now uses Miller–Madow bias-corrected entropy normalized by `log2(min(N, bins))` to a 0–1 scale, plus an interval coefficient-of-variation signal.
- Flags: `hourly_flag` (`hourly_entropy_norm >= 0.85`), `interval_flag` (`interval_entropy_norm <= 0.53`), `cv_flag` (`interval_cv <= 0.5`), `is_bot_like = hourly_flag AND (interval_flag OR cv_flag)`.
- Env vars: replace the two old threshold vars with the three new ones (names/defaults from Task 3).
- Output columns: document the four new columns.

Keep each file's existing structure and tone; change only what is now inaccurate.

**Verification:**
Run: `grep -n 'ACCOUNT_ENTROPY_HOURLY_ENTROPY_THRESHOLD\|ACCOUNT_ENTROPY_INTERVAL_ENTROPY_THRESHOLD' account_entropy/README.md account_entropy/CLAUDE.md`
Expected: no matches (old env vars fully purged from docs).

**Commit:** `docs: update account_entropy README and CLAUDE.md for normalized entropy`
<!-- END_TASK_6 -->

<!-- START_TASK_7 -->
### Task 7: Full suite gate

**Verifies:** stats-methodology.AC9.1 (account_entropy scope)

**Files:** none (verification only)

**Step 1: Run the full test suite**

Run: `cd account_entropy && uv run pytest`
Expected: all tests pass, zero failures/errors.

**Step 2: Lint**

Run: `cd account_entropy && uv run ruff check src tests && uv run ruff format --check src tests`
Expected: clean. (Ruff is configured in `pyproject.toml`; single quotes, 120 cols.)

**Step 3: Confirm working tree is fully committed**

Run: `git status --short` in both repos.
Expected: clean tree in `osprey-sidecars`; the `skywatch-osprey` schema change from Task 5 committed in that repo.
<!-- END_TASK_7 -->
