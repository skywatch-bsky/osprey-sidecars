# Statistical Methodology Fixes — Phase 7: Documentation and Calibration Implementation Plan

**Goal:** Cross-repo documentation reflects the methods shipped in Phases 1–6, the missing quote_* sections exist, and a calibration playbook describes how to validate flag rates once deployed (validation itself is out of scope).

**Architecture:** Documentation only — no runtime code. Three deliverables: a rewritten `skywatch-osprey/docs/statistical-sidecars.md`, a new `osprey-sidecars/docs/calibration.md`, and an updated methods table in `osprey-sidecars/README.md`. Ends with a cross-repo consistency sweep for AC7.1.

**Tech Stack:** Markdown; ClickHouse SQL for the validation queries embedded in calibration.md.

**Scope:** Phase 7 of 7 from `docs/design-plans/2026-07-06-stats-methodology.md`. **Depends on Phases 1–6 being merged** — every section documents shipped behaviour, so writers must read the final merged source, not this plan, as ground truth for details like column names and defaults.

**Codebase verified:** 2026-07-06 via codebase-investigator agents (pre-Phase-1 state; re-verify against merged code when executing).

---

## Acceptance Criteria Coverage

This phase implements and tests:

### stats-methodology.AC7: Schemas
- **stats-methodology.AC7.1 Success:** All seven clickhouse-init files updated consistently with sidecar insert column lists (unit-tested per sidecar) — *this phase: final cross-file consistency sweep*

### stats-methodology.AC8: Documentation
- **stats-methodology.AC8.1 Success:** statistical-sidecars.md describes NB, beta-binomial, FDR, normalized entropy, and Newman weighting; includes quote_* sections
- **stats-methodology.AC8.2 Success:** Per-sidecar README/CLAUDE.md match shipped behaviour
- **stats-methodology.AC8.3 Success:** `docs/calibration.md` exists with per-sidecar validation queries, healthy ranges, and tuning levers

---

## Context from codebase verification

- `/Users/scarndp/dev/skywatch/skywatch-osprey/docs/statistical-sidecars.md` (547 lines): overview table lists all six sidecars (lines 1–16); detailed method sections exist for PDS Signup Anomaly (19–159), URL Overdispersion (162–235), Account Entropy (237–325), URL Co-Sharing (390–534); **quote_overdispersion and quote_cosharing have no method sections**; General Limitations at 536–547.
- `/Users/scarndp/dev/skywatch/osprey-sidecars/README.md`: overview table (lines 1–26) currently claims "Poisson statistics", "Poisson volume + density scoring", "Shannon entropy", "Leiden community detection"; env var reference at 49–60.
- `/Users/scarndp/dev/skywatch/osprey-sidecars/docs/calibration.md`: **does not exist**.
- Per-sidecar README/CLAUDE.md files were updated within Phases 1–6 (each phase's docs task); AC8.2 here is a verification sweep, not a rewrite.
- Line numbers above predate Phases 1–6; treat them as section locators, not exact anchors.

Method facts the docs must state (as shipped by Phases 1–6; verify against merged code):
- **Counts (signup/url/quote):** NB via MoM on (rolling median, φ·median) where φ = rolling variance/mean of a dense zero-filled window; Poisson fallback when φ ≤ 1 or variance unavailable; `P(X ≥ observed)` one-sided; hourly baselines partition by (entity, hour-of-day); `min_sharers` applies to scored rows only; population fallbacks are medians over today's scored entities without dispersion/rolling-mean qualifiers.
- **Density (url/quote):** beta-binomial `sf` with α = μM, β = (1−μ)M, M = μ(1−μ)/σ² − 1 from rolling density mean/variance (NULL-density zero days excluded); plain binomial fallback when the fit is invalid; explicit one-sided guard (at/below baseline → p = 1.0).
- **FDR:** BH per analysis cycle, per granularity (daily and hourly are separate families because each `score_rows` call is one family), per signal (volume and density adjusted separately); `is_anomaly = q_value < threshold` with surviving guards; threshold env vars keep their names but are FDR targets; BH on discrete p-values is conservative (documented, accepted).
- **Entropy:** Miller–Madow correction `H + (K_occupied − 1)/(2N ln 2)` bits, normalized by `log2(min(N, bins))`, clamped to [0, 1]; thresholds 0.85 (hourly ≥), 0.53 (interval ≤), CV ≤ 0.5; `is_bot_like = hourly_flag AND (interval_flag OR cv_flag)`; raw bit entropies retained as context.
- **Co-sharing:** `newman_weight = Σ 1/(k − 1)` per pair in the MV (k = per-URL/URI sharer count from the qualifying CTE, k ≥ 3); Leiden/CPM clusters on `newman_weight`; `min_edge_weight` filters raw `weight`; cluster `total_weight` remains the raw co-share sum; duplicate pairs aggregated before batch edge construction.
- **Accepted approximations** (from the design, to be restated in General Limitations): binomial/beta-binomial ignores the first-share-is-always-unique dependence; BH conservative on discrete p-values; entropy normalization treats bins as exchangeable.

---

<!-- START_SUBCOMPONENT_A (tasks 1-4: statistical-sidecars.md rewrite) -->
<!-- START_TASK_1 -->
### Task 1: statistical-sidecars.md — overview table and signup section

**Verifies:** stats-methodology.AC8.1 (partial)

**Files:**
- Modify: `/Users/scarndp/dev/skywatch/skywatch-osprey/docs/statistical-sidecars.md` (overview, lines ~1–16; PDS Signup Anomaly section, lines ~19–159)

**Implementation:**

1. Overview table: update each sidecar's one-line method — signup: "Negative binomial / Poisson on dense median baselines, BH-FDR"; url/quote overdispersion: "NB volume + beta-binomial density, BH-FDR per signal"; account_entropy: "Bias-corrected normalized entropy + interval CV"; cosharing pair: "Leiden (CPM) on Newman-weighted co-sharing graph".
2. Rewrite the signup section's model subsections against the merged Phase 1 code:
   - Model: NB via MoM (r = μ²/(σ²−μ), p = μ/σ² with μ = rolling median, σ² = φ·median), Poisson fallback conditions, one-sided survival p-value, worked example updated to the new maths.
   - Baselines: dense zero-filled calendar from each host's first appearance, `medianExact` centre, mean/variance for φ, hour-of-day-matched hourly windows (7 same-hour observations at default `baseline_days=7` — thin-window caveat with pointer to calibration.md), population medians without the old qualifiers.
   - FDR: BH per cycle per granularity; `q_value` column; `is_anomaly = q < target`; env var names unchanged, reinterpreted; population-source rows still compared against the stricter daily target.
   - Terminology: rename the "dispersion diagnostics" prose to **dispersion factor** (variance-to-mean ratio of the baseline window) and note the `dispersion_index` column name is historical — this closes the design's "overdispersion naming clarified in docs" item.
   - Output schema table: add `q_value`, correct semantics of `expected_lambda` (median-based).

**Verification:**
Run: `grep -n 'negative binomial\|q_value\|medianExact\|hour-of-day' /Users/scarndp/dev/skywatch/skywatch-osprey/docs/statistical-sidecars.md | head -20`
Expected: all four concepts present in the signup section.

**Commit:** one commit for Tasks 1–4 (in skywatch-osprey): `Rewrite statistical-sidecars.md for NB, beta-binomial, FDR, normalized entropy, Newman weighting`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: statistical-sidecars.md — url_overdispersion rewrite and new quote_overdispersion section

**Verifies:** stats-methodology.AC8.1 (partial)

**Files:**
- Modify: `/Users/scarndp/dev/skywatch/skywatch-osprey/docs/statistical-sidecars.md` (URL Overdispersion section, lines ~162–235; new quote_overdispersion section after it)

**Implementation:**

1. Rewrite URL Overdispersion: NB volume test (as Task 1's model text, entity = domain), beta-binomial density test (MoM fit, validity conditions, binomial fallback, explicit one-sided guard, worked example — reuse the Phase 2 hand example: μ=0.5, σ²=0.05 → α=β=2, P(X≥9 | n=10) = 186/1716 ≈ 0.108), dense baselines with `min_sharers` on scored rows only, NULL-density zero-days convention, hour-of-day matching, per-signal BH families and OR-logic on q-values, new output columns (`volume_q_value`, `density_q_value`, rolling-stat diagnostics), accepted first-share approximation.
2. Add a **quote_overdispersion** section (currently missing): mirror the URL section with entity = quoted-post AT-URI (`PostEmbedRecordUri`/`PostEmbedRecordWithMediaUri` coalescing), `quoted_author_did` extraction, no watchlist, output table `quote_overdispersion_results`, and the lifecycle note (quoted posts are short-lived entities, so population fallback dominates — expected behaviour).

**Verification:**
Run: `grep -n 'beta-binomial\|quote_overdispersion\|quoted_uri' /Users/scarndp/dev/skywatch/skywatch-osprey/docs/statistical-sidecars.md | head -20`
Expected: beta-binomial in both sections; a quote_overdispersion heading exists.

**Commit:** combined with Task 1's commit.
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: statistical-sidecars.md — account_entropy rewrite

**Verifies:** stats-methodology.AC8.1 (partial)

**Files:**
- Modify: `/Users/scarndp/dev/skywatch/skywatch-osprey/docs/statistical-sidecars.md` (Account Entropy section, lines ~237–325)

**Implementation:**

Rewrite against merged Phase 4 code: Miller–Madow bias correction (formula and why finite samples underestimate entropy), normalization by `log2(min(N, bins))` with the 10-posts-in-10-hours worked example (raw 3.32 bits could never cross the old 3.9-bit rule; normalized 1.0 crosses 0.85), interval CV signal, revised conjunction `is_bot_like = hourly_flag AND (interval_flag OR cv_flag)`, thresholds and their env vars, raw entropies retained as context, new output columns, exchangeable-bins approximation note in the edge-cases subsection.

**Verification:**
Run: `grep -n 'Miller\|normalized\|interval_cv\|cv_flag' /Users/scarndp/dev/skywatch/skywatch-osprey/docs/statistical-sidecars.md | head -20`
Expected: all present in the Account Entropy section.

**Commit:** combined with Task 1's commit.
<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: statistical-sidecars.md — url_cosharing rewrite, new quote_cosharing section, limitations

**Verifies:** stats-methodology.AC8.1 (completes it)

**Files:**
- Modify: `/Users/scarndp/dev/skywatch/skywatch-osprey/docs/statistical-sidecars.md` (URL Co-Sharing section, lines ~390–534; new quote_cosharing section; General Limitations, lines ~536–547)

**Implementation:**

1. URL Co-Sharing rewrite: dual edge weights (raw co-share count `weight` for `min_edge_weight` filtering and investigations; `newman_weight = Σ 1/(k_url − 1)` with k from the qualifying CTE, Newman 2001, viral-URL down-weighting rationale with a two-line worked example — niche URL k=3 contributes 0.5, viral URL k=500 contributes ~0.002), Leiden/CPM on `newman_weight`, `total_weight` semantics (still raw), duplicate-pair aggregation + batch construction hardening, pairs-table migration note (drop/recreate, 7-day TTL bounds loss).
2. **Jaccard-threshold tuning guidance** in the evolution-tracking subsection: default `*_JACCARD_THRESHOLD=0.5` is strict; community-evolution literature commonly uses ~0.3 for continuation matching; lowering it links more clusters day-over-day (fewer birth/death events, more continuations) at the cost of looser identity — tuning lever, cross-reference calibration.md.
3. Add a **quote_cosharing** section mirroring the URL one (entity = quoted URI, `shared_uris`, `quote_cosharing_*` tables).
4. General Limitations: append the design's accepted approximations (first-share dependence ignored by the density model; BH conservative on discrete p-values; entropy bins treated as exchangeable) and the CPM-resolution-scales-with-weights caveat.

**Verification:**
Run: `grep -n 'newman\|Jaccard\|quote_cosharing' /Users/scarndp/dev/skywatch/skywatch-osprey/docs/statistical-sidecars.md | head -20`
Expected: Newman weighting in both cosharing sections; Jaccard guidance present; quote_cosharing heading exists.

**Commit:** completes the Tasks 1–4 commit in skywatch-osprey.
<!-- END_TASK_4 -->
<!-- END_SUBCOMPONENT_A -->

<!-- START_TASK_5 -->
### Task 5: `docs/calibration.md` — post-deploy validation playbook

**Verifies:** stats-methodology.AC8.3

**Files:**
- Create: `osprey-sidecars/docs/calibration.md`

**Implementation:**

Create the calibration playbook. Structure (one section per sidecar plus a shared preamble); every query must be runnable ClickHouse SQL against the shipped schemas — copy exact table/column names from the merged code, not from memory:

1. **Preamble:** purpose (validate flag rates and baseline health after deploying the Phase 1–6 methodology; this playbook is executed post-deploy, it was explicitly out of scope to run it now); how to read q-values vs p-values; reminder that threshold env vars are FDR targets.
2. **Per count-test sidecar (signup_anomaly, url_overdispersion, quote_overdispersion):**
   - Validation queries: daily flag rate over time (`countIf(is_anomaly = 1) / count()` grouped by day and granularity); q-value distribution histogram; `baseline_source` split (entity vs population share); dispersion-factor distribution (how often the NB branch engages: share of rows with φ > 1); for url/quote, volume-vs-density attribution (`countIf(volume_q_value < target)` vs `countIf(density_q_value < target)`).
   - Healthy ranges: flagged share of scored rows in the low single-digit percent and, under the null, bounded by the FDR target among true nulls; population-source share falling over the first `baseline_days` after deploy; NB branch engaging on a meaningful fraction of rows (if ~0%, variance plumbing is broken; if ~100%, baselines are noisier than expected).
   - Tuning levers: FDR targets (env vars, names unchanged), `baseline_days` (the lever for thin hour-of-day windows: 7 same-hour observations for signup, 14 for url/quote at defaults), `min_sharers`, `cold_start_min_days`.
3. **account_entropy:** bot-rate over time; joint distribution of `hourly_entropy_norm` × `interval_entropy_norm` × `interval_cv`; share of flags attributable to `cv_flag` vs `interval_flag`. Healthy: `is_bot_like` rate stable and small; normalized entropies spanning (0, 1) rather than saturating. Levers: the three normalized thresholds and `min_posts`.
4. **Cosharing pair (url_cosharing, quote_cosharing):**
   - Validation queries: clusters per day, cluster-size distribution, `total_weight` vs member_count sanity, evolution-type mix (birth/continuation/merge/split/death rates).
   - **CPM resolution re-tuning procedure** (required by the design: Newman weights are ~an order of magnitude smaller than raw counts, and CPM resolution scales with edge weights): sweep `*_RESOLUTION` over a grid (e.g. 0.005, 0.01, 0.02, 0.05, 0.1) on one day's pairs, compare cluster count and size distribution against the pre-change baseline, pick the value that restores comparable granularity; document the chosen value.
   - Jaccard threshold guidance (0.5 default, literature ~0.3; effect on evolution-type mix).
   - Healthy: no giant component swallowing most accounts (resolution too low) and no all-singleton dust (too high); continuation share dominating evolution events once tuned.
5. **Escalation:** what to do when a range is violated (check dense-baseline queries with `EXPLAIN`, verify schema migration applied, compare q-value vs p-value flag counts).

**Verification:**
Run: `ls osprey-sidecars/docs/calibration.md && grep -c 'SELECT' osprey-sidecars/docs/calibration.md`
Expected: file exists; at least one validation query per sidecar (≥ 6 SELECTs).

**Commit:** `docs: add post-deploy calibration playbook`
<!-- END_TASK_5 -->

<!-- START_TASK_6 -->
### Task 6: Root README methods table

**Verifies:** stats-methodology.AC8.2 (root-level slice)

**Files:**
- Modify: `osprey-sidecars/README.md` (overview table, lines ~1–26; env var reference, lines ~49–60)

**Implementation:**

Update the methods column per sidecar (same one-liners as Task 1's overview table) and the env var reference: account_entropy's renamed threshold vars (`ACCOUNT_ENTROPY_HOURLY_NORM_THRESHOLD`, `ACCOUNT_ENTROPY_INTERVAL_NORM_THRESHOLD`, `ACCOUNT_ENTROPY_CV_THRESHOLD`); note that `*_P_THRESHOLD` vars are FDR targets. Add a pointer to `docs/calibration.md`.

**Verification:**
Run: `grep -n 'Poisson statistics' osprey-sidecars/README.md`
Expected: no bare "Poisson statistics" claims remain.

**Commit:** `docs: update root README methods table for new statistical methodology`
<!-- END_TASK_6 -->

<!-- START_TASK_7 -->
### Task 7: Cross-repo consistency sweep

**Verifies:** stats-methodology.AC7.1 (final check), stats-methodology.AC8.2 (verification sweep)

**Files:** none (verification only; fix anything found and fold fixes into the relevant docs commit)

**Step 1: Schema ↔ insert-list consistency (AC7.1)**

For each sidecar, compare the insert `column_names` list in `src/<sidecar>/db.py` against the corresponding `clickhouse-init` CREATE TABLE (02, 03, 04, 05, 06, 07; `01-init.sql` defines the shared source table and needs no change — confirm it was not touched):

Run, for each pair (example for signup):
```bash
grep -o "'[a-z_]*'" signup_anomaly/src/signup_anomaly/db.py | tr -d "'" | sort > /tmp/cols_sidecar.txt
grep -oE '^\s+[a-z_]+ ' /Users/scarndp/dev/skywatch/skywatch-osprey/clickhouse-init/02-signup-anomalies.sql | tr -d ' ' | sort > /tmp/cols_schema.txt
comm -23 /tmp/cols_sidecar.txt /tmp/cols_schema.txt
```
Expected: no insert column missing from its schema (the `comm` output contains no insert-list names). Do the equivalent check for all six sidecars; each sidecar's own `test_db.py` already unit-tests the list shape — this sweep catches cross-repo drift.

**Step 2: Stale-method sweep (AC8.2)**

Run:
```bash
grep -rn 'normal-approximation\|z-test\|3.9 bits\|3\.9-bit' \
    osprey-sidecars/*/README.md osprey-sidecars/*/CLAUDE.md osprey-sidecars/README.md \
    /Users/scarndp/dev/skywatch/skywatch-osprey/docs/statistical-sidecars.md
```
Expected: no matches describing replaced methods as current (historical "previously used X" notes are acceptable if clearly past-tense). Also confirm each per-sidecar README/CLAUDE.md mentions its phase's headline change (NB/FDR for 1–3, normalized entropy for 4, Newman for 5–6) — these were written in Phases 1–6; fix any that drifted.

**Step 3: Final commit check**

Run: `git status --short` in both repos; `git log --oneline -5` in each.
Expected: clean trees; documentation commits present.
<!-- END_TASK_7 -->
