# Post-Deploy Calibration Playbook

## Purpose

This playbook validates flag rates and baseline health immediately after deploying the statistical methodology (Phases 1–6) to production. It is not intended for execution during development; instead, run these queries in your ClickHouse environment once all sidecars have populated results for at least one full cycle.

**Key context:**
- Threshold env vars (e.g., `SIGNUP_ANOMALY_DAILY_P_THRESHOLD`) are **FDR targets**, not raw p-value cutoffs. See the "q-value vs p-value" section below.
- `is_anomaly` decisions use q-values (Benjamini–Hochberg adjusted p-values), which are monotone non-decreasing and conservative on discrete p-values.
- Baseline health depends on sufficient historical data (rolling medians over 7–14 days); new entities or sparse days will trigger cold-start fallback to population medians.

---

## Understanding q-values vs p-values

**p-value:** raw probability under the null hypothesis, computed per row.

**q-value:** adjusted p-value via Benjamini–Hochberg FDR control, computed per batch per granularity per signal. When you set `SIGNUP_ANOMALY_DAILY_P_THRESHOLD=0.05`, the code interprets this as "achieve 5% FDR among flagged rows" — i.e., q-value < 0.05 triggers anomaly. The env var name uses "p" for historical reasons but the semantics are now q-value thresholds.

**During validation:** Compare raw p-value flag counts against q-value flag counts. If they differ significantly, verify that your batch contained sufficient rows for BH adjustment to take effect.

---

## Signup Anomaly Detector

### Table: `pds_signup_anomalies`

Validates flag rates and baseline composition for PDS signup count anomalies.

#### Query 1: Daily flag rate over time

```sql
SELECT
    toDate(run_timestamp) as run_date,
    granularity,
    count() as total_rows,
    countIf(is_anomaly = 1) as flagged,
    ROUND(countIf(is_anomaly = 1) / count() * 100, 2) as flag_rate_pct
FROM pds_signup_anomalies
GROUP BY run_date, granularity
ORDER BY run_date DESC, granularity
LIMIT 100;
```

**Healthy ranges:**
- Flag rate in low single digits (1–5%) for the null population.
- Under FDR control with q-value threshold T, expect ~T × (true negatives / total rows) flags at steady state.
- Flag rate should stabilize after 3–5 days; high volatility in the first week suggests unstable baselines or threshold miscalibration.

#### Query 2: q-value distribution

```sql
SELECT
    granularity,
    countIf(q_value < 0.01) as flag_q_lt_001,
    countIf(q_value < 0.05) as flag_q_lt_005,
    countIf(q_value < 0.1) as flag_q_lt_01,
    countIf(q_value >= 0.1) as not_flagged,
    count() as total
FROM pds_signup_anomalies
WHERE toDate(run_timestamp) = today()
GROUP BY granularity
ORDER BY granularity;
```

**Healthy ranges:**
- q-values should span a wide range; saturation at 1.0 (all rows p=1.0) suggests baseline computation is failing (check EXPLAIN on baseline queries).
- If using FDR target = 0.05, the bulk of flagged rows should cluster below 0.05; rows above 0.05 are tolerated but should be rare.

#### Query 3: Baseline source distribution

```sql
SELECT
    toDate(run_timestamp) as run_date,
    granularity,
    baseline_source,
    count() as rows,
    ROUND(count() / sum(count()) OVER (PARTITION BY toDate(run_timestamp), granularity) * 100, 1) as pct_of_batch
FROM pds_signup_anomalies
WHERE toDate(run_timestamp) >= today() - 7
GROUP BY run_date, granularity, baseline_source
ORDER BY run_date DESC, granularity, pct_of_batch DESC;
```

**Healthy ranges:**
- Entity baseline should dominate (70–90%+) after the first 3–5 days.
- Population baseline share should fall as rolling medians stabilize; persistent >30% population rows after a week suggests entities have sparse, irregular history.
- Cold-start fallback rows (zero baseline) should be rare (<5%).

#### Query 4: Dispersion factor distribution (NB engagement rate)

```sql
SELECT
    toDate(run_timestamp) as run_date,
    granularity,
    countIf(dispersion_index > 1.0) as nb_branch,
    countIf(dispersion_index <= 1.0 OR dispersion_index IS NULL) as poisson_branch,
    count() as total,
    ROUND(countIf(dispersion_index > 1.0) / count() * 100, 1) as nb_pct
FROM pds_signup_anomalies
WHERE toDate(run_timestamp) >= today() - 7
GROUP BY run_date, granularity
ORDER BY run_date DESC, granularity;
```

**Healthy ranges:**
- NB branch (dispersion_index φ > 1.0) should engage 20–60% of rows.
- If NB engagement is ~0%, variance plumbing is broken — check that rolling variance is being computed correctly.
- If NB engagement is ~100%, baselines are noisier than expected (high variance relative to mean); either the baseline window is too short or the signal includes genuine spikes in non-anomalous accounts.

### Tuning Levers

- **FDR target (env var `SIGNUP_ANOMALY_DAILY_P_THRESHOLD`, `SIGNUP_ANOMALY_HOURLY_P_THRESHOLD`):** Controls q-value cutoff. Defaults 0.01 (daily) and 0.05 (hourly). Lower → stricter, fewer flags. Typical range: 0.01–0.1.
- **Baseline days (`SIGNUP_ANOMALY_BASELINE_DAYS`):** Default 7; controls rolling window for median/variance. Shorter windows → thinner hour-of-day bins (at default 7, hourly matching uses ~7 same-hour observations from the past 7 days). Increase to 14 for more stable baselines; decrease to 3 for faster response to emerging behaviour.
- **Min sharers:** Not applicable to signup (per-host counts, no sharer dimension).
- **Cold-start threshold (`SIGNUP_ANOMALY_COLD_START_MIN_DAYS`):** Default 3; when entity has <3 days of history, population median is used instead. Increase to 7 for more conservative fallback; decrease to 1 for faster entity bootstrap.

### Escalation

If flag rate is anomalously high (>10%) or distribution is severely skewed:
1. Check that dense-baseline queries in `queries.py` are returning expected medians by running them manually with `EXPLAIN QUERY PLAN` to verify index usage.
2. Compare raw p-value flag counts (from the same day, filtering `p_value < threshold`) against q-value flag counts. Large discrepancy suggests BH is over-correcting; check batch size and sorted p-value order.
3. Verify that `pds_signup_anomalies` table exists and has the `q_value` column (schema-migration check).

---

## URL Overdispersion Detector

### Table: `url_overdispersion_results`

Validates flag rates, baseline health, and volume-vs-density attribution for domain sharing anomalies.

#### Query 1: Daily flag rate over time

```sql
SELECT
    toDate(run_timestamp) as run_date,
    granularity,
    count() as total_rows,
    countIf(is_anomaly = 1) as flagged,
    ROUND(countIf(is_anomaly = 1) / count() * 100, 2) as flag_rate_pct
FROM url_overdispersion_results
WHERE toDate(run_timestamp) >= today() - 7
GROUP BY run_date, granularity
ORDER BY run_date DESC, granularity
LIMIT 50;
```

**Healthy ranges:**
- Flag rate 1–5% in steady state.
- Both daily and hourly granularities should be independently calibrated (separate FDR families); expect similar or slightly higher hourly rates due to tighter windows.

#### Query 2: q-value distribution (volume and density separate)

```sql
SELECT
    granularity,
    'volume' as signal,
    countIf(volume_q_value < 0.01) as q_lt_001,
    countIf(volume_q_value < 0.05) as q_lt_005,
    countIf(volume_q_value >= 0.05) as q_ge_005
FROM url_overdispersion_results
WHERE toDate(run_timestamp) = today()
GROUP BY granularity
UNION ALL
SELECT
    granularity,
    'density' as signal,
    countIf(density_q_value < 0.01) as q_lt_001,
    countIf(density_q_value < 0.05) as q_lt_005,
    countIf(density_q_value >= 0.05) as q_ge_005
FROM url_overdispersion_results
WHERE toDate(run_timestamp) = today()
GROUP BY granularity
ORDER BY granularity, signal;
```

**Healthy ranges:**
- Both volume and density q-values should have independent spread; if one is consistently all 1.0, that signal's baseline is missing (check population fallback logic).

#### Query 3: Baseline source distribution

```sql
SELECT
    toDate(run_timestamp) as run_date,
    granularity,
    baseline_source,
    count() as rows,
    ROUND(count() / sum(count()) OVER (PARTITION BY toDate(run_timestamp), granularity) * 100, 1) as pct_of_batch
FROM url_overdispersion_results
WHERE toDate(run_timestamp) >= today() - 7
GROUP BY run_date, granularity, baseline_source
ORDER BY run_date DESC, granularity, pct_of_batch DESC;
```

**Healthy ranges:**
- Entity baseline 70–90%+ after 3–5 days.
- Population baseline share should decline; persistent >30% after one week indicates many domains lack rolling density measurements.

#### Query 4: Dispersion factor distribution (NB engagement, volume test)

```sql
SELECT
    toDate(run_timestamp) as run_date,
    granularity,
    countIf(rolling_volume_variance > rolling_volume_median) as nb_branch,
    count() - countIf(rolling_volume_variance > rolling_volume_median) as poisson_branch,
    count() as total,
    ROUND(countIf(rolling_volume_variance > rolling_volume_median) / count() * 100, 1) as nb_pct
FROM url_overdispersion_results
WHERE toDate(run_timestamp) >= today() - 7 AND rolling_volume_variance IS NOT NULL
GROUP BY run_date, granularity
ORDER BY run_date DESC, granularity;
```

**Healthy ranges:**
- NB branch engagement 20–70% (higher than signup due to more variable domain sharing).
- If <10%, overdispersion signal is weak across domains; if >90%, many domains have atypical variance.

#### Query 5: Volume vs. density anomaly attribution

```sql
SELECT
    toDate(run_timestamp) as run_date,
    granularity,
    countIf(volume_q_value < 0.05) as volume_flagged,
    countIf(density_q_value < 0.05) as density_flagged,
    countIf(is_anomaly = 1) as total_flagged,
    count() as total,
    ROUND(countIf(volume_q_value < 0.05) / countIf(is_anomaly = 1) * 100, 1) as pct_volume_in_flags
FROM url_overdispersion_results
WHERE toDate(run_timestamp) >= today() - 7 AND is_anomaly = 1
GROUP BY run_date, granularity
ORDER BY run_date DESC, granularity;
```

**Healthy ranges:**
- Both volume and density should contribute to flagged anomalies; neither should completely dominate.
- If volume dominates >90%, density baseline may be miscalibrated (low variance → too many flags from small concentration swings).

### Tuning Levers

- **FDR targets (`URL_OVERDISPERSION_VOLUME_P_THRESHOLD`, `URL_OVERDISPERSION_DENSITY_P_THRESHOLD`):** Defaults 0.05. BH-FDR at 0.05 allows ~5% false discovery rate, which should bring flag rates into the 1–5% target range. The previous default of 0.01 was too conservative, producing 0.2–0.3% flag rates. Independent control over volume and density strictness.
- **Baseline days (`URL_OVERDISPERSION_BASELINE_DAYS`):** Default 14 for URL (longer than signup due to sparser domain-specific events). Increase to 21 for volatile domains; decrease to 7 for faster response.
- **Min sharers (`URL_OVERDISPERSION_MIN_SHARERS`):** Default 3; filters scored rows (final WHERE clause only, not baseline construction). Note: the current SQL restricts the domain population to domains meeting current-bucket `min_sharers` before building baselines. Lowering from 3 to 2 would broaden coverage but may introduce noisier baselines. If flag rates remain below target after the FDR threshold change, consider reducing `min_sharers` as a secondary lever.
- **Cold-start threshold (`URL_OVERDISPERSION_COLD_START_MIN_DAYS`):** Default 3.

### Escalation

1. Check ClickHouse baseline queries with `EXPLAIN` for index usage and scan volume.
2. If volume and density q-values are both all 1.0: rolling statistics not computed. Verify that `rolling_volume_median`, `rolling_volume_variance`, `rolling_density_mean`, `rolling_density_variance` are populated.
3. If density flags are absent but volume spikes occur: re-check density baseline calculation (mean and variance of sharer density per domain per day).

---

## Quote Post Overdispersion Detector

### Table: `quote_overdispersion_results`

Validates flag rates and baseline health for quoted-post concentration anomalies.

#### Query 1: Daily flag rate over time

```sql
SELECT
    toDate(run_timestamp) as run_date,
    granularity,
    count() as total_rows,
    countIf(is_anomaly = 1) as flagged,
    ROUND(countIf(is_anomaly = 1) / count() * 100, 2) as flag_rate_pct
FROM quote_overdispersion_results
WHERE toDate(run_timestamp) >= today() - 7
GROUP BY run_date, granularity
ORDER BY run_date DESC, granularity
LIMIT 50;
```

**Healthy ranges:**
- Flag rate 1–5%, similar to URL overdispersion.
- Quote posts have shorter lifespans than static URLs, so population fallback is expected to dominate even after several days (design expectation).

#### Query 2: q-value distribution

```sql
SELECT
    granularity,
    'volume' as signal,
    countIf(volume_q_value < 0.01) as q_lt_001,
    countIf(volume_q_value < 0.05) as q_lt_005,
    countIf(volume_q_value >= 0.05) as q_ge_005
FROM quote_overdispersion_results
WHERE toDate(run_timestamp) = today()
GROUP BY granularity
UNION ALL
SELECT
    granularity,
    'density' as signal,
    countIf(density_q_value < 0.01) as q_lt_001,
    countIf(density_q_value < 0.05) as q_lt_005,
    countIf(density_q_value >= 0.05) as q_ge_005
FROM quote_overdispersion_results
WHERE toDate(run_timestamp) = today()
GROUP BY granularity
ORDER BY granularity, signal;
```

**Healthy ranges:**
- Similar to URL overdispersion; quoted URIs have shorter observation windows, so cluster dynamics differ.

#### Query 3: Baseline source and short-lived entity effect

```sql
SELECT
    toDate(run_timestamp) as run_date,
    granularity,
    baseline_source,
    count() as rows,
    ROUND(count() / sum(count()) OVER (PARTITION BY toDate(run_timestamp), granularity) * 100, 1) as pct_of_batch
FROM quote_overdispersion_results
WHERE toDate(run_timestamp) >= today() - 7
GROUP BY run_date, granularity, baseline_source
ORDER BY run_date DESC, granularity, pct_of_batch DESC;
```

**Healthy ranges:**
- Population baseline may be 50–70%+ even after multiple days (expected for short-lived entities). This is normal.
- Quote overdispersion uses shorter baseline windows than URL overdispersion (7-day lookback, 1-day cold start) because quoted posts are short-lived. Entity baseline rates should be higher than with the previous 14-day/3-day defaults, but population baseline dominance remains expected.
- If entity baseline approaches 70%+ after 3 days, quoted posts are being tracked longer than design expectation; verify that the age-out logic in the query is working.

#### Query 4: Dispersion factor distribution

```sql
SELECT
    toDate(run_timestamp) as run_date,
    granularity,
    countIf(rolling_volume_variance > rolling_volume_median) as nb_branch,
    count() - countIf(rolling_volume_variance > rolling_volume_median) as poisson_branch,
    count() as total,
    ROUND(countIf(rolling_volume_variance > rolling_volume_median) / count() * 100, 1) as nb_pct
FROM quote_overdispersion_results
WHERE toDate(run_timestamp) >= today() - 7 AND rolling_volume_variance IS NOT NULL
GROUP BY run_date, granularity
ORDER BY run_date DESC, granularity;
```

**Healthy ranges:**
- NB branch engagement 10–50% (lower than URL because fewer quoted posts reach high sharing counts).

### Tuning Levers

- **FDR targets (`QUOTE_OVERDISPERSION_VOLUME_P_THRESHOLD`, `QUOTE_OVERDISPERSION_DENSITY_P_THRESHOLD`):** Defaults 0.01.
- **Baseline days (`QUOTE_OVERDISPERSION_BASELINE_DAYS`):** Default 7. Shorter than URL overdispersion (14) because quoted posts are short-lived — most don't survive 3 days. A 7-day lookback allows entity baselines for any URI quoted on consecutive days.
- **Min sharers (`QUOTE_OVERDISPERSION_MIN_SHARERS`):** Default 3 (same rationale as URL).
- **Cold-start threshold (`QUOTE_OVERDISPERSION_COLD_START_MIN_DAYS`):** Default 1. Lower than URL overdispersion (3) because a 1-day baseline is weak but better than population fallback for volatility reduction. A quoted URI quoted on two consecutive days qualifies for an entity baseline.

### Escalation

Same as URL overdispersion. Additionally, if population fallback dominates:
1. Verify that quoted posts are being tracked correctly (check `PostEmbedRecordUri` / `PostEmbedRecordWithMediaUri` coalescing in queries).
2. Quoted posts typically have short observation windows; high population baseline is expected and not a failure mode.

---

## Account Entropy Detector

### Table: `account_entropy_results`

Validates bot-like flag rates and entropy signal distributions.

#### Query 1: Bot-like flag rate over time

```sql
SELECT
    toDate(run_timestamp) as run_date,
    count() as total_accounts,
    countIf(is_bot_like = 1) as bot_flagged,
    ROUND(countIf(is_bot_like = 1) / count() * 100, 2) as bot_rate_pct
FROM account_entropy_results
WHERE toDate(run_timestamp) >= today() - 7
GROUP BY run_date
ORDER BY run_date DESC
LIMIT 30;
```

**Healthy ranges:**
- Bot-like rate 0.5–2% for a healthy Bluesky population.
- Rate should be stable across days; sudden spikes suggest a cohort of new accounts or a legitimate automation wave.

#### Query 2: Entropy signal joint distribution

```sql
SELECT
    COUNT() as count,
    ROUND(hourly_entropy_norm, 1) as hourly_norm,
    ROUND(interval_entropy_norm, 1) as interval_norm,
    ROUND(interval_cv, 1) as cv_bucket
FROM account_entropy_results
WHERE toDate(run_timestamp) = today()
GROUP BY hourly_norm, interval_norm, cv_bucket
ORDER BY count DESC
LIMIT 20;
```

**Healthy ranges:**
- Normalized entropies should span much of [0, 1] (diverse posting patterns across accounts).
- CV should range from 0.0 (metronomic) to >1.0 (highly variable inter-post intervals).
- Saturation at extremes (many accounts at 0.0 or 1.0 on multiple axes) suggests thresholds may be miscalibrated.

#### Query 3: Flag attribution (which signal triggers bot-like)

```sql
SELECT
    countIf(is_bot_like = 1 AND hourly_flag = 1 AND interval_flag = 0 AND cv_flag = 0) AS hourly_only,
    countIf(is_bot_like = 1 AND hourly_flag = 1 AND interval_flag = 1 AND cv_flag = 0) AS interval_only,
    countIf(is_bot_like = 1 AND hourly_flag = 1 AND interval_flag = 0 AND cv_flag = 1) AS cv_only,
    countIf(is_bot_like = 1 AND hourly_flag = 1 AND interval_flag = 1 AND cv_flag = 1) AS both_interval_and_cv,
    countIf(is_bot_like = 1) AS total_bot_like,
    countIf(hourly_flag = 1) AS hourly_flagged_total
FROM account_entropy_results
WHERE toDate(run_timestamp) = today();
```

**Healthy ranges:**
- Bot-like decisions should be distributed across the secondary signals (interval and CV); if one accounts for >80% of flags, that threshold may be too loose.
- The conjunction rule (hourly_flag AND (interval_flag OR cv_flag)) requires hourly entropy to be high (disordered posting) and *either* low inter-post randomness or metronomic CV to trigger bot-like. This filters out seasonal peak posters.
- The `hourly_only` category (interval_flag=0 AND cv_flag=0) is structurally zero for bot-like rows because `is_bot_like` requires at least one of interval_flag or cv_flag. It is included for completeness of the partition; the four categories are mutually exclusive and exhaustive over bot-like rows.
- The `hourly_flagged_total` column shows how many accounts pass the hourly gate regardless of bot-like status. A large gap between `hourly_flagged_total` and `total_bot_like` is expected (most high-entropy accounts are human). If `hourly_flagged_total` itself is very high (>10% of scored accounts), the hourly threshold (0.85) may be too permissive for low-N accounts where `log2(min(N, 24))` normalization inflates scores.

**Note on attribution:** The predicates above are mutually exclusive by construction — each bot-like row falls into exactly one category defined by the (interval_flag, cv_flag) pair. Previous versions of this query used `is_bot_like = 1 AND hourly_flag = 1` for the `hourly_only` category, which was tautological (is_bot_like already implies hourly_flag=1) and counted all bot-like rows, not just those driven by hourly entropy alone. Always use the full mutually exclusive predicates when decomposing flag attribution.

### Tuning Levers

- **Hourly entropy threshold (`ACCOUNT_ENTROPY_HOURLY_NORM_THRESHOLD`):** Default 0.85 (normalized). Raised to 0.90 to accept more variance; lowered to 0.75 to flag more posting-pattern disorder.
- **Interval entropy threshold (`ACCOUNT_ENTROPY_INTERVAL_NORM_THRESHOLD`):** Default 0.53 (normalized). Lower values (e.g., 0.40) flag more metronomic posting timing.
- **Coefficient-of-variation threshold (`ACCOUNT_ENTROPY_CV_THRESHOLD`):** Default 0.5. Raised to 0.7 for more regular human patterns; lowered to 0.3 for stricter regularity filtering.
- **Min posts (`ACCOUNT_ENTROPY_MIN_POSTS`):** Default 10; accounts with <10 posts are not scored. Increase to 50 for more stable estimates; decrease to 5 for broader coverage at lower confidence.

### Escalation

If bot-like rate is unexpectedly high (>5%):
1. Decompose by signal (Query 3) to identify the dominant driver. Ensure the attribution query uses mutually exclusive predicates (the four categories must partition all bot-like rows).
2. Check that entropies are normalized correctly: raw `hourly_entropy` should be in bits (0–log₂24 ≈ 4.58), normalized should be [0, 1].
3. Verify Miller–Madow correction is applied: `hourly_entropy_norm` should incorporate `(K_occupied − 1) / (2 * N * ln 2)` bias term (where N is post count).
4. If attribution shows a single signal dominating, investigate whether the hourly threshold (0.85) is too permissive for low-N accounts where `log2(min(N, 24))` normalization inflates scores. Threshold tuning should follow accurate attribution data from the fixed Query 3, not precede it.

---

## URL Co-Sharing Clusters

### Table: `url_cosharing_clusters` and `url_cosharing_runs`

Validates cluster granularity, network health, and dismantling pipeline execution via run metadata.

#### Query 1: Cluster count and size distribution

```sql
SELECT
    run_date,
    count() as cluster_count,
    countIf(member_count >= 2 AND member_count <= 5) as tiny_pct,
    countIf(member_count >= 6 AND member_count <= 20) as small_pct,
    countIf(member_count >= 21 AND member_count <= 100) as medium_pct,
    countIf(member_count >= 101) as large_pct,
    MIN(member_count) as min_size,
    MAX(member_count) as max_size,
    ROUND(AVG(member_count), 1) as avg_size
FROM url_cosharing_clusters
WHERE run_date >= today() - 7
GROUP BY run_date
ORDER BY run_date DESC;
```

**Healthy ranges:**
- Tiny/small clusters (2–20 members) should dominate (60–80% of clusters).
- Density-based dismantling (precision-first pipeline) flags far fewer accounts than Leiden-over-everything: expect smaller, more focused clusters and lower overall cluster counts compared to prior methodology.
- Large clusters (>100 members) should be rare (<5%); if frequent, CPM resolution is too low.
- Avg size 3–10 accounts per cluster indicates appropriate granularity for coordinated behaviour.

#### Query 2: Total weight sanity (raw co-share magnitude per cluster)

```sql
SELECT
    run_date,
    ROUND(AVG(total_weight), 1) as avg_total_weight,
    ROUND(quantile(0.5)(total_weight), 1) as p50_total_weight,
    ROUND(quantile(0.95)(total_weight), 1) as p95_total_weight,
    MAX(total_weight) as max_total_weight
FROM url_cosharing_clusters
WHERE run_date >= today() - 7
GROUP BY run_date
ORDER BY run_date DESC;
```

**Healthy ranges:**
- Avg total_weight (Σ over cluster URLs of C(members, 2)) 2–10 co-shares.
- P95 10–50 co-shares; extreme clusters indicate strong coordination.
- If avg < 2, clusters are too granular; if avg > 50, resolution is too low or thresholds need tightening.

#### Query 3: Evolution-type mix (birth/continuation/merge/split/death)

```sql
SELECT
    run_date,
    evolution_type,
    count() as event_count,
    ROUND(count() / sum(count()) OVER (PARTITION BY run_date) * 100, 1) as pct_of_day
FROM url_cosharing_clusters
WHERE run_date >= today() - 7
GROUP BY run_date, evolution_type
ORDER BY run_date DESC, pct_of_day DESC;
```

**Healthy ranges:**
- Continuation should dominate (60–80%) once clusters have stabilized (after day 2).
- With a 7-day rolling window advancing daily (6/7 data overlap), evolution is heavily skewed toward continuation. If continuation rate drops below 50%, investigate whether the Jaccard threshold needs retuning (see "Jaccard threshold guidance" below).
- Birth rate 10–20% (new clusters emerging daily).
- Merge/split/death rates low (5–10% combined).

#### Query 4: Run metadata health (dismantling execution)

```sql
SELECT
    run_date,
    knee_found,
    guardrail_triggered,
    accounts_raw,
    accounts_eligible,
    urls_eligible,
    graph_edges,
    edge_quantile,
    centrality_quantile,
    ROUND(min_component_density, 3) as min_density,
    flagged_accounts,
    ROUND(flagged_accounts / nullIf(accounts_eligible, 0) * 100, 2) as flagged_pct,
    cluster_count
FROM url_cosharing_runs
WHERE run_date >= today() - 14
ORDER BY run_date DESC;
```

**Healthy ranges:**
- **Denominator caveat:** the paper's observed 0.4–1.5% coordinated-account band is relative to full dataset populations (6k–178k accounts), which corresponds to our `accounts_raw`, not `accounts_eligible`. `flagged_accounts / accounts_raw` in the 0.05–0.5% range is consistent with the paper; `flagged_pct` against `accounts_eligible` runs much higher (2–5%) because the ≥10-unique-URLs eligibility filter shrinks the denominator ~50×.
- Absolute `flagged_accounts` should sit in the low hundreds (paper observed 25–764). Investigate sustained values near the `MAX_FLAGGED_ACCOUNTS` ceiling (default 750).
- `knee_found = false` days are **correct behaviour**: they indicate no sharp phase transition in the density grid (no obvious knee = low confidence in a single optimal point). Alert only if paired with consecutive false runs and rising `accounts_eligible`.
- Frequent `guardrail_triggered = true` paired with rising `flagged_pct` signals the knee rule is being overridden: inspect the density surface (see "Density-Surface Calibration" below) before loosening `MAX_FLAGGED_FRACTION`.

### Density-Surface Calibration

The dismantling surface is a grid of (edge_quantile, centrality_quantile) pairs, each point yielding a component density and surviving node count. Operators inspect this surface offline to calibrate `DENSITY_FLOOR` and validate `MAX_FLAGGED_FRACTION`.

**How to read the surface:** Run `uv run python -m url_cosharing.calibrate` to dump the per-cell grid as TSV, including `min_component_density`, `surviving_nodes`, `surviving_edges`. The header row names the columns.

**Phase transitions:** A healthy surface shows a sharp density jump (phase transition) between two adjacent cells, indicating a transition from sparse to dense regions. The post-jump plateau is the natural equilibrium for the day's data.

**Tuning `DENSITY_FLOOR`:**
- Set it just below the post-jump plateau (e.g., if density jumps from 0.3 to 0.7, set `DENSITY_FLOOR` to 0.65).
- Too high: results in permanent `knee_found = false` (no cell survives the threshold), which is safe but provides no flagging.
- Too low: the knee rule flags weak phase transitions, producing noise and raising `flagged_pct`.
- Adjust incrementally (±0.05 steps) over 3–5 days and re-run Query 4 above to verify the effect.

**Validating the guardrail (`MAX_FLAGGED_FRACTION` + `MAX_FLAGGED_ACCOUNTS`):**
- The effective ceiling is `min(MAX_FLAGGED_FRACTION × accounts_eligible, MAX_FLAGGED_ACCOUNTS)`. The guardrail is an operational safety bound of this implementation — the paper prescribes no size cap; its threshold selection is the knee rule alone.
- If `flagged_accounts` regularly hugs the ceiling AND post-hoc triage shows false positives, the dismantling is over-flagging: raise `DENSITY_FLOOR` (quality lever) before touching the caps.
- If `guardrail_triggered = true` recurs on candidates with density ≥ 0.9 sitting just above the ceiling, the cap is clipping plausible cores: raise `MAX_FLAGGED_ACCOUNTS` in small steps and re-check precision.
- Sanity band: `flagged_accounts / accounts_raw` between 0.05% and 0.5% is comfortably below the paper's observed 0.4–1.5%.

### CPM Resolution Re-Tuning Procedure

Leiden CPM now runs on the **dismantled core only** with cosine-similarity edge weights (not Newman weights; weights are ≤ 1 per edge). The resolution sweep procedure remains the same but applies to a much smaller subgraph and uses the calibrate module instead of raw pairs.

**Step 1: Baseline**
Run the density surface dump (`uv run python -m url_cosharing.calibrate`) and record cluster count and size distribution from the selected cell (Query 1 above).

**Step 2: Sweep**
Modify `URL_COSHARING_RESOLUTION` and re-run the calibrate module with resolution parameters 0.01, 0.02, 0.05, 0.1, 0.2. For each, check Query 1 metrics.

**Step 3: Compare**
Pick the resolution that yields cluster metrics closest to baseline (e.g., ≤20% change in cluster count). Update `URL_COSHARING_RESOLUTION`.

**Step 4: Monitor**
After applying the change, re-run Query 1 for 3 days. Verify evolution-type mix remains continuation-dominant.

### Tuning Levers

- **Density Floor (`URL_COSHARING_DENSITY_FLOOR`):** Default 0.5. Lower to flag more weak knees; raise to reduce noise. See "Density-Surface Calibration" above.
- **Max Flagged Fraction (`URL_COSHARING_MAX_FLAGGED_FRACTION`):** Default 0.05 (5% of eligible accounts; raised from 0.02 on 2026-07-07). One half of the guardrail ceiling `min(fraction × eligible, max_flagged_accounts)`; its remaining job is degenerate days when the eligible graph itself is tiny and flagging most of it would be implausible.
- **Max Flagged Accounts (`URL_COSHARING_MAX_FLAGGED_ACCOUNTS`):** Default 750. Absolute half of the guardrail ceiling, anchored to Cinus et al.'s observed core sizes (25–764 accounts across networks of 6k–178k). Observed coordinated cores are roughly constant in absolute size, so this bound — not the fraction — should govern normal days. Calibrate down only on precision evidence from post-hoc cluster triage (but fix false positives by raising `DENSITY_FLOOR` first); calibrate up only if runs repeatedly show `guardrail_triggered = true` on dense (≥0.9) candidates just above the cap.
- **Edge Epsilon (`URL_COSHARING_EDGE_EPSILON`):** Default 0.05. Similarity threshold for including edges; lower includes more weak ties.
- **Edge Quantile Grid (`URL_COSHARING_EDGE_QUANTILE_GRID`):** Default `0.50,0.60,0.70,0.80,0.90,0.95,0.99`. Coarser grid (fewer points) speeds computation; finer grid refines knee detection.
- **Centrality Quantile Grid (`URL_COSHARING_CENTRALITY_QUANTILE_GRID`):** Default `0.50,0.60,0.70,0.80,0.90,0.95,0.99`. Controls dismantling grid density; same semantics as edge quantile grid.
- **Min Unique URLs (`URL_COSHARING_MIN_UNIQUE_URLS`):** Default 10. Accounts sharing fewer URLs are excluded; raise to focus on heavy sharers.
- **Min URL Sharers (`URL_COSHARING_MIN_URL_SHARERS`):** Default 5. URLs shared by fewer accounts are excluded; raise to filter niche URLs.
- **Max URL DF Fraction (`URL_COSHARING_MAX_URL_DF_FRACTION`):** Default 0.90. Excludes URLs shared by more than this fraction of accounts (sklearn max_df semantics; a rarely-binding safety valve — viral downweighting is handled by TF-IDF). Lower to be more aggressive on viral filtering.
- **Window Days (`URL_COSHARING_WINDOW_DAYS`):** Default 7. Historical window for URL shares; raise to include older data (smoother, less volatile).
- **Resolution (`URL_COSHARING_RESOLUTION`):** Default 0.05. CPM parameter; see re-tuning procedure above.
- **Jaccard Threshold (`URL_COSHARING_JACCARD_THRESHOLD`):** Default 0.5 (strict). See "Jaccard threshold guidance" below.

### Jaccard Threshold Guidance

The Jaccard similarity threshold controls whether a cluster on day N and a cluster on day N+1 are considered the "same cluster" (continuation) or distinct (split/merge/birth/death).

- **Default 0.5 (strict):** Clusters must share ≥50% of members to be linked. Best for detecting rapid composition changes.
- **~0.3 (loose):** Clusters must share ≥30% of members. Better for tracking long-lived communities with gradual member churn. Produces more continuations, fewer evolution events.

**Effect:** Lowering threshold increases continuation rate and decreases birth/death/merge/split rates. With the 7-day rolling window (6/7 daily overlap), evolution naturally skews toward continuation; adjust threshold if the observed balance seems off.

---

## Quote Post Co-Sharing Clusters

### Table: `quote_cosharing_clusters`

Validates cluster granularity and evolution for quote-post coordination.

#### Query 1: Cluster count and size distribution

```sql
SELECT
    run_date,
    count() as cluster_count,
    countIf(member_count >= 2 AND member_count <= 5) as tiny_pct,
    countIf(member_count >= 6 AND member_count <= 20) as small_pct,
    countIf(member_count >= 21 AND member_count <= 100) as medium_pct,
    countIf(member_count >= 101) as large_pct,
    MIN(member_count) as min_size,
    MAX(member_count) as max_size,
    ROUND(AVG(member_count), 1) as avg_size
FROM quote_cosharing_clusters
WHERE run_date >= today() - 7
GROUP BY run_date
ORDER BY run_date DESC;
```

**Healthy ranges:**
- Identical to URL cosharing; quoted URIs are subject to the same viral-URL effects.
- If cluster sizes are smaller than URL clusters, quoted posts may be less coordinated or more short-lived (both reasonable).

#### Query 2: Total weight sanity

```sql
SELECT
    run_date,
    ROUND(AVG(total_weight), 1) as avg_total_weight,
    ROUND(quantile(0.5)(total_weight), 1) as p50_total_weight,
    ROUND(quantile(0.95)(total_weight), 1) as p95_total_weight,
    MAX(total_weight) as max_total_weight
FROM quote_cosharing_clusters
WHERE run_date >= today() - 7
GROUP BY run_date
ORDER BY run_date DESC;
```

**Healthy ranges:**
- Avg total_weight 1–5 (quoted URIs are more fragmented than URLs).
- P95 5–20 co-quotes; very high weights suggest coordinated quote-bombing campaigns.

#### Query 3: Evolution-type mix

```sql
SELECT
    run_date,
    evolution_type,
    count() as event_count,
    ROUND(count() / sum(count()) OVER (PARTITION BY run_date) * 100, 1) as pct_of_day
FROM quote_cosharing_clusters
WHERE run_date >= today() - 7
GROUP BY run_date, evolution_type
ORDER BY run_date DESC, pct_of_day DESC;
```

**Healthy ranges:**
- Continuation 50–70% (lower than URL because quoted posts are short-lived).
- Birth rate 20–40% (new quote targets emerging daily).
- Merge/split/death rates moderate (10–20% combined).

### CPM Resolution Re-Tuning Procedure

Same as URL cosharing (see above). Run the procedure independently for `QUOTE_COSHARING_RESOLUTION`.

### Tuning Levers

- **Resolution (`QUOTE_COSHARING_RESOLUTION`):** Default 0.05.
- **Min edge weight (`QUOTE_COSHARING_MIN_EDGE_WEIGHT`):** Default 2.
- **Jaccard threshold (`QUOTE_COSHARING_JACCARD_THRESHOLD`):** Default 0.5; see guidance above.

---

## Escalation Procedures

### Symptom: All flag rates are zero

**Probable cause:** Baseline queries failed or returned no data.

**Steps:**
1. Manually run the baseline generation queries (see src/<sidecar>/queries.py).
2. Check that `osprey_execution_results` table exists and is populated.
3. Run `EXPLAIN QUERY PLAN` on a baseline query to verify index usage and confirm no timeouts.
4. Check ClickHouse logs for query errors or memory limits.

### Symptom: Flag rate is extremely high (>20%)

**Probable cause:** Threshold is too loose, or baseline computation is broken (returning all nulls → all fallback to population → very broad flag distribution).

**Steps:**
1. Check baseline_source split (Query 3 for each sidecar): if >50% is "zero_baseline", baseline computation is failing.
2. Verify that rolling medians and variances are being computed: run `rolling_volume_median IS NOT NULL` filter and count.
3. Reduce FDR threshold by half (e.g., 0.05 → 0.025) and re-run next cycle.

### Symptom: q-value saturation (all q-values = 1.0)

**Probable cause:** P-values are all 1.0 (baseline is zero or entity not scored).

**Steps:**
1. Check that expected_lambda (for counts) or expected_density_lambda (for density) are non-zero.
2. Verify `rolling_median` and rolling baseline queries return expected values.
3. If population fallback is dominating, baseline_days may be too short. Increase to 14–21 and re-run.

### Symptom: Entropy signals are all 0.0 or all 1.0

**Probable cause:** Normalization is broken or all accounts have identical patterns.

**Steps:**
1. Verify Miller–Madow bias correction is applied: `hourly_entropy` should be raw bits (0–4.58), then normalized to [0, 1].
2. Check that normalized entropies span [0, 1]; if all accounts cluster at one extreme, thresholds may be set incorrectly.
3. Run the joint distribution query (Query 2 for account_entropy) to inspect real distributions.

### Symptom: Clusters are all singletons or one giant component

**Probable cause:** CPM resolution is too high (singletons) or too low (giant component).

**Steps:**
1. Follow the CPM resolution re-tuning procedure (above).
2. Verify that Newman weights are being computed correctly; raw weight should be ≥ Newman weight.
3. Check min_edge_weight: if too high, all pairs are filtered; if too low, noise pairs enter.

---

## Summary Checklist

- [ ] All tables (`pds_signup_anomalies`, `url_overdispersion_results`, `quote_overdispersion_results`, `account_entropy_results`, `url_cosharing_clusters`, `url_cosharing_membership`, `quote_cosharing_clusters`, `quote_cosharing_membership`) exist and have recent data.
- [ ] Flag rates (count-test sidecars) are in the 1–5% range.
- [ ] q-values are distributed across the full [0, 1] range, not saturated.
- [ ] Baseline source split shows entity baseline dominating (>70%) after 3 days.
- [ ] NB branch engagement is 20–70% for count tests.
- [ ] Bot-like rate (account_entropy) is 0.5–2%.
- [ ] Cluster sizes are appropriate (no giant component, no all-singleton dust).
- [ ] Evolution-type mix is continuation-dominant (>50%) after day 2.
- [ ] If resolution tuning is needed, CPM sweep procedure has been executed and documented.

---

## References

- Phase 1–6 implementation: `/Users/scarndp/dev/skywatch/osprey-sidecars/src`
- Schema definitions: `/Users/scarndp/dev/skywatch/skywatch-osprey/clickhouse-init/`
- Design documentation: `/Users/scarndp/dev/skywatch/skywatch-osprey/docs/statistical-sidecars.md`
