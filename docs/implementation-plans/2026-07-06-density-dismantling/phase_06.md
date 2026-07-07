# Density-Dismantling Implementation Plan — Phase 6: Documentation and calibration

> **Superseded (2026-07-07, issue #3):** the URL df ceiling described in this document as a percentile of the df distribution (`max_url_df_pctl` / `quantile(max_url_df_pctl)(df)`) was a mis-transcription of Cinus et al.'s published code and is degenerate on production data. The implemented contract is `max_url_df_fraction` (`URL_COSHARING_MAX_URL_DF_FRACTION`): eligible URLs satisfy `df <= max_url_df_fraction * distinct_account_count` (sklearn `max_df` semantics), applied in SQL only. Do not reintroduce percentile/quantile ceiling logic from this document.

**Goal:** Operators can understand the new methodology, calibrate grids/guardrails against production data, and dump the density surface for offline inspection.

**Architecture:** Documentation updates (`url_cosharing/CLAUDE.md`, `url_cosharing/README.md`, `docs/calibration.md`) plus one thin Imperative Shell module `calibrate.py` invoked as `uv run python -m url_cosharing.calibrate` (there is **no** scripts/ directory convention in this repo — module invocation matches the established `uv run python -m <sidecar>.main` pattern). Surface formatting is a pure function so the shell stays trivially thin.

**Tech Stack:** Markdown docs; Python (reuses Phase 2–4 Core functions and `CosharingDb`).

**Scope:** Phase 6 of 6 from `docs/design-plans/2026-07-06-density-dismantling.md`.

**Codebase verified:** 2026-07-07 (codebase-investigator: README structure, calibration.md url_cosharing section lines 441–569, freshness-date and invocation conventions).

---

## Acceptance Criteria Coverage

**Verifies: None** (documentation/calibration phase — no design ACs attach here; design "Done when": docs updated with freshness dates; calibration script runs against a ClickHouse instance, which is a deploy-time human verification step since CI has no ClickHouse).

---

## Codebase Context (verified 2026-07-07)

- `url_cosharing/CLAUDE.md`: sections Purpose / Architecture / Weighting Scheme / Contract / Commands; carries `Last verified:` date (line 3) — the repo convention is CLAUDE.md-only freshness dates; READMEs carry none.
- `url_cosharing/README.md` (~45 lines): "How it works" (numbered algorithm flow), "Usage", "Configuration" (3-column env-var table), "Output schema".
- `docs/calibration.md` (~715 lines): per-detector sections following `## Name` → `### Table:` → `#### Query N:` + SQL + `**Healthy ranges:**` → `### Tuning Levers` → escalation guidance. The URL Co-Sharing section (lines 441–569) currently documents: Query 1 cluster count/size distribution, Query 2 total_weight sanity, Query 3 evolution mix, Query 4 Newman-weight verification, the CPM Resolution Re-Tuning Procedure, Tuning Levers (incl. the now-removed `URL_COSHARING_MIN_EDGE_WEIGHT`), and Jaccard Threshold Guidance.
- No `[project.scripts]` entries anywhere; helper invocation is `uv run python -m <module>`.

---

<!-- START_SUBCOMPONENT_A (tasks 1-2) -->

<!-- START_TASK_1 -->
### Task 1: Update `url_cosharing/CLAUDE.md` and `url_cosharing/README.md`

**Verifies:** None (documentation)

**Files:**
- Modify: `url_cosharing/CLAUDE.md`
- Modify: `url_cosharing/README.md`

**Implementation:**

`CLAUDE.md` — rewrite to describe the shipped state (update `Last verified:` to the execution date):

- **Purpose:** detection via TF-IDF cosine-similarity network over per-account URL-sharing vectors (rolling 7-day window) + density-based dismantling isolating a high-precision coordinated core (Cinus, Minici, Luceri & Ferrara, WWW '25), Leiden CPM decomposing the core, existing evolution tracking. State the contract change plainly: **reads `osprey_execution_results` directly; `url_cosharing_pairs` is no longer consumed by the sidecar** (the pairs MV stays for investigation tooling).
- **Architecture:** add `similarity.py` (share filters, sparse TF-IDF, cosine graph — Core) and `dismantling.py` (centrality/edge quantile grid search, knee detection, guardrails — Core) and `calibrate.py` (density-surface dump — Shell) to the module list.
- **Replace the "Weighting Scheme" section** with a "Detection methodology" section covering: TF-IDF weighting (tf × ln(N/df), L2-normalized), edge weights = cosine similarity in [0, 1], the dismantling grid + knee rule + guardrails (`density_floor`, `max_flagged_fraction`), precision-first semantics (paper: precision > 0.9 at recall ≈ 0.1 — membership is a strong signal, not exhaustive coverage), empty-result-is-correct-behaviour (`knee_found = false` days), and the retained `total_weight` co-share-count semantics (Σ over cluster URLs of C(sharing members, 2)).
- **Contract:** input `osprey_execution_results`; outputs `url_cosharing_clusters` (+ `mean_edge_similarity`, `subgraph_density`), `url_cosharing_membership`, `url_cosharing_runs` (per-run observability).
- **Commands:** keep test/compose lines; add `uv run python -m url_cosharing.calibrate` (density-surface dump).

`README.md` — same structure it has today:

- "How it works": replace the pairs/Newman flow with: 1) fetch per-account URL share counts (7-day window) with activity/df filters, 2) TF-IDF + cosine similarity network, 3) density-based dismantling (grid search + knee detection + guardrails) isolates the coordinated core, 4) Leiden CPM decomposes the core, 5) evolution tracking + writes (clusters, membership, run metadata).
- "Configuration": extend the env-var table with every `URL_COSHARING_*` variable and its documented default — `WINDOW_DAYS` 7, `MIN_UNIQUE_URLS` 10, `MIN_URL_SHARERS` 5, `MAX_URL_DF_PCTL` 0.90, `EDGE_EPSILON` 0.05, `EDGE_QUANTILE_GRID` / `CENTRALITY_QUANTILE_GRID` `0.50,0.60,0.70,0.80,0.90,0.95,0.99`, `DENSITY_FLOOR` 0.5, `MAX_FLAGGED_FRACTION` 0.02, `RUNS_TABLE` `url_cosharing_runs`, plus the retained `RESOLUTION`, `MIN_CLUSTER_SIZE`, `JACCARD_THRESHOLD`, `EVOLUTION_WINDOW_DAYS`, `INTERVAL_SECONDS`, table names. **Remove** `MIN_EDGE_WEIGHT` and `MIN_COSHARERS` rows (and `PAIRS_TABLE`). Cross-check the final table against `config.py` field-for-field.
- "Output schema": add `url_cosharing_runs` and the two new cluster columns.

**Verification:**
```bash
grep -rn 'MIN_EDGE_WEIGHT\|MIN_COSHARERS\|PAIRS_TABLE\|newman' url_cosharing/CLAUDE.md url_cosharing/README.md
```
Expected: no matches describing current behaviour (a historical "previously Newman-weighted" aside is acceptable if clearly past-tense). `Last verified:` in CLAUDE.md is the execution date.

**Commit:** `docs(url_cosharing): document density-dismantling methodology and config`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Rewrite the URL Co-Sharing section of `docs/calibration.md`

**Verifies:** None (documentation)

**Files:**
- Modify: `docs/calibration.md` (URL Co-Sharing section, currently lines 441–569 — re-locate by the `## URL Co-Sharing Clusters` heading)

**Implementation:**

Keep the section's house format (`### Table:` / `#### Query N:` / SQL / `**Healthy ranges:**` / `### Tuning Levers`). Changes:

- **Keep** Query 1 (cluster count/size distribution) and Query 3 (evolution mix) as-is structurally, but revise healthy-range prose: the precision-first pipeline flags far fewer accounts than Leiden-over-everything — expect fewer, smaller clusters; note that with a 7-day window advancing daily (6/7 data overlap) evolution skews heavily toward `continuation`, and the Jaccard threshold may warrant retuning (link the existing Jaccard guidance).
- **Keep** Query 2 (total_weight sanity) — semantics unchanged (Σ C(k, 2)) — but drop the `min_edge_weight` reference in its healthy-range bullet.
- **Replace** Query 4 (Newman-weight verification) with run-metadata health queries against `url_cosharing_runs`:

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

  **Healthy ranges:** `flagged_pct` in the paper's observed 0.4–1.5% coordinated-account band (investigate sustained values above ~2%, which is the `MAX_FLAGGED_FRACTION` default ceiling); `knee_found = false` days are **correct behaviour**, alert only on consecutive false runs paired with *rising* `accounts_eligible`; frequent `guardrail_triggered = true` means the knee rule wants to over-flag — inspect the surface before loosening `MAX_FLAGGED_FRACTION`.
- **Add** a "Density-Surface Calibration" subsection: how to read the grid (`uv run python -m url_cosharing.calibrate` dumps per-cell `edge_quantile`, `centrality_quantile`, `min_component_density`, survivors), what a phase transition looks like (a sharp density jump between adjacent cells), how to choose `DENSITY_FLOOR` (just below the post-jump plateau; too high → permanent `knee_found = false`, too low → weak knees flag noise), and how to validate `MAX_FLAGGED_FRACTION` against observed `flagged_pct`.
- **Update the CPM Resolution Re-Tuning Procedure**: Leiden now runs on cosine-similarity weights (≤ 1 per edge) over the dismantled core only; the sweep procedure stays but is run via the calibrate module/core functions rather than "the same pairs"; expected cluster counts are much smaller.
- **Update Tuning Levers:** remove `URL_COSHARING_MIN_EDGE_WEIGHT`; add `DENSITY_FLOOR`, `MAX_FLAGGED_FRACTION`, `EDGE_EPSILON`, `EDGE_QUANTILE_GRID`/`CENTRALITY_QUANTILE_GRID`, `MIN_UNIQUE_URLS`/`MIN_URL_SHARERS`/`MAX_URL_DF_PCTL`, `WINDOW_DAYS`, each with a one-line effect description. Keep `RESOLUTION` and the Jaccard guidance.

Also grep the **Quote Co-Sharing** section and Summary Checklist for cross-references to url_cosharing behaviour that this change invalidates (quote_cosharing itself is out of scope and keeps the pairs methodology — make sure the doc still describes quote_cosharing's Newman weighting as its own, not by reference to url_cosharing).

**Verification:**
```bash
grep -n 'MIN_EDGE_WEIGHT' docs/calibration.md        # expect: no url_cosharing references (quote_cosharing may keep its own)
grep -n 'url_cosharing_runs' docs/calibration.md     # expect: >= 1
```

**Commit:** `docs: density-surface calibration methodology for url_cosharing`
<!-- END_TASK_2 -->

<!-- END_SUBCOMPONENT_A -->

<!-- START_SUBCOMPONENT_B (tasks 3-4) -->

<!-- START_TASK_3 -->
### Task 3: `calibrate.py` — density-surface dump module

**Verifies:** None (operator tooling; live run against ClickHouse is a deploy-time human verification step)

**Files:**
- Create: `url_cosharing/src/url_cosharing/calibrate.py`
- Test: `url_cosharing/tests/test_calibrate.py` (unit, for the pure formatter)

**Implementation:**

```python
# pattern: Imperative Shell
"""Dump the density-dismantling grid surface for offline calibration.

Usage: uv run python -m url_cosharing.calibrate
Reads the same URL_COSHARING_* / OSPREY_CLICKHOUSE_* env vars as the sidecar,
fetches the current window from ClickHouse, runs the full similarity +
dismantling pipeline, and prints the per-cell surface as TSV to stdout.
"""
from __future__ import annotations

import logging
import sys

from url_cosharing.config import AppConfig
from url_cosharing.db import CosharingDb
from url_cosharing.dismantling import DismantlingResult, dismantle
from url_cosharing.queries import fetch_url_shares_query
from url_cosharing.similarity import SimilarityNetwork, similarity_network

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger('url_cosharing.calibrate')


def format_surface(network: SimilarityNetwork, result: DismantlingResult) -> str:
    """Pure formatter: TSV surface plus a summary footer (Functional Core logic
    kept separate so the shell below stays untestable-thin)."""
    lines = ['edge_quantile\tcentrality_quantile\tmin_component_density\tsurviving_nodes\tsurviving_edges']
    for cell in result.surface:
        lines.append(
            f'{cell.edge_quantile}\t{cell.centrality_quantile}\t'
            f'{cell.min_component_density:.4f}\t{cell.surviving_nodes}\t{cell.surviving_edges}'
        )
    lines.append('')
    lines.append(
        f'# accounts_raw={network.accounts_raw} accounts_eligible={network.accounts_eligible} '
        f'urls_eligible={network.urls_eligible} graph_edges={network.graph_edges}'
    )
    lines.append(
        f'# knee_found={result.knee_found} edge_quantile={result.edge_quantile} '
        f'centrality_quantile={result.centrality_quantile} '
        f'min_component_density={result.min_component_density:.4f} '
        f'guardrail_triggered={result.guardrail_triggered} flagged_accounts={result.core.vcount()}'
    )
    return '\n'.join(lines)


def main() -> None:
    config = AppConfig.from_env()
    analysis = config.analysis
    db = CosharingDb(config.clickhouse)
    try:
        rows = db.fetch_url_shares(fetch_url_shares_query(analysis))
        logger.info(f'fetched {len(rows)} share rows')
        network = similarity_network(
            rows,
            analysis.min_unique_urls,
            analysis.min_url_sharers,
            analysis.max_url_df_pctl,
            analysis.edge_epsilon,
            logger,
        )
        result = dismantle(
            network.graph,
            analysis.edge_quantile_grid,
            analysis.centrality_quantile_grid,
            analysis.density_floor,
            analysis.max_flagged_fraction,
            analysis.min_cluster_size,
            logger,
        )
        sys.stdout.write(format_surface(network, result) + '\n')
    finally:
        db.close()


if __name__ == '__main__':
    main()
```

**Testing:**

`tests/test_calibrate.py`, class `TestFormatSurface` (pure function, no mocks): feed a hand-built `SimilarityNetwork` + `DismantlingResult` (reuse Phase 3/4 constructors on a tiny graph, or construct dataclasses directly) and assert: header row present, one line per surface cell with tab-separated values, summary footer lines carry the exact counts/flags. Empty surface (no-knee empty result) produces header + footer without error.

**Verification:**
```bash
cd url_cosharing
uv run pytest tests/test_calibrate.py
uv run python -c "import url_cosharing.calibrate"
```
Expected: tests pass; import clean. (A live `uv run python -m url_cosharing.calibrate` against a populated ClickHouse is the deploy-time human step — record it in test-requirements.md, not CI.)

**Commit:** `feat(url_cosharing): density-surface calibration dump module`
<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: Full-suite gate and freshness sweep

**Verifies:** density-dismantling.AC4.3 (final gate)

**Files:**
- None new — verification only.

**Step 1: Full suite**

```bash
cd url_cosharing && uv run pytest
```
Expected: all tests pass.

**Step 2: Freshness/consistency sweep**

```bash
grep -n 'Last verified' url_cosharing/CLAUDE.md                    # execution date
grep -rn 'min_edge_weight\|min_cosharers\|pairs_table' url_cosharing/src url_cosharing/tests   # empty
grep -c 'URL_COSHARING_' url_cosharing/README.md                   # matches config.py env var count
```

**Step 3: Commit any stragglers**

```bash
git status --porcelain   # if clean, nothing to do
```

**Commit (if needed):** `docs(url_cosharing): freshness sweep`
<!-- END_TASK_4 -->

<!-- END_SUBCOMPONENT_B -->

---

## Phase completion checklist

- [ ] CLAUDE.md/README.md describe the shipped methodology; CLAUDE.md `Last verified:` updated; env-var table matches `config.py` exactly.
- [ ] calibration.md URL Co-Sharing section rewritten: runs-table queries, density-surface methodology, updated tuning levers; quote_cosharing section still self-consistent.
- [ ] `uv run python -m url_cosharing.calibrate` importable and formatter unit-tested; live run recorded as human verification.
- [ ] `cd url_cosharing && uv run pytest` passes (density-dismantling.AC4.3).
