# Density-Dismantling Implementation Plan — Phase 3: Similarity network core

> **Superseded (2026-07-07, issue #3):** the URL df ceiling described in this document as a percentile of the df distribution (`max_url_df_pctl` / `quantile(max_url_df_pctl)(df)`) was a mis-transcription of Cinus et al.'s published code and is degenerate on production data. The implemented contract is `max_url_df_fraction` (`URL_COSHARING_MAX_URL_DF_FRACTION`): eligible URLs satisfy `df <= max_url_df_fraction * distinct_account_count` (sklearn `max_df` semantics), applied in SQL only. Do not reintroduce percentile/quantile ceiling logic from this document.

**Goal:** Pure Functional Core turning `(did, url, share_count)` rows into a TF-IDF cosine-similarity igraph network, with in-Python re-application of the activity/df filters (defense in depth over the SQL prefilter).

**Architecture:** All new code goes in `similarity.py` (`# pattern: Functional Core`, seeded with `UrlShareRow` in Phase 2) — no I/O, no ClickHouse imports, loggers permitted. scipy sparse arrays (`csr_array`) hold the account×URL matrix; TF-IDF is hand-rolled (no sklearn); cosine similarity is a sparse matrix product; the result is an undirected igraph graph with a `similarity` edge attribute.

**Tech Stack:** scipy >= 1.15 (`csr_array`, `diags_array`, `triu`, sparse `.multiply`), numpy, python-igraph >= 1.0, pytest.

**Scope:** Phase 3 of 6 from `docs/design-plans/2026-07-06-density-dismantling.md`.

**Codebase verified:** 2026-07-07 (codebase-investigator + external API research; sibling sidecars pin `scipy>=1.15.0`).

---

## Acceptance Criteria Coverage

This phase implements and tests:

### density-dismantling.AC1: Similarity network construction
- **density-dismantling.AC1.2 Success:** Accounts sharing fewer than `min_unique_urls` (default 10) unique URLs in the window are excluded before matrix construction.
- **density-dismantling.AC1.3 Success:** URLs shared by fewer than `min_url_sharers` (default 5) accounts, or with document frequency above the `max_url_df_pctl` (default 0.90) percentile, are excluded before TF-IDF.
- **density-dismantling.AC1.4 Success:** Edge weights are cosine similarities between account TF-IDF vectors, in [0, 1]; similarities below `edge_epsilon` are not materialized as edges.
- **density-dismantling.AC1.5 Edge:** An empty or fully-filtered input produces an empty graph; the run completes normally and writes a run-metadata row with zero counts. *(This phase delivers the empty-graph half; the run-metadata half is Phase 5.)*

### density-dismantling.AC4: Cross-cutting
- **density-dismantling.AC4.2:** `similarity.py` and `dismantling.py` are pure functional-core modules (no I/O, no ClickHouse imports). *(`similarity.py` half; `dismantling.py` is Phase 4.)*

---

## Design decisions fixed by this phase

- **Sparse-array API, not sparse-matrix:** use `scipy.sparse.csr_array` / `diags_array` (the maintained API — `sum(axis=1)` returns `ndarray`, avoiding `np.matrix` pitfalls; `diags` is deprecated in favour of `diags_array`).
- **TF-IDF formula (per design):** `tf × ln(N / df)`, natural log, no smoothing — `df ≥ 1` by construction (a column only exists because some account shared it), so no division by zero. A URL shared by every eligible account (`df == N`) gets weight 0 and stops contributing, which is exactly the intended down-weighting. Rows are L2-normalized; all-zero rows (possible when every URL an account shares has `df == N`) are left as zero vectors, not NaN.
- **Filter semantics mirror Phase 2's SQL exactly (single pass over raw rows):** account activity (unique-URL count) and URL df are both computed over the *unfiltered* window rows, then both filters applied together. An account may survive with fewer than `min_unique_urls` *remaining* URLs — by design (see phase_02.md "Filter-order decision").
- **df ceiling in Python uses `np.quantile(dfs, max_url_df_pctl)`** (linear interpolation), while ClickHouse `quantile()` is an approximate t-digest. Small disagreements near the ceiling are acceptable — the Python check is a defense-in-depth backstop, not an exactness contract.
- **All eligible accounts become graph vertices** (including ones left with no edges after the epsilon cutoff) — keeps vertex indices aligned with matrix rows; Phase 4's dismantling drops isolates itself.
- **Cosine values clamped to [0, 1]:** counts are non-negative so cosine is mathematically in [0, 1], but floating-point can yield `1 + 1e-16`; clamp with `np.minimum(data, 1.0)` so AC1.4's interval claim holds exactly.
- **Chunk-friendliness (design "Additional Considerations"):** the account×account product lives in one function (`build_similarity_graph`) so a future blockwise implementation can replace the single `@` without touching callers. Not required now.

---

<!-- START_TASK_1 -->
### Task 1: Add scipy/numpy dependencies

**Verifies:** None (infrastructure)

**Files:**
- Modify: `url_cosharing/pyproject.toml`
- Modify: `url_cosharing/uv.lock` (regenerated)

**Step 1: Edit dependencies**

In `url_cosharing/pyproject.toml`, extend the `dependencies` list (currently `clickhouse-connect>=0.8.0`, `igraph>=1.0.0`, `leidenalg>=0.11.0`):

```toml
dependencies = [
    "clickhouse-connect>=0.8.0",
    "igraph>=1.0.0",
    "leidenalg>=0.11.0",
    "numpy>=1.26.0",
    "scipy>=1.15.0",
]
```

`scipy>=1.15.0` matches the sibling sidecars (`url_overdispersion`, `quote_overdispersion`). numpy is declared explicitly because `similarity.py`/`dismantling.py` import it directly (not only transitively via scipy).

**Step 2: Verify operationally**

```bash
cd url_cosharing
uv sync
uv run python -c "import numpy, scipy.sparse; print(scipy.__version__)"
uv run pytest
```
Expected: sync succeeds, import prints a >= 1.15 version, suite still green.

**Step 3: Commit**

```bash
git add url_cosharing/pyproject.toml url_cosharing/uv.lock
git commit -m "chore(url_cosharing): add scipy and numpy dependencies"
```
<!-- END_TASK_1 -->

<!-- START_SUBCOMPONENT_A (tasks 2-5) -->

<!-- START_TASK_2 -->
### Task 2: `filter_shares` — in-Python activity/df filters

**Verifies:** density-dismantling.AC1.2, density-dismantling.AC1.3 (Python level)

**Files:**
- Modify: `url_cosharing/src/url_cosharing/similarity.py`
- Test: `url_cosharing/tests/test_similarity.py` (unit; create file with `# pattern: Functional Core` comment per repo test convention)

**Implementation:**

Add to `similarity.py` (imports at top: `import numpy as np`; keep the module ClickHouse-free):

```python
def filter_shares(
    rows: list[UrlShareRow],
    min_unique_urls: int,
    min_url_sharers: int,
    max_url_df_pctl: float,
    logger: logging.Logger | None = None,
) -> list[UrlShareRow]:
    """Re-apply the SQL prefilters in Python (defense in depth).

    Semantics match fetch_url_shares_query: activity and df are both computed
    over the raw input rows in a single pass, then applied together.
    """
    if not rows:
        return []

    urls_by_did: dict[str, set[str]] = {}
    dids_by_url: dict[str, set[str]] = {}
    for row in rows:
        urls_by_did.setdefault(row.did, set()).add(row.url)
        dids_by_url.setdefault(row.url, set()).add(row.did)

    dfs = np.array([len(dids) for dids in dids_by_url.values()])
    df_ceiling = float(np.quantile(dfs, max_url_df_pctl))

    active_dids = {did for did, urls in urls_by_did.items() if len(urls) >= min_unique_urls}
    eligible_urls = {
        url
        for url, dids in dids_by_url.items()
        if min_url_sharers <= len(dids) <= df_ceiling
    }

    kept = [row for row in rows if row.did in active_dids and row.url in eligible_urls]
    if logger:
        logger.info(
            f'filter_shares: {len(rows)} rows -> {len(kept)} '
            f'(accounts {len(urls_by_did)} -> {len({r.did for r in kept})}, '
            f'urls {len(dids_by_url)} -> {len({r.url for r in kept})}, df_ceiling={df_ceiling})'
        )
    return kept
```

Add `import logging` to the module imports.

**Testing:**

Create `tests/test_similarity.py` with class `TestFilterShares` (fixtures: small hand-built `UrlShareRow` lists). Tests:

- AC1.2: an account with 2 unique URLs is dropped when `min_unique_urls=3`; an account with exactly 3 is kept (boundary).
- AC1.3 floor: a URL shared by 1 account is dropped when `min_url_sharers=2`; exactly 2 kept (boundary).
- AC1.3 ceiling (hand-computed): URLs with dfs `[1, 1, 1, 1, 10]` and `max_url_df_pctl=0.5` → `np.quantile` ceiling `1.0` → the df-10 URL is dropped, df-1 URLs kept (subject to the floor).
- Single-pass semantics: an account whose raw unique-URL count passes but whose *surviving* URLs number fewer than `min_unique_urls` is still kept (regression pin for the SQL-mirror semantics).
- Empty input returns `[]`; fully-filtered input returns `[]`.

**Verification:**
Run: `cd url_cosharing && uv run pytest tests/test_similarity.py`
Expected: pass.

**Commit:** `feat(url_cosharing): in-Python share filters for similarity pipeline`
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: `ShareMatrix` and `build_share_matrix`

**Verifies:** density-dismantling.AC1.5 (empty-input matrix half); groundwork for AC1.4

**Files:**
- Modify: `url_cosharing/src/url_cosharing/similarity.py`
- Test: `url_cosharing/tests/test_similarity.py` (unit)

**Implementation:**

```python
@dataclass(frozen=True)
class ShareMatrix:
    counts: csr_array          # accounts × urls share counts
    accounts: tuple[str, ...]  # row index -> did (sorted)
    urls: tuple[str, ...]      # col index -> url (sorted)


def build_share_matrix(rows: list[UrlShareRow]) -> ShareMatrix:
    if not rows:
        return ShareMatrix(
            counts=csr_array((0, 0), dtype=np.float64),
            accounts=(),
            urls=(),
        )

    accounts = tuple(sorted({row.did for row in rows}))
    urls = tuple(sorted({row.url for row in rows}))
    did_to_idx = {did: idx for idx, did in enumerate(accounts)}
    url_to_idx = {url: idx for idx, url in enumerate(urls)}

    data = np.array([row.share_count for row in rows], dtype=np.float64)
    row_idx = np.array([did_to_idx[row.did] for row in rows])
    col_idx = np.array([url_to_idx[row.url] for row in rows])

    counts = csr_array((data, (row_idx, col_idx)), shape=(len(accounts), len(urls)))
    return ShareMatrix(counts=counts, accounts=accounts, urls=urls)
```

Imports to add: `from dataclasses import dataclass` is already there (Phase 2); add `from scipy.sparse import csr_array`.

Duplicate `(did, url)` rows are summed automatically by the COO→CSR construction — acceptable (the SQL GROUP BY means duplicates shouldn't occur, but summing is the right degradation).

**Testing:**

`TestBuildShareMatrix`:
- Known 2×3 case: exact `counts.toarray()` comparison, `accounts`/`urls` sorted, indices aligned.
- Duplicate `(did, url)` entries sum.
- Empty input → shape `(0, 0)`, empty tuples.

**Verification:**
Run: `cd url_cosharing && uv run pytest tests/test_similarity.py`
Expected: pass.

**Commit:** `feat(url_cosharing): sparse account-by-url share matrix`
<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: `tfidf_transform`

**Verifies:** groundwork for density-dismantling.AC1.4 (TF-IDF vectors)

**Files:**
- Modify: `url_cosharing/src/url_cosharing/similarity.py`
- Test: `url_cosharing/tests/test_similarity.py` (unit)

**Implementation:**

```python
def tfidf_transform(counts: csr_array) -> csr_array:
    """tf * ln(N / df), rows L2-normalized. Zero rows stay zero."""
    n_accounts, n_urls = counts.shape
    if n_accounts == 0 or n_urls == 0:
        return counts.copy()

    df = np.asarray((counts > 0).sum(axis=0)).ravel()
    idf = np.log(n_accounts / df)
    weighted = counts @ diags_array(idf, format='csr')

    norms = np.sqrt(np.asarray(weighted.multiply(weighted).sum(axis=1)).ravel())
    inv_norms = np.divide(1.0, norms, where=norms > 0, out=np.zeros_like(norms))
    return csr_array(diags_array(inv_norms, format='csr') @ weighted)
```

Imports to add: `from scipy.sparse import csr_array, diags_array` (extend Task 3's import).

**Testing:**

`TestTfidfTransform` with hand-computed expectations (use `pytest.approx`):
- Known-vector case: 2 accounts, 2 URLs — `a1` shares `u1`×2 and `u2`×1, `a2` shares `u2`×3. df(u1)=1, df(u2)=2, N=2 → idf(u1)=ln 2, idf(u2)=0. Row `a1` pre-norm = `[2·ln2, 0]` → normalized `[1.0, 0.0]`; row `a2` = `[0, 0]` (zero row from ubiquitous URL, no NaN).
- 3-account case where no df equals N: verify each cell against hand-computed `tf·ln(N/df)` then row-normalized values; verify every nonzero row has L2 norm ≈ 1.0.
- Empty (0×0) matrix returns empty, no error.
- Property check (bounds): all values finite, ≥ 0.

**Verification:**
Run: `cd url_cosharing && uv run pytest tests/test_similarity.py`
Expected: pass.

**Commit:** `feat(url_cosharing): hand-rolled sparse TF-IDF transform`
<!-- END_TASK_4 -->

<!-- START_TASK_5 -->
### Task 5: `build_similarity_graph` and the `similarity_network` entry point

**Verifies:** density-dismantling.AC1.4, density-dismantling.AC1.5 (empty-graph half)

**Files:**
- Modify: `url_cosharing/src/url_cosharing/similarity.py`
- Test: `url_cosharing/tests/test_similarity.py` (unit)

**Implementation:**

```python
@dataclass(frozen=True)
class SimilarityNetwork:
    graph: ig.Graph
    matrix: ShareMatrix
    tfidf: csr_array
    accounts_raw: int        # distinct accounts in the unfiltered input
    accounts_eligible: int   # distinct accounts after filters
    urls_eligible: int       # distinct urls after filters
    graph_edges: int


def build_similarity_graph(tfidf: csr_array, accounts: tuple[str, ...], edge_epsilon: float) -> ig.Graph:
    """Undirected graph over all accounts; edges are cosine similarities >= edge_epsilon."""
    graph = ig.Graph(n=len(accounts))
    if len(accounts) > 0:
        graph.vs['name'] = list(accounts)
    if tfidf.shape[0] == 0:
        graph.es['similarity'] = []
        return graph

    sims = triu(tfidf @ tfidf.T, k=1).tocoo()
    keep = sims.data >= edge_epsilon
    edges = list(zip(sims.row[keep].tolist(), sims.col[keep].tolist()))
    weights = np.minimum(sims.data[keep], 1.0).tolist()

    graph.add_edges(edges)
    graph.es['similarity'] = weights
    return graph


def similarity_network(
    rows: list[UrlShareRow],
    min_unique_urls: int,
    min_url_sharers: int,
    max_url_df_pctl: float,
    edge_epsilon: float,
    logger: logging.Logger | None = None,
) -> SimilarityNetwork:
    accounts_raw = len({row.did for row in rows})
    kept = filter_shares(rows, min_unique_urls, min_url_sharers, max_url_df_pctl, logger)
    matrix = build_share_matrix(kept)
    tfidf = tfidf_transform(matrix.counts)
    graph = build_similarity_graph(tfidf, matrix.accounts, edge_epsilon)
    return SimilarityNetwork(
        graph=graph,
        matrix=matrix,
        tfidf=tfidf,
        accounts_raw=accounts_raw,
        accounts_eligible=len(matrix.accounts),
        urls_eligible=len(matrix.urls),
        graph_edges=graph.ecount(),
    )
```

Imports to add: `import igraph as ig`; extend scipy import with `triu`.

Note `keep = sims.data >= edge_epsilon`: AC1.4 says similarities *below* `edge_epsilon` are not materialized, so `>=` keeps the boundary value.

**Testing:**

`TestBuildSimilarityGraph`:
- Two accounts with identical share vectors → one edge with `similarity == pytest.approx(1.0)` (AC1.4 upper bound; also pins the clamp).
- Two accounts with disjoint URL sets → cosine 0 → no edge.
- Epsilon boundary: a pair whose similarity is just below `edge_epsilon` gets no edge; a pair exactly at `edge_epsilon` gets one (AC1.4).
- All edge weights within `[0, 1]` on a 4-account mixed case; vertex `name` order matches `matrix.accounts`.
- Accounts with no surviving edges still appear as vertices (isolates preserved for Phase 4).

`TestSimilarityNetwork`:
- End-to-end small case: stage counts (`accounts_raw`, `accounts_eligible`, `urls_eligible`, `graph_edges`) all correct against a hand-built input where filters drop known rows.
- AC1.5: empty input → `graph.vcount() == 0`, `graph.ecount() == 0`, all counts zero except nothing raises.
- AC1.5: non-empty input that is fully filtered (every account below `min_unique_urls`) → `accounts_raw > 0`, `accounts_eligible == 0`, empty graph, completes normally.

**Verification:**
Run: `cd url_cosharing && uv run pytest`
Expected: full suite passes.

**Commit:** `feat(url_cosharing): TF-IDF cosine similarity network construction`
<!-- END_TASK_5 -->

<!-- END_SUBCOMPONENT_A -->

---

## Phase completion checklist

- [ ] `cd url_cosharing && uv run pytest` passes.
- [ ] `similarity.py` contains no ClickHouse or db imports (`grep -n 'clickhouse\|from url_cosharing.db' src/url_cosharing/similarity.py` → empty) — density-dismantling.AC4.2 half.
- [ ] Tests pin AC1.2, AC1.3 (Python level), AC1.4 (bounds + epsilon), AC1.5 (empty graph).
- [ ] scipy/numpy in `pyproject.toml` and `uv.lock` committed.
