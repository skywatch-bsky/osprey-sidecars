# URL co-sharing graph sidecar

Identifies clusters of accounts that repeatedly share the same URLs using TF-IDF cosine-similarity networks combined with density-based dismantling to isolate high-precision coordinated cores, then applies Leiden community detection.

## How it works

1. Fetch per-account URL share counts from `osprey_execution_results` (7-day rolling window) with activity and document-frequency filters
2. Compute TF-IDF vectors (tf = co-share count, idf = ln(N / df), L2-normalized) and build cosine-similarity network (edges = similarity in [0, 1])
3. Density-based dismantling: grid search over edge/centrality quantile pairs to isolate high-density coordinated cores via knee-detection heuristic and guardrails
4. Leiden CPM decomposes the dismantled core with cosine-similarity edge weights to identify clusters
5. Evolution tracking: Jaccard similarity matching against prior day membership snapshots to classify cluster transitions (birth/continuation/merge/split/death)
6. Write cluster results with similarity metrics to `url_cosharing_clusters`, daily membership snapshots to `url_cosharing_membership`, and run metadata to `url_cosharing_runs`

## Usage

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Run locally
uv run python -m url_cosharing.main

# Run via Docker
docker build -t url-cosharing .
docker run --env-file .env url-cosharing
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OSPREY_CLICKHOUSE_HOST` | `localhost` | ClickHouse server host |
| `OSPREY_CLICKHOUSE_PORT` | `8123` | ClickHouse HTTP port |
| `OSPREY_CLICKHOUSE_USER` | `default` | ClickHouse user |
| `OSPREY_CLICKHOUSE_PASSWORD` | `clickhouse` | ClickHouse password |
| `OSPREY_CLICKHOUSE_DB` | `default` | Database name |
| `URL_COSHARING_WINDOW_DAYS` | `7` | Rolling window for URL share history (days) |
| `URL_COSHARING_MIN_UNIQUE_URLS` | `10` | Minimum unique URLs per account to be included |
| `URL_COSHARING_MIN_URL_SHARERS` | `5` | Minimum accounts sharing a URL to be included |
| `URL_COSHARING_MAX_URL_DF_FRACTION` | `0.90` | Exclude URLs shared by more than this fraction of accounts (sklearn max_df semantics) |
| `URL_COSHARING_EDGE_EPSILON` | `0.05` | Similarity threshold for including edges in graph |
| `URL_COSHARING_EDGE_QUANTILE_GRID` | `0.50,0.60,0.70,0.80,0.90,0.95,0.99` | Edge-weight quantiles for dismantling grid search |
| `URL_COSHARING_CENTRALITY_QUANTILE_GRID` | `0.50,0.60,0.70,0.80,0.90,0.95,0.99` | Centrality quantiles for dismantling grid search |
| `URL_COSHARING_DENSITY_FLOOR` | `0.5` | Minimum component density threshold for dismantling |
| `URL_COSHARING_MAX_FLAGGED_FRACTION` | `0.05` | Maximum fraction of eligible accounts to flag (guardrail) |
| `URL_COSHARING_MAX_FLAGGED_ACCOUNTS` | `750` | Maximum absolute number of accounts to flag; effective cap is `min(max_flagged_fraction × eligible_accounts, max_flagged_accounts)` |
| `URL_COSHARING_RESOLUTION` | `0.05` | Leiden CPM resolution parameter |
| `URL_COSHARING_MIN_CLUSTER_SIZE` | `3` | Minimum cluster membership size |
| `URL_COSHARING_JACCARD_THRESHOLD` | `0.5` | Jaccard similarity threshold for evolution matching |
| `URL_COSHARING_EVOLUTION_WINDOW_DAYS` | `7` | Historical window for cluster matching |
| `URL_COSHARING_INTERVAL_SECONDS` | `3600` | Seconds between analysis cycles |
| `URL_COSHARING_RUNS_TABLE` | `url_cosharing_runs` | ClickHouse table for run metadata |
| `URL_COSHARING_CLUSTERS_TABLE` | `url_cosharing_clusters` | ClickHouse table for cluster results |
| `URL_COSHARING_MEMBERSHIP_TABLE` | `url_cosharing_membership` | ClickHouse table for membership snapshots |
| `URL_COSHARING_SOURCE_TABLE` | `osprey_execution_results` | Source table for URL shares |
| `URL_COSHARING_EXCLUSIONS_FILE` | unset | Path to an `exclusions.yaml` file declaring benign accounts/URLs to exclude (see [Exclusions](#exclusions); unset = no exclusions) |
| `URL_COSHARING_OTEL_ENABLED` | `false` | Enable OpenTelemetry traces/metrics |
| `URL_COSHARING_OTEL_SERVICE_NAME` | `url-cosharing` | OTel service name |
| `URL_COSHARING_OTEL_SERVICE_VERSION` | `0.1.0` | OTel service version |
| `URL_COSHARING_OTEL_ENVIRONMENT` | `local` | OTel deployment environment |
| `URL_COSHARING_OTEL_TRACES_ENABLED` | follows `URL_COSHARING_OTEL_ENABLED` | Enable OTel traces |
| `URL_COSHARING_OTEL_METRICS_ENABLED` | follows `URL_COSHARING_OTEL_ENABLED` | Enable OTel metrics |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | unset | OTLP collector endpoint used by the standard OTel exporter |

## Observability

OpenTelemetry is disabled by default. Set `URL_COSHARING_OTEL_ENABLED=true` and configure `OTEL_EXPORTER_OTLP_ENDPOINT` to export coarse run/stage traces and metrics to an OTLP collector. Traces cover the detector run and fixed pipeline stages such as fetching URL shares, building the similarity graph, dismantling, clustering, evolution, and persistence. Metrics record run success/failure, run and stage durations, knee/guardrail counters, and per-run counts such as accounts, URLs, graph edges, flagged accounts, and cluster count.

Telemetry is intentionally low-cardinality and privacy-preserving. DIDs, URLs, domains, cluster IDs, sample URLs, and sample DIDs must not be emitted as span attributes or metric labels. Those high-cardinality domain details belong in ClickHouse result tables. `url_cosharing_runs` remains the durable audit table for detector methodology and stage counts; OpenTelemetry is operational observability for timings, failures, and coarse health signals.

## Exclusions

Some share patterns are benign by construction — weather bots reposting identical links, accounts embedding the same GIF CDN URL. Set `URL_COSHARING_EXCLUSIONS_FILE` to point at a YAML file declaring them:

```yaml
excluded_domains:
  # Exact host OR any subdomain of it. static.klipy.com also covers
  # a.static.klipy.com; it never matches evil-klipy.com (the suffix
  # match is dot-anchored).
  - static.klipy.com
excluded_dids:
  # Exact DIDs (did:plc: + 24 chars of [a-z2-7]); the account and all
  # its shares are removed.
  - did:plc:aaaaaaaaaaaaaaaaaaaaaaaa  # example DID
```

Semantics:

- **Layer**: exclusions apply at the SQL eligibility layer, inside the share query and **before** document-frequency, eligibility, and mono-URL computation. An excluded URL stops counting toward `min_unique_urls` and the ≥2-URL floor; a fleet whose entire link set is excluded simply drops out via the existing rules.
- **Matching**: domain entries match the exact host or a subdomain of it (`lower(domain(url)) = entry` OR `endsWith(lower(domain(url)), '.entry')`). The leading dot prevents `klipy.com` from matching lookalike hosts like `evil-klipy.com`.
- **Hot reload**: the daemon reloads the file every cycle — edits take effect next cycle, no restart. `backfill` and `calibrate` load once at startup instead.
- **Failure handling**: unset env var → no exclusions (the file is not required). Env var set but the file missing/invalid at startup → the process exits non-zero. A file that breaks **mid-flight** → last-known-good exclusions stay in effect for that cycle and an ERROR is logged.
- **Auditability**: every run writes the applied revision to `url_cosharing_runs` (`excluded_domains_count`, `excluded_dids_count`, `exclusions_hash`, `excluded_shares_suppressed`), so results remain attributable to a specific exclusions revision.
- **Validation**: domains are normalized to lowercase and must be hostnames (`label.second-label` shape); DIDs must be full `did:plc` identifiers (`did:plc:` + 24 chars of `[a-z2-7]`). Unknown top-level keys are rejected (typo protection). These restrictions also make direct interpolation into the generated SQL safe.
- **Deploy order**: the four audit columns on `url_cosharing_runs` require the companion DDL in skywatch-osprey (`clickhouse-init/05-url-cosharing.sql` ALTERs). The compose `clickhouse-init` service only runs on first volume init, so apply the ALTERs to an existing cluster **before** deploying this sidecar version — otherwise run-metadata inserts fail every cycle.

## Output schema

- `url_cosharing_clusters` — cluster results with member count, metrics (`mean_edge_similarity`, `subgraph_density`), and evolution tracking
- `url_cosharing_membership` — daily membership snapshots per cluster (no TTL; retained for post-hoc analysis)
- `url_cosharing_runs` — run metadata including stage counts, quantile choices, knee-finding result, flagged account count, cluster count, and the exclusion audit fields (`excluded_domains_count`, `excluded_dids_count`, `exclusions_hash`, `excluded_shares_suppressed`). `accounts_raw` is the pre-exclusion window population, so exclusion suppression appears as extra `accounts_raw → accounts_eligible` attrition quantified by `excluded_shares_suppressed`
