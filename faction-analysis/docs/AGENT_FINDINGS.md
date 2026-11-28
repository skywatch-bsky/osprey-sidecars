# Agent findings — faction-analysis sidecar

Date: 2026-11-27 · Branch: `main` · Repo: `skywatch-bsky/osprey-sidecars` (sidecars submodule of `skywatch-osprey`)

Goal (operator direction): design a new sidecar that derives moderation intelligence about factions on
Bluesky from Osprey's ClickHouse, following Sahu, "Efficient Tracking of Communities on Evolving Graphs
with Leiden Algorithm" (HPDC '26, DOI 10.1145/3806645.3807601): dynamic Leiden on the follow/repost graph
with consistent faction IDs, a normalized faction-by-faction conflict matrix from hostile interactions,
per-user roles from Guimerà–Amaral cartography plus cascade timing, a mediator-versus-agitator estimate
from matched moments, a monthly full scan and a weekly refresh.

Method: direct local verification (reads of `skywatch-osprey`, `skywatch_rules`, and this repo); read-only
queries against the production ClickHouse on host `sys2` (schema and `system.*` metadata, plus
partition-pruned aggregates capped with `max_threads = 4` and a time limit); the full paper text supplied by
the operator; and five researcher passes (paper and code; ClickHouse schema for membership history; Osprey
engine; conflict and role-measurement prior art; follow-graph backfill sources). Claims are marked
**verified** (read locally or queried live), **cited** (external URL), or **inferred**.

Guard note: nothing was changed in `skywatch-osprey`, `skywatch_rules`, or on `sys2`. The ClickHouse on
`localhost` is a bare 24.1.8.22 with no user tables; all live facts below come from `sys2`.

---

## 1. Summary

1. **The data is there.** Every edge type the pipeline needs exists as a column in the production
   `osprey_execution_results`: follows, reposts, replies with the parent author, quote targets and blocks.
   Likes are dropped upstream. Recent volume is about 12M rows a day (§2).
2. **Three properties of the source shape the design.** It keeps two months of history (`TTL __timestamp +
   2 months`), and the operator wants recent history only, so the sidecar keeps nothing longer than that.
   Follow data is a flow, not a stock: only follows created since 2026-09-12 are observable, and backfill is
   ruled out. The hostility scores are not usable (0.35% coverage, intolerance never populated); the operator
   has ruled them out anyway.
3. **Scale is mid-size.** About 4–8M accounts and 150–230M edges over the full history. That is within reach
   of the Python `igraph` + `leidenalg` stack the existing sidecars already use, so the first version does
   not need the paper's C++ implementation (§5.2).
4. **The reusable part of the paper is its tracking step.** Algorithm 1 is a small majority-overlap
   renumbering that keeps community IDs stable across updates. It can be reproduced exactly in NumPy on top
   of a warm-started Leiden run. The incremental machinery is a speedup we do not need at this size, and the
   paper never tests users joining or leaving (§4).
5. **Storage: one current image plus a change log.** The mapping is one row per account per run, promoted as
   a unit; history is an append-only log of moves; identity is a small lineage registry. No per-run copy of
   the edge list, no derived edge tables, a retention policy on every output, and no query-time diff (§5.4).
6. **Starter packs should stay out of the affinity graph.** They are 14.5% of follow creates but come from
   1.7% of followers at 131 follows per adopter, mostly from one pack-generating account (§5.1).
7. **First version: the weekly refresh and the monthly full scan.** The conflict matrix is computed per run,
   from three text-free signals and a hostility classifier owned by the sidecar. The daily or hourly
   conflict monitor, the mediator-versus-agitator estimate and cascade summaries come later (§5.5, §5.8).

---

## 2. Production ClickHouse facts (verified on `sys2`, 2026-11-27)

### Table definition

The deployed table differs from `clickhouse-init/01-init.sql:5-10` in the repo (same commit `5ff0b07` on
both hosts). Live definition from `system.tables`:

```
MergeTree PARTITION BY toYYYYMM(__timestamp) ORDER BY (__action_id, __timestamp)
TTL toDateTime(__timestamp) + toIntervalMonth(2)
```

| Property | Value |
|---|---|
| Columns | 1,081 (the init file declares 3; the sink adds the rest at runtime) |
| Rows / size | 361.5M / 256 GiB, 19 active parts |
| Partitions | `202609`: 20.8M rows from 2026-09-12; `202610`: 166.5M; `202611`: 174.2M to 2026-11-27 21:39 |
| Column types | every string feature is `Nullable(String)`; booleans `Nullable(UInt8)`; scores `Nullable(Float64)`; lists `Array(String)` |

The repo's init file shows no partition key and no TTL. Treat the live table, not the file, as the source
of truth for this table.

### Columns the pipeline needs (all present)

`UserId`, `Handle`, `PdsHost`, `AccountCreatedAt`, `AccountAgeSeconds`, `Collection`, `OperationKind`,
`Path`, `AtUri`, `PostText`, `PostIsReply`, `PostIsSelfReply`, `PostReplyParent`, `PostReplyParentDid`,
`PostReplyRoot`, `PostEmbedRecordUri`, `PostEmbedRecordWithMediaUri`, `PostLanguages`, `PostIsEnglish`,
`FacetMentionList`, `FacetTagList`, `RepostSubject`, `RepostSubjectDid`, `IsRepost`, `FollowSubjectDid`,
`FollowSubjectVia`, `IsStarterPackFollow`, `IsOrganicFollow`, `BlockSubjectDid`, `HateBertScore`,
`IntoleranceScore`.

### Volume

| Measure | November 1–27 | Notes |
|---|---|---|
| Post creates | 51.06M | 22.99M are replies (45%), all with the parent author's DID; 3.64M quotes; 30.2M flagged English |
| Repost creates | 49.09M | 1.06M reposters, 772k original authors, 0 unresolved subjects |
| Follow creates | 28.23M | 1.78M followers, 4.32M followees; 4.45M deletes |
| Block creates | 3.78M | 380k blockers, 1.16M blocked; 115k deletes |
| `HateBertScore` non-null | 106k | 0.35% of English posts; 4.5k above 0.6, 3.4k above 0.85 |
| `IntoleranceScore` non-null | 0 | the Shield endpoint is not configured |

Full days Nov 20–26 run at 11.2–13.2M rows: 3.3–3.8M posts, 3.2–3.5M reposts, 2.0–2.4M follows,
244k–295k blocks. Nov 19 has 1.26M rows, which is an ingestion gap; the pipeline must not assume contiguous
coverage.

Whole history (2026-09-12 to 2026-11-27):

| Edge type | Creates | Distinct sources | Distinct targets |
|---|---|---|---|
| Follow | 63.3M | 2.89M followers | 6.27M followees |
| Repost | 105.9M | 1.41M reposters | 1.29M authors |
| Post (create) | 109.6M | 2.22M authors | — |
| Reply | 48.8M | — | 1.32M replied-to authors |
| Block | 8.87M | 652k blockers | 2.11M blocked |

Distinct counts are `uniqHLL12` estimates.

### Starter-pack follows (November)

| | Follows | Accounts | Follows per account |
|---|---|---|---|
| Via starter pack | 4.08M (14.5%) | 31,034 adopters | 131.5 |
| Organic | 24.16M (85.5%) | 1,774,655 followers | 13.6 |

17,436 distinct packs. The 15 largest are all owned by one account, `did:plc:pifkcjimdcfwaxkanzhwxufp`,
with 13–117 members each and 1,077–3,837 adopters each (inferred: a pack generator rather than a person).

### Host

`sys2`: 16 cores, 62 GB RAM (40 GB available when queried), 414 GB free disk. The sidecar would share this
host with 12 worker processes, ClickHouse, Kafka, Postgres and six sidecars.

### Other tables in `default`

`account_entropy_results` (312M rows, 44 GiB), `quote_overdispersion_results` (66.7M),
`url_overdispersion_results` (63.9M), `quote_cosharing_pairs` (6.1M), `url_cosharing_pairs` (454k), the
co-sharing clusters/membership/runs tables, `pds_signup_anomalies`, and a `domain_shares_hourly`
AggregatingMergeTree with 0 rows that no file in `clickhouse-init` declares.

---

## 3. Caveats on the source (verified unless marked)

1. **Two-month retention.** Follow, repost and reply history older than about two months is deleted from
   the source. The sidecar works inside that window and accumulates nothing beyond it (operator: recent
   history only, disk is limited).
2. **The follow blind spot is permanent.** Only follow create and delete events since 2026-09-12 exist, and
   backfill is ruled out. A follow edge is visible to the sidecar from its create until it ages out of the
   window or is deleted. Reposts and replies are flows and are complete for the window (modulo gaps).
3. **Deletes carry no target.** `FollowSubjectDid` and `BlockSubjectDid` are NULL on delete rows; the
   deleted record is identified by `Path` / `AtUri` (`models/record/base.sml:15-23`).
4. **`__timestamp` is the producer's send time.** It is taken from the action envelope
   (`osprey_worker/src/osprey/engine/executor/execution_context.py`,
   `osprey_async_worker/.../executor.py:596`), not from the worker's clock. The worker does not preserve
   per-account ordering (12 replicas, 24 partitions), so order by `__timestamp`, never by row position
   (researcher finding, cited to those files).
5. **Rows exist for non-hit actions.** The sink receives every executed action
   (`skywatch_plugins/src/output_sinks/clickhouse_execution_results_sink.py:48-65`).
6. **A sampler exists but is not configured.** `ActionSampler` drops an action only when its action config
   sets a rate other than 100 (`osprey_worker/src/osprey/async_worker/sinks/sink/rules_sink.py:36-53`).
   Nothing in `docker-compose.yaml` sets one and the live table has no `__sample_rate` column. Not fully
   excluded, because the engine's action configs were not inspected.
7. **`quote_cosharing` does not set a Leiden seed** (`quote_cosharing/src/quote_cosharing/analyzer.py:140-145`),
   so its partitions are not reproducible run to run. `url_cosharing` uses `seed=42`
   (`url_cosharing/src/url_cosharing/analyzer.py:273-279`).
8. **Deployment artifacts live in the other repo.** Sidecar tables are created by
   `skywatch-osprey/clickhouse-init/*.sql`, and sidecar processes are compose services there
   (`docker-compose.yaml:360-383`). Implementing this sidecar needs one migration file and one service block
   in `skywatch-osprey`. That is a later, separate change.
9. **Documentation drift on the closest precedent.** `05-url-cosharing.sql:55` declares a 7-day TTL on
   `url_cosharing_membership` while `url_cosharing/CLAUDE.md:49` says it has none.

---

## 4. Paper review (read in full from the operator's copy)

Sahu, "Efficient Tracking of Communities on Evolving Graphs with Leiden Algorithm", HPDC '26,
[DOI](https://dl.acm.org/doi/10.1145/3806645.3807601), CC BY 4.0. Code:
[puzzlef/leiden-communities-openmp-heuristic-dynamic](https://github.com/puzzlef/leiden-communities-openmp-heuristic-dynamic)
(MIT, research code). Technical report: [arXiv:2410.15451](https://arxiv.org/abs/2410.15451).

**Goal.** The paper names three properties a dynamic community algorithm needs: performance, quality and
stability, where stability means "community assignments for nodes that haven't significantly changed should
remain consistent over time". Its motivating example is "continuously track the top five most cohesive
political discussion groups and alert when they merge", which is our use case.

**Variants.** Naive-dynamic starts from the previous membership and processes every vertex. Delta-screening
uses modularity scores to pick likely-affected vertices. Dynamic Frontier marks the endpoints of deleted
edges inside a community and of inserted edges across communities, then spreads to neighbours when a vertex
moves. All three update after a batch of edge insertions and deletions.

**Contribution 1: tracking (§4.1, Algorithm 1).** After an update:
1. For each old community, find the current community that holds most of its vertices. The paper does this
   with a weighted Boyer–Moore majority vote, weighting each vertex by its degree. It is an approximation of
   "largest overlap", not an exact total.
2. For each current community, among the old communities that point to it, the one with the largest weight
   wins. That current community takes the old community's ID.
3. Current communities with no predecessor get new unique IDs.

So an ID survives when its community changes slowly; on a split the largest part keeps the ID and the rest
are new; on a merge one predecessor's ID survives. The paper records no lineage events; it only renumbers.
Effect (Fig. 2): with tracking, vertex-level agreement with static Leiden after an update-and-restore cycle
is 84% at a batch of 10⁻⁷|E| and 44% at 0.1|E|; without it, near zero. Tracking adds little runtime.

**Contribution 2: fewer refinements (§4.2).** A community is refined only when its accumulated edge-weight
change exceeds a fraction τre of its total weight, with insertions and deletions both counted. Suggested
τre = 0.6.

**Contribution 3: fewer splits (§4.3).** Only communities that could have come apart (deleted internal edges,
or vertex migration) are split into connected components, with subset renumbering so two disconnected parts
never share an ID.

**Results (§5).** Mean speedups over static Leiden of 3.9× (ND), 4.4× (DS) and 6.1× (DF) on graphs up to
214M vertices and 3.8B edges, on a 64-core machine; smaller on social networks. Modularity within 0.002 of
static Leiden, slightly better on social networks because the dynamic runs start from a good partition.
About 7× scaling at 32 threads, limited beyond that.

**Limits for us.**
- Batches contain only edge insertions and deletions: "No new vertices are added or removed" (§5.1.4).
  Users joining and leaving is our normal case and is untested. The released code has the same scope: it
  generates random edge batches itself, reads MatrixMarket or SNAP files, and exposes no API.
- Modularity with unit weights; CPM and hierarchy tracking are future work. The existing sidecars here use
  CPM.
- It is an OpenMP C++ program tuned for 64 threads. `sys2` has 16 cores, shared with production.

**What the paper points at in Python.** §1 notes that the multislice method of Mucha et al. "is present in
the Leiden official package (leidenalg on PyPI)" and objects that it "requires processing the entire
multislice graph simultaneously". Verified in the [`leidenalg` reference](https://leidenalg.readthedocs.io/en/stable/reference.html):
`find_partition_multiplex(graphs, partition_type, layer_weights=None, …)` optimises one partition over
several edge layers on the same vertices, and `find_partition_temporal(graphs, partition_type,
interslice_weight=1, …)` couples consecutive snapshots by matching vertex ids. Neither takes an initial
membership. For a two-slice window the objection is minor; the cost is roughly double the edges in memory.

**Fit.** Take the tracking idea and implement it exactly. Treat the incremental algorithms, and the C++
code, as an option only if runtime or memory becomes a problem.

---

## 5. Design consequences

### 5.1 Graph layers

- **Affinity (what Leiden runs on): organic follows plus windowed reposts**, as the operator proposed.
  Follows give stable structure and include quiet accounts; reposts reflect current alignment and are
  complete for the window. With no backfill the follow layer contains only follows created inside the
  window, so it recency-weights ties by construction: an account that has followed nobody new lately is
  still present through the follows it receives. Weighting and window length are calibration questions.
- **Starter packs stay out of the affinity layer** (operator: exclusion acceptable, organic probably more
  important). Reasons: one click adds about 131 follows, so a single decision would define an adopter's
  neighbourhood; and the largest packs come from one generator account, which would import a recommender's
  idea of similarity into a graph we want to derive. Pack follows attach adopters to members and add no
  member-to-member edges, so they do not fabricate cliques; excluding them mostly changes where adopters
  land. Keep `FollowSubjectVia` in the derived follow table so adoption can be added back as a separate,
  down-weighted layer.
- **Targeting lists are a different signal.** A list used for targeting produces blocks and replies, not
  follows, so follow-via-pack cannot reveal one. Curated list membership is observable (`app.bsky.graph.
  listitem` creates: 5.73M in November) and could later support "lists whose members receive disproportionate
  blocks or replies". Not in the first version.
- **Replies, quotes and blocks feed the conflict layer, not the partition.**

### 5.2 Partition engine

Recommended for the first version: Python `leidenalg`, one combined weighted layer, warm-started from the
previous membership with a fixed seed, then exact tracking (below). Inferred cost at 150–230M edges:
roughly 10–15 GB for the graph and tens of minutes single-threaded, to be measured in a spike. That fits
`sys2` and avoids a 64-thread binary on a 16-core production host.

The monthly full scan is the correction for warm-start bias: it rebuilds the graph from the derived tables
and re-runs from a cold start with the fixed seed, then tracks against the previous partition. Record the
quality of each run (modularity, agreement with the previous run) so drift is visible.

Two alternatives to compare in the spike, not to adopt up front: `find_partition_temporal` over the previous
and current snapshots (coupling instead of warm start), and `find_partition_multiplex` over follow and
repost layers.

**Measuring noise.** Warm-starting keeps IDs stable by construction, which also hides real change. A
multi-seed run (same graph, several seeds) gives each account a confidence: the share of seeds that agree
with its tracked faction. Monthly is enough.

### 5.3 Faction identity

Implement Algorithm 1 exactly with NumPy rather than by majority vote. A contingency table of old ID by new
ID, weighted by degree, is one `bincount` over the vertices, and it also yields the events the paper does
not record:

| Pattern in the table | Event | ID handling |
|---|---|---|
| one old, one new, high overlap | continuation | kept |
| one old, several new above a share threshold | split | largest part keeps the ID; others are births with the parent recorded |
| several old, one new | merge | largest predecessor keeps the ID; the others are recorded as merged into it |
| new with no predecessor above threshold | birth | new ID |
| old with no successor above threshold | death | ID retired, never reused |

This differs from the existing precedent, `url_cosharing/src/url_cosharing/analyzer.py:126-252`
(`compute_evolution`), which mints new IDs for every merge and every part of a split (`:207-230`) and
compares clusters pairwise with Jaccard over member sets. Pairwise comparison is also quadratic in the
number of communities and linear in members for each pair, which will not do for thousands of communities
over millions of members.

### 5.4 Storage

**Requirement.** Map DIDs to a faction. Serve the current mapping. Answer how membership is changing: which
factions exchange users, who moved, how long a user has been where. No full copy of the mapping per run, no
copies of the edge list, no query-time diff.

**Principle: store by how often things change.**

| Class | Changes | Storage |
|---|---|---|
| Assignment (DID → faction) | rarely, for few accounts per run | one **current image** (one row per account) plus an append-only **moves log** |
| Continuous features (z-score, participation, cascade ranks) | slightly, for nearly everyone, every run | one **current image**, rewritten each run; retain history only at a coarse cadence if wanted |
| Faction identity (registry, lineage) | on structural events | small registry plus an event log |
| Faction aggregates (sizes, conflict cells, cascades) | every run, tiny | append-only time series |
| Run metadata | once per run | append-only |
| Graph edges | every run | **never stored**; read from the source at run time |

Proposed tables (names follow the `<sidecar>_<noun>` convention and are proposals):

| Table | Engine and key | Written per run |
|---|---|---|
| `faction_analysis_runs` | MergeTree, `ORDER BY run_id` | one row |
| `faction_analysis_members` | MergeTree, `PARTITION BY run_id`, `ORDER BY did` | one partition |
| `faction_analysis_member_features` | MergeTree, `PARTITION BY run_id`, `ORDER BY (faction_id, did)` | one partition |
| `faction_analysis_moves` | MergeTree, `ORDER BY (did, run_id)` | one row per changed assignment |
| `faction_analysis_lineage` | MergeTree, `ORDER BY (run_id, faction_id)` | one row per event |
| `faction_analysis_factions` | ReplacingMergeTree, `ORDER BY faction_id` | changed factions only |
| `faction_analysis_faction_runs` | MergeTree, `ORDER BY (run_id, faction_id)` | one row per faction |
| `faction_analysis_conflict` | MergeTree, `ORDER BY (window_start, from_faction_id, to_faction_id)` | non-zero cells |
| `faction_analysis_cascades` | MergeTree, `ORDER BY (first_seen, root_uri)` | cascades above a size floor |

No table stores graph edges. The operator wants recent history only and disk is limited, and the source
already retains two months, which is the longest window the sidecar needs.

**The current image and its history.**
- `…_members` holds exactly one row per account for the run: `did, faction_id, since_run_id` (tenure),
  `prev_faction_id`, `confidence`. A run is promoted by writing its row to `…_runs`; readers use the latest
  promoted `run_id`, and partition pruning keeps every read to one partition. Old partitions are dropped
  except those a retention policy keeps (monthly is cheap: about 100 MB compressed per image at 5M
  accounts). This gives exact state, "as-of run" reads for free, and no deduplication semantics.
- The researcher pass advises against ReplacingMergeTree as the authoritative current mapping when
  freshness must be exact, because deduplication is eventual
  ([ReplacingMergeTree](https://clickhouse.com/docs/en/engines/table-engines/mergetree-family/replacingmergetree)).
  The image approach takes that advice; ReplacingMergeTree is used only where a fold is wanted (follow
  state, the faction registry).
- `…_moves` is the whole assignment history: `(run_id, did, from_faction_id, to_faction_id, move_kind)`.
  A per-run transition matrix is a `GROUP BY` over one run's rows; a user's trajectory is a point lookup.
  Rows are proportional to movers, not members.

**Where the diff happens.** Once, in sidecar memory, at write time: compare the new assignment vector with
the previous one (a linear pass), which produces the rows for both tables. Nothing is diffed at query time.

**Answer to the operator's question.** Yes: movement trends can be captured without per-run copies and
without query-time diffs, provided two things hold.

1. *Faction identity must be stable*, or a "move" is not a move. This is why §5.3 comes before storage.
2. *Noise must be separable from movement.* Accounts on a boundary flip between two factions on successive
   runs. `prev_faction_id` lets a return to the previous faction be recognised, `since_run_id` lets a
   consumer require persistence, and `confidence` (§5.2) discounts unstable accounts.

**Joining the mapping at scale.** The hourly or daily conflict aggregate joins tens of millions of
interaction rows against the mapping. Expose the promoted image as a ClickHouse dictionary keyed by DID
(`complex_key_hashed` for a string key; about 0.5–1 GB resident at 5M keys) rather than a table join, and
reload it after each promotion. Fallback if memory is tight: a DID → UInt64 surrogate table and a `hashed`
dictionary ([dictionary best practices](https://clickhouse.com/docs/best-practices/dictionaries)).

**Reading edges.** Each run reads the window's follow, repost, reply, quote and block rows straight from the
source, partition-pruned on `__timestamp`. The existing sidecars read seven-day windows the same way
(`url_cosharing/src/url_cosharing/queries.py:64-73`). Current follow state inside the window is creates minus
deletes keyed by record URI. Measure the scan in the spike before adding any aggregate table; daily pair
aggregates would be the first optimisation to reach for, refreshable views the second (still experimental on
the pinned 24.1, `05-url-cosharing.sql:86`).

**Retention and footprint.** Every output table gets an explicit policy. Sizes below are inferred from the
measured volumes, not measured.

| Table | Policy | Estimated size |
|---|---|---|
| `…_runs`, `…_lineage`, `…_factions`, `…_faction_runs` | keep | a few MB a year |
| `…_moves` | keep, or TTL by policy | a few hundred MB a year compressed if 5–10% of accounts move per week |
| `…_members`, `…_member_features` | latest image plus monthly images | about 100 MB and 200 MB compressed per image at 5M accounts |
| `…_conflict` | TTL, for example one year | thousands to millions of cells per window |
| `…_cascades` | TTL, for example 90 days | cascades above the size floor only |

### 5.5 Conflict matrix

Hostility scores are ruled out. What remains, and what each is good for:

| Signal | Direction | Volume | Notes |
|---|---|---|---|
| Blocks (creates) | blocker → blocked | 8.9M in history | explicit and abundant; also used for curation and boundary management |
| Reply pile-ons | replier → target author | derived from 48.8M replies | many accounts of one faction replying to one author in a short window; no text needed |
| Quote-post swarms | quoter → quoted author | 3.6M quotes in November | the usual "dunking" vector; no text needed |
| Text classification | author → target | replies and quotes | a hostility classifier owned by this sidecar; in the first version (operator, 2026-11-27) |

Normalisation: raw hostile counts conflate faction size with hostility, because a large faction sends more
of everything. The researcher pass recommends activity-adjusted hostile-event rates and expected-count
residuals. Concretely, for each ordered pair (A → B): the hostile count, the total interaction count, and a
score relative to the hostility expected from interaction volume alone, so that the score answers "is A more
hostile to B than A is to everyone, and than everyone is to B?". Direction comes from counting each
direction separately. Bluesky studies so far document structural and stance polarisation, toxicity and
blocks, not a directed faction-by-faction hostility matrix (researcher finding), so the definition is ours
to calibrate.

**The classifier.** The operator chose to add a hostility classifier owned by this sidecar (2026-11-27). It
is the first sidecar here that interprets text, and it changes the dependency contract: sidecars are
documented as depending on ClickHouse only (`README.md:3`), though `exact_dup_classifier` already calls an
external classification API (`README.md:76-85`). Sizing: about 1.5–1.7M replies and 135k–250k quotes a day
at the recent run-rate, so scoring everything is about 20 texts a second sustained. Inferred: a small local
transformer on one or two cores can do that with batching; an external API makes it expensive. Scoring only
interactions whose two ends are in different factions cuts it to the part the matrix needs. Open questions
for planning: a local model or an API; which texts are scored; where scores live and for how long (a
sidecar table keyed by post URI with a TTL, or recomputed per run); how the model is versioned and
calibrated; and how a conflict cell records which model produced it.

### 5.6 Roles

- **Cartography.** Within-faction degree z-score: $(\kappa_i - \bar\kappa_{s_i}) / \sigma_{\kappa_{s_i}}$.
  Participation coefficient: $P_i = 1 - \sum_s (\kappa_{i,s} / \kappa_i)^2$. Both depend on the partition:
  a finer partition raises participation for everyone, so they are comparable within one run and across
  runs only with lineage in hand. Both have undefined cases (single-member factions, zero degree, zero
  variance) that need an explicit "undefined" value. Store the two numbers separately, as the operator asked.
- **Cascade timing.** A cascade is a root post with its replies, reposts and quotes (no likes). For each
  participant, the rank of its first participation time in that cascade, normalised to a percentile; per
  user, the mean over cascades above a minimum count. Split by role in the cascade: author of content
  others spread ("amplified") versus spreading someone else's ("amplifier"). Percentile ranks remove a
  size bias but do not remove the small-sample bias (researcher finding).
- **Mediator versus agitator.** Deferred to a later version (operator, 2026-11-27). The matched-moment
  design is the right shape: a boundary user tends to post
  *because* hostility is already rising, so a raw before/after comparison measures the user's timing, not
  their effect. Remaining risks: the user's own post draws hostile replies aimed at the user, which is not
  the same as inflaming the room; threads differ in topic; power is low. The researcher pass describes
  matched event-time controls plus within-thread comparison as a pragmatic estimator with strong
  self-selection caveats. Report an effect size with a sample size and an explicit "not enough evidence"
  state. Without scores, "hostility level" inside a thread has to be defined behaviourally: blocks of thread
  participants, rate of new distinct repliers from other factions, or a classifier if one is added.

### 5.7 Backfill

Ruled out by the operator (2026-11-27): disk is limited and only recent history is wanted. Consequence:
follow coverage begins at the source's first row, 2026-09-12, and a follow edge exists for the sidecar only
while its create event is inside the source window. The earlier blind spot is permanent and accepted.

For the record, the option that existed: [Jetstream Replay](https://bsky.network/docs/jetstream-replay/),
an official Bluesky-hosted archive with collection-filtered, resumable, byte-metered downloads behind an
API key. Its oldest segment and its pricing were not checked.

### 5.8 Cadence

| Tier | Cadence | Work | Cost |
|---|---|---|---|
| Source scan | each run | read follow, repost, reply, quote and block rows in the window from the source | one scan per run; measure before optimising |
| Conflict monitor | daily or hourly | hostile and total interactions per ordered faction pair, joined through the dictionary; compared with a trailing baseline | aggregate only |
| Refresh | weekly | rebuild the graph from derived tables, warm-started Leiden, track IDs, write image, moves, lineage, features, stats | the heavy one |
| Boundary users | weekly | mediator-versus-agitator estimates for accounts with enough events | per-thread work, few accounts |
| Full scan | monthly | cold-start recompute, re-track, quality report, multi-seed confidence | heaviest |

The conflict monitor is what makes near-real-time monitoring possible without re-running Leiden: the
partition only has to be approximately right for the aggregate to be useful.

The operator scoped the first version to the refresh and the full scan (2026-11-27). The conflict matrix is
still computed once per run over the window, so the later monitor only has to raise the cadence (inferred
from the choice to include the classifier now). The other rows describe later versions.

---

## 6. Conventions a new sidecar must follow (verified)

- Standalone project, Functional Core / Imperative Shell: `config.py`, `queries.py`, `analyzer.py` are core;
  `db.py`, `main.py` are shell; `telemetry.py` holds all OpenTelemetry setup; core modules never import
  OTel (`README.md:7-14`, `quote_cosharing/CLAUDE.md:9-16`).
- No imports from `osprey_worker` or other sidecars. Depends on ClickHouse only.
- Connection settings come from `OSPREY_CLICKHOUSE_HOST/PORT/USER/PASSWORD/DB`; sidecar settings are
  prefixed with the sidecar name in upper case (`quote_cosharing/src/quote_cosharing/config.py:26-79`).
- Leiden precedent: `leidenalg.find_partition(graph, leidenalg.CPMVertexPartition, weights=…,
  resolution_parameter=…, seed=42)`; dependencies `igraph >= 1.0.0`, `leidenalg >= 0.11.0`
  (`url_cosharing/pyproject.toml:11-13`).
- Each sidecar carries a `CLAUDE.md`, a `Dockerfile`, and `tests/test_docs.py`.
- OTel must not carry DIDs, URLs, cluster or faction ids, handles, query text or exception messages
  (`url_cosharing/CLAUDE.md:42`, `quote_cosharing/CLAUDE.md:51`).
- Outputs are consumed by `skywatch-mcp` as read-only investigation tools named `<sidecar>_<noun>`
  (`docs/design-plans/2026-07-09-mcp-sidecar-tools.md:14-51`). Faction tables should stay friendly to that:
  sanitizable DID and date filters, sensible defaults, bounded results.
- Schema ground truth is `skywatch-osprey/clickhouse-init/*.sql`; sidecars never create tables.

---

## 7. Researcher passes

Each pass returns a short conclusion and a source list.

1. **Paper and code.** Conclusion: the dynamic work was evaluated on edge-only changes, has no Python API
   and no vertex insertion or deletion; use Python `igraph`/`leidenalg` as quality baselines. Superseded in
   part by the measured scale (§2): at this size Python is sufficient for the first version.
   Sources: [arXiv:2410.15451](https://arxiv.org/html/2410.15451),
   [arXiv:2405.11658](https://arxiv.org/html/2405.11658),
   [arXiv:2312.13936](https://arxiv.org/html/2312.13936), [code](https://github.com/puzzlef/leiden-communities-openmp-heuristic-dynamic),
   [earlier dynamic code](https://github.com/puzzlef/leiden-communities-openmp-dynamic),
   [leidenalg reference](https://leidenalg.readthedocs.io/en/stable/reference.html),
   [igraph communities](https://python.igraph.org/en/stable/api/igraph.community.html),
   [NetworKit](https://networkit.github.io/dev-docs/python_api/community.html),
   [cuGraph Leiden](https://docs.rapids.ai/api/cugraph/stable/api_docs/api/cugraph/cugraph.leiden/),
   [TILES](https://github.com/GiulioRossetti/TILES),
   [survey](https://link.springer.com/article/10.1007/s10994-016-5582-8),
   [NMI](https://jmlr.org/papers/volume11/vinh10a/vinh10a.pdf), [seed sensitivity](https://arxiv.org/abs/1802.10582).
2. **Storage.** Conclusion: one atomically promoted current image for feature-heavy state, append-only
   change and lineage tables, moves only in the history log, no weekly history of continuous features,
   dictionary for the join. Sources: [ReplacingMergeTree](https://clickhouse.com/docs/en/engines/table-engines/mergetree-family/replacingmergetree),
   [CollapsingMergeTree](https://clickhouse.com/docs/en/engines/table-engines/mergetree-family/collapsingmergetree),
   [dictionaries](https://clickhouse.com/docs/reference/statements/create/dictionary/layouts/overview),
   [hashed layout](https://clickhouse.com/docs/reference/statements/create/dictionary/layouts/hashed),
   [ssd-cache layout](https://clickhouse.com/docs/reference/statements/create/dictionary/layouts/ssd-cache),
   [best practices](https://clickhouse.com/docs/concepts/features/dictionaries/best-practices),
   [join algorithms](https://clickhouse.com/docs/concepts/features/operations/select/join-algorithms),
   [EXCHANGE](https://clickhouse.com/docs/reference/statements/exchange),
   [refreshable MV](https://clickhouse.com/docs/concepts/features/materialized-views/refreshable-materialized-view),
   [Palla et al. 2007](https://arxiv.org/pdf/0704.0744),
   [Rossetti and Cazabet survey](https://dl.acm.org/doi/10.1145/3172867),
   [Aynaud and Guillaume 2010](https://doi.org/10.1109/asonam.2010.17),
   [tnetwork matching](https://tnetwork.readthedocs.io/en/latest/reference/dyn_communities.html).
3. **Osprey engine.** Conclusion: `__timestamp` is the envelope's send time; per-account order is not
   guaranteed; rows exist for executed actions including non-hits; a sampler and skip paths exist. Git
   history and the `atkafka` enrichment were not resolved by that pass; live facts in §2 fill the gap.
4. **Conflict and roles.** Conclusion: Bluesky work covers structural and stance polarisation, toxicity and
   blocks, not directed faction matrices; use activity-adjusted rates and expected-count residuals;
   cartography formulas and degeneracies confirmed; cascade ranks are size-biased; matched controls are
   pragmatic with strong self-selection caveats. Sources include [Kumar et al., WWW 2018](https://doi.org/10.1145/3178876.3186141),
   [Maclure 1991](https://quantifiedself.com/wp-content/uploads/2012/06/1991-Maclure.pdf),
   [Brady et al. 2021](https://www.science.org/doi/10.1126/sciadv.abe5641),
   [arXiv:2505.01174](https://arxiv.org/html/2505.01174), [arXiv:2505.02317](https://arxiv.org/html/2505.02317),
   [arXiv:2506.03443](https://arxiv.org/abs/2506.03443), [arXiv:1910.09563](https://arxiv.org/abs/1910.09563).
5. **Backfill.** Conclusion: Jetstream archive replay is the best apparent path, pending its oldest date
   and byte pricing. Not pursued (operator decision, §8.3). Sources: [Jetstream Replay](https://bsky.network/docs/jetstream-replay/),
   [rate limits](https://docs.bsky.app/docs/advanced-guides/rate-limits),
   [getFollows](https://docs.bsky.app/docs/api/app-bsky-graph-get-follows),
   [arXiv:2501.11605](https://arxiv.org/html/2501.11605v2).

URLs are listed as returned; titles were not individually checked unless cited with a name in the text.

---

## 8. Operator decisions

Recorded so far (2026-11-27):

1. **Hostility scores are ignored.** `HateBertScore` and `IntoleranceScore` play no part in the design.
2. **Starter-pack follows may be excluded.** Organic follows are probably the more important signal;
   starter packs could also serve as targeting lists. Design consequence in §5.1: excluded from the
   affinity layer, retained as a column.
3. **No backfill; recent history only; disk is limited.** Follow coverage starts 2026-09-12. The sidecar
   stores outputs only, each with a retention policy (§5.4, §5.7).

Answered in session (2026-11-27):

4. **Hostile interactions come from three text-free signals plus a classifier owned by this sidecar**:
   blocks, reply pile-ons, quote-post swarms, and a hostility score over reply and quote text (§5.5).
5. **One level of factions, about 100 to 300**, with a minimum faction size. The conflict matrix has on the
   order of 40,000 ordered cells at the top of that range.
6. **The first implementation is the weekly refresh and the monthly full scan.** The daily or hourly
   conflict monitor, the mediator-versus-agitator estimate and cascade summaries come later. Inferred: the
   conflict matrix is still computed once per run (§5.8).
7. **Outputs are consumed on demand** through read-only `skywatch-mcp` tools; alerting comes later, once a
   conflict baseline exists. Each faction carries a readable description (sample handles, common tags,
   languages), for which `url_cosharing_clusters.sample_dids` is precedent.

Answered in session (2026-11-28), during the build:

8. **The window is at most 60 days.** A read-only measurement against production ClickHouse on `sys2` is
   approved under the plan's guardrails. Changes in `skywatch-osprey` are requested as GitHub issues after
   the sidecar is built.
9. **The classifier is a linear model now**, trained on block-derived labels plus a bounded Jev-labelled
   teaching sample. The build also exports a dataset with parent and child text, so a transformer can be
   trained later on a GPU machine. Cascade timing is deferred.
10. **One level of factions in a flexible band of 20 to 300**, which replaces the 100 to 300 of decision 5.
    The operator first suggested a target of 20 to 30 communities and then judged 52 factions reasonable,
    to leave the calibration room to move. The measurement put two configurations in the band; the one with
    the better seed stability is the default (25 factions at resolution 0.5), and resolution 1.0 gives the
    partition of 52 factions (§9.3, `MEASUREMENT.md`).

---

## 9. Design note, classifier decisions and the build (2026-11-28)

### 9.1 Bree's design note

The note (`birdbree/a501149e1b6f89e2d3a8daa18d3e0729`) shaped the build. What was taken, and what changed:

| Element | Disposition |
|---|---|
| Window of 90 days | 60 days: the source keeps two months |
| `follow + alpha * log(1 + reposts)`, mutual follows heavier | Adopted, with damping of edges into large accounts |
| CPM at several resolutions, 20 to 50 seeds | One level. CPM at 0.01 fragmented into 1.97M communities and did not converge in an hour; modularity at 0.5 is the default |
| Jaccard matching with a Hungarian assignment | Rejected for the paper's contingency-table tracking, which is linear and records lineage |
| Hostility as a probability summed, never thresholded early | Adopted for the text column |
| Classifier sees parent and child together | Adopted for Jev and for the exported dataset. The linear model at scale reads the child only |
| Conflict needs contact volume and a hostility rate above expectation | Adopted: `conflict_score` is the smaller of the contact lift and the best signal lift |
| Beta-binomial shrinkage | Adopted as `(H + k*g) / (N + k)` |
| Streaming monitor, hour-of-week baseline, per-episode roles, mediators | Deferred with the monitor tier (operator, 2026-11-27) |

### 9.2 Classifier decisions (operator, 2026-11-28)

- A linear model now: character n-grams and structural features, scikit-learn.
- Labels: block-derived positives and negatives, plus a bounded Jev-labelled teaching sample. Jev never
  runs at scale. The request and token budgets are hard and count every attempt.
- The build exports a portable dataset with parent and child text, so a transformer can be trained on a
  GPU machine later.
- Cascade timing is deferred.

### 9.3 What the build found

The numbers behind the defaults are in `MEASUREMENT.md` and its interpretation. Two sources of findings
are worth remembering because the unit tests, with their fakes, could not see them:

- **ClickHouse 24.1**, through the gated integration suite: a TTL on a `DateTime64` column is refused; an
  alias equal to a column name is substituted into the other expressions of its `SELECT`; a column read in
  an outer statement must be passed through the inner one; clickhouse-connect streams same-typed columns
  as a plain two-dimensional array; rows of one run share `recorded_at`, so the status must break the tie.
- **The implementation review**: deleted blocks were not subtracted from the conflict layer; a late block
  inside the horizon was still used as a training negative; a block-only cell could get a score of 0; the
  TypeSafe SDK retried outside the budget and the concurrency limit was not enforced; paid answers were
  held in memory until a batch ended; the model bytes were not flushed before the pointer named them; the
  cleanup used a mutation instead of a lightweight delete. All are fixed, each with a test that fails
  without the fix.
