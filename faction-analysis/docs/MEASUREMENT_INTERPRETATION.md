# Faction analysis: reading the measurement

The numbers are in `MEASUREMENT.md`, which `measure report` writes. This file is the reading of them and
the decision they lead to. It was written by hand on 2026-11-28.

## Where the time and the memory go

The extraction pulled 60 days in 180 statements, one per day and signal, in 20.0 minutes of wall time
(17.7 minutes of statement time). It removed 3,206,700 follows that a later delete had cancelled and folded
the rest into 93,800,621 directed account pairs, which are 90,731,819 undirected edges over 6,942,920
accounts. That is well below the 150–230M edges of the first estimate, because the window holds a follow
only while its create event is inside it.

A Leiden run spends about 23 seconds on building the igraph graph. The first local-moving pass over the
whole graph took about 9.4 minutes (564 s). The aggregation and refinement levels take the rest, 15 to 27
minutes depending on the resolution.

Peak memory is 17.7 to 17.8 GiB for every modularity run: the edge arrays and the igraph graph, plus the
Python list of float weights, which is released once the partition holds its own copy. The runs were made
on the development machine (62 GiB of memory, 32 cores), not on `sys2`. The deployed service needs about
20 GiB of memory for a run.

## Why CPM is not the default

At resolution 0.01 on weights rescaled to a mean of 1, CPM's local moving produced about 1.97 million
communities on its first level and did not converge in 60 minutes. The parent stopped it at 60.0 minutes
with a peak of 19.0 GiB. CPM compares edge density against the resolution, so each weight scale needs its
own resolution. Modularity needs none and finished every time.

## Stability decides the default

The same seed gives the same partition: the repeat of `modularity-1` with seed 0 is identical. Two seeds
agree less. The adjusted Rand index between two seeds is 0.882 at resolution 0.5 (25 and 26 factions) and
0.689 at resolution 1.0 (52 and 49 factions).

Both counts lie in the band of 20 to 300 factions. Gate C takes the configuration with the better seed
stability, so the default is modularity at resolution 0.5 in two iterations: 25 factions, 25.2 minutes,
17.7 GiB.

A weekly run starts from the previous image, which hides most of the seed difference from week to week.
The monthly cold start does not. At resolution 1.0 about a third of the co-membership structure changes
with the seed, and the lineage would record it as splits and merges.

The operator judged 52 factions reasonable as well. `FACTION_ANALYSIS_RESOLUTION=1.0` gives that partition.

In every run 19,533 accounts stayed unplaced: their community had fewer than 500 accounts and no
established neighbour.

## The labels

The 60-day window holds 45,169,478 replies and quotes. Of these:

| Followed by a block within | Interactions | Share |
|---|---|---|
| 15 minutes | 66,116 | 0.15% |
| 1 hour | 103,314 | 0.23% |
| 4 hours | 139,343 | 0.31% |
| 24 hours | 180,360 | 0.40% |

187,147 are followed by a block at any time. Gate B passes with five times its minimum.

The Jev stratum of 100,000 block positives per refresh fits in one window. Labels are cached and never
bought twice, so a monthly refresh pays only for the positives that are new to it.

## What the integration run against ClickHouse 24.1 found

The gated suite in `tests/test_integration_clickhouse.py` runs the sidecar against a local ClickHouse 24.1.
It rejected four things that the unit tests, with their fakes, could not see:

- A TTL on a `DateTime64` column is refused. Every TTL wraps its column in `toDateTime`.
- An alias equal to a column name is substituted into the other expressions of the same `SELECT`.
  `min(did) AS did` next to `uniqExact(did)` failed. The columns are now qualified (`a.did`), and
  `tests/test_queries.py` guards every statement.
- The interactions statement read `PostEmbedRecordWithMediaUri` in its outer part without passing it
  through the inner one.
- clickhouse-connect streams a block of same-typed columns as a plain two-dimensional array, with no
  column names. The repository read columns by name and now reads them by position.
