# Faction analysis: production measurement

Date: 2026-11-28 · Source: `osprey_execution_results` on `sys2`

## Access path

A Python process cannot call the MCP query tool, so the measurement used the documented SSH fallback:
`ssh -o BatchMode=yes -o ConnectTimeout=10 sys2` running `sudo -n docker exec -i clickhouse clickhouse-client`
with `--readonly=1 --max_execution_time=60 --max_threads=4` and
`--max_memory_usage=10000000000`, one statement on standard input. Nothing was written on that host.
Every statement started with SELECT or WITH and ended with a LIMIT. Each LIMIT was an expected ceiling. No
statement reached its ceiling.

## Probes (one day)

Counts, flag values and column types. The sampled rows are left out: they carry account identifiers.

### block_flags

```
create	1	1	244769
delete	1	\N	6761
```

### collections

```
app.bsky.feed.post	create	3325171
app.bsky.feed.repost	create	2922402
app.bsky.graph.listitem	create	2613196
app.bsky.graph.follow	create	1799527
app.bsky.graph.follow	delete	291152
app.bsky.graph.block	create	244769
fm.teal.feed.play	create	224124
app.bsky.feed.post	delete	197850
app.bsky.feed.threadgate	create	186294
app.bsky.feed.repost	delete	126605
dev.sensorthings.observationBatch	update	107784
app.bsky.feed.postgate	create	107221
app.bsky.actor.profile	update	82903
\N	\N	60185
app.rocksky.scrobble	create	33782
dev.sensorthings.observationBatch	create	24833
site.standard.document	create	24360
app.bsky.actor.profile	create	23893
app.bsky.graph.listitem	delete	22006
place.stream.livestream	update	14902
place.stream.broadcast.origin	update	14863
fm.teal.feed.play	delete	13977
fm.teal.actor.status	update	13400
cx.vmx.matadisco	create	12287
games.gamesgamesgamesgames.contribution	create	11770
app.bsky.feed.generator	update	11661
app.bsky.actor.status	update	10247
is.currents.feed.save	create	10030
cloud.substratum.place.joinPackage	update	10027
cx.vmx.matadisco.test	delete	10000
app.rocksky.actor.status	update	8070
community.lexicon.media.book.edition	update	8061
app.bsky.actor.status	create	7918
app.rocksky.song	create	7882
app.bsky.actor.status	delete	7850
pub.techne.draft.source	update	7646
ai.afrilingua.feed.content	create	7294
app.bsky.actor.link	create	7004
games.gamesgamesgamesgames.contributionVerification	create	7000
games.gamesgamesgamesgames.contributionReview	create	6999
games.gamesgamesgamesgames.game	update	6968
so.sprk.feed.post	create	6904
app.bsky.graph.block	delete	6761
link.bridgebeats.lookup	update	6400
com.publicdomainrelay.temp.market.heartbeat	update	5762
app.bsky.graph.list	update	5147
app.bsky.graph.starterpack	update	5041
rsvp.atmo.eventStarterPack	update	4895
town.delve.feed.post	create	4182
app.studynext.settings	update	3812
app.bsky.feed.post	update	3764
com.puzzmo.streak	update	3736
site.standard.document	update	3702
net.atmowx.observation	create	3479
org.tangled.git.refs	update	3473
app.rocksky.album	create	3407
tech.transparencia.news.article	create	3207
town.delve.feed.like	create	2731
app.bsky.graph.list	delete	2625
so.sprk.feed.post	delete	2567
```

### columns

```
AtUri	Nullable(String)
BlockSubjectDid	Nullable(String)
Collection	Nullable(String)
FacetTagList	Array(String)
FollowSubjectDid	Nullable(String)
FollowSubjectVia	Nullable(String)
Handle	Nullable(String)
IsOrganicFollow	Nullable(UInt8)
IsRepost	Nullable(UInt8)
IsStarterPackFollow	Nullable(UInt8)
OperationKind	Nullable(String)
Path	Nullable(String)
PostEmbedRecordUri	Nullable(String)
PostEmbedRecordWithMediaUri	Nullable(String)
PostIsReply	Nullable(UInt8)
PostIsSelfReply	Nullable(UInt8)
PostLanguages	Array(String)
PostReplyParent	Nullable(String)
PostReplyParentDid	Nullable(String)
PostReplyRoot	Nullable(String)
PostText	Nullable(String)
RepostSubjectDid	Nullable(String)
UserId	Nullable(String)
__timestamp	DateTime64(3)
```

### day_hours

```
2026-11-27 00:00:00	575898
2026-11-27 01:00:00	582323
2026-11-27 02:00:00	468016
2026-11-27 03:00:00	561179
2026-11-27 04:00:00	492214
2026-11-27 05:00:00	456731
2026-11-27 06:00:00	425065
2026-11-27 07:00:00	438205
2026-11-27 08:00:00	453939
2026-11-27 09:00:00	419121
2026-11-27 10:00:00	332623
2026-11-27 11:00:00	368088
2026-11-27 12:00:00	405915
2026-11-27 13:00:00	443938
2026-11-27 14:00:00	484150
2026-11-27 15:00:00	512774
2026-11-27 16:00:00	541281
2026-11-27 17:00:00	536122
2026-11-27 18:00:00	521271
2026-11-27 19:00:00	718202
2026-11-27 20:00:00	759430
2026-11-27 21:00:00	780018
2026-11-27 22:00:00	787527
2026-11-27 23:00:00	759742
```

### follow_flags

```
create	0	1	1	1589768
delete	-1	1	\N	291152
create	1	1	1	209759
```

### media_embed_flags

```
app.bsky.feed.post	34546
app.bsky.feed.post	app.bsky.feed.post	13
```

### post_flags

```
0	0	\N		0	1684913
1	0	1		0	1170922
1	1	1		0	260574
0	0	\N	app.bsky.feed.post	0	163577
0	0	\N		1	34024
1	0	1	app.bsky.feed.post	0	6639
1	1	1	app.bsky.feed.post	0	3737
1	1	1		1	269
1	0	1		1	253
0	0	\N	app.bsky.graph.starterpack	0	113
0	0	\N	app.bsky.feed.generator	0	38
1	0	1	app.bsky.graph.starterpack	0	30
1	0	1	app.bsky.feed.generator	0	16
0	0	\N	app.bsky.graph.list	0	15
0	0	\N	app.bsky.feed.post	1	13
1	0	1	app.bsky.graph.list	0	13
1	1	1	app.bsky.feed.generator	0	11
1	1	1	app.bsky.graph.list	0	7
1	1	1	app.bsky.graph.starterpack	0	7
```

### repost_flags

```
create	1	1	2922402
delete	\N	-1	126605
update	1	1	11
```

## Extraction

```json
{
  "accounts": 6942920,
  "deleted_follows_removed": 3206700,
  "directed_pairs": 93800621,
  "follow_creates": 53492369,
  "follow_deletes": 10618857,
  "median_in_strength": 2.0,
  "repost_creates": 101001019,
  "rescale": 0.26046250105465907,
  "statement_seconds": 1059.2651992449537,
  "statements": 180,
  "undirected_edges": 90731819,
  "wall_seconds": 1196.0751546290703
}
```

## Leiden grid

Each run built the graph from the extracted edges, ran Leiden once, and absorbed communities below the minimum
size (20 to 300 factions is the target).

| Configuration | Run | Graph build min | Total min | Peak GiB | Edges | Communities above minimum | Factions | Unplaced | Modularity |
|---|---|---|---|---|---|---|---|---|---|
| modularity-0.5 | seed +0 | 0.4 | 25.2 | 17.70 | 90,731,819 | 25 | 25 | 19,533 | 0.6048 |
| modularity-0.5 | seed +1 | 0.4 | 26.4 | 17.72 | 90,731,819 | 26 | 26 | 19,533 | 0.6003 |
| modularity-1 | repeat | 0.4 | 36.5 | 17.76 | 90,731,819 | 52 | 52 | 19,533 | 0.6529 |
| modularity-1 | seed +0 | 0.4 | 32.7 | 17.75 | 90,731,819 | 52 | 52 | 19,533 | 0.6529 |
| modularity-1 | seed +1 | 0.4 | 30.7 | 17.75 | 90,731,819 | 49 | 49 | 19,533 | 0.6521 |
| cpm-0.01 | seed +0 | | 60.0 | 19.04 | stopped | | | | |

## Seed stability and repeat

The adjusted Rand index of two partitions over the accounts both placed. 1.0 is the same partition.

| Comparison | Accounts placed in both | Adjusted Rand index | Identical |
|---|---|---|---|
| modularity-0.5: seed 0 against seed 1 | 6,923,387 | 0.882 | no |
| modularity-1: seed 0 against its repeat | 6,923,387 | 1.000 | yes |
| modularity-1: seed 0 against seed 1 | 6,923,387 | 0.689 | no |

## Label audit

```json
{
  "interactions": 45169478,
  "positive_15m": 66116,
  "positive_1h": 103314,
  "positive_24h": 180360,
  "positive_4h": 139343,
  "with_block": 187147
}
```


## Gates

- **Gate A** (a single Leiden run within 60 minutes and 20 GiB): passed. 5 runs finished inside the limits; cpm-0.01 seed+0 stopped at 60 min and 19.0 GiB.
- **Gate B** (at least 20,000 block positives at one hour in the window): passed.
- **Gate C** (a configuration with 20 to 300 factions, with the better seed stability when several lie in the range): passed: `modularity-0.5`. `modularity-0.5` gave 25 factions and its seeds agree at 0.882. `modularity-1` gave 52 factions and its seeds agree at 0.689.

## Reading the numbers

This file holds what was measured and is written by `measure report`. The hand-written reading of these
numbers, with the decision they lead to, is in `docs/MEASUREMENT_INTERPRETATION.md`.
