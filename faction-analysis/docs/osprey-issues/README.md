# Deployment issues on `skywatch-osprey`

The sidecar never creates its tables and is deployed from the other repository. Two issues carry the
changes it needs. Their bodies are the files in this directory, read back from GitHub after creation.

| Issue | Title | Body |
|---|---|---|
| [skywatch-osprey#84](https://github.com/skywatch-bsky/skywatch-osprey/issues/84) | Add the `faction_analysis` tables for the faction-analysis sidecar (DDL and cleanup grants) | `01-clickhouse-tables.md` |
| [skywatch-osprey#85](https://github.com/skywatch-bsky/skywatch-osprey/issues/85) | Add the faction-analysis sidecar as a compose service | `02-compose-service.md` |

`01-clickhouse-tables.md` embeds `sql/001_faction_analysis.sql` unchanged. When that file changes, update
the issue and this copy together.
