## What

Add the `faction-analysis` sidecar as a compose service. The image is built from
`./sidecars/faction-analysis`, which is the `faction-analysis` directory of the `osprey-sidecars`
repository.

The service needs the tables of #84 and one named volume.

## Service

Add to `services`:

```yaml
  faction-analysis:
    build:
      context: ./sidecars/faction-analysis
      dockerfile: Dockerfile
    container_name: faction-analysis
    depends_on:
      clickhouse:
        condition: service_healthy
      clickhouse-init:
        condition: service_completed_successfully
    environment:
      - OSPREY_CLICKHOUSE_HOST=clickhouse
      - OSPREY_CLICKHOUSE_PORT=8123
      - OSPREY_CLICKHOUSE_USER=default
      - OSPREY_CLICKHOUSE_PASSWORD=clickhouse
      - OSPREY_CLICKHOUSE_DB=default
      - FACTION_ANALYSIS_POLL_INTERVAL_SECONDS=3600
      - FACTION_ANALYSIS_MODEL_DIR=/data/models
    secrets:
      - typesafe_api_key
    volumes:
      - faction_analysis_models:/data/models
    mem_limit: 24g
    memswap_limit: 24g
    restart: unless-stopped
```

Add to `secrets` at the top level, and to the existing `volumes`:

```yaml
secrets:
  typesafe_api_key:
    environment: TYPESAFE_API_KEY

volumes:
  faction_analysis_models: {}
```

## Why each part

- **The volume must be named and persistent.** `/data/models` holds the hostility model and the
  promotion-intent file. The intent file records that a run is fully written and must be published. A run
  whose intent file is lost on a restart can be mistaken for a run that never finished.
- **The secret.** `secrets.typesafe_api_key.environment` takes the key from the environment of the host that
  runs `docker compose` (Compose 2.23 or later) and mounts it as `/run/secrets/typesafe_api_key`. The
  sidecar reads that file, or `TYPESAFE_API_KEY` in `environment`, whichever is set. Without a key the
  monthly model refresh buys no new TypeSafe labels. It can still train a model when enough labels are
  stored from earlier refreshes. With none stored, no model is promoted and the text columns of the
  conflict matrix stay NULL. The rest of the sidecar works.
- **Memory.** A run held 17.7 GiB at the peak on the production graph, in a measurement on a development
  machine. `mem_limit` of 24g leaves room for the interaction pass, which was not measured.
- **Time.** The first run has no previous image. On the production graph it needs about 20 minutes to
  extract the follow and repost edges and 25 minutes for the partition. The sidecar polls hourly and
  starts a run only when one is due: a monthly cold start every 28 days, a weekly warm start otherwise.
- **One instance.** The sidecar holds a lock file in `/data/models` for its whole lifetime and a second
  process on the same volume exits at once. Run one replica.
- **Defaults.** The configuration is the one the production measurement chose: Leiden with the modularity
  objective at resolution 0.5, seed 42, two iterations, a minimum faction size of 500, and a window of 60
  days. Every setting is an environment variable; the list is in the README of the sidecar.

## Load on ClickHouse

Every statement of the sidecar sets `max_threads` to 4, `max_execution_time` to 3600 seconds and
`max_memory_usage` to 20 GB. A run scans two monthly partitions of `osprey_execution_results` for the
graph and once more for the interactions.

## Documentation

Add a section for the sidecar to `docs/statistical-sidecars.md`, in the pattern of the others. The
operator-facing description, the tables, the formulas and the limits are in the README of the sidecar.
