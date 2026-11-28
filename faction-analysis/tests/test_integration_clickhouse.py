"""The pipeline against a real ClickHouse of the production version.

These tests are gated on ``FACTION_ANALYSIS_INTEGRATION_CH_HOST``. ``docker/compose.test.yml`` starts the
server. Each test gets an empty database that holds the sidecar's tables and a small source table, seeds
the two-faction world, and drives the real repository. ``DelayingClient`` stands in for a commit that is
late or whose response is lost, and the fault hook of the context stands in for the death of the process.
"""

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from faction_analysis import db as db_module
from faction_analysis import export_dataset, lock, model_store, pipeline, schema
from faction_analysis.config import AppConfig
from faction_analysis.db import ClickHouseClient, FactionAnalysisDb
from faction_analysis.graph import drop_deleted_creates
from faction_analysis.lock import CrashPoint, SimulatedCrash
from faction_analysis.model import FEATURE_CONFIG, TrainingRows, train
from faction_analysis.promotion import IntentUnreadable
from faction_analysis.queries import Window
from faction_analysis.runstate import RunRow
from tests import integration_world as world
from tests.conftest import fabricated_rows

pytestmark = pytest.mark.integration

INTENT = lock.INTENT_NAME
NOW = datetime(2026, 11, 27, 12, tzinfo=UTC)
# A cell with two hostile events of six contacts, against a prior of 20: (2 + 20 * 2/6) / (3 + 20) / (2/6).
PRIOR_LIFT = pytest.approx(1.130435, abs=1e-5)
WRITE_STAGES = {
    'after_write_members': 'write_members',
    'after_write_features': 'write_features',
    'after_write_conflict': 'write_conflict',
    'after_write_registry': 'write_registry',
}


def run_id_at(moment: datetime, suffix: str) -> str:
    return f'{int(moment.timestamp() * 1000):013d}.{suffix}'


def stored(moment: datetime) -> Any:
    """A moment as ClickHouse hands it back: the column is DateTime64(3, 'UTC') and arrives without a zone."""
    return pd.Timestamp(moment.replace(tzinfo=None))


def cell(cells: pd.DataFrame, window: int, source: int, target: int) -> pd.Series:
    """The one cell of ``window`` days from ``source`` to ``target``."""
    rows = cells.loc[
        (cells['window_days'] == window) & (cells['from_faction_id'] == source) & (cells['to_faction_id'] == target)
    ]
    assert len(rows) == 1, f'expected one cell {window} {source} -> {target}, found {len(rows)}'
    return rows.iloc[0]


def image_of(live: 'Live', run_id: str | None) -> dict[str, int]:
    assert run_id is not None, 'the outcome names no run'
    members = live.rows(schema.MEMBERS, run_id)
    return dict(zip(members['did'], members['faction_id'].astype(int), strict=True))


class Live:
    """The real repository over a scratch database, a model directory and a clock the tests move."""

    def __init__(self, scratch: Any, tmp_path: Path, **overrides: str) -> None:
        self.scratch = scratch
        self.now = scratch.now
        self.client = world.DelayingClient(scratch.sidecar)
        self.model_dir = tmp_path / 'models'
        self.model_dir.mkdir()
        values = {
            'OSPREY_CLICKHOUSE_DB': scratch.database,
            'FACTION_ANALYSIS_SOURCE_TABLE': scratch.source,
            'FACTION_ANALYSIS_OBJECTIVE': 'modularity',
            'FACTION_ANALYSIS_RESOLUTION': '1.0',
            'FACTION_ANALYSIS_N_ITERATIONS': '-1',
            'FACTION_ANALYSIS_MIN_FACTION_SIZE': '3',
            'FACTION_ANALYSIS_CONFIDENCE_SEEDS': '1',
            'FACTION_ANALYSIS_PILEON_MIN_REPLIERS': '2',
            'FACTION_ANALYSIS_SWARM_MIN_QUOTERS': '2',
            'FACTION_ANALYSIS_MIN_TEXT_SCORED': '1',
        }
        self.config = AppConfig.from_mapping({**values, **overrides})
        self.repo = FactionAnalysisDb(
            self.config.clickhouse, self.config.query, client=cast(ClickHouseClient, self.client)
        )
        self.dids: list[str] = []
        self.restart()

    def restart(self) -> None:
        """The next start of the process over the same database and the same model directory."""
        self.ctx = pipeline.RunContext(self.config, self.repo, self.model_dir, now=lambda: self.now)

    def seed(self) -> None:
        self.dids = world.seed(self.scratch.client, self.scratch.source, self.now)

    def poll(self, kind: pipeline.RunKind | None = None) -> pipeline.RunOutcome:
        return pipeline.poll(self.ctx, kind=kind)

    def advance(self, days: int) -> None:
        self.now += timedelta(days=days)

    def install_model(self) -> str:
        rows: TrainingRows = fabricated_rows(jev_rows=300, weak_rows=60, random_rows=60)
        payload = model_store.serialise(train(rows, seed=1))
        manifest = model_store.ModelManifest(
            version='h-it-1',
            sha256=hashlib.sha256(payload).hexdigest(),
            trained_at=self.now.isoformat(),
            dataset_version='ds-it',
            label_policy_hash='policy',
            feature_config=dict(FEATURE_CONFIG),
            row_counts={'train': 1},
            metrics={},
            calibrated=True,
        )
        model_store.promote(self.model_dir, payload, manifest)
        return manifest.version

    # reads straight from the tables

    def rows(self, noun: str, run_id: str | None = None) -> pd.DataFrame:
        """Every row of a table, with or without a ``run_id`` filter, exactly as it is stored."""
        table = f'{self.scratch.database}.{self.config.query.table(noun)}'
        where = '' if run_id is None else f" WHERE run_id = '{run_id}'"
        return self.scratch.client.query_df(f'SELECT * FROM {table}{where}')

    def snapshot(self, noun: str, run_id: str) -> pd.DataFrame:
        """A table's rows of one run as text in a fixed order, so that two reads compare column by column.

        Rows that share the sort key come back in any order, and the run rows of one run all do.
        """
        frame = self.rows(noun, run_id).astype(str)
        return frame.sort_values(list(frame.columns), kind='stable').reset_index(drop=True)

    def statuses(self, run_id: str | None) -> list[str]:
        assert run_id is not None, 'the outcome names no run'
        return [row.status for row in self.repo.fetch_run_rows(run_id)]

    def promoted_ids(self) -> list[str]:
        return sorted(run_id for run_id, state in self.repo.fetch_run_states().items() if state.promoted)

    def run_row(self, run_id: str | None) -> RunRow:
        """The promoted row of a run, which carries the counts of the finished run."""
        assert run_id is not None, 'the outcome names no run'
        rows = [row for row in self.repo.fetch_run_rows(run_id) if row.status == 'promoted']
        assert rows, f'{run_id} has no promoted row'
        return rows[-1]

    def intent(self) -> RunRow | None:
        record = lock.load(self.model_dir)
        return None if record is None else record.run

    def intended_run(self) -> RunRow:
        """The run the intent file names. Fails when there is no intent file."""
        run = self.intent()
        assert run is not None, 'the intent file is missing'
        return run

    # writes by hand

    def frame(self, noun: str, **columns: Any) -> pd.DataFrame:
        """A frame with exactly the columns of a table, in their order."""
        return pd.DataFrame({name: columns[name] for name in schema.COLUMNS[noun]})

    def write_rows(self, noun: str, run_id: str, **columns: Any) -> None:
        """Write one hand-made row of a run-owned table. A column that is not given holds 0."""
        row: dict[str, Any] = {name: [0] for name in schema.COLUMNS[noun]}
        row.update({name: [value] for name, value in columns.items()})
        row['run_id'] = [run_id]
        self.repo.insert_frame(noun, pd.DataFrame(row)[list(schema.COLUMNS[noun])])


def crash_at(point: str) -> Any:
    def crash(where: str) -> None:
        if where == point:
            raise SimulatedCrash(where)

    return crash


def fail_at(point: str) -> Any:
    def fail(where: str) -> None:
        if where == point:
            raise RuntimeError('injected')

    return fail


def assert_consistent_first_image(live: Live, outcome: pipeline.RunOutcome) -> RunRow:
    """What the first promoted run of a seeded world always holds."""
    assert outcome.status == 'promoted', outcome.reason
    assert outcome.run_id is not None
    members = live.rows(schema.MEMBERS, outcome.run_id)
    assert len(members) == 12 and sorted(members['did'].tolist()) == sorted(world.account_dids())
    image = image_of(live, outcome.run_id)
    assert {image[did] for did in live.dids[:6]} == {image[live.dids[0]]}
    assert {image[did] for did in live.dids[6:]} == {image[live.dids[6]]}
    assert image[live.dids[0]] != image[live.dids[6]]
    run = live.run_row(outcome.run_id)
    assert (run.factions, run.next_faction_id, run.accounts_placed) == (2, 3, 12)
    assert not (live.model_dir / INTENT).exists()
    return run


# ---- the schema ----


def test_the_ddl_carries_a_ttl_for_every_table_but_runs(scratch: Any) -> None:
    tables = scratch.client.query_df(
        f"SELECT name, create_table_query FROM system.tables WHERE database = '{scratch.database}' "
        "AND name LIKE 'faction_analysis_%' ORDER BY name"
    )
    assert len(tables) == 11
    for name, statement in zip(tables['name'], tables['create_table_query'], strict=True):
        assert ('TTL' in statement.upper()) == (name != 'faction_analysis_runs'), name
        partitioned = name in {'faction_analysis_members', 'faction_analysis_member_features'}
        assert ('PARTITION BY run_id' in statement) == partitioned, name


# ---- the scenario of the plan: full, then a weekly run with a split, then a weekly run with a model ----


def test_end_to_end_full_then_weekly(scratch: Any, tmp_path: Path) -> None:
    live = Live(scratch, tmp_path)
    live.seed()
    first = live.poll('full')
    run = assert_consistent_first_image(live, first)
    assert run.kind == 'full'
    assert (run.births, run.merges, run.splits, run.deaths, run.moves) == (2, 0, 0, 0, 12)
    assert run.modularity is not None and run.modularity > 0.4
    assert (run.agreement_prev, run.source_gap_days, run.hash_collisions, run.accounts_unplaced) == (None, 0, 0, 0)
    assert (run.text_model_version, run.text_coverage) == ('', 0.0)
    assert live.statuses(first.run_id) == ['running', 'promoting', 'promoted']
    assert live.promoted_ids() == [first.run_id]
    # The run row is completed before the promotion, so it holds every stage up to the registry.
    assert set(run.stage_seconds) >= {
        'extract_graph',
        'partition',
        'track',
        'extract_interactions',
        'write_registry',
    }

    image = image_of(live, first.run_id)
    art, music = image[live.dids[0]], image[live.dids[6]]
    lineage = live.rows(schema.LINEAGE, first.run_id)
    assert sorted(lineage['event'].tolist()) == ['birth', 'birth']
    assert sorted(lineage['faction_id'].tolist()) == sorted([art, music])
    moves = live.rows(schema.MOVES, first.run_id)
    assert len(moves) == 12 and set(moves['move_kind'].tolist()) == {'assign'}
    assert moves['from_faction_id'].isna().all() and set(moves['to_faction_id'].dropna().astype(int)) == {art, music}
    members = live.rows(schema.MEMBERS, first.run_id)
    assert members['prev_faction_id'].isna().all() and members['confidence'].notna().all()
    assert (members['since_run_id'] == first.run_id).all()
    assert (members['retire_at'] == stored(NOW + timedelta(days=395))).all()

    registry = live.rows(schema.FACTIONS, first.run_id).set_index('faction_id')
    assert sorted(registry.index.tolist()) == sorted([art, music])
    assert registry['status'].eq('active').all() and registry['parent_faction_id'].isna().all()
    assert (registry['born_run_id'] == first.run_id).all()
    assert sorted(registry.loc[art, 'top_tags']) == ['art', 'ink'] and list(registry.loc[music, 'top_tags']) == [
        'music'
    ]
    assert sorted(registry.loc[art, 'top_languages']) == ['de', 'en']
    assert sorted(registry.loc[music, 'top_languages']) == ['de', 'en']
    for faction in (art, music):
        handles = [str(handle) for handle in registry.loc[faction, 'sample_handles']]
        assert handles and len(handles) <= 6, faction
        indexes = set()
        for handle in handles:
            account = handle.removeprefix('user').split('.')[0]
            assert handle.endswith('.bsky.social') and account.isdigit(), handle
            indexes.add(int(account))
        expected = set(range(0, 6)) if faction == art else set(range(6, 12))
        assert indexes <= expected, (faction, indexes)
    statistics = live.rows(schema.FACTION_RUNS, first.run_id).set_index('faction_id')
    assert statistics['member_count'].to_dict() == {art: 6, music: 6}
    assert (statistics['hostile_sent'].to_dict(), statistics['hostile_received'].to_dict()) == (
        {art: 3, music: 2},
        {art: 2, music: 3},
    )
    assert statistics['conflict_cells'].to_dict() == {art: 1, music: 1}

    features = live.rows(schema.MEMBER_FEATURES, first.run_id).set_index('did')
    assert features['blocks_sent'].sum() == 1 and features['blocks_received'].sum() == 1
    assert features['pileon_replies_sent'].sum() == 2 and features['pileon_replies_received'].sum() == 2
    assert features['swarm_quotes_sent'].sum() == 2 and features['swarm_quotes_received'].sum() == 2
    assert features['within_z'].notna().all() and features['participation'].notna().all()
    assert features.loc[live.dids[0], 'blocks_sent'] == 1 and features.loc[live.dids[2], 'blocks_sent'] == 0
    assert features.loc[live.dids[6], 'pileon_replies_received'] == 2
    assert features.loc[live.dids[1], 'swarm_quotes_received'] == 2
    assert features.loc[live.dids[7], 'blocks_received'] == 0

    # Hand-computed. Faction ``art`` sends three replies to ``music`` and ``music`` one reply and two quotes
    # back. Accounts 0, 1 and 3 reply and account 0 blocks: three sources on one target. The block of account
    # 2 was deleted a minute after it was made and does not count.
    cells = live.rows(schema.CONFLICT, first.run_id)
    assert sorted(cells['window_days'].unique().tolist()) == [7, 60]
    for window in (7, 60):
        forward = cell(cells, window, art, music)
        backward = cell(cells, window, music, art)
        assert (forward['contact'], forward['contact_expected'], forward['contact_lift']) == (3, 1.5, 2.0)
        assert (forward['pileon_replies'], forward['pileon_lift'], forward['direction_pileon']) == (2, PRIOR_LIFT, 1.0)
        # One block of six contacts is exactly the global block rate, so its shrunk lift is 1.1304 as well.
        assert (forward['blocks'], forward['blocks_lift'], forward['direction_blocks']) == (1, PRIOR_LIFT, 1.0)
        assert forward['swarm_quotes'] == 0 and forward['direction_swarm'] == -1.0
        assert (backward['swarm_quotes'], backward['swarm_lift'], backward['direction_swarm']) == (2, PRIOR_LIFT, 1.0)
        assert backward['pileon_replies'] == 0 and backward['blocks'] == 0
        assert (forward['distinct_sources'], forward['distinct_targets']) == (3, 1)
        assert (backward['distinct_sources'], backward['distinct_targets']) == (3, 2)
        assert forward['conflict_score'] == PRIOR_LIFT and backward['conflict_score'] == PRIOR_LIFT
        # No model, so every text column is empty.
        assert (forward['text_hostility'], forward['text_scored']) == (0.0, 0)
        assert pd.isna(forward['text_lift']) and pd.isna(forward['direction_text'])
    assert not (cells['from_faction_id'] == cells['to_faction_id']).any()
    assert len(cells) == 4

    # A week later the first faction splits. The half with the reposts keeps the ID.
    world.split_first_clique(scratch.client, scratch.source, live.now)
    live.advance(7)
    second = live.poll()
    assert second.status == 'promoted', second.reason
    assert second.kind == 'weekly' and second.run_id is not None
    after = image_of(live, second.run_id)
    child = after[live.dids[3]]
    assert child not in {art, music}
    assert {after[did] for did in live.dids[:3]} == {art}
    assert {after[did] for did in live.dids[3:6]} == {child}
    assert {after[did] for did in live.dids[6:]} == {music}
    lineage = live.rows(schema.LINEAGE, second.run_id)
    assert len(lineage) == 1
    assert (lineage['faction_id'].iloc[0], lineage['event'].iloc[0]) == (child, 'split')
    assert lineage['related_faction_id'].iloc[0] == art
    moves = live.rows(schema.MOVES, second.run_id).sort_values('did')
    assert len(moves) == 3 and set(moves['move_kind'].tolist()) == {'move'}
    assert moves['did'].tolist() == sorted(live.dids[3:6])
    assert (moves['from_faction_id'].astype(int) == art).all() and (moves['to_faction_id'].astype(int) == child).all()
    members = live.rows(schema.MEMBERS, second.run_id)
    assert (members['retire_at'] == stored(NOW + timedelta(days=7, weeks=8))).all()
    registry = live.rows(schema.FACTIONS, second.run_id).set_index('faction_id')
    assert registry.loc[child, 'parent_faction_id'] == art and registry.loc[child, 'status'] == 'active'
    run = live.run_row(second.run_id)
    assert (run.factions, run.splits, run.births, run.deaths, run.next_faction_id) == (3, 1, 0, 0, 4)
    assert run.agreement_prev is not None and run.agreement_prev < 1.0

    # A week later again, with a model on disk. The texts are scored once and attributed by this image.
    live.advance(7)
    version = live.install_model()
    third = live.poll()
    assert third.status == 'promoted', third.reason
    assert third.kind == 'weekly' and third.run_id is not None
    run = live.run_row(third.run_id)
    assert (run.text_model_version, run.next_faction_id) == (version, 4)
    assert 0 < run.text_coverage < 1
    scores = live.rows(schema.TEXT_SCORES)
    assert len(scores) == 5 and scores['uri_hash'].is_unique
    assert scores['model_version'].eq(version).all()
    examples = scores['is_example'] == 1
    # A text is an example when its hostility is high or when its hash falls in the sample.
    assert ((scores['h'] >= 0.5) <= examples).all()
    sampled = (scores['uri_hash'].astype(int) % 200) == 0
    assert ((examples) <= ((scores['h'] >= 0.5) | sampled)).all()
    assert scores.loc[examples, 'post_text'].notna().all() and scores.loc[~examples, 'post_text'].isna().all()
    assert scores.loc[examples, 'author_did'].isin(live.dids).all()
    assert scores.loc[~examples, 'author_did'].isna().all()
    assert set(scores['kind'].tolist()) <= {'reply', 'quote'}
    # The interactions are fifteen days old now, so only the long window holds them.
    cells = live.rows(schema.CONFLICT, third.run_id)
    assert sorted(cells['window_days'].unique().tolist()) == [60]
    assert cells['text_scored'].sum() == 5 and cells['text_lift'].notna().any()
    forward = cell(cells, 60, art, music)
    assert forward['contact'] == 2 and forward['text_scored'] == 2 and forward['pileon_replies'] == 2
    assert cell(cells, 60, child, music)['contact'] == 1

    # A fourth run scores nothing again.
    live.advance(7)
    fourth = live.poll()
    assert fourth.status == 'promoted', fourth.reason
    assert fourth.run_id is not None and second.run_id is not None and third.run_id is not None
    assert first.run_id is not None
    assert len(live.rows(schema.TEXT_SCORES)) == 5
    assert live.promoted_ids() == sorted([first.run_id, second.run_id, third.run_id, fourth.run_id])
    assert len(live.rows(schema.MOVES, fourth.run_id)) == 0
    assert len(live.rows(schema.LINEAGE, fourth.run_id)) == 0


def test_interactions_uppercase_matches_ascii_count(scratch: Any, tmp_path: Path) -> None:
    live = Live(scratch, tmp_path)
    live.seed()
    run = live.poll('full')
    assert run.status == 'promoted', run.reason
    assert run.run_id is not None
    row = live.run_row(run.run_id)
    interactions = pd.concat(live.repo.stream_interactions(Window(row.window_start, row.window_end)), ignore_index=True)
    expected = [sum('A' <= char <= 'Z' for char in text) for text in interactions['text'].fillna('')]
    assert interactions['uppercase'].astype(int).tolist() == expected
    assert any(expected)


def test_classifier_lookup_chunks_fit_clickhouse_query_size(scratch: Any, tmp_path: Path) -> None:
    live = Live(scratch, tmp_path)
    live.seed()
    first = live.poll('full')
    assert first.status == 'promoted', first.reason
    row = live.run_row(first.run_id)
    window = Window(row.window_start, row.window_end)
    hashes = list(range(2**63, 2**63 + db_module.HASH_CHUNK))
    frame = live.repo.fetch_interaction_texts(window, hashes)
    assert isinstance(frame, pd.DataFrame)
    uris = [f'at://did:plc:{index:024x}/app.bsky.feed.post/{index:040d}' for index in range(db_module.URI_CHUNK)]
    source = world.Source()
    source.add(
        world.recent_ms(live.now),
        Collection=world.POST,
        OperationKind=world.CREATE,
        UserId=world.account_dids()[0],
        AtUri=uris[0],
        PostText='parent text',
    )
    scratch.client.insert(scratch.source, source.as_rows(), column_names=list(world.SOURCE_COLUMNS))
    parents = live.repo.fetch_parent_texts(window, uris)
    assert isinstance(parents, dict)


def test_extraction_to_membership_preserves_dids(scratch: Any, tmp_path: Path) -> None:
    live = Live(scratch, tmp_path)
    live.seed()
    outcome = live.poll('full')
    assert outcome.status == 'promoted', outcome.reason
    members = live.rows(schema.MEMBERS, outcome.run_id)
    assert sorted(members['did'].tolist()) == sorted(world.account_dids())
    assert members['did'].is_unique and len(members) == 12

    # The graph leaves out the deleted follow and the starter-pack follow.
    row = live.run_row(outcome.run_id)
    window = Window(row.window_start, row.window_end)
    blocks = list(live.repo.stream_follow_creates(window))
    src = np.concatenate([block['src'] for block in blocks])
    dst = np.concatenate([block['dst'] for block in blocks])
    uri = np.concatenate([block['uri_hash'] for block in blocks])
    event_ms = np.concatenate([block['event_ms'] for block in blocks])
    # Thirty follow creates per clique and two across: the live one and the one deleted later. The query
    # itself leaves out the starter-pack follow.
    assert len(src) == 62
    src, dst, removed = drop_deleted_creates(src, dst, uri, event_ms, live.repo.fetch_deletes(window, world.FOLLOW))
    assert removed == 1
    accounts = live.repo.fetch_account_hashes(window)
    # The hashes are UInt64: kept as Python integers they never pass through a signed 64-bit integer.
    hash_of = dict(zip(accounts['did'], accounts['h'].tolist(), strict=True))
    assert len(accounts) == 12 and accounts['dids'].eq(1).all()
    edges = {(int(a), int(b)) for a, b in zip(src, dst, strict=True)}
    assert (hash_of[live.dids[6]], hash_of[live.dids[0]]) in edges
    assert (hash_of[live.dids[0]], hash_of[live.dids[6]]) not in edges
    assert (hash_of[live.dids[1]], hash_of[live.dids[11]]) not in edges


def test_cached_scores_follow_a_split(scratch: Any, tmp_path: Path) -> None:
    live = Live(scratch, tmp_path)
    live.seed()
    live.install_model()
    first = live.poll('full')
    assert first.status == 'promoted', first.reason
    assert len(live.rows(schema.TEXT_SCORES)) == 5
    art, music = image_of(live, first.run_id)[live.dids[0]], image_of(live, first.run_id)[live.dids[6]]
    before = live.rows(schema.CONFLICT, first.run_id)
    hostility = {
        (int(source), int(target)): hostility
        for source, target, hostility in before.loc[
            before['window_days'] == 60, ['from_faction_id', 'to_faction_id', 'text_hostility']
        ].itertuples(index=False)
    }

    # After the split the two replies of accounts 0 and 1 leave faction ``art`` and the reply without text
    # leaves the new faction. No text is scored again.
    world.split_first_clique(scratch.client, scratch.source, live.now)
    live.advance(7)
    second = live.poll()
    assert second.status == 'promoted', second.reason
    assert len(live.rows(schema.TEXT_SCORES)) == 5
    after = image_of(live, second.run_id)
    child = after[live.dids[3]]
    cells = live.rows(schema.CONFLICT, second.run_id)
    assert sorted(cells['window_days'].unique().tolist()) == [60]
    from_art = cell(cells, 60, art, music)
    assert (from_art['contact'], from_art['text_scored']) == (2, 2)
    assert from_art['text_hostility'] == pytest.approx(hostility[(art, music)])
    assert (cell(cells, 60, child, music)['contact'], cell(cells, 60, child, music)['text_scored']) == (1, 0)
    assert cell(cells, 60, music, art)['text_hostility'] == pytest.approx(hostility[(music, art)])
    assert cells['text_scored'].sum() == before.loc[before['window_days'] == 60, 'text_scored'].sum()


# ---- publication safety ----


@pytest.mark.parametrize(('point', 'stage'), sorted(WRITE_STAGES.items()))
def test_failed_run_after_each_write_stage_leaves_state_unchanged(
    scratch: Any, tmp_path: Path, point: str, stage: str
) -> None:
    live = Live(scratch, tmp_path)
    live.seed()
    good = live.poll('full')
    assert good.status == 'promoted', good.reason
    assert good.run_id is not None
    before = {noun: live.snapshot(noun, good.run_id) for noun in schema.RUN_OWNED if noun != schema.RUNS}
    good_rows = live.repo.fetch_run_rows(good.run_id)

    live.advance(7)
    live.ctx.fault = fail_at(point)
    outcome = live.poll()
    assert outcome.status == 'failed' and (outcome.stage, outcome.reason) == (stage, 'RuntimeError')
    assert outcome.run_id is not None and outcome.run_id != good.run_id
    assert live.statuses(outcome.run_id) == ['running', 'failed']
    failed = live.repo.fetch_run_rows(outcome.run_id)[-1]
    assert (failed.failed_stage, failed.error_class) == (stage, 'RuntimeError')
    # Nothing of the failed run is left in any table, and the good run is exactly as it was.
    for noun in schema.RUN_OWNED:
        if noun != schema.RUNS:
            assert live.rows(noun, outcome.run_id).empty, noun
    for noun, snapshot in before.items():
        pd.testing.assert_frame_equal(live.snapshot(noun, good.run_id), snapshot)
    assert live.repo.fetch_run_rows(good.run_id) == good_rows
    assert live.promoted_ids() == [good.run_id]
    assert not (live.model_dir / INTENT).exists()

    # The next poll retries at once and reaches a consistent state.
    live.ctx.fault = None
    retry = live.poll()
    assert retry.status == 'promoted', retry.reason
    run = assert_consistent_first_image(live, retry)
    assert retry.kind == 'weekly' and retry.run_id is not None
    assert (run.moves, run.births, run.next_faction_id) == (0, 0, 3)
    assert live.promoted_ids() == sorted([good.run_id, retry.run_id])
    assert live.snapshot(schema.MEMBERS, good.run_id).equals(before[schema.MEMBERS])


def test_retry_in_same_second_gets_distinct_run_id(scratch: Any, tmp_path: Path) -> None:
    live = Live(scratch, tmp_path)
    live.seed()
    first = live.poll('full')
    second = live.poll('full')
    assert first.status == second.status == 'promoted', (first.reason, second.reason)
    assert first.run_id is not None and second.run_id is not None
    assert first.run_id != second.run_id
    # The same clock gives the same epoch milliseconds; the random suffix tells the attempts apart.
    assert first.run_id.split('.')[0] == second.run_id.split('.')[0] == f'{int(NOW.timestamp() * 1000):013d}'
    assert live.promoted_ids() == sorted([first.run_id, second.run_id])
    # Two promoted runs, each with a whole image, and the second continued the IDs of the first.
    assert len(live.rows(schema.MEMBERS, first.run_id)) == 12 and len(live.rows(schema.MEMBERS, second.run_id)) == 12
    assert live.run_row(second.run_id).next_faction_id == 3
    assert len(live.rows(schema.MOVES, second.run_id)) == 0


def test_promotion_committed_but_response_lost(scratch: Any, tmp_path: Path) -> None:
    live = Live(scratch, tmp_path)
    live.seed()
    live.client.once = {'promoted': 'commit_then_fail'}
    outcome = live.poll('full')
    # The failure handler re-sent both rows, the read confirmed them, and the run counts as promoted.
    assert outcome.status == 'promoted' and outcome.exit_code == 0
    assert 'completed after RuntimeError' in outcome.reason
    run = assert_consistent_first_image(live, outcome)
    assert run.next_faction_id == 3
    # Rows of one run share their recorded time, so the status orders them.
    assert live.statuses(outcome.run_id) == ['running', 'promoting', 'promoting', 'promoted', 'promoted']
    assert live.promoted_ids() == [outcome.run_id]
    assert live.poll().status == 'none'


def test_delayed_commit_is_left_unresolved_and_nothing_is_removed(scratch: Any, tmp_path: Path) -> None:
    live = Live(scratch, tmp_path)
    live.seed()
    live.client.once = {'promoting': 'hold'}
    live.client.plan = {'promoting': 'fail', 'promoted': 'fail'}
    outcome = live.poll('full')
    assert outcome.status == 'unresolved' and outcome.exit_code == 1
    assert outcome.run_id is not None
    run_id = outcome.run_id
    assert (live.model_dir / INTENT).is_file() and live.intent() is not None
    # The first promoting row has not committed, so the table shows only the running row.
    assert live.statuses(run_id) == ['running']
    assert len(live.rows(schema.MEMBERS, run_id)) == 12
    assert len(live.rows(schema.CONFLICT, run_id)) == 4

    # While the server cannot be reached, a restart completes nothing and starts no run.
    live.restart()
    with pytest.raises(RuntimeError):
        live.poll()
    assert sorted(live.repo.fetch_run_states()) == [run_id]

    # The server is back: the next start completes the run from the intent file.
    live.client.plan = {}
    live.restart()
    assert live.poll().status == 'none'
    assert not (live.model_dir / INTENT).exists()
    assert live.statuses(run_id) == ['running', 'promoting', 'promoted']
    assert len(image_of(live, run_id)) == 12

    # The delayed promoting row commits now. It only adds a row that readers ignore.
    assert live.client.release() == 1
    assert live.statuses(run_id) == ['running', 'promoting', 'promoting', 'promoted']
    assert live.promoted_ids() == [run_id]
    assert live.run_row(run_id).next_faction_id == 3
    assert len(live.rows(schema.MEMBERS, run_id)) == 12
    assert live.poll().status == 'none'


def test_unresolved_run_blocks_new_runs(scratch: Any, tmp_path: Path) -> None:
    live = Live(scratch, tmp_path)
    live.seed()
    live.client.plan = {'promoted': 'fail'}
    outcome = live.poll('full')
    assert outcome.status == 'unresolved' and outcome.run_id is not None
    run_id = outcome.run_id
    # The failure handler re-sent the promoting row before the promoted insert failed again.
    assert live.statuses(run_id) == ['running', 'promoting', 'promoting']
    assert (live.model_dir / INTENT).is_file()
    with pytest.raises(RuntimeError):
        live.poll()
    assert sorted(live.repo.fetch_run_states()) == [run_id]
    assert live.promoted_ids() == []

    # The IDs the unfinished run reserved are issued to no other run.
    reserved = live.intent()
    assert reserved is not None and reserved.next_faction_id == 3
    live.client.plan = {}
    assert live.poll().status == 'none'
    assert live.promoted_ids() == [run_id]
    assert live.run_row(run_id).next_faction_id == 3
    live.advance(7)
    following = live.poll()
    assert following.status == 'promoted', following.reason
    assert following.run_id is not None
    assert live.run_row(following.run_id).next_faction_id == 3
    assert set(image_of(live, following.run_id).values()) == {1, 2}


def test_crash_after_promoting_recovers_on_restart(scratch: Any, tmp_path: Path) -> None:
    live = Live(scratch, tmp_path)
    live.seed()
    live.ctx.fault = crash_at('after_promoting_row')
    with pytest.raises(SimulatedCrash):
        live.poll('full')
    run_id = live.intended_run().run_id
    assert live.statuses(run_id) == ['running', 'promoting']
    assert live.intended_run().next_faction_id == 3

    # The next start completes the same run, with its outputs and its faction IDs.
    live.restart()
    assert live.poll().status == 'none'
    assert not (live.model_dir / INTENT).exists()
    statuses = live.statuses(run_id)
    assert statuses[0] == 'running' and statuses[-1] == 'promoted' and statuses.count('promoting') >= 1
    assert live.promoted_ids() == [run_id]
    image = image_of(live, run_id)
    assert len(image) == 12 and set(image.values()) == {1, 2}
    assert live.run_row(run_id).next_faction_id == 3
    assert len(live.rows(schema.LINEAGE, run_id)) == 2 and len(live.rows(schema.CONFLICT, run_id)) == 4
    assert sorted(live.rows(schema.MEMBERS)['run_id'].unique().tolist()) == [run_id]


def test_crash_before_first_marker_is_visible(scratch: Any, tmp_path: Path) -> None:
    live = Live(scratch, tmp_path)
    live.seed()
    live.ctx.fault = crash_at('after_intent')
    with pytest.raises(SimulatedCrash):
        live.poll('full')
    run_id = live.intended_run().run_id
    assert live.statuses(run_id) == ['running']
    assert (live.model_dir / INTENT).is_file()

    live.restart()
    assert live.poll().status == 'none'
    assert live.statuses(run_id) == ['running', 'promoting', 'promoted']
    assert len(image_of(live, run_id)) == 12


def test_first_marker_commits_after_empty_reread(scratch: Any, tmp_path: Path) -> None:
    live = Live(scratch, tmp_path)
    live.seed()
    live.client.plan = {'running': 'hold', 'promoting': 'hold'}
    live.ctx.fault = crash_at('after_promoting_row')
    with pytest.raises(SimulatedCrash):
        live.poll('full')
    run_id = live.intended_run().run_id
    # The re-read of the table finds nothing of the run. Its outputs are written.
    assert live.repo.fetch_run_states() == {}
    assert len(live.rows(schema.MEMBERS, run_id)) == 12

    # The intent file is the source of truth: the next start completes the run whatever the table shows.
    live.client.plan = {}
    live.restart()
    assert live.poll().status == 'none'
    assert not (live.model_dir / INTENT).exists()
    assert live.promoted_ids() == [run_id]

    # The delayed rows of the first attempt commit now and only add rows that readers ignore.
    assert live.client.release() == 2
    assert live.statuses(run_id) == ['running', 'promoting', 'promoting', 'promoted']
    assert live.promoted_ids() == [run_id]
    assert live.run_row(run_id).next_faction_id == 3
    assert len(live.rows(schema.MEMBERS, run_id)) == 12
    assert live.poll().status == 'none'


def test_late_promotion_commit_is_harmless(scratch: Any, tmp_path: Path) -> None:
    live = Live(scratch, tmp_path)
    live.seed()
    live.client.plan = {'promoted': 'hold'}
    live.ctx.fault = crash_at('after_promoted_row')
    with pytest.raises(SimulatedCrash):
        live.poll('full')
    run_id = live.intended_run().run_id
    assert live.statuses(run_id) == ['running', 'promoting']

    # The next start resends both rows. The held insert stays held.
    live.client.plan = {}
    live.restart()
    assert live.poll().status == 'none'
    assert live.statuses(run_id) == ['running', 'promoting', 'promoting', 'promoted']
    # The original promoted insert commits after the recovery added its own rows.
    assert live.client.release() == 1
    assert live.statuses(run_id) == ['running', 'promoting', 'promoting', 'promoted', 'promoted']
    assert live.promoted_ids() == [run_id]
    assert len(image_of(live, run_id)) == 12
    assert len(live.rows(schema.CONFLICT, run_id)) == 4
    assert live.run_row(run_id).next_faction_id == 3


@pytest.mark.parametrize('point', ['temp_written', 'temp_fsynced'])
def test_torn_intent_write_leaves_nothing(scratch: Any, tmp_path: Path, point: str, monkeypatch: Any) -> None:
    live = Live(scratch, tmp_path)
    live.seed()
    install = lock.install
    torn: Any = lambda model_dir, record: install(model_dir, record, fail_at=cast(CrashPoint, point))  # noqa: E731
    monkeypatch.setattr(lock, 'install', torn)
    with pytest.raises(SimulatedCrash):
        live.poll('full')
    assert not (live.model_dir / INTENT).exists()
    # The temporary file is left behind and ignored.
    left = [path.name for path in live.model_dir.iterdir()]
    assert len(left) == 1 and left[0].startswith(lock.TEMP_PREFIX) and left[0].endswith('.tmp')
    crashed = live.repo.latest_promoted_run()
    assert crashed is None
    assert len(live.repo.fetch_run_states()) == 1

    # The next start has no intent to honour and runs again, with an intent writer that works.
    monkeypatch.setattr(lock, 'install', install)
    live.restart()
    outcome = live.poll()
    assert outcome.status == 'promoted', outcome.reason
    assert outcome.run_id is not None
    assert len(image_of(live, outcome.run_id)) == 12
    assert live.promoted_ids() == [outcome.run_id]


def test_torn_intent_write_after_the_rename_leaves_a_complete_file(
    scratch: Any, tmp_path: Path, monkeypatch: Any
) -> None:
    live = Live(scratch, tmp_path)
    live.seed()
    install = lock.install
    torn: Any = lambda model_dir, record: install(model_dir, record, fail_at='renamed')  # noqa: E731
    monkeypatch.setattr(lock, 'install', torn)
    with pytest.raises(SimulatedCrash):
        live.poll('full')
    record = live.intended_run()
    assert record.next_faction_id == 3
    crashed_id = record.run_id
    assert live.statuses(crashed_id) == ['running']

    # The file is complete, so the next start completes the run it names.
    live.restart()
    assert live.poll().status == 'none'
    assert live.statuses(crashed_id) == ['running', 'promoting', 'promoted']
    assert len(image_of(live, crashed_id)) == 12
    assert live.run_row(crashed_id).next_faction_id == 3


def test_unreadable_intent_fails_closed(scratch: Any, tmp_path: Path) -> None:
    live = Live(scratch, tmp_path)
    live.seed()
    first = live.poll('full')
    assert first.status == 'promoted', first.reason
    assert first.run_id is not None
    runs_before = live.repo.fetch_run_rows(first.run_id)
    members_before = live.snapshot(schema.MEMBERS, first.run_id)
    (live.model_dir / INTENT).write_text('{not json')

    live.restart()
    with pytest.raises(IntentUnreadable):
        live.poll()
    assert live.repo.fetch_run_rows(first.run_id) == runs_before
    pd.testing.assert_frame_equal(live.snapshot(schema.MEMBERS, first.run_id), members_before)
    assert (live.model_dir / INTENT).read_text() == '{not json'
    assert live.promoted_ids() == [first.run_id]


def test_cleanup_preserves_promoted_runs(scratch: Any, tmp_path: Path) -> None:
    live = Live(scratch, tmp_path)
    live.seed()
    good = live.poll('full')
    assert good.status == 'promoted', good.run_id
    assert good.run_id is not None
    template = live.run_row(good.run_id)
    stale = run_id_at(NOW - timedelta(days=2), '00000009')
    fresh = run_id_at(NOW - timedelta(hours=2), '00000008')
    for run_id in (stale, fresh):
        at = datetime.fromtimestamp(int(run_id[:13]) / 1000, tz=UTC)
        live.repo.insert_run_row(replace(template, run_id=run_id, status='failed', recorded_at=at, started_at=at))
        live.write_rows(
            schema.MEMBERS,
            run_id,
            did='did:plc:stale',
            faction_id=9,
            since_run_id='',
            retire_at=NOW + timedelta(days=30),
        )
        live.write_rows(schema.CONFLICT, run_id, written_at=NOW)

    assert live.poll().status == 'none'
    assert live.rows(schema.MEMBERS, stale).empty and live.rows(schema.CONFLICT, stale).empty
    assert len(live.rows(schema.MEMBERS, fresh)) == 1 and len(live.rows(schema.CONFLICT, fresh)) == 1
    assert len(live.rows(schema.MEMBERS, good.run_id)) == 12
    assert live.promoted_ids() == [good.run_id]
    assert live.statuses(stale) == ['failed'] and live.statuses(fresh) == ['failed']


def test_startup_after_success_is_allowed(scratch: Any, tmp_path: Path) -> None:
    live = Live(scratch, tmp_path)
    live.seed()
    good = live.poll('full')
    assert good.status == 'promoted' and good.run_id is not None
    before = {noun: live.snapshot(noun, good.run_id) for noun in schema.RUN_OWNED}
    rows_before = live.repo.fetch_run_rows(good.run_id)

    # A restart right after a success changes nothing and starts nothing.
    live.restart()
    assert live.poll().status == 'none'
    for noun, snapshot in before.items():
        pd.testing.assert_frame_equal(live.snapshot(noun, good.run_id), snapshot)
    assert live.repo.fetch_run_rows(good.run_id) == rows_before
    assert len(live.repo.fetch_run_states()) == 1

    # A week later the same process runs the weekly run.
    live.advance(7)
    weekly = live.poll()
    assert weekly.status == 'promoted' and weekly.kind == 'weekly'
    assert len(live.repo.fetch_run_states()) == 2


def test_immediate_retry_after_failure_is_allowed(scratch: Any, tmp_path: Path) -> None:
    live = Live(scratch, tmp_path)
    live.seed()
    live.ctx.fault = fail_at('after_write_members')
    failed = live.poll('full')
    assert failed.status == 'failed' and failed.run_id is not None
    assert live.rows(schema.MEMBERS, failed.run_id).empty

    live.ctx.fault = None
    retry = live.poll()
    assert retry.status == 'promoted' and retry.kind == 'full'
    assert_consistent_first_image(live, retry)
    assert live.promoted_ids() == [retry.run_id]


# ---- the caches ----


def test_paid_labels_survive_a_failed_run(scratch: Any, tmp_path: Path) -> None:
    live = Live(scratch, tmp_path)
    live.seed()
    paid = live.frame(
        schema.JEV_LABELS,
        uri_hash=np.asarray([77], dtype=np.uint64),
        prompt_version='hostility-v1',
        h=0.9,
        p_hostile=0.5,
        p_threat=0.4,
        model='jev-1.14.0',
        input_tokens=120,
        labelled_at=NOW - timedelta(hours=1),
    )
    live.repo.insert_frame(schema.JEV_LABELS, paid)
    live.ctx.fault = fail_at('after_write_members')
    outcome = live.poll('full')
    assert outcome.status == 'failed' and outcome.run_id is not None
    assert live.rows(schema.MEMBERS, outcome.run_id).empty
    # The cleanup of a failed run never touches a cache.
    assert live.rows(schema.JEV_LABELS)['uri_hash'].tolist() == [77]
    assert live.repo.fetch_jev_labels('hostility-v1')['h'].tolist() == pytest.approx([0.9])


def test_duplicate_cache_writes_aggregate_once(scratch: Any, tmp_path: Path) -> None:
    live = Live(scratch, tmp_path)
    for hostility, age in ((0.2, 2), (0.8, 1)):
        live.repo.insert_frame(
            schema.TEXT_SCORES,
            live.frame(
                schema.TEXT_SCORES,
                uri_hash=np.asarray([555, 556], dtype=np.uint64),
                h=hostility,
                model_version='m1',
                kind='reply',
                scored_at=NOW - timedelta(minutes=age),
                is_example=0,
                author_did=None,
                target_did=None,
                post_text=None,
                event_at=None,
            ),
        )
    hashes, scores = live.repo.fetch_text_scores('m1')
    assert hashes.tolist() == [555, 556] and scores.tolist() == pytest.approx([0.8, 0.8])
    other_hashes, other_scores = live.repo.fetch_text_scores('m2')
    assert len(other_hashes) == 0 and len(other_scores) == 0


def test_dataset_keeps_each_example_once(scratch: Any, tmp_path: Path) -> None:
    live = Live(scratch, tmp_path)
    live.seed()
    parent = world.post_uri(live.dids, 6, '101')
    rows = live.frame(
        schema.SCORER_DATASET,
        dataset_version='ds-it',
        uri_hash=np.asarray([901, 902, 903], dtype=np.uint64),
        kind=['reply', 'quote', 'reply'],
        split=['train', 'test', 'train'],
        label_source=['jev', 'block', 'random'],
        target=np.asarray([1, 1, 0], dtype=np.uint8),
        weight=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
        label_h=np.asarray([0.9, None, None], dtype=object),
        author_did=np.asarray([11, 12, 13], dtype=np.uint64),
        target_did=np.asarray([21, 22, 23], dtype=np.uint64),
        text=['you are an idiot', 'SHUT UP loser', 'thanks'],
        parent_uri=[parent, 'at://did:plc:missing/app.bsky.feed.post/none', parent],
        event_ms=np.asarray([1_794_000_000_000] * 3, dtype=np.int64),
        languages=[('en',)] * 3,
        text_length=np.asarray([16, 14, 6], dtype=np.uint32),
        uppercase=np.asarray([0, 5, 0], dtype=np.uint32),
        exclamations=np.asarray([0, 0, 0], dtype=np.uint32),
        questions=np.asarray([0, 0, 0], dtype=np.uint32),
        mentions=np.asarray([0, 0, 0], dtype=np.uint32),
        written_at=NOW - timedelta(minutes=5),
    )
    # The same dataset is written twice, as a retried write would.
    live.repo.insert_frame(schema.SCORER_DATASET, rows)
    live.repo.insert_frame(schema.SCORER_DATASET, rows.assign(written_at=NOW - timedelta(minutes=4)))
    writes = live.repo.fetch_dataset('ds-it')
    assert len(writes) == 6 and writes['uri_hash'].nunique() == 3
    assert len(export_dataset.latest_writes(writes)) == 3

    out_dir = tmp_path / 'export'
    window = Window(NOW - timedelta(days=60), NOW)
    manifest = export_dataset.export(live.repo, out_dir, 'ds-it', window)
    assert (manifest.dataset_version, manifest.rows, len(manifest.shards)) == ('ds-it', 3, 1)
    document = json.loads((out_dir / export_dataset.MANIFEST_FILE).read_text())
    assert document['rows'] == 3 and document['shards'][0]['rows'] == 3
    lines = (out_dir / manifest.shards[0]['file']).read_text().splitlines()
    assert len(lines) == 3
    records = [json.loads(line) for line in lines]
    assert sorted(record['uri_hash'] for record in records) == [901, 902, 903]
    for record in records:
        # The row key set names no DID and no handle.
        assert not any('did' in key or 'handle' in key for key in record)
        assert set(record) == set(export_dataset.EXPORT_FIELDS) | {'parent_text'}
        assert isinstance(record['author_hash'], int) and isinstance(record['target_hash'], int)
    with_parent = {record['uri_hash']: record['parent_text'] for record in records}
    assert with_parent[901] == 'root post 101' and with_parent[903] == 'root post 101'
    assert with_parent[902] == ''
    assert {record['label_h'] for record in records if record['uri_hash'] == 902} == {None}
