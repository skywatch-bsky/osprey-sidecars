import hashlib
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from faction_analysis import model_store, pipeline, schema
from faction_analysis.config import AppConfig
from faction_analysis.lock import SimulatedCrash
from faction_analysis.model import FEATURE_CONFIG, TrainingRows, train
from faction_analysis.promotion import IntentUnreadable
from faction_analysis.runstate import RunRow, RunStatus
from tests.conftest import fabricated_rows
from tests.fake_repository import FakeRepository, two_faction_world

NOW = datetime(2026, 11, 27, 12, tzinfo=UTC)
INTENT = 'promote-pending.json'
WRITE_STAGES = {
    'after_write_members': 'write_members',
    'after_write_features': 'write_features',
    'after_write_conflict': 'write_conflict',
    'after_write_registry': 'write_registry',
}
CRASH_POINTS = {
    'after_intent': ['running'],
    'after_promoting_row': ['running', 'promoting'],
    'after_promoted_row': ['running', 'promoting', 'promoted'],
}


def config(**overrides: str) -> AppConfig:
    values = {
        'FACTION_ANALYSIS_OBJECTIVE': 'modularity',
        'FACTION_ANALYSIS_RESOLUTION': '1.0',
        'FACTION_ANALYSIS_N_ITERATIONS': '-1',
        'FACTION_ANALYSIS_MIN_FACTION_SIZE': '3',
        'FACTION_ANALYSIS_CONFIDENCE_SEEDS': '1',
        'FACTION_ANALYSIS_PILEON_MIN_REPLIERS': '2',
        'FACTION_ANALYSIS_SWARM_MIN_QUOTERS': '2',
        'FACTION_ANALYSIS_MIN_TEXT_SCORED': '1',
    }
    return AppConfig.from_mapping({**values, **overrides})


def a_run(run_row: RunRow, run_id: str, status: RunStatus, at: datetime) -> RunRow:
    return replace(run_row, run_id=run_id, status=status, recorded_at=at, started_at=at)


def run_id_at(moment: datetime, suffix: str = '00000001') -> str:
    """A run ID that starts at ``moment``, as the IDs the sidecar issues do."""
    return f'{int(moment.timestamp() * 1000):013d}.{suffix}'


class Harness:
    """A repository over the two-faction world, a model directory and a clock the tests move."""

    def __init__(self, tmp_path: Path, *, split: bool = False, collision: bool = False, **overrides: str) -> None:
        self.now = NOW
        self.world, self.dids = two_faction_world(NOW, split=split, collision=collision)
        self.repo = FakeRepository(self.world)
        self.model_dir = tmp_path / 'models'
        self.model_dir.mkdir()
        self.config = config(**overrides)
        self.ctx = pipeline.RunContext(self.config, self.repo, self.model_dir, now=lambda: self.now)

    def poll(self, kind: pipeline.RunKind | None = None) -> pipeline.RunOutcome:
        return pipeline.poll(self.ctx, kind=kind)

    def advance(self, days: int) -> None:
        self.now += timedelta(days=days)

    def install_model(self) -> str:
        rows: TrainingRows = fabricated_rows(jev_rows=300, weak_rows=60, random_rows=60)
        payload = model_store.serialise(train(rows, seed=1))
        manifest = model_store.ModelManifest(
            version='h-test-1',
            sha256=hashlib.sha256(payload).hexdigest(),
            trained_at=NOW.isoformat(),
            dataset_version='ds-test',
            label_policy_hash='policy',
            feature_config=dict(FEATURE_CONFIG),
            row_counts={'train': 1},
            metrics={},
            calibrated=True,
        )
        model_store.promote(self.model_dir, payload, manifest)
        return manifest.version


def frame_of(repo: FakeRepository, noun: str, run_id: str) -> pd.DataFrame:
    """What a run wrote to a table. An empty write leaves no frame, as it leaves no rows in ClickHouse."""
    return repo.run_owned[noun].get(run_id, pd.DataFrame())


def promoted(h: Harness) -> RunRow:
    run = h.repo.latest_promoted_run()
    assert run is not None
    return run


def test_run_promotes_without_model(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    outcome = h.poll('full')
    assert outcome.status == 'promoted', outcome.reason
    run_id = outcome.run_id
    assert run_id is not None
    assert h.repo.statuses(run_id) == ['running', 'promoting', 'promoted']
    assert not (h.model_dir / INTENT).exists()
    for noun in schema.RUN_OWNED:
        if noun != schema.RUNS:
            assert run_id in h.repo.run_owned[noun], noun

    members = h.repo.fetch_image(run_id)
    assert len(members) == 12
    assert set(members['faction_id'].tolist()) == {1, 2}
    assert members['confidence'].notna().all()
    assert members['prev_faction_id'].isna().all()
    assert (members['since_run_id'] == run_id).all()

    run = promoted(h)
    assert (run.factions, run.accounts_placed, run.accounts_unplaced) == (2, 12, 0)
    assert (run.births, run.merges, run.splits, run.deaths) == (2, 0, 0, 0)
    assert run.next_faction_id == 3
    assert run.modularity is not None and run.modularity > 0.4
    assert run.agreement_prev is None
    assert run.text_model_version == '' and run.text_coverage == 0.0

    assert sorted(frame_of(h.repo, 'lineage', run_id)['event'].tolist()) == ['birth', 'birth']
    assert (frame_of(h.repo, 'moves', run_id)['move_kind'] == 'assign').all()
    registry = frame_of(h.repo, 'factions', run_id)
    assert sorted(registry['top_tags'].tolist(), key=len) == [['music'], ['art', 'ink']]
    assert all(handles for handles in registry['sample_handles'].tolist())
    assert frame_of(h.repo, 'faction_runs', run_id)['member_count'].tolist() == [6, 6]

    features = frame_of(h.repo, 'member_features', run_id)
    assert features['blocks_sent'].sum() == 2 and features['blocks_received'].sum() == 2
    assert features['pileon_replies_sent'].sum() == 2
    assert features['swarm_quotes_received'].sum() == 2
    assert features['within_z'].notna().all()

    cells = frame_of(h.repo, 'conflict', run_id)
    assert sorted(cells['window_days'].unique().tolist()) == [7, 60]
    assert cells['text_scored'].sum() == 0
    assert cells['text_lift'].isna().all()
    # Hand-computed. Faction 1 sends three replies to faction 2 and faction 2 one reply and two quotes
    # back, so each side expects 1.5 of the 6 contacts. The pile-on, the swarm and the two blocks are two
    # events each: with a prior of 20, (2 + 20 * 2/6) / (3 + 20) divided by 2/6 is 1.1304.
    forward = cells.loc[(cells['window_days'] == 7) & (cells['from_faction_id'] == 1) & (cells['to_faction_id'] == 2)]
    backward = cells.loc[(cells['window_days'] == 7) & (cells['from_faction_id'] == 2) & (cells['to_faction_id'] == 1)]
    assert forward['contact'].tolist() == [3] and backward['contact'].tolist() == [3]
    assert forward['contact_lift'].tolist() == [2.0]
    assert forward['pileon_replies'].tolist() == [2] and forward['pileon_lift'].tolist() == pytest.approx([1.130435])
    assert forward['direction_pileon'].tolist() == [1.0] and backward['direction_pileon'].tolist() == [-1.0]
    assert forward['blocks'].tolist() == [2] and backward['blocks'].tolist() == [0]
    assert forward['swarm_quotes'].tolist() == [0] and backward['swarm_quotes'].tolist() == [2]
    # Accounts 0, 1 and 3 reply and account 2 blocks: four sources on two targets.
    assert forward['distinct_sources'].tolist() == [4] and forward['distinct_targets'].tolist() == [2]
    assert forward['conflict_score'].tolist() == pytest.approx([1.130435])
    # Nothing happens inside a faction, so neither diagonal cell is written.
    assert not ((cells['from_faction_id'] == cells['to_faction_id']).any())
    assert len(
        frame_of(h.repo, 'conflict', run_id).groupby(['window_days', 'from_faction_id', 'to_faction_id'])
    ) == len(frame_of(h.repo, 'conflict', run_id))


def test_a_second_run_keeps_every_id_and_logs_no_move(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    first = h.poll('full')
    h.advance(7)
    second = h.poll()
    assert second.status == 'promoted', second.reason
    assert second.kind == 'weekly'
    assert first.run_id is not None and second.run_id is not None
    before = h.repo.fetch_image(first.run_id).set_index('did')['faction_id']
    after = h.repo.fetch_image(second.run_id).set_index('did')['faction_id']
    assert before.to_dict() == after.to_dict()
    assert len(frame_of(h.repo, 'moves', second.run_id)) == 0
    assert len(frame_of(h.repo, 'lineage', second.run_id)) == 0
    assert (h.repo.fetch_image(second.run_id)['since_run_id'] == first.run_id).all()
    assert promoted(h).agreement_prev == pytest.approx(1.0)
    assert promoted(h).next_faction_id == 3


def test_a_split_keeps_the_id_with_one_part_and_records_the_other(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    first = h.poll('full')
    assert first.run_id is not None
    before = h.repo.fetch_image(first.run_id).set_index('did')['faction_id']
    split_at = h.now
    h.world.follows = two_faction_world(split_at, split=True)[0].follows
    h.advance(7)
    second = h.poll()
    assert second.status == 'promoted', second.reason
    assert second.run_id is not None
    after = h.repo.fetch_image(second.run_id).set_index('did')['faction_id']
    child = after.loc[h.dids[3]]
    parent = after.loc[h.dids[0]]
    assert child != parent
    assert parent == before.loc[h.dids[0]]
    lineage = frame_of(h.repo, 'lineage', second.run_id)
    splits = lineage.loc[lineage['event'] == 'split']
    assert splits['faction_id'].tolist() == [child]
    assert splits['related_faction_id'].tolist() == [parent]
    moves = frame_of(h.repo, 'moves', second.run_id)
    assert set(moves['move_kind'].tolist()) == {'move'} and len(moves) == 3
    registry = frame_of(h.repo, 'factions', second.run_id)
    assert registry.loc[registry['faction_id'] == child]['parent_faction_id'].tolist() == [parent]
    run = promoted(h)
    assert (run.factions, run.splits, run.next_faction_id) == (3, 1, 4)


def test_a_deleted_block_is_no_conflict_signal(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    start = int((NOW - timedelta(days=1)).replace(minute=0, second=0, microsecond=0).timestamp() * 1000)
    # The block of account 2 is deleted a minute after it was made. The delete of account 0's block carries
    # the very time of its creation, which leaves the block: the record was made again after that delete.
    h.world.block_deletes += [(7, start + 7 * 60_000), (6, start + 5 * 60_000)]
    outcome = h.poll('full')
    assert outcome.status == 'promoted', outcome.reason
    assert outcome.run_id is not None
    cells = frame_of(h.repo, 'conflict', outcome.run_id)
    forward = cells.loc[(cells['window_days'] == 7) & (cells['from_faction_id'] == 1) & (cells['to_faction_id'] == 2)]
    assert forward['blocks'].tolist() == [1]
    assert forward['blocks_lift'].tolist() == pytest.approx([1.130435])
    assert (forward['distinct_sources'].tolist(), forward['distinct_targets'].tolist()) == ([3], [1])
    features = frame_of(h.repo, 'member_features', outcome.run_id).set_index('did')
    assert features['blocks_sent'].sum() == 1 and features['blocks_received'].sum() == 1
    assert features.loc[h.dids[0], 'blocks_sent'] == 1 and features.loc[h.dids[2], 'blocks_sent'] == 0
    assert features.loc[h.dids[6], 'blocks_received'] == 1 and features.loc[h.dids[7], 'blocks_received'] == 0


def test_a_promoted_run_whose_image_is_gone_retires_its_factions_and_never_reuses_ids(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    first = h.poll('full')
    assert first.run_id is not None
    # Retention removed the image of the first run. Its run row and its registry rows remain.
    h.repo.run_owned[schema.MEMBERS].pop(first.run_id)
    h.repo.run_owned[schema.MEMBER_FEATURES].pop(first.run_id)
    h.advance(7)
    second = h.poll()
    assert second.status == 'promoted', second.reason
    assert second.run_id is not None
    members = h.repo.fetch_image(second.run_id)
    assert set(members['faction_id'].tolist()) == {3, 4}
    assert members['prev_faction_id'].isna().all()
    assert (frame_of(h.repo, 'moves', second.run_id)['move_kind'] == 'assign').all()
    assert sorted(frame_of(h.repo, 'lineage', second.run_id)['event'].tolist()) == ['birth', 'birth', 'death', 'death']
    registry = frame_of(h.repo, 'factions', second.run_id)
    retired = registry.loc[registry['status'] == 'retired', 'faction_id'].tolist()
    assert sorted(retired) == [1, 2]
    run = promoted(h)
    assert (run.next_faction_id, run.agreement_prev) == (5, None)


def test_text_is_scored_once_and_the_score_follows_the_image(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    version = h.install_model()
    first = h.poll('full')
    assert first.run_id is not None
    scores = h.repo.cache_frames['text_scores'][0]
    # Five interactions have text. The reply without text is never scored.
    assert len(scores) == 5
    assert (scores['model_version'] == version).all()
    assert scores.loc[scores['is_example'] == 1]['post_text'].notna().all()
    assert scores.loc[scores['is_example'] == 0]['post_text'].isna().all()
    assert scores.loc[scores['h'] >= 0.5, 'is_example'].eq(1).all()
    run = promoted(h)
    assert run.text_model_version == version and 0 < run.text_coverage < 1
    cells = frame_of(h.repo, 'conflict', first.run_id)
    assert cells['text_scored'].sum() > 0

    # A week later the first faction splits. The replies of accounts 0 and 1 now leave faction 1 and the
    # reply without text leaves the new faction 3, and no text is scored again. The interactions are eight
    # days old by now, so only the long window holds them.
    h.world.follows = two_faction_world(h.now, split=True)[0].follows
    h.advance(7)
    second = h.poll()
    assert second.status == 'promoted', second.reason
    assert len(h.repo.cache_frames['text_scores']) == 1
    assert second.run_id is not None
    again = frame_of(h.repo, 'conflict', second.run_id)
    assert sorted(again['window_days'].unique().tolist()) == [60]
    long = again.set_index(['from_faction_id', 'to_faction_id'])
    before = cells.loc[cells['window_days'] == 60].set_index(['from_faction_id', 'to_faction_id'])
    assert long.loc[(1, 2), 'contact'] == 2 and long.loc[(3, 2), 'contact'] == 1
    assert long.loc[(1, 2), 'text_scored'] == 2 and long.loc[(3, 2), 'text_scored'] == 0
    assert long.loc[(1, 2), 'text_hostility'] == before.loc[(1, 2), 'text_hostility']
    assert long.loc[(2, 1), 'text_hostility'] == before.loc[(2, 1), 'text_hostility']
    assert again['text_lift'].notna().any()
    assert again['text_scored'].sum() == before['text_scored'].sum()


@pytest.mark.parametrize(('point', 'stage'), sorted(WRITE_STAGES.items()))
def test_a_failure_after_a_write_stage_cleans_the_run_up(tmp_path: Path, point: str, stage: str) -> None:
    h = Harness(tmp_path)

    def fail(where: str) -> None:
        if where == point:
            raise RuntimeError('injected')

    h.ctx.fault = fail
    outcome = h.poll('full')
    assert outcome.status == 'failed' and outcome.stage == stage
    assert outcome.reason == 'RuntimeError' and outcome.exit_code == 1
    assert outcome.run_id is not None
    assert h.repo.statuses(outcome.run_id) == ['running', 'failed']
    assert h.repo.promoted_run_ids() == []
    assert not (h.model_dir / INTENT).exists()
    for noun in schema.IMAGE_TABLES:
        assert (noun, outcome.run_id) in h.repo.dropped
    for noun in schema.RUN_OWNED:
        if noun not in schema.IMAGE_TABLES and noun != schema.RUNS:
            assert (noun, outcome.run_id) in h.repo.deleted
    failed = h.repo.fetch_run_rows(outcome.run_id)[-1]
    assert (failed.failed_stage, failed.error_class) == (stage, 'RuntimeError')
    # The next poll starts a clean run.
    h.ctx.fault = None
    h.advance(1)
    assert h.poll().status == 'promoted'


@pytest.mark.parametrize(('point', 'visible'), sorted(CRASH_POINTS.items()))
def test_a_crash_around_the_promotion_is_completed_on_the_next_poll(
    tmp_path: Path, point: str, visible: list[str]
) -> None:
    h = Harness(tmp_path)

    def crash(where: str) -> None:
        if where == point:
            raise SimulatedCrash(where)

    h.ctx.fault = crash
    with pytest.raises(SimulatedCrash):
        h.poll('full')
    run_id = h.repo.run_ids()[0]
    assert (h.model_dir / INTENT).is_file()
    assert h.repo.statuses(run_id) == visible
    h.ctx.fault = None
    outcome = h.poll()
    assert outcome.status == 'none'
    assert not (h.model_dir / INTENT).exists()
    assert h.repo.promoted_run_ids() == [run_id]
    statuses = h.repo.statuses(run_id)
    assert statuses[0] == 'running' and statuses[-1] == 'promoted'
    assert statuses.count('promoting') >= 1
    assert run_id in h.repo.run_owned[schema.MEMBERS]
    assert run_id in h.repo.run_owned[schema.CONFLICT]
    assert h.repo.dropped == [] and h.repo.deleted == []


def test_a_failure_after_the_intent_completes_the_run_forward(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    lost: list[str] = []

    def fail(where: str) -> None:
        # The response of the promoted insert is lost once. The re-send that follows succeeds.
        if where == 'after_promoted_row' and not lost:
            lost.append(where)
            raise RuntimeError('response lost')

    h.ctx.fault = fail
    outcome = h.poll('full')
    assert outcome.status == 'promoted' and outcome.exit_code == 0
    assert 'completed after RuntimeError' in outcome.reason
    assert outcome.run_id is not None
    assert not (h.model_dir / INTENT).exists()
    assert h.repo.promoted_run_ids() == [outcome.run_id]
    assert h.repo.dropped == [] and h.repo.deleted == []


def test_a_delayed_commit_is_left_unresolved_and_nothing_is_removed(tmp_path: Path) -> None:
    h = Harness(tmp_path)

    def fail(where: str) -> None:
        if where == 'after_promoting_row':
            raise RuntimeError('response lost')

    h.repo.hold_next_promoting = True
    h.ctx.fault = fail
    first = h.poll('full')
    assert first.status == 'unresolved' and first.exit_code == 1
    run_id = first.run_id
    assert run_id is not None
    assert (h.model_dir / INTENT).is_file()
    assert h.repo.statuses(run_id) == ['running']
    assert h.repo.dropped == [] and h.repo.deleted == []
    assert run_id in h.repo.run_owned[schema.MEMBERS]

    h.ctx.fault = None
    assert h.poll().status == 'unresolved'
    assert h.repo.run_ids() == [run_id]

    h.repo.release(run_id)
    assert h.poll().status == 'none'
    assert not (h.model_dir / INTENT).exists()
    assert h.repo.promoted_run_ids() == [run_id]


def test_an_unreadable_intent_fails_closed(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    first = h.poll('full')
    assert first.run_id is not None
    (h.model_dir / INTENT).write_text('{not json')
    with pytest.raises(IntentUnreadable):
        h.poll()
    assert first.run_id in h.repo.run_owned[schema.MEMBERS]
    assert h.repo.dropped == [] and h.repo.deleted == []


def test_an_unfinished_run_blocks_new_runs_until_it_is_completed(tmp_path: Path, run_row: RunRow) -> None:
    h = Harness(tmp_path)
    # An unfinished monthly run: once it is promoted, no run is due for a month.
    old = run_id_at(NOW - timedelta(hours=1))
    h.repo.run_rows[old] = [a_run(replace(run_row, kind='full'), old, 'promoting', NOW)]
    h.repo.fail_on_status = {'promoted'}
    assert h.poll().status == 'unresolved'
    assert h.repo.run_ids() == [old]
    h.repo.fail_on_status = set()
    assert h.poll().status == 'none'
    assert h.repo.promoted_run_ids() == [old]
    # The IDs the unfinished run reserved are never issued again.
    assert h.poll('full').status == 'promoted'
    assert promoted(h).next_faction_id >= run_row.next_faction_id


def test_cleanup_removes_stale_runs_and_spares_everything_else(tmp_path: Path, run_row: RunRow) -> None:
    h = Harness(tmp_path)
    good = h.poll('full')
    stale = run_id_at(NOW - timedelta(days=2), '00000009')
    fresh = run_id_at(NOW - timedelta(hours=2), '00000008')
    for run_id in (stale, fresh):
        h.repo.run_owned[schema.MEMBERS][run_id] = pd.DataFrame({'run_id': [run_id], 'did': ['x']})
        h.repo.run_owned[schema.CONFLICT][run_id] = pd.DataFrame({'run_id': [run_id], 'a': [1]})
        h.repo.run_rows[run_id] = [a_run(run_row, run_id, 'failed', NOW - timedelta(days=2))]
    assert h.poll().status == 'none'
    assert (schema.MEMBERS, stale) in h.repo.dropped and (schema.CONFLICT, stale) in h.repo.deleted
    assert stale not in h.repo.run_owned[schema.MEMBERS]
    assert fresh in h.repo.run_owned[schema.MEMBERS] and fresh in h.repo.run_owned[schema.CONFLICT]
    assert good.run_id in h.repo.run_owned[schema.MEMBERS]
    assert h.repo.promoted_run_ids() == [good.run_id]


def test_a_running_run_older_than_a_day_is_marked_failed_and_cleaned(tmp_path: Path, run_row: RunRow) -> None:
    h = Harness(tmp_path)
    old = run_id_at(NOW - timedelta(days=2), '00000007')
    h.repo.run_rows[old] = [a_run(run_row, old, 'running', NOW - timedelta(days=2))]
    h.repo.run_owned[schema.MEMBERS][old] = pd.DataFrame({'run_id': [old], 'did': ['x']})
    assert h.poll('full').status == 'promoted'
    assert h.repo.statuses(old) == ['running', 'failed']
    assert h.repo.fetch_run_rows(old)[-1].error_class == 'RunAbandoned'
    assert (schema.MEMBERS, old) in h.repo.dropped


def test_no_faction_at_the_minimum_size_fails_the_partition(tmp_path: Path) -> None:
    h = Harness(tmp_path, FACTION_ANALYSIS_MIN_FACTION_SIZE='100')
    outcome = h.poll('full')
    assert outcome.status == 'failed'
    assert outcome.stage == 'partition' and outcome.reason == 'NoFactionError'
    assert outcome.run_id is not None
    failed = h.repo.fetch_run_rows(outcome.run_id)[-1]
    assert (failed.failed_stage, failed.error_class) == ('partition', 'NoFactionError')
    assert h.repo.promoted_run_ids() == []


def test_a_hash_collision_leaves_every_account_behind_it_unplaced(tmp_path: Path) -> None:
    h = Harness(tmp_path, collision=True)
    outcome = h.poll('full')
    assert outcome.status == 'promoted', outcome.reason
    run = promoted(h)
    assert (run.accounts_placed, run.accounts_unplaced, run.hash_collisions) == (10, 2, 1)
    assert outcome.run_id is not None
    placed = h.repo.fetch_image(outcome.run_id)['did'].tolist()
    assert h.dids[0] not in placed and h.dids[1] not in placed
    assert len(placed) == 10


def test_a_dry_run_computes_everything_and_writes_nothing(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    recorder = pipeline.RecordingRepository(h.repo)
    h.ctx = pipeline.RunContext(
        h.config, cast(pipeline.RunRepository, recorder), h.model_dir, now=lambda: h.now, dry_run=True
    )
    outcome = h.poll('full')
    assert outcome.status == 'promoted', outcome.reason
    assert h.repo.run_ids() == []
    assert all(not frames for frames in h.repo.run_owned.values())
    assert h.repo.dropped == [] and h.repo.deleted == []
    assert not (h.model_dir / INTENT).exists()
    written = {name for name, _ in recorder.writes}
    assert written >= set(schema.RUN_OWNED) - {schema.RUNS}


def test_a_failure_after_a_success_changes_no_promoted_state(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    good = h.poll('full')
    assert good.run_id is not None
    before = {
        noun: {run_id: frame.copy() for run_id, frame in frames.items()} for noun, frames in h.repo.run_owned.items()
    }
    h.advance(8)

    def fail(where: str) -> None:
        if where == 'after_write_conflict':
            raise RuntimeError('injected')

    h.ctx.fault = fail
    assert h.poll().status == 'failed'
    for noun, frames in before.items():
        for run_id, frame in frames.items():
            pd.testing.assert_frame_equal(h.repo.run_owned[noun][run_id], frame)
    assert h.repo.promoted_run_ids() == [good.run_id]
    h.ctx.fault = None
    assert h.poll().status == 'promoted'
