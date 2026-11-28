import subprocess
from argparse import Namespace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from faction_analysis import measure
from faction_analysis import measurement_rules as rules
from faction_analysis.config import GraphConfig
from faction_analysis.graph import aggregate_window_edges, build_graph
from faction_analysis.queries import Window

SOURCE = 'osprey_execution_results'
WINDOW = Window(datetime(2026, 11, 25, tzinfo=UTC), datetime(2026, 11, 27, tzinfo=UTC))
DAY = Window(datetime(2026, 11, 26, tzinfo=UTC), datetime(2026, 11, 27, tzinfo=UTC))


def records(dtype: np.dtype, rows: tuple[tuple[int, ...], ...]) -> np.ndarray:
    array = np.zeros(len(rows), dtype=dtype)
    for index, name in enumerate(dtype.names or ()):
        array[name] = [row[index] for row in rows]
    return array


def follow_rows(*rows: tuple[int, int, int, int]) -> np.ndarray:
    return records(measure.FOLLOWS_DTYPE, rows)


def pair_rows(*rows: tuple[int, int]) -> np.ndarray:
    return records(measure.PAIR_DTYPE, rows)


def delete_rows(*rows: tuple[int, int]) -> np.ndarray:
    return records(measure.DELETE_DTYPE, rows)


class FakeRunner:
    """Answers each kind of statement once with canned bytes, then with nothing. Records every statement."""

    def __init__(
        self,
        follows: np.ndarray | None = None,
        reposts: np.ndarray | None = None,
        deletes: np.ndarray | None = None,
        audits: dict[str, int] | None = None,
        text: str = 'ok',
    ) -> None:
        self.follows = follows if follows is not None else follow_rows()
        self.reposts = reposts if reposts is not None else pair_rows()
        self.deletes = deletes if deletes is not None else delete_rows()
        self.audits = audits or dict.fromkeys(rules.AUDIT_COLUMNS, 0)
        self.text = text
        self.statements: list[rules.MeasuredStatement] = []
        self.seconds = 0.0

    def __call__(self, statement: rules.MeasuredStatement) -> bytes:
        self.statements.append(statement)
        rules.validate_statement(statement.sql)
        name = statement.name
        if name.startswith('follows_'):
            data = self.follows.tobytes()
            self.follows = follow_rows()
        elif name.startswith('reposts_'):
            data = self.reposts.tobytes()
            self.reposts = pair_rows()
        elif name.startswith('follow_deletes_'):
            data = self.deletes.tobytes()
            self.deletes = delete_rows()
        elif name.startswith('audit_'):
            data = ('\t'.join(str(self.audits[column]) for column in rules.AUDIT_COLUMNS) + '\n').encode()
        else:
            data = (self.text + '\n').encode()
        if statement.output_format == rules.BINARY:
            return data
        return data


def test_every_measurement_statement_is_select_with_limit() -> None:
    runner = FakeRunner(
        follows=follow_rows((1, 2, 100, 1000), (2, 1, 101, 1001)),
        reposts=pair_rows((1, 2), (3, 4)),
        deletes=delete_rows((100, 2000)),
    )
    measure.extract_edges(runner, SOURCE, WINDOW, GraphConfig(2, 2.0, 0.5, True))
    measure.run_probes(runner, SOURCE, DAY)
    measure.run_audit(runner, SOURCE, WINDOW)
    assert len(runner.statements) == 6 + 11 + 2
    for statement in runner.statements:
        assert statement.sql.lstrip().upper().startswith(('SELECT', 'WITH')), statement.name
        assert statement.sql.rstrip().upper().endswith(f'LIMIT {statement.ceiling}'), statement.name
        assert ';' not in statement.sql
        assert 'INTO OUTFILE' not in statement.sql.upper()


def test_validate_statement_rejects_everything_else() -> None:
    for bad in [
        'INSERT INTO t VALUES (1)',
        'SELECT 1',
        'SELECT 1;',
        'WITH x AS (SELECT 1) SELECT * FROM x',
        "SELECT 1 INTO OUTFILE 'a' LIMIT 1",
        'SELECT 1 FORMAT CSV LIMIT 1',
        'DROP TABLE t',
        'SELECT 1 LIMIT 1; SELECT 2 LIMIT 1',
    ]:
        with pytest.raises(rules.MeasurementRuleError):
            rules.validate_statement(bad)
    rules.validate_statement('WITH x AS (SELECT 1) SELECT * FROM x LIMIT 5')


def test_ssh_command_is_readonly_and_bounded() -> None:
    for output_format in (rules.TEXT, rules.BINARY):
        argv, remote = rules.ssh_invocation('sys2', output_format)
        assert argv[:6] == ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'sys2']
        assert argv[6] == remote
        assert remote.startswith('sudo -n docker exec -i clickhouse clickhouse-client')
        assert '--readonly=1' in remote
        assert '--max_execution_time=60' in remote
        assert '--max_threads=4' in remote
        assert f'--max_memory_usage={rules.MAX_MEMORY_USAGE}' in remote
        assert f'--format={output_format}' in remote
        assert remote.count(';') == 0 and '\n' not in remote


def test_ssh_invocation_refuses_a_host_or_format_it_cannot_quote() -> None:
    for host in ('sys2; rm -rf /', 'a b', '', '-oProxyCommand=x'):
        with pytest.raises(rules.MeasurementRuleError):
            rules.ssh_invocation(host, rules.TEXT)
    with pytest.raises(rules.MeasurementRuleError):
        rules.ssh_invocation('sys2', 'JSON; rm')


def test_measure_is_read_only(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[list[str], bytes]] = []

    def fake_run(argv: list[str], *, input: bytes, capture_output: bool, check: bool, timeout: int) -> object:
        seen.append((argv, input))
        return subprocess.CompletedProcess(argv, 0, stdout=b'1\n', stderr=b'')

    monkeypatch.setattr(measure.subprocess, 'run', fake_run)
    runner = measure.SshRunner('sys2')
    statement = rules.probe_statements(SOURCE, DAY)[0]
    assert runner(statement) == b'1\n'
    argv, sent = seen[0]
    remote = argv[6]
    assert remote.startswith('sudo -n docker exec -i clickhouse clickhouse-client --readonly=1')
    assert '--max_execution_time=60' in remote and '--max_threads=4' in remote
    assert '--max_memory_usage=' in remote
    assert sent.decode().strip().upper().startswith('SELECT')
    assert runner.statements == 1
    with pytest.raises(rules.MeasurementRuleError):
        runner(rules.MeasuredStatement('bad', 'DELETE FROM t', rules.TEXT))
    assert len(seen) == 1


def test_resource_floor_blocks_before_any_extraction(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert rules.resource_shortfall(rules.MIN_FREE_MEMORY, rules.MIN_FREE_DISK) == ()
    assert len(rules.resource_shortfall(2**30, rules.MIN_FREE_DISK)) == 1
    assert len(rules.resource_shortfall(rules.MIN_FREE_MEMORY, 2**30)) == 1
    monkeypatch.setattr(measure, 'free_memory_bytes', lambda: 2**30)
    spawned: list[list[str]] = []

    def fake_child(argv: list[str], **_kwargs: object) -> measure.ChildOutcome:
        spawned.append(list(argv))
        return measure.ChildOutcome(0, 0.0, 0)

    monkeypatch.setattr(measure, 'run_child', fake_child)
    args = Namespace(
        work_dir=tmp_path,
        host='sys2',
        source=SOURCE,
        start=WINDOW.start,
        end=WINDOW.end,
        mutual_follow_weight=2.0,
        repost_alpha=0.5,
        no_damping=False,
    )
    with pytest.raises(SystemExit) as excinfo:
        measure.command_extract(args)
    assert excinfo.value.code == measure.EXIT_REFUSED
    assert spawned == []


def test_a_statement_at_its_ceiling_stops_as_truncated(monkeypatch: pytest.MonkeyPatch) -> None:
    statement = rules.MeasuredStatement('follows_x', 'SELECT 1 LIMIT 3', rules.BINARY, 16, 2)
    with pytest.raises(measure.MeasurementTruncated):
        measure.binary_rows(statement, b'\0' * 32)
    with pytest.raises(measure.MeasurementError):
        measure.binary_rows(statement, b'\0' * 31)
    assert measure.binary_rows(statement, b'\0' * 16) == 1
    monkeypatch.setattr(rules, 'EDGE_ROWS_CEILING', 1)
    runner = FakeRunner(follows=follow_rows((1, 2, 3, 4)))
    with pytest.raises(measure.MeasurementTruncated, match='its ceiling'):
        measure.extract_edges(runner, SOURCE, WINDOW, GraphConfig(2, 2.0, 0.5, True))


class FakeProcess:
    def __init__(self, pid: int = 4242) -> None:
        self.pid = pid
        self.killed = False
        self.polls = 0

    def poll(self) -> int | None:
        self.polls += 1
        return -9 if self.killed else None

    def kill(self) -> None:
        self.killed = True

    def wait(self) -> int:
        return -9


def test_child_is_stopped_at_the_memory_limit() -> None:
    process = FakeProcess()
    ticks = iter([0.0, 1.0, 2.0, 3.0])
    outcome = measure.run_child(
        ['true'],
        max_seconds=3600,
        max_rss=1024,
        spawn=lambda argv, cwd=None: process,
        rss_of=lambda pid: 2048,
        clock=lambda: next(ticks),
        sleep=lambda seconds: None,
    )
    assert process.killed
    assert not outcome.completed
    assert outcome.returncode == -1
    assert outcome.peak_rss == 2048


def test_child_is_stopped_at_the_time_limit() -> None:
    process = FakeProcess()
    ticks = iter([0.0, 61.0, 62.0])
    outcome = measure.run_child(
        ['true'],
        max_seconds=60,
        max_rss=1024,
        spawn=lambda argv, cwd=None: process,
        rss_of=lambda pid: 10,
        clock=lambda: next(ticks),
        sleep=lambda seconds: None,
    )
    assert process.killed and not outcome.completed


def test_a_child_that_exits_is_not_stopped() -> None:
    class Done(FakeProcess):
        def poll(self) -> int:
            return 0

    outcome = measure.run_child(
        ['true'],
        max_seconds=60,
        max_rss=1024,
        spawn=lambda argv, cwd=None: Done(),
        rss_of=lambda pid: 10,
        clock=lambda: 0.0,
    )
    assert outcome.completed


def test_extract_edges_builds_the_graph_from_the_pulled_rows(tmp_path: Path) -> None:
    runner = FakeRunner(
        follows=follow_rows((1, 2, 100, 1000), (2, 1, 101, 1001), (5, 6, 102, 1002)),
        reposts=pair_rows((1, 2), (1, 2), (3, 4), (3, 4), (3, 4)),
        deletes=delete_rows((102, 1500)),
    )
    summary, graph = measure.extract_edges(runner, SOURCE, WINDOW, GraphConfig(60, 2.0, 0.5, True))
    # URI 102 was deleted after its create, so the follow 5 -> 6 is gone. 1 and 2 follow each other.
    assert summary.deleted_follows_removed == 1
    assert summary.follow_creates == 3 and summary.repost_creates == 5 and summary.follow_deletes == 1
    assert graph.account_hash.tolist() == [1, 2, 3, 4]
    assert graph.mutual_follow.tolist() == [True, False]
    assert graph.degree().mean() == pytest.approx(graph.weight.sum() * 2 / 4)
    assert runner.statements[0].name.startswith('follows_2026-11-25')


def test_extract_edges_refuses_an_empty_window() -> None:
    from faction_analysis.queries import QueryError

    with pytest.raises(QueryError):
        measure.extract_edges(FakeRunner(), SOURCE, Window(WINDOW.start, WINDOW.start), GraphConfig(60, 2.0, 0.5, True))


def test_audit_totals_sum_every_day() -> None:
    runner = FakeRunner(
        audits={
            'interactions': 100,
            'with_block': 50,
            'positive_15m': 1,
            'positive_1h': 2,
            'positive_4h': 3,
            'positive_24h': 4,
        }
    )
    # The canned answer is returned for both days.
    totals = measure.run_audit(runner, SOURCE, WINDOW)
    assert totals == {
        'interactions': 200,
        'with_block': 100,
        'positive_15m': 2,
        'positive_1h': 4,
        'positive_4h': 6,
        'positive_24h': 8,
    }
    first = runner.statements[0]
    assert first.name == 'audit_2026-11-25'
    assert 'LEFT ASOF JOIN' in first.sql
    assert 'if(b.block_ms > 0, b.block_ms - i.event_ms, NULL) AS gap_ms' in first.sql
    assert "toDateTime64('2026-11-25 00:00:00.000', 3, 'UTC')" in first.sql
    # The block side spans the interaction day and the whole next day, so a late block still counts.
    block_side = first.sql.split('LEFT ASOF JOIN')[1]
    assert "toDateTime64('2026-11-25 00:00:00.000', 3, 'UTC')" in block_side
    assert "toDateTime64('2026-11-27 00:00:00.000', 3, 'UTC')" in block_side
    assert first.sql.rstrip().endswith('LIMIT 1')


def test_audit_uses_the_same_reply_and_quote_predicates_as_the_sidecar() -> None:
    from faction_analysis.queries import interactions_sql

    audit = rules.block_positive_audit_sql(SOURCE, DAY, Window(DAY.start, DAY.end + timedelta(days=1)))
    sidecar = interactions_sql(SOURCE, DAY)
    for predicate in (rules.REPLY_PREDICATE, rules.QUOTE_PREDICATE):
        assert predicate in audit
        assert predicate in sidecar


def test_gates() -> None:
    def result(label: str, factions: int, **overrides: object) -> rules.LeidenResult:
        objective, _, resolution = label.partition('-')
        fields: dict[str, object] = {
            'configuration': rules.Configuration('cpm' if objective == 'cpm' else 'modularity', float(resolution)),
            'seed_offset': 0,
            'completed': True,
            'seconds': 600.0,
            'peak_rss': 2**33,
            'factions': factions,
        }
        fields.update(overrides)
        return rules.LeidenResult(**fields)  # ty: ignore[invalid-argument-type]

    assert rules.gate_a_failures([result('cpm-0.01', 120)]) == []
    assert rules.gate_a_passes([result('cpm-0.01', 120)])
    stopped = result('cpm-0.1', 0, completed=False, seconds=3601.0, peak_rss=21 * 2**30)
    # A configuration the parent stopped at a limit is named, but it does not fail the gate on its own:
    # the gate asks whether a run can finish at all.
    assert rules.gate_a_failures([stopped]) == []
    assert 'stopped at' in rules.stopped_runs([stopped])[0]
    assert not rules.gate_a_passes([stopped])
    assert rules.gate_a_passes([stopped, result('modularity-1', 52)])
    too_big = result('cpm-0.1', 0, peak_rss=21 * 2**30)
    assert 'exceed the limits' in rules.gate_a_failures([too_big])[0]
    assert not rules.gate_a_passes([too_big, result('modularity-1', 52)])
    assert rules.gate_b_failures({'positive_1h': 20_000}) == []
    assert rules.gate_b_failures({'positive_1h': 19_999})
    assert rules.gate_b_failures({}) == ['0 block positives at one hour, fewer than 20000']
    grid = [result('cpm-0.01', 4000), result('modularity-1', 52), result('modularity-0.5', 25)]
    stability = {'modularity-1': 0.689, 'modularity-0.5': 0.882}
    # Both modularity runs are in the band. The one whose seeds agree better is the default.
    chosen = rules.choose_default(grid, stability)
    assert chosen is not None and chosen.label == 'modularity-0.5'
    # Without a comparison the run nearer the middle of the band wins, and a compared run ranks first.
    chosen = rules.choose_default(grid)
    assert chosen is not None and chosen.label == 'modularity-1'
    chosen = rules.choose_default(grid, {'modularity-0.5': 0.3})
    assert chosen is not None and chosen.label == 'modularity-0.5'
    # The bounds count as in range. Outside the band on either side does not.
    assert rules.choose_default([result('modularity-0.3', 20)]) is not None
    assert rules.choose_default([result('modularity-3', 300)]) is not None
    assert rules.choose_default([result('modularity-0.3', 19), result('modularity-3', 301)]) is None
    assert rules.choose_default([result('modularity-0.5', 25, seed_offset=1)]) is None


def test_only_two_seeds_of_one_configuration_measure_stability() -> None:
    comparisons = [
        {
            'first': 'modularity-0.5-seed0.npy',
            'second': 'modularity-0.5-seed1.npy',
            'identical': False,
            'adjusted_rand': 0.882,
        },
        # A repeat of one seed shows determinism, not stability.
        {
            'first': 'modularity-1-seed0.npy',
            'second': 'modularity-1-seed0-repeat.npy',
            'identical': True,
            'adjusted_rand': 1.0,
        },
        {
            'first': 'modularity-1-seed0.npy',
            'second': 'modularity-1-seed1.npy',
            'identical': False,
            'adjusted_rand': 0.689,
        },
        # Two configurations against each other say nothing about either.
        {'first': 'modularity-0.5-seed0.npy', 'second': 'modularity-1-seed0.npy', 'adjusted_rand': 0.7},
        {'first': 'modularity-2-seed0.npy', 'second': 'modularity-2-seed1.npy', 'adjusted_rand': None},
    ]
    assert rules.seed_stability(comparisons) == {'modularity-0.5': 0.882, 'modularity-1': 0.689}
    assert rules.describe_comparison(comparisons[0]) == 'modularity-0.5: seed 0 against seed 1'
    assert rules.describe_comparison(comparisons[1]) == 'modularity-1: seed 0 against its repeat'


def test_the_report_leaves_out_sampled_rows_and_names_the_gates(tmp_path: Path) -> None:
    work = {
        'probes': {'follow_flags': 'create\t1\t1\t100', 'reply_samples': 'at://did:plc:abc/x\tdid:plc:abc'},
        'extraction': {'accounts': 12},
        'leiden': [
            {
                'configuration': 'modularity-0.5',
                'seed_offset': 0,
                'repeat': False,
                'completed': True,
                'seconds': 1500.0,
                'peak_rss': 2**34,
                'vertices': 12,
                'edges': 9,
                'communities_above_min': 25,
                'factions': 25,
                'unplaced': 0,
                'modularity': 0.6,
                'quality': 1.0,
            },
            {
                'configuration': 'modularity-1',
                'seed_offset': 0,
                'repeat': False,
                'completed': True,
                'seconds': 1900.0,
                'peak_rss': 2**34,
                'vertices': 12,
                'edges': 9,
                'communities_above_min': 52,
                'factions': 52,
                'unplaced': 0,
                'modularity': 0.65,
                'quality': 2.0,
            },
        ],
        'stopped': [],
        'comparisons': [
            {'first': 'modularity-0.5-seed0.npy', 'second': 'modularity-0.5-seed1.npy', 'adjusted_rand': 0.882},
            {'first': 'modularity-1-seed0.npy', 'second': 'modularity-1-seed1.npy', 'adjusted_rand': 0.689},
        ],
        'audit': {'interactions': 45, 'positive_1h': 103_314},
    }
    text = measure.render_report('sys2', work)
    assert 'did:plc' not in text and 'follow_flags' in text
    assert 'Gate A' in text and 'passed. 2 runs finished inside the limits' in text
    assert '**Gate C**' in text and 'passed: `modularity-0.5`' in text
    assert '| modularity-0.5: seed 0 against seed 1 |' in text and '| 0.882 |' in text


def test_leiden_payload_round_trips() -> None:
    payload = rules.leiden_payload(
        rules.LeidenResult(rules.Configuration('modularity', 0.5), 1, True, 12.5, 99, 10, 9, 8, 7, 6, 0.25)
    )
    assert payload['configuration'] == 'modularity-0.5'
    restored = rules.leiden_result(payload)
    assert restored.configuration.label == 'modularity-0.5'
    assert restored.modularity == 0.25 and restored.factions == 7
    with pytest.raises(rules.MeasurementRuleError):
        rules.leiden_result({**payload, 'factions': 'many'})


def test_window_helpers() -> None:
    now = datetime(2026, 11, 27, 13, 45, tzinfo=UTC)
    day = rules.last_full_day(now)
    assert day == Window(datetime(2026, 11, 26, tzinfo=UTC), datetime(2026, 11, 27, tzinfo=UTC))
    window = rules.measurement_window(now, 60)
    assert window.end == datetime(2026, 11, 27, 13, tzinfo=UTC)
    assert window.days == 60
    assert len(rules.extraction_statements(SOURCE, window)) == 180
    assert len(rules.audit_statements(SOURCE, window)) == 60


def test_a_cached_day_is_not_pulled_again(tmp_path: Path) -> None:
    runner = FakeRunner(follows=follow_rows((1, 2, 100, 1000), (2, 1, 101, 1001)))
    first = FakeRunner(follows=runner.follows, reposts=pair_rows((1, 2)))
    measure.extract_edges(first, SOURCE, WINDOW, GraphConfig(2, 2.0, 0.5, True), cache_dir=tmp_path)
    assert len(first.statements) == 6
    assert len(list(tmp_path.glob('*.bin'))) == 6
    again = FakeRunner()
    measure.extract_edges(again, SOURCE, WINDOW, GraphConfig(2, 2.0, 0.5, True), cache_dir=tmp_path)
    assert again.statements == []


def test_leiden_child_writes_its_result_into_the_work_directory(tmp_path: Path) -> None:
    follows = np.asarray([(1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2), (10, 11), (11, 10)], dtype=measure.PAIR_DTYPE)
    edges = aggregate_window_edges(
        follows['src'], follows['dst'], np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.uint64)
    )
    graph = build_graph(edges, GraphConfig(60, 2.0, 0.5, True))
    np.savez(
        tmp_path / measure.EDGES_FILE, u=graph.u, v=graph.v, weight=graph.weight, vertices=np.int64(graph.vertices)
    )
    args = Namespace(
        work_dir=tmp_path,
        objective='modularity',
        resolution=0.5,
        seed=1,
        seed_offset=0,
        repeat=False,
        n_iterations=-1,
        min_faction_size=3,
    )
    assert measure.child_leiden(args) == 0
    stem = tmp_path / measure.LEIDEN_DIR / 'modularity-0.5-seed0'
    assert Path(f'{stem}.npy').is_file()
    payload = measure.read_json(Path(f'{stem}.json'))
    assert payload['vertices'] == 5 and payload['edges'] == 4
    assert payload['factions'] == 1 and payload['unplaced'] == 2
    assert payload['completed'] is True
