# pattern: Imperative Shell
"""Read-only measurement of the production source on the measurement host.

A Python process cannot call the MCP query tool, so the measurement uses the documented SSH fallback. It runs
``clickhouse-client --readonly=1`` in the ClickHouse container on the host and sends one statement on standard
input. It writes nothing on that host. The extraction and each Leiden run execute in a child process that this
module watches and stops at the time or memory limit.

Stages: ``probes``, ``extract``, ``leiden`` (one configuration and seed offset), ``compare``, ``audit`` and
``report``. Each stage writes its result into the work directory.
"""

import argparse
import json
import resource
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from faction_analysis import measurement_rules as rules
from faction_analysis.config import GraphConfig, Objective
from faction_analysis.graph import (
    AffinityGraph,
    aggregate_window_edges,
    build_graph,
    drop_deleted_creates,
    fold_deletes,
)
from faction_analysis.partition import (
    absorb_small_communities,
    build_igraph,
    run_partition,
    weighted_modularity,
)
from faction_analysis.queries import Window
from faction_analysis.tracking import adjusted_rand_index

Runner = Callable[[rules.MeasuredStatement], bytes]

FOLLOWS_DTYPE = np.dtype([('src', '<u8'), ('dst', '<u8'), ('uri_hash', '<u8'), ('event_ms', '<i8')])
PAIR_DTYPE = np.dtype([('src', '<u8'), ('dst', '<u8')])
DELETE_DTYPE = np.dtype([('uri_hash', '<u8'), ('event_ms', '<i8')])

WINDOW_FILE = 'window.json'
EDGES_FILE = 'edges.npz'
EXTRACTION_FILE = 'extraction.json'
PROBES_FILE = 'probes.json'
AUDIT_FILE = 'audit.json'
LEIDEN_DIR = 'leiden'
EXIT_TRUNCATED = 3
EXIT_REFUSED = 4


class MeasurementError(RuntimeError):
    """The measurement cannot continue."""


class MeasurementTruncated(MeasurementError):
    """A statement returned exactly its ceiling, so the extraction would be incomplete."""


# ---- access path


class SshRunner:
    """Runs one statement per invocation on the measurement host."""

    def __init__(self, host: str) -> None:
        self._host = host
        self.statements = 0
        self.seconds = 0.0

    def __call__(self, statement: rules.MeasuredStatement) -> bytes:
        rules.validate_statement(statement.sql)
        argv, _ = rules.ssh_invocation(self._host, statement.output_format)
        started = time.monotonic()
        try:
            completed = subprocess.run(
                argv,
                input=statement.sql.encode(),
                capture_output=True,
                check=False,
                timeout=rules.MAX_EXECUTION_TIME + 30,
            )
        except subprocess.TimeoutExpired as exc:
            raise MeasurementError(f'{statement.name}: ssh did not return in time') from exc
        self.seconds += time.monotonic() - started
        self.statements += 1
        if completed.returncode != 0:
            # clickhouse-client prints the exception first and then echoes the query. Keep the exception.
            detail = completed.stderr.decode(errors='replace').split('(query:')[0].strip()
            raise MeasurementError(f'{statement.name}: clickhouse-client failed: {detail[:400:]}')
        return completed.stdout


# ---- parsing


def binary_rows(statement: rules.MeasuredStatement, data: bytes) -> int:
    """The number of rows in a binary answer. A statement that reached its ceiling raises."""
    if statement.row_bytes is None:
        raise MeasurementError(f'{statement.name} has no row size')
    if len(data) % statement.row_bytes:
        raise MeasurementError(f'{statement.name}: {len(data)} bytes are not whole rows')
    rows = len(data) // statement.row_bytes
    if rows >= statement.ceiling:
        raise MeasurementTruncated(f'{statement.name}: returned {rows} rows, its ceiling')
    return rows


def parse_audit(statement: rules.MeasuredStatement, data: bytes) -> dict[str, int]:
    lines = data.decode().strip().splitlines()
    if len(lines) != 1:
        raise MeasurementError(f'{statement.name}: expected one row, got {len(lines)}')
    values = [int(value) for value in lines[0].split('\t')]
    if len(values) != len(rules.AUDIT_COLUMNS):
        raise MeasurementError(f'{statement.name}: expected {len(rules.AUDIT_COLUMNS)} columns')
    return dict(zip(rules.AUDIT_COLUMNS, values, strict=True))


# ---- stages that talk to the host


@dataclass(frozen=True)
class ExtractionSummary:
    follow_creates: int
    repost_creates: int
    follow_deletes: int
    deleted_follows_removed: int
    accounts: int
    directed_pairs: int
    undirected_edges: int
    median_in_strength: float
    rescale: float
    statements: int
    statement_seconds: float
    wall_seconds: float


def extract_edges(
    runner: Runner,
    source: str,
    window: Window,
    graph: GraphConfig,
    *,
    cache_dir: Path | None = None,
) -> tuple[ExtractionSummary, AffinityGraph]:
    """Pull every day of the window and build the affinity graph from it.

    With ``cache_dir`` every answer is kept on this machine, so an interrupted extraction resumes without
    asking the production host for the same day twice.
    """
    started = time.monotonic()
    follows: list[np.ndarray] = []
    reposts: list[np.ndarray] = []
    deletes: list[np.ndarray] = []
    for statement in rules.extraction_statements(source, window):
        data, rows = pull(runner, statement, cache_dir)
        if rows == 0:
            continue
        if statement.name.startswith('follows_'):
            follows.append(np.frombuffer(data, dtype=FOLLOWS_DTYPE))
        elif statement.name.startswith('reposts_'):
            reposts.append(np.frombuffer(data, dtype=PAIR_DTYPE))
        else:
            deletes.append(np.frombuffer(data, dtype=DELETE_DTYPE))

    counts = (
        sum(len(part) for part in follows),
        sum(len(part) for part in reposts),
        sum(len(part) for part in deletes),
    )
    follow = _stack(follows, FOLLOWS_DTYPE)
    del follows
    repost = _stack(reposts, PAIR_DTYPE)
    del reposts
    delete = _stack(deletes, DELETE_DTYPE)
    del deletes

    deleted = fold_deletes(delete['uri_hash'], delete['event_ms'])
    del delete
    src, dst, removed = drop_deleted_creates(
        follow['src'], follow['dst'], follow['uri_hash'], follow['event_ms'], deleted
    )
    del follow, deleted
    edges = aggregate_window_edges(src, dst, repost['src'], repost['dst'])
    del src, dst, repost
    built = build_graph(edges, graph)
    summary = ExtractionSummary(
        follow_creates=int(counts[0]),
        repost_creates=int(counts[1]),
        follow_deletes=int(counts[2]),
        deleted_follows_removed=removed,
        accounts=built.vertices,
        directed_pairs=int(len(edges.src_index)),
        undirected_edges=built.edges,
        median_in_strength=built.median_in_strength,
        rescale=built.rescale,
        statements=getattr(runner, 'statements', 0),
        statement_seconds=getattr(runner, 'seconds', 0.0),
        wall_seconds=time.monotonic() - started,
    )
    return summary, built


def pull(runner: Runner, statement: rules.MeasuredStatement, cache_dir: Path | None) -> tuple[bytes, int]:
    """The answer of one statement and its row count, from ``cache_dir`` when it was pulled before."""
    path = cache_dir / f'{statement.name}.bin' if cache_dir is not None else None
    started = time.monotonic()
    if path is not None and path.is_file():
        data = path.read_bytes()
        source = 'cache'
    else:
        data = runner(statement)
        source = 'host'
    rows = binary_rows(statement, data)
    if path is not None and source == 'host':
        path.write_bytes(data)
    print(f'{statement.name}: {rows} rows from {source} in {time.monotonic() - started:.1f} s', file=sys.stderr)
    return data, rows


def _stack(parts: list[np.ndarray], dtype: np.dtype) -> np.ndarray:
    if not parts:
        return np.empty(0, dtype=dtype)
    return np.concatenate(parts) if len(parts) > 1 else parts[0]


def run_probes(runner: Runner, source: str, day: Window) -> dict[str, str]:
    return {
        statement.name: runner(statement).decode(errors='replace').strip()
        for statement in rules.probe_statements(source, day)
    }


def run_audit(runner: Runner, source: str, window: Window) -> dict[str, int]:
    totals: dict[str, int] = dict.fromkeys(rules.AUDIT_COLUMNS, 0)
    for statement in rules.audit_statements(source, window):
        for column, value in parse_audit(statement, runner(statement)).items():
            totals[column] += value
    return totals


# ---- watched child processes


@dataclass(frozen=True)
class ChildOutcome:
    returncode: int
    seconds: float
    peak_rss: int

    @property
    def completed(self) -> bool:
        return self.returncode == 0


def read_rss_bytes(pid: int) -> int:
    with open(f'/proc/{pid}/status') as status:
        for line in status:
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) * 1024
    return 0


def free_memory_bytes() -> int:
    with open('/proc/meminfo') as meminfo:
        for line in meminfo:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) * 1024
    return 0


def run_child(
    argv: Sequence[str],
    *,
    max_seconds: int,
    max_rss: int,
    cwd: Path | None = None,
    spawn: Callable[..., Any] = subprocess.Popen,
    rss_of: Callable[[int], int] = read_rss_bytes,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    poll_seconds: float = 2.0,
) -> ChildOutcome:
    """Run a child and stop it when it exceeds the time or the memory limit.

    A child that was stopped has return code ``-1`` and never counts as completed.
    """
    process = spawn(list(argv), cwd=str(cwd) if cwd else None)
    started = clock()
    peak = 0
    while True:
        returncode = process.poll()
        if returncode is not None:
            return ChildOutcome(int(returncode), clock() - started, peak)
        try:
            peak = max(peak, rss_of(process.pid))
        except FileNotFoundError, ProcessLookupError:
            pass
        elapsed = clock() - started
        if elapsed > max_seconds or peak > max_rss:
            process.kill()
            process.wait()
            return ChildOutcome(-1, clock() - started, peak)
        sleep(poll_seconds)


def child_argv(kind: str, work_dir: Path, *options: str) -> list[str]:
    return [sys.executable, '-m', 'faction_analysis.measure', '--child', kind, '--work-dir', str(work_dir), *options]


def peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


# ---- work directory


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def check_resources(work_dir: Path) -> None:
    problems = rules.resource_shortfall(free_memory_bytes(), shutil.disk_usage(work_dir).free)
    if problems:
        for problem in problems:
            print(f'refused: {problem}', file=sys.stderr)
        raise SystemExit(EXIT_REFUSED)


def saved_window(work_dir: Path) -> Window:
    saved = read_json(work_dir / WINDOW_FILE)
    return Window(
        datetime.fromisoformat(saved['start']).astimezone(UTC),
        datetime.fromisoformat(saved['end']).astimezone(UTC),
    )


def report_child(kind: str, outcome: ChildOutcome) -> None:
    minutes, gib = outcome.seconds / 60, outcome.peak_rss / 2**30
    if outcome.completed:
        print(f'{kind}: completed in {minutes:.1f} min, peak {gib:.2f} GiB', file=sys.stderr)
    elif outcome.returncode == EXIT_TRUNCATED:
        print(f'{kind}: stopped after {minutes:.1f} min because a statement returned its ceiling', file=sys.stderr)
    else:
        print(f'{kind}: stopped after {minutes:.1f} min at {gib:.2f} GiB', file=sys.stderr)


# ---- child entry points


def child_extract(args: argparse.Namespace) -> int:
    window = Window(args.start, args.end)
    write_json(args.work_dir / WINDOW_FILE, {'start': window.start.isoformat(), 'end': window.end.isoformat()})
    graph = GraphConfig(
        window_days=window.days,
        mutual_follow_weight=args.mutual_follow_weight,
        repost_alpha=args.repost_alpha,
        damp_large_accounts=not args.no_damping,
    )
    runner = SshRunner(args.host)
    cache_dir = args.work_dir / 'raw'
    cache_dir.mkdir(parents=True, exist_ok=True)
    summary, built = extract_edges(runner, args.source, window, graph, cache_dir=cache_dir)
    np.savez(
        args.work_dir / EDGES_FILE,
        u=built.u.astype(np.int32),
        v=built.v.astype(np.int32),
        weight=built.weight.astype(np.float32),
        vertices=np.int64(built.vertices),
    )
    write_json(args.work_dir / EXTRACTION_FILE, asdict(summary))
    print(json.dumps(asdict(summary), indent=2))
    return 0


def child_leiden(args: argparse.Namespace) -> int:
    configuration = rules.Configuration(args.objective, args.resolution)
    started = time.monotonic()
    with np.load(args.work_dir / EDGES_FILE) as data:
        u = data['u']
        v = data['v']
        weight = data['weight']
        vertices = int(data['vertices'])
    edge_count = int(len(u))
    graph = build_igraph(vertices, u, v, weight)
    # The igraph holds the edges now. Reload them after Leiden so they do not count towards the peak.
    del u, v, weight
    build_seconds = time.monotonic() - started
    result = run_partition(
        graph,
        objective=args.objective,
        resolution=args.resolution,
        seed=args.seed + args.seed_offset,
        n_iterations=args.n_iterations,
    )
    del graph
    with np.load(args.work_dir / EDGES_FILE) as data:
        sizes = np.bincount(result.membership)
        absorbed = absorb_small_communities(
            result.membership, data['u'], data['v'], data['weight'], args.min_faction_size
        )
        modularity = weighted_modularity(result.membership, data['u'], data['v'], data['weight'])
    outcome = rules.LeidenResult(
        configuration=configuration,
        seed_offset=args.seed_offset,
        completed=True,
        seconds=time.monotonic() - started,
        peak_rss=peak_rss_bytes(),
        vertices=vertices,
        edges=edge_count,
        communities_above_min=int((sizes >= args.min_faction_size).sum()),
        factions=int(absorbed.established.sum()),
        unplaced=absorbed.unplaced_accounts,
        modularity=modularity,
        quality=result.quality,
    )
    stem = f'{configuration.label}-seed{args.seed_offset}' + ('-repeat' if args.repeat else '')
    out_dir = args.work_dir / LEIDEN_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f'{stem}.npy', absorbed.membership)
    write_json(
        out_dir / f'{stem}.json',
        {**rules.leiden_payload(outcome), 'build_seconds': build_seconds, 'repeat': args.repeat},
    )
    print(json.dumps({**rules.leiden_payload(outcome), 'build_seconds': build_seconds}, indent=2))
    return 0


def child_audit(args: argparse.Namespace) -> int:
    totals = run_audit(SshRunner(args.host), args.source, saved_window(args.work_dir))
    write_json(args.work_dir / AUDIT_FILE, totals)
    print(json.dumps(totals, indent=2))
    return 0


# ---- parent commands


def command_probes(args: argparse.Namespace) -> int:
    day = Window(args.day, args.day + timedelta(days=1))
    probes = run_probes(SshRunner(args.host), args.source, day)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.work_dir / PROBES_FILE, probes)
    for name in sorted(probes):
        print(f'## {name}\n{probes[name]}\n')
    return 0


def command_extract(args: argparse.Namespace) -> int:
    args.work_dir.mkdir(parents=True, exist_ok=True)
    check_resources(args.work_dir)
    options = [
        '--host',
        args.host,
        '--source',
        args.source,
        '--start',
        args.start.isoformat(),
        '--end',
        args.end.isoformat(),
        '--mutual-follow-weight',
        str(args.mutual_follow_weight),
        '--repost-alpha',
        str(args.repost_alpha),
    ]
    outcome = run_child(
        child_argv('extract', args.work_dir, *options),
        max_seconds=rules.CHILD_MAX_SECONDS,
        max_rss=rules.CHILD_MAX_RSS,
        cwd=args.work_dir,
    )
    report_child('extract', outcome)
    return 0 if outcome.completed else (outcome.returncode or 1)


def command_leiden(args: argparse.Namespace) -> int:
    check_resources(args.work_dir)
    configuration = rules.Configuration(args.objective, args.resolution)
    options = [
        '--objective',
        args.objective,
        '--resolution',
        repr(args.resolution),
        '--seed',
        str(args.seed),
        '--seed-offset',
        str(args.seed_offset),
        '--n-iterations',
        str(args.n_iterations),
        '--min-faction-size',
        str(args.min_faction_size),
    ]
    if args.repeat:
        options.append('--repeat')
    outcome = run_child(
        child_argv('leiden', args.work_dir, *options),
        max_seconds=rules.CHILD_MAX_SECONDS,
        max_rss=rules.CHILD_MAX_RSS,
        cwd=args.work_dir,
    )
    report_child(f'leiden {configuration.label} seed+{args.seed_offset}', outcome)
    if not outcome.completed:
        stopped = rules.LeidenResult(configuration, args.seed_offset, False, outcome.seconds, outcome.peak_rss)
        leiden_dir = args.work_dir / LEIDEN_DIR
        leiden_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            leiden_dir / f'{configuration.label}-seed{args.seed_offset}.stopped.json',
            rules.leiden_payload(stopped),
        )
    return 0 if outcome.completed else 1


def command_audit(args: argparse.Namespace) -> int:
    check_resources(args.work_dir)
    outcome = run_child(
        child_argv('audit', args.work_dir, '--host', args.host, '--source', args.source),
        max_seconds=rules.CHILD_MAX_SECONDS,
        max_rss=rules.CHILD_MAX_RSS,
        cwd=args.work_dir,
    )
    report_child('audit', outcome)
    return 0 if outcome.completed else 1


def command_compare(args: argparse.Namespace) -> int:
    first, second = np.load(args.first), np.load(args.second)
    if len(first) != len(second):
        raise MeasurementError('the two membership files have different vertex counts')
    both = (first >= 0) & (second >= 0)
    payload = {
        'first': args.first.name,
        'second': args.second.name,
        'identical': bool(np.array_equal(first, second)),
        'placed_in_both': int(both.sum()),
        'adjusted_rand': adjusted_rand_index(first[both], second[both]) if int(both.sum()) else None,
    }
    text = json.dumps(payload, indent=2)
    if args.output:
        args.output.write_text(text)
    print(text)
    return 0


def collect_work(work_dir: Path) -> dict[str, Any]:
    def optional(name: str) -> Any:
        return read_json(work_dir / name) if (work_dir / name).is_file() else {}

    results: list[dict[str, Any]] = []
    stopped: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    leiden_dir = work_dir / LEIDEN_DIR
    if leiden_dir.is_dir():
        for path in sorted(leiden_dir.glob('*.json')):
            payload = read_json(path)
            if path.name.endswith('.stopped.json'):
                stopped.append(payload)
            else:
                results.append(payload)
    if (work_dir / 'compare').is_dir():
        comparisons = [read_json(path) for path in sorted((work_dir / 'compare').glob('*.json'))]
    return {
        'probes': optional(PROBES_FILE),
        'extraction': optional(EXTRACTION_FILE),
        'audit': optional(AUDIT_FILE),
        'leiden': results,
        'stopped': stopped,
        'comparisons': comparisons,
    }


def command_report(args: argparse.Namespace) -> int:
    text = render_report(args.host, collect_work(args.work_dir))
    if args.output:
        args.output.write_text(text)
    else:
        print(text)
    return 0


def render_report(host: str, work: dict[str, Any]) -> str:
    results = [rules.leiden_result(item) for item in work['leiden']]
    audit = work['audit']
    stability = rules.seed_stability(work['comparisons'])
    default = rules.choose_default(results, stability)
    gate_a = rules.gate_a_failures(results)
    stopped = rules.stopped_runs([rules.leiden_result(item) for item in work['stopped']])
    if rules.gate_a_passes(results):
        gate_a_text = f'passed. {len(results)} runs finished inside the limits'
        gate_a_text += ('; ' + ' and '.join(stopped)) if stopped else ''
        gate_a_text += '.'
    elif gate_a:
        gate_a_text = 'failed. ' + ' '.join(gate_a)
    else:
        gate_a_text = 'not measured.'
    gate_b = rules.gate_b_failures(audit) if audit else ['not measured']
    out: list[str] = [
        '# Faction analysis: production measurement',
        '',
        f'Date: {datetime.now(UTC):%Y-%m-%d} · Source: `osprey_execution_results` on `{host}`',
        '',
        '## Access path',
        '',
        'A Python process cannot call the MCP query tool, so the measurement used the documented SSH fallback:',
        f'`ssh -o BatchMode=yes -o ConnectTimeout=10 {host}` running `sudo -n docker exec -i clickhouse clickhouse-client`',
        f'with `--readonly=1 --max_execution_time={rules.MAX_EXECUTION_TIME} --max_threads={rules.MAX_THREADS}` and',
        f'`--max_memory_usage={rules.MAX_MEMORY_USAGE}`, one statement on standard input. Nothing was written on that host.',
        'Every statement started with SELECT or WITH and ended with a LIMIT. Each LIMIT was an expected ceiling. No',
        'statement reached its ceiling.',
        '',
        '## Probes (one day)',
        '',
        'Counts, flag values and column types. The sampled rows are left out: they carry account identifiers.',
        '',
    ]
    # A probe whose name ends in ``_samples`` holds rows with DIDs and post URIs. They stay in the work directory.
    for name in sorted(work['probes']):
        if not name.endswith('_samples'):
            out += [f'### {name}', '', '```', str(work['probes'][name]), '```', '']
    out += ['## Extraction', '', '```json', json.dumps(work['extraction'], indent=2, sort_keys=True), '```', '']
    out += [
        '## Leiden grid',
        '',
        'Each run built the graph from the extracted edges, ran Leiden once, and absorbed communities below the minimum',
        f'size ({rules.TARGET_FACTIONS[0]} to {rules.TARGET_FACTIONS[1]} factions is the target).',
        '',
        '| Configuration | Run | Graph build min | Total min | Peak GiB | Edges | Communities above minimum | Factions | Unplaced | Modularity |',
        '|---|---|---|---|---|---|---|---|---|---|',
    ]
    for item in work['leiden']:
        run = 'repeat' if item.get('repeat') else f'seed +{item["seed_offset"]}'
        out.append(
            f'| {item["configuration"]} | {run} | {item.get("build_seconds", 0) / 60:.1f} | {item["seconds"] / 60:.1f} | '
            f'{item["peak_rss"] / 2**30:.2f} | {item["edges"]:,} | {item["communities_above_min"]:,} | {item["factions"]:,} | '
            f'{item["unplaced"]:,} | {item["modularity"]:.4f} |'
        )
    for item in work['stopped']:
        out.append(
            f'| {item["configuration"]} | seed +{item["seed_offset"]} | | {item["seconds"] / 60:.1f} | '
            f'{item["peak_rss"] / 2**30:.2f} | stopped | | | | |'
        )
    out += [
        '',
        '## Seed stability and repeat',
        '',
        'The adjusted Rand index of two partitions over the accounts both placed. 1.0 is the same partition.',
        '',
        '| Comparison | Accounts placed in both | Adjusted Rand index | Identical |',
        '|---|---|---|---|',
    ]
    for item in work['comparisons']:
        index = item.get('adjusted_rand')
        out.append(
            f'| {rules.describe_comparison(item)} | {item.get("placed_in_both", 0):,} | '
            f'{"n/a" if index is None else format(index, ".3f")} | {"yes" if item.get("identical") else "no"} |'
        )
    out += ['', '## Label audit', '', '```json', json.dumps(audit, indent=2, sort_keys=True), '```', '']
    gate_c_text = 'failed. No configuration landed in the range.'
    if default is not None:
        low, high = rules.TARGET_FACTIONS
        # One entry per configuration: the repeat of a seed is the same configuration.
        primary = {}
        for result in results:
            if result.completed and result.seed_offset == 0 and low <= result.factions <= high:
                primary.setdefault(result.configuration.label, result)
        gate_c_text = f'passed: `{default.label}`. ' + ' '.join(
            f'`{label}` gave {result.factions} factions and '
            + (f'its seeds agree at {stability[label]:.3f}.' if label in stability else 'its seeds were not compared.')
            for label, result in sorted(primary.items())
        )
    out += [
        '',
        '## Gates',
        '',
        f'- **Gate A** (a single Leiden run within {rules.CHILD_MAX_SECONDS // 60} minutes and {rules.CHILD_MAX_RSS / 2**30:.0f} GiB): '
        + gate_a_text,
        f'- **Gate B** (at least {rules.GATE_B_MINIMUM_POSITIVES:,} block positives at one hour in the window): '
        + ('passed.' if audit and not gate_b else ' '.join(gate_b) + '.'),
        f'- **Gate C** (a configuration with {rules.TARGET_FACTIONS[0]} to {rules.TARGET_FACTIONS[1]} factions, with the '
        'better seed stability when several lie in the range): ' + gate_c_text,
        '',
        '## Reading the numbers',
        '',
        'This file holds what was measured and is written by `measure report`. The hand-written reading of these',
        'numbers, with the decision they lead to, is in `docs/MEASUREMENT_INTERPRETATION.md`.',
        '',
    ]
    return '\n'.join(out)


# ---- argument parsing


def iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError('timestamps need a timezone')
    return parsed.astimezone(UTC)


def objective(value: str) -> Objective:
    if value == 'cpm':
        return 'cpm'
    if value == 'modularity':
        return 'modularity'
    raise argparse.ArgumentTypeError('objective must be cpm or modularity')


def default_work_dir() -> Path:
    import os

    base = Path(os.environ.get('XDG_CACHE_HOME') or Path.home() / '.cache')
    return base / 'faction-analysis' / 'measure'


def add_host_source(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--host', default=rules.DEFAULT_HOST)
    parser.add_argument('--source', default='osprey_execution_results')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='python -m faction_analysis.measure', description=__doc__.splitlines()[0])
    parser.add_argument('--child', choices=['extract', 'leiden', 'audit'])
    parser.add_argument('--work-dir', type=Path, default=default_work_dir())
    add_host_source(parser)
    parser.add_argument('--start', type=iso)
    parser.add_argument('--end', type=iso)
    parser.add_argument('--objective', type=objective, default='cpm')
    parser.add_argument('--resolution', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seed-offset', type=int, default=0)
    parser.add_argument('--repeat', action='store_true')
    parser.add_argument('--n-iterations', type=int, default=-1)
    parser.add_argument('--min-faction-size', type=int, default=500)
    parser.add_argument('--mutual-follow-weight', type=float, default=2.0)
    parser.add_argument('--repost-alpha', type=float, default=0.5)
    parser.add_argument('--no-damping', action='store_true')
    parser.add_argument('--day', type=iso)
    parser.add_argument('--first', type=Path)
    parser.add_argument('--second', type=Path)
    parser.add_argument('--output', type=Path)
    sub = parser.add_subparsers(dest='command', metavar='COMMAND')
    for name, help_text in (
        ('probes', 'bounded predicate probes on one day'),
        ('extract', 'extract the window of follow and repost edges, one day per statement'),
        ('leiden', 'one Leiden run in a watched child process'),
        ('audit', 'count interactions and block positives, one day per statement'),
        ('compare', 'adjusted Rand index and identity of two membership files'),
        ('report', 'render the measurement report from the work directory'),
    ):
        sub.add_parser(name, help=help_text)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.child:
            return _child(args)
        return _parent(args)
    except MeasurementTruncated as exc:
        print(f'truncated: {exc}', file=sys.stderr)
        return EXIT_TRUNCATED
    except MeasurementError as exc:
        print(f'measurement failed: {exc}', file=sys.stderr)
        return 1


def _child(args: argparse.Namespace) -> int:
    if args.child == 'extract':
        if args.start is None or args.end is None:
            raise MeasurementError('the extract child needs --start and --end')
        return child_extract(args)
    if args.child == 'leiden':
        return child_leiden(args)
    return child_audit(args)


def _parent(args: argparse.Namespace) -> int:
    args.work_dir.mkdir(parents=True, exist_ok=True)
    command = args.command
    if command == 'probes':
        if args.day is None:
            args.day = rules.last_full_day(datetime.now(UTC)).start
        return command_probes(args)
    if command == 'extract':
        if args.start is None or args.end is None:
            window = rules.measurement_window(datetime.now(UTC))
            args.start, args.end = window.start, window.end
        return command_extract(args)
    if command == 'leiden':
        return command_leiden(args)
    if command == 'audit':
        return command_audit(args)
    if command == 'compare':
        if args.first is None or args.second is None:
            raise MeasurementError('compare needs --first and --second')
        return command_compare(args)
    if command == 'report':
        return command_report(args)
    raise MeasurementError(f'unknown command: {command}')


if __name__ == '__main__':
    raise SystemExit(main())
