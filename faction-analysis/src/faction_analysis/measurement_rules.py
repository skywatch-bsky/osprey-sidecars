# pattern: Functional Core
"""Guardrails and statement plans of the production measurement.

The measurement reads production ClickHouse through SSH with ``clickhouse-client --readonly=1``. The project
query rules apply to every statement it sends: SELECT or WITH only, one statement per invocation, no
semicolons, no export clause, a LIMIT on every statement and bounded time, threads and memory.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta

from faction_analysis.config import Objective
from faction_analysis.queries import (
    BLOCK_COLLECTION,
    FOLLOW_COLLECTION,
    POST_COLLECTION,
    QUOTE_PREDICATE,
    QUOTED_AUTHOR,
    QUOTED_COLLECTION,
    REPLY_PREDICATE,
    REPOST_COLLECTION,
    Window,
    deletes_sql,
    follow_creates_sql,
    identifier,
    in_window_sql,
    quote_literal,
    repost_creates_sql,
)

MAX_EXECUTION_TIME = 60
MAX_THREADS = 4
# Ten gigabytes per statement leaves the production host its own headroom.
MAX_MEMORY_USAGE = 10_000_000_000
MIN_FREE_MEMORY = 24 * 2**30
MIN_FREE_DISK = 20 * 2**30
CHILD_MAX_SECONDS = 60 * 60
CHILD_MAX_RSS = 20 * 2**30
# Expected ceilings. A chunk that returns exactly its ceiling stops the measurement as truncated.
EDGE_ROWS_CEILING = 20_000_000
DELETES_ROWS_CEILING = 1_000_000
BINARY = 'RowBinary'
TEXT = 'TabSeparated'
DEFAULT_HOST = 'sys2'
FOLLOW_CREATE_BYTES = 32
PAIR_BYTES = 16
AUDIT_COLUMNS = ('interactions', 'with_block', 'positive_15m', 'positive_1h', 'positive_4h', 'positive_24h')
AUDIT_DELAYS_MINUTES = (15, 60, 240, 1440)
GATE_B_MINIMUM_POSITIVES = 20_000
# Factions (communities of at least the minimum size, after absorption) that the default configuration
# should produce. The operator widened the first range of 100 to 300 to this band on 2026-11-28, to leave
# the calibration room to move. The operator judged 52 factions reasonable.
TARGET_FACTIONS = (20, 300)

_OUTPUT_FORMATS = frozenset({BINARY, TEXT})
_HOST_PATTERN = re.compile(r'^[A-Za-z0-9][A-Za-z0-9.-]*$')
_EXPORT_CLAUSE = re.compile(r'\b(INTO\s+(OUTFILE|FORMAT)|FORMAT\s+[A-Za-z]+)', re.IGNORECASE)
_TRAILING_LIMIT = re.compile(r'\bLIMIT\s+\d+\s*$', re.IGNORECASE)

_PROBE_COLUMNS = (
    'UserId',
    'Handle',
    'Collection',
    'OperationKind',
    'Path',
    'AtUri',
    'PostText',
    'PostLanguages',
    'FacetTagList',
    'PostIsReply',
    'PostIsSelfReply',
    'PostReplyParent',
    'PostReplyParentDid',
    'PostReplyRoot',
    'PostEmbedRecordUri',
    'PostEmbedRecordWithMediaUri',
    'RepostSubjectDid',
    'IsRepost',
    'FollowSubjectDid',
    'FollowSubjectVia',
    'IsStarterPackFollow',
    'IsOrganicFollow',
    'BlockSubjectDid',
    '__timestamp',
)


class MeasurementRuleError(ValueError):
    """A statement or an invocation breaks a query rule of the measurement."""


@dataclass(frozen=True)
class MeasuredStatement:
    """One statement, the format the client must return, the size of a binary row and the row ceiling."""

    name: str
    sql: str
    output_format: str = TEXT
    row_bytes: int | None = None
    ceiling: int = 1


@dataclass(frozen=True)
class Configuration:
    """One point of the Leiden grid."""

    objective: Objective
    resolution: float

    @property
    def label(self) -> str:
        return f'{self.objective}-{self.resolution:g}'


@dataclass(frozen=True)
class LeidenResult:
    """What one Leiden child process reported, or that the parent stopped it."""

    configuration: Configuration
    seed_offset: int
    completed: bool
    seconds: float
    peak_rss: int
    vertices: int = 0
    edges: int = 0
    communities_above_min: int = 0
    factions: int = 0
    unplaced: int = 0
    modularity: float | None = None
    quality: float | None = None


def validate_statement(sql: str) -> None:
    text = sql.strip()
    if not re.match(r'(SELECT|WITH)\b', text, re.IGNORECASE):
        raise MeasurementRuleError('statement must start with SELECT or WITH')
    if ';' in text:
        raise MeasurementRuleError('statement must not contain a semicolon')
    if _EXPORT_CLAUSE.search(text):
        raise MeasurementRuleError('statement must not contain an export clause')
    if not _TRAILING_LIMIT.search(text):
        raise MeasurementRuleError('statement must end with LIMIT n')


def ssh_invocation(host: str, output_format: str) -> tuple[list[str], str]:
    """The ``ssh`` argument vector and the remote command for one statement.

    The statement travels on the standard input of the returned command.
    """
    if not _HOST_PATTERN.fullmatch(host):
        raise MeasurementRuleError(f'invalid ssh host: {host!r}')
    if output_format not in _OUTPUT_FORMATS:
        raise MeasurementRuleError(f'unsupported output format: {output_format!r}')
    remote = (
        'sudo -n docker exec -i clickhouse clickhouse-client'
        f' --readonly=1 --max_execution_time={MAX_EXECUTION_TIME}'
        f' --max_threads={MAX_THREADS} --max_memory_usage={MAX_MEMORY_USAGE}'
        f' --format={output_format}'
    )
    argv = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host, remote]
    return argv, remote


def resource_shortfall(free_memory: int, free_disk: int) -> tuple[str, ...]:
    """What the machine lacks before an extraction may start. Empty when it lacks nothing."""
    problems: list[str] = []
    if free_memory < MIN_FREE_MEMORY:
        problems.append(f'free memory {free_memory / 2**30:.1f} GiB is below {MIN_FREE_MEMORY / 2**30:.0f} GiB')
    if free_disk < MIN_FREE_DISK:
        problems.append(f'free disk {free_disk / 2**30:.1f} GiB is below {MIN_FREE_DISK / 2**30:.0f} GiB')
    return tuple(problems)


def last_full_day(now: datetime) -> Window:
    """The most recent UTC day that has completely elapsed."""
    end = now.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    return Window(end - timedelta(days=1), end)


def measurement_window(now: datetime, days: int = 60) -> Window:
    """The window a run would use at ``now``: ``days`` days back from the start of the current hour."""
    end = now.astimezone(UTC).replace(minute=0, second=0, microsecond=0)
    return Window(end - timedelta(days=days), end)


def _text(name: str, sql: str, ceiling: int = 1) -> MeasuredStatement:
    return MeasuredStatement(name, sql, TEXT, None, ceiling)


def _binary(name: str, sql: str, row_bytes: int, ceiling: int) -> MeasuredStatement:
    return MeasuredStatement(name, sql, BINARY, row_bytes, ceiling)


def probe_statements(source: str, day: Window) -> list[MeasuredStatement]:
    """Bounded statements that confirm the predicate values of the source on one day."""
    table = identifier(source)
    window = in_window_sql(day)
    columns = ', '.join(quote_literal(c) for c in _PROBE_COLUMNS)
    return [
        _text(
            'columns',
            f"""SELECT name, type
FROM system.columns
WHERE database = 'default' AND table = {quote_literal(source)} AND name IN ({columns})
ORDER BY name
LIMIT 50""",
            ceiling=50,
        ),
        _text(
            'collections',
            f"""SELECT Collection, OperationKind, count() AS n
FROM {table}
WHERE {window}
GROUP BY Collection, OperationKind
ORDER BY n DESC
LIMIT 60""",
            ceiling=60,
        ),
        _text(
            'post_flags',
            f"""SELECT
    coalesce(PostIsReply, -1) AS is_reply,
    coalesce(PostIsSelfReply, -1) AS is_self_reply,
    PostReplyParentDid != '' AS has_parent_did,
    coalesce({QUOTED_COLLECTION}, '') AS embed_collection,
    PostEmbedRecordWithMediaUri != '' AS has_media_embed,
    count() AS n
FROM {table}
WHERE Collection = {quote_literal(POST_COLLECTION)} AND OperationKind = 'create' AND {window}
GROUP BY is_reply, is_self_reply, has_parent_did, embed_collection, has_media_embed
ORDER BY n DESC
LIMIT 60""",
            ceiling=60,
        ),
        _text(
            'media_embed_flags',
            f"""SELECT
    coalesce({QUOTED_COLLECTION}, '') AS record_embed_collection,
    splitByChar('/', assumeNotNull(PostEmbedRecordWithMediaUri))[4] AS media_embed_collection,
    count() AS n
FROM {table}
WHERE Collection = {quote_literal(POST_COLLECTION)} AND OperationKind = 'create' AND {window}
  AND PostEmbedRecordWithMediaUri != ''
GROUP BY record_embed_collection, media_embed_collection
ORDER BY n DESC
LIMIT 60""",
            ceiling=60,
        ),
        _text(
            'follow_flags',
            f"""SELECT
    OperationKind,
    coalesce(IsStarterPackFollow, -1) AS starter_pack,
    AtUri != '' AS has_uri,
    FollowSubjectDid != '' AS has_subject,
    count() AS n
FROM {table}
WHERE Collection = {quote_literal(FOLLOW_COLLECTION)} AND {window}
GROUP BY OperationKind, starter_pack, has_uri, has_subject
ORDER BY n DESC
LIMIT 60""",
            ceiling=60,
        ),
        _text(
            'repost_flags',
            f"""SELECT
    OperationKind,
    RepostSubjectDid != '' AS has_subject,
    coalesce(IsRepost, -1) AS is_repost,
    count() AS n
FROM {table}
WHERE Collection = {quote_literal(REPOST_COLLECTION)} AND {window}
GROUP BY OperationKind, has_subject, is_repost
ORDER BY n DESC
LIMIT 60""",
            ceiling=60,
        ),
        _text(
            'block_flags',
            f"""SELECT
    OperationKind,
    AtUri != '' AS has_uri,
    BlockSubjectDid != '' AS has_subject,
    count() AS n
FROM {table}
WHERE Collection = {quote_literal(BLOCK_COLLECTION)} AND {window}
GROUP BY OperationKind, has_uri, has_subject
ORDER BY n DESC
LIMIT 60""",
            ceiling=60,
        ),
        _text(
            'delete_samples',
            f"""SELECT Collection, OperationKind, AtUri, Path
FROM {table}
WHERE Collection IN ({quote_literal(FOLLOW_COLLECTION)}, {quote_literal(BLOCK_COLLECTION)})
  AND OperationKind = 'delete'
  AND {window}
LIMIT 5""",
            ceiling=5,
        ),
        _text(
            'reply_samples',
            f"""SELECT AtUri, PostReplyParent, PostReplyRoot, PostReplyParentDid, PostIsReply, PostIsSelfReply,
       PostEmbedRecordUri, PostEmbedRecordWithMediaUri
FROM {table}
WHERE Collection = {quote_literal(POST_COLLECTION)} AND OperationKind = 'create' AND {window}
  AND {REPLY_PREDICATE}
LIMIT 5""",
            ceiling=5,
        ),
        _text(
            'quote_samples',
            f"""SELECT AtUri, PostIsReply, PostEmbedRecordUri, PostEmbedRecordWithMediaUri
FROM {table}
WHERE Collection = {quote_literal(POST_COLLECTION)} AND OperationKind = 'create' AND {window}
  AND PostEmbedRecordUri != ''
LIMIT 5""",
            ceiling=5,
        ),
        _text(
            'day_hours',
            f"""SELECT toStartOfHour(__timestamp) AS hour, count() AS n
FROM {table}
WHERE {window}
GROUP BY hour
ORDER BY hour
LIMIT 30""",
            ceiling=30,
        ),
    ]


def extraction_statements(source: str, window: Window) -> list[MeasuredStatement]:
    """Follow creates, repost creates and follow deletes, one day per statement."""
    statements: list[MeasuredStatement] = []
    for day in window.day_chunks():
        stamp = day.start.astimezone(UTC).strftime('%Y-%m-%d')
        statements.append(
            _binary(
                f'follows_{stamp}',
                follow_creates_sql(source, day, EDGE_ROWS_CEILING),
                FOLLOW_CREATE_BYTES,
                EDGE_ROWS_CEILING,
            )
        )
        statements.append(
            _binary(
                f'reposts_{stamp}', repost_creates_sql(source, day, EDGE_ROWS_CEILING), PAIR_BYTES, EDGE_ROWS_CEILING
            )
        )
        statements.append(
            _binary(
                f'follow_deletes_{stamp}',
                deletes_sql(source, day, FOLLOW_COLLECTION, DELETES_ROWS_CEILING),
                PAIR_BYTES,
                DELETES_ROWS_CEILING,
            )
        )
    return statements


def block_positive_audit_sql(source: str, interactions: Window, blocks: Window) -> str:
    """Replies and quotes on one day, and how many the target answered with a block within each delay.

    ``blocks`` must cover ``interactions`` and the longest delay. The ASOF join pairs each interaction with
    the first block the target placed on the author at or after the interaction. A LEFT ASOF JOIN fills an
    unmatched row with 0, not NULL, so the gap is NULL unless a block was found: otherwise every interaction
    without a block would count as a positive.
    """
    positives = ', '.join(
        f'countIf(gap_ms <= {minutes * 60_000}) AS positive_{label}'
        for minutes, label in zip(AUDIT_DELAYS_MINUTES, ('15m', '1h', '4h', '24h'), strict=True)
    )
    return f"""SELECT
    count() AS interactions,
    countIf(gap_ms IS NOT NULL) AS with_block,
    {positives}
FROM (
    SELECT
        i.event_ms,
        if(b.block_ms > 0, b.block_ms - i.event_ms, NULL) AS gap_ms
    FROM (
        SELECT
            toInt64(toUnixTimestamp64Milli(__timestamp)) AS event_ms,
            assumeNotNull(UserId) AS author_did,
            assumeNotNull(target_did) AS target_did
        FROM (
            SELECT __timestamp, UserId, PostReplyParentDid, PostIsReply, PostIsSelfReply, PostEmbedRecordUri,
                if({REPLY_PREDICATE}, PostReplyParentDid, {QUOTED_AUTHOR}) AS target_did,
                {REPLY_PREDICATE} AS is_reply,
                {QUOTE_PREDICATE} AS is_quote
            FROM {identifier(source)}
            WHERE Collection = {quote_literal(POST_COLLECTION)}
              AND OperationKind = 'create'
              AND {in_window_sql(interactions)}
        )
        WHERE is_reply OR is_quote
    ) AS i
    LEFT ASOF JOIN (
        SELECT
            assumeNotNull(UserId) AS blocker_did,
            assumeNotNull(BlockSubjectDid) AS blocked_did,
            toInt64(toUnixTimestamp64Milli(__timestamp)) AS block_ms
        FROM {identifier(source)}
        WHERE Collection = {quote_literal(BLOCK_COLLECTION)}
          AND OperationKind = 'create'
          AND {in_window_sql(blocks)}
          AND BlockSubjectDid != ''
          AND UserId != BlockSubjectDid
        ORDER BY blocker_did, blocked_did, block_ms
    ) AS b
        ON i.author_did = b.blocked_did AND i.target_did = b.blocker_did AND i.event_ms <= b.block_ms
)
LIMIT 1"""


def audit_statements(source: str, window: Window) -> list[MeasuredStatement]:
    """One label audit per day. The block side of each statement also covers the next day."""
    statements: list[MeasuredStatement] = []
    for day in window.day_chunks():
        stamp = day.start.astimezone(UTC).strftime('%Y-%m-%d')
        blocks = Window(day.start, min(day.end + timedelta(days=1), window.end + timedelta(days=1)))
        statements.append(_text(f'audit_{stamp}', block_positive_audit_sql(source, day, blocks)))
    return statements


def leiden_payload(result: LeidenResult) -> dict[str, object]:
    """A ``LeidenResult`` as JSON, with the configuration as its label."""
    payload: dict[str, object] = asdict(result)
    payload['configuration'] = result.configuration.label
    return payload


def leiden_result(payload: dict[str, object]) -> LeidenResult:
    """Rebuild a ``LeidenResult`` from ``leiden_payload`` output."""

    def integer(key: str) -> int:
        value = payload.get(key)
        try:
            if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                raise ValueError
            return int(value)
        except ValueError as exc:
            raise MeasurementRuleError(f'{key} is not an integer: {value!r}') from exc

    def number(key: str) -> float | None:
        value = payload.get(key)
        if value is None:
            return None
        try:
            if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                raise ValueError
            return float(value)
        except ValueError as exc:
            raise MeasurementRuleError(f'{key} is not a number: {value!r}') from exc

    label = str(payload['configuration'])
    objective, _, resolution = label.partition('-')
    configuration = Configuration('cpm' if objective == 'cpm' else 'modularity', float(resolution))
    return LeidenResult(
        configuration=configuration,
        seed_offset=integer('seed_offset'),
        completed=bool(payload.get('completed', False)),
        seconds=number('seconds') or 0.0,
        peak_rss=integer('peak_rss'),
        vertices=integer('vertices'),
        edges=integer('edges'),
        communities_above_min=integer('communities_above_min'),
        factions=integer('factions'),
        unplaced=integer('unplaced'),
        modularity=number('modularity'),
        quality=number('quality'),
    )


def gate_a_failures(results: list[LeidenResult]) -> list[str]:
    """Gate A, per run: every Leiden run that completed did so within the time and memory limits.

    A run the parent stopped at a limit is named by ``stopped_runs`` instead. One such configuration does
    not fail the gate: the gate asks whether a run can finish at all, which ``gate_a_passes`` answers.
    """
    failures: list[str] = []
    for result in results:
        if result.completed and (result.seconds > CHILD_MAX_SECONDS or result.peak_rss > CHILD_MAX_RSS):
            name = f'{result.configuration.label} seed+{result.seed_offset}'
            failures.append(
                f'{name}: {result.seconds / 60:.0f} min and {result.peak_rss / 2**30:.1f} GiB exceed the limits'
            )
    return failures


def stopped_runs(results: list[LeidenResult]) -> list[str]:
    """The runs the parent stopped at a limit, with the time and memory they had reached."""
    stopped = []
    for result in results:
        if not result.completed:
            name = f'{result.configuration.label} seed+{result.seed_offset}'
            stopped.append(f'{name} stopped at {result.seconds / 60:.0f} min and {result.peak_rss / 2**30:.1f} GiB')
    return stopped


def gate_a_passes(results: list[LeidenResult]) -> bool:
    """Gate A: at least one Leiden run finished, and no finished run exceeded the limits."""
    return any(result.completed for result in results) and not gate_a_failures(results)


def gate_b_failures(totals: dict[str, int], delay_label: str = 'positive_1h') -> list[str]:
    """Gate B: at least 20,000 block positives at one hour in the window."""
    if totals.get(delay_label, 0) >= GATE_B_MINIMUM_POSITIVES:
        return []
    return [f'{totals.get(delay_label, 0)} block positives at one hour, fewer than {GATE_B_MINIMUM_POSITIVES}']


_MEMBERSHIP_FILE = re.compile(r'^(?P<configuration>.+)-seed(?P<seed>\d+)(?P<repeat>-repeat)?\.npy$')


def _parts(name: object) -> re.Match[str] | None:
    return _MEMBERSHIP_FILE.match(name) if isinstance(name, str) else None


def describe_comparison(item: Mapping[str, object]) -> str:
    """A comparison of two membership files, in words: the configuration and what is compared."""
    first, second = _parts(item.get('first')), _parts(item.get('second'))
    if first is None or second is None:
        return f'{item.get("first")} against {item.get("second")}'
    if first['repeat'] or second['repeat']:
        return f'{first["configuration"]}: seed {first["seed"]} against its repeat'
    return f'{first["configuration"]}: seed {first["seed"]} against seed {second["seed"]}'


def seed_stability(comparisons: Sequence[Mapping[str, object]]) -> dict[str, float]:
    """The adjusted Rand index between two seeds, per configuration label.

    Only a comparison of two different seeds of one configuration counts. A repeat of one seed shows that
    the run is deterministic, not that the partition is stable.
    """
    stability: dict[str, float] = {}
    for item in comparisons:
        first, second = _parts(item.get('first')), _parts(item.get('second'))
        value = item.get('adjusted_rand')
        if first is None or second is None or not isinstance(value, int | float):
            continue
        same_configuration = first['configuration'] == second['configuration']
        two_seeds = not first['repeat'] and not second['repeat'] and first['seed'] != second['seed']
        if same_configuration and two_seeds:
            stability[first['configuration']] = float(value)
    return stability


def choose_default(results: list[LeidenResult], stability: Mapping[str, float] | None = None) -> Configuration | None:
    """Gate C: the completed primary run whose faction count lies in the target range.

    When several lie in range, the one whose seeds agree best wins. ``stability`` maps a configuration label
    to the adjusted Rand index between its seeds; a configuration without such a comparison ranks after
    one with it. A remaining tie goes to the run whose faction count is nearer the middle of the range.
    """
    low, high = TARGET_FACTIONS
    middle = (low + high) / 2
    known = stability or {}

    def rank(result: LeidenResult) -> tuple[int, float, float, str]:
        index = known.get(result.configuration.label)
        compared = index is not None
        return (
            0 if compared else 1,
            -index if compared else 0.0,
            abs(result.factions - middle),
            result.configuration.label,
        )

    in_range = [
        result for result in results if result.completed and result.seed_offset == 0 and low <= result.factions <= high
    ]
    return min(in_range, key=rank).configuration if in_range else None
