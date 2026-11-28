"""The two-faction world as rows of ``osprey_execution_results``, and a client that can delay a commit.

Only the integration tests use this module. They run against a real ClickHouse of the production version,
so the rows here carry the column types the production table has.
"""

import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any

from faction_analysis.schema import RUNS_COLUMNS

POST = 'app.bsky.feed.post'
REPOST = 'app.bsky.feed.repost'
FOLLOW = 'app.bsky.graph.follow'
BLOCK = 'app.bsky.graph.block'
CREATE = 'create'
DELETE = 'delete'
MINUTE = 60_000
DAY = 86_400_000

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DDL_FILE = PROJECT_ROOT / 'sql' / '001_faction_analysis.sql'

# The columns the queries read, with the types the production table has. Every feature column is Nullable
# there, and so it is here.
SOURCE_COLUMNS = (
    '__action_id',
    '__timestamp',
    'AtUri',
    'BlockSubjectDid',
    'Collection',
    'FacetTagList',
    'FollowSubjectDid',
    'Handle',
    'IsStarterPackFollow',
    'OperationKind',
    'Path',
    'PostEmbedRecordUri',
    'PostEmbedRecordWithMediaUri',
    'PostIsReply',
    'PostIsSelfReply',
    'PostLanguages',
    'PostReplyParent',
    'PostReplyParentDid',
    'PostReplyRoot',
    'PostText',
    'RepostSubjectDid',
    'UserId',
)

SOURCE_DDL = """CREATE TABLE {database}.osprey_execution_results (
    __action_id Int64,
    __timestamp DateTime64(3),
    AtUri Nullable(String),
    BlockSubjectDid Nullable(String),
    Collection Nullable(String),
    FacetTagList Array(String),
    FollowSubjectDid Nullable(String),
    Handle Nullable(String),
    IsStarterPackFollow Nullable(UInt8),
    OperationKind Nullable(String),
    Path Nullable(String),
    PostEmbedRecordUri Nullable(String),
    PostEmbedRecordWithMediaUri Nullable(String),
    PostIsReply Nullable(UInt8),
    PostIsSelfReply Nullable(UInt8),
    PostLanguages Array(String),
    PostReplyParent Nullable(String),
    PostReplyParentDid Nullable(String),
    PostReplyRoot Nullable(String),
    PostText Nullable(String),
    RepostSubjectDid Nullable(String),
    UserId Nullable(String)
) ENGINE = MergeTree
PARTITION BY toYYYYMM(__timestamp)
ORDER BY (__action_id, __timestamp)"""


def ddl_statements(database: str) -> list[str]:
    """The sidecar's DDL, one statement per item, written for ``database``."""
    text = re.sub(r'^\s*--.*$', '', DDL_FILE.read_text(), flags=re.MULTILINE)
    return [statement.strip().replace('default.', f'{database}.') for statement in text.split(';') if statement.strip()]


def account_dids() -> list[str]:
    """The twelve accounts: 0 to 5 form the first faction and 6 to 11 the second."""
    return [f'did:plc:{index:024x}' for index in range(12)]


def tags_of(index: int) -> list[str]:
    return ['art', 'ink'] if index < 6 else ['music']


def languages_of(index: int) -> list[str]:
    return ['en'] if index % 2 else ['de', 'en']


def handle_of(index: int) -> str:
    return f'user{index}.bsky.social'


def _moment(at_ms: int) -> datetime:
    return datetime.fromtimestamp(at_ms / 1000, tz=UTC)


class Source:
    """Collects rows for the source table."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self._action = 0

    def add(self, at_ms: int, **values: Any) -> None:
        self._action += 1
        row: dict[str, Any] = {name: None for name in SOURCE_COLUMNS}
        row.update(
            {
                '__action_id': self._action,
                '__timestamp': _moment(at_ms),
                'FacetTagList': [],
                'PostLanguages': [],
                'IsStarterPackFollow': 0,
                'PostIsReply': 0,
                'PostIsSelfReply': 0,
                'Path': '/jetstream',
            }
        )
        row.update(values)
        self.rows.append(row)

    def as_rows(self) -> list[list[Any]]:
        """The rows as tuples in column order, which is what the client inserts."""
        return [[row[name] for name in SOURCE_COLUMNS] for row in self.rows]


def follow_uri(accounts: list[str], a: int, b: int) -> str:
    return f'at://{accounts[a]}/app.bsky.graph.follow/{a}-{b}'


def post_uri(accounts: list[str], index: int, rkey: str) -> str:
    return f'at://{accounts[index]}/app.bsky.feed.post/{rkey}'


def _follow(source: Source, at_ms: int, accounts: list[str], a: int, b: int, **values: Any) -> None:
    source.add(
        at_ms,
        Collection=FOLLOW,
        OperationKind=CREATE,
        UserId=accounts[a],
        FollowSubjectDid=accounts[b],
        AtUri=follow_uri(accounts, a, b),
        **values,
    )


def _post(source: Source, at_ms: int, accounts: list[str], index: int, text: str, rkey: str, **values: Any) -> None:
    # Account 5 has no handle on its post, as some production rows have none.
    source.add(
        at_ms,
        Collection=POST,
        OperationKind=CREATE,
        UserId=accounts[index],
        Handle=None if index == 5 else handle_of(index),
        FacetTagList=tags_of(index),
        PostLanguages=languages_of(index),
        PostText=text,
        AtUri=post_uri(accounts, index, rkey),
        **values,
    )


def recent_ms(now: datetime) -> int:
    """The hour that starts the day before ``now``. Interactions are placed from here on."""
    start = (now - timedelta(days=1)).replace(minute=0, second=0, microsecond=0)
    return int(start.timestamp() * 1000)


def seed(client: Any, table: str, now: datetime, *, split: bool = False) -> list[str]:
    """Insert the two-faction world into ``table``. Returns the twelve DIDs.

    Two cliques of six accounts follow each other in both directions. Across the cliques one follow is
    live, one is deleted two days after its creation and one comes from a starter pack, so only the first
    counts. Accounts 0 and 1 reply to account 6 in one thread, which is a pile-on. Accounts 7 and 8 quote
    one post of account 1, which is a swarm; the second quote attaches media. Account 0 blocks account 6
    and account 2 blocks account 7. The second block is deleted a minute later, so only the first counts.
    Account 3 replies to account 6 without text.

    With ``split`` the nine follow pairs between {0, 1, 2} and {3, 4, 5} are deleted in both directions,
    so the first clique falls apart and the next run sees a split.
    """
    accounts = account_dids()
    recent = recent_ms(now)
    old = recent - 29 * DAY
    source = Source()
    pairs = [(a, b) for start in (0, 6) for a in range(start, start + 6) for b in range(a + 1, start + 6)]
    for a, b in pairs:
        _follow(source, old, accounts, a, b)
        _follow(source, old, accounts, b, a)
    _follow(source, old, accounts, 6, 0)
    _follow(source, old, accounts, 0, 6)
    source.add(old + 2 * DAY, Collection=FOLLOW, OperationKind=DELETE, AtUri=follow_uri(accounts, 0, 6))
    _follow(source, old, accounts, 1, 11, IsStarterPackFollow=1)
    if split:
        delete_between_halves(source, accounts, recent)
    for a, b in ((0, 1), (0, 2), (6, 7)):
        source.add(
            old,
            Collection=REPOST,
            OperationKind=CREATE,
            UserId=accounts[a],
            RepostSubjectDid=accounts[b],
            AtUri=f'at://{accounts[a]}/app.bsky.feed.repost/{a}-{b}',
        )
    # One plain post per account carries its tags, its languages and its handle for the descriptions.
    for index in range(12):
        _post(source, recent + index * MINUTE, accounts, index, 'a plain post', f'p{index}')
    for index, rkey in ((6, '101'), (0, '102'), (6, '103'), (1, '201')):
        _post(source, recent, accounts, index, f'root post {rkey}', rkey)

    def reply(author: int, target: int, text: str, root: str, at_ms: int, rkey: str) -> None:
        _post(
            source,
            at_ms,
            accounts,
            author,
            text,
            rkey,
            PostIsReply=1,
            PostIsSelfReply=0,
            PostReplyParentDid=accounts[target],
            PostReplyRoot=root,
            PostReplyParent=root,
        )

    root = post_uri(accounts, 6, '101')
    reply(0, 6, 'you are an idiot', root, recent, 'r0')
    reply(1, 6, 'nobody asked you!!!', root, recent + MINUTE, 'r1')
    reply(6, 0, 'thanks, that helped', post_uri(accounts, 0, '102'), recent + 2 * MINUTE, 'r6')
    quoted = post_uri(accounts, 1, '201')
    _post(source, recent + 3 * MINUTE, accounts, 7, 'SHUT UP loser', 'q7', PostEmbedRecordUri=quoted)
    # A quote that attaches media carries the quoted post in the other column.
    _post(
        source,
        recent + 4 * MINUTE,
        accounts,
        8,
        'what an idiot',
        'q8',
        PostEmbedRecordWithMediaUri=quoted,
    )
    for author, target, at_ms in ((0, 6, recent + 5 * MINUTE), (2, 7, recent + 6 * MINUTE)):
        source.add(
            at_ms,
            Collection=BLOCK,
            OperationKind=CREATE,
            UserId=accounts[author],
            BlockSubjectDid=accounts[target],
            AtUri=f'at://{accounts[author]}/app.bsky.graph.block/b{author}',
        )
    # The block of account 2 is deleted a minute after it was made, so only the block of account 0 counts.
    source.add(
        recent + 7 * MINUTE,
        Collection=BLOCK,
        OperationKind=DELETE,
        AtUri=f'at://{accounts[2]}/app.bsky.graph.block/b2',
    )
    reply(3, 6, '', post_uri(accounts, 6, '103'), recent + 7 * MINUTE, 'r3')
    client.insert(table, source.as_rows(), column_names=list(SOURCE_COLUMNS))
    return accounts


def delete_between_halves(source: Source, accounts: list[str], at_ms: int) -> None:
    """Delete the nine follow pairs between {0, 1, 2} and {3, 4, 5} in both directions."""
    for a in (0, 1, 2):
        for b in (3, 4, 5):
            source.add(at_ms, Collection=FOLLOW, OperationKind=DELETE, AtUri=follow_uri(accounts, a, b))
            source.add(at_ms, Collection=FOLLOW, OperationKind=DELETE, AtUri=follow_uri(accounts, b, a))


def split_first_clique(client: Any, table: str, now: datetime) -> None:
    """Make the first clique of a seeded world fall apart for the next run."""
    source = Source()
    delete_between_halves(source, account_dids(), recent_ms(now))
    client.insert(table, source.as_rows(), column_names=list(SOURCE_COLUMNS))


class DelayingClient:
    """The real client, with a say in each insert of a run row.

    ``plan`` maps a status of run row to what happens to every insert of such a row, and ``once`` to what
    happens to the first one only: ``send`` (the default), ``hold``, ``fail`` or ``commit_then_fail``. A held
    insert stands for one that reached the server but is not visible yet; ``release`` lets it commit, in
    order. A failing insert stands for a server that cannot be reached. ``commit_then_fail`` stands for a
    commit whose response is lost.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.plan: dict[str, str] = {}
        # ``once`` acts on the first insert of a status and is then spent.
        self.once: dict[str, str] = {}
        self.held: list[Callable[[], None]] = []
        self.run_row_inserts: list[str] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def insert(self, table: str | None = None, data: Any = None, column_names: Any = None, **keywords: Any) -> Any:
        if column_names is not None and list(column_names) == list(RUNS_COLUMNS):
            status = str(data[0][1])
            self.run_row_inserts.append(status)
            action = self.once.pop(status, None) or self.plan.get(status, 'send')
            send = partial(self._inner.insert, table, data, column_names=column_names, **keywords)
            if action == 'hold':
                self.held.append(send)
                return None
            if action == 'fail':
                raise RuntimeError('the server cannot be reached')
            if action == 'commit_then_fail':
                send()
                raise RuntimeError('the response was lost')
        return self._inner.insert(table, data, column_names=column_names, **keywords)

    def release(self) -> int:
        """Let every held insert commit. Returns how many there were."""
        held, self.held = self.held, []
        for send in held:
            send()
        return len(held)
