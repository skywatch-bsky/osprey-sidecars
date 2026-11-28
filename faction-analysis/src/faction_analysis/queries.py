# pattern: Functional Core
"""SQL builders for the source table and for the output tables.

Every source statement filters ``__timestamp``. A table name passes ``identifier`` and a value passes
``quote_literal`` before either enters a statement, so no builder interpolates unchecked text.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

POST_COLLECTION = 'app.bsky.feed.post'
REPOST_COLLECTION = 'app.bsky.feed.repost'
FOLLOW_COLLECTION = 'app.bsky.graph.follow'
BLOCK_COLLECTION = 'app.bsky.graph.block'
OPERATION_CREATE = 'create'
OPERATION_DELETE = 'delete'

_SOURCE_COLLECTIONS = (POST_COLLECTION, REPOST_COLLECTION, FOLLOW_COLLECTION, BLOCK_COLLECTION)


class QueryError(ValueError):
    """A caller passed a value that cannot be put into a statement safely."""


@dataclass(frozen=True)
class Window:
    """Half-open time range ``[start, end)`` in UTC."""

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        for value in (self.start, self.end):
            if value.tzinfo is None or value.utcoffset() is None:
                raise QueryError('window bounds must be timezone-aware')
        if self.end <= self.start:
            raise QueryError('window end must be after window start')

    @property
    def days(self) -> int:
        seconds = (self.end - self.start).total_seconds()
        return -(-int(seconds) // 86_400)

    def day_chunks(self) -> list[Window]:
        """Consecutive one-day windows, the last of which may be shorter."""
        chunks: list[Window] = []
        cursor = self.start
        while cursor < self.end:
            stop = min(cursor + timedelta(days=1), self.end)
            chunks.append(Window(cursor, stop))
            cursor = stop
        return chunks


def _timestamp(value: datetime) -> str:
    utc = value.astimezone(UTC)
    return f"toDateTime64('{utc:%Y-%m-%d %H:%M:%S}.{utc.microsecond // 1000:03d}', 3, 'UTC')"


def quote_literal(value: str) -> str:
    return "'" + value.replace('\\', '\\\\').replace("'", "\\'") + "'"


_IDENTIFIER_PATTERN = re.compile(r'^[a-zA-Z0-9_.]+$')


def identifier(value: str) -> str:
    """A table name. Only characters that can appear in an identifier are accepted."""
    if not _IDENTIFIER_PATTERN.fullmatch(value):
        raise QueryError(f'invalid table identifier: {value!r}')
    return value


# A reply addresses the author of its parent. A self-reply, or a reply without a parent author, is not an
# interaction with another account.
REPLY_PREDICATE = """(coalesce(PostIsReply, 0) = 1
            AND coalesce(PostIsSelfReply, 0) = 0
            AND PostReplyParentDid != ''
            AND PostReplyParentDid != UserId)"""
# The quoted post of a quote. A quote that also attaches media carries the post in PostEmbedRecordWithMediaUri
# and leaves PostEmbedRecordUri empty (about a sixth of the quotes on a day), so both columns are read.
QUOTED_URI = (
    "if(PostEmbedRecordUri != '', assumeNotNull(PostEmbedRecordUri), assumeNotNull(PostEmbedRecordWithMediaUri))"
)
# Parts of an AT-URI such as at://did:plc:abc/app.bsky.feed.post/rkey, split on '/'. The third part is the
# authority (the DID) and the fourth is the collection. An index past the end gives ''.
QUOTED_AUTHOR = f"splitByChar('/', {QUOTED_URI})[3]"
QUOTED_COLLECTION = f"splitByChar('/', {QUOTED_URI})[4]"
# A quote embeds a post of another account. The author is the authority of the embedded URI.
QUOTE_PREDICATE = f"""({QUOTED_URI} != ''
            AND {QUOTED_COLLECTION} = {quote_literal(POST_COLLECTION)}
            AND {QUOTED_AUTHOR} != UserId)"""


# Source string columns are Nullable. ``assumeNotNull`` keeps a hash a plain UInt64, which a fixed-size
# binary row and a NumPy array both need. The predicates exclude empty values, so NULL never passes.
def _hash(column: str) -> str:
    return f'cityHash64(assumeNotNull({column}))'


def _limit(limit: int | None) -> str:
    if limit is None:
        return ''
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise QueryError(f'limit must be a positive integer or None: {limit!r}')
    return f'\nLIMIT {limit}'


def in_window_sql(window: Window) -> str:
    return f'__timestamp >= {_timestamp(window.start)}\n  AND __timestamp < {_timestamp(window.end)}'


def _source(source: str, window: Window, collection: str, operation: str) -> str:
    return f"""FROM {identifier(source)}
WHERE Collection = {quote_literal(collection)}
  AND OperationKind = {quote_literal(operation)}
  AND {in_window_sql(window)}"""


def follow_creates_sql(source: str, window: Window, limit: int | None = None) -> str:
    """Columns: src, dst, uri_hash (UInt64) and event_ms (Int64). One row per live starter-pack-free follow create.

    Deletes are subtracted by the caller from ``deletes_sql`` output, because a delete can land on a later day
    than its create.
    """
    return f"""SELECT
    {_hash('UserId')} AS src,
    {_hash('FollowSubjectDid')} AS dst,
    {_hash('AtUri')} AS uri_hash,
    toUnixTimestamp64Milli(__timestamp) AS event_ms
{_source(source, window, FOLLOW_COLLECTION, OPERATION_CREATE)}
  AND coalesce(IsStarterPackFollow, 0) = 0
  AND UserId != ''
  AND FollowSubjectDid != ''
  AND UserId != FollowSubjectDid{_limit(limit)}"""


def repost_creates_sql(source: str, window: Window, limit: int | None = None) -> str:
    """Columns: src, dst (UInt64). One row per repost create."""
    return f"""SELECT
    {_hash('UserId')} AS src,
    {_hash('RepostSubjectDid')} AS dst
{_source(source, window, REPOST_COLLECTION, OPERATION_CREATE)}
  AND RepostSubjectDid != ''
  AND UserId != RepostSubjectDid{_limit(limit)}"""


def deletes_sql(source: str, window: Window, collection: str, limit: int | None = None) -> str:
    """Columns: uri_hash (UInt64) and event_ms (Int64). One row per delete.

    Delete rows carry no subject. The record is identified by its URI.
    """
    if collection not in {FOLLOW_COLLECTION, BLOCK_COLLECTION}:
        raise QueryError(f'deletes are only defined for follow and block records: {collection!r}')
    return f"""SELECT
    {_hash('AtUri')} AS uri_hash,
    toUnixTimestamp64Milli(__timestamp) AS event_ms
{_source(source, window, collection, OPERATION_DELETE)}
  AND AtUri != ''{_limit(limit)}"""


def account_hashes_sql(source: str, window: Window, limit: int | None = None) -> str:
    """Columns: h (UInt64), did (String) and dids (UInt64). One row per account hash in the window.

    ``dids`` is the number of distinct DIDs that hash to ``h``. A hash with ``dids`` above one is a collision.
    """
    collections = ', '.join(quote_literal(c) for c in _SOURCE_COLLECTIONS)
    # Every column is qualified: an alias equal to a column name (``min(did) AS did``) would otherwise be
    # substituted into the other expressions of the same SELECT.
    return f"""SELECT
    cityHash64(a.did) AS h,
    min(a.did) AS did,
    uniqExact(a.did) AS dids
FROM (
    SELECT arrayJoin(arrayFilter(x -> x != '', [
        coalesce(UserId, ''),
        coalesce(FollowSubjectDid, ''),
        coalesce(RepostSubjectDid, ''),
        coalesce(BlockSubjectDid, ''),
        coalesce(PostReplyParentDid, ''),
        coalesce({QUOTED_AUTHOR}, '')
    ])) AS did
    FROM {identifier(source)}
    WHERE Collection IN ({collections})
      AND OperationKind = {quote_literal(OPERATION_CREATE)}
      AND {in_window_sql(window)}
) AS a
GROUP BY h{_limit(limit)}"""


# Structural features of the text, computed where the text is read. ClickHouse counts them on tens of
# millions of rows at no cost, and the training set and the scoring read the same columns. ``uppercase`` is
# the count of ASCII uppercase letters: shouting in Latin script.
STRUCTURAL_COLUMNS = """toUInt32(lengthUTF8(assumeNotNull(PostText))) AS text_length,
    toUInt32(countMatches(assumeNotNull(PostText), '[A-Z]')) AS uppercase,
    toUInt32(countSubstrings(assumeNotNull(PostText), '!')) AS exclamations,
    toUInt32(countSubstrings(assumeNotNull(PostText), '?')) AS questions,
    toUInt32(countSubstrings(assumeNotNull(PostText), '@')) AS mentions"""
STRUCTURAL_ZERO = """toUInt32(0) AS text_length,
    toUInt32(0) AS uppercase,
    toUInt32(0) AS exclamations,
    toUInt32(0) AS questions,
    toUInt32(0) AS mentions"""


def interactions_sql(source: str, window: Window, limit: int | None = None) -> str:
    """Columns: event_ms, kind, uri_hash, author_did, target_did, thread_key, parent_uri, text, languages and
    the five structural columns.

    ``kind`` is 0 for a reply, 1 for a quote and 2 for a block. A post that is both a reply and a quote is
    counted once, as a reply. ``thread_key`` is the hash of the thread root for a reply and of the quoted post
    for a quote. ``text`` is NULL, ``parent_uri`` is empty and the structural columns are 0 for a block.
    """
    return f"""SELECT
    toInt64(toUnixTimestamp64Milli(__timestamp)) AS event_ms,
    if(is_reply, toInt8(0), toInt8(1)) AS kind,
    {_hash('AtUri')} AS uri_hash,
    UserId AS author_did,
    if(is_reply, PostReplyParentDid, {QUOTED_AUTHOR}) AS target_did,
    if(is_reply, {_hash('PostReplyRoot')}, cityHash64({QUOTED_URI})) AS thread_key,
    if(is_reply, PostReplyParent, {QUOTED_URI}) AS parent_uri,
    PostText AS text,
    PostLanguages AS languages,
    {STRUCTURAL_COLUMNS}
FROM (
    SELECT
        __timestamp,
        AtUri,
        UserId,
        PostReplyParent,
        PostReplyParentDid,
        PostReplyRoot,
        PostEmbedRecordUri,
        PostEmbedRecordWithMediaUri,
        PostText,
        PostLanguages,
        {REPLY_PREDICATE} AS is_reply,
        {QUOTE_PREDICATE} AS is_quote
    {_source(source, window, POST_COLLECTION, OPERATION_CREATE)}
)
WHERE is_reply OR is_quote
UNION ALL
SELECT
    toInt64(toUnixTimestamp64Milli(__timestamp)) AS event_ms,
    toInt8(2) AS kind,
    {_hash('AtUri')} AS uri_hash,
    UserId AS author_did,
    BlockSubjectDid AS target_did,
    toUInt64(0) AS thread_key,
    '' AS parent_uri,
    NULL AS text,
    CAST([], 'Array(String)') AS languages,
    {STRUCTURAL_ZERO}
{_source(source, window, BLOCK_COLLECTION, OPERATION_CREATE)}
  AND BlockSubjectDid != ''
  AND UserId != BlockSubjectDid{_limit(limit)}"""


def daily_row_counts_sql(source: str, window: Window, limit: int | None = None) -> str:
    """Columns: day (Date) and rows (UInt64). Source rows per day, for gap detection."""
    return f"""SELECT
    toDate(__timestamp) AS day,
    count() AS rows
FROM {identifier(source)}
WHERE {in_window_sql(window)}
GROUP BY day
ORDER BY day{_limit(limit)}"""


def _image_subquery(members_table: str, run_id: str) -> str:
    return f"""(
    SELECT did, faction_id
    FROM {identifier(members_table)}
    WHERE run_id = {quote_literal(run_id)}
)"""


def faction_tags_sql(source: str, members_table: str, run_id: str, window: Window, per_faction: int = 10) -> str:
    """Columns: faction_id, tag and n. The most used tags per faction over the window."""
    if per_faction < 1:
        raise QueryError(f'per_faction must be positive: {per_faction!r}')
    return f"""SELECT
    faction_id,
    tag,
    count() AS n
FROM (
    SELECT
        m.faction_id AS faction_id,
        arrayJoin(arrayFilter(x -> x != '', p.FacetTagList)) AS tag
    FROM {identifier(source)} AS p
    INNER JOIN {_image_subquery(members_table, run_id)} AS m ON p.UserId = m.did
    WHERE p.Collection = {quote_literal(POST_COLLECTION)}
      AND p.OperationKind = {quote_literal(OPERATION_CREATE)}
      AND p.__timestamp >= {_timestamp(window.start)}
      AND p.__timestamp < {_timestamp(window.end)}
)
GROUP BY faction_id, tag
ORDER BY faction_id ASC, n DESC, tag ASC
LIMIT {int(per_faction)} BY faction_id"""


def faction_languages_sql(source: str, members_table: str, run_id: str, window: Window, per_faction: int = 10) -> str:
    """Columns: faction_id, language and n. The most used languages per faction over the window."""
    if per_faction < 1:
        raise QueryError(f'per_faction must be positive: {per_faction!r}')
    return f"""SELECT
    faction_id,
    language,
    count() AS n
FROM (
    SELECT
        m.faction_id AS faction_id,
        arrayJoin(arrayFilter(x -> x != '', p.PostLanguages)) AS language
    FROM {identifier(source)} AS p
    INNER JOIN {_image_subquery(members_table, run_id)} AS m ON p.UserId = m.did
    WHERE p.Collection = {quote_literal(POST_COLLECTION)}
      AND p.OperationKind = {quote_literal(OPERATION_CREATE)}
      AND p.__timestamp >= {_timestamp(window.start)}
      AND p.__timestamp < {_timestamp(window.end)}
)
GROUP BY faction_id, language
ORDER BY faction_id ASC, n DESC, language ASC
LIMIT {int(per_faction)} BY faction_id"""


def faction_handles_sql(source: str, features_table: str, run_id: str, window: Window, per_faction: int = 10) -> str:
    """Columns: faction_id, did and handle. Handles of the accounts with the highest within-faction z-score.

    A row can carry no handle. The last one that does is taken, so an account keeps its handle as long as one
    of its posts in the window has it.
    """
    if per_faction < 1:
        raise QueryError(f'per_faction must be positive: {per_faction!r}')
    return f"""SELECT
    f.faction_id AS faction_id,
    f.did AS did,
    anyLastIf(p.Handle, p.Handle IS NOT NULL AND p.Handle != '') AS handle
FROM (
    SELECT
        did,
        faction_id,
        row_number() OVER (PARTITION BY faction_id ORDER BY within_z DESC, did ASC) AS z_rank
    FROM {identifier(features_table)}
    WHERE run_id = {quote_literal(run_id)}
      AND within_z IS NOT NULL
) AS f
INNER JOIN {identifier(source)} AS p ON p.UserId = f.did
WHERE f.z_rank <= {int(per_faction)}
  AND p.Collection = {quote_literal(POST_COLLECTION)}
  AND p.OperationKind = {quote_literal(OPERATION_CREATE)}
  AND p.__timestamp >= {_timestamp(window.start)}
  AND p.__timestamp < {_timestamp(window.end)}
GROUP BY faction_id, did
ORDER BY faction_id ASC, did ASC"""


def run_states_sql(runs_table: str) -> str:
    """Columns: run_id, promoted, promoting, failed, running and last_recorded_at. One row per run."""
    return f"""SELECT
    run_id,
    countIf(status = 'promoted') > 0 AS promoted,
    countIf(status = 'promoting') > 0 AS promoting,
    countIf(status = 'failed') > 0 AS failed,
    countIf(status = 'running') > 0 AS running,
    max(recorded_at) AS last_recorded_at
FROM {identifier(runs_table)}
GROUP BY run_id
ORDER BY run_id"""


def run_rows_sql(runs_table: str, run_id: str) -> str:
    """Every row of one run, oldest first.

    A run writes its rows seconds apart at most, so the status breaks ties: ``running`` precedes
    ``promoting`` precedes ``promoted``, and ``failed`` comes last.
    """
    return f"""SELECT {_run_row_columns()}
FROM {identifier(runs_table)}
WHERE run_id = {quote_literal(run_id)}
ORDER BY recorded_at ASC, status ASC"""


def latest_promoted_sql(runs_table: str, kind: str | None = None) -> str:
    """The promoted run that started last, with ``kind`` equal to ``kind`` when a kind is given.

    A run ID starts with epoch milliseconds, so its lexical order is its start order.
    """
    kind_clause = '' if kind is None else f'\n  AND kind = {quote_literal(kind)}'
    return f"""SELECT {_run_row_columns()}
FROM {identifier(runs_table)}
WHERE status = 'promoted'{kind_clause}
ORDER BY run_id DESC
LIMIT 1"""


def _run_row_columns() -> str:
    return ', '.join(
        [
            'run_id',
            'status',
            'kind',
            'recorded_at',
            'started_at',
            'window_start',
            'window_end',
            'window_days',
            'objective',
            'resolution',
            'seed',
            'next_faction_id',
            'accounts_placed',
            'accounts_unplaced',
            'hash_collisions',
            'factions',
            'moves',
            'births',
            'merges',
            'splits',
            'deaths',
            'agreement_prev',
            'modularity',
            'source_gap_days',
            'text_model_version',
            'text_coverage',
            'stage_seconds',
            'failed_stage',
            'error_class',
        ]
    )


def image_sql(members_table: str, run_id: str) -> str:
    """Columns: did, faction_id, since_run_id, prev_faction_id and confidence for one promoted image."""
    return f"""SELECT
    did,
    faction_id,
    since_run_id,
    prev_faction_id,
    confidence
FROM {identifier(members_table)}
WHERE run_id = {quote_literal(run_id)}
ORDER BY did"""


def text_scores_for_model_sql(text_scores_table: str, model_version: str) -> str:
    """Columns: uri_hash and h. One row per uri_hash for a model version.

    ``argMax`` picks the latest write, so a retried insert cannot double-count. The columns are qualified so
    that the alias of an aggregate is not substituted into the aggregate itself.
    """
    return f"""SELECT
    uri_hash,
    argMax(t.h, t.scored_at) AS h
FROM {identifier(text_scores_table)} AS t
WHERE model_version = {quote_literal(model_version)}
GROUP BY uri_hash
ORDER BY uri_hash"""


def jev_labels_sql(labels_table: str, prompt_version: str) -> str:
    """Columns: uri_hash, h, p_hostile, p_threat, model and input_tokens. One row per uri_hash."""
    return f"""SELECT
    uri_hash,
    argMax(t.h, t.labelled_at) AS h,
    argMax(t.p_hostile, t.labelled_at) AS p_hostile,
    argMax(t.p_threat, t.labelled_at) AS p_threat,
    argMax(t.model, t.labelled_at) AS model,
    argMax(t.input_tokens, t.labelled_at) AS input_tokens
FROM {identifier(labels_table)} AS t
WHERE prompt_version = {quote_literal(prompt_version)}
GROUP BY uri_hash
ORDER BY uri_hash"""


def parent_texts_sql(source: str, dataset_table: str, dataset_version: str, window: Window) -> str:
    """Columns: uri_hash and parent_text. The text of the parent post of each dataset row.

    The dataset rows carry ``parent_uri``. One scan of the source joins those URIs to their post text.
    """
    return f"""SELECT
    d.uri_hash AS uri_hash,
    anyLast(p.PostText) AS parent_text
FROM {identifier(dataset_table)} AS d
INNER JOIN {identifier(source)} AS p ON p.AtUri = d.parent_uri
WHERE d.dataset_version = {quote_literal(dataset_version)}
  AND p.Collection = {quote_literal(POST_COLLECTION)}
  AND p.OperationKind = {quote_literal(OPERATION_CREATE)}
  AND {in_window_sql(window)}
GROUP BY uri_hash
ORDER BY uri_hash"""


def label_interactions_sql(source: str, window: Window) -> str:
    """Columns: uri_hash, kind, event_ms, author and target (UInt64). Every reply and quote of the window.

    Accounts are hashes, so tens of millions of rows fit in memory and no DID is stored.
    """
    return f"""SELECT
    uri_hash,
    kind,
    event_ms,
    {_hash('author_did')} AS author,
    {_hash('target_did')} AS target
FROM (
    SELECT
        {_hash('AtUri')} AS uri_hash,
        if(is_reply, toInt8(0), toInt8(1)) AS kind,
        toInt64(toUnixTimestamp64Milli(__timestamp)) AS event_ms,
        UserId AS author_did,
        if(is_reply, PostReplyParentDid, {QUOTED_AUTHOR}) AS target_did,
        {REPLY_PREDICATE} AS is_reply,
        {QUOTE_PREDICATE} AS is_quote
    {_source(source, window, POST_COLLECTION, OPERATION_CREATE)}
)
WHERE is_reply OR is_quote"""


def block_events_sql(source: str, window: Window) -> str:
    """Columns: uri_hash, event_ms, blocker and blocked (UInt64). Every block create.

    Deletes are subtracted by the caller, because a delete can land later than its create.
    """
    return f"""SELECT
    {_hash('AtUri')} AS uri_hash,
    toInt64(toUnixTimestamp64Milli(__timestamp)) AS event_ms,
    {_hash('UserId')} AS blocker,
    {_hash('BlockSubjectDid')} AS blocked
{_source(source, window, BLOCK_COLLECTION, OPERATION_CREATE)}
  AND BlockSubjectDid != ''
  AND UserId != BlockSubjectDid"""


def texts_for_hashes_sql(source: str, window: Window, uri_hashes: Sequence[int]) -> str:
    """The columns of ``interactions_sql`` for the given URI hashes."""
    if not uri_hashes:
        raise QueryError('uri_hashes must not be empty')
    for value in uri_hashes:
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**64:
            raise QueryError(f'uri_hash must be an unsigned 64-bit integer: {value!r}')
    hashes = ', '.join(str(int(value)) for value in uri_hashes)
    return f"""SELECT *
FROM (
{interactions_sql(source, window)}
)
WHERE uri_hash IN ({hashes})
ORDER BY uri_hash"""


def parent_texts_for_uris_sql(source: str, window: Window, parent_uris: Sequence[str]) -> str:
    """Columns: parent_uri and parent_text for the given post URIs."""
    if not parent_uris:
        raise QueryError('parent_uris must not be empty')
    for uri in parent_uris:
        if not isinstance(uri, str) or not uri.startswith('at://') or len(uri) > 4096:
            raise QueryError(f'not a post URI: {uri!r}')
    uris = ', '.join(quote_literal(uri) for uri in parent_uris)
    return f"""SELECT
    AtUri AS parent_uri,
    anyLast(PostText) AS parent_text
FROM {identifier(source)}
WHERE Collection = {quote_literal(POST_COLLECTION)}
  AND OperationKind = {quote_literal(OPERATION_CREATE)}
  AND {in_window_sql(window)}
  AND AtUri IN ({uris})
GROUP BY parent_uri
ORDER BY parent_uri"""


def dataset_rows_sql(dataset_table: str, dataset_version: str) -> str:
    """Every write of a dataset row, oldest first. Readers keep the last write of each URI hash."""
    return f"""SELECT
    uri_hash,
    kind,
    split,
    label_source,
    target,
    weight,
    label_h,
    author_did,
    target_did,
    text,
    parent_uri,
    event_ms,
    languages,
    text_length,
    uppercase,
    exclamations,
    questions,
    mentions,
    written_at
FROM {identifier(dataset_table)}
WHERE dataset_version = {quote_literal(dataset_version)}
ORDER BY uri_hash, written_at"""


def faction_registry_sql(factions_table: str, runs_table: str) -> str:
    """Columns: faction_id, status, parent_faction_id and born_run_id.

    The latest row of each faction among promoted runs, which is the current state of the faction.
    """
    return f"""SELECT
    faction_id,
    argMax(f.status, f.run_id) AS status,
    argMax(f.parent_faction_id, f.run_id) AS parent_faction_id,
    argMax(f.born_run_id, f.run_id) AS born_run_id
FROM {identifier(factions_table)} AS f
WHERE run_id IN (
    SELECT run_id FROM {identifier(runs_table)} WHERE status = 'promoted'
)
GROUP BY faction_id
ORDER BY faction_id"""
