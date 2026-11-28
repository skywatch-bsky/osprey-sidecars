import re
from datetime import UTC, datetime, timedelta, timezone

import pytest

from faction_analysis import queries
from faction_analysis.queries import QueryError, Window

WINDOW = Window(datetime(2026, 11, 1, tzinfo=UTC), datetime(2026, 11, 3, tzinfo=UTC))
SOURCE = 'osprey_execution_results'


def source_statements() -> dict[str, str]:
    return {
        'follows': queries.follow_creates_sql(SOURCE, WINDOW),
        'reposts': queries.repost_creates_sql(SOURCE, WINDOW),
        'follow_deletes': queries.deletes_sql(SOURCE, WINDOW, queries.FOLLOW_COLLECTION),
        'block_deletes': queries.deletes_sql(SOURCE, WINDOW, queries.BLOCK_COLLECTION),
        'account_hashes': queries.account_hashes_sql(SOURCE, WINDOW),
        'interactions': queries.interactions_sql(SOURCE, WINDOW),
        'daily_counts': queries.daily_row_counts_sql(SOURCE, WINDOW),
        'tags': queries.faction_tags_sql(SOURCE, 'fa_members', '1764241600000.0123abcd', WINDOW),
        'languages': queries.faction_languages_sql(SOURCE, 'fa_members', '1764241600000.0123abcd', WINDOW),
        'handles': queries.faction_handles_sql(SOURCE, 'fa_features', '1764241600000.0123abcd', WINDOW),
        'parent_texts': queries.parent_texts_sql(SOURCE, 'fa_dataset', 'v1', WINDOW),
    }


def test_every_source_statement_filters_the_timestamp_range() -> None:
    for name, sql in source_statements().items():
        assert "__timestamp >= toDateTime64('2026-11-01 00:00:00.000', 3, 'UTC')" in sql, name
        assert "__timestamp < toDateTime64('2026-11-03 00:00:00.000', 3, 'UTC')" in sql, name
        assert f'FROM {SOURCE}' in sql or f'JOIN {SOURCE} AS p' in sql, name


def test_window_bounds_are_written_in_utc() -> None:
    zone = timezone(timedelta(hours=5, minutes=30))
    local = Window(datetime(2026, 11, 1, 5, 30, tzinfo=zone), datetime(2026, 11, 1, 6, 30, tzinfo=zone))
    sql = queries.repost_creates_sql(SOURCE, local)
    assert "toDateTime64('2026-11-01 00:00:00.000', 3, 'UTC')" in sql
    assert "toDateTime64('2026-11-01 01:00:00.000', 3, 'UTC')" in sql


def test_limit_is_appended_only_when_given_and_must_be_positive() -> None:
    assert not queries.follow_creates_sql(SOURCE, WINDOW).rstrip().endswith('LIMIT 100')
    assert queries.follow_creates_sql(SOURCE, WINDOW, 100).rstrip().endswith('LIMIT 100')
    for bad in (0, -1, True, '10'):
        with pytest.raises(QueryError):
            queries.follow_creates_sql(SOURCE, WINDOW, bad)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize('name', ['follows', 'reposts', 'account_hashes', 'interactions'])
def test_limit_closes_every_branch_of_a_statement(name: str) -> None:
    statements = source_statements()
    assert statements[name].count('UNION ALL') == source_statements()[name].count('UNION ALL')


def test_interactions_limit_applies_to_the_whole_union() -> None:
    sql = queries.interactions_sql(SOURCE, WINDOW, 50)
    assert sql.count('LIMIT') == 1
    assert sql.rstrip().endswith('LIMIT 50')


def test_table_names_cannot_inject_sql() -> None:
    with pytest.raises(QueryError, match='invalid table identifier'):
        queries.follow_creates_sql('osprey_results; DROP TABLE x', WINDOW)
    with pytest.raises(QueryError):
        queries.run_states_sql('fa_runs --')


def test_values_are_escaped_as_string_literals() -> None:
    assert queries.quote_literal("it's") == "'it\\'s'"
    assert queries.quote_literal('a\\b') == "'a\\\\b'"
    sql = queries.text_scores_for_model_sql('fa_scores', "v'1")
    assert "model_version = 'v\\'1'" in sql


def test_window_requires_a_timezone_and_a_positive_length() -> None:
    naive = datetime(2026, 11, 1)
    with pytest.raises(QueryError):
        Window(naive, datetime(2026, 11, 2, tzinfo=UTC))
    with pytest.raises(QueryError):
        Window(WINDOW.start, WINDOW.start)
    with pytest.raises(QueryError):
        Window(WINDOW.end, WINDOW.start)


def test_day_chunks_are_consecutive_one_day_windows() -> None:
    window = Window(datetime(2026, 11, 1, 12, tzinfo=UTC), datetime(2026, 11, 4, tzinfo=UTC))
    chunks = window.day_chunks()
    assert len(chunks) == 3
    assert chunks[0] == Window(datetime(2026, 11, 1, 12, tzinfo=UTC), datetime(2026, 11, 2, 12, tzinfo=UTC))
    assert chunks[-1].end == window.end
    assert window.days == 3


def test_deletes_are_only_defined_for_follow_and_block_records() -> None:
    with pytest.raises(QueryError):
        queries.deletes_sql(SOURCE, WINDOW, queries.POST_COLLECTION)
    sql = queries.deletes_sql(SOURCE, WINDOW, queries.BLOCK_COLLECTION)
    assert "Collection = 'app.bsky.graph.block'" in sql
    assert "OperationKind = 'delete'" in sql


def test_follows_exclude_starter_packs_and_self_follows() -> None:
    sql = queries.follow_creates_sql(SOURCE, WINDOW)
    assert "Collection = 'app.bsky.graph.follow'" in sql
    assert 'coalesce(IsStarterPackFollow, 0) = 0' in sql
    assert 'UserId != FollowSubjectDid' in sql
    assert 'cityHash64(assumeNotNull(AtUri)) AS uri_hash' in sql


def test_reposts_read_the_repost_collection() -> None:
    sql = queries.repost_creates_sql(SOURCE, WINDOW)
    assert "Collection = 'app.bsky.feed.repost'" in sql
    assert 'cityHash64(assumeNotNull(RepostSubjectDid)) AS dst' in sql


def test_every_hash_is_plain_uint64_even_though_source_columns_are_nullable() -> None:
    # A Nullable(UInt64) adds a byte per value in RowBinary and arrives as None in a DataFrame.
    for sql in source_statements().values():
        for call in re.findall(r'cityHash64\([^)]*\)', sql):
            # ``a.did`` in the account query is not nullable: it comes out of an array of plain strings.
            assert call == 'cityHash64(a.did)' or 'assumeNotNull(' in call, f'{call} in {sql[:60]}'


def test_interactions_count_ascii_uppercase_letters_directly() -> None:
    sql = queries.interactions_sql(SOURCE, WINDOW)
    assert "toUInt32(countMatches(assumeNotNull(PostText), '[A-Z]')) AS uppercase" in sql
    assert 'length(assumeNotNull(PostText)) - length(lower(assumeNotNull(PostText)))' not in sql


def test_replies_require_a_non_self_reply_with_a_parent_author() -> None:
    sql = queries.interactions_sql(SOURCE, WINDOW)
    assert 'coalesce(PostIsReply, 0) = 1' in sql
    assert 'coalesce(PostIsSelfReply, 0) = 0' in sql
    assert "PostReplyParentDid != ''" in sql
    assert 'PostReplyParentDid != UserId' in sql


def test_quotes_take_the_author_from_the_uri_authority_and_need_a_post_path() -> None:
    sql = queries.interactions_sql(SOURCE, WINDOW)
    assert queries.QUOTE_PREDICATE in sql
    # A quote with media carries the post in PostEmbedRecordWithMediaUri.
    assert 'PostEmbedRecordWithMediaUri' in queries.QUOTE_PREDICATE
    assert f'if(is_reply, PostReplyParentDid, {queries.QUOTED_AUTHOR}) AS target_did' in sql
    assert f'if(is_reply, PostReplyParent, {queries.QUOTED_URI}) AS parent_uri' in sql
    assert 'extract(' not in sql


def test_blocks_carry_no_text_and_a_zero_thread_key() -> None:
    sql = queries.interactions_sql(SOURCE, WINDOW)
    block_branch = sql.split('UNION ALL')[1]
    assert 'toInt8(2) AS kind' in block_branch
    assert 'toUInt64(0) AS thread_key' in block_branch
    assert 'NULL AS text' in block_branch
    assert "Collection = 'app.bsky.graph.block'" in block_branch


def test_run_state_reads_group_by_run_and_rank_by_run_id() -> None:
    assert 'GROUP BY run_id' in queries.run_states_sql('fa_runs')
    latest = queries.latest_promoted_sql('fa_runs', 'full')
    assert "WHERE status = 'promoted'" in latest
    assert "AND kind = 'full'" in latest
    assert latest.rstrip().endswith('ORDER BY run_id DESC\nLIMIT 1')


def test_cache_reads_deduplicate_with_arg_max() -> None:
    scores = queries.text_scores_for_model_sql('fa_scores', 'v1')
    assert 'argMax(t.h, t.scored_at) AS h' in scores
    assert 'GROUP BY uri_hash' in scores
    labels = queries.jev_labels_sql('fa_labels', 'p1')
    assert 'argMax(t.h, t.labelled_at) AS h' in labels
    assert 'GROUP BY uri_hash' in labels


def every_statement() -> dict[str, str]:
    run_id = '1764241600000.0123abcd'
    return {
        **source_statements(),
        'run_states': queries.run_states_sql('fa_runs'),
        'run_rows': queries.run_rows_sql('fa_runs', run_id),
        'latest': queries.latest_promoted_sql('fa_runs', 'full'),
        'image': queries.image_sql('fa_members', run_id),
        'text_scores': queries.text_scores_for_model_sql('fa_scores', 'v1'),
        'jev_labels': queries.jev_labels_sql('fa_labels', 'p1'),
        'label_interactions': queries.label_interactions_sql(SOURCE, WINDOW),
        'block_events': queries.block_events_sql(SOURCE, WINDOW),
        'dataset_rows': queries.dataset_rows_sql('fa_dataset', 'v1'),
        'registry': queries.faction_registry_sql('fa_factions', 'fa_runs'),
    }


def test_no_alias_shadows_the_column_its_expression_reads() -> None:
    """ClickHouse substitutes an alias into the other expressions of its SELECT.

    ``min(did) AS did`` next to ``uniqExact(did)`` fails on 24.1 with "Unknown function min", because the
    second ``did`` becomes the first alias. ClickHouse 24.1 rejected this exact shape in the integration
    run, so every alias here must differ from the unqualified columns its expression reads.
    """
    alias_pattern = re.compile(r'(?:SELECT|,)\s*(.*?)\s+AS\s+(\w+)', re.DOTALL)
    literal = re.compile(r"'(?:[^'\\]|\\.)*'")
    for name, sql in every_statement().items():
        for expression, alias in alias_pattern.findall(sql):
            # A string literal is not a column: ``countIf(status = 'promoted') > 0 AS promoted`` is fine.
            unqualified = re.search(rf'(?<![\w.]){re.escape(alias)}\b', literal.sub("''", expression))
            assert unqualified is None, f'{alias} is both an alias and a column in {name}: {expression.strip()}'


def test_description_queries_join_the_image_of_the_run() -> None:
    run_id = '1764241600000.0123abcd'
    tags = queries.faction_tags_sql(SOURCE, 'fa_members', run_id, WINDOW, per_faction=5)
    assert "arrayJoin(arrayFilter(x -> x != '', p.FacetTagList)) AS tag" in tags
    assert "WHERE run_id = '1764241600000.0123abcd'" in tags
    assert 'INNER JOIN p.UserId = m.did' not in tags
    assert 'LIMIT 5 BY faction_id' in tags
    handles = queries.faction_handles_sql(SOURCE, 'fa_features', run_id, WINDOW, per_faction=3)
    assert 'ORDER BY within_z DESC' in handles
    # A row without a handle is passed over, so an account keeps its handle when one of its posts has it.
    assert "anyLastIf(p.Handle, p.Handle IS NOT NULL AND p.Handle != '') AS handle" in handles
    assert 'f.z_rank <= 3' in handles


def test_description_queries_refuse_a_non_positive_per_faction() -> None:
    with pytest.raises(QueryError):
        queries.faction_tags_sql(SOURCE, 'fa_members', 'x', WINDOW, per_faction=0)


def test_no_statement_contains_a_statement_separator() -> None:
    for name, sql in source_statements().items():
        assert ';' not in sql, name
        assert re.search(r'\bINTO\s+(OUTFILE|FORMAT)\b', sql, re.IGNORECASE) is None, name
