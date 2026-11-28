import os
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from sys import modules
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from faction_analysis.labels import BLOCK, CALIBRATION, JEV, RANDOM, TEST, TRAIN
from faction_analysis.model import TrainingRows, text_frame
from faction_analysis.runstate import RunRow
from tests import training_world
from tests.integration_world import SOURCE_DDL, ddl_statements
from tests.training_world import FakeTrainingRepo

HOSTILE = [
    'you are an idiot and nobody likes you!!!',
    'SHUT UP nobody asked you',
    'go away and never come back, you loser',
    'everyone block this clown @someone @someone else',
    'why are you so stupid???',
]
NEUTRAL = [
    'thanks, that helped',
    'I think the second option works better',
    'here is the link I mentioned',
    'nice photo, where was it taken?',
    'agree, lets do it next week',
]


def structural(texts: list[str]) -> dict[str, np.ndarray]:
    """The five structural columns as ClickHouse computes them."""
    return {
        'text_length': np.asarray([len(text) for text in texts], dtype=np.int64),
        'uppercase': np.asarray([sum(1 for char in text if char.isupper()) for text in texts], dtype=np.int64),
        'exclamations': np.asarray([text.count('!') for text in texts], dtype=np.int64),
        'questions': np.asarray([text.count('?') for text in texts], dtype=np.int64),
        'mentions': np.asarray([text.count('@') for text in texts], dtype=np.int64),
    }


def fabricated_rows(
    *,
    jev_rows: int = 900,
    weak_rows: int = 300,
    random_rows: int = 300,
    seed: int = 0,
    test_all_positive: bool = False,
) -> TrainingRows:
    """Labelled training rows from fabricated text.

    Jev-labelled rows are split 70 / 15 / 15. Weak positives and random negatives only train. German rows are
    rarely hostile, so German has too few positives for a per-language metric.
    """
    rng = np.random.default_rng(seed)

    def pick(hostile: bool) -> str:
        pool = HOSTILE if hostile else NEUTRAL
        return f'{pool[int(rng.integers(len(pool)))]} {int(rng.integers(10_000))}'

    texts: list[str] = []
    targets: list[int] = []
    splits: list[int] = []
    sources: list[int] = []
    languages: list[tuple[str, ...]] = []
    for index in range(jev_rows):
        hostile = bool(rng.random() < 0.5) or test_all_positive
        texts.append(pick(hostile))
        targets.append(int(hostile))
        # A 70 / 15 / 15 split by a hash of the index, independent of anything else in the row.
        share = (index * 7919 + seed) % 100
        splits.append(TRAIN if share < 70 else CALIBRATION if share < 85 else TEST)
        sources.append(JEV)
        roll = rng.random()
        languages.append(('de',) if roll < 0.17 else ('en', 'fr') if roll < 0.34 else ('en',))
    for _ in range(weak_rows):
        texts.append(pick(True))
        targets.append(1)
        splits.append(TRAIN)
        sources.append(BLOCK)
        languages.append(('en',))
    for _ in range(random_rows):
        texts.append(pick(rng.random() < 0.1))
        targets.append(0)
        splits.append(TRAIN)
        sources.append(RANDOM)
        languages.append(('en',))
    frame = text_frame(texts, kind=np.zeros(len(texts), dtype=np.int8), **structural(texts))
    return TrainingRows(
        frame=frame,
        target=np.asarray(targets, dtype=np.uint8),
        weight=np.ones(len(texts), dtype=np.float32),
        split=np.asarray(splits, dtype=np.int8),
        label_source=np.asarray(sources, dtype=np.int8),
        languages=languages,
    )


@pytest.fixture
def fabricated() -> Callable[..., TrainingRows]:
    return fabricated_rows


@pytest.fixture
def helpers() -> ModuleType:
    """This module, for its text samples and its ``structural`` helper."""
    return modules[__name__]


@pytest.fixture
def fake_world() -> FakeTrainingRepo:
    """The fabricated world of the classifier shell: interactions, blocks, texts and parents."""
    inputs, texts, parents = training_world.world()
    return FakeTrainingRepo(inputs, texts, parents, training_world.empty_labels())


@pytest.fixture
def run_row() -> RunRow:
    started = datetime(2025, 11, 27, 12, tzinfo=UTC)
    return RunRow(
        run_id='1764241600000.0123abcd',
        status='running',
        kind='weekly',
        recorded_at=started,
        started_at=started,
        window_start=started - timedelta(days=60),
        window_end=started,
        window_days=60,
        objective='modularity',
        resolution=1.0,
        seed=42,
        next_faction_id=0,
        accounts_placed=0,
        accounts_unplaced=0,
        hash_collisions=0,
        factions=0,
        moves=0,
        births=0,
        merges=0,
        splits=0,
        deaths=0,
        agreement_prev=None,
        modularity=None,
        source_gap_days=0,
        text_model_version='',
        text_coverage=0.0,
        stage_seconds={},
        failed_stage='',
        error_class='',
    )


# ---- integration against a real ClickHouse of the production version ----

INTEGRATION_HOST = 'FACTION_ANALYSIS_INTEGRATION_CH_HOST'
INTEGRATION_NOW = datetime(2026, 11, 27, 12, tzinfo=UTC)


@pytest.fixture(scope='session')
def server() -> dict[str, Any]:
    """The connection values of the test server, after a check that it runs ClickHouse 24.1.

    The integration tests skip unless the host is configured. ``docker/compose.test.yml`` starts a server:
    ClickHouse 24.1, the version on ``sys2``, with its data in memory.
    """
    settings = _settings()
    client = _client(settings, 'default')
    try:
        version = str(client.query('SELECT version()').result_rows[0][0])
        assert version.split('.')[:2] == ['24', '1'], f'these tests run against ClickHouse 24.1, not {version}'
    finally:
        client.close()
    return settings


def _settings() -> dict[str, Any]:
    host = os.environ.get(INTEGRATION_HOST)
    if not host:
        pytest.skip(f'set {INTEGRATION_HOST} to run the integration tests; docker/compose.test.yml starts a server')
    return {
        'host': host,
        'port': int(os.environ.get('FACTION_ANALYSIS_INTEGRATION_CH_PORT', '18123')),
        'username': os.environ.get('FACTION_ANALYSIS_INTEGRATION_CH_USER', 'default'),
        'password': os.environ.get('FACTION_ANALYSIS_INTEGRATION_CH_PASSWORD', ''),
    }


def _client(server: dict[str, Any], database: str) -> Any:
    """A client for the test server with ``database`` as its default database."""
    import clickhouse_connect

    return clickhouse_connect.get_client(**server, database=database)


@dataclass
class Scratch:
    """An empty database with the sidecar's tables and a small source table.

    ``client`` works in the default database and seeds the source. ``sidecar`` has the scratch database as
    its default database, as the sidecar's own client has ``OSPREY_CLICKHOUSE_DB``.
    """

    client: Any
    sidecar: Any
    database: str
    source: str
    now: datetime


@pytest.fixture
def scratch(server: dict[str, Any]) -> Iterator[Scratch]:
    """The sidecar's DDL with the database substituted, and the source table with the columns it reads."""
    database = 'fa_it_' + uuid.uuid4().hex[:10]
    client = _client(server, 'default')
    client.command(f'CREATE DATABASE {database}')
    sidecar = None
    try:
        for statement in ddl_statements(database):
            client.command(statement)
        client.command(SOURCE_DDL.format(database=database))
        sidecar = _client(server, database)
        yield Scratch(client, sidecar, database, f'{database}.osprey_execution_results', INTEGRATION_NOW)
    finally:
        if sidecar is not None:
            sidecar.close()
        client.command(f'DROP DATABASE {database} SYNC')
        client.close()
