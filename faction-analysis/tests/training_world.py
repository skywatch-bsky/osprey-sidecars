"""A fabricated world for the classifier shell: interactions, blocks, texts, parents and a fake Jev."""

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import pandas as pd

from faction_analysis import queries
from faction_analysis.jev import JevLabel, TextPair
from faction_analysis.labels import LabelInput

HOSTILE_WORD = 'idiot'
PARENT_TEXT = 'the parent post says otherwise'
DAY = 86_400_000
NOW_MS = 1_795_700_000_000


class FakeTrainingRepo:
    """Answers the ``TrainingRepository`` protocol from fabricated data and records every insert."""

    def __init__(
        self, inputs: LabelInput, texts: pd.DataFrame, parents: dict[str, str], existing: pd.DataFrame
    ) -> None:
        self.inputs = inputs
        self.texts = texts
        self.parents = parents
        self.existing = existing
        self.inserted: dict[str, list[pd.DataFrame]] = defaultdict(list)

    def fetch_label_inputs(self, window: queries.Window) -> LabelInput:
        return self.inputs

    def fetch_interaction_texts(self, window: queries.Window, uri_hashes: Sequence[int]) -> pd.DataFrame:
        wanted = set(int(value) for value in uri_hashes)
        return self.texts[self.texts['uri_hash'].isin(wanted)].reset_index(drop=True)

    def fetch_parent_texts(self, window: queries.Window, parent_uris: Sequence[str]) -> dict[str, str]:
        wanted = set(parent_uris)
        return {uri: text for uri, text in self.parents.items() if uri in wanted}

    def fetch_jev_labels(self, prompt_version: str) -> pd.DataFrame:
        return self.existing

    def insert_frame(self, noun: str, frame: pd.DataFrame) -> None:
        self.inserted[noun].append(frame)


class FakeJev:
    """Labels a pair by its text, like Jev would for these samples. Counts its requests."""

    def __init__(self, calls: list[str] | None = None) -> None:
        self.calls = calls if calls is not None else []

    async def label(self, pair: TextPair) -> JevLabel:
        self.calls.append(pair.text)
        hostile = HOSTILE_WORD in pair.text
        return JevLabel(0.9 if hostile else 0.05, 0.6 if hostile else 0.03, 0.3 if hostile else 0.02, 'fake', 100)


def world(count: int = 3000, seed: int = 1) -> tuple[LabelInput, pd.DataFrame, dict[str, str]]:
    """``count`` interactions: half hostile. Each hostile one is answered by a block a minute later."""
    rng = np.random.default_rng(seed)
    uri_hash = rng.integers(1, 2**63, size=count, dtype=np.uint64)
    uri_hash = np.unique(uri_hash).astype(np.uint64)
    count = len(uri_hash)
    hostile = np.zeros(count, dtype=bool)
    hostile[::2] = True
    author = rng.integers(1, 2**40, size=count, dtype=np.uint64)
    target = rng.integers(2**40 + 1, 2**41, size=count, dtype=np.uint64)
    event_ms = np.full(count, NOW_MS - 30 * DAY, dtype=np.int64) + np.arange(count, dtype=np.int64)
    blocks = np.flatnonzero(hostile)
    inputs = LabelInput(
        uri_hash=uri_hash,
        kind=np.zeros(count, dtype=np.int8),
        event_ms=event_ms,
        author=author,
        target=target,
        block_ms=event_ms[blocks] + 60_000,
        block_blocker=target[blocks].copy(),
        block_blocked=author[blocks].copy(),
    )
    texts = pd.DataFrame(
        {
            'uri_hash': uri_hash,
            'kind': np.zeros(count, dtype=np.int8),
            'author_did': [f'did:plc:{value:x}' for value in author],
            'target_did': [f'did:plc:{value:x}' for value in target],
            'thread_key': uri_hash,
            'parent_uri': [f'at://did:plc:{value:x}/app.bsky.feed.post/abc' for value in target],
            'text': [
                f'you are an {HOSTILE_WORD} {index}' if hostile[index] else f'thanks for the post {index}'
                for index in range(count)
            ],
            'languages': [('en',)] * count,
            'text_length': np.full(count, 20, dtype=np.int64),
            'uppercase': np.zeros(count, dtype=np.int64),
            'exclamations': np.zeros(count, dtype=np.int64),
            'questions': np.zeros(count, dtype=np.int64),
            'mentions': np.zeros(count, dtype=np.int64),
            'event_ms': event_ms,
        }
    )
    parents = {uri: PARENT_TEXT for uri in texts['parent_uri'].tolist()}
    return inputs, texts, parents


def empty_labels() -> pd.DataFrame:
    return pd.DataFrame({'uri_hash': pd.Series(dtype='uint64'), 'h': pd.Series(dtype='float32')})
