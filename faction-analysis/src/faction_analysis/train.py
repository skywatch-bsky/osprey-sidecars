# pattern: Imperative Shell
"""Refresh the hostility model.

Build the dataset from the window, label with Jev what is not yet labelled, train, evaluate, and promote the
model when it passes the test gates. The dataset and every row of it are also written to the durable
``scorer_dataset`` table, so a later export can join the text of the parent post.
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import secrets
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import pandas as pd

from faction_analysis import model_store, queries, schema
from faction_analysis.config import AppConfig
from faction_analysis.jev import JevLabel, JevLabeller, TextPair, TypeSafeJevLabeller, label_pairs, new_budget
from faction_analysis.labels import (
    LABEL_SOURCES,
    SPLITS,
    LabelInput,
    LabelSet,
    block_positive_mask,
    build_labels,
    choose_jev_sample,
    elapsed_horizon_mask,
    label_policy_hash,
    negative_mask,
)
from faction_analysis.model import (
    EXCLAMATIONS,
    FEATURE_CONFIG,
    LENGTH,
    MENTIONS,
    QUESTIONS,
    UPPERCASE,
    TrainingError,
    TrainingRows,
    evaluate,
    may_promote,
    text_frame,
    train,
)

logger = logging.getLogger(__name__)
KIND_NAMES = ('reply', 'quote')
MIN_TRAIN_ROWS = 1000


class TrainingRepository(Protocol):
    def fetch_label_inputs(self, window: queries.Window) -> LabelInput: ...

    def fetch_interaction_texts(self, window: queries.Window, uri_hashes: Sequence[int]) -> pd.DataFrame: ...

    def fetch_parent_texts(self, window: queries.Window, parent_uris: Sequence[str]) -> dict[str, str]: ...

    def fetch_jev_labels(self, prompt_version: str) -> pd.DataFrame: ...

    def insert_frame(self, noun: str, frame: pd.DataFrame) -> None: ...


LabellerFactory = Callable[[], JevLabeller]
RefreshStatus = Literal['promoted', 'kept', 'no_model']


@dataclass(frozen=True)
class RefreshOutcome:
    """What a refresh did. ``kept`` means the previous model stays and ``reason`` says why."""

    status: RefreshStatus
    reason: str
    dataset_version: str
    labelled_now: int
    model_version: str | None = None
    metrics: dict | None = None

    def as_dict(self) -> dict:
        return {
            'status': self.status,
            'reason': self.reason,
            'dataset_version': self.dataset_version,
            'labelled_now': self.labelled_now,
            'model_version': self.model_version,
            'metrics': self.metrics,
        }


def _close(labeller: JevLabeller) -> None:
    close = getattr(labeller, 'close', None)
    if close is not None:
        asyncio.run(close())


def _existing_labels(repo: TrainingRepository, inputs: LabelInput, prompt_version: str) -> np.ndarray:
    """The hostility Jev already assigned to each interaction, ``NaN`` where it assigned none."""
    jev_h = np.full(len(inputs.uri_hash), np.nan, dtype=np.float32)
    if len(inputs.uri_hash) == 0:
        return jev_h
    existing = repo.fetch_jev_labels(prompt_version)
    if len(existing) == 0:
        return jev_h
    hashes = np.asarray(existing['uri_hash'], dtype=np.uint64)
    values = np.asarray(existing['h'], dtype=np.float32)
    # The search needs a sorted table. A cache read may come back in any order.
    order = np.argsort(hashes, kind='stable')
    hashes, values = hashes[order], values[order]
    position = np.searchsorted(hashes, inputs.uri_hash)
    clipped = np.minimum(position, len(hashes) - 1)
    found = hashes[clipped] == inputs.uri_hash
    jev_h[found] = values[clipped][found]
    return jev_h


def _label_row(uri_hash: int, result: JevLabel, prompt_version: str, now: datetime) -> dict[str, object]:
    return {
        'uri_hash': uri_hash,
        'prompt_version': prompt_version,
        'h': result.h,
        'p_hostile': result.p_hostile,
        'p_threat': result.p_threat,
        'model': result.model,
        'input_tokens': result.input_tokens,
        'labelled_at': now,
    }


def _label_with_jev(
    repo: TrainingRepository,
    config: AppConfig,
    window: queries.Window,
    inputs: LabelInput,
    sample: np.ndarray,
    factory: LabellerFactory,
    *,
    dry_run: bool,
    now: datetime,
) -> dict[int, float]:
    """Label the sample and store each answer as soon as it exists. Returns the labels by URI hash.

    The answers go into the label cache in batches while the labelling runs, not when it ends. A crash, an
    authentication error or a failed cache write can then cost at most the answers of the last batch, which
    the next refresh buys again. The rest is never paid for twice.
    """
    sampled = [int(value) for value in inputs.uri_hash[sample]]
    texts = repo.fetch_interaction_texts(window, sampled)
    parent_uris = sorted({str(uri) for uri in texts['parent_uri'].tolist() if uri})
    parents = repo.fetch_parent_texts(window, parent_uris) if parent_uris else {}
    pairs: list[tuple[int, TextPair]] = []
    columns = zip(texts['uri_hash'], texts['kind'], texts['text'], texts['parent_uri'], strict=True)
    for uri_hash, kind, text, parent_uri in columns:
        parent = parents.get(str(parent_uri) or '', '')
        pairs.append((int(uri_hash), TextPair('' if text is None else str(text), parent, int(kind))))
    label = config.label
    budget = new_budget(label)
    labeller = factory()

    def store(batch: list[tuple[int, JevLabel]]) -> None:
        if not dry_run:
            rows = [_label_row(uri_hash, result, label.prompt_version, now) for uri_hash, result in batch]
            repo.insert_frame('jev_labels', pd.DataFrame(rows, columns=list(schema.COLUMNS[schema.JEV_LABELS])))

    failure: BaseException | None = None
    try:
        results = asyncio.run(
            label_pairs(
                labeller,
                pairs,
                budget,
                concurrency=label.jev_concurrency,
                max_text_chars=label.jev_max_text_chars,
                store=store,
            )
        )
    except BaseException as exc:
        failure = exc
        raise
    finally:
        try:
            _close(labeller)
        except Exception as close_error:  # noqa: BLE001 - closing must not hide why labelling ended
            if failure is None:
                raise
            failure.add_note(f'the labeller could not be closed: {close_error!r}')
    return {uri: result.h for uri, result in results.items()}


def refresh(
    repo: TrainingRepository,
    config: AppConfig,
    model_dir: Path,
    *,
    now: datetime,
    factory: LabellerFactory | None = None,
    dry_run: bool = False,
) -> RefreshOutcome:
    """One refresh. ``factory`` builds the Jev labeller; ``None`` skips labelling and keeps the stored labels."""
    now = now.astimezone(UTC)
    now_ms = int(now.timestamp() * 1000)
    dataset_version = f'ds-{now_ms}-{secrets.token_hex(2)}'
    window = queries.Window(now - timedelta(days=config.graph.window_days), now)
    label = config.label
    inputs = repo.fetch_label_inputs(window)
    positives = block_positive_mask(inputs, positive_minutes=label.block_positive_minutes)
    elapsed = elapsed_horizon_mask(inputs.event_ms, now_ms=now_ms, horizon_days=label.block_negative_horizon_days)
    negatives = negative_mask(
        inputs, positives=positives, elapsed=elapsed, horizon_days=label.block_negative_horizon_days
    )
    jev_h = _existing_labels(repo, inputs, label.prompt_version)
    labelled = ~np.isnan(jev_h)
    sample = choose_jev_sample(inputs, positives=positives, elapsed=elapsed, labelled=labelled, config=label)
    labelled_now = 0
    if sample.any() and factory is not None:
        labels = _label_with_jev(repo, config, window, inputs, sample, factory, dry_run=dry_run, now=now)
        position_of = {
            int(value): index for index, value in zip(np.flatnonzero(sample), inputs.uri_hash[sample], strict=True)
        }
        for uri_hash, value in labels.items():
            jev_h[position_of[uri_hash]] = value
        labelled_now = len(labels)
    elif sample.any():
        logger.warning('no labeller configured: %d pairs stay unlabelled', int(sample.sum()))

    rows = build_labels(inputs, jev_h, positives=positives, negatives=negatives, config=label)
    if len(rows) == 0:
        return RefreshOutcome('no_model', 'the window holds no labelled rows', dataset_version, labelled_now)
    order = np.argsort(rows.uri_hash, kind='stable')
    rows = _take(rows, order)
    texts = repo.fetch_interaction_texts(window, [int(value) for value in rows.uri_hash])
    joined = _join_texts(rows, inputs, texts, dataset_version, now)
    if not dry_run:
        repo.insert_frame('scorer_dataset', joined)
    training_rows = _training_rows(joined)
    counts: dict[str, int] = {SPLITS[code]: int((rows.split == code).sum()) for code in range(len(SPLITS))}
    if counts['train'] < MIN_TRAIN_ROWS:
        return RefreshOutcome(
            'no_model', f'the training split has {counts["train"]} rows', dataset_version, labelled_now
        )
    try:
        model = train(training_rows, seed=config.partition.seed)
    except TrainingError as exc:
        return RefreshOutcome('kept', str(exc), dataset_version, labelled_now)
    metrics = evaluate(model, training_rows)
    promoted, reason = may_promote(
        metrics, min_test_auc=label.min_test_auc, min_test_positives=label.min_test_positives
    )
    if not promoted:
        return RefreshOutcome('kept', reason, dataset_version, labelled_now, metrics=metrics.as_dict())
    payload = model_store.serialise(model)
    digest = hashlib.sha256(payload).hexdigest()
    version = model_store.model_version(now_ms, digest)
    manifest = model_store.ModelManifest(
        version=version,
        sha256=digest,
        trained_at=now.isoformat(),
        dataset_version=dataset_version,
        label_policy_hash=label_policy_hash(label),
        feature_config=dict(FEATURE_CONFIG),
        row_counts=counts,
        metrics=metrics.as_dict(),
        calibrated=model.calibrated,
    )
    if not dry_run:
        current = model_store.current_manifest(model_dir)
        # The same bytes trained at the same instant are the same model. Nothing is written twice.
        if current is None or current.version != version:
            model_store.promote(model_dir, payload, manifest)
    return RefreshOutcome('promoted', reason, dataset_version, labelled_now, version, metrics.as_dict())


def _take(rows: LabelSet, order: np.ndarray) -> LabelSet:
    return LabelSet(
        rows.uri_hash[order],
        rows.kind[order],
        rows.target[order],
        rows.weight[order],
        rows.split[order],
        rows.label_source[order],
        rows.label_h[order],
    )


def _join_texts(
    rows: LabelSet, inputs: LabelInput, texts: pd.DataFrame, dataset_version: str, now: datetime
) -> pd.DataFrame:
    """The dataset rows with their text and structural columns.

    Texts join on the URI hash, never on position. A labelled row without a text (for example a post that
    left the window between the two reads) is left out.
    """
    left = pd.DataFrame(
        {
            'dataset_version': dataset_version,
            'uri_hash': np.asarray(rows.uri_hash, dtype=np.uint64),
            'kind': [KIND_NAMES[code] for code in rows.kind.tolist()],
            'split': [SPLITS[code] for code in rows.split.tolist()],
            'label_source': [LABEL_SOURCES[code] for code in rows.label_source.tolist()],
            'target': rows.target,
            'weight': rows.weight,
            'label_h': [None if np.isnan(value) else float(value) for value in rows.label_h.tolist()],
            # The author and target hashes come from the label input, which carries no text.
            'author_did': _lookup(inputs.uri_hash, inputs.author, rows.uri_hash),
            'target_did': _lookup(inputs.uri_hash, inputs.target, rows.uri_hash),
        }
    )
    right = texts.loc[
        :,
        [
            'uri_hash',
            'text',
            'parent_uri',
            'event_ms',
            'languages',
            LENGTH,
            UPPERCASE,
            EXCLAMATIONS,
            QUESTIONS,
            MENTIONS,
        ],
    ].copy()
    right['uri_hash'] = np.asarray(right['uri_hash'], dtype=np.uint64)
    joined = left.merge(right, on='uri_hash', how='inner', validate='one_to_one')
    joined['event_ms'] = np.asarray(joined['event_ms'], dtype=np.int64)
    joined['languages'] = [tuple(value) if value is not None else () for value in joined['languages'].tolist()]
    joined['written_at'] = now
    return joined


def _lookup(keys: np.ndarray, values: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    order = np.argsort(keys, kind='stable')
    sorted_keys = keys[order]
    position = np.searchsorted(sorted_keys, wanted)
    clipped = np.minimum(position, len(sorted_keys) - 1)
    found = sorted_keys[clipped] == wanted
    result = np.zeros(len(wanted), dtype=values.dtype)
    result[found] = values[order][clipped][found]
    return result


def _training_rows(joined: pd.DataFrame) -> TrainingRows:
    frame = text_frame(
        ['' if value is None else str(value) for value in joined['text'].tolist()],
        np.asarray(joined[LENGTH], dtype=np.int64),
        np.asarray(joined[UPPERCASE], dtype=np.int64),
        np.asarray(joined[EXCLAMATIONS], dtype=np.int64),
        np.asarray(joined[QUESTIONS], dtype=np.int64),
        np.asarray(joined[MENTIONS], dtype=np.int64),
        np.asarray([KIND_NAMES.index(value) for value in joined['kind'].tolist()], dtype=np.int8),
    )
    return TrainingRows(
        frame=frame,
        target=np.asarray(joined['target'], dtype=np.uint8),
        weight=np.asarray(joined['weight'], dtype=np.float32),
        split=np.asarray([SPLITS.index(value) for value in joined['split'].tolist()], dtype=np.int8),
        label_source=np.asarray(
            [LABEL_SOURCES.index(value) for value in joined['label_source'].tolist()], dtype=np.int8
        ),
        languages=[tuple(value) for value in joined['languages'].tolist()],
    )


def default_factory(api_key: str, config: AppConfig) -> LabellerFactory:
    def build() -> JevLabeller:
        return TypeSafeJevLabeller(api_key, config.label.jev_model, max_text_chars=config.label.jev_max_text_chars)

    return build


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Refresh the hostility model')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--no-label', action='store_true', help='do not call Jev; reuse the stored labels')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    config = AppConfig.from_env()
    from faction_analysis.db import FactionAnalysisDb

    db = FactionAnalysisDb(config.clickhouse, config.query)
    api_key = os.environ.get('TYPESAFE_API_KEY', '')
    try:
        outcome = refresh(
            db,
            config,
            Path(config.storage.model_dir),
            now=datetime.now(UTC),
            factory=None if args.no_label or not api_key else default_factory(api_key, config),
            dry_run=args.dry_run,
        )
    finally:
        db.close()
    print(json.dumps(outcome.as_dict(), indent=2, default=str))
    return 0 if outcome.status != 'no_model' else 1


if __name__ == '__main__':
    raise SystemExit(main())
