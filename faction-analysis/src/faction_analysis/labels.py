# pattern: Functional Core
"""Labels for the hostility classifier: block positives, elapsed horizons, quotas and splits.

An interaction is a reply or a quote from an author to a target. It is a block positive when the target blocks
the author within ``block_positive_minutes`` of it. It can only be a negative when no such block occurred
before ``block_negative_horizon_days`` had elapsed, and that moment lies in the past.

Accounts are 64-bit hashes of DIDs. Every sample is deterministic: a pool is ordered by a hash of its URI
and the first rows are taken.
"""

import hashlib
from dataclasses import dataclass
from typing import Literal

import numpy as np

from faction_analysis.config import LabelConfig

Split = Literal['train', 'calibration', 'test']
SPLITS: tuple[Split, ...] = ('train', 'calibration', 'test')
LabelSource = Literal['jev', 'block', 'random']
LABEL_SOURCES: tuple[LabelSource, ...] = ('jev', 'block', 'random')
REPLY = 0
QUOTE = 1
JEV, BLOCK, RANDOM = 0, 1, 2
TRAIN, CALIBRATION, TEST = 0, 1, 2
SPLIT_SHARES = (70, 15, 15)
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)


@dataclass(frozen=True)
class LabelInput:
    """The replies and quotes of a window, and the blocks that might answer them."""

    uri_hash: np.ndarray
    kind: np.ndarray
    event_ms: np.ndarray
    author: np.ndarray
    target: np.ndarray
    block_ms: np.ndarray
    block_blocker: np.ndarray
    block_blocked: np.ndarray


@dataclass(frozen=True)
class LabelSet:
    """The training rows: one per interaction that received a label.

    ``split``, ``label_source`` and ``label_h`` use the codes above. ``label_h`` is ``NaN`` for a weak or a
    random label.
    """

    uri_hash: np.ndarray
    kind: np.ndarray
    target: np.ndarray
    weight: np.ndarray
    split: np.ndarray
    label_source: np.ndarray
    label_h: np.ndarray

    def __len__(self) -> int:
        return int(len(self.uri_hash))


def mix(values: np.ndarray, salt: int) -> np.ndarray:
    """A deterministic pseudo-random 64-bit value per input (SplitMix64's finalizer)."""
    with np.errstate(over='ignore'):
        z = np.asarray(values, dtype=np.uint64) + np.uint64(salt) * _GOLDEN
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def rank_order(uri_hash: np.ndarray, salt: int = 0) -> np.ndarray:
    """Positions of the interactions in the order of their mixed hash."""
    return np.argsort(mix(uri_hash, salt), kind='stable')


def _first(mask: np.ndarray, count: int, rank: np.ndarray) -> np.ndarray:
    """The ``count`` lowest-ranked positions among the ``True`` positions of ``mask``."""
    chosen = np.flatnonzero(mask)
    if len(chosen) > count:
        chosen = chosen[np.argsort(rank[chosen], kind='stable')[:count]]
    return chosen


def _pair_key(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    with np.errstate(over='ignore'):
        return mix(first, 1) ^ (np.asarray(second, dtype=np.uint64) * np.uint64(0xC2B2AE3D27D4EB4F))


def block_answered_mask(inputs: LabelInput, *, within_ms: int) -> np.ndarray:
    """True for each interaction that the target answered with a block within ``within_ms`` of it.

    A block before the interaction, by the other account, or after the time does not count.
    """
    count = int(len(inputs.uri_hash))
    answered = np.zeros(count, dtype=bool)
    if count == 0 or len(inputs.block_ms) == 0:
        return answered
    block_pair = _pair_key(inputs.block_blocker, inputs.block_blocked)
    order = np.lexsort((inputs.block_ms, block_pair))
    pair_sorted = block_pair[order]
    ms_sorted = np.asarray(inputs.block_ms, dtype=np.int64)[order]
    pair = _pair_key(inputs.target, inputs.author)
    low = np.searchsorted(pair_sorted, pair, side='left')
    high = np.searchsorted(pair_sorted, pair, side='right')
    event_ms = np.asarray(inputs.event_ms, dtype=np.int64)
    # Only interactions whose pair has any block in the answering direction need a closer look.
    for index in np.flatnonzero(high > low):
        start = int(low[index])
        position = start + int(np.searchsorted(ms_sorted[start : high[index]], event_ms[index]))
        if position < high[index] and ms_sorted[position] <= event_ms[index] + within_ms:
            answered[index] = True
    return answered


def block_positive_mask(inputs: LabelInput, *, positive_minutes: int) -> np.ndarray:
    """True for each interaction that the target answered with a block within the delay."""
    return block_answered_mask(inputs, within_ms=positive_minutes * 60_000)


def elapsed_horizon_mask(event_ms: np.ndarray, *, now_ms: int, horizon_days: int) -> np.ndarray:
    """True when the whole negative horizon lies in the past, so a non-block means something."""
    return np.asarray(event_ms, dtype=np.int64) + horizon_days * 86_400_000 <= now_ms


def negative_mask(inputs: LabelInput, *, positives: np.ndarray, elapsed: np.ndarray, horizon_days: int) -> np.ndarray:
    """True for each interaction that is a block negative.

    The horizon has elapsed and no block of the author by the target came inside it. An interaction whose
    block came after the positive delay but inside the horizon is neither a positive nor a negative, so it
    enters no training source.
    """
    answered = block_answered_mask(inputs, within_ms=horizon_days * 86_400_000)
    return np.asarray(elapsed, dtype=bool) & ~np.asarray(positives, dtype=bool) & ~answered


def split_codes(uri_hash: np.ndarray) -> np.ndarray:
    """70 / 15 / 15 into train, calibration and test, by the URI hash."""
    share = np.asarray(uri_hash, dtype=np.uint64) % np.uint64(100)
    return np.where(share < SPLIT_SHARES[0], TRAIN, np.where(share < 85, CALIBRATION, TEST)).astype(np.int8)


def choose_jev_sample(
    inputs: LabelInput,
    *,
    positives: np.ndarray,
    elapsed: np.ndarray,
    labelled: np.ndarray,
    config: LabelConfig,
) -> np.ndarray:
    """The interactions to send to Jev in this refresh.

    Two disjoint strata: block positives, and interactions with an elapsed horizon that are not in the first
    stratum. A pool larger than its quota is reduced to its quota by the deterministic order. Interactions
    that already have a label are left out, so a pair is never paid for twice.
    """
    rank = np.empty(len(inputs.uri_hash), dtype=np.int64)
    rank[rank_order(inputs.uri_hash)] = np.arange(len(rank))
    sample = np.zeros(len(rank), dtype=bool)
    sample[_first(positives & ~labelled, config.jev_stratum_positives, rank)] = True
    sample[_first(elapsed & ~labelled & ~sample, config.jev_stratum_random, rank)] = True
    return sample


def build_labels(
    inputs: LabelInput,
    jev_h: np.ndarray,
    *,
    positives: np.ndarray,
    negatives: np.ndarray,
    config: LabelConfig,
) -> LabelSet:
    """Compose the training rows. Precedence: Jev label, then block positive, then random negative.

    A Jev-labelled row takes its target from the label and its split from the URI hash. A weak positive and a
    random negative have no Jev label and always train, so a calibration or test pair is in no training
    source. ``negatives`` marks the block negatives, as ``negative_mask`` does: nothing else trains as a 0.
    """
    jev_h = np.asarray(jev_h, dtype=np.float32)
    labelled = ~np.isnan(jev_h)
    rank = np.empty(len(inputs.uri_hash), dtype=np.int64)
    rank[rank_order(inputs.uri_hash)] = np.arange(len(rank))

    source = np.full(len(rank), -1, dtype=np.int8)
    target = np.zeros(len(rank), dtype=np.uint8)
    weight = np.zeros(len(rank), dtype=np.float32)
    split = np.full(len(rank), -1, dtype=np.int8)
    source[labelled] = JEV
    target[labelled] = (jev_h[labelled] >= 0.5).astype(np.uint8)
    weight[labelled] = 1.0
    split[labelled] = split_codes(inputs.uri_hash)[labelled]

    weak = _first(positives & ~labelled, config.max_weak_positives, rank)
    source[weak] = BLOCK
    target[weak] = 1
    weight[weak] = config.weak_positive_weight
    split[weak] = TRAIN

    negative = _first(negatives & ~labelled, config.negative_sample, rank)
    source[negative] = RANDOM
    target[negative] = 0
    weight[negative] = config.negative_weight
    split[negative] = TRAIN

    keep = np.flatnonzero(source >= 0)
    keep = keep[np.argsort(rank[keep], kind='stable')]
    return LabelSet(
        uri_hash=np.asarray(inputs.uri_hash, dtype=np.uint64)[keep],
        kind=np.asarray(inputs.kind, dtype=np.int8)[keep],
        target=target[keep],
        weight=weight[keep],
        split=split[keep],
        label_source=source[keep],
        label_h=np.where(labelled, jev_h, np.nan).astype(np.float32)[keep],
    )


def label_policy_hash(config: LabelConfig) -> str:
    """A digest of every setting that changes which rows a dataset holds."""
    fields = (
        config.block_positive_minutes,
        config.block_negative_horizon_days,
        config.jev_stratum_positives,
        config.jev_stratum_random,
        config.max_weak_positives,
        config.negative_sample,
        config.weak_positive_weight,
        config.negative_weight,
        config.jev_model,
        config.prompt_version,
    )
    return hashlib.sha256(repr(fields).encode()).hexdigest()
