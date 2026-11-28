# pattern: Functional Core
"""The conflict matrix: pile-ons, swarms and one row per ordered faction pair and window.

Pure NumPy and pandas. Accounts are vertex indices and factions are dense indices. A cell is an ordered
pair (A to B), the diagonal included. Hostility is a probability that is summed, never thresholded early.
"""

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

REPLY = 0
QUOTE = 1
BLOCK = 2

CELL_COLUMNS = (
    'from_faction_id',
    'to_faction_id',
    'contact',
    'contact_expected',
    'contact_lift',
    'pileon_replies',
    'pileon_lift',
    'direction_pileon',
    'swarm_quotes',
    'swarm_lift',
    'direction_swarm',
    'blocks',
    'blocks_lift',
    'direction_blocks',
    'text_hostility',
    'text_scored',
    'text_lift',
    'direction_text',
    'conflict_score',
    'distinct_sources',
    'distinct_targets',
)


@dataclass(frozen=True)
class Events:
    """Interactions whose author and target are both placed in the current image.

    ``kind`` is ``REPLY``, ``QUOTE`` or ``BLOCK``. ``from_faction`` and ``to_faction`` are dense faction
    indices. ``thread_key`` groups the replies of one thread and the quotes of one post. ``h`` is the
    hostility of the text and ``NaN`` where there is no score.
    """

    event_ms: np.ndarray
    kind: np.ndarray
    author: np.ndarray
    target: np.ndarray
    from_faction: np.ndarray
    to_faction: np.ndarray
    thread_key: np.ndarray
    h: np.ndarray
    accounts: int
    factions: int
    faction_ids: np.ndarray

    def __len__(self) -> int:
        return int(len(self.event_ms))

    def in_window(self, start_ms: int, end_ms: int) -> Events:
        mask = (self.event_ms >= start_ms) & (self.event_ms < end_ms)
        return self.take(mask)

    def take(self, mask: np.ndarray) -> Events:
        return Events(
            event_ms=self.event_ms[mask],
            kind=self.kind[mask],
            author=self.author[mask],
            target=self.target[mask],
            from_faction=self.from_faction[mask],
            to_faction=self.to_faction[mask],
            thread_key=self.thread_key[mask],
            h=self.h[mask],
            accounts=self.accounts,
            factions=self.factions,
            faction_ids=self.faction_ids,
        )


def swarm_flags(
    events: Events,
    *,
    min_repliers: int,
    reply_window_minutes: int,
    min_quoters: int,
    quote_window_minutes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Mark the pile-on replies and the swarm quotes.

    Buckets are tumbling windows aligned to the epoch. Within a bucket, replies group by thread root, target
    account and author faction, and quotes by quoted post and author faction. A group with enough distinct
    authors marks every event in it, the earlier ones included.
    """
    size = len(events)
    pileon = np.zeros(size, dtype=bool)
    swarm = np.zeros(size, dtype=bool)
    threads = np.asarray(events.thread_key).view(np.int64)
    for kind, minutes, minimum, flags in (
        (REPLY, reply_window_minutes, min_repliers, pileon),
        (QUOTE, quote_window_minutes, min_quoters, swarm),
    ):
        selected = np.flatnonzero(events.kind == kind)
        if len(selected) == 0:
            continue
        bucket = events.event_ms[selected] // (minutes * 60_000)
        span = events.accounts * events.factions
        packed = bucket * span + events.target[selected] * events.factions + events.from_faction[selected]
        order = np.lexsort((events.author[selected], threads[selected], packed))
        sorted_thread = threads[selected][order]
        sorted_packed = packed[order]
        sorted_author = events.author[selected][order]
        new_group = np.empty(len(order), dtype=bool)
        new_group[0] = True
        new_group[1:] = (sorted_thread[1:] != sorted_thread[:-1]) | (sorted_packed[1:] != sorted_packed[:-1])
        new_author = new_group.copy()
        new_author[1:] |= sorted_author[1:] != sorted_author[:-1]
        group = np.cumsum(new_group) - 1
        distinct = np.bincount(group, weights=new_author)
        flagged = np.empty(len(order), dtype=bool)
        flagged[order] = distinct[group] >= minimum
        flags[selected[flagged]] = True
    return pileon, swarm


@dataclass(frozen=True)
class CellCounts:
    """Observed counts per dense cell ``from_faction * factions + to_faction``."""

    factions: int
    faction_ids: np.ndarray
    contact: np.ndarray
    pileon: np.ndarray
    swarm: np.ndarray
    blocks: np.ndarray
    text_hostility: np.ndarray
    text_scored: np.ndarray
    sources: np.ndarray
    targets: np.ndarray


def cell_counts(events: Events, pileon: np.ndarray, swarm: np.ndarray) -> CellCounts:
    """Count every signal per ordered faction pair over the whole set of events."""
    cells = events.factions * events.factions
    cell = events.from_faction * events.factions + events.to_faction
    is_block = events.kind == BLOCK
    scored = ~np.isnan(events.h) & ~is_block
    h = np.where(scored, events.h, 0.0).astype(np.float64)
    return CellCounts(
        factions=events.factions,
        faction_ids=events.faction_ids,
        contact=np.bincount(cell[~is_block], minlength=cells),
        pileon=np.bincount(cell[pileon], minlength=cells),
        swarm=np.bincount(cell[swarm], minlength=cells),
        blocks=np.bincount(cell[is_block], minlength=cells),
        text_hostility=np.bincount(cell[scored], weights=h[scored], minlength=cells),
        text_scored=np.bincount(cell[scored], minlength=cells),
        sources=_distinct_per_cell(cell, events.author, events.accounts, cells),
        targets=_distinct_per_cell(cell, events.target, events.accounts, cells),
    )


def _distinct_per_cell(cell: np.ndarray, account: np.ndarray, accounts: int, cells: int) -> np.ndarray:
    """How many distinct accounts each cell has on one side."""
    keys = np.unique(cell * accounts + account)
    return np.bincount(keys // accounts, minlength=cells)


@dataclass(frozen=True)
class HostileCounts:
    """Hostile events sent and received per account, by signal. Indexed by vertex."""

    pileon_sent: np.ndarray
    pileon_received: np.ndarray
    swarm_sent: np.ndarray
    swarm_received: np.ndarray
    blocks_sent: np.ndarray
    blocks_received: np.ndarray
    text_sent: np.ndarray
    text_received: np.ndarray


def hostile_counts(events: Events, pileon: np.ndarray, swarm: np.ndarray) -> HostileCounts:
    """Hostile events sent and received per account over the whole set of events."""
    n = events.accounts
    is_block = events.kind == BLOCK
    scored = ~np.isnan(events.h) & ~is_block
    h = np.where(scored, events.h, 0.0).astype(np.float64)
    return HostileCounts(
        pileon_sent=np.bincount(events.author[pileon], minlength=n),
        pileon_received=np.bincount(events.target[pileon], minlength=n),
        swarm_sent=np.bincount(events.author[swarm], minlength=n),
        swarm_received=np.bincount(events.target[swarm], minlength=n),
        blocks_sent=np.bincount(events.author[is_block], minlength=n),
        blocks_received=np.bincount(events.target[is_block], minlength=n),
        text_sent=np.bincount(events.author[scored], weights=h[scored], minlength=n),
        text_received=np.bincount(events.target[scored], weights=h[scored], minlength=n),
    )


def shrunk_lift(observed: np.ndarray, contact: np.ndarray, contact_total: float, k: float) -> np.ndarray:
    """``((observed + k * g) / (contact + k)) / g`` with ``g = total / contact_total``.

    The result is the intensity relative to the global rate, pulled toward 1 where the cell has little
    contact. It is ``NaN`` everywhere when the signal never occurs or there is no contact at all.
    """
    total = float(observed.sum())
    if total <= 0 or contact_total <= 0:
        return np.full(len(observed), np.nan, dtype=np.float64)
    g = total / contact_total
    with np.errstate(divide='ignore', invalid='ignore'):
        return ((observed + k * g) / (contact + k)) / g


def direction(forward: np.ndarray, backward: np.ndarray) -> np.ndarray:
    """``(forward - backward) / (forward + backward)``, ``NaN`` where the sum is 0."""
    total = forward + backward
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(total > 0, (forward - backward) / np.where(total > 0, total, 1.0), np.nan)


def cells_frame(counts: CellCounts, *, prior_strength: float, min_text_scored: int, has_model: bool) -> pd.DataFrame:
    """One row per cell with contact or blocks, in ``CELL_COLUMNS`` order.

    ``contact`` is replies plus quotes. The lifts are shrunk toward the global rate with ``prior_strength``
    pseudo-events. ``conflict_score`` is the smaller of the contact lift and the largest lift of a signal.
    """
    factions = counts.factions
    grid = (factions, factions)
    contact_total = float(counts.contact.sum())
    out = counts.contact.reshape(grid).sum(axis=1)
    into = counts.contact.reshape(grid).sum(axis=0)
    expected = (np.outer(out, into) / contact_total).ravel() if contact_total > 0 else np.zeros(factions * factions)
    with np.errstate(divide='ignore', invalid='ignore'):
        contact_lift = np.where(expected > 0, counts.contact / np.where(expected > 0, expected, 1.0), np.nan)

    reverse = np.arange(factions * factions).reshape(grid).T.ravel()
    k = prior_strength
    pileon_lift = shrunk_lift(counts.pileon, counts.contact, contact_total, k)
    swarm_lift = shrunk_lift(counts.swarm, counts.contact, contact_total, k)
    blocks_lift = shrunk_lift(counts.blocks, counts.contact, contact_total, k)

    scored_total = float(counts.text_scored.sum())
    hostility_total = float(counts.text_hostility.sum())
    text_lift = np.full(factions * factions, np.nan, dtype=np.float64)
    if has_model and scored_total > 0 and hostility_total > 0:
        g_text = hostility_total / scored_total
        enough = counts.text_scored >= min_text_scored
        with np.errstate(divide='ignore', invalid='ignore'):
            text_lift = np.where(
                enough, ((counts.text_hostility + k * g_text) / (counts.text_scored + k)) / g_text, np.nan
            )

    lifts = np.vstack([pileon_lift, swarm_lift, blocks_lift, text_lift])
    best = _nan_max(lifts)
    # The score needs contact, the contact lift and at least one signal lift. A cell with blocks but no
    # contact has a contact lift of zero when its expectation is positive, and still gets no score:
    # ``np.fmin`` would ignore a NaN and a zero lift would read as "no conflict".
    with np.errstate(invalid='ignore'):
        score = np.where(
            (counts.contact == 0) | np.isnan(contact_lift) | np.isnan(best), np.nan, np.minimum(contact_lift, best)
        )

    written = (counts.contact > 0) | (counts.blocks > 0)
    index = np.flatnonzero(written)
    frame = pd.DataFrame(
        {
            'from_faction_id': counts.faction_ids[index // factions],
            'to_faction_id': counts.faction_ids[index % factions],
            'contact': counts.contact[index].astype(np.int64),
            'contact_expected': expected[index],
            'contact_lift': _optional(contact_lift[index]),
            'pileon_replies': counts.pileon[index].astype(np.int64),
            'pileon_lift': _optional(pileon_lift[index]),
            'direction_pileon': _optional(direction(counts.pileon, counts.pileon[reverse])[index]),
            'swarm_quotes': counts.swarm[index].astype(np.int64),
            'swarm_lift': _optional(swarm_lift[index]),
            'direction_swarm': _optional(direction(counts.swarm, counts.swarm[reverse])[index]),
            'blocks': counts.blocks[index].astype(np.int64),
            'blocks_lift': _optional(blocks_lift[index]),
            'direction_blocks': _optional(direction(counts.blocks, counts.blocks[reverse])[index]),
            'text_hostility': counts.text_hostility[index],
            'text_scored': counts.text_scored[index].astype(np.int64),
            'text_lift': _optional(text_lift[index]),
            'direction_text': _optional(direction(counts.text_hostility, counts.text_hostility[reverse])[index]),
            'conflict_score': _optional(score[index]),
            'distinct_sources': counts.sources[index].astype(np.int64),
            'distinct_targets': counts.targets[index].astype(np.int64),
        }
    )
    return frame.loc[:, list(CELL_COLUMNS)]


def _nan_max(values: np.ndarray) -> np.ndarray:
    """The largest lift per cell. A cell where every lift is NaN gets NaN and no warning."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmax(values, axis=0)


def _optional(values: np.ndarray) -> np.ndarray:
    """An object column that holds ``None`` for ``NaN``.

    pandas would turn a list with ``None`` into floats with ``NaN``, and ClickHouse would store that as a
    NaN and not as a NULL.
    """
    return np.array([None if not np.isfinite(value) else float(value) for value in values], dtype=object)
