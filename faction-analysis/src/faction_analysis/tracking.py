# pattern: Functional Core
"""Faction identity across runs: tracking, lineage, moves and agreement.

Tracking follows the contingency table of the reference paper. Each old faction points at the new community
that holds most of its weighted degree. Among the old factions that point at one new community, the one with
the largest overlap wins and that community takes its ID. Everything here is a pure function of arrays.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

LineageKind = Literal['birth', 'split', 'merge', 'death']
MoveKind = Literal['assign', 'move', 'drop']
MOVE_KINDS: tuple[MoveKind, ...] = ('assign', 'move', 'drop')
UNPLACED = -1


@dataclass(frozen=True)
class LineageEvent:
    """One event of one faction in one run.

    ``related_faction_id`` is the parent of a split and the winner of a merge. It is ``None`` for a birth
    and for a death.
    """

    faction_id: int
    event: LineageKind
    related_faction_id: int | None


@dataclass(frozen=True)
class Tracked:
    """The partition of one run after its communities received faction IDs."""

    faction: np.ndarray
    next_faction_id: int
    lineage: tuple[LineageEvent, ...]
    adjusted_rand: float | None


@dataclass(frozen=True)
class MoveTable:
    """Assignment changes between two runs. Code ``i`` of ``kind`` is ``MOVE_KINDS[i]``."""

    did: np.ndarray
    kind: np.ndarray
    from_faction_id: np.ndarray
    to_faction_id: np.ndarray

    def __len__(self) -> int:
        return int(len(self.did))


@dataclass(frozen=True)
class MoveDiff:
    """Moves of a run, plus per current vertex the faction it had before and when its tenure started."""

    moves: MoveTable
    since_run_id: np.ndarray
    prev_faction_id: np.ndarray


def _dense(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels, index = np.unique(values, return_inverse=True)
    return labels, index.astype(np.int64).ravel()


def _contingency(
    old_index: np.ndarray, new_index: np.ndarray, weights: np.ndarray, old_count: int, new_count: int
) -> np.ndarray:
    """Weighted degree in every cell of the old-by-new table."""
    if old_count == 0 or new_count == 0 or len(old_index) == 0:
        return np.zeros((old_count, new_count), dtype=np.float64)
    cells = np.bincount(old_index * new_count + new_index, weights=weights, minlength=old_count * new_count)
    return cells.reshape(old_count, new_count)


def _winners(table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The old faction each old row points at, and the old row that won each new community.

    A row points at its largest cell. On a tie it points at the smaller community index. Among the rows that
    point at one community, the row with the larger cell wins. On a tie the smaller old faction ID wins,
    which is the smaller row index because ``old_labels`` is sorted.
    """
    old_count, new_count = table.shape
    points_to = np.full(old_count, UNPLACED, dtype=np.int64)
    winner = np.full(new_count, UNPLACED, dtype=np.int64)
    has_degree = table.sum(axis=1) > 0
    if new_count:
        points_to[has_degree] = np.argmax(table[has_degree], axis=1)
    for old in np.flatnonzero(has_degree):
        community = int(points_to[old])
        current = int(winner[community])
        if current < 0 or table[old, community] > table[current, community]:
            winner[community] = old
    return points_to, winner


def _split_parents(table: np.ndarray, points_to: np.ndarray, winner: np.ndarray, split_share: float) -> np.ndarray:
    """The old row that a new community split from, or ``UNPLACED``. Indexed by new community."""
    old_count, new_count = table.shape
    parent = np.full(new_count, UNPLACED, dtype=np.int64)
    row_total = table.sum(axis=1)
    winners = [old for old in range(old_count) if row_total[old] > 0 and winner[points_to[old]] == old]
    for community in np.flatnonzero(winner < 0):
        column = table[:, community]
        eligible = [old for old in winners if column[old] > 0 and column[old] >= split_share * row_total[old]]
        if eligible:
            # Largest cell wins. The lower row index breaks a tie.
            parent[community] = max(eligible, key=lambda old: (column[old], -old))
    return parent


def track(
    previous_faction: np.ndarray | None,
    current_community: np.ndarray,
    degree: np.ndarray,
    *,
    next_faction_id: int,
    split_share: float,
    previous_faction_ids: Sequence[int] | np.ndarray | None = None,
) -> Tracked:
    """Give every current community a faction ID.

    ``previous_faction`` holds the previous faction ID per current vertex and ``UNPLACED`` where the vertex
    had none. ``current_community`` holds the community index per current vertex and ``UNPLACED`` where the
    account is unplaced now. ``previous_faction_ids`` lists every faction of the previous image, so a faction
    whose accounts all left the graph is still retired. New IDs start at ``next_faction_id``.
    """
    current = np.asarray(current_community, dtype=np.int64)
    n = int(len(current))
    if n == 0:
        return Tracked(np.empty(0, dtype=np.int64), int(next_faction_id), (), None)
    placed_now = current >= 0
    weights = np.asarray(degree, dtype=np.float64)
    if previous_faction is None:
        previous = np.full(n, UNPLACED, dtype=np.int64)
    else:
        previous = np.asarray(previous_faction, dtype=np.int64)
    both = placed_now & (previous >= 0)

    community_labels, community_index = _dense(current[placed_now])
    new_count = int(len(community_labels))
    dense = np.full(n, UNPLACED, dtype=np.int64)
    dense[placed_now] = community_index

    seen = previous[both]
    known = [seen]
    if previous_faction_ids is not None:
        known.append(np.asarray(previous_faction_ids, dtype=np.int64))
    old_labels = np.unique(np.concatenate(known))
    old_count = int(len(old_labels))
    old_index = np.searchsorted(old_labels, seen)
    table = _contingency(old_index, dense[both], weights[both], old_count, new_count)

    points_to, winner = _winners(table)
    split_parent = _split_parents(table, points_to, winner, split_share)

    community_faction = np.full(new_count, UNPLACED, dtype=np.int64)
    for community in range(new_count):
        if winner[community] >= 0:
            community_faction[community] = old_labels[winner[community]]
        else:
            community_faction[community] = next_faction_id
            next_faction_id += 1

    lineage: list[LineageEvent] = []
    for community in np.flatnonzero(winner < 0):
        parent = int(split_parent[community])
        lineage.append(
            LineageEvent(
                faction_id=int(community_faction[community]),
                event='split' if parent >= 0 else 'birth',
                related_faction_id=int(old_labels[parent]) if parent >= 0 else None,
            )
        )
    for old in range(old_count):
        if table[old].sum() <= 0:
            lineage.append(LineageEvent(int(old_labels[old]), 'death', None))
        elif winner[points_to[old]] != old:
            lineage.append(LineageEvent(int(old_labels[old]), 'merge', int(old_labels[winner[points_to[old]]])))

    faction = np.full(n, UNPLACED, dtype=np.int64)
    faction[placed_now] = community_faction[dense[placed_now]]
    agreement = adjusted_rand_index(previous[both], current[both]) if int(both.sum()) else None
    return Tracked(faction, int(next_faction_id), tuple(lineage), agreement)


def _choose_two(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values * (values - 1.0) / 2.0


def adjusted_rand_index(first: np.ndarray, second: np.ndarray) -> float:
    """Adjusted Rand index of two label vectors over the same items.

    ``1.0`` for identical partitions. One item makes no pair, so the index has no denominator. It is ``1.0``
    then, here and in scikit-learn.
    """
    first = np.asarray(first, dtype=np.int64)
    second = np.asarray(second, dtype=np.int64)
    n = int(len(first))
    if n == 0 or len(second) != n:
        raise ValueError('adjusted_rand_index needs two label vectors of equal, non-zero length')
    if n == 1:
        return 1.0
    _, a = _dense(first)
    _, b = _dense(second)
    width = int(b.max()) + 1
    cells = np.bincount(a * width + b, minlength=(int(a.max()) + 1) * width)
    sum_cells = float(_choose_two(cells).sum())
    sum_rows = float(_choose_two(np.bincount(a)).sum())
    sum_columns = float(_choose_two(np.bincount(b)).sum())
    total = float(_choose_two(np.asarray([n]))[0])
    expected = sum_rows * sum_columns / total
    maximum = 0.5 * (sum_rows + sum_columns)
    if maximum == expected:
        return 1.0 if sum_cells == expected else 0.0
    return float((sum_cells - expected) / (maximum - expected))


_FNV_OFFSET = np.uint64(0xCBF29CE484222325)
_FNV_PRIME = np.uint64(0x100000001B3)
_ONE = np.uint64(1)


def stable_hash(values: Sequence[str] | np.ndarray) -> np.ndarray:
    """FNV-1a over the UTF-8 bytes of each string. The same string gives the same hash in every process."""
    if len(values) == 0:
        return np.empty(0, dtype=np.uint64)
    text = [value if isinstance(value, str) else str(value) for value in values]
    lengths = np.fromiter((len(value.encode()) for value in text), dtype=np.int64, count=len(text))
    width = int(lengths.max()) if len(lengths) else 0
    if width == 0:
        return np.full(len(text), _FNV_OFFSET, dtype=np.uint64)
    padded = b''.join(value.encode().ljust(width, b'\0') for value in text)
    matrix = np.frombuffer(padded, dtype=np.uint8).reshape(len(text), width)
    result = np.full(len(text), _FNV_OFFSET, dtype=np.uint64)
    for column in range(width):
        active = column < lengths
        result ^= np.where(active, matrix[:, column].astype(np.uint64), np.uint64(0))
        result *= np.where(active, _FNV_PRIME, _ONE)
    return result


def _lookup(sorted_keys: np.ndarray, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each key, its position in ``sorted_keys`` and whether it was found."""
    if len(sorted_keys) == 0:
        return np.zeros(len(keys), dtype=np.int64), np.zeros(len(keys), dtype=bool)
    position = np.searchsorted(sorted_keys, keys)
    clipped = np.minimum(position, len(sorted_keys) - 1)
    return clipped, sorted_keys[clipped] == keys


def diff_moves(
    previous_did: np.ndarray,
    previous_faction: np.ndarray,
    previous_since: np.ndarray,
    current_did: np.ndarray,
    current_faction: np.ndarray,
    run_id: str,
) -> MoveDiff:
    """Compare a new image with the previous promoted image.

    ``assign`` is an account placed now that the previous image did not place. ``move`` is an account in a
    different faction. ``drop`` is an account the previous image placed that is not placed now. An unchanged
    account produces no move and keeps its ``since_run_id``. The current arrays hold the placed accounts, and
    ``since_run_id`` and ``prev_faction_id`` line up with them.
    """
    previous_did = np.asarray(previous_did, dtype=object)
    current_did = np.asarray(current_did, dtype=object)
    prev_faction = np.asarray(previous_faction, dtype=np.int64)
    current = np.asarray(current_faction, dtype=np.int64)

    prev_hash = stable_hash(previous_did)
    prev_order = np.argsort(prev_hash, kind='stable')
    prev_hash_sorted = prev_hash[prev_order]
    prev_faction_sorted = prev_faction[prev_order]
    prev_since_sorted = np.asarray(previous_since, dtype=object)[prev_order]
    cur_hash = stable_hash(current_did)

    forward, found = _lookup(prev_hash_sorted, cur_hash)
    if len(prev_hash_sorted) == 0:
        before = np.full(len(current_did), UNPLACED, dtype=np.int64)
        since = np.full(len(current_did), run_id, dtype=object)
    else:
        before = np.where(found, prev_faction_sorted[forward], UNPLACED)
        unchanged = found & (before == current)
        since = np.where(unchanged, prev_since_sorted[forward], run_id)
    unchanged = found & (before == current)
    assign = ~found
    move = found & ~unchanged
    _, still_placed = _lookup(np.sort(cur_hash), prev_hash)
    count_assign, count_move = int(assign.sum()), int(move.sum())
    did = np.concatenate([current_did[assign], current_did[move], previous_did[~still_placed]])
    kind = np.concatenate([np.zeros(count_assign), np.ones(count_move), np.full(int((~still_placed).sum()), 2)]).astype(
        np.int8
    )
    from_faction = np.concatenate([np.full(count_assign, UNPLACED), before[move], prev_faction[~still_placed]])
    to_faction = np.concatenate([current[assign], current[move], np.full(int((~still_placed).sum()), UNPLACED)])
    return MoveDiff(
        moves=MoveTable(did, kind, from_faction.astype(np.int64), to_faction.astype(np.int64)),
        since_run_id=np.asarray(since, dtype=object),
        prev_faction_id=before,
    )


def seed_confidence(
    main_community: np.ndarray,
    extra_partitions: Sequence[np.ndarray],
    degree: np.ndarray,
    *,
    split_share: float,
) -> np.ndarray:
    """Share of extra runs that put each account in its main faction. ``NaN`` where the main run left it unplaced."""
    main = np.asarray(main_community, dtype=np.int64)
    if not extra_partitions:
        return np.full(len(main), np.nan, dtype=np.float64)
    placed = main >= 0
    agree = np.zeros(len(main), dtype=np.float64)
    for extra in extra_partitions:
        tracked = track(main, extra, degree, next_faction_id=int(main.max()) + 1, split_share=split_share)
        agree += (tracked.faction == main) & placed
    return np.where(placed, agree / len(extra_partitions), np.nan)
