# pattern: Functional Core
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

import igraph as ig
import leidenalg

EvolutionType = Literal['birth', 'death', 'continuation', 'merge', 'split']


@dataclass(frozen=True)
class PairRow:
    date: date
    account_a: str
    account_b: str
    weight: int
    newman_weight: float
    shared_uris: list[str]


@dataclass(frozen=True)
class ClusterResult:
    cluster_id: str
    members: frozenset[str]
    member_count: int
    total_edges: int
    total_weight: int
    unique_uris: int
    sample_dids: list[str]
    sample_uris: list[str]
    resolution_parameter: float


@dataclass(frozen=True)
class TimestampedCluster(ClusterResult):
    temporal_spread_hours: float
    mean_posting_interval_seconds: float


@dataclass(frozen=True)
class EvolutionEvent:
    cluster_id: str
    members: frozenset[str]
    evolution_type: EvolutionType
    predecessor_cluster_ids: tuple[str, ...]
    jaccard_score: float


def compute_jaccard(set_a: frozenset[str], set_b: frozenset[str]) -> float:
    """
    Compute Jaccard similarity between two sets.

    Args:
        set_a: First set of DIDs.
        set_b: Second set of DIDs.

    Returns:
        Jaccard similarity: |A ∩ B| / |A ∪ B|.
        Returns 0.0 if both sets are empty.
    """
    if not set_a and not set_b:
        return 0.0

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)

    return intersection / union


def build_graph(pairs: list[PairRow], min_edge_weight: int) -> ig.Graph:
    """
    Build an undirected weighted graph from pairs, filtering by minimum raw edge weight.

    Duplicate (account_a, account_b) pairs are aggregated before edge creation:
    raw weights and Newman weights are summed, shared URI lists are unioned.
    Edges are added in a single batch with attribute lists, so the graph can
    never contain parallel edges or None-valued attributes.

    Returns an empty graph (0 vertices, 0 edges) if no qualifying pairs.
    """
    aggregated: dict[tuple[str, str], tuple[int, float, set[str]]] = {}
    for pair in pairs:
        key = (pair.account_a, pair.account_b) if pair.account_a < pair.account_b else (pair.account_b, pair.account_a)
        if key in aggregated:
            weight, newman_weight, uris = aggregated[key]
            aggregated[key] = (
                weight + pair.weight,
                newman_weight + pair.newman_weight,
                uris | set(pair.shared_uris),
            )
        else:
            aggregated[key] = (pair.weight, pair.newman_weight, set(pair.shared_uris))

    # Filter by min_edge_weight on the *aggregated* raw weight so that
    # fragmented duplicate pairs (each below threshold) are combined first.
    aggregated = {key: val for key, val in aggregated.items() if val[0] >= min_edge_weight}

    if not aggregated:
        return ig.Graph()

    sorted_dids = sorted({did for key in aggregated for did in key})
    did_to_idx = {did: idx for idx, did in enumerate(sorted_dids)}

    graph = ig.Graph(len(sorted_dids))
    graph.vs['name'] = sorted_dids

    sorted_keys = sorted(aggregated)
    graph.add_edges([(did_to_idx[a], did_to_idx[b]) for a, b in sorted_keys])
    graph.es['weight'] = [aggregated[key][0] for key in sorted_keys]
    graph.es['newman_weight'] = [aggregated[key][1] for key in sorted_keys]
    graph.es['shared_uris'] = [sorted(aggregated[key][2]) for key in sorted_keys]

    return graph


def cluster_graph(graph: ig.Graph, resolution: float, min_cluster_size: int) -> list[ClusterResult]:
    """
    Run Leiden community detection on a weighted graph and compute per-cluster metrics.

    Community detection is optimized over Newman-weighted edges, where weights are
    assigned using Newman's collaboration weighting (Σ 1/(k_uri − 1) per pair). This
    down-weights viral URIs that appear in many shares. The CPM resolution parameter
    compares edge-weight density against the threshold; after switching to Newman weights,
    the default resolution may require re-tuning to maintain desired cluster density.

    Args:
        graph: igraph Graph object with weighted edges (weight, newman_weight) and shared_uris attributes.
        resolution: CPM resolution parameter for Leiden algorithm.
        min_cluster_size: Minimum number of members required to keep a cluster.

    Returns:
        List of ClusterResult objects for clusters meeting the size threshold.
    """
    if graph.vcount() == 0:
        return []

    partition = leidenalg.find_partition(
        graph,
        leidenalg.CPMVertexPartition,
        weights='newman_weight',
        resolution_parameter=resolution,
    )

    membership = partition.membership

    clusters_by_id = {}
    for vertex_idx, cluster_id in enumerate(membership):
        if cluster_id not in clusters_by_id:
            clusters_by_id[cluster_id] = []
        clusters_by_id[cluster_id].append(vertex_idx)

    results = []
    for cluster_id, vertex_indices in clusters_by_id.items():
        if len(vertex_indices) < min_cluster_size:
            continue

        members = frozenset(graph.vs[idx]['name'] for idx in vertex_indices)
        member_list = sorted(members)

        subgraph = graph.induced_subgraph(vertex_indices)

        total_edges = subgraph.ecount()
        total_weight = sum(subgraph.es['weight']) if subgraph.ecount() > 0 else 0

        unique_uris_set = set()
        for edge in subgraph.es:
            if 'shared_uris' in edge.attributes():
                unique_uris_set.update(edge['shared_uris'])

        unique_uris = len(unique_uris_set)

        sample_dids = member_list[:10]

        sample_uris_list = list(unique_uris_set)[:10]

        result = ClusterResult(
            cluster_id='',
            members=members,
            member_count=len(members),
            total_edges=total_edges,
            total_weight=total_weight,
            unique_uris=unique_uris,
            sample_dids=sample_dids,
            sample_uris=sample_uris_list,
            resolution_parameter=resolution,
        )
        results.append(result)

    return results


def compute_temporal_metrics(
    cluster: ClusterResult,
    member_timestamps: dict[str, list[datetime]],
) -> TimestampedCluster:
    """
    Compute temporal metrics for a cluster given member timestamps.

    Args:
        cluster: ClusterResult to extend with temporal metrics.
        member_timestamps: Mapping of DID -> sorted list of datetime objects.

    Returns:
        TimestampedCluster with temporal_spread_hours and mean_posting_interval_seconds.
    """
    all_timestamps = []
    for did in cluster.members:
        if did in member_timestamps:
            all_timestamps.extend(member_timestamps[did])

    if not all_timestamps:
        temporal_spread_hours = 0.0
        mean_posting_interval_seconds = 0.0
    else:
        all_timestamps_sorted = sorted(all_timestamps)

        earliest = all_timestamps_sorted[0]
        latest = all_timestamps_sorted[-1]

        temporal_spread_hours = (latest - earliest).total_seconds() / 3600.0

        if len(all_timestamps_sorted) < 2:
            mean_posting_interval_seconds = 0.0
        else:
            intervals = []
            for i in range(len(all_timestamps_sorted) - 1):
                interval_seconds = (all_timestamps_sorted[i + 1] - all_timestamps_sorted[i]).total_seconds()
                intervals.append(interval_seconds)

            mean_posting_interval_seconds = sum(intervals) / len(intervals)

    return TimestampedCluster(
        cluster_id=cluster.cluster_id,
        members=cluster.members,
        member_count=cluster.member_count,
        total_edges=cluster.total_edges,
        total_weight=cluster.total_weight,
        unique_uris=cluster.unique_uris,
        sample_dids=cluster.sample_dids,
        sample_uris=cluster.sample_uris,
        resolution_parameter=cluster.resolution_parameter,
        temporal_spread_hours=temporal_spread_hours,
        mean_posting_interval_seconds=mean_posting_interval_seconds,
    )


def compute_evolution(
    current_clusters: list[ClusterResult],
    previous_membership: dict[str, frozenset[str]],
    run_date: date,
    jaccard_threshold: float,
) -> list[EvolutionEvent]:
    """
    Classify current clusters by evolution type (birth, death, continuation, merge, split)
    and assign stable cluster IDs.

    Args:
        current_clusters: List of clusters detected today.
        previous_membership: Mapping of previous_cluster_id -> frozenset of DIDs.
        run_date: Today's date, used for generating birth IDs.
        jaccard_threshold: Minimum Jaccard for considering a previous cluster as a match.

    Returns:
        List of EvolutionEvent objects classifying each current cluster and any deaths.
    """
    events = []

    if not current_clusters and not previous_membership:
        return []

    if not previous_membership:
        birth_counter = 1
        for cluster in current_clusters:
            cluster_id = f'{run_date.isoformat()}-{birth_counter:04d}'
            event = EvolutionEvent(
                cluster_id=cluster_id,
                members=cluster.members,
                evolution_type='birth',
                predecessor_cluster_ids=(),
                jaccard_score=0.0,
            )
            events.append(event)
            birth_counter += 1
        return events

    current_matches: dict[frozenset[str], list[tuple[str, float]]] = {}
    for curr_cluster in current_clusters:
        matches = []
        for prev_id, prev_members in previous_membership.items():
            jaccard = compute_jaccard(prev_members, curr_cluster.members)
            if jaccard >= jaccard_threshold:
                matches.append((prev_id, jaccard))

        current_matches[curr_cluster.members] = matches

    previous_matches: dict[str, list[tuple[frozenset[str], float]]] = {}
    for prev_id, prev_members in previous_membership.items():
        matches = []
        for curr_cluster in current_clusters:
            jaccard = compute_jaccard(prev_members, curr_cluster.members)
            if jaccard >= jaccard_threshold:
                matches.append((curr_cluster.members, jaccard))

        previous_matches[prev_id] = matches

    birth_counter = 1

    for curr_cluster in current_clusters:
        curr_members = curr_cluster.members
        matches = current_matches[curr_members]

        if not matches:
            cluster_id = f'{run_date.isoformat()}-{birth_counter:04d}'
            event = EvolutionEvent(
                cluster_id=cluster_id,
                members=curr_members,
                evolution_type='birth',
                predecessor_cluster_ids=(),
                jaccard_score=0.0,
            )
            events.append(event)
            birth_counter += 1
            continue

        best_match_id, best_jaccard = max(matches, key=lambda x: x[1])
        matching_prev_ids = [m[0] for m in matches]

        if len(matching_prev_ids) > 1:
            cluster_id = f'{run_date.isoformat()}-{birth_counter:04d}'
            event = EvolutionEvent(
                cluster_id=cluster_id,
                members=curr_members,
                evolution_type='merge',
                predecessor_cluster_ids=tuple(sorted(matching_prev_ids)),
                jaccard_score=best_jaccard,
            )
            events.append(event)
            birth_counter += 1
        else:
            num_current_from_prev = len(previous_matches.get(best_match_id, []))
            if num_current_from_prev > 1:
                cluster_id = f'{run_date.isoformat()}-{birth_counter:04d}'
                event = EvolutionEvent(
                    cluster_id=cluster_id,
                    members=curr_members,
                    evolution_type='split',
                    predecessor_cluster_ids=(best_match_id,),
                    jaccard_score=best_jaccard,
                )
                events.append(event)
                birth_counter += 1
            else:
                event = EvolutionEvent(
                    cluster_id=best_match_id,
                    members=curr_members,
                    evolution_type='continuation',
                    predecessor_cluster_ids=(best_match_id,),
                    jaccard_score=best_jaccard,
                )
                events.append(event)

    for prev_id in previous_membership:
        if not previous_matches.get(prev_id):
            event = EvolutionEvent(
                cluster_id=prev_id,
                members=previous_membership[prev_id],
                evolution_type='death',
                predecessor_cluster_ids=(prev_id,),
                jaccard_score=0.0,
            )
            events.append(event)

    return events
