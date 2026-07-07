# pattern: Functional Core
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

import igraph as ig
import leidenalg
import numpy as np
from scipy.sparse import csr_array

from url_cosharing.similarity import ShareMatrix

EvolutionType = Literal['birth', 'death', 'continuation', 'merge', 'split']


@dataclass(frozen=True)
class ClusterResult:
    cluster_id: str
    members: frozenset[str]
    member_count: int
    total_edges: int
    total_weight: int
    unique_urls: int
    sample_dids: list[str]
    sample_urls: list[str]
    resolution_parameter: float
    mean_edge_similarity: float
    subgraph_density: float


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
        unique_urls=cluster.unique_urls,
        sample_dids=cluster.sample_dids,
        sample_urls=cluster.sample_urls,
        resolution_parameter=cluster.resolution_parameter,
        mean_edge_similarity=cluster.mean_edge_similarity,
        subgraph_density=cluster.subgraph_density,
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


def cluster_core(
    core: ig.Graph,
    matrix: ShareMatrix,
    tfidf: csr_array,
    resolution: float,
    min_cluster_size: int,
) -> list[ClusterResult]:
    """Leiden CPM over cosine-similarity weights on the dismantled core.

    Cluster metrics come from two sources: edge metrics (mean_edge_similarity,
    subgraph_density, total_edges) from the core subgraph; URL metrics
    (unique_urls, total_weight, sample_urls) from the bipartite share matrix.
    total_weight keeps its co-share-count semantics: sum over cluster URLs of
    C(k, 2) where k is the number of cluster members sharing that URL.
    """
    if core.vcount() == 0:
        return []

    partition = leidenalg.find_partition(
        core,
        leidenalg.CPMVertexPartition,
        weights='similarity',
        resolution_parameter=resolution,
    )

    account_to_row = {did: idx for idx, did in enumerate(matrix.accounts)}

    clusters_by_id: dict[int, list[int]] = {}
    for vertex_idx, cluster_id in enumerate(partition.membership):
        clusters_by_id.setdefault(cluster_id, []).append(vertex_idx)

    results = []
    for vertex_indices in clusters_by_id.values():
        if len(vertex_indices) < min_cluster_size:
            continue

        members = frozenset(core.vs[idx]['name'] for idx in vertex_indices)
        subgraph = core.induced_subgraph(vertex_indices)
        total_edges = subgraph.ecount()
        mean_edge_similarity = (
            float(np.mean(subgraph.es['similarity'])) if total_edges > 0 else 0.0
        )
        subgraph_density = float(subgraph.density(loops=False))

        member_rows = [account_to_row[did] for did in sorted(members)]
        sub_counts = matrix.counts[member_rows, :]
        sharers = np.asarray((sub_counts > 0).sum(axis=0)).ravel()
        unique_urls = int((sharers >= 2).sum())
        total_weight = int((sharers * (sharers - 1) // 2).sum())

        mass = np.asarray(tfidf[member_rows, :].sum(axis=0)).ravel()
        order = np.argsort(-mass, kind='stable')
        sample_urls = [matrix.urls[k] for k in order[:10] if mass[k] > 0]

        results.append(
            ClusterResult(
                cluster_id='',
                members=members,
                member_count=len(members),
                total_edges=total_edges,
                total_weight=total_weight,
                unique_urls=unique_urls,
                sample_dids=sorted(members)[:10],
                sample_urls=sample_urls,
                resolution_parameter=resolution,
                mean_edge_similarity=mean_edge_similarity,
                subgraph_density=subgraph_density,
            )
        )
    return results
