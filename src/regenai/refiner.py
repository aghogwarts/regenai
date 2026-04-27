"""
Refiner — Module 6 of the ReGenAI pipeline.

Takes cluster assignments from Module 5 and cleans them using
structural signals (project root tags, name prefixes).

Two operations:
    1. SPLIT: Clusters containing files from multiple project roots
       get split — each chunk moves to the existing pure cluster
       that owns its project, or becomes a new sub-cluster.
    2. REASSIGN NOISE: Noise points with a project_root tag get
       assigned to the dominant cluster for that project. Unanchored
       noise stays as noise (goes to _uncategorized later).
"""

import numpy as np
from collections import Counter, defaultdict
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

from regenai.config import GMM_SECONDARY_THRESHOLD
from regenai.clusterer import ClusterAssignment, ClusteringResult

console = Console()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RefinedResult:
    """Output of the refinement step."""

    assignments: list[ClusterAssignment]
    n_clusters: int
    n_noise: int
    splits_performed: int
    noise_reassigned: int
    chunk_ids: list[str]
    chunk_metadata: list[dict]


# ---------------------------------------------------------------------------
# Helper: find dominant cluster for a project root
# ---------------------------------------------------------------------------


def _find_project_cluster_map(
    assignments: list[ClusterAssignment],
    metadata: list[dict],
) -> dict[str, int]:
    """
    For each project root, find which cluster contains the most chunks
    from that project (excluding noise). This is the "home cluster"
    for that project.

    Returns:
        Mapping of project_root → cluster_id
    """
    # Count chunks per (project_root, cluster) pair
    project_cluster_counts: dict[str, Counter] = defaultdict(Counter)

    for assignment, meta in zip(assignments, metadata):
        root = meta.get("project_root", "")
        if not root or assignment.is_noise:
            continue
        project_cluster_counts[root][assignment.primary_cluster] += 1

    # For each project, pick the cluster with the most chunks
    project_home: dict[str, int] = {}
    for root, counts in project_cluster_counts.items():
        if counts:
            project_home[root] = counts.most_common(1)[0][0]

    return project_home


# ---------------------------------------------------------------------------
# Operation 1: Split mixed clusters
# ---------------------------------------------------------------------------


def _split_mixed_clusters(
    assignments: list[ClusterAssignment],
    metadata: list[dict],
    reduced_embeddings: np.ndarray,
) -> tuple[list[ClusterAssignment], int]:
    """
    For clusters containing chunks from multiple project roots,
    reassign each chunk to the home cluster of its project root.

    Unanchored chunks (no project root) in mixed clusters stay in
    their current cluster — they'll be resolved by semantic similarity.
    """
    project_home = _find_project_cluster_map(assignments, metadata)
    splits_performed = 0

    # Identify mixed clusters
    cluster_roots: dict[int, set[str]] = defaultdict(set)
    for assignment, meta in zip(assignments, metadata):
        if assignment.is_noise:
            continue
        root = meta.get("project_root", "")
        if root:
            cluster_roots[assignment.primary_cluster].add(root)

    mixed_clusters = {cid for cid, roots in cluster_roots.items() if len(roots) > 1}

    if not mixed_clusters:
        return assignments, 0

    # Reassign chunks in mixed clusters
    new_assignments: list[ClusterAssignment] = []
    for assignment, meta in zip(assignments, metadata):
        if assignment.primary_cluster in mixed_clusters and not assignment.is_noise:
            root = meta.get("project_root", "")
            if root and root in project_home:
                home_cluster = project_home[root]
                if home_cluster != assignment.primary_cluster:
                    # Move to home cluster
                    new_assignments.append(
                        ClusterAssignment(
                            chunk_id=assignment.chunk_id,
                            primary_cluster=home_cluster,
                            confidence=assignment.confidence,
                            secondary_cluster=assignment.primary_cluster,
                            secondary_confidence=GMM_SECONDARY_THRESHOLD,
                            is_noise=False,
                        )
                    )
                    splits_performed += 1
                    continue

        new_assignments.append(assignment)

    return new_assignments, splits_performed


# ---------------------------------------------------------------------------
# Operation 2: Reassign noise points
# ---------------------------------------------------------------------------


def _reassign_noise(
    assignments: list[ClusterAssignment],
    metadata: list[dict],
    reduced_embeddings: np.ndarray,
) -> tuple[list[ClusterAssignment], int]:
    """
    Noise points that have a project_root tag get assigned to the
    dominant cluster for that project.

    Noise points without a project root get assigned to the nearest
    cluster by embedding distance (cosine similarity in reduced space).

    Truly isolated points (very far from all clusters) stay as noise.
    """
    project_home = _find_project_cluster_map(assignments, metadata)
    noise_reassigned = 0

    # Compute cluster centroids in reduced space
    cluster_points: dict[int, list[int]] = defaultdict(list)
    for i, assignment in enumerate(assignments):
        if not assignment.is_noise and assignment.primary_cluster >= 0:
            cluster_points[assignment.primary_cluster].append(i)

    centroids: dict[int, np.ndarray] = {}
    for cid, indices in cluster_points.items():
        centroids[cid] = np.mean(reduced_embeddings[indices], axis=0)

    new_assignments: list[ClusterAssignment] = []

    for i, (assignment, meta) in enumerate(zip(assignments, metadata)):
        if not assignment.is_noise:
            new_assignments.append(assignment)
            continue

        root = meta.get("project_root", "")

        # Strategy 1: Has a project root — assign to project's home cluster
        if root and root in project_home:
            new_assignments.append(
                ClusterAssignment(
                    chunk_id=assignment.chunk_id,
                    primary_cluster=project_home[root],
                    confidence=0.6,  # moderate confidence for noise reassignment
                    secondary_cluster=None,
                    secondary_confidence=None,
                    is_noise=False,
                )
            )
            noise_reassigned += 1
            continue

        # Strategy 2: No project root — find nearest cluster by distance
        if centroids:
            point = reduced_embeddings[i]
            min_dist = float("inf")
            nearest_cluster = -1

            for cid, centroid in centroids.items():
                dist = np.linalg.norm(point - centroid)
                if dist < min_dist:
                    min_dist = dist
                    nearest_cluster = cid

            # Only reassign if reasonably close (within 2x average intra-cluster distance)
            if nearest_cluster >= 0:
                cluster_indices = cluster_points[nearest_cluster]
                avg_dist = np.mean(
                    [
                        np.linalg.norm(
                            reduced_embeddings[j] - centroids[nearest_cluster]
                        )
                        for j in cluster_indices
                    ]
                )

                if min_dist <= avg_dist * 2.0:
                    new_assignments.append(
                        ClusterAssignment(
                            chunk_id=assignment.chunk_id,
                            primary_cluster=nearest_cluster,
                            confidence=0.4,  # low confidence for distance-based reassignment
                            secondary_cluster=None,
                            secondary_confidence=None,
                            is_noise=False,
                        )
                    )
                    noise_reassigned += 1
                    continue

        # Still noise — keep it
        new_assignments.append(assignment)

    return new_assignments, noise_reassigned


# ---------------------------------------------------------------------------
# Main refiner
# ---------------------------------------------------------------------------


def refine_clusters(cluster_result: ClusteringResult) -> RefinedResult:
    """
    Refine cluster assignments using structural signals.

    Steps:
        1. Split mixed clusters (multiple project roots → separate)
        2. Reassign noise points (project-root-based, then distance-based)

    Args:
        cluster_result: Output from Module 5 (clusterer).

    Returns:
        RefinedResult with cleaned cluster assignments.
    """
    assignments = list(cluster_result.assignments)
    metadata = list(cluster_result.chunk_metadata)
    reduced = cluster_result.reduced_embeddings

    console.print("  [dim]Splitting mixed clusters...[/dim]")
    assignments, splits = _split_mixed_clusters(assignments, metadata, reduced)
    console.print(f"  [dim]  Reassigned {splits} chunks from mixed clusters[/dim]")

    console.print("  [dim]Reassigning noise points...[/dim]")
    assignments, noise_fixed = _reassign_noise(assignments, metadata, reduced)
    console.print(f"  [dim]  Reassigned {noise_fixed} noise points[/dim]")

    # Recount clusters and noise
    active_clusters = set()
    remaining_noise = 0
    for a in assignments:
        if a.is_noise:
            remaining_noise += 1
        else:
            active_clusters.add(a.primary_cluster)

    result = RefinedResult(
        assignments=assignments,
        n_clusters=len(active_clusters),
        n_noise=remaining_noise,
        splits_performed=splits,
        noise_reassigned=noise_fixed,
        chunk_ids=cluster_result.chunk_ids,
        chunk_metadata=metadata,
    )

    _print_refined_summary(result)

    return result


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def _print_refined_summary(result: RefinedResult) -> None:
    """Print a summary of refinement results."""
    console.print()
    console.print(f"  [bold]Refinement complete:[/bold]")
    console.print(f"    Active clusters:  {result.n_clusters}")
    console.print(f"    Remaining noise:  {result.n_noise}")
    console.print(f"    Chunks split:     {result.splits_performed}")
    console.print(f"    Noise reassigned: {result.noise_reassigned}")
    console.print()

    # Per-cluster breakdown (post-refinement)
    table = Table(title="Refined Cluster Breakdown", show_lines=False)
    table.add_column("Cluster", justify="center")
    table.add_column("Chunks", justify="right")
    table.add_column("Files", justify="right")
    table.add_column("Top File Types", max_width=30)
    table.add_column("Project Roots", max_width=40)

    cluster_data: dict[int, list[dict]] = defaultdict(list)
    for assignment, meta in zip(result.assignments, result.chunk_metadata):
        cid = -1 if assignment.is_noise else assignment.primary_cluster
        cluster_data[cid].append(meta)

    for cid in sorted(cluster_data.keys()):
        metas = cluster_data[cid]
        chunk_count = len(metas)
        files = set(m["source_file"] for m in metas)
        file_count = len(files)
        type_counts = Counter(m["file_type"] for m in metas)
        top_types = ", ".join(f"{t}({c})" for t, c in type_counts.most_common(3))
        roots = set(m["project_root"] for m in metas if m["project_root"])
        roots_str = ", ".join(sorted(roots)) if roots else "[dim]unanchored[/dim]"

        label = "noise" if cid == -1 else str(cid)
        table.add_row(label, str(chunk_count), str(file_count), top_types, roots_str)

    console.print(table)
    console.print()
