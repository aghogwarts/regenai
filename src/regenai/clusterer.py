"""
Clusterer — Module 5 of the ReGenAI pipeline.

The core algorithmic layer.  Takes embeddings from ChromaDB, reduces
dimensionality via UMAP, discovers clusters via HDBSCAN, then refines
with GMM for soft (probabilistic) cluster assignments.

Pipeline:
    1. Pull embeddings + metadata from ChromaDB
    2. UMAP → reduce to ~25 dimensions
    3. HDBSCAN → discover clusters, flag noise
    4. GMM → soft assignments with confidence scores
    5. Silhouette check → retry with adjusted params if below threshold
"""

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from collections import Counter

from rich.console import Console
from rich.table import Table

from regenai.config import (
    UMAP_N_COMPONENTS,
    UMAP_N_NEIGHBORS,
    UMAP_MIN_DIST,
    UMAP_METRIC,
    HDBSCAN_MIN_CLUSTER_SIZE,
    HDBSCAN_MIN_SAMPLES,
    GMM_COVARIANCE_TYPE,
    GMM_SECONDARY_THRESHOLD,
    SILHOUETTE_MIN_THRESHOLD,
    CLUSTER_RETRY_MAX,
    CHROMA_CHUNK_COLLECTION,
    CHROMA_PERSIST_DIR,
)

console = Console()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ClusterAssignment:
    """Cluster assignment for a single chunk."""

    chunk_id: str
    primary_cluster: int
    confidence: float
    secondary_cluster: int | None
    secondary_confidence: float | None
    is_noise: bool


@dataclass
class ClusteringResult:
    """Full output of the clustering pipeline."""

    assignments: list[ClusterAssignment]
    n_clusters: int
    n_noise: int
    silhouette_score: float
    reduced_embeddings: np.ndarray  # UMAP-reduced, for potential visualization
    cluster_labels: list[int]  # raw HDBSCAN labels per chunk
    chunk_ids: list[str]  # ordered chunk IDs matching the arrays
    chunk_metadata: list[dict]  # ordered metadata dicts matching the arrays


# ---------------------------------------------------------------------------
# Step 1: Pull embeddings from ChromaDB
# ---------------------------------------------------------------------------


def _load_from_chroma(input_dir: Path) -> tuple[np.ndarray, list[str], list[dict]]:
    """
    Load all chunk embeddings, IDs, and metadata from ChromaDB.

    Returns:
        embeddings: numpy array of shape (n_chunks, embedding_dim)
        ids: list of chunk IDs
        metadatas: list of metadata dicts
    """
    import chromadb

    persist_path = input_dir / CHROMA_PERSIST_DIR
    client = chromadb.PersistentClient(path=str(persist_path))
    collection = client.get_collection(CHROMA_CHUNK_COLLECTION)

    # Get everything
    result = collection.get(
        include=["embeddings", "metadatas"],
    )

    embeddings = np.array(result["embeddings"], dtype=np.float32)
    ids = result["ids"]
    metadatas = result["metadatas"]

    return embeddings, ids, metadatas


# ---------------------------------------------------------------------------
# Step 2: UMAP dimensionality reduction
# ---------------------------------------------------------------------------


def _reduce_umap(embeddings: np.ndarray) -> np.ndarray:
    """Reduce embedding dimensions via UMAP."""
    import umap

    n_samples = embeddings.shape[0]

    # Adjust n_neighbors if we have few samples
    n_neighbors = min(UMAP_N_NEIGHBORS, n_samples - 1)
    n_components = min(UMAP_N_COMPONENTS, n_samples - 1)

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=42,
    )

    reduced = reducer.fit_transform(embeddings)
    return reduced


# ---------------------------------------------------------------------------
# Step 3: HDBSCAN clustering
# ---------------------------------------------------------------------------


def _cluster_hdbscan(
    reduced: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
) -> tuple[np.ndarray, int, int]:
    """
    Run HDBSCAN on reduced embeddings.

    Returns:
        labels: array of cluster labels (-1 = noise)
        n_clusters: number of clusters found (excluding noise)
        n_noise: number of noise points
    """
    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )

    labels = clusterer.fit_predict(reduced)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))

    return labels, n_clusters, n_noise


# ---------------------------------------------------------------------------
# Step 4: GMM soft assignments
# ---------------------------------------------------------------------------


def _fit_gmm(
    reduced: np.ndarray,
    n_clusters: int,
    labels: np.ndarray,
) -> np.ndarray:
    """
    Fit a GMM initialized from HDBSCAN results to get soft cluster
    assignments (probability per cluster for each point).

    Returns:
        probabilities: array of shape (n_samples, n_clusters)
    """
    from sklearn.mixture import GaussianMixture

    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type=GMM_COVARIANCE_TYPE,
        random_state=42,
        max_iter=300,
        n_init=3,
    )

    gmm.fit(reduced)
    probabilities = gmm.predict_proba(reduced)

    return probabilities


# ---------------------------------------------------------------------------
# Step 5: Silhouette scoring
# ---------------------------------------------------------------------------


def _compute_silhouette(reduced: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute silhouette score for the clustering.
    Only considers non-noise points.  Returns -1 if not enough clusters.
    """
    from sklearn.metrics import silhouette_score

    # Filter out noise
    mask = labels >= 0
    if mask.sum() < 2:
        return -1.0

    unique_labels = set(labels[mask])
    if len(unique_labels) < 2:
        return -1.0

    return float(silhouette_score(reduced[mask], labels[mask]))


# ---------------------------------------------------------------------------
# Build assignments from GMM probabilities
# ---------------------------------------------------------------------------


def _build_assignments(
    chunk_ids: list[str],
    labels: np.ndarray,
    probabilities: np.ndarray | None,
) -> list[ClusterAssignment]:
    """
    Build ClusterAssignment objects from HDBSCAN labels and GMM probabilities.
    """
    assignments: list[ClusterAssignment] = []

    for i, chunk_id in enumerate(chunk_ids):
        is_noise = labels[i] == -1

        if probabilities is not None and not is_noise:
            probs = probabilities[i]
            sorted_indices = np.argsort(probs)[::-1]

            primary = int(sorted_indices[0])
            primary_conf = float(probs[sorted_indices[0]])

            # Secondary cluster if above threshold
            secondary = None
            secondary_conf = None
            if len(sorted_indices) > 1:
                sec_prob = float(probs[sorted_indices[1]])
                if sec_prob >= GMM_SECONDARY_THRESHOLD:
                    secondary = int(sorted_indices[1])
                    secondary_conf = sec_prob
        else:
            # Noise point or no GMM — use HDBSCAN label directly
            primary = int(labels[i])
            primary_conf = 0.0 if is_noise else 1.0
            secondary = None
            secondary_conf = None

        assignments.append(
            ClusterAssignment(
                chunk_id=chunk_id,
                primary_cluster=primary,
                confidence=primary_conf,
                secondary_cluster=secondary,
                secondary_confidence=secondary_conf,
                is_noise=is_noise,
            )
        )

    return assignments


# ---------------------------------------------------------------------------
# Main clustering pipeline with retry loop
# ---------------------------------------------------------------------------


def cluster_all(input_dir: Path) -> ClusteringResult:
    """
    Run the full clustering pipeline: UMAP → HDBSCAN → GMM → validate.

    If silhouette score is below threshold, adjusts min_cluster_size
    and retries up to CLUSTER_RETRY_MAX times.

    Args:
        input_dir: Root directory being scanned (contains .regenai_store/).

    Returns:
        ClusteringResult with assignments, scores, and metadata.
    """

    # -- Load embeddings --
    console.print("  [dim]Loading embeddings from ChromaDB...[/dim]")
    embeddings, chunk_ids, chunk_metadata = _load_from_chroma(input_dir)
    console.print(
        f"  [dim]Loaded {len(chunk_ids)} chunk embeddings ({embeddings.shape[1]} dims)[/dim]"
    )

    # -- UMAP reduction --
    console.print("  [dim]Reducing dimensions with UMAP...[/dim]")
    reduced = _reduce_umap(embeddings)
    console.print(f"  [dim]Reduced to {reduced.shape[1]} dimensions[/dim]")

    # -- Clustering with retry loop --
    best_result: dict | None = None
    best_silhouette: float = -1.0

    min_cluster_size = HDBSCAN_MIN_CLUSTER_SIZE
    min_samples = HDBSCAN_MIN_SAMPLES

    for attempt in range(1, CLUSTER_RETRY_MAX + 1):
        console.print(
            f"  [dim]Attempt {attempt}/{CLUSTER_RETRY_MAX}: "
            f"HDBSCAN(min_cluster_size={min_cluster_size}, "
            f"min_samples={min_samples})...[/dim]"
        )

        # HDBSCAN
        labels, n_clusters, n_noise = _cluster_hdbscan(
            reduced, min_cluster_size, min_samples
        )

        console.print(
            f"  [dim]  Found {n_clusters} clusters, {n_noise} noise points[/dim]"
        )

        # Need at least 2 clusters for meaningful results
        if n_clusters < 2:
            console.print("  [yellow]  Too few clusters, adjusting params...[/yellow]")
            min_cluster_size = max(3, min_cluster_size - 2)
            min_samples = max(2, min_samples - 1)
            continue

        # Silhouette score
        sil_score = _compute_silhouette(reduced, labels)
        console.print(f"  [dim]  Silhouette score: {sil_score:.3f}[/dim]")

        # Track best result
        if sil_score > best_silhouette:
            best_silhouette = sil_score
            best_result = {
                "labels": labels,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "silhouette": sil_score,
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
            }

        # Good enough — stop
        if sil_score >= SILHOUETTE_MIN_THRESHOLD:
            console.print(
                f"  [green]  Score above threshold ({SILHOUETTE_MIN_THRESHOLD}), accepting.[/green]"
            )
            break

        # Adjust for next attempt
        console.print(
            f"  [yellow]  Below threshold ({SILHOUETTE_MIN_THRESHOLD}), retrying...[/yellow]"
        )
        min_cluster_size = max(3, min_cluster_size - 2)
        min_samples = max(2, min_samples - 1)

    # Use best result found
    if best_result is None:
        # Absolute fallback — everything in one cluster
        console.print("  [red]Clustering failed, falling back to single cluster[/red]")
        labels = np.zeros(len(chunk_ids), dtype=int)
        n_clusters = 1
        n_noise = 0
        best_silhouette = 0.0
    else:
        labels = best_result["labels"]
        n_clusters = best_result["n_clusters"]
        n_noise = best_result["n_noise"]

    # -- GMM soft assignments --
    probabilities = None
    if n_clusters >= 2:
        console.print(f"  [dim]Fitting GMM with {n_clusters} components...[/dim]")
        try:
            # For GMM, reassign noise points temporarily to nearest cluster
            non_noise_mask = labels >= 0
            if non_noise_mask.any():
                probabilities = _fit_gmm(reduced, n_clusters, labels)
                console.print("  [green]GMM soft assignments computed[/green]")
        except Exception as e:
            console.print(
                f"  [yellow]GMM failed ({e}), using hard assignments[/yellow]"
            )

    # -- Build final assignments --
    assignments = _build_assignments(chunk_ids, labels, probabilities)

    result = ClusteringResult(
        assignments=assignments,
        n_clusters=n_clusters,
        n_noise=n_noise,
        silhouette_score=best_silhouette,
        reduced_embeddings=reduced,
        cluster_labels=labels.tolist(),
        chunk_ids=chunk_ids,
        chunk_metadata=chunk_metadata,
    )

    # -- Display summary --
    _print_cluster_summary(result)

    return result


# ---------------------------------------------------------------------------
# Display utilities
# ---------------------------------------------------------------------------


def _print_cluster_summary(result: ClusteringResult) -> None:
    """Print a summary of clustering results."""
    console.print()
    console.print(f"  [bold]Clustering complete:[/bold]")
    console.print(f"    Clusters found:   {result.n_clusters}")
    console.print(f"    Noise points:     {result.n_noise}")
    console.print(f"    Silhouette score: {result.silhouette_score:.3f}")
    console.print()

    # Per-cluster breakdown
    table = Table(title="Cluster Breakdown", show_lines=False)
    table.add_column("Cluster", justify="center")
    table.add_column("Chunks", justify="right")
    table.add_column("Files", justify="right")
    table.add_column("Top File Types", max_width=30)
    table.add_column("Project Roots", max_width=40)

    # Group by primary cluster
    cluster_data: dict[int, list[dict]] = {}
    for assignment, meta in zip(result.assignments, result.chunk_metadata):
        cid = assignment.primary_cluster
        if cid not in cluster_data:
            cluster_data[cid] = []
        cluster_data[cid].append(meta)

    for cid in sorted(cluster_data.keys()):
        metas = cluster_data[cid]
        chunk_count = len(metas)

        # Unique files
        files = set(m["source_file"] for m in metas)
        file_count = len(files)

        # Top file types
        type_counts = Counter(m["file_type"] for m in metas)
        top_types = ", ".join(f"{t}({c})" for t, c in type_counts.most_common(3))

        # Project roots
        roots = set(m["project_root"] for m in metas if m["project_root"])
        roots_str = ", ".join(sorted(roots)) if roots else "[dim]unanchored[/dim]"

        label = "noise" if cid == -1 else str(cid)
        table.add_row(label, str(chunk_count), str(file_count), top_types, roots_str)

    console.print(table)

    # Cross-cluster chunks (secondary assignment)
    cross_count = sum(1 for a in result.assignments if a.secondary_cluster is not None)
    if cross_count:
        console.print(
            f"\n  [dim]{cross_count} chunks have significant secondary "
            f"cluster membership (>{GMM_SECONDARY_THRESHOLD:.0%})[/dim]"
        )
    console.print()
