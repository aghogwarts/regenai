"""
Embedder — Module 4 of the ReGenAI pipeline.

Converts chunks into vector embeddings using BAAI/bge-m3 and stores
them in ChromaDB with full provenance metadata.

Two collections are created:
    regenai_chunks  — one entry per chunk (for fine-grained clustering)
    regenai_files   — one entry per file (mean-pooled chunk embeddings)
"""

import numpy as np
from pathlib import Path

from rich.console import Console

from regenai.config import (
    EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE,
    CHROMA_CHUNK_COLLECTION,
    CHROMA_FILE_COLLECTION,
    CHROMA_PERSIST_DIR,
)
from regenai.chunker import Chunk

console = Console()


# ---------------------------------------------------------------------------
# Embedding model loader
# ---------------------------------------------------------------------------

_model = None


def _load_model():
    """
    Load bge-m3 via sentence-transformers.
    Tries GPU first, falls back to CPU on OOM or CUDA errors.
    """
    global _model
    if _model is not None:
        return _model

    from sentence_transformers import SentenceTransformer

    try:
        console.print(f"  [dim]Loading {EMBEDDING_MODEL} on GPU...[/dim]")
        _model = SentenceTransformer(EMBEDDING_MODEL, device="cuda")
        # Quick test to catch OOM early
        _model.encode(["test"], show_progress_bar=False)
        console.print("  [green]Model loaded on GPU[/green]")
    except Exception as e:
        console.print(f"  [yellow]GPU failed ({e}), falling back to CPU...[/yellow]")
        _model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
        console.print("  [green]Model loaded on CPU[/green]")

    return _model


# ---------------------------------------------------------------------------
# ChromaDB setup
# ---------------------------------------------------------------------------


def _get_chroma_client(input_dir: Path):
    """Create a persistent ChromaDB client in the input directory."""
    import chromadb

    persist_path = input_dir / CHROMA_PERSIST_DIR
    persist_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(persist_path))
    return client


def _reset_collections(client):
    """Delete and recreate both collections for a clean run."""
    for name in (CHROMA_CHUNK_COLLECTION, CHROMA_FILE_COLLECTION):
        try:
            client.delete_collection(name)
        except Exception:
            pass

    chunk_col = client.create_collection(
        name=CHROMA_CHUNK_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    file_col = client.create_collection(
        name=CHROMA_FILE_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    return chunk_col, file_col


# ---------------------------------------------------------------------------
# Batch embedding
# ---------------------------------------------------------------------------


def _embed_texts(texts: list[str], model) -> np.ndarray:
    """Embed a list of texts in batches, returns numpy array of embeddings."""
    all_embeddings = []

    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i : i + EMBEDDING_BATCH_SIZE]
        embeddings = model.encode(
            batch,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        all_embeddings.append(embeddings)

    return np.vstack(all_embeddings)


# ---------------------------------------------------------------------------
# Main embedder
# ---------------------------------------------------------------------------


def embed_all(chunks: list[Chunk], input_dir: Path) -> Path:
    """
    Embed all chunks and store in ChromaDB.

    Steps:
    1. Load bge-m3 model
    2. Embed all chunk texts in batches
    3. Store chunk embeddings + metadata in ChromaDB
    4. Compute file-level embeddings (mean-pool) and store separately

    Args:
        chunks: List of Chunk objects from the chunker.
        input_dir: Root directory being scanned (ChromaDB lives here).

    Returns:
        Path to the ChromaDB persist directory.
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

    if not chunks:
        console.print("  [yellow]No chunks to embed.[/yellow]")
        return input_dir / CHROMA_PERSIST_DIR

    # -- Load model --
    model = _load_model()
    console.print()

    # -- Embed all chunks --
    console.print("  [dim]Embedding chunks...[/dim]")
    texts = [c.text for c in chunks]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Embedding", total=len(texts))

        all_embeddings: list[np.ndarray] = []
        for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[i : i + EMBEDDING_BATCH_SIZE]
            embs = model.encode(
                batch,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            all_embeddings.append(embs)
            progress.advance(task, len(batch))

    embeddings = np.vstack(all_embeddings)
    console.print(
        f"  [green]Embedded {len(chunks)} chunks[/green] "
        f"[dim]({embeddings.shape[1]} dimensions)[/dim]"
    )

    # -- Store in ChromaDB --
    console.print("  [dim]Storing in ChromaDB...[/dim]")
    client = _get_chroma_client(input_dir)
    chunk_col, file_col = _reset_collections(client)

    # ChromaDB add — in batches (ChromaDB has a limit of ~41666 per add)
    batch_size = 5000
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_embs = embeddings[i : i + batch_size]

        chunk_col.add(
            ids=[c.id for c in batch_chunks],
            embeddings=batch_embs.tolist(),
            documents=[c.text for c in batch_chunks],
            metadatas=[
                {
                    "source_file": str(c.source_file),
                    "file_type": c.file_type,
                    "language": c.language or "",
                    "chunk_index": c.chunk_index,
                    "total_chunks": c.total_chunks,
                    "project_root": c.project_root or "",
                    "is_support_file": c.is_support_file,
                    "name_prefix": c.name_prefix or "",
                }
                for c in batch_chunks
            ],
        )

    console.print(
        f"  [green]Stored {chunk_col.count()} chunks in "
        f"'{CHROMA_CHUNK_COLLECTION}'[/green]"
    )

    # -- File-level embeddings (mean-pool) --
    file_embeddings: dict[str, list[np.ndarray]] = {}
    file_metadata: dict[str, dict] = {}

    for chunk, emb in zip(chunks, embeddings):
        key = str(chunk.source_file)
        if key not in file_embeddings:
            file_embeddings[key] = []
            file_metadata[key] = {
                "source_file": key,
                "file_type": chunk.file_type,
                "language": chunk.language or "",
                "project_root": chunk.project_root or "",
                "is_support_file": chunk.is_support_file,
                "name_prefix": chunk.name_prefix or "",
                "chunk_count": chunk.total_chunks,
            }
        file_embeddings[key].append(emb)

    # Mean-pool and store
    file_ids: list[str] = []
    file_embs: list[list[float]] = []
    file_metas: list[dict] = []

    for file_path, emb_list in file_embeddings.items():
        pooled = np.mean(emb_list, axis=0)
        pooled = pooled / np.linalg.norm(pooled)  # re-normalize after pooling

        file_ids.append(file_path)
        file_embs.append(pooled.tolist())
        file_metas.append(file_metadata[file_path])

    if file_ids:
        file_col.add(
            ids=file_ids,
            embeddings=file_embs,
            metadatas=file_metas,
        )

    console.print(
        f"  [green]Stored {file_col.count()} file embeddings in "
        f"'{CHROMA_FILE_COLLECTION}'[/green]"
    )

    persist_path = input_dir / CHROMA_PERSIST_DIR
    console.print(f"  [dim]ChromaDB persisted at {persist_path}[/dim]")
    console.print()

    return persist_path
