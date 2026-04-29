"""
RAPTOR — Module 7 of the ReGenAI pipeline.

2-level summarization using a local LLM (Ollama) or any
OpenAI-compatible API (Dassault DevAssistant, OpenRouter, etc.):

    Level 1 — Cluster summaries:
        For each cluster, gather chunk texts and summarize.
        Large clusters use map-reduce (sub-group → summarize → merge).

    Level 2 — Project summaries:
        Group cluster summaries by project root.
        Produce a unified project overview per project.
        Unanchored clusters get LLM-named topics (grouped by cluster).
"""

from dataclasses import dataclass, field
from collections import defaultdict

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from regenai.config import (
    CHROMA_CHUNK_COLLECTION,
    CHROMA_PERSIST_DIR,
    CHUNK_MAX_TOKENS,
    CHARS_PER_TOKEN,
    OLLAMA_BASE_URL,
    OPENROUTER_BASE_URL,
)
from regenai.refiner import RefinedResult

console = Console()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ClusterSummary:
    """Level 1: summary of a single cluster."""

    cluster_id: int
    summary: str
    source_files: list[str]
    project_roots: list[str]
    unanchored_files: list[str]
    chunk_count: int
    composition: str


@dataclass
class ProjectSummary:
    """Level 2: unified summary for a project or topic."""

    project_name: str
    project_root: str | None
    cluster_summaries: list[str]
    project_summary: str
    source_files: list[str]
    avg_confidence: float


# ---------------------------------------------------------------------------
# LLM wrapper — Ollama (local) or any OpenAI-compatible API
# ---------------------------------------------------------------------------


def _is_api_model(model: str) -> bool:
    """Models with '/' are API models, otherwise Ollama."""
    return "/" in model


def _call_llm(prompt: str, model: str, max_retries: int = 3) -> str:
    if _is_api_model(model):
        return _call_openai_compatible(prompt, model, max_retries)
    else:
        return _call_ollama(prompt, model, max_retries)


def _call_openai_compatible(prompt: str, model: str, max_retries: int = 5) -> str:
    """
    Call any OpenAI-compatible API using httpx directly.
    Uses LLM_BASE_URL + LLM_API_KEY from .env if set,
    falls back to OpenRouter with OPENROUTER_API_KEY.
    """
    import os
    import time
    import json
    import httpx

    base_url = os.environ.get("BASE_URL", "")
    api_key = os.environ.get("API_KEY", "")

    if not base_url:
        base_url = OPENROUTER_BASE_URL
        api_key = os.environ.get("OPENROUTER_API_KEY", "")

    if not api_key:
        console.print(
            "  [red]No API key! Add BASE_URL + API_KEY or OPENROUTER_API_KEY to .env[/red]"
        )
        return "[Summary generation failed: no API key]"

    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 2048,
        "stream": False,
    }

    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(url, headers=headers, json=payload)

            if response.status_code == 400:
                console.print(f"  [red]Invalid model ID: {model}[/red]")
                return "[Summary generation failed: invalid model]"
            elif response.status_code == 429:
                wait = 15 * (attempt + 1)
                console.print(f"  [yellow]Rate limited, waiting {wait}s...[/yellow]")
                time.sleep(wait)
                continue

            response.raise_for_status()
            data = response.json()

            # Extract content from OpenAI-compatible response
            content = data["choices"][0]["message"]["content"]
            return content.strip()

        except (httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException) as e:
            if attempt == max_retries - 1:
                console.print(f"  [red]API error: {e}[/red]")
                return f"[Summary generation failed: {e}]"
            console.print(f"  [yellow]Retry {attempt + 1}: {e}[/yellow]")
            time.sleep(5)
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            console.print(f"  [red]Failed to parse API response: {e}[/red]")
            return f"[Summary generation failed: {e}]"

    return "[Summary generation failed]"


def _call_ollama(prompt: str, model: str, max_retries: int = 3) -> str:
    """Call local Ollama server."""
    import ollama

    for attempt in range(max_retries):
        try:
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3, "num_ctx": 4096},
            )
            return response["message"]["content"].strip()
        except Exception as e:
            if attempt == max_retries - 1:
                console.print(f"  [red]Ollama error: {e}[/red]")
                return f"[Summary generation failed: {e}]"
            console.print(f"  [yellow]Ollama retry {attempt + 1}: {e}[/yellow]")

    return "[Summary generation failed]"


# ---------------------------------------------------------------------------
# Load chunk texts from ChromaDB
# ---------------------------------------------------------------------------


def _load_chunk_texts(input_dir) -> dict[str, str]:
    import chromadb
    from pathlib import Path

    persist_path = Path(input_dir) / CHROMA_PERSIST_DIR
    client = chromadb.PersistentClient(path=str(persist_path))
    collection = client.get_collection(CHROMA_CHUNK_COLLECTION)

    result = collection.get(include=["documents"])
    return dict(zip(result["ids"], result["documents"]))


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------


def _cluster_prompt(chunks_text: str, composition: str) -> str:
    if composition == "code-heavy":
        instruction = (
            "Summarize the following code and documentation. Focus on:\n"
            "- What the code does (purpose and functionality)\n"
            "- Key components, functions, and classes\n"
            "- Dependencies and technology stack\n"
            "- Architecture patterns\n"
            "Be specific — mention actual function names, file names, and technical details."
        )
    elif composition == "document-heavy":
        instruction = (
            "Summarize the following documents. Focus on:\n"
            "- Key topics and themes\n"
            "- Important decisions, requirements, or deliverables\n"
            "- People, dates, and deadlines mentioned\n"
            "- Relationships between documents\n"
            "Be specific — mention actual names, dates, and concrete details."
        )
    else:
        instruction = (
            "Summarize the following collection of code and documents. Focus on:\n"
            "- The project's purpose and goals\n"
            "- Key code components and what they do\n"
            "- Important documentation points\n"
            "- How the code and documents relate to each other\n"
            "Be specific — mention actual names, files, and technical details."
        )

    return f"{instruction}\n\n---\n\n{chunks_text}"


def _project_prompt(cluster_summaries: str, project_name: str) -> str:
    return (
        f"The following are summaries of different aspects of the project '{project_name}'. "
        f"Combine them into a single comprehensive project overview. Include:\n"
        f"- Overall purpose and goals of the project\n"
        f"- Key components and how they fit together\n"
        f"- Technology stack and architecture\n"
        f"- Important details, decisions, and deliverables\n"
        f"Write a well-structured summary with clear sections.\n\n"
        f"---\n\n{cluster_summaries}"
    )


def _topic_naming_prompt(summary: str) -> str:
    return (
        "Based on this summary, give a short descriptive name (2-5 words) for this "
        "collection of files. Respond with ONLY the name, nothing else.\n\n"
        f"{summary}"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def _get_composition(metadata_list: list[dict]) -> str:
    type_counts = defaultdict(int)
    for m in metadata_list:
        type_counts[m.get("file_type", "")] += 1

    code = type_counts.get("code", 0) + type_counts.get("config", 0)
    docs = type_counts.get("document", 0)
    total = code + docs

    if total == 0:
        return "mixed"
    if code / total >= 0.6:
        return "code-heavy"
    if docs / total >= 0.6:
        return "document-heavy"
    return "mixed"


# ---------------------------------------------------------------------------
# Level 1: Cluster summaries
# ---------------------------------------------------------------------------


def _summarize_cluster(
    cluster_id: int,
    chunk_ids: list[str],
    chunk_texts: dict[str, str],
    metadata_list: list[dict],
    model: str,
) -> ClusterSummary:
    """
    Summarize a single cluster. Uses map-reduce for large clusters
    that exceed the context window.
    """
    composition = _get_composition(metadata_list)

    texts = [chunk_texts.get(cid, "") for cid in chunk_ids]
    texts = [t for t in texts if t.strip()]

    source_files = sorted(set(m["source_file"] for m in metadata_list))
    project_roots = sorted(
        set(m["project_root"] for m in metadata_list if m["project_root"])
    )
    unanchored = sorted(
        set(m["source_file"] for m in metadata_list if not m["project_root"])
    )

    max_input_tokens = 3000
    combined = "\n\n---\n\n".join(texts)

    if _estimate_tokens(combined) <= max_input_tokens:
        prompt = _cluster_prompt(combined, composition)
        summary = _call_llm(prompt, model)
    else:
        # Map-reduce
        groups: list[str] = []
        current_group: list[str] = []
        current_tokens = 0

        for text in texts:
            t_tokens = _estimate_tokens(text)
            if current_tokens + t_tokens > max_input_tokens and current_group:
                groups.append("\n\n---\n\n".join(current_group))
                current_group = []
                current_tokens = 0
            current_group.append(text)
            current_tokens += t_tokens

        if current_group:
            groups.append("\n\n---\n\n".join(current_group))

        # Map
        sub_summaries: list[str] = []
        for group_text in groups:
            prompt = _cluster_prompt(group_text, composition)
            sub_summary = _call_llm(prompt, model)
            sub_summaries.append(sub_summary)

        # Reduce
        merged = "\n\n---\n\n".join(sub_summaries)
        reduce_prompt = (
            "The following are partial summaries of the same project component. "
            "Merge them into a single cohesive summary, removing redundancy "
            "and preserving all important details.\n\n"
            f"---\n\n{merged}"
        )
        summary = _call_llm(reduce_prompt, model)

    return ClusterSummary(
        cluster_id=cluster_id,
        summary=summary,
        source_files=source_files,
        project_roots=project_roots,
        unanchored_files=unanchored,
        chunk_count=len(chunk_ids),
        composition=composition,
    )


# ---------------------------------------------------------------------------
# Level 2: Project summaries
# ---------------------------------------------------------------------------


def _build_project_summaries(
    cluster_summaries: list[ClusterSummary],
    refined: RefinedResult,
    model: str,
) -> list[ProjectSummary]:
    """
    Group cluster summaries by project root → unified project summaries.
    Unanchored clusters grouped by cluster ID → LLM-named topics.
    """
    # Group clusters by project root
    project_clusters: dict[str, list[ClusterSummary]] = defaultdict(list)
    # Track unanchored clusters separately, preserving cluster identity
    unanchored_clusters: dict[int, ClusterSummary] = {}

    for cs in cluster_summaries:
        if cs.project_roots:
            for root in cs.project_roots:
                project_clusters[root].append(cs)
        if cs.unanchored_files:
            unanchored_clusters[cs.cluster_id] = cs

    # Avg confidence per project
    assignment_map: dict[str, list[float]] = defaultdict(list)
    for a, m in zip(refined.assignments, refined.chunk_metadata):
        root = m.get("project_root", "")
        if root:
            assignment_map[root].append(a.confidence)

    results: list[ProjectSummary] = []

    # --- Named project summaries ---
    for root, summaries in sorted(project_clusters.items()):
        all_files = sorted(
            set(
                f
                for cs in summaries
                for f in cs.source_files
                if any(
                    m.get("project_root") == root
                    for m in refined.chunk_metadata
                    if m["source_file"] == f
                )
            )
        )

        cluster_texts = [cs.summary for cs in summaries]
        confs = assignment_map.get(root, [1.0])
        avg_conf = sum(confs) / max(len(confs), 1)

        if len(cluster_texts) == 1:
            project_summary = cluster_texts[0]
        else:
            combined = "\n\n---\n\n".join(
                f"[Aspect {i+1}]:\n{s}" for i, s in enumerate(cluster_texts)
            )
            prompt = _project_prompt(combined, root)
            project_summary = _call_llm(prompt, model)

        results.append(
            ProjectSummary(
                project_name=root,
                project_root=root,
                cluster_summaries=cluster_texts,
                project_summary=project_summary,
                source_files=all_files,
                avg_confidence=avg_conf,
            )
        )

    # --- Unanchored topic summaries (one per cluster) ---
    for cid, cs in sorted(unanchored_clusters.items()):
        # Name the topic
        topic_name = _call_llm(_topic_naming_prompt(cs.summary), model)
        topic_name = topic_name.strip().strip('"').strip("'").strip("#").strip("*")

        # Confidence for unanchored chunks in this cluster
        unanchored_confs = [
            a.confidence
            for a, m in zip(refined.assignments, refined.chunk_metadata)
            if not m.get("project_root") and a.primary_cluster == cid
        ]
        avg_conf = (
            sum(unanchored_confs) / max(len(unanchored_confs), 1)
            if unanchored_confs
            else 0.5
        )

        results.append(
            ProjectSummary(
                project_name=topic_name,
                project_root=None,
                cluster_summaries=[cs.summary],
                project_summary=cs.summary,
                source_files=cs.unanchored_files,
                avg_confidence=avg_conf,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Main RAPTOR entrypoint
# ---------------------------------------------------------------------------


def raptor_summarize(
    refined: RefinedResult,
    input_dir,
    model: str,
) -> list[ProjectSummary]:
    """
    Run the 2-level RAPTOR summarization pipeline.

    Level 1: Summarize each cluster via LLM (with map-reduce for large clusters).
    Level 2: Group by project root and produce unified project summaries.
             Unanchored clusters get their own LLM-named topics.
    """
    from pathlib import Path

    input_dir = Path(input_dir)

    # Load chunk texts
    console.print("  [dim]Loading chunk texts from ChromaDB...[/dim]")
    chunk_texts = _load_chunk_texts(input_dir)

    # Group chunks by cluster
    cluster_chunks: dict[int, tuple[list[str], list[dict]]] = defaultdict(
        lambda: ([], [])
    )
    for assignment, meta in zip(refined.assignments, refined.chunk_metadata):
        if assignment.is_noise:
            continue
        cid = assignment.primary_cluster
        ids_list, meta_list = cluster_chunks[cid]
        ids_list.append(assignment.chunk_id)
        meta_list.append(meta)

    # -- Level 1: Cluster summaries --
    console.print(f"  [dim]Summarizing {len(cluster_chunks)} clusters...[/dim]")
    cluster_summaries: list[ClusterSummary] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Cluster summaries", total=len(cluster_chunks))

        for cid in sorted(cluster_chunks.keys()):
            chunk_ids, metadata_list = cluster_chunks[cid]
            cs = _summarize_cluster(cid, chunk_ids, chunk_texts, metadata_list, model)
            cluster_summaries.append(cs)
            progress.advance(task)

    console.print(
        f"  [green]Generated {len(cluster_summaries)} cluster summaries[/green]"
    )

    # -- Level 2: Project summaries --
    console.print("  [dim]Building project summaries...[/dim]")
    project_summaries = _build_project_summaries(cluster_summaries, refined, model)

    console.print(
        f"  [green]Generated {len(project_summaries)} project summaries[/green]"
    )
    console.print()

    for ps in project_summaries:
        console.print(
            f"    📁 {ps.project_name} "
            f"({len(ps.source_files)} files, conf={ps.avg_confidence:.2f})"
        )

    console.print()

    return project_summaries
