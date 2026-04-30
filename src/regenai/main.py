"""
CLI entrypoint for ReGenAI.

Handles argument parsing, interactive prompts for directory paths,
and orchestrates the pipeline modules in sequence.
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from rich.console import Console
from rich.prompt import Prompt, Confirm

from regenai.config import DEFAULT_LLM_MODEL, DEFAULT_OUTPUT_DIR
from regenai.registry import (
    build_registry,
    print_registry_summary,
    print_registry_detail,
)
from regenai.parser import parse_all
from regenai.chunker import chunk_all
from regenai.embedder import embed_all
from regenai.clusterer import cluster_all
from regenai.refiner import refine_clusters
from regenai.raptor import raptor_summarize
from regenai.output import write_output

console = Console()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="regenai",
        description="ReGenAI — Crawl, cluster, and summarize project directories using local AI.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_LLM_MODEL,
        help=f"Ollama model to use for summarization (default: {DEFAULT_LLM_MODEL})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan directory and show file registry only — no processing",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging and show file-level registry table",
    )
    return parser


def _prompt_directory(label: str, default: str | None = None) -> Path:
    """Prompt the user for a directory path, validate it exists."""
    while True:
        if default:
            raw = Prompt.ask(f"  {label}", default=default)
        else:
            raw = Prompt.ask(f"  {label}")

        path = Path(raw).expanduser().resolve()

        if "output" in label.lower():
            # Output directory will be created if it doesn't exist
            return path

        if not path.exists():
            console.print(f"  [red]Path does not exist:[/red] {path}")
            continue
        if not path.is_dir():
            console.print(f"  [red]Not a directory:[/red] {path}")
            continue
        return path


def cli() -> None:
    """Main CLI entrypoint."""
    parser = _build_parser()
    args = parser.parse_args()

    # -- Banner --
    console.print()
    console.print("[bold cyan]  ReGenAI[/bold cyan] — Project Directory Summarizer")
    console.print("  ─────────────────────────────────────────")
    console.print()

    # -- Interactive directory prompts --
    input_dir = _prompt_directory("Enter input directory path")
    output_dir = _prompt_directory(
        "Enter output directory path", default=DEFAULT_OUTPUT_DIR
    )

    console.print()
    console.print(f"  [dim]Model:[/dim]  {args.model}")
    console.print()

    # -- Module 1: File Registry --
    with console.status("[bold green]Scanning directory..."):
        registry = build_registry(input_dir)

    print_registry_summary(registry)

    if args.verbose or args.dry_run:
        print_registry_detail(registry)

    if len(registry.entries) == 0:
        console.print(
            "[yellow]No processable files found. Check your directory and ignore patterns.[/yellow]"
        )
        sys.exit(0)

    if args.dry_run:
        console.print("[dim]Dry run complete. No processing performed.[/dim]")
        sys.exit(0)

    # -- Confirmation before heavy processing --
    proceed = Confirm.ask(
        f"  Proceed with {len(registry.entries)} files?",
        default=True,
    )
    if not proceed:
        console.print("[dim]Aborted.[/dim]")
        sys.exit(0)

    # -- Module 2: Parse all files --
    console.print("[bold]Parsing files...[/bold]")
    parsed_files = parse_all(registry.entries)

    successful = [p for p in parsed_files if p.parse_success]
    console.print(f"  [dim]{len(successful)} files ready for chunking[/dim]")
    console.print()

    # -- Module 3: Chunk all parsed files --
    console.print("[bold]Chunking files...[/bold]")
    chunks = chunk_all(parsed_files)

    # -- TEMP: Dump chunks to JSON for inspection (safe to delete) --
    import json

    chunks_dump = [
        {
            "id": c.id,
            "source_file": str(c.source_file),
            "file_type": c.file_type,
            "language": c.language,
            "chunk_index": c.chunk_index,
            "total_chunks": c.total_chunks,
            "project_root": c.project_root,
            "is_support_file": c.is_support_file,
            "name_prefix": c.name_prefix,
            "token_estimate": len(c.text) // 4,
            "text": c.text,
        }
        for c in chunks
    ]
    dump_path = Path("_chunks_debug.json")
    dump_path.write_text(
        json.dumps(chunks_dump, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    console.print(
        f"  [dim]Chunks dumped to {dump_path.resolve()} (safe to delete)[/dim]"
    )
    # -- END TEMP --

    # -- Module 4: Embed chunks and store in ChromaDB --
    console.print("[bold]Embedding chunks...[/bold]")
    persist_path = embed_all(chunks, input_dir)

    # -- Module 5: Cluster embeddings --
    console.print("[bold]Clustering...[/bold]")
    cluster_result = cluster_all(input_dir)

    # -- Module 6: Refine clusters using structural signals --
    console.print("[bold]Refining clusters...[/bold]")
    refined_result = refine_clusters(cluster_result)

    # -- TEMP: Dump cluster contents to JSON for inspection (safe to delete) --
    import json

    chunk_text_map = {c.id: c.text for c in chunks}
    cluster_dump: dict[str, list] = {}
    for assignment, meta in zip(
        refined_result.assignments, refined_result.chunk_metadata
    ):
        cid = "noise" if assignment.is_noise else str(assignment.primary_cluster)
        if cid not in cluster_dump:
            cluster_dump[cid] = []
        cluster_dump[cid].append(
            {
                "chunk_id": assignment.chunk_id,
                "source_file": meta["source_file"],
                "file_type": meta["file_type"],
                "project_root": meta["project_root"],
                "confidence": round(assignment.confidence, 3),
                "text_preview": chunk_text_map.get(assignment.chunk_id, "")[:200],
            }
        )
    dump_path = Path("_clusters_debug.json")
    dump_path.write_text(
        json.dumps(cluster_dump, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    console.print(
        f"  [dim]Clusters dumped to {dump_path.resolve()} (safe to delete)[/dim]"
    )
    # -- END TEMP --

    # -- Module 7: RAPTOR summarization --
    console.print(f"[bold]Generating summaries via {args.model}...[/bold]")
    project_summaries = raptor_summarize(refined_result, input_dir, args.model)

    # -- Module 8: Write output markdown files --
    from datetime import datetime

    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = output_dir / run_timestamp

    console.print("[bold]Writing output files...[/bold]")
    write_output(project_summaries, run_dir)

    console.print("[bold green]Pipeline complete![/bold green]")
    console.print(f"  [dim]Output directory: {run_dir.resolve()}[/dim]")


if __name__ == "__main__":
    cli()
