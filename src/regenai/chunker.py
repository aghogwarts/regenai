"""
Chunker — Module 3 of the ReGenAI pipeline.

Splits parsed files into semantically meaningful chunks with provenance
metadata.  Uses different strategies based on file type:

    Code files    → tree-sitter AST (function/class boundaries), regex fallback
    Sectioned docs → split on parser-detected sections, sub-split if too large
    Plain text     → paragraph-boundary splitting with overlap
    Support files  → embed whole (small config/infra files)
"""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from regenai.config import (
    CHUNK_MAX_TOKENS,
    CHUNK_MIN_SPLIT_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    CHARS_PER_TOKEN,
    LANGUAGE_MAP,
)
from regenai.parser import ParsedFile

console = Console()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """A single chunk of text with full provenance metadata."""

    id: str  # unique ID: hash of source path + chunk index
    text: str
    source_file: Path  # relative path to source
    file_type: str  # "code", "document", "config", etc.
    language: str | None  # for code: "python", "javascript", etc.
    chunk_index: int  # 0-based index within this file
    total_chunks: int  # total chunks produced from this file
    project_root: str | None  # inherited from FileEntry
    is_support_file: bool  # inherited from FileEntry
    name_prefix: str | None  # inherited from FileEntry


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


def _estimate_tokens(text: str) -> int:
    """Rough token count — chars / 4."""
    return len(text) // CHARS_PER_TOKEN


# ---------------------------------------------------------------------------
# Chunk ID generation
# ---------------------------------------------------------------------------


def _make_chunk_id(source_path: Path, chunk_index: int) -> str:
    """Generate a deterministic unique ID for a chunk."""
    raw = f"{source_path}::{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Text splitting utilities
# ---------------------------------------------------------------------------


def _split_at_paragraphs(text: str) -> list[str]:
    """Split text at double-newline paragraph boundaries."""
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def _split_at_sentences(text: str) -> list[str]:
    """Split text at sentence boundaries (period/question/exclamation + space)."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def _merge_small_pieces(
    pieces: list[str],
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """
    Merge small text pieces into chunks up to max_tokens.
    Adds overlap between consecutive chunks for context continuity.
    """
    if not pieces:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    overlap_chars = overlap_tokens * CHARS_PER_TOKEN

    for piece in pieces:
        piece_tokens = _estimate_tokens(piece)

        # If a single piece exceeds max, it becomes its own chunk
        if piece_tokens > max_tokens:
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_tokens = 0
            chunks.append(piece)
            continue

        # Would adding this piece exceed the limit?
        if current_tokens + piece_tokens > max_tokens and current:
            chunk_text = "\n\n".join(current)
            chunks.append(chunk_text)

            # Build overlap from end of current chunk
            if overlap_chars > 0:
                overlap_text = chunk_text[-overlap_chars:]
                current = [overlap_text]
                current_tokens = _estimate_tokens(overlap_text)
            else:
                current = []
                current_tokens = 0

        current.append(piece)
        current_tokens += piece_tokens

    if current:
        chunks.append("\n\n".join(current))

    return chunks


# ---------------------------------------------------------------------------
# Code chunking — tree-sitter with regex fallback
# ---------------------------------------------------------------------------


def _chunk_code_treesitter(text: str, language: str) -> list[str] | None:
    """
    Attempt to chunk code using tree-sitter AST parsing.
    Returns list of chunks (one per top-level function/class), or None if
    tree-sitter isn't available or fails.
    """
    try:
        from tree_sitter_languages import get_parser

        parser = get_parser(language)
        tree = parser.parse(text.encode("utf-8"))
        root = tree.root_node

        # Node types that represent top-level blocks we want to split on
        # Varies by language but these cover the common ones
        block_types = {
            "function_definition",  # Python
            "class_definition",  # Python
            "function_declaration",  # JS, C, Go
            "class_declaration",  # JS, Java
            "method_definition",  # JS class methods
            "arrow_function",  # JS
            "impl_item",  # Rust
            "function_item",  # Rust
            "struct_item",  # Rust
            "method_declaration",  # Java
            "interface_declaration",  # Java, TS
            "type_alias_declaration",  # TS
            "export_statement",  # JS/TS
        }

        # Collect top-level block boundaries
        blocks: list[tuple[int, int]] = []
        for child in root.children:
            if child.type in block_types:
                blocks.append((child.start_byte, child.end_byte))
            # Handle exported functions: export default function ...
            elif child.type == "export_statement":
                for grandchild in child.children:
                    if grandchild.type in block_types:
                        blocks.append((child.start_byte, child.end_byte))
                        break

        if not blocks:
            return None  # no recognizable blocks, fall back to regex

        # Build chunks: include any code between blocks (imports, globals)
        # as part of the first chunk or as a preamble chunk
        text_bytes = text.encode("utf-8")
        chunks: list[str] = []

        # Preamble: everything before the first block (imports, constants)
        preamble = text_bytes[: blocks[0][0]].decode("utf-8").strip()
        if preamble and _estimate_tokens(preamble) > 20:
            chunks.append(preamble)

        # Each block
        for start, end in blocks:
            block_text = text_bytes[start:end].decode("utf-8").strip()
            if block_text:
                chunks.append(block_text)

        return chunks if chunks else None

    except Exception:
        return None  # fall back to regex


# Regex patterns for function/class detection per language family
_CODE_BLOCK_PATTERNS: dict[str, re.Pattern] = {
    "python": re.compile(r"^(?:class\s|def\s|async\s+def\s)", re.MULTILINE),
    "javascript": re.compile(
        r"^(?:function\s|class\s|const\s+\w+\s*=\s*(?:async\s*)?\(|export\s+(?:default\s+)?(?:function|class)\s)",
        re.MULTILINE,
    ),
    "typescript": re.compile(
        r"^(?:function\s|class\s|const\s+\w+\s*=\s*(?:async\s*)?\(|export\s+(?:default\s+)?(?:function|class|interface|type)\s|interface\s)",
        re.MULTILINE,
    ),
    "java": re.compile(
        r"^(?:\s*(?:public|private|protected|static)\s+.*(?:class|void|int|String|boolean|interface)\s)",
        re.MULTILINE,
    ),
}


def _chunk_code_regex(text: str, language: str | None) -> list[str]:
    """
    Fallback code chunking using regex to detect function/class boundaries.
    """
    # Try language-specific pattern
    pattern = None
    if language:
        pattern = _CODE_BLOCK_PATTERNS.get(language)
        # Try language family
        if not pattern and language in ("typescript",):
            pattern = _CODE_BLOCK_PATTERNS.get("javascript")

    if not pattern:
        # Generic: split on lines that start with common keywords
        pattern = re.compile(
            r"^(?:def\s|class\s|function\s|public\s|private\s|func\s|fn\s)",
            re.MULTILINE,
        )

    # Find all match positions
    matches = list(pattern.finditer(text))

    if len(matches) < 2:
        # Not enough structure to split meaningfully
        return [text]

    chunks: list[str] = []
    lines = text.split("\n")

    # Preamble
    first_match_line = text[: matches[0].start()].count("\n")
    if first_match_line > 0:
        preamble = "\n".join(lines[:first_match_line]).strip()
        if preamble and _estimate_tokens(preamble) > 20:
            chunks.append(preamble)

    # Split at each match
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end].strip()
        if block:
            chunks.append(block)

    return chunks if chunks else [text]


def _chunk_code(text: str, language: str | None) -> list[str]:
    """
    Chunk code file: try tree-sitter first, regex fallback.
    Files under CHUNK_MIN_SPLIT_TOKENS are kept whole.
    """
    if _estimate_tokens(text) <= CHUNK_MIN_SPLIT_TOKENS:
        return [text]

    # Try tree-sitter if we have a language
    if language:
        ts_chunks = _chunk_code_treesitter(text, language)
        if ts_chunks:
            return ts_chunks

    # Regex fallback
    return _chunk_code_regex(text, language)


# ---------------------------------------------------------------------------
# Document chunking
# ---------------------------------------------------------------------------


def _chunk_with_sections(sections: list[str]) -> list[str]:
    """
    Chunk a document that has pre-detected sections (slides, pages, sheets).
    Sections that are too large get sub-split at paragraph boundaries.
    """
    chunks: list[str] = []

    for section in sections:
        if _estimate_tokens(section) <= CHUNK_MAX_TOKENS:
            chunks.append(section)
        else:
            # Sub-split large sections at paragraph boundaries
            paragraphs = _split_at_paragraphs(section)
            sub_chunks = _merge_small_pieces(
                paragraphs, CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS
            )
            chunks.extend(sub_chunks)

    return chunks


def _chunk_plain_text(text: str) -> list[str]:
    """
    Chunk plain text / unsectioned documents.
    Split at paragraphs, merge small ones, add overlap.
    """
    if _estimate_tokens(text) <= CHUNK_MIN_SPLIT_TOKENS:
        return [text]

    paragraphs = _split_at_paragraphs(text)

    # If we only got one big paragraph, try sentence splitting
    if len(paragraphs) <= 1:
        sentences = _split_at_sentences(text)
        if len(sentences) > 1:
            return _merge_small_pieces(
                sentences, CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS
            )
        # Can't split meaningfully, return whole
        return [text]

    return _merge_small_pieces(paragraphs, CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS)


# ---------------------------------------------------------------------------
# Main chunker
# ---------------------------------------------------------------------------


def chunk_file(parsed: ParsedFile) -> list[Chunk]:
    """
    Chunk a single parsed file using the appropriate strategy.

    Strategy selection:
        Support/config files (small) → embed whole
        Code files                   → AST-based / regex splitting
        Documents with sections      → section-based splitting
        Documents without sections   → paragraph-based splitting
    """
    entry = parsed.entry

    # Skip files that failed to parse
    if not parsed.parse_success or not parsed.raw_text.strip():
        return []

    # Support files and config: embed whole (they're small and their value
    # is in project association, not internal structure)
    if entry.is_support_file or entry.file_type == "config":
        raw_chunks = [parsed.raw_text]

    # Code files
    elif entry.file_type == "code":
        raw_chunks = _chunk_code(parsed.raw_text, entry.language)

    # Documents with sections from parser
    elif parsed.sections and len(parsed.sections) > 1:
        raw_chunks = _chunk_with_sections(parsed.sections)

    # Everything else: plain text splitting
    else:
        raw_chunks = _chunk_plain_text(parsed.raw_text)

    # Filter empty chunks and build Chunk objects
    raw_chunks = [c.strip() for c in raw_chunks if c.strip()]

    if not raw_chunks:
        return []

    total = len(raw_chunks)
    chunks: list[Chunk] = []

    for i, text in enumerate(raw_chunks):
        chunk = Chunk(
            id=_make_chunk_id(entry.relative_path, i),
            text=text,
            source_file=entry.relative_path,
            file_type=entry.file_type,
            language=entry.language,
            chunk_index=i,
            total_chunks=total,
            project_root=entry.project_root,
            is_support_file=entry.is_support_file,
            name_prefix=entry.name_prefix,
        )
        chunks.append(chunk)

    return chunks


# ---------------------------------------------------------------------------
# Batch chunking with progress
# ---------------------------------------------------------------------------


def chunk_all(parsed_files: list[ParsedFile]) -> list[Chunk]:
    """
    Chunk all parsed files with a progress indicator.
    Returns flat list of all chunks across all files.
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

    all_chunks: list[Chunk] = []
    files_chunked = 0
    files_skipped = 0

    # Only process successfully parsed files
    valid_files = [p for p in parsed_files if p.parse_success]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Chunking files", total=len(valid_files))

        for parsed in valid_files:
            chunks = chunk_file(parsed)

            if chunks:
                all_chunks.extend(chunks)
                files_chunked += 1
            else:
                files_skipped += 1

            progress.advance(task)

    console.print(f"  [green]Files chunked:[/green] {files_chunked}")
    console.print(f"  [green]Total chunks:[/green]  {len(all_chunks)}")
    if files_skipped:
        console.print(f"  [yellow]Files skipped (no content):[/yellow] {files_skipped}")

    # Token stats
    token_counts = [_estimate_tokens(c.text) for c in all_chunks]
    if token_counts:
        avg_tokens = sum(token_counts) // len(token_counts)
        min_tokens = min(token_counts)
        max_tokens = max(token_counts)
        console.print(
            f"  [dim]Chunk sizes: avg={avg_tokens}, min={min_tokens}, "
            f"max={max_tokens} tokens (est.)[/dim]"
        )
    console.print()

    return all_chunks
