"""
File Registry — Module 1 of the ReGenAI pipeline.

Recursively walks the input directory, filters ignored paths, and builds
a structured registry of every file to be processed.  Each file gets
classified by type, associated with a project root (if detected), and
tagged with naming patterns for downstream clustering.

This module does NO file I/O beyond stat calls — parsing happens later.
"""

import re
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.table import Table

from regenai.config import (
    ANCHOR_FILES,
    MAX_FILE_SIZE_MB,
    classify_file_type,
    get_language,
    is_support_file,
)
from regenai.ignore import build_ignore_spec, should_ignore


console = Console()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FileEntry:
    """Metadata record for a single discovered file."""

    path: Path  # absolute path
    relative_path: Path  # relative to input root
    extension: str  # e.g. ".py", ".docx"
    file_type: str  # "code", "document", "config", "data", "unknown"
    language: str | None  # for code files: "python", "javascript", etc.
    size_bytes: int
    modified_at: datetime
    parent_dir: str  # immediate parent directory name
    depth: int  # directory depth from scan root
    project_root: str | None  # detected project root (relative path), if anchored
    is_support_file: bool  # Dockerfile, requirements.txt, etc.
    name_prefix: str | None  # extracted naming pattern for version detection


@dataclass
class RegistryResult:
    """Output of the file discovery phase."""

    root_dir: Path
    entries: list[FileEntry]
    project_roots: list[str]  # list of detected project root relative paths
    total_files_scanned: int  # before filtering
    total_files_ignored: int
    total_files_oversized: int
    total_files_binary: int


# ---------------------------------------------------------------------------
# Name prefix extraction
# ---------------------------------------------------------------------------

# Patterns stripped from filenames to extract a canonical "prefix" for
# detecting versions/duplicates: Q3_budget_v2_FINAL.docx → "q3_budget"
_VERSION_SUFFIXES = re.compile(
    r"[_\-\s]*"
    r"(?:"
    r"v\d+|ver\d*|version\d*|"
    r"final|draft|old|new|latest|copy|backup|revised|updated|"
    r"edit(?:ed)?|review|approved|signed|"
    r"\(\d+\)|"  # (1), (2), etc.
    r"\d{4}[-_]\d{2}[-_]\d{2}"  # date suffixes like 2024-03-15
    r")"
    r"[_\-\s]*",
    re.IGNORECASE,
)


def _extract_name_prefix(filename: str) -> str | None:
    """
    Strip version/status suffixes from a filename stem to get a canonical
    prefix.  Returns None if the result is too short to be meaningful.

    Examples:
        "Q3_budget_v2_FINAL" → "q3_budget"
        "client_pitch_old"   → "client_pitch"
        "a"                  → None
    """
    stem = Path(filename).stem
    cleaned = _VERSION_SUFFIXES.sub("", stem).strip("_- ").lower()
    # Too short or empty after cleaning — not useful as a prefix
    if len(cleaned) < 3:
        return None
    return cleaned


# ---------------------------------------------------------------------------
# Project root detection
# ---------------------------------------------------------------------------


def _detect_project_roots(root_dir: Path, all_dirs: set[Path]) -> dict[Path, str]:
    """
    Scan directories for anchor files and return a mapping from
    absolute directory path → relative path string for each project root.

    A directory is a project root if it contains at least one anchor file
    (package.json, requirements.txt, Cargo.toml, etc.).

    README files are "weak" anchors — they count for subdirectories but
    are ignored at the scan root to avoid tagging the entire input as
    one project.
    """
    weak_anchors = {"README.md", "README.rst", "README.txt", "README"}
    roots: dict[Path, str] = {}

    for dir_path in sorted(all_dirs):
        is_scan_root = dir_path == root_dir
        for anchor in ANCHOR_FILES:
            # Skip weak anchors at scan root
            if is_scan_root and anchor in weak_anchors:
                continue
            # Handle glob patterns like *.csproj
            if "*" in anchor:
                if list(dir_path.glob(anchor)):
                    rel = str(dir_path.relative_to(root_dir))
                    roots[dir_path] = rel
                    break
            else:
                if (dir_path / anchor).exists():
                    rel = str(dir_path.relative_to(root_dir))
                    roots[dir_path] = rel
                    break

    # Remove nested roots — if a root is inside another root, the parent wins.
    # e.g. expenses-pwa\backend gets absorbed into expenses-pwa
    to_remove: set[Path] = set()
    sorted_roots = sorted(roots.keys())
    for i, child in enumerate(sorted_roots):
        for parent in sorted_roots[:i]:
            if parent in to_remove:
                continue
            try:
                child.relative_to(parent)
                # child is inside parent — remove child
                to_remove.add(child)
                break
            except ValueError:
                continue

    for path in to_remove:
        del roots[path]

    return roots


def _find_project_root_for_file(
    file_path: Path,
    project_roots: dict[Path, str],
) -> str | None:
    """
    Walk up from a file's directory to find the nearest project root.
    Returns the relative path string of the project root, or None.
    """
    current = file_path.parent
    while current:
        if current in project_roots:
            return project_roots[current]
        # Don't walk above any known root — nested roots stay separate
        current_parent = current.parent
        if current_parent == current:
            break
        current = current_parent
    return None


# ---------------------------------------------------------------------------
# Main registry builder
# ---------------------------------------------------------------------------


def build_registry(root_dir: Path) -> RegistryResult:
    """
    Walk the directory tree and build a complete file registry.

    Steps:
    1. Build ignore spec from default + user patterns
    2. Walk tree, collect all directories and non-ignored files
    3. Detect project roots from anchor files
    4. Classify each file and build FileEntry objects

    Args:
        root_dir: Absolute path to the directory to scan.

    Returns:
        RegistryResult with all discovered file entries and metadata.
    """
    root_dir = root_dir.resolve()
    if not root_dir.is_dir():
        raise ValueError(f"Not a directory: {root_dir}")

    ignore_spec = build_ignore_spec(root_dir)

    # -- Pass 1: Collect all directories and file paths --
    all_dirs: set[Path] = {root_dir}
    candidate_files: list[Path] = []
    total_scanned = 0
    total_ignored = 0

    for item in sorted(root_dir.rglob("*")):
        relative = item.relative_to(root_dir)

        if should_ignore(relative, ignore_spec):
            total_ignored += 1
            continue

        if item.is_dir():
            all_dirs.add(item)
        elif item.is_file():
            total_scanned += 1
            candidate_files.append(item)

    # -- Pass 2: Detect project roots --
    project_roots = _detect_project_roots(root_dir, all_dirs)

    # -- Pass 3: Build file entries --
    entries: list[FileEntry] = []
    total_oversized = 0
    total_binary = 0

    for fpath in candidate_files:
        relative = fpath.relative_to(root_dir)
        ext = fpath.suffix.lower()
        ftype = classify_file_type(ext, fpath.name)

        # Skip binary files that slipped through ignore patterns
        if ftype == "binary":
            total_binary += 1
            continue

        # Skip oversized files
        try:
            size = fpath.stat().st_size
        except OSError:
            continue

        if size > MAX_FILE_SIZE_MB * 1024 * 1024:
            total_oversized += 1
            continue

        # Skip empty files
        if size == 0:
            continue

        try:
            mtime = datetime.fromtimestamp(fpath.stat().st_mtime)
        except OSError:
            mtime = datetime.now()

        entry = FileEntry(
            path=fpath,
            relative_path=relative,
            extension=ext,
            file_type=ftype,
            language=get_language(ext),
            size_bytes=size,
            modified_at=mtime,
            parent_dir=fpath.parent.name,
            depth=len(relative.parts) - 1,
            project_root=_find_project_root_for_file(fpath, project_roots),
            is_support_file=is_support_file(fpath.name, ext),
            name_prefix=_extract_name_prefix(fpath.name),
        )
        entries.append(entry)

    return RegistryResult(
        root_dir=root_dir,
        entries=entries,
        project_roots=list(project_roots.values()),
        total_files_scanned=total_scanned,
        total_files_ignored=total_ignored,
        total_files_oversized=total_oversized,
        total_files_binary=total_binary,
    )


# ---------------------------------------------------------------------------
# Display utilities (used by --dry-run and --verbose)
# ---------------------------------------------------------------------------


def print_registry_summary(result: RegistryResult) -> None:
    """Print a concise summary of what the registry found."""
    console.print()
    console.print(f"[bold]Scan root:[/bold] {result.root_dir}")
    console.print(f"  Files scanned:    {result.total_files_scanned}")
    console.print(f"  Files ignored:    {result.total_files_ignored}")
    console.print(f"  Files oversized:  {result.total_files_oversized}")
    console.print(f"  Binary skipped:   {result.total_files_binary}")
    console.print(f"  [green]Files to process: {len(result.entries)}[/green]")
    console.print()

    if result.project_roots:
        console.print(
            f"[bold]Detected {len(result.project_roots)} project root(s):[/bold]"
        )
        for root in result.project_roots:
            console.print(f"  📁 {root}")
        console.print()

    # Type breakdown
    type_counts: dict[str, int] = {}
    for entry in result.entries:
        type_counts[entry.file_type] = type_counts.get(entry.file_type, 0) + 1

    console.print("[bold]File type breakdown:[/bold]")
    for ftype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        console.print(f"  {ftype:<12} {count}")
    console.print()


def print_registry_detail(result: RegistryResult) -> None:
    """Print a detailed table of all registered files (for --verbose or --dry-run)."""
    table = Table(title="File Registry", show_lines=False, pad_edge=False)
    table.add_column("Relative Path", style="cyan", max_width=60)
    table.add_column("Type", style="green")
    table.add_column("Lang", style="yellow")
    table.add_column("Size", justify="right")
    table.add_column("Project Root", style="magenta", max_width=30)
    table.add_column("Support?", justify="center")
    table.add_column("Name Prefix", style="dim", max_width=25)

    for entry in result.entries:
        size_str = _format_size(entry.size_bytes)
        table.add_row(
            str(entry.relative_path),
            entry.file_type,
            entry.language or "",
            size_str,
            entry.project_root or "",
            "✓" if entry.is_support_file else "",
            entry.name_prefix or "",
        )

    console.print(table)
    console.print()


def _format_size(size_bytes: int) -> str:
    """Format bytes into human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"
