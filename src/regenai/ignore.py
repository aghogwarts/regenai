"""
Ignore pattern engine for ReGenAI.

Loads ignore patterns from two sources:
1. Built-in defaults (embedded in this module)
2. User-defined .regenai-ignore in the target directory being scanned

Patterns follow .gitignore syntax via the `pathspec` library.
"""

from pathlib import Path

import pathspec

from regenai.config import BINARY_EXTENSIONS


# Built-in default patterns — same syntax as .gitignore.
# These are embedded directly so they survive package installation.
_DEFAULT_PATTERNS: list[str] = [
    # Package & dependency directories
    "node_modules/",
    "vendor/",
    "bower_components/",
    ".gradle/",
    "target/",
    "__pycache__/",
    "*.pyc",
    "*.pyo",
    ".eggs/",
    "*.egg-info/",
    "site-packages/",
    ".tox/",
    # Virtual environments
    "venv/",
    ".venv/",
    "env/",
    ".env/",
    "ENV/",
    "conda-envs/",
    # Build artifacts
    "dist/",
    "build/",
    "out/",
    ".next/",
    ".nuxt/",
    "*.min.js",
    "*.min.css",
    "*.map",
    "*.bundle.js",
    "_build/",
    # Cache & temp
    ".cache/",
    ".tmp/",
    "*.tmp",
    "*.swp",
    "*.swo",
    "*.bak",
    ".DS_Store",
    "Thumbs.db",
    "*.log",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    "__pypackages__/",
    # Lock files (no semantic content)
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "Cargo.lock",
    "go.sum",
    "Gemfile.lock",
    # Database artifacts
    "chroma_db/",
    "*.sqlite",
    "*.sqlite3",
    "*.db",
    # Version control
    ".git/",
    ".svn/",
    ".hg/",
    ".gitmodules",
    # IDE & editor
    ".idea/",
    ".vscode/",
    "*.suo",
    "*.user",
    "*.workspace",
    ".project",
    ".settings/",
    # OS generated
    "desktop.ini",
    "ehthumbs.db",
    # ReGenAI's own output
    "regenai-output/",
    ".regenai_store/",
]


def _load_patterns_from_file(filepath: Path) -> list[str]:
    """Read non-empty, non-comment lines from a patterns file."""
    if not filepath.exists():
        return []
    lines = filepath.read_text(encoding="utf-8").splitlines()
    return [line for line in lines if line.strip() and not line.strip().startswith("#")]


def build_ignore_spec(target_dir: Path) -> pathspec.PathSpec:
    """
    Build a PathSpec matcher combining:
    1. Built-in default patterns
    2. User patterns (.regenai-ignore in target_dir, if present)

    Args:
        target_dir: The root directory being scanned.

    Returns:
        A compiled PathSpec that can match relative file paths.
    """
    patterns: list[str] = list(_DEFAULT_PATTERNS)

    # Load user overrides from the target directory
    user_ignore = target_dir / ".regenai-ignore"
    patterns.extend(_load_patterns_from_file(user_ignore))

    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)


def should_ignore(
    relative_path: Path,
    ignore_spec: pathspec.PathSpec,
) -> bool:
    """
    Determine whether a file or directory should be ignored.

    Checks against:
    1. The compiled pathspec patterns (gitignore-style)
    2. Binary file extensions (always skipped regardless of patterns)

    Args:
        relative_path: File path relative to the scan root.
        ignore_spec: Compiled PathSpec matcher.

    Returns:
        True if the path should be skipped.
    """
    # Pathspec matching on the string representation
    path_str = str(relative_path)
    if ignore_spec.match_file(path_str):
        return True

    # Also check each parent directory component — if any directory in the
    # path is ignored, the whole path is ignored.  This handles cases like
    # "some/deep/node_modules/package/index.js" where node_modules/ is the
    # pattern but the file is nested deeper.
    for parent in relative_path.parents:
        parent_str = str(parent)
        if parent_str == ".":
            continue
        # Append trailing slash so directory patterns match
        if ignore_spec.match_file(parent_str + "/"):
            return True

    # Binary extension check (always skip, even without explicit patterns)
    if relative_path.suffix.lower() in BINARY_EXTENSIONS:
        return True

    return False
