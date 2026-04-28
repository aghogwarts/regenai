"""
Shared configuration for the ReGenAI pipeline.

All tunable parameters, model names, file type mappings, and thresholds
live here so they can be adjusted without touching module logic.
"""

from pathlib import Path
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIMENSIONS: int = 768  # bge-base-en-v1.5 outputs 768-dim dense vectors

DEFAULT_LLM_MODEL: str = "llama3.1:8b"
OLLAMA_BASE_URL: str = "http://localhost:11434"

# OpenRouter (used when model name contains "/", e.g. "meta-llama/llama-3.3-70b-instruct:free")
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

# ---------------------------------------------------------------------------
# Chunking parameters
# ---------------------------------------------------------------------------

# Maximum tokens per chunk (approx — we count by characters / 4 as heuristic)
CHUNK_MAX_TOKENS: int = 800
CHUNK_OVERLAP_TOKENS: int = 100
# Files smaller than this (in tokens) are embedded whole, no splitting
CHUNK_MIN_SPLIT_TOKENS: int = 500
# Character-to-token ratio approximation
CHARS_PER_TOKEN: int = 4

# ---------------------------------------------------------------------------
# Embedding parameters
# ---------------------------------------------------------------------------

EMBEDDING_BATCH_SIZE: int = 32

# ChromaDB collection names
CHROMA_CHUNK_COLLECTION: str = "regenai_chunks"
CHROMA_FILE_COLLECTION: str = "regenai_files"
CHROMA_PERSIST_DIR: str = ".regenai_store"

# ---------------------------------------------------------------------------
# Clustering parameters
# ---------------------------------------------------------------------------

# UMAP
UMAP_N_COMPONENTS: int = 25
UMAP_N_NEIGHBORS: int = 15
UMAP_MIN_DIST: float = 0.1
UMAP_METRIC: str = "cosine"

# HDBSCAN
HDBSCAN_MIN_CLUSTER_SIZE: int = 10
HDBSCAN_MIN_SAMPLES: int = 5

# GMM
GMM_COVARIANCE_TYPE: str = "diag"  # safer than "full" for our dimensionality
GMM_SECONDARY_THRESHOLD: float = 0.15  # minimum probability to record secondary cluster

# Quality thresholds
SILHOUETTE_MIN_THRESHOLD: float = 0.5
CLUSTER_RETRY_MAX: int = 5

# ---------------------------------------------------------------------------
# RAPTOR recursion
# ---------------------------------------------------------------------------

RAPTOR_STOP_CLUSTER_COUNT: int = 3  # stop recursing when clusters <= this
RAPTOR_MAX_DEPTH: int = 5  # hard cap on recursion depth

# ---------------------------------------------------------------------------
# File type classification
# ---------------------------------------------------------------------------

# Extension → file_type mapping
CODE_EXTENSIONS: set[str] = {
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".kts",
    ".scala",
    ".r",
    ".R",
    ".m",
    ".mm",
    ".lua",
    ".pl",
    ".pm",
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    ".ps1",
    ".bat",
    ".cmd",
    ".zig",
    ".nim",
    ".dart",
    ".ex",
    ".exs",
    ".erl",
    ".hs",
    ".ml",
    ".mli",
    ".clj",
    ".cljs",
    ".v",
    ".sv",
    ".css",
    ".scss",
    ".sass",
    ".less",
    ".styl",
}

DOCUMENT_EXTENSIONS: set[str] = {
    ".md",
    ".txt",
    ".rst",
    ".adoc",
    ".org",
    ".tex",
    ".rtf",
    ".pdf",
    ".docx",
    ".doc",
    ".odt",
    ".pptx",
    ".ppt",
    ".odp",
    ".xlsx",
    ".xls",
    ".ods",
    ".csv",
    ".tsv",
    ".html",
    ".htm",
    ".epub",
    ".cls",
    ".bib",
    ".bst",
    ".sty",
}

CONFIG_EXTENSIONS: set[str] = {
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".env",
    ".env.example",
    ".env.local",
    ".properties",
    ".xml",
}

DATA_EXTENSIONS: set[str] = {
    ".sql",
    ".graphql",
    ".gql",
    ".proto",
}

# Files that signal "this directory is a project root"
ANCHOR_FILES: set[str] = {
    # Python
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "requirements.txt",
    "Pipfile",
    # JavaScript/TypeScript
    "package.json",
    "tsconfig.json",
    # Java/JVM
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    # Go
    "go.mod",
    # Rust
    "Cargo.toml",
    # C/C++
    "CMakeLists.txt",
    "Makefile",
    "makefile",
    # .NET
    "*.csproj",
    "*.sln",
    # Docker
    "Dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    "compose.yml",
    "compose.yaml",
    # General — README is a weak anchor, ignored at scan root level
    "README.md",
    "README.rst",
    "README.txt",
    "README",
}

# Infrastructure/support files that inherit their project root's cluster
SUPPORT_FILE_NAMES: set[str] = {
    "Dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    "compose.yml",
    "compose.yaml",
    ".dockerignore",
    ".gitignore",
    ".editorconfig",
    "Makefile",
    "makefile",
    "Justfile",
    "LICENSE",
    "LICENSE.md",
    "LICENSE.txt",
    "CHANGELOG.md",
    "CHANGELOG",
    "CHANGES.md",
    "CONTRIBUTING.md",
    "CONTRIBUTING",
    "requirements.txt",
    "requirements-dev.txt",
    "setup.py",
    "setup.cfg",
    "pyproject.toml",
    "package.json",
    "package-lock.json",
    "yarn.lock",
    "tsconfig.json",
    "jest.config.js",
    "babel.config.js",
    ".eslintrc.js",
    ".eslintrc.json",
    ".prettierrc",
    "go.mod",
    "go.sum",
    "Cargo.toml",
    "Cargo.lock",
    "pom.xml",
    "build.gradle",
    "Procfile",
    "Vagrantfile",
    ".env",
    ".env.example",
    ".env.local",
    ".env.production",
    "tox.ini",
    "pytest.ini",
    ".flake8",
    "mypy.ini",
}

SUPPORT_FILE_EXTENSIONS: set[str] = {
    ".lock",
    ".sum",
}

# Extension → language name for tree-sitter grammar loading
LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "c_sharp",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
}

# Binary/media extensions to always skip (even if not in .regenai-ignore)
BINARY_EXTENSIONS: set[str] = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".ico",
    ".svg",
    ".webp",
    ".mp3",
    ".wav",
    ".ogg",
    ".flac",
    ".aac",
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
    ".7z",
    ".rar",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".bin",
    ".woff",
    ".woff2",
    ".ttf",
    ".otf",
    ".eot",
    ".pyc",
    ".pyo",
    ".class",
    ".o",
    ".obj",
    ".db",
    ".sqlite",
    ".sqlite3",
}


# ---------------------------------------------------------------------------
# Output configuration
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR: str = "regenai-output"
UNCATEGORIZED_DIR: str = "_uncategorized"


# ---------------------------------------------------------------------------
# File size limits
# ---------------------------------------------------------------------------

# Skip files larger than this (likely binary or generated)
MAX_FILE_SIZE_MB: int = 50


# Specific filenames (no extension) → file_type
KNOWN_FILENAMES: dict[str, str] = {
    "Dockerfile": "config",
    "Makefile": "config",
    "makefile": "config",
    "Justfile": "config",
    "Procfile": "config",
    "Vagrantfile": "config",
    "Gemfile": "config",
    "Rakefile": "code",
    "Brewfile": "config",
    ".env": "config",
    ".env.example": "config",
    ".env.local": "config",
    ".env.production": "config",
    ".gitignore": "config",
    ".gitattributes": "config",
    ".dockerignore": "config",
    ".editorconfig": "config",
    ".flake8": "config",
    ".eslintrc": "config",
    ".prettierrc": "config",
    "LICENSE": "document",
    "LICENCE": "document",
}


def classify_file_type(extension: str, filename: str = "") -> str:
    """Classify a file by its name (for extensionless files) or extension."""
    # Check filename first — handles Dockerfile, .env, etc.
    if filename in KNOWN_FILENAMES:
        return KNOWN_FILENAMES[filename]
    ext = extension.lower()
    if ext in CODE_EXTENSIONS:
        return "code"
    if ext in DOCUMENT_EXTENSIONS:
        return "document"
    if ext in CONFIG_EXTENSIONS:
        return "config"
    if ext in DATA_EXTENSIONS:
        return "data"
    if ext in BINARY_EXTENSIONS:
        return "binary"
    return "unknown"


def get_language(extension: str) -> str | None:
    """Get the tree-sitter language name for a code file extension."""
    return LANGUAGE_MAP.get(extension.lower())


def is_support_file(filename: str, extension: str) -> bool:
    """Check if a file is a support/infra file that should inherit its project root."""
    return (
        filename in SUPPORT_FILE_NAMES or extension.lower() in SUPPORT_FILE_EXTENSIONS
    )
