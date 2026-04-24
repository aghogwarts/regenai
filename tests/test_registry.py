"""
Quick integration test for Module 1 (registry + ignore).

Creates a mock messy directory structure and verifies the registry
correctly discovers, classifies, filters, and tags files.
"""

from pathlib import Path
import tempfile
import shutil

from regenai.registry import (
    build_registry,
    print_registry_summary,
    print_registry_detail,
)


def create_mock_directory(root: Path) -> None:
    """Create a realistic messy directory for testing."""

    # -- Project A: a Python project with proper structure --
    proj_a = root / "project_alpha" / "src"
    proj_a.mkdir(parents=True)
    (proj_a / "main.py").write_text("def main():\n    print('hello')\n")
    (proj_a / "utils.py").write_text("def helper():\n    return 42\n")
    (proj_a.parent / "requirements.txt").write_text("requests\nflask\n")
    (proj_a.parent / "Dockerfile").write_text("FROM python:3.11\nCOPY . .\n")
    (proj_a.parent / "README.md").write_text("# Project Alpha\nA demo project.\n")
    (proj_a.parent / ".env").write_text("SECRET_KEY=abc123\n")
    (proj_a / "schema.sql").write_text("CREATE TABLE users (id INT);\n")

    # -- Project B: a JS project --
    proj_b = root / "web_app"
    proj_b.mkdir()
    (proj_b / "package.json").write_text('{"name": "web-app", "version": "1.0"}\n')
    (proj_b / "index.js").write_text("const app = require('express')();\n")
    (proj_b / "style.css").write_text("body { margin: 0; }\n")

    # -- Scattered docs (no project root) --
    docs = root / "john_docs"
    docs.mkdir()
    (docs / "Q3_budget_draft.txt").write_text("Q3 budget: $50k for infra.\n")
    (docs / "meeting_notes_2024.md").write_text(
        "# Kickoff\nDiscussed alpha and budget.\n"
    )

    old = root / "old_stuff"
    old.mkdir()
    (old / "Q3_budget_FINAL.txt").write_text("Q3 budget finalized: $55k.\n")
    (old / "client_pitch_old.txt").write_text("Our proposal for client X.\n")

    presentations = root / "presentations"
    presentations.mkdir()
    (presentations / "client_pitch_v2.txt").write_text(
        "Updated proposal for client X.\n"
    )
    (presentations / "onboarding_deck.txt").write_text("Welcome to the team!\n")

    # -- Things that should be ignored --
    cache = root / "project_alpha" / "__pycache__"
    cache.mkdir()
    (cache / "main.cpython-311.pyc").write_bytes(b"\x00\x01\x02")

    venv = root / "project_alpha" / "venv" / "lib"
    venv.mkdir(parents=True)
    (venv / "site.py").write_text("# venv internals\n")

    git = root / ".git" / "objects"
    git.mkdir(parents=True)
    (git / "abc123").write_bytes(b"\x00")

    node_modules = root / "web_app" / "node_modules" / "express"
    node_modules.mkdir(parents=True)
    (node_modules / "index.js").write_text("module.exports = {};\n")

    # -- Binary file that should be skipped --
    (root / "photo.jpg").write_bytes(b"\xff\xd8\xff\xe0")

    # -- Empty file that should be skipped --
    (root / "empty.txt").write_text("")


def main():
    tmp = Path(tempfile.mkdtemp(prefix="regenai_test_"))
    print(f"Created test directory: {tmp}\n")

    try:
        create_mock_directory(tmp)
        result = build_registry(tmp)
        print_registry_summary(result)
        print_registry_detail(result)

        # -- Assertions --
        print("Running assertions...")

        # File counts
        assert len(result.entries) > 0, "Should find some files"

        # Project roots detected
        assert (
            len(result.project_roots) >= 2
        ), f"Should detect at least 2 project roots, found {result.project_roots}"

        # Ignored files
        cached = [e for e in result.entries if "__pycache__" in str(e.relative_path)]
        assert len(cached) == 0, "Should ignore __pycache__ files"

        venv_files = [e for e in result.entries if "venv" in str(e.relative_path)]
        assert len(venv_files) == 0, "Should ignore venv files"

        git_files = [e for e in result.entries if ".git" in str(e.relative_path)]
        assert len(git_files) == 0, "Should ignore .git files"

        nm_files = [e for e in result.entries if "node_modules" in str(e.relative_path)]
        assert len(nm_files) == 0, "Should ignore node_modules files"

        # Binary skip
        jpg_files = [e for e in result.entries if e.extension == ".jpg"]
        assert len(jpg_files) == 0, "Should skip binary files"

        # Empty file skip
        empty = [e for e in result.entries if "empty.txt" in str(e.relative_path)]
        assert len(empty) == 0, "Should skip empty files"

        # Project root assignment
        main_py = [e for e in result.entries if e.relative_path.name == "main.py"][0]
        assert main_py.project_root is not None, "main.py should have a project root"
        assert (
            "project_alpha" in main_py.project_root
        ), "main.py should be under project_alpha"

        # Support file detection
        dockerfile = [
            e for e in result.entries if e.relative_path.name == "Dockerfile"
        ][0]
        assert (
            dockerfile.is_support_file
        ), "Dockerfile should be flagged as support file"

        # Name prefix extraction (version detection)
        q3_files = [
            e for e in result.entries if e.name_prefix and "q3_budget" in e.name_prefix
        ]
        assert (
            len(q3_files) >= 2
        ), f"Should detect Q3 budget name prefix in at least 2 files, found {len(q3_files)}"

        # File type classification
        py_files = [e for e in result.entries if e.file_type == "code"]
        assert len(py_files) >= 2, "Should classify .py/.js as code"

        # Extensionless file classification (Dockerfile, .env)
        dockerfile = [
            e for e in result.entries if e.relative_path.name == "Dockerfile"
        ][0]
        assert (
            dockerfile.file_type == "config"
        ), f"Dockerfile should be config, got {dockerfile.file_type}"

        env_file = [e for e in result.entries if e.relative_path.name == ".env"][0]
        assert (
            env_file.file_type == "config"
        ), f".env should be config, got {env_file.file_type}"

        # CSS classification
        css_file = [e for e in result.entries if e.relative_path.name == "style.css"][0]
        assert (
            css_file.file_type == "code"
        ), f"style.css should be code, got {css_file.file_type}"

        print("\n✅ All assertions passed!\n")

    finally:
        shutil.rmtree(tmp)
        print(f"Cleaned up: {tmp}")


if __name__ == "__main__":
    main()
