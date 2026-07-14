"""Collect safe repository context for summary generation.

Walks the repository, skips anything sensitive, generated, or binary, and
returns a size-bounded :class:`RepoContext` that a summary provider (today the
deterministic one, later Copilot/an LLM) can turn into a project summary.
No official/company documents are read — only the repository itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Directories never entered.
IGNORED_DIRECTORIES = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    "dist",
    "build",
    "artifacts",
}

# File suffixes never included (neither tree nor content).
IGNORED_SUFFIXES = {".docx", ".csv", ".pyc", ".pem", ".key"}

# File-name patterns that indicate secrets.
_SECRET_MARKERS = ("secret", "credential", "token")

# Suffixes whose content is safe and useful to read.
TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".json", ".yml", ".yaml", ".toml", ".ini", ".cfg",
}

# Size limits to keep future token usage reasonable.
MAX_FILE_CHARS = 6_000
MAX_TOTAL_CHARS = 150_000
MAX_CONTENT_FILES = 50
MAX_TREE_ENTRIES = 400


@dataclass
class RepoContext:
    """Everything a summary provider may look at."""

    root: str
    project_name: str
    file_tree: list[str] = field(default_factory=list)
    files: dict[str, str] = field(default_factory=dict)  # path -> content
    truncated_files: list[str] = field(default_factory=list)
    skipped_binary: list[str] = field(default_factory=list)
    total_files: int = 0
    collected_chars: int = 0


def _is_ignored_file(path: Path) -> bool:
    name = path.name.lower()
    if path.suffix.lower() in IGNORED_SUFFIXES:
        return True
    if name == ".env" or name.startswith(".env."):
        return True
    if any(marker in name for marker in _SECRET_MARKERS):
        return True
    if name == ".ds_store":
        return True
    return False


def _read_text_safely(path: Path) -> str | None:
    """Return file text, or ``None`` for binary/undecodable content."""
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in raw:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def collect_repo_context(repo_path: str | Path = ".") -> RepoContext:
    """Walk the repository and collect a bounded, safe context."""
    root = Path(repo_path).resolve()
    context = RepoContext(root=str(root), project_name=root.name)

    candidates: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        relative = path.relative_to(root)
        if any(part in IGNORED_DIRECTORIES for part in relative.parts):
            continue
        if _is_ignored_file(path):
            continue
        # Only text-like files enter the context at all — the provider (and
        # any future LLM) must never see binary or unknown formats.
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in (
            "requirements.txt",
            "pytest.ini",
            ".gitignore",
        ):
            continue
        candidates.append(path)

    context.total_files = len(candidates)
    context.file_tree = [
        str(path.relative_to(root)) for path in candidates[:MAX_TREE_ENTRIES]
    ]

    for path in candidates:
        if len(context.files) >= MAX_CONTENT_FILES:
            break
        if context.collected_chars >= MAX_TOTAL_CHARS:
            break

        relative = str(path.relative_to(root))
        text = _read_text_safely(path)
        if text is None:
            context.skipped_binary.append(relative)
            continue

        if len(text) > MAX_FILE_CHARS:
            text = text[:MAX_FILE_CHARS]
            context.truncated_files.append(relative)

        context.files[relative] = text
        context.collected_chars += len(text)

    return context
