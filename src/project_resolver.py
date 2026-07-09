"""Map repository names to project and document metadata.

The first layer of the GitHub automation phase: given the name of a pushed
repository, resolve which project it belongs to and which master document
that project maintains. The mapping lives in ``config/projects.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Every project entry in the config must provide these fields.
REQUIRED_FIELDS: tuple[str, ...] = (
    "project_id",
    "production_branch",
    "document_name",
    "document_location",
)


@dataclass
class ProjectConfig:
    """Project and document metadata for one configured repository."""

    repository_name: str
    project_id: str
    production_branch: str
    document_name: str
    document_location: str


def load_project_configs(config_path: str | Path) -> dict[str, ProjectConfig]:
    """Load the repository -> project mapping from a JSON config file.

    Raises ``FileNotFoundError`` when the file does not exist and
    ``ValueError`` when the JSON root is not an object or a project entry is
    missing required fields. The repository name is taken from the JSON key.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Project config not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError(
            f"Project config root must be a JSON object mapping repository "
            f"names to project entries, got {type(data).__name__}: {path}"
        )

    configs: dict[str, ProjectConfig] = {}
    for repository_name, entry in data.items():
        if not isinstance(entry, dict):
            raise ValueError(
                f"Project entry for {repository_name!r} must be a JSON "
                f"object, got {type(entry).__name__}"
            )

        missing = [field for field in REQUIRED_FIELDS if field not in entry]
        if missing:
            raise ValueError(
                f"Project entry for {repository_name!r} is missing required "
                f"fields: {', '.join(missing)}"
            )

        configs[repository_name] = ProjectConfig(
            repository_name=repository_name,
            project_id=entry["project_id"],
            production_branch=entry["production_branch"],
            document_name=entry["document_name"],
            document_location=entry["document_location"],
        )

    return configs


def resolve_project(
    repository_name: str,
    config_path: str | Path = "config/projects.json",
) -> ProjectConfig:
    """Return the :class:`ProjectConfig` for one repository.

    Raises ``KeyError`` with a clear message when the repository is not
    configured.
    """
    configs = load_project_configs(config_path)

    if repository_name not in configs:
        known = ", ".join(sorted(configs)) or "none"
        raise KeyError(
            f"Repository {repository_name!r} is not configured in "
            f"{config_path} (known repositories: {known})"
        )

    return configs[repository_name]
