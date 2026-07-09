"""Tests for the repository -> project metadata resolver."""

from __future__ import annotations

import json

import pytest

from src.project_resolver import ProjectConfig, load_project_configs, resolve_project

VALID_CONFIG = {
    "TechDocker": {
        "project_id": "techdocker",
        "production_branch": "main",
        "document_name": "TechDocker Master Technical Document.docx",
        "document_location": "sharepoint-placeholder",
    },
    "OtherRepo": {
        "project_id": "other",
        "production_branch": "production",
        "document_name": "Other Master.docx",
        "document_location": "sharepoint-placeholder",
    },
}


def write_config(tmp_path, data):
    config_path = tmp_path / "projects.json"
    config_path.write_text(json.dumps(data), encoding="utf-8")
    return config_path


def test_load_project_configs_loads_valid_config(tmp_path):
    config_path = write_config(tmp_path, VALID_CONFIG)

    configs = load_project_configs(config_path)

    assert set(configs) == {"TechDocker", "OtherRepo"}
    techdocker = configs["TechDocker"]
    assert isinstance(techdocker, ProjectConfig)
    assert techdocker.repository_name == "TechDocker"
    assert techdocker.project_id == "techdocker"
    assert techdocker.production_branch == "main"
    assert techdocker.document_name == "TechDocker Master Technical Document.docx"
    assert techdocker.document_location == "sharepoint-placeholder"


def test_resolve_project_returns_correct_project(tmp_path):
    config_path = write_config(tmp_path, VALID_CONFIG)

    config = resolve_project("OtherRepo", config_path)

    assert config.repository_name == "OtherRepo"
    assert config.project_id == "other"
    assert config.production_branch == "production"


def test_unknown_repository_raises_key_error(tmp_path):
    config_path = write_config(tmp_path, VALID_CONFIG)

    with pytest.raises(KeyError, match="UnknownRepo"):
        resolve_project("UnknownRepo", config_path)


def test_missing_config_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_project_configs(tmp_path / "does-not-exist.json")


def test_missing_required_field_raises_value_error(tmp_path):
    broken = {
        "TechDocker": {
            "project_id": "techdocker",
            "production_branch": "main",
            # document_name and document_location missing
        }
    }
    config_path = write_config(tmp_path, broken)

    with pytest.raises(ValueError, match="document_name"):
        load_project_configs(config_path)


def test_invalid_json_root_raises_value_error(tmp_path):
    config_path = write_config(tmp_path, ["not", "an", "object"])

    with pytest.raises(ValueError, match="JSON object"):
        load_project_configs(config_path)
