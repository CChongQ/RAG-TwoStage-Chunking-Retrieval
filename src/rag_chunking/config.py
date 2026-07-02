"""Project-level path helpers and YAML configuration loading."""

from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any, TypeVar

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CONFIG_DIR = PROJECT_ROOT / "configs"

T = TypeVar("T")


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a project-relative path from the repository root."""
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping from an absolute or project-relative path."""
    config_path = resolve_project_path(path)
    with config_path.open("r", encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {config_path}")
    return data


def load_dataclass_config(config_type: type[T], path: str | Path, **overrides: Any) -> T:
    """Load a dataclass config from YAML and apply non-None overrides."""
    if not is_dataclass(config_type):
        raise TypeError(f"{config_type!r} must be a dataclass type")

    data = load_yaml(path)
    field_names = {field.name for field in fields(config_type)}
    unknown_keys = sorted(set(data) - field_names)
    if unknown_keys:
        raise ValueError(
            f"Unknown config key(s) for {config_type.__name__}: {', '.join(unknown_keys)}"
        )

    config = config_type(**data)
    active_overrides = {
        key: value
        for key, value in overrides.items()
        if value is not None
    }
    return replace(config, **active_overrides)
