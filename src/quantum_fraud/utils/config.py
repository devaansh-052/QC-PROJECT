"""Strict configuration with paths relative to the configuration's project root."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    """Validated settings for the current development phase."""

    seed: int
    log_level: str
    data_path: Path
    results_dir: Path


def load_config(path: str | Path, *, project_root: Path | None = None) -> ProjectConfig:
    """Load YAML; default root is the parent of the configs directory."""
    config_path = Path(path).resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot load configuration {config_path}: {exc}") from exc
    expected = {"seed", "log_level", "data_path", "results_dir"}
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError(f"Configuration must contain exactly: {sorted(expected)}")
    if type(raw["seed"]) is not int or not 0 <= raw["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if raw["log_level"] not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        raise ValueError("Invalid log_level")
    root = Path(project_root).resolve() if project_root else config_path.parent.parent
    paths = {}
    for key in ("data_path", "results_dir"):
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise ValueError(f"{key} must be a nonempty path string")
        paths[key] = (root / raw[key]).resolve()
    return ProjectConfig(seed=raw["seed"], log_level=raw["log_level"], **paths)
