"""Input/output helpers for the VR piloting experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml


def project_root(start: Path | None = None) -> Path:
    """Return the project root by walking upward until `vrdataset` is found."""
    current = (start or Path.cwd()).resolve()
    for path in [current, *current.parents]:
        if (path / "vrdataset").is_dir():
            return path
    raise FileNotFoundError("Could not locate project root containing 'vrdataset'.")


def ensure_directory(path: Path | str) -> Path:
    """Create a directory if needed and return it as a Path."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def load_config(path: Path | str) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def save_config(config: dict[str, Any], path: Path | str) -> Path:
    """Save a YAML configuration file."""
    target = Path(path)
    ensure_directory(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return target


def save_json(data: dict[str, Any], path: Path | str) -> Path:
    """Save JSON with stable indentation."""
    target = Path(path)
    ensure_directory(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    return target


def save_dataframe(df: pd.DataFrame, path: Path | str, index: bool = False) -> Path:
    """Save a DataFrame as CSV after creating the parent directory."""
    target = Path(path)
    ensure_directory(target.parent)
    df.to_csv(target, index=index)
    return target


def list_files(root: Path | str, suffixes: Iterable[str] | None = None) -> list[Path]:
    """List files recursively, optionally filtering by lowercase suffix."""
    root_path = Path(root)
    suffix_set = {suffix.lower() for suffix in suffixes} if suffixes else None
    files = [path for path in root_path.rglob("*") if path.is_file()]
    if suffix_set is None:
        return sorted(files)
    return sorted(path for path in files if path.suffix.lower() in suffix_set)
