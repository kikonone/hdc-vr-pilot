"""Path parsing helpers for the VR piloting dataset."""

from __future__ import annotations

import re
from pathlib import Path

LEVEL_RE = re.compile(r"level-(?P<level>\d{2,3})(?P<suffix>[A-Za-z]*)")
RUN_RE = re.compile(r"run-(?P<run>\d+)")
SUBJECT_RE = re.compile(r"sub-(?P<subject>[A-Za-z0-9]+)")
SESSION_RE = re.compile(r"ses-(?P<session>[A-Za-z0-9]+)")
STREAM_RE = re.compile(r"stream-(?P<stream>[^_]+)")


def parse_level_token(text: str) -> dict[str, str | int | None]:
    """Parse level tokens such as `level-01B` or `level-000`."""
    match = LEVEL_RE.search(text)
    if not match:
        return {"level_token": None, "difficulty_level": None, "level_suffix": None}
    numeric = match.group("level")
    difficulty = int(numeric) if numeric != "000" else 0
    return {
        "level_token": match.group(0),
        "difficulty_level": difficulty,
        "level_suffix": match.group("suffix") or None,
    }


def parse_dataset_path(path: Path | str) -> dict[str, str | int | None]:
    """Extract common subject, session, level, run, task, and stream metadata from a dataset path."""
    text = Path(path).as_posix()
    parts = Path(path).parts
    parsed = parse_level_token(text)
    parsed.update(
        {
            "subject": _first_group(SUBJECT_RE, text, "subject"),
            "session": _first_group(SESSION_RE, text, "session"),
            "run": _int_or_none(_first_group(RUN_RE, text, "run")),
            "stream": _first_group(STREAM_RE, text, "stream"),
            "task": next((part for part in parts if part.startswith("task-")), None),
        }
    )
    return parsed


def is_four_class_task_level(level: int | None) -> bool:
    """Return True for the thesis target levels 1, 2, 3, and 4."""
    return level in {1, 2, 3, 4}


def _first_group(pattern: re.Pattern[str], text: str, group: str) -> str | None:
    match = pattern.search(text)
    return match.group(group) if match else None


def _int_or_none(value: str | None) -> int | None:
    return int(value) if value is not None else None
