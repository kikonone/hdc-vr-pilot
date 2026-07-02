"""Split helpers for subject-aware experiment phases."""

from __future__ import annotations

from collections.abc import Iterable

from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold


def make_group_holdout_split(
    labels: Iterable[int],
    groups: Iterable[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> GroupShuffleSplit:
    """Create a reusable group holdout splitter."""
    return GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)


def make_stratified_cv(n_splits: int = 5, random_state: int = 42) -> StratifiedKFold:
    """Create a stratified cross-validation splitter."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
