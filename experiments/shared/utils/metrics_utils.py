"""Metric helpers for later four-class classification phases."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def classification_summary(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    average: str = "macro",
) -> dict[str, float]:
    """Return common classification metrics without training a model."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average=average)),
    }


def metrics_table(rows: list[dict]) -> pd.DataFrame:
    """Create a stable metrics table from a list of result dictionaries."""
    return pd.DataFrame(rows)
