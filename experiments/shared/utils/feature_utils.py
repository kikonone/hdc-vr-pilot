"""Feature helper functions for later multimodal phases.

Phase 00 does not extract features. These helpers are intentionally generic so later
phases can use them after the raw modality audit defines valid signals and windows.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric column names from a DataFrame."""
    return df.select_dtypes(include=[np.number]).columns.tolist()


def summarize_numeric_columns(df: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Summarize numeric columns without changing the source data."""
    selected = list(columns) if columns is not None else numeric_columns(df)
    if not selected:
        return pd.DataFrame(columns=["column", "count", "mean", "std", "min", "max"])
    summary = df[selected].agg(["count", "mean", "std", "min", "max"]).T.reset_index()
    return summary.rename(columns={"index": "column"})


def safe_zscore(values: np.ndarray) -> np.ndarray:
    """Compute a z-score while avoiding division by zero."""
    arr = np.asarray(values, dtype=float)
    std = arr.std()
    if std == 0 or np.isnan(std):
        return np.zeros_like(arr, dtype=float)
    return (arr - arr.mean()) / std
