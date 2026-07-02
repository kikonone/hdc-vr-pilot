"""Plotting helpers shared by later experiment phases."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def set_plot_style() -> None:
    """Apply a simple, readable Matplotlib style."""
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: plt.Figure, path: Path | str) -> Path:
    """Save a figure after creating the parent directory."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, bbox_inches="tight")
    return target
