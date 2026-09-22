"""
src/utils.py
------------
Shared helper utilities.
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import pandas as pd


def ensure_dirs(*paths: str | pathlib.Path) -> None:
    """Create directories (and parents) if they don't already exist."""
    for p in paths:
        pathlib.Path(p).mkdir(parents=True, exist_ok=True)


def save_fig(fig: plt.Figure, name: str, subdir: str = "outputs/plots") -> None:
    """
    Save a matplotlib figure to outputs/plots/<name>.png.

    Parameters
    ----------
    fig   : matplotlib Figure
    name  : filename without extension
    subdir: target directory
    """
    ensure_dirs(subdir)
    path = pathlib.Path(subdir) / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_table(df: pd.DataFrame, name: str, subdir: str = "outputs/reports") -> None:
    """Save a DataFrame as CSV to outputs/reports/<name>.csv."""
    ensure_dirs(subdir)
    path = pathlib.Path(subdir) / f"{name}.csv"
    df.to_csv(path, index=True)


def pct(numerator: pd.Series | float, denominator: pd.Series | float, decimals: int = 1) -> pd.Series | float:
    """Return percentage, safe against zero-division."""
    return (numerator / denominator * 100).round(decimals)
