"""Dataset access for Stocki.

    from stocki.datasets import load_stocki, load_bars, load_panel

Nothing in here imports the API layer, so training and notebooks never need
FastAPI installed.
"""

from .labels import direction_labels
from .loaders import (
    Bunch,
    Panel,
    coverage,
    describe,
    known_tickers,
    load_bars,
    load_fundamentals,
    load_panel,
    load_raw,
    load_stocki,
)
from .windows import DEFAULT_CHANNELS, WindowSet, build_windows, split_mask

__all__ = [
    "Bunch",
    "DEFAULT_CHANNELS",
    "Panel",
    "WindowSet",
    "build_windows",
    "coverage",
    "describe",
    "direction_labels",
    "known_tickers",
    "load_bars",
    "load_fundamentals",
    "load_panel",
    "load_raw",
    "load_stocki",
    "split_mask",
]
