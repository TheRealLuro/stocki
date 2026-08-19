"""Target definition.

The label answers one question: `horizon` bars after the window ends, is the
close higher than it was at the window end? It never reads a bar inside the
window, and the caller never gets a label whose future bar falls in the next
session, because windowing hands this function one session at a time.
"""

from __future__ import annotations

import numpy as np


def window_count(n_bars: int, window: int, horizon: int) -> int:
    """How many labelled windows fit in a session of `n_bars`. Never negative."""
    return max(0, n_bars - window - horizon + 1)


def direction_labels(
    close: np.ndarray,
    window: int,
    horizon: int = 1,
    threshold: float = 0.0,
) -> np.ndarray:
    """1 when the close rises by more than `threshold` over the next `horizon` bars.

    `threshold` is a fractional return, so 0.001 means "up by at least 0.1%".
    With the default of 0.0 the label is simply whether the next close is higher.
    """
    close = np.asarray(close, dtype=np.float64)
    count = window_count(len(close), window, horizon)
    if count == 0:
        return np.empty(0, dtype=np.int8)

    ends = np.arange(count) + window - 1
    at_end = close[ends]
    ahead = close[ends + horizon]

    with np.errstate(divide="ignore", invalid="ignore"):
        change = np.where(at_end != 0, (ahead - at_end) / at_end, 0.0)

    return (change > threshold).astype(np.int8)
