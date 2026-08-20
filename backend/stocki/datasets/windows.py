"""Turning tidy bars into CNN-ready windows.

Every session is windowed on its own. Trading days are not contiguous -- day 13
may be a week after day 12 -- so a window spanning two files would describe a
price move that never happened.

Derived channels only ever look at the current bar or the one before it, and
normalisation is per window, so nothing here can see the future.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .labels import direction_labels, window_count

#: Raw OHLCV plus three scale-free shapes computed from the bars themselves.
DEFAULT_CHANNELS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "log_return",
    "hl_range",
    "co_range",
)

DEFAULT_WINDOW = 32
DEFAULT_HORIZON = 1
DEFAULT_TEST_DAYS = 4
NORMALIZATIONS = ("window-z", None)


@dataclass(frozen=True)
class WindowSet:
    """Windows plus the metadata needed to slice, split, and audit them."""

    X: np.ndarray
    y: np.ndarray
    ticker: np.ndarray
    day: np.ndarray
    timestamps: np.ndarray
    feature_names: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.X)


def _naive_utc(series: pd.Series) -> np.ndarray:
    """datetime64[ns] holding UTC instants, so numpy can sort and compare them."""
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        series = series.dt.tz_convert("UTC").dt.tz_localize(None)
    return series.to_numpy(dtype="datetime64[ns]")


def derive_channels(session: pd.DataFrame) -> dict[str, np.ndarray]:
    """Every available channel for one time-ordered session."""
    open_ = session["open"].to_numpy(dtype=np.float64)
    high = session["high"].to_numpy(dtype=np.float64)
    low = session["low"].to_numpy(dtype=np.float64)
    close = session["close"].to_numpy(dtype=np.float64)
    volume = session["volume"].to_numpy(dtype=np.float64)

    log_return = np.zeros_like(close)
    if len(close) > 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            log_return[1:] = np.log(close[1:] / close[:-1])
    log_return = np.nan_to_num(log_return, nan=0.0, posinf=0.0, neginf=0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        hl_range = np.nan_to_num((high - low) / close, nan=0.0, posinf=0.0, neginf=0.0)
        co_range = np.nan_to_num((close - open_) / open_, nan=0.0, posinf=0.0, neginf=0.0)

    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "log_return": log_return,
        "hl_range": hl_range,
        "co_range": co_range,
    }


def normalize_windows(x: np.ndarray, method: str | None = "window-z") -> np.ndarray:
    """Normalise within each window and channel, so the model sees shape not price level.

    Using dataset-wide statistics would fold the test set into the training set,
    so the statistics never leave the window they came from. A channel that is
    flat inside a window becomes zeros rather than NaN.
    """
    if method is None:
        return x
    if method != "window-z":
        raise ValueError(f"unknown normalization {method!r}; expected one of {NORMALIZATIONS}")

    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return (x - mean) / np.where(std > 0, std, 1.0)


def _stack_windows(matrix: np.ndarray, window: int, count: int) -> np.ndarray:
    """(T, C) -> (count, window, C) without copying until the end."""
    strided = np.lib.stride_tricks.sliding_window_view(matrix, window, axis=0)
    return np.ascontiguousarray(strided.transpose(0, 2, 1)[:count])


def split_mask(
    days: np.ndarray,
    subset: str = "all",
    test_days: int = DEFAULT_TEST_DAYS,
) -> np.ndarray:
    """Chronological split by day number: the last `test_days` days are the test set.

    Splitting on days rather than on windows is what makes leakage structurally
    impossible -- no window can straddle the boundary, because none straddles a day.
    """
    days = np.asarray(days)
    if subset == "all":
        return np.ones(len(days), dtype=bool)
    if len(days) == 0:
        return np.zeros(0, dtype=bool)

    cutoff = days.max() - test_days
    if subset == "train":
        return days <= cutoff
    if subset == "test":
        return days > cutoff
    raise ValueError(f"unknown subset {subset!r}; expected 'all', 'train', or 'test'")


def build_windows(
    bars: pd.DataFrame,
    window: int = DEFAULT_WINDOW,
    horizon: int = DEFAULT_HORIZON,
    threshold: float = 0.0,
    channels: Sequence[str] | None = None,
    normalize: str | None = "window-z",
    channels_first: bool = False,
) -> WindowSet:
    """Window a tidy bars frame, one session at a time.

    `bars` needs ticker, day, timestamp, open, high, low, close, volume.
    """
    names = tuple(DEFAULT_CHANNELS if channels is None else channels)
    _check_parameters(window, horizon, names)

    ordered = bars.sort_values(["ticker", "day", "timestamp"], kind="stable")

    blocks: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    tickers: list[np.ndarray] = []
    days: list[np.ndarray] = []
    stamps: list[np.ndarray] = []

    for (ticker, day), session in ordered.groupby(["ticker", "day"], sort=True):
        count = window_count(len(session), window, horizon)
        if count == 0:
            continue

        available = derive_channels(session)
        matrix = np.column_stack([available[name] for name in names])

        blocks.append(_stack_windows(matrix, window, count))
        labels.append(direction_labels(available["close"], window, horizon, threshold))
        tickers.append(np.full(count, ticker, dtype=object))
        days.append(np.full(count, day, dtype=np.int16))
        stamps.append(_naive_utc(session["timestamp"])[window - 1 : window - 1 + count])

    if not blocks:
        return WindowSet(
            X=np.empty((0, window, len(names)), dtype=np.float32),
            y=np.empty(0, dtype=np.int8),
            ticker=np.empty(0, dtype=object),
            day=np.empty(0, dtype=np.int16),
            timestamps=np.empty(0, dtype="datetime64[ns]"),
            feature_names=names,
        )

    x = normalize_windows(np.concatenate(blocks), normalize).astype(np.float32)
    if channels_first:
        x = np.ascontiguousarray(x.transpose(0, 2, 1))

    return WindowSet(
        X=x,
        y=np.concatenate(labels),
        ticker=np.concatenate(tickers).astype(str),
        day=np.concatenate(days),
        timestamps=np.concatenate(stamps),
        feature_names=names,
    )


def _check_parameters(window: int, horizon: int, names: Iterable[str]) -> None:
    if window < 2:
        raise ValueError(f"window must be at least 2 bars, got {window}")
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1 bar, got {horizon}")
    unknown = [n for n in names if n not in DEFAULT_CHANNELS]
    if unknown:
        raise ValueError(f"unknown channels {unknown}; available: {list(DEFAULT_CHANNELS)}")
