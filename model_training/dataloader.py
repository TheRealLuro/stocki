"""Dataloader for stocki training examples.

Where the data comes from
-------------------------
Bars are read from the backend API over HTTP (``api_client.fetch_bars``), then
windowed, labelled and normalised here. The transform below is a deliberate
mirror of ``backend/stocki/datasets/{windows,labels}.py`` -- same formulas, same
session grouping, same ordering -- so the tensors this file produces are the
same tensors ``/api/v1/dataset/windows`` serves. ``verify_against_api()`` proves
that on every run rather than trusting the comment.

Why the transform is repeated instead of paged from the API: the
``/api/v1/dataset/windows`` route accepts no ``offset`` and is capped at 500
windows, so it can only ever hand back the first 500 of 8,602. ``/api/v1/bars``
is paginated and serves all 14,586 bars, so the windowing has to happen
client-side.

Leakage controls, all three inherited from the backend design:
  1. No window crosses a session boundary -- windowing runs per (ticker, day).
  2. Splits are by trading day, never random -- overlapping windows would
     otherwise appear on both sides of the boundary.
  3. Normalisation statistics come from inside a single window only.

Example contract
----------------
``get_example(i)`` returns ``(x, y)``:

    x : float32 (NUM_INPUT_FEATURES, sequence_length)  channels-first, as Conv1d wants
    y : float32 (NUM_OUTPUTS,)                          1.0 = UP, 0.0 = DOWN
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
import torch

import api_client
import config

# =====================================================================
# Transform -- mirrors backend/stocki/datasets/{windows,labels}.py
# =====================================================================


def window_count(n_bars: int, window: int, horizon: int) -> int:
    """How many labelled windows fit in a session of ``n_bars``. Never negative."""
    return max(0, n_bars - window - horizon + 1)


def direction_labels(
    close: np.ndarray,
    window: int,
    horizon: int = config.HORIZON,
    threshold: float = config.LABEL_THRESHOLD,
) -> np.ndarray:
    """1 when the close rises by more than ``threshold`` over the next ``horizon`` bars.

    ``threshold`` is a fractional return, so 0.001 means "up by at least 0.1%".
    The bar being compared against sits *after* the window, so the model never
    sees its own answer.
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


def derive_channels(session: np.ndarray) -> dict[str, np.ndarray]:
    """Every available channel for one time-ordered session of bars.

    The three derived channels read the current bar and the one immediately
    before it, and nothing later.
    """
    open_ = session["open"].astype(np.float64)
    high = session["high"].astype(np.float64)
    low = session["low"].astype(np.float64)
    close = session["close"].astype(np.float64)
    volume = session["volume"].astype(np.float64)

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


def normalize_windows(x: np.ndarray, method: str | None = config.NORMALIZE) -> np.ndarray:
    """Z-score within each window and channel, so the model sees shape not price level.

    A channel that is flat inside a window becomes zeros rather than NaN.
    """
    if method is None:
        return x
    if method != "window-z":
        raise ValueError(f"unknown normalization {method!r}; expected 'window-z' or None")

    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return (x - mean) / np.where(std > 0, std, 1.0)


def _stack_windows(matrix: np.ndarray, window: int, count: int) -> np.ndarray:
    """(T, C) -> (count, window, C) without copying until the end."""
    strided = np.lib.stride_tricks.sliding_window_view(matrix, window, axis=0)
    return np.ascontiguousarray(strided.transpose(0, 2, 1)[:count])


def _session_bounds(bars: np.ndarray) -> Iterator[tuple[int, int]]:
    """(start, stop) row ranges for each (ticker, day) group of a sorted table."""
    if len(bars) == 0:
        return
    changed = (bars["ticker"][1:] != bars["ticker"][:-1]) | (bars["day"][1:] != bars["day"][:-1])
    starts = np.concatenate(([0], np.nonzero(changed)[0] + 1))
    stops = np.concatenate((starts[1:], [len(bars)]))
    yield from zip(starts.tolist(), stops.tolist())


@dataclass(frozen=True)
class WindowSet:
    """Windows plus the metadata needed to slice, split and audit them."""

    X: np.ndarray  # float32 (N, channels, window) -- channels-first
    y: np.ndarray  # int8 (N,) -- 1 = UP
    ticker: np.ndarray  # (N,) str
    day: np.ndarray  # (N,) int16
    timestamps: np.ndarray  # (N,) datetime64[ns] -- last bar *inside* each window
    feature_names: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.X)

    @property
    def up_rate(self) -> float:
        return float(self.y.mean()) if len(self.y) else float("nan")


def build_windows(
    bars: np.ndarray,
    window: int = config.SEQUENCE_LENGTH,
    horizon: int = config.HORIZON,
    threshold: float = config.LABEL_THRESHOLD,
    channels: Sequence[str] = config.FEATURE_CHANNELS,
    normalize: str | None = config.NORMALIZE,
) -> WindowSet:
    """Window a tidy bars table, one session at a time.

    Trading days are not contiguous -- day 13 may be a week after day 12 -- so a
    window spanning two sessions would describe a price move that never
    happened. Sessions are therefore windowed independently and concatenated in
    ``(ticker, day)`` order, which is the order the API's own windows come in.
    """
    names = tuple(channels)
    if window < 2:
        raise ValueError(f"window must be at least 2 bars, got {window}")
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1 bar, got {horizon}")

    # Stable sort by (ticker, day, timestamp) -- the API already returns this
    # order, but relying on it silently would be a trap for a future caller.
    order = np.lexsort((bars["timestamp"], bars["day"], bars["ticker"]))
    ordered = bars[order]

    blocks, labels, tickers, days, stamps = [], [], [], [], []

    for start, stop in _session_bounds(ordered):
        session = ordered[start:stop]
        count = window_count(len(session), window, horizon)
        if count == 0:
            continue

        available = derive_channels(session)
        unknown = [n for n in names if n not in available]
        if unknown:
            raise ValueError(f"unknown channels {unknown}; available: {list(available)}")

        matrix = np.column_stack([available[name] for name in names])
        blocks.append(_stack_windows(matrix, window, count))
        labels.append(direction_labels(available["close"], window, horizon, threshold))
        tickers.append(np.full(count, session["ticker"][0]))
        days.append(np.full(count, session["day"][0], dtype=np.int16))
        stamps.append(session["timestamp"][window - 1 : window - 1 + count])

    if not blocks:
        raise ValueError(
            f"no session had enough bars for window={window}, horizon={horizon} "
            f"(a session is 78 bars, so window + horizon must be <= 78)"
        )

    x = normalize_windows(np.concatenate(blocks), normalize).astype(np.float32)
    x = np.ascontiguousarray(x.transpose(0, 2, 1))  # (N, T, C) -> (N, C, T)

    return WindowSet(
        X=x,
        y=np.concatenate(labels),
        ticker=np.concatenate(tickers).astype(str),
        day=np.concatenate(days),
        timestamps=np.concatenate(stamps),
        feature_names=names,
    )


# =====================================================================
# Splits -- by trading day, never random
# =====================================================================


@dataclass(frozen=True)
class Splits:
    """Index arrays into a ``WindowSet``, one per subset."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    train_days: tuple[int, ...]
    val_days: tuple[int, ...]
    test_days: tuple[int, ...]

    def __iter__(self):
        return iter((self.train, self.val, self.test))


def day_splits(
    day: np.ndarray,
    test_days: int = config.TEST_DAYS,
    val_days: int = config.VAL_DAYS,
) -> Splits:
    """Chronological three-way split: the last ``test_days`` days are held out.

    ``test`` uses the same cutoff as the backend's ``subset="test"``, so it is
    the same set of windows the rest of the team evaluates on. ``val`` is the
    ``val_days`` days immediately before that, carved out of what the backend
    calls ``train``.

    Splitting on days rather than on windows is what makes leakage structurally
    impossible: no window straddles the boundary, because no window straddles a
    day.
    """
    day = np.asarray(day)
    if len(day) == 0:
        raise ValueError("cannot split an empty window set")

    last = int(day.max())
    test_cutoff = last - test_days  # days > this are test
    val_cutoff = test_cutoff - val_days  # val_cutoff < day <= test_cutoff is val

    if val_cutoff < int(day.min()):
        raise ValueError(
            f"days {day.min()}-{last} cannot give up {test_days} test + {val_days} val days "
            f"and still leave a training set -- lower config.TEST_DAYS / config.VAL_DAYS"
        )

    is_test = day > test_cutoff
    is_val = (day > val_cutoff) & ~is_test
    is_train = ~is_val & ~is_test

    def days_in(mask: np.ndarray) -> tuple[int, ...]:
        return tuple(int(d) for d in np.unique(day[mask]))

    return Splits(
        train=np.nonzero(is_train)[0],
        val=np.nonzero(is_val)[0],
        test=np.nonzero(is_test)[0],
        train_days=days_in(is_train),
        val_days=days_in(is_val),
        test_days=days_in(is_test),
    )


# =====================================================================
# Loader
# =====================================================================


class StockDataLoader:
    """Serves ``(features, label)`` training examples read from the backend API.

    The whole dataset is 8.4 MB of float32, so it is held in memory and batching
    is pure fancy-indexing -- there is no per-example file I/O to hide.
    """

    def __init__(
        self,
        sequence_length: int = config.SEQUENCE_LENGTH,
        num_input_features: int = config.NUM_INPUT_FEATURES,
        num_outputs: int = config.NUM_OUTPUTS,
        batch_size: int = config.BATCH_SIZE,
        horizon: int = config.HORIZON,
        threshold: float = config.LABEL_THRESHOLD,
        channels: Sequence[str] = config.FEATURE_CHANNELS,
        normalize: str | None = config.NORMALIZE,
        base_url: str | None = None,
        refresh: bool = False,
        seed: int = 0,
        verbose: bool = True,
        windows: WindowSet | None = None,
    ):
        self.sequence_length = sequence_length
        self.num_input_features = num_input_features
        self.num_outputs = num_outputs
        self.batch_size = batch_size
        self.rng = random.Random(seed)

        if windows is None:
            bars = api_client.fetch_bars(base_url=base_url, refresh=refresh, verbose=verbose)
            windows = build_windows(
                bars,
                window=sequence_length,
                horizon=horizon,
                threshold=threshold,
                channels=channels,
                normalize=normalize,
            )
        self.windows = windows

        expected = (num_input_features, sequence_length)
        if self.windows.X.shape[1:] != expected:
            raise ValueError(
                f"built windows of shape {self.windows.X.shape[1:]} but config expects "
                f"{expected} -- config.FEATURE_CHANNELS has {num_input_features} entries"
            )

        # (N, 1) float32: what nn.BCELoss wants next to a sigmoid output.
        self._y = self.windows.y.astype(np.float32).reshape(-1, num_outputs)

    # -- example contract --------------------------------------------

    def __len__(self) -> int:
        return len(self.windows)

    def get_example(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """One ``(x, y)`` pair.

        x : float32 (num_input_features, sequence_length)
        y : float32 (num_outputs,)
        """
        return self.windows.X[index], self._y[index]

    def get_batch(self, indices: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Collate the examples at ``indices`` into stacked float32 tensors.

        Shapes ``(len(indices), num_input_features, sequence_length)`` and
        ``(len(indices), num_outputs)``.
        """
        rows = np.asarray(indices, dtype=np.intp)
        return (
            torch.from_numpy(self.windows.X[rows]),
            torch.from_numpy(self._y[rows]),
        )

    def iter_batches(
        self,
        shuffle: bool = True,
        drop_last: bool = False,
        indices: Sequence[int] | None = None,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yield ``(x, y)`` batches over ``indices`` (all examples by default)."""
        order = list(range(len(self))) if indices is None else [int(i) for i in indices]
        if shuffle:
            self.rng.shuffle(order)

        for start in range(0, len(order), self.batch_size):
            chunk = order[start : start + self.batch_size]
            if drop_last and len(chunk) < self.batch_size:
                break
            yield self.get_batch(chunk)

    def num_batches(self, drop_last: bool = False, num_examples: int | None = None) -> int:
        n = len(self) if num_examples is None else num_examples
        if drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    # -- splitting and reporting -------------------------------------

    def splits(
        self,
        test_days: int = config.TEST_DAYS,
        val_days: int = config.VAL_DAYS,
    ) -> Splits:
        """Train / val / test index arrays, split by trading day.

        There is deliberately no random-split option: windows overlap by
        ``sequence_length - 1`` bars, so a random split scores the model on
        near-copies of its training data.
        """
        split = day_splits(self.windows.day, test_days=test_days, val_days=val_days)
        self.assert_no_leakage(split)
        return split

    def assert_no_leakage(self, split: Splits) -> None:
        """Every training window must end before every validation window, and so on."""
        stamps = self.windows.timestamps
        for earlier, later, names in (
            (split.train, split.val, ("train", "val")),
            (split.val, split.test, ("val", "test")),
        ):
            if len(earlier) == 0 or len(later) == 0:
                continue
            if not stamps[earlier].max() < stamps[later].min():
                raise AssertionError(
                    f"{names[0]} windows overlap {names[1]} in time: "
                    f"{names[0]} ends {stamps[earlier].max()}, "
                    f"{names[1]} starts {stamps[later].min()}"
                )

    def up_rate(self, indices: Sequence[int] | None = None) -> float:
        """Fraction of UP labels -- the majority-class baseline to beat."""
        y = self.windows.y if indices is None else self.windows.y[np.asarray(indices, dtype=np.intp)]
        return float(y.mean()) if len(y) else float("nan")

    def summary(self, split: Splits | None = None) -> str:
        """A printable data card for the run."""
        split = split or self.splits()
        lines = [
            f"{len(self)} windows of shape "
            f"({self.num_input_features}, {self.sequence_length}) "
            f"[{self.windows.X.nbytes / 1024**2:.1f} MB]",
            f"channels: {', '.join(self.windows.feature_names)}",
            f"tickers:  {', '.join(sorted(set(self.windows.ticker.tolist())))}",
            "",
            f"{'subset':<7} {'windows':>8}  {'up rate':>8}  days",
        ]
        for name, idx, days in (
            ("train", split.train, split.train_days),
            ("val", split.val, split.val_days),
            ("test", split.test, split.test_days),
        ):
            span = f"{min(days)}-{max(days)}" if days else "-"
            lines.append(f"{name:<7} {len(idx):>8}  {self.up_rate(idx):>8.4f}  {span}")
        return "\n".join(lines)


# =====================================================================
# Verification against the API's own windows
# =====================================================================


def verify_against_api(
    loader: StockDataLoader | None = None,
    limit: int = 500,
    base_url: str | None = None,
    verbose: bool = True,
) -> dict:
    """Check the local transform reproduces ``/api/v1/dataset/windows``.

    The route returns the *first* ``limit`` windows of the same ``(ticker, day)``
    ordering this file builds in, so the comparison is index for index. It is
    capped at 500, which is why training cannot use the route -- but 500 windows
    across the first sessions is plenty to catch a formula or ordering mistake.

    Returns a report dict; raises ``AssertionError`` when the tensors disagree.
    """
    loader = loader or StockDataLoader(verbose=verbose)
    remote = api_client.dataset_windows_npz(
        limit=limit,
        base_url=base_url,
        window=loader.sequence_length,
        horizon=config.HORIZON,
        threshold=config.LABEL_THRESHOLD,
    )

    n = len(remote["target"])
    # The API serves channels-last (N, T, C); transpose to compare.
    remote_x = np.ascontiguousarray(remote["data"].transpose(0, 2, 1))
    local_x = loader.windows.X[:n]

    max_diff = float(np.abs(local_x - remote_x).max())
    label_mismatches = int((loader.windows.y[:n] != remote["target"]).sum())
    ticker_mismatches = int((loader.windows.ticker[:n] != remote["ticker"]).sum())
    stamp_mismatches = int((loader.windows.timestamps[:n] != remote["timestamps"]).sum())

    report = {
        "compared": n,
        "max_abs_feature_diff": max_diff,
        "label_mismatches": label_mismatches,
        "ticker_mismatches": ticker_mismatches,
        "timestamp_mismatches": stamp_mismatches,
        "feature_names_match": tuple(remote["feature_names"].tolist())
        == loader.windows.feature_names,
    }

    problems = []
    # float32 round-off only; the JSON/npz round-trip is not bit-exact by design.
    if max_diff > 1e-4:
        problems.append(f"features differ by up to {max_diff:.3e}")
    if label_mismatches:
        problems.append(f"{label_mismatches}/{n} labels differ")
    if ticker_mismatches or stamp_mismatches:
        problems.append("metadata is misaligned -- window ordering does not match the API")
    if not report["feature_names_match"]:
        problems.append(f"channel order differs: API says {remote['feature_names'].tolist()}")

    if problems:
        raise AssertionError(
            "local windows disagree with /api/v1/dataset/windows: " + "; ".join(problems)
        )

    if verbose:
        print(
            f"verified {n} windows against /api/v1/dataset/windows: "
            f"labels identical, max |local - api| = {max_diff:.3e}"
        )
    return report


if __name__ == "__main__":
    loader = StockDataLoader()
    split = loader.splits()
    print()
    print(loader.summary(split))
    print()
    verify_against_api(loader)

    x, y = loader.get_batch(split.train[:4])
    print(f"batch shapes: x {tuple(x.shape)} {x.dtype}, y {tuple(y.shape)} {y.dtype}")
    print(f"window 0 per-channel mean {loader.windows.X[0].mean(axis=1).round(3)}")
    print(f"window 0 per-channel std  {loader.windows.X[0].std(axis=1).round(3)}")
