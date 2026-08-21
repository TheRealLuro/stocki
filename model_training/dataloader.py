"""Dataloader for stock training examples.

The batching / shuffling / collation plumbing is finished; the two methods that
depend on the (not yet decided) on-disk data structure are deliberately left as
stubs -- see the "NOT IMPLEMENTED YET" section. Fill those in and the training
loop in train.py runs as-is.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import torch

import config


class StockDataLoader:
    """Serves ``(features, label)`` training examples read from ``data_dir``.

    Example contract
    ----------------
    ``get_example(index)`` returns a tuple ``(x, y)`` where

        x : float32 array of shape (NUM_INPUT_FEATURES, sequence_length)
            a contiguous segment of stock data, channels-first
        y : float32 array of shape (NUM_OUTPUTS,)
            the regression target for that segment

    Every downstream method (``get_batch``, ``iter_batches``) is built on top of
    that single contract, so implementing ``get_example`` + ``_index_examples``
    is enough to make the whole pipeline work.
    """

    def __init__(
        self,
        data_dir: Path | str = config.DATA_DIR,
        sequence_length: int = config.SEQUENCE_LENGTH,
        num_input_features: int = config.NUM_INPUT_FEATURES,
        num_outputs: int = config.NUM_OUTPUTS,
        batch_size: int = config.BATCH_SIZE,
        seed: int = 0,
    ):
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.num_input_features = num_input_features
        self.num_outputs = num_outputs
        self.batch_size = batch_size
        self.rng = random.Random(seed)

        # Flat list of example descriptors (whatever shape ends up being useful:
        # (ticker, file, row_offset) tuples, memmap slices, ...). Populated by
        # _index_examples().
        self._examples: list = []
        self._index_examples()

    # =================================================================
    # NOT IMPLEMENTED YET -- these two depend on the final data structure
    # =================================================================

    def _index_examples(self) -> None:
        """Build a flat list of valid windows from the on-disk CSV sessions.

        Each descriptor is ``(csv_path, start_row)`` for one contiguous window of
        length ``sequence_length``. The actual row materialization happens in
        ``get_example`` so it stays lazy and simple.
        """
        self._examples = []
        if not self.data_dir.exists():
            return

        for csv_path in sorted(self.data_dir.glob("*/day*.csv")):
            try:
                frame = pd.read_csv(csv_path)
            except Exception:
                continue
            if frame.empty:
                continue

            n_rows = len(frame)
            if n_rows < self.sequence_length + 1:
                continue

            for start in range(0, n_rows - self.sequence_length):
                self._examples.append((csv_path, start))

    def get_example(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return one ``(x, y)`` pair for ``self._examples[index]``.

        x is channels-first ``(NUM_INPUT_FEATURES, sequence_length)`` and y is the
        predicted next-close target, matching the current regression model in
        ``config.py``.
        """
        csv_path, start = self._examples[index]
        frame = pd.read_csv(csv_path)

        window = frame.iloc[start : start + self.sequence_length].copy()
        if len(window) != self.sequence_length:
            raise ValueError(f"window starting at {start} in {csv_path} is shorter than required")

        price = window["close"].to_numpy(dtype=np.float64)
        open_ = window["open"].to_numpy(dtype=np.float64)
        high = window["high"].to_numpy(dtype=np.float64)
        low = window["low"].to_numpy(dtype=np.float64)
        volume = window["volume"].to_numpy(dtype=np.float64)

        log_return = np.zeros_like(price, dtype=np.float64)
        if len(price) > 1:
            with np.errstate(divide="ignore", invalid="ignore"):
                log_return[1:] = np.log(price[1:] / price[:-1])
        log_return = np.nan_to_num(log_return, nan=0.0, posinf=0.0, neginf=0.0)

        with np.errstate(divide="ignore", invalid="ignore"):
            hl_range = np.nan_to_num((high - low) / price, nan=0.0, posinf=0.0, neginf=0.0)
            co_range = np.nan_to_num((price - open_) / open_, nan=0.0, posinf=0.0, neginf=0.0)

        features = np.column_stack(
            [
                open_.astype(np.float32),
                high.astype(np.float32),
                low.astype(np.float32),
                price.astype(np.float32),
                volume.astype(np.float32),
                log_return.astype(np.float32),
                hl_range.astype(np.float32),
                co_range.astype(np.float32),
            ]
        )
        x = features.T.astype(np.float32)

        if x.shape != (self.num_input_features, self.sequence_length):
            raise ValueError(
                f"feature shape mismatch for {csv_path} at offset {start}: "
                f"expected {(self.num_input_features, self.sequence_length)}, got {x.shape}"
            )

        future_close = float(frame.iloc[start + self.sequence_length]["close"])
        y = np.full(self.num_outputs, future_close, dtype=np.float32)
        return x, y

    # =================================================================
    # Implemented: batching, shuffling, collation, splitting
    # =================================================================

    def __len__(self) -> int:
        return len(self._examples)

    def get_batch(self, indices: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Collate the examples at ``indices`` into stacked float32 tensors.

        Returns ``(x, y)`` with shapes
        ``(len(indices), num_input_features, sequence_length)`` and
        ``(len(indices), num_outputs)``.
        """
        xs, ys = [], []
        for i in indices:
            x, y = self.get_example(i)
            x = np.asarray(x, dtype=np.float32)
            y = np.asarray(y, dtype=np.float32).reshape(-1)

            if x.shape != (self.num_input_features, self.sequence_length):
                raise ValueError(
                    f"example {i}: expected features of shape "
                    f"{(self.num_input_features, self.sequence_length)}, got {x.shape}"
                )
            if y.shape != (self.num_outputs,):
                raise ValueError(
                    f"example {i}: expected label of shape {(self.num_outputs,)}, got {y.shape}"
                )
            xs.append(x)
            ys.append(y)

        return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))

    def iter_batches(
        self,
        shuffle: bool = True,
        drop_last: bool = False,
        indices: Sequence[int] | None = None,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yield ``(x, y)`` batches over ``indices`` (all examples by default)."""
        order = list(indices) if indices is not None else list(range(len(self)))
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

    def split_indices(self, val_fraction: float = 0.2, shuffle: bool = True) -> tuple[list[int], list[int]]:
        """Split example indices into ``(train_indices, val_indices)``.

        Note: a random split leaks information across overlapping time windows.
        Once ``_index_examples`` exists, prefer splitting by day/ticker here.
        """
        order = list(range(len(self)))
        if shuffle:
            self.rng.shuffle(order)
        cut = int(len(order) * (1.0 - val_fraction))
        return order[:cut], order[cut:]
