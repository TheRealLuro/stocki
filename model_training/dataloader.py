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
        """Build ``self._examples``: one descriptor per available training example.

        Called once from ``__init__``. A descriptor only has to carry enough
        information for ``get_example`` to materialize the segment -- e.g.
        ``(csv_path, start_row)`` for every valid window in every day file.

        TODO: implement once the data structure is decided.
        """
        self._examples = []

    def get_example(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return one ``(x, y)`` pair for ``self._examples[index]``.

        Returns
        -------
        x : np.ndarray, shape (num_input_features, sequence_length), float32
        y : np.ndarray, shape (num_outputs,), float32

        TODO: implement once the data structure is decided. Notes for whoever
        fills this in:
          * The CSVs are channels-last (one row per timestep), so the feature
            block needs a transpose before it is returned.
          * Normalization/scaling belongs here (or in a helper called from
            here) so it applies identically at train and inference time.
        """
        raise NotImplementedError(
            "StockDataLoader.get_example is not implemented yet -- the data structure "
            "has not been decided. See the docstring for the expected return shapes."
        )

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
