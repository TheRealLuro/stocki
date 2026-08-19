"""ONNX Runtime wrapper around the exported 1D-CNN."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import onnxruntime as ort


class ModelNotLoadedError(RuntimeError):
    """Raised when a prediction is requested but the ONNX model failed to load."""


class OnnxPredictor:
    """Loads ``model.onnx`` and runs inference on stock-data segments.

    The graph expects channels-first input ``(batch, features, timesteps)``.
    Callers hand in the more natural row-per-timestep layout
    ``(timesteps, features)`` and this class transposes.
    """

    def __init__(self, model_path: Path | str, providers: Sequence[str] | None = None):
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"no ONNX model at {self.model_path}")

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=list(providers) if providers else ["CPUExecutionProvider"],
        )

        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        self.output_names = [o.name for o in self.session.get_outputs()]

        # Dimensions the exporter left dynamic come back as strings.
        shape = model_input.shape
        self.num_input_features = shape[1] if isinstance(shape[1], int) else None
        self.sequence_length = shape[2] if isinstance(shape[2], int) else None

    # -- helpers -----------------------------------------------------
    def _to_channels_first(self, sequence: Sequence[Sequence[float]], label: str) -> np.ndarray:
        """Validate one (timesteps, features) sequence and return (features, timesteps)."""
        try:
            array = np.asarray(sequence, dtype=np.float32)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"{label}: not a rectangular (timesteps, features) array -- every timestep "
                f"must have the same number of features ({exc})"
            ) from exc

        if array.ndim != 2:
            raise ValueError(f"{label}: expected a 2D array (timesteps, features), got {array.ndim}D")
        if array.size == 0:
            raise ValueError(f"{label}: empty sequence")
        if not np.isfinite(array).all():
            raise ValueError(f"{label}: contains NaN or infinite values")

        timesteps, features = array.shape
        if self.num_input_features is not None and features != self.num_input_features:
            raise ValueError(
                f"{label}: model expects {self.num_input_features} features per timestep, got {features}"
            )
        if self.sequence_length is not None and timesteps != self.sequence_length:
            raise ValueError(
                f"{label}: model expects {self.sequence_length} timesteps, got {timesteps}"
            )

        return array.T  # (features, timesteps)

    # -- inference ---------------------------------------------------
    def predict(self, sequence: Sequence[Sequence[float]]) -> list[float]:
        """Run one segment through the model. Returns the output values."""
        batch = self._to_channels_first(sequence, "sequence")[None, ...]
        return self._run(batch)[0].tolist()

    def predict_many(self, sequences: Sequence[Sequence[Sequence[float]]]) -> list[list[float]]:
        """Run several segments. Returns one list of output values per segment."""
        if not sequences:
            return []

        prepared = [
            self._to_channels_first(seq, f"sequences[{i}]") for i, seq in enumerate(sequences)
        ]

        lengths = {p.shape[1] for p in prepared}
        if len(lengths) == 1:
            return self._run(np.stack(prepared)).tolist()

        # Ragged lengths can't share a batch dimension -- run them one at a time.
        return [self._run(p[None, ...])[0].tolist() for p in prepared]

    def _run(self, batch: np.ndarray) -> np.ndarray:
        outputs = self.session.run(self.output_names, {self.input_name: batch})
        return np.asarray(outputs[0], dtype=np.float32).reshape(batch.shape[0], -1)

    # -- introspection -----------------------------------------------
    def info(self) -> dict:
        return {
            "model_path": str(self.model_path),
            "input_features": self.num_input_features,
            "sequence_length": self.sequence_length or "dynamic",
            "providers": self.session.get_providers(),
        }
