"""Convert a saved checkpoint / state_dict into an ONNX Runtime file.

    python main.py export --weights checkpoints/latest.pt --output artifacts/model.onnx

Requires the ``onnx`` package (``pip install onnx``); ``onnxruntime`` is only
needed for the optional verification pass.
"""

from __future__ import annotations

from pathlib import Path

import torch

import config
from model import StockCNN1D, build_model


def load_model_for_export(weights_path: Path | str) -> StockCNN1D:
    """Build a model and load ``weights_path`` into it.

    Accepts either a training checkpoint from ``train.save_checkpoint`` (which
    carries the architecture in ``model_config``) or a bare ``state_dict``.
    """
    weights_path = Path(weights_path)
    if not weights_path.is_file():
        raise FileNotFoundError(f"no weights at {weights_path}")

    blob = torch.load(weights_path, map_location="cpu", weights_only=False)

    if isinstance(blob, dict) and "model_state_dict" in blob:
        state_dict = blob["model_state_dict"]
        model = build_model(**blob.get("model_config", {}))
    else:  # bare state_dict -> architecture comes from config.py
        state_dict = blob
        model = build_model()

    model.load_state_dict(state_dict)
    model.eval()
    return model


def export_to_onnx(
    weights_path: Path | str = config.LATEST_CHECKPOINT,
    output_path: Path | str = config.ARTIFACT_DIR / "model.onnx",
    sequence_length: int = config.SEQUENCE_LENGTH,
    opset_version: int = config.ONNX_OPSET,
    verify: bool = True,
) -> Path:
    """Export the weights at ``weights_path`` to an ONNX file at ``output_path``.

    The exported graph has dynamic ``batch`` and ``sequence_length`` axes:
        input  "input"  : (batch, num_input_features, sequence_length)
        output "output" : (batch, num_outputs)
    """
    model = load_model_for_export(weights_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(1, model.num_input_features, sequence_length)

    torch.onnx.export(
        model,
        (dummy_input,),
        str(output_path),
        dynamo=False,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 2: "sequence_length"},
            "output": {0: "batch"},
        },
    )
    print(f"exported {weights_path} -> {output_path}")

    if verify:
        _verify_export(output_path, model, dummy_input)
    return output_path


def _verify_export(onnx_path: Path, model: StockCNN1D, dummy_input: torch.Tensor) -> None:
    """Run the ONNX graph and compare against the PyTorch output."""
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        print("skipped verification (onnxruntime not installed)")
        return

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {"input": dummy_input.numpy()})[0]
    with torch.no_grad():
        torch_out = model(dummy_input).numpy()

    max_diff = float(np.abs(onnx_out - torch_out).max())
    print(f"verified: output shape {onnx_out.shape}, max |onnx - torch| = {max_diff:.3e}")
    if max_diff > 1e-4:
        print("WARNING: outputs diverge more than expected")


if __name__ == "__main__":
    export_to_onnx()
