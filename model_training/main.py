"""Entry point for the training framework.

    python main.py train  [--epochs 50] [--batch-size 64] [--lr 1e-3] [--fresh]
    python main.py export [--weights checkpoints/latest.pt] [--output artifacts/model.onnx]
"""

from __future__ import annotations

import argparse

import config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="stocki 1D-CNN training framework")
    sub = parser.add_subparsers(dest="command", required=True)

    train_cmd = sub.add_parser("train", help="train the model (resumes from latest.pt if present)")
    train_cmd.add_argument("--epochs", type=int, default=config.EPOCHS)
    train_cmd.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    train_cmd.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    train_cmd.add_argument("--val-fraction", type=float, default=0.2)
    train_cmd.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    train_cmd.add_argument(
        "--fresh", action="store_true", help="ignore any checkpoint and start from random weights"
    )

    export_cmd = sub.add_parser("export", help="convert saved weights to an ONNX file")
    export_cmd.add_argument("--weights", default=str(config.LATEST_CHECKPOINT))
    export_cmd.add_argument("--output", default=str(config.ARTIFACT_DIR / "model.onnx"))
    export_cmd.add_argument("--sequence-length", type=int, default=config.SEQUENCE_LENGTH)
    export_cmd.add_argument("--opset", type=int, default=config.ONNX_OPSET)
    export_cmd.add_argument("--no-verify", action="store_true")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "train":
        from train import train

        train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            val_fraction=args.val_fraction,
            checkpoint_path=None if args.fresh else config.LATEST_CHECKPOINT,
            device=args.device,
        )
    elif args.command == "export":
        from export_onnx import export_to_onnx

        export_to_onnx(
            weights_path=args.weights,
            output_path=args.output,
            sequence_length=args.sequence_length,
            opset_version=args.opset,
            verify=not args.no_verify,
        )


if __name__ == "__main__":
    main()
