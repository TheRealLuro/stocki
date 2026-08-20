"""Entry point for the training framework.

    python main.py data                   # fetch/inspect the data, verify against the API
    python main.py train  [--epochs 50] [--batch-size 64] [--lr 1e-3] [--fresh]
    python main.py evaluate [--subset test] [--weights checkpoints/best.pt]
    python main.py export [--weights checkpoints/best.pt] [--output artifacts/model.onnx]
"""

from __future__ import annotations

import argparse

import config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="stocki 1D-CNN training framework")
    sub = parser.add_subparsers(dest="command", required=True)

    data_cmd = sub.add_parser(
        "data", help="fetch bars from the API, print the data card, verify the transform"
    )
    data_cmd.add_argument(
        "--refresh", action="store_true", help="ignore the bar cache and refetch from the API"
    )
    data_cmd.add_argument("--api-url", default=None, help=f"default: {config.API_BASE_URL}")
    data_cmd.add_argument("--no-verify", action="store_true")

    train_cmd = sub.add_parser("train", help="train the model (resumes from latest.pt if present)")
    train_cmd.add_argument("--epochs", type=int, default=config.EPOCHS)
    train_cmd.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    train_cmd.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    train_cmd.add_argument("--test-days", type=int, default=config.TEST_DAYS)
    train_cmd.add_argument("--val-days", type=int, default=config.VAL_DAYS)
    train_cmd.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    train_cmd.add_argument(
        "--fresh", action="store_true", help="ignore any checkpoint and start from random weights"
    )
    train_cmd.add_argument(
        "--refresh-data", action="store_true", help="refetch bars from the API before training"
    )

    eval_cmd = sub.add_parser("evaluate", help="score a checkpoint on a held-out subset")
    eval_cmd.add_argument("--weights", default=str(config.BEST_CHECKPOINT))
    eval_cmd.add_argument("--subset", default="test", choices=("train", "val", "test"))
    eval_cmd.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    eval_cmd.add_argument("--test-days", type=int, default=config.TEST_DAYS)
    eval_cmd.add_argument("--val-days", type=int, default=config.VAL_DAYS)
    eval_cmd.add_argument("--device", default=None)

    export_cmd = sub.add_parser("export", help="convert saved weights to an ONNX file")
    export_cmd.add_argument("--weights", default=str(config.BEST_CHECKPOINT))
    export_cmd.add_argument("--output", default=str(config.ARTIFACT_DIR / "model.onnx"))
    export_cmd.add_argument("--sequence-length", type=int, default=config.SEQUENCE_LENGTH)
    export_cmd.add_argument("--opset", type=int, default=config.ONNX_OPSET)
    export_cmd.add_argument("--no-verify", action="store_true")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "data":
        from dataloader import StockDataLoader, verify_against_api

        loader = StockDataLoader(base_url=args.api_url, refresh=args.refresh)
        print()
        print(loader.summary())
        print()
        if not args.no_verify:
            verify_against_api(loader, base_url=args.api_url)

    elif args.command == "train":
        from train import train

        train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            test_days=args.test_days,
            val_days=args.val_days,
            checkpoint_path=None if args.fresh else config.LATEST_CHECKPOINT,
            device=args.device,
            refresh_data=args.refresh_data,
        )

    elif args.command == "evaluate":
        from train import evaluate

        evaluate(
            checkpoint_path=args.weights,
            subset=args.subset,
            batch_size=args.batch_size,
            test_days=args.test_days,
            val_days=args.val_days,
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
