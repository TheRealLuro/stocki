"""Training loop.

Resumes from ``config.LATEST_CHECKPOINT`` when it exists, otherwise starts from
randomly initialized weights. Writes a checkpoint at the end of every epoch:
``checkpoints/epoch_XXX.pt`` plus ``checkpoints/latest.pt`` (the resume point).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

import config
from dataloader import StockDataLoader
from model import StockCNN1D, build_model


def pick_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_or_create_model(
    checkpoint_path: Path | str | None = config.LATEST_CHECKPOINT,
    device: torch.device | str = "cpu",
) -> tuple[StockCNN1D, dict]:
    """Load a model from disk, or create a new one with randomized weights.

    Returns ``(model, checkpoint)`` where ``checkpoint`` is the loaded checkpoint
    dict (``{}`` when a fresh model was created) so the caller can also restore
    optimizer state and the epoch counter.
    """
    path = Path(checkpoint_path) if checkpoint_path else None

    if path is not None and path.is_file():
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model_kwargs = checkpoint.get("model_config", {})
        model = build_model(**model_kwargs)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"resumed from {path} (epoch {checkpoint.get('epoch', '?')})")
    else:
        checkpoint = {}
        model = build_model()
        where = f" (no checkpoint at {path})" if path is not None else ""
        print(f"created a new model with random weights{where}")

    model.to(device)
    print(f"model: {model.num_parameters:,} trainable parameters on {device}")
    return model, checkpoint


def save_checkpoint(
    model: StockCNN1D,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    checkpoint_dir: Path | None = None,
) -> Path:
    """Write ``epoch_XXX.pt`` and refresh ``latest.pt``. Returns the epoch file."""
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else config.CHECKPOINT_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": model.config_dict(),
        "sequence_length": config.SEQUENCE_LENGTH,
        "metrics": metrics,
    }
    epoch_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
    torch.save(payload, epoch_path)
    torch.save(payload, checkpoint_dir / config.LATEST_CHECKPOINT.name)
    return epoch_path


def run_epoch(
    model: StockCNN1D,
    loader: StockDataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    indices=None,
) -> float:
    """One pass over ``indices``. Trains when ``optimizer`` is given, else evaluates.

    Returns the example-weighted mean loss.
    """
    training = optimizer is not None
    model.train(training)

    total_loss, total_examples = 0.0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader.iter_batches(shuffle=training, indices=indices):
            x, y = x.to(device), y.to(device)

            predictions = model(x)
            loss = loss_fn(predictions, y)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if config.GRAD_CLIP_NORM:
                    nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
                optimizer.step()

            total_loss += loss.item() * x.shape[0]
            total_examples += x.shape[0]

    return total_loss / total_examples if total_examples else float("nan")


def train(
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.LEARNING_RATE,
    val_fraction: float = 0.2,
    checkpoint_path: Path | str | None = config.LATEST_CHECKPOINT,
    device: str | None = None,
) -> StockCNN1D:
    device = pick_device(device)

    loader = StockDataLoader(batch_size=batch_size)
    if len(loader) == 0:
        raise RuntimeError(
            "the dataloader indexed 0 examples -- implement "
            "StockDataLoader._index_examples() and .get_example() in dataloader.py first"
        )

    train_idx, val_idx = loader.split_indices(val_fraction=val_fraction)
    print(f"{len(loader)} examples -> {len(train_idx)} train / {len(val_idx)} val")

    model, checkpoint = load_or_create_model(checkpoint_path, device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=config.WEIGHT_DECAY
    )
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    loss_fn = nn.BCELoss() # Loss function for binary classification task with Sigmoid output
    start_epoch = checkpoint.get("epoch", 0) + 1

    for epoch in range(start_epoch, start_epoch + epochs):
        train_loss = run_epoch(model, loader, loss_fn, device, optimizer, indices=train_idx)
        val_loss = (
            run_epoch(model, loader, loss_fn, device, indices=val_idx) if val_idx else float("nan")
        )

        saved = save_checkpoint(
            model, optimizer, epoch, {"train_loss": train_loss, "val_loss": val_loss}
        )
        print(
            f"epoch {epoch:3d} | train {train_loss:.6f} | val {val_loss:.6f} | saved {saved.name}"
        )

    return model


if __name__ == "__main__":
    train()
