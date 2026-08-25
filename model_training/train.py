# """Training loop.
#
# Resumes from ``config.LATEST_CHECKPOINT`` when it exists, otherwise starts from
# randomly initialized weights. Writes a checkpoint at the end of every epoch:
# ``checkpoints/epoch_XXX.pt`` plus ``checkpoints/latest.pt`` (the resume point),
# and refreshes ``checkpoints/best.pt`` whenever validation loss improves.
#
# Splits come from ``dataloader.StockDataLoader.splits()`` and are by trading day:
# train on days 1-13, validate on 14-16, and leave 17-20 untouched until the run
# is finished. The test set is loaded but never read here -- ``evaluate()`` is a
# separate call, so a test score cannot leak into a training decision by accident.
# """
#
# from __future__ import annotations
#
# from dataclasses import dataclass, field
# from pathlib import Path
#
# import numpy as np
# import torch
# import torch.nn as nn
#
# import config
# from dataloader import StockDataLoader
# from model import StockCNN1D, build_model
#
#
# def pick_device(device: str | None = None) -> torch.device:
#     if device:
#         return torch.device(device)
#     return torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#
# # =====================================================================
# # Metrics
# # =====================================================================
#
#
# @dataclass
# class EpochResult:
#     """Loss plus everything needed to score the pass afterwards."""
#
#     loss: float
#     probs: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
#     targets: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
#
#     def metrics(self, threshold: float = 0.5) -> dict:
#         return {"loss": self.loss, **classification_metrics(self.probs, self.targets, threshold)}
#
#
# def classification_metrics(probs: np.ndarray, targets: np.ndarray, threshold: float = 0.5) -> dict:
#     """Accuracy, precision, recall and F1 for the UP class.
#
#     Precision and recall are 0.0 rather than NaN when the model predicts no
#     positives at all -- a degenerate all-DOWN model should read as bad, not as
#     undefined.
#     """
#     if len(targets) == 0:
#         return {"accuracy": float("nan"), "precision": float("nan"),
#                 "recall": float("nan"), "f1": float("nan")}
#
#     predicted = (probs >= threshold).astype(np.int8)
#     actual = (targets >= 0.5).astype(np.int8)
#
#     true_positive = int(((predicted == 1) & (actual == 1)).sum())
#     false_positive = int(((predicted == 1) & (actual == 0)).sum())
#     false_negative = int(((predicted == 0) & (actual == 1)).sum())
#
#     precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
#     recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
#     f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
#
#     return {
#         "accuracy": float((predicted == actual).mean()),
#         "precision": float(precision),
#         "recall": float(recall),
#         "f1": float(f1),
#         "predicted_up_rate": float(predicted.mean()),
#     }
#
#
# def baseline_metrics(targets: np.ndarray) -> dict:
#     """The majority-class baseline: always guess whichever label is more common.
#
#     This is the number in the README that the model has to beat. On this dataset
#     it sits near 0.50, so anything below ~0.52 is noise.
#     """
#     if len(targets) == 0:
#         return {"accuracy": float("nan"), "majority_class": None}
#     up_rate = float((targets >= 0.5).mean())
#     majority = 1 if up_rate >= 0.5 else 0
#     constant = np.full(len(targets), float(majority), dtype=np.float32)
#     return {
#         "majority_class": majority,
#         **classification_metrics(constant, targets),
#     }
#
#
# # =====================================================================
# # Checkpoints
# # =====================================================================
#
#
# def load_or_create_model(
#     checkpoint_path: Path | str | None = config.LATEST_CHECKPOINT,
#     device: torch.device | str = "cpu",
# ) -> tuple[StockCNN1D, dict]:
#     """Load a model from disk, or create a new one with randomized weights.
#
#     Returns ``(model, checkpoint)`` where ``checkpoint`` is the loaded checkpoint
#     dict (``{}`` when a fresh model was created) so the caller can also restore
#     optimizer state and the epoch counter.
#     """
#     path = Path(checkpoint_path) if checkpoint_path else None
#
#     if path is not None and path.is_file():
#         checkpoint = torch.load(path, map_location="cpu", weights_only=False)
#         model_kwargs = checkpoint.get("model_config", {})
#         model = build_model(**model_kwargs)
#         model.load_state_dict(checkpoint["model_state_dict"])
#         print(f"resumed from {path} (epoch {checkpoint.get('epoch', '?')})")
#     else:
#         checkpoint = {}
#         model = build_model()
#         where = f" (no checkpoint at {path})" if path is not None else ""
#         print(f"created a new model with random weights{where}")
#
#     model.to(device)
#     print(f"model: {model.num_parameters:,} trainable parameters on {device}")
#     return model, checkpoint
#
#
# def save_checkpoint(
#     model: StockCNN1D,
#     optimizer: torch.optim.Optimizer,
#     epoch: int,
#     metrics: dict,
#     checkpoint_dir: Path | None = None,
#     is_best: bool = False,
#     extra: dict | None = None,
# ) -> Path:
#     """Write ``epoch_XXX.pt``, refresh ``latest.pt``, and ``best.pt`` when asked.
#
#     Each file carries the weights, the optimizer state, the epoch number, the
#     architecture config and the data contract, so ``export_onnx.py`` can rebuild
#     the right model from the checkpoint alone.
#     """
#     checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else config.CHECKPOINT_DIR
#     checkpoint_dir.mkdir(parents=True, exist_ok=True)
#     payload = {
#         "epoch": epoch,
#         "model_state_dict": model.state_dict(),
#         "optimizer_state_dict": optimizer.state_dict(),
#         "model_config": model.config_dict(),
#         "sequence_length": config.SEQUENCE_LENGTH,
#         "feature_channels": list(config.FEATURE_CHANNELS),
#         "horizon": config.HORIZON,
#         "normalize": config.NORMALIZE,
#         "metrics": metrics,
#         **(extra or {}),
#     }
#     epoch_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
#     torch.save(payload, epoch_path)
#     torch.save(payload, checkpoint_dir / config.LATEST_CHECKPOINT.name)
#     if is_best:
#         torch.save(payload, checkpoint_dir / config.BEST_CHECKPOINT.name)
#     return epoch_path
#
# # =================
# # loss functions
# # ============
#
# class WeightedBCELoss(nn.Module):
#     """ A Weighted BCE Loss function, where each possibility is weighted by a custom factor, negative or positive cases contribute more or less to the loss
#     """
#
#     def __init__(self, eps: float = 1e-7, positive_weight: float = 1.0, negative_weight: float = 1.0):
#         super().__init__()
#         self.eps = eps
#         self.positive_weight = positive_weight
#         self.negative_weight = negative_weight
#
#     def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
#         predictions = predictions.clamp(self.eps, 1 - self.eps)
#
#         per_example_loss = -(
#             targets * torch.log(predictions) * self.positive_weight + (1 - targets) * torch.log(1 - predictions) * self.negative_weight
#         )
#
#         return per_example_loss.mean()
#
#
# class BalancedBCELoss(nn.Module):
#     """ A Weighted BCE Loss function, where each possibility is weighted by a custom factor, negative or positive cases contribute more or less to the loss
#     """
#
#     def __init__(self, eps: float = 1e-7, balance_factor : float = 0.1, std_eps : float = 1e-7):
#         super().__init__()
#         self.eps = eps
#         self.balance_factor = balance_factor
#         self.std_eps = std_eps
#
#     def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
#         predictions = predictions.clamp(self.eps, 1 - self.eps)
#
#         bce = -(
#             targets * torch.log(predictions) + (1 - targets) * torch.log(1 - predictions)
#         )
#
#         homo_penalty = self.balance_factor / (predictions.std(unbiased=False) + self.std_eps)
#
#         return bce.mean() + homo_penalty
#
# # =====================================================================
# # Passes
# # =====================================================================
#
#
#
#
#
# def run_epoch(
#     model: StockCNN1D,
#     loader: StockDataLoader,
#     loss_fn: nn.Module,
#     device: torch.device,
#     optimizer: torch.optim.Optimizer | None = None,
#     indices=None,
# ) -> EpochResult:
#     """One pass over ``indices``. Trains when ``optimizer`` is given, else evaluates.
#
#     Evaluation passes keep insertion order (no shuffle) so the returned
#     probabilities line up with ``indices``.
#     """
#     training = optimizer is not None
#     model.train(training)
#
#     total_loss, total_examples = 0.0, 0
#     probs, targets = [], []
#
#     with torch.set_grad_enabled(training):
#         for x, y in loader.iter_batches(shuffle=training, indices=indices):
#             x, y = x.to(device), y.to(device)
#
#             predictions = model(x)
#             loss = loss_fn(predictions, y)
#
#             if training:
#                 optimizer.zero_grad(set_to_none=True)
#                 loss.backward()
#                 if config.GRAD_CLIP_NORM:
#                     nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
#                 optimizer.step()
#
#             total_loss += loss.item() * x.shape[0]
#             total_examples += x.shape[0]
#             probs.append(predictions.detach().cpu().numpy().reshape(-1))
#             targets.append(y.detach().cpu().numpy().reshape(-1))
#
#     return EpochResult(
#         loss=total_loss / total_examples if total_examples else float("nan"),
#         probs=np.concatenate(probs) if probs else np.empty(0, dtype=np.float32),
#         targets=np.concatenate(targets) if targets else np.empty(0, dtype=np.float32),
#     )
#
#
# def train(
#     epochs: int = config.EPOCHS,
#     batch_size: int = config.BATCH_SIZE,
#     learning_rate: float = config.LEARNING_RATE,
#     test_days: int = config.TEST_DAYS,
#     val_days: int = config.VAL_DAYS,
#     checkpoint_path: Path | str | None = config.LATEST_CHECKPOINT,
#     device: str | None = None,
#     refresh_data: bool = False,
# ) -> StockCNN1D:
#     device = pick_device(device)
#
#     loader = StockDataLoader(batch_size=batch_size, refresh=refresh_data)
#     split = loader.splits(test_days=test_days, val_days=val_days)
#     print()
#     print(loader.summary(split))
#     print()
#
#     if len(split.train) == 0:
#         raise ValueError(
#             f"the training subset is empty -- {test_days} test + {val_days} val days leaves "
#             f"nothing to fit on"
#         )
#
#     baseline = baseline_metrics(loader.windows.y[split.val].astype(np.float32))
#     if len(split.val):
#         print(
#             f"validation baseline (always predict "
#             f"{'UP' if baseline['majority_class'] else 'DOWN'}): "
#             f"accuracy {baseline['accuracy']:.4f}, f1 {baseline['f1']:.4f}"
#         )
#     else:
#         print("no validation days -- val metrics will be nan and best.pt will not be written")
#
#     model, checkpoint = load_or_create_model(checkpoint_path, device)
#
#     optimizer = torch.optim.Adam(
#         model.parameters(), lr=learning_rate, weight_decay=config.WEIGHT_DECAY
#     )
#     opt_state = checkpoint.get("optimizer_state_dict")
#     if isinstance(opt_state, dict) and opt_state and "param_groups" in opt_state:
#         optimizer.load_state_dict(opt_state)
#
#
#
#     loss_fn : nn.Module = None
#     match config.LOSS_FUNCTION:
#         case 'bce':
#             loss_fn = nn.BCELoss()
#         case 'wbce':
#             loss_fn = WeightedBCELoss(positive_weight=config.POSITIVE_WEIGHT, negative_weight=config.NEGATIVE_WEIGHT)
#         case 'bbce':
#             loss_fn = BalancedBCELoss(balance_factor=config.BALANCE_FACTOR)
#         case _:
#             raise ValueError(f"Unknown loss function {config.LOSS_FUNCTION}")
#     start_epoch = checkpoint.get("epoch", 0) + 1
#     best_val_loss = checkpoint.get("best_val_loss", float("inf"))
#
#     for epoch in range(start_epoch, start_epoch + epochs):
#         train_result = run_epoch(model, loader, loss_fn, device, optimizer, indices=split.train)
#         val_result = run_epoch(model, loader, nn.BCELoss(), device, indices=split.val)
#
#         train_metrics = train_result.metrics()
#         val_metrics = val_result.metrics(threshold=config.CLASS_THRESHOLD)
#
#         is_best = val_metrics["loss"] < best_val_loss
#         best_val_loss = min(best_val_loss, val_metrics["loss"])
#
#         saved = save_checkpoint(
#             model,
#             optimizer,
#             epoch,
#             {"train": train_metrics, "val": val_metrics, "val_baseline": baseline},
#             is_best=is_best,
#             extra={"best_val_loss": best_val_loss},
#         )
#         print(
#             f"epoch {epoch:3d} | train loss {train_metrics['loss']:.5f} acc {train_metrics['accuracy']:.4f}"
#             f" | val loss {val_metrics['loss']:.5f} acc {val_metrics['accuracy']:.4f}"
#             f" f1 {val_metrics['f1']:.4f} | {saved.name}{'  <- best' if is_best else ''}"
#         )
#
#     return model
#
#
# def evaluate(
#     checkpoint_path: Path | str = config.BEST_CHECKPOINT,
#     subset: str = "test",
#     batch_size: int = config.BATCH_SIZE,
#     test_days: int = config.TEST_DAYS,
#     val_days: int = config.VAL_DAYS,
#     device: str | None = None,
# ) -> dict:
#     """Score a saved checkpoint on a held-out subset, against the baseline.
#
#     Run this once, at the end. Tuning against the test set turns it into a
#     second validation set and the reported number stops meaning anything.
#     """
#     device = pick_device(device)
#     loader = StockDataLoader(batch_size=batch_size)
#     split = loader.splits(test_days=test_days, val_days=val_days)
#     indices = {"train": split.train, "val": split.val, "test": split.test}[subset]
#     if len(indices) == 0:
#         raise ValueError(
#             f"the {subset} subset is empty -- check config.TEST_DAYS / config.VAL_DAYS"
#         )
#
#     model, _ = load_or_create_model(checkpoint_path, device)
#     loss_fn = nn.BCELoss()
#     result = run_epoch(model, loader, loss_fn, device, indices=indices)
#
#     metrics = result.metrics(threshold=config.CLASS_THRESHOLD)
#     baseline = baseline_metrics(result.targets)
#     days = {"train": split.train_days, "val": split.val_days, "test": split.test_days}[subset]
#
#     print(f"\n{subset} ({len(indices)} windows, days {min(days)}-{max(days)})")
#     print(f"  loss       {metrics['loss']:.5f}")
#     for name in ("accuracy", "precision", "recall", "f1"):
#         print(f"  {name:<10} {metrics[name]:.4f}   baseline {baseline[name]:.4f}")
#     print(f"  predicted UP on {metrics['predicted_up_rate']:.1%} of windows "
#           f"(actual {float((result.targets >= 0.5).mean()):.1%})")
#
#     return {"metrics": metrics, "baseline": baseline, "subset": subset, "n": len(indices)}
#
#
# if __name__ == "__main__":
#     train()



"""Training loop.

Resumes from ``config.LATEST_CHECKPOINT`` when it exists, otherwise starts from
randomly initialized weights. Writes a checkpoint at the end of every epoch:
``checkpoints/epoch_XXX.pt`` plus ``checkpoints/latest.pt`` (the resume point),
and refreshes ``checkpoints/best.pt`` whenever validation loss improves.

Splits come from ``dataloader.StockDataLoader.splits()`` and are by trading day:
train on days 1-13, validate on 14-16, and leave 17-20 untouched until the run
is finished. The test set is loaded but never read here -- ``evaluate()`` is a
separate call, so a test score cannot leak into a training decision by accident.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import config
from dataloader import StockDataLoader
from model import StockCNN1D, build_model

try:
    import onnxruntime as ort
except ImportError:  # onnxruntime is only required if you actually evaluate .onnx files
    ort = None


def pick_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================================
# Metrics
# =====================================================================


@dataclass
class EpochResult:
    """Loss plus everything needed to score the pass afterwards."""

    loss: float
    probs: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    targets: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))

    def metrics(self, threshold: float = 0.5) -> dict:
        return {"loss": self.loss, **classification_metrics(self.probs, self.targets, threshold)}


def classification_metrics(probs: np.ndarray, targets: np.ndarray, threshold: float = 0.5) -> dict:
    """Accuracy, precision, recall and F1 for the UP class.

    Precision and recall are 0.0 rather than NaN when the model predicts no
    positives at all -- a degenerate all-DOWN model should read as bad, not as
    undefined.
    """
    if len(targets) == 0:
        return {"accuracy": float("nan"), "precision": float("nan"),
                "recall": float("nan"), "f1": float("nan")}

    predicted = (probs >= threshold).astype(np.int8)
    actual = (targets >= 0.5).astype(np.int8)

    true_positive = int(((predicted == 1) & (actual == 1)).sum())
    false_positive = int(((predicted == 1) & (actual == 0)).sum())
    false_negative = int(((predicted == 0) & (actual == 1)).sum())

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "accuracy": float((predicted == actual).mean()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "predicted_up_rate": float(predicted.mean()),
    }


def baseline_metrics(targets: np.ndarray) -> dict:
    """The majority-class baseline: always guess whichever label is more common.

    This is the number in the README that the model has to beat. On this dataset
    it sits near 0.50, so anything below ~0.52 is noise.
    """
    if len(targets) == 0:
        return {"accuracy": float("nan"), "majority_class": None}
    up_rate = float((targets >= 0.5).mean())
    majority = 1 if up_rate >= 0.5 else 0
    constant = np.full(len(targets), float(majority), dtype=np.float32)
    return {
        "majority_class": majority,
        **classification_metrics(constant, targets),
    }


# =====================================================================
# Checkpoints
# =====================================================================


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
    is_best: bool = False,
    extra: dict | None = None,
) -> Path:
    """Write ``epoch_XXX.pt``, refresh ``latest.pt``, and ``best.pt`` when asked.

    Each file carries the weights, the optimizer state, the epoch number, the
    architecture config and the data contract, so ``export_onnx.py`` can rebuild
    the right model from the checkpoint alone.
    """
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else config.CHECKPOINT_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": model.config_dict(),
        "sequence_length": config.SEQUENCE_LENGTH,
        "feature_channels": list(config.FEATURE_CHANNELS),
        "horizon": config.HORIZON,
        "normalize": config.NORMALIZE,
        "metrics": metrics,
        **(extra or {}),
    }
    epoch_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
    torch.save(payload, epoch_path)
    torch.save(payload, checkpoint_dir / config.LATEST_CHECKPOINT.name)
    if is_best:
        torch.save(payload, checkpoint_dir / config.BEST_CHECKPOINT.name)
    return epoch_path


def load_onnx_session(path: Path | str) -> "ort.InferenceSession":
    """Load an .onnx file into an onnxruntime inference session.

    Raises a clear error if onnxruntime isn't installed rather than failing
    on a confusing AttributeError later.
    """
    if ort is None:
        raise ImportError(
            "onnxruntime is required to evaluate .onnx checkpoints -- "
            "install it with `pip install onnxruntime` (or `onnxruntime-gpu`)"
        )
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no .onnx file at {path}")
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    print(f"loaded onnx model from {path}")
    return session


# =================
# loss functions
# ============

class WeightedBCELoss(nn.Module):
    """ A Weighted BCE Loss function, where each possibility is weighted by a custom factor, negative or positive cases contribute more or less to the loss
    """

    def __init__(self, eps: float = 1e-7, positive_weight: float = 1.0, negative_weight: float = 1.0):
        super().__init__()
        self.eps = eps
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predictions = predictions.clamp(self.eps, 1 - self.eps)

        per_example_loss = -(
            targets * torch.log(predictions) * self.positive_weight + (1 - targets) * torch.log(1 - predictions) * self.negative_weight
        )

        return per_example_loss.mean()


class BalancedBCELoss(nn.Module):
    """ A Weighted BCE Loss function, where each possibility is weighted by a custom factor, negative or positive cases contribute more or less to the loss
    """

    def __init__(self, eps: float = 1e-7, balance_factor : float = 0.1, std_eps : float = 1e-7):
        super().__init__()
        self.eps = eps
        self.balance_factor = balance_factor
        self.std_eps = std_eps

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predictions = predictions.clamp(self.eps, 1 - self.eps)

        bce = -(
            targets * torch.log(predictions) + (1 - targets) * torch.log(1 - predictions)
        )

        homo_penalty = self.balance_factor / (predictions.std(unbiased=False) + self.std_eps)

        return bce.mean() + homo_penalty


def _numpy_bce(probs: np.ndarray, targets: np.ndarray, eps: float = 1e-7) -> float:
    """Plain-numpy BCE, used for the onnx eval path where there's no nn.Module loss."""
    if len(targets) == 0:
        return float("nan")
    p = np.clip(probs, eps, 1 - eps)
    return float(-(targets * np.log(p) + (1 - targets) * np.log(1 - p)).mean())


# =====================================================================
# Passes
# =====================================================================


def run_epoch(
    model: StockCNN1D,
    loader: StockDataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    indices=None,
) -> EpochResult:
    """One pass over ``indices``. Trains when ``optimizer`` is given, else evaluates.

    Evaluation passes keep insertion order (no shuffle) so the returned
    probabilities line up with ``indices``.
    """
    training = optimizer is not None
    model.train(training)

    total_loss, total_examples = 0.0, 0
    probs, targets = [], []

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
            probs.append(predictions.detach().cpu().numpy().reshape(-1))
            targets.append(y.detach().cpu().numpy().reshape(-1))

    return EpochResult(
        loss=total_loss / total_examples if total_examples else float("nan"),
        probs=np.concatenate(probs) if probs else np.empty(0, dtype=np.float32),
        targets=np.concatenate(targets) if targets else np.empty(0, dtype=np.float32),
    )


def run_epoch_onnx(
    session: "ort.InferenceSession",
    loader: StockDataLoader,
    indices=None,
) -> EpochResult:
    """Same contract as ``run_epoch``, but scores a loaded ONNX session instead
    of a torch model. Always an evaluation pass -- there's no training an
    exported graph -- so it always keeps insertion order (no shuffle).

    Assumes the ONNX graph's single output is already a sigmoid probability,
    same as the torch model (it's fed straight into BCELoss elsewhere), so
    loss is computed with plain-numpy BCE rather than nn.BCELoss.
    """
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    probs, targets = [], []
    for x, y in loader.iter_batches(shuffle=False, indices=indices):
        x_np = x.detach().cpu().numpy().astype(np.float32)
        y_np = y.detach().cpu().numpy().reshape(-1)

        outputs = session.run([output_name], {input_name: x_np})[0]

        probs.append(np.asarray(outputs).reshape(-1))
        targets.append(y_np)

    probs = np.concatenate(probs) if probs else np.empty(0, dtype=np.float32)
    targets = np.concatenate(targets) if targets else np.empty(0, dtype=np.float32)

    return EpochResult(loss=_numpy_bce(probs, targets), probs=probs, targets=targets)


def train(
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.LEARNING_RATE,
    test_days: int = config.TEST_DAYS,
    val_days: int = config.VAL_DAYS,
    checkpoint_path: Path | str | None = config.LATEST_CHECKPOINT,
    device: str | None = None,
    refresh_data: bool = False,
) -> StockCNN1D:
    device = pick_device(device)

    loader = StockDataLoader(batch_size=batch_size, refresh=refresh_data)
    split = loader.splits(test_days=test_days, val_days=val_days)
    print()
    print(loader.summary(split))
    print()

    if len(split.train) == 0:
        raise ValueError(
            f"the training subset is empty -- {test_days} test + {val_days} val days leaves "
            f"nothing to fit on"
        )

    baseline = baseline_metrics(loader.windows.y[split.val].astype(np.float32))
    if len(split.val):
        print(
            f"validation baseline (always predict "
            f"{'UP' if baseline['majority_class'] else 'DOWN'}): "
            f"accuracy {baseline['accuracy']:.4f}, f1 {baseline['f1']:.4f}"
        )
    else:
        print("no validation days -- val metrics will be nan and best.pt will not be written")

    model, checkpoint = load_or_create_model(checkpoint_path, device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=config.WEIGHT_DECAY
    )
    opt_state = checkpoint.get("optimizer_state_dict")
    if isinstance(opt_state, dict) and opt_state and "param_groups" in opt_state:
        optimizer.load_state_dict(opt_state)



    loss_fn : nn.Module = None
    match config.LOSS_FUNCTION:
        case 'bce':
            loss_fn = nn.BCELoss()
        case 'wbce':
            loss_fn = WeightedBCELoss(positive_weight=config.POSITIVE_WEIGHT, negative_weight=config.NEGATIVE_WEIGHT)
        case 'bbce':
            loss_fn = BalancedBCELoss(balance_factor=config.BALANCE_FACTOR)
        case _:
            raise ValueError(f"Unknown loss function {config.LOSS_FUNCTION}")
    start_epoch = checkpoint.get("epoch", 0) + 1
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))

    for epoch in range(start_epoch, start_epoch + epochs):
        train_result = run_epoch(model, loader, loss_fn, device, optimizer, indices=split.train)
        val_result = run_epoch(model, loader, nn.BCELoss(), device, indices=split.val)

        train_metrics = train_result.metrics()
        val_metrics = val_result.metrics(threshold=config.CLASS_THRESHOLD)

        is_best = val_metrics["loss"] < best_val_loss
        best_val_loss = min(best_val_loss, val_metrics["loss"])

        saved = save_checkpoint(
            model,
            optimizer,
            epoch,
            {"train": train_metrics, "val": val_metrics, "val_baseline": baseline},
            is_best=is_best,
            extra={"best_val_loss": best_val_loss},
        )
        print(
            f"epoch {epoch:3d} | train loss {train_metrics['loss']:.5f} acc {train_metrics['accuracy']:.4f}"
            f" | val loss {val_metrics['loss']:.5f} acc {val_metrics['accuracy']:.4f}"
            f" f1 {val_metrics['f1']:.4f} | {saved.name}{'  <- best' if is_best else ''}"
        )

    return model


def evaluate(
    checkpoint_path: Path | str = config.BEST_CHECKPOINT,
    subset: str = "test",
    batch_size: int = config.BATCH_SIZE,
    test_days: int = config.TEST_DAYS,
    val_days: int = config.VAL_DAYS,
    device: str | None = None,
) -> dict:
    """Score a saved checkpoint on a held-out subset, against the baseline.

    Accepts either a PyTorch checkpoint (``.pt``) or an exported ONNX graph
    (``.onnx``) -- the file extension picks the branch. Both paths run the
    exact same splits/baseline/printing so the numbers are directly comparable,
    e.g. to confirm an ONNX export didn't drift from the checkpoint it came from.

    Run this once, at the end. Tuning against the test set turns it into a
    second validation set and the reported number stops meaning anything.
    """
    device = pick_device(device)
    loader = StockDataLoader(batch_size=batch_size)
    split = loader.splits(test_days=test_days, val_days=val_days)
    indices = {"train": split.train, "val": split.val, "test": split.test}[subset]
    if len(indices) == 0:
        raise ValueError(
            f"the {subset} subset is empty -- check config.TEST_DAYS / config.VAL_DAYS"
        )

    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.suffix == ".onnx":
        session = load_onnx_session(checkpoint_path)
        result = run_epoch_onnx(session, loader, indices=indices)
    else:
        model, _ = load_or_create_model(checkpoint_path, device)
        loss_fn = nn.BCELoss()
        result = run_epoch(model, loader, loss_fn, device, indices=indices)

    metrics = result.metrics(threshold=config.CLASS_THRESHOLD)
    baseline = baseline_metrics(result.targets)
    days = {"train": split.train_days, "val": split.val_days, "test": split.test_days}[subset]

    print(f"\n{subset} ({len(indices)} windows, days {min(days)}-{max(days)})")
    print(f"  loss       {metrics['loss']:.5f}")
    for name in ("accuracy", "precision", "recall", "f1"):
        print(f"  {name:<10} {metrics[name]:.4f}   baseline {baseline[name]:.4f}")
    print(f"  predicted UP on {metrics['predicted_up_rate']:.1%} of windows "
          f"(actual {float((result.targets >= 0.5).mean()):.1%})")

    return {"metrics": metrics, "baseline": baseline, "subset": subset, "n": len(indices)}


if __name__ == "__main__":
    train()
