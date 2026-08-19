"""Base model: a 1D CNN over a contiguous segment of stock data.

Tensor layout
-------------
    input : (batch, NUM_INPUT_FEATURES, sequence_length)   <- channels-first
    output: (batch, NUM_OUTPUTS)

The network pools over the time axis with an adaptive pool, so a trained model
accepts any sequence length >= the receptive field, not just the length it was
trained on. `sequence_length` is therefore a dynamic axis in the ONNX export.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import config


class ConvBlock(nn.Module):
    """Conv1d -> BatchNorm -> GELU -> Dropout, with length-preserving padding."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,  # 'same' length for odd kernel sizes
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.norm(self.conv(x))))


class StockCNN1D(nn.Module):
    """1D convolutional regressor for stock-price prediction.

    Parameters
    ----------
    num_input_features:
        Features per timestep == Conv1d input channels. Defaults to
        ``config.NUM_INPUT_FEATURES`` -- change it there, not here.
    """

    def __init__(
        self,
        num_input_features: int = config.NUM_INPUT_FEATURES,
        num_outputs: int = config.NUM_OUTPUTS,
        conv_channels: tuple[int, ...] = config.CONV_CHANNELS,
        kernel_size: int = config.KERNEL_SIZE,
        dropout: float = config.DROPOUT,
    ):
        super().__init__()
        self.num_input_features = num_input_features
        self.num_outputs = num_outputs

        blocks = []
        in_ch = num_input_features
        for out_ch in conv_channels:
            blocks.append(ConvBlock(in_ch, out_ch, kernel_size, dropout))
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)

        # Mean + max pooling over time -> fixed-size representation.
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.head = nn.Sequential(
            nn.Linear(in_ch * 2, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Skipped while tracing so the ONNX exporter doesn't bake the checks in
        # as constants (and warn about it).
        if not torch.jit.is_tracing():
            if x.dim() != 3:
                raise ValueError(
                    f"expected a 3D tensor (batch, features, time), got shape {tuple(x.shape)}"
                )
            if x.shape[1] != self.num_input_features:
                raise ValueError(
                    f"expected {self.num_input_features} input features on dim 1, got {x.shape[1]}"
                )

        h = self.features(x)
        pooled = torch.cat([self.avg_pool(h), self.max_pool(h)], dim=1).flatten(1)
        return self.head(pooled)

    # -- convenience -------------------------------------------------
    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def config_dict(self) -> dict:
        """Architecture description stored alongside the weights in checkpoints."""
        return {
            "num_input_features": self.num_input_features,
            "num_outputs": self.num_outputs,
            "conv_channels": [b.conv.out_channels for b in self.features],
            "kernel_size": self.features[0].conv.kernel_size[0],
            "dropout": self.features[0].drop.p,
        }


def build_model(**overrides) -> StockCNN1D:
    """Create a freshly initialized model. Keyword args override config defaults."""
    return StockCNN1D(**overrides)


if __name__ == "__main__":
    m = build_model()
    dummy = torch.randn(4, config.NUM_INPUT_FEATURES, config.SEQUENCE_LENGTH)
    print(m)
    print("parameters:", f"{m.num_parameters:,}")
    print("output shape:", tuple(m(dummy).shape))
