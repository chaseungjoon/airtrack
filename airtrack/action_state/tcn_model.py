"""Lightweight Temporal Convolutional Network for TYPING / GESTURE classification.

Architecture: (B, T, feature_dim) → 3 dilated causal conv blocks → linear → 2-class logits.
Exported to TorchScript for CoreML / ANE inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _CausalConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, dilation=dilation, padding=padding)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)
        self._padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self._padding > 0:
            out = out[:, :, : -self._padding]
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        out = self.dropout(torch.relu(out))
        return out + x


class GestureClassifierTCN(nn.Module):
    """Streaming-compatible TCN: TYPING vs GESTURE per 200 ms window.

    Args:
        feature_dim: Input feature vector size (default 7).
        hidden_channels: Conv channel count per block.
        num_blocks: Number of dilated conv blocks.
        dropout: Dropout probability.
    """

    NUM_CLASSES: int = 2  # 0 = TYPING, 1 = GESTURE

    def __init__(
        self,
        feature_dim: int = 7,
        hidden_channels: int = 32,
        num_blocks: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(feature_dim, hidden_channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [_CausalConvBlock(hidden_channels, kernel_size=3, dilation=2 ** i, dropout=dropout)
             for i in range(num_blocks)]
        )
        self.classifier = nn.Linear(hidden_channels, self.NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x: (B, T, feature_dim). Returns: (B, NUM_CLASSES) logits."""
        h = self.input_proj(x.transpose(1, 2))
        for block in self.blocks:
            h = block(h)
        return self.classifier(h.mean(dim=2))

    def export_torchscript(self, path: str) -> None:
        self.eval()
        dummy = torch.zeros(1, 12, 7)
        torch.jit.trace(self, dummy).save(path)  # type: ignore[no-untyped-call]
