"""Lightweight Temporal Convolutional Network for TYPING / GESTURE classification.

Architecture (input → output):
  (B, FEATURE_DIM, T) → 3 × dilated causal conv blocks → linear → 2-class logits

Design rationale:
  - TCN over LSTM: fully parallelisable, no hidden state, fixed receptive field
    that exactly covers our 200 ms window at 60 fps.
  - Causal convolutions (no future leakage) so the model is streaming-compatible.
  - Dilation [1, 2, 4] gives receptive field = 1+(kernel-1)*(1+2+4) = 15 frames
    at kernel_size=3, which comfortably spans 200 ms at 60 fps.
  - Exported to TorchScript / CoreML for <10 ms ANE inference.

Training approach (supervised):
  Phase 1 — Bootstrap:
    The heuristic classifier labels "clearly typing" and "clearly gesture"
    windows from a 10-minute recording session (keystroke rate > 4 Hz or
    velocity > 15 px/frame). Ambiguous frames are discarded from training.
  Phase 2 — Correction fine-tuning (online SGD):
    User presses a configurable hot-key ("correction trigger") to mark the
    previous 200 ms as mislabelled. The corrected sample is added to a
    local replay buffer and one SGD step is taken per correction.
  Why not RL:
    The reward signal (did the cursor go where intended?) is hard to observe
    automatically and introduces significant delay. Supervised learning with
    an explicit correction mechanism gives faster, more stable convergence for
    a binary classification task with near-instantaneous ground truth.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _CausalConvBlock(nn.Module):
    """Single dilated causal convolution + LayerNorm + ReLU + residual.

    Args:
        channels: Number of input and output channels.
        kernel_size: Convolution kernel size.
        dilation: Dilation factor.
        dropout: Dropout probability applied after activation.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation  # causal: pad only the left
        self.conv = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)
        self._padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (B, C, T) tensor.
        Returns:
            (B, C, T) tensor with residual added.
        """
        out = self.conv(x)
        # Remove the right-side padding introduced for causality
        if self._padding > 0:
            out = out[:, :, : -self._padding]
        # LayerNorm expects (B, T, C)
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        out = self.dropout(torch.relu(out))
        return out + x  # residual


class GestureClassifierTCN(nn.Module):
    """Streaming-compatible TCN that classifies TYPING vs GESTURE per window.

    Args:
        feature_dim: Dimensionality of the input feature vector (default 7).
        hidden_channels: Number of conv channels in each block.
        num_blocks: Number of dilated conv blocks (dilation doubles each block).
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
            [
                _CausalConvBlock(
                    hidden_channels,
                    kernel_size=3,
                    dilation=2 ** i,
                    dropout=dropout,
                )
                for i in range(num_blocks)
            ]
        )
        self.classifier = nn.Linear(hidden_channels, self.NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify a batch of feature sequences.

        Args:
            x: (B, T, feature_dim) float32 tensor.

        Returns:
            (B, NUM_CLASSES) logits tensor.
        """
        # Conv1d expects (B, C, T)
        h = self.input_proj(x.transpose(1, 2))
        for block in self.blocks:
            h = block(h)
        # Mean-pool over time → (B, C)
        h = h.mean(dim=2)
        return self.classifier(h)

    def export_torchscript(self, path: str) -> None:
        """Trace and save the model as TorchScript for CoreML conversion.

        Args:
            path: Output .pt file path.
        """
        self.eval()
        dummy = torch.zeros(1, 12, 7)  # batch=1, T=12, feature_dim=7
        traced = torch.jit.trace(self, dummy)
        traced.save(path)
