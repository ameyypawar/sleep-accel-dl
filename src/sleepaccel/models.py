"""The network: per-epoch CNN encoder, attention pooling, sequence context.

Shape of the idea. A 30-second epoch of triaxial acceleration goes through a
small 1D CNN, which learns local motion micropatterns. Attention pooling
collapses the time axis to one embedding per epoch, weighting the moments that
matter rather than averaging them flat. A BiLSTM then reads the sequence of
epoch embeddings, which is the part that separates Deep from REM: both are
near-motionless in isolation, and only their position in the night's cycle
tells them apart.

``context_model="none"`` exists to demonstrate exactly that. It is the ablation
that shows the sequence model is doing the work, not decoration.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data.hr_features import N_HR_FEATURES
from .data.labels import INVALID, N_CLASSES

#: HR features plus the validity flag, so "missing" is distinguishable from 0 bpm.
HR_INPUT_DIM = N_HR_FEATURES + 1


def select_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


class ConvEncoder(nn.Module):
    """1D CNN over one epoch's waveform."""

    def __init__(self, in_channels: int = 4, widths=(32, 64, 128), dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_channels
        for width in widths:
            layers += [
                nn.Conv1d(prev, width, kernel_size=7, padding=3),
                nn.BatchNorm1d(width),
                nn.GELU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout),
            ]
            prev = width
        self.net = nn.Sequential(*layers)
        self.out_dim = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionPool(nn.Module):
    """Additive attention over the time axis, with a learned query."""

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, D, T] -> [B, T, D]
        x = x.transpose(1, 2)
        weights = torch.softmax(self.score(x).squeeze(-1), dim=1)
        pooled = (x * weights.unsqueeze(-1)).sum(dim=1)
        return pooled, weights


class SleepStagingNet(nn.Module):
    """Full model. ``input_variant`` selects which branches even exist."""

    def __init__(
        self,
        input_variant: str = "accel_only",
        context_model: str = "bilstm",
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_variant = input_variant
        self.context_model = context_model

        embed_dim = 0
        self.encoder = None
        self.pool = None
        if input_variant in ("accel_only", "accel_hr"):
            self.encoder = ConvEncoder(in_channels=4, dropout=dropout)
            self.pool = AttentionPool(self.encoder.out_dim)
            embed_dim += self.encoder.out_dim

        self.hr_proj = None
        if input_variant in ("accel_hr", "hr_only"):
            self.hr_proj = nn.Sequential(
                nn.Linear(HR_INPUT_DIM, 32), nn.GELU(), nn.Linear(32, 32)
            )
            embed_dim += 32

        if embed_dim == 0:
            raise ValueError(f"input_variant {input_variant!r} selects no inputs")

        if context_model == "bilstm":
            self.context = nn.LSTM(
                embed_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True
            )
            head_dim = hidden_dim * 2
        elif context_model == "none":
            self.context = None
            head_dim = embed_dim
        else:
            raise ValueError(f"context_model must be 'bilstm' or 'none', got {context_model!r}")

        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(head_dim, N_CLASSES))

    def forward(self, waveform: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        """waveform ``[B, L, C, T]``, hr ``[B, L, 6]`` -> logits ``[B, L, 4]``."""
        batch, length = hr.shape[0], hr.shape[1]
        parts: list[torch.Tensor] = []

        if self.encoder is not None:
            flat = waveform.reshape(batch * length, waveform.shape[2], waveform.shape[3])
            pooled, _ = self.pool(self.encoder(flat))
            parts.append(pooled.reshape(batch, length, -1))

        if self.hr_proj is not None:
            parts.append(self.hr_proj(hr))

        x = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
        if self.context is not None:
            x, _ = self.context(x)
        return self.head(x)


def masked_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor, class_weights: torch.Tensor | None = None
) -> torch.Tensor:
    """Cross entropy that skips masked epochs (label == -1)."""
    return F.cross_entropy(
        logits.reshape(-1, N_CLASSES),
        labels.reshape(-1),
        weight=class_weights,
        ignore_index=INVALID,
    )


def make_class_weights(counts: np.ndarray, strategy: str = "class_weights") -> np.ndarray | None:
    """Inverse-frequency weights.

    Light is over half the epochs and Deep under a seventh, so an unweighted
    loss reaches a decent accuracy by mostly predicting Light -- and a model
    that never predicts Deep scores near zero kappa on the class that matters.
    """
    if strategy == "none":
        return None
    counts = np.asarray(counts, dtype=np.float64)
    weights = counts.sum() / np.maximum(counts, 1.0)
    return (weights / weights.mean()).astype(np.float32)
