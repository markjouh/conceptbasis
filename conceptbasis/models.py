"""Reusable model components for ConceptBasis experiments."""

from __future__ import annotations

import torch
import torch.nn as nn


class Adapter(nn.Module):
    """Two-layer projection head shared by training and downstream tools."""

    def __init__(self, d_in: int, d_out: int, hidden: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
