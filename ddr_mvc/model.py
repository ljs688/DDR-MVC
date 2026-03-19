from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class ViewEncoder(nn.Module):
    """Per-view encoder for DDR-MVC."""

    def __init__(self, input_dim: int, hidden_dim: int, feature_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ViewDecoder(nn.Module):
    """Per-view decoder for DDR-MVC reconstruction."""

    def __init__(self, output_dim: int, hidden_dim: int, feature_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class DDRMVCModel(nn.Module):
    """Minimal DDR-MVC model with attention fusion over view embeddings."""

    def __init__(self, view_dims: List[int], hidden_dim: int, feature_dim: int):
        super().__init__()
        self.encoders = nn.ModuleList(
            [ViewEncoder(dim, hidden_dim, feature_dim) for dim in view_dims]
        )
        self.decoders = nn.ModuleList(
            [ViewDecoder(dim, hidden_dim, feature_dim) for dim in view_dims]
        )
        self.attention_score = nn.Linear(feature_dim, 1, bias=False)

    def encode(self, xs: List[torch.Tensor]) -> List[torch.Tensor]:
        return [encoder(x) for encoder, x in zip(self.encoders, xs)]

    def fuse(self, z_views: List[torch.Tensor]):
        stacked = torch.stack(z_views, dim=1)  # [B, V, D]
        logits = self.attention_score(stacked).squeeze(-1)  # [B, V]
        weights = torch.softmax(logits, dim=1)
        fused = torch.sum(stacked * weights.unsqueeze(-1), dim=1)  # [B, D]
        return fused, weights

    def forward(self, xs: List[torch.Tensor]):
        z_views = self.encode(xs)
        z_fused, weights = self.fuse(z_views)
        reconstructions = [decoder(z_fused) for decoder in self.decoders]
        return reconstructions, z_views, z_fused, weights
