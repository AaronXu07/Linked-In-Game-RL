"""Neural network models for Patches candidate-action agents."""

from __future__ import annotations

import torch
from torch import nn


class PatchesCandidateDQN(nn.Module):
    """Candidate-conditioned Q-network for dynamic rectangle placements."""

    def __init__(
        self,
        grid_shape: tuple[int, int, int],
        candidate_feature_count: int,
        max_actions: int,
        *,
        candidate_hidden: int = 128,
        global_hidden: int = 128,
        head_hidden: int = 256,
    ) -> None:
        super().__init__()
        channels, rows, cols = grid_shape
        if channels <= 0 or rows <= 0 or cols <= 0:
            raise ValueError("grid_shape must be positive")
        if candidate_feature_count <= 0 or max_actions <= 0:
            raise ValueError("candidate_feature_count and max_actions must be positive")

        self.grid_shape = tuple(grid_shape)
        self.candidate_feature_count = int(candidate_feature_count)
        self.max_actions = int(max_actions)

        self.grid_encoder = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_mlp = nn.Sequential(
            nn.Linear(64, global_hidden),
            nn.ReLU(),
        )
        self.candidate_mlp = nn.Sequential(
            nn.Linear(candidate_feature_count, candidate_hidden),
            nn.ReLU(),
            nn.Linear(candidate_hidden, candidate_hidden),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(global_hidden + candidate_hidden + 64, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

    def forward(
        self,
        grid: torch.Tensor,
        candidates: torch.Tensor,
        candidate_footprints: torch.Tensor,
    ) -> torch.Tensor:
        spatial = self.grid_encoder(grid)
        batch, channels, rows, cols = spatial.shape
        if candidate_footprints.shape[-2:] != (rows, cols):
            candidate_footprints = torch.nn.functional.interpolate(
                candidate_footprints,
                size=(rows, cols),
                mode="nearest",
            )

        global_embedding = self.global_mlp(
            self.global_pool(spatial).reshape(batch, channels)
        )
        global_embedding = global_embedding[:, None, :].expand(
            -1,
            candidates.shape[1],
            -1,
        )

        candidate_embedding = self.candidate_mlp(candidates)
        footprints = candidate_footprints.to(dtype=spatial.dtype)
        footprint_area = footprints.sum(dim=(-1, -2)).clamp_min(1.0)
        footprint_embedding = torch.einsum("bchw,bahw->bac", spatial, footprints)
        footprint_embedding = footprint_embedding / footprint_area.unsqueeze(-1)

        combined = torch.cat(
            (global_embedding, candidate_embedding, footprint_embedding),
            dim=-1,
        )
        return self.head(combined).squeeze(-1)


class FlatCandidateDQN(nn.Module):
    """Small MLP baseline useful for tiny smoke tests."""

    def __init__(
        self,
        grid_shape: tuple[int, int, int],
        candidate_feature_count: int,
        max_actions: int,
        *,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        channels, rows, cols = grid_shape
        flat_grid = channels * rows * cols
        flat_footprint = rows * cols
        self.max_actions = int(max_actions)
        self.head = nn.Sequential(
            nn.Linear(flat_grid + candidate_feature_count + flat_footprint, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        grid: torch.Tensor,
        candidates: torch.Tensor,
        candidate_footprints: torch.Tensor,
    ) -> torch.Tensor:
        batch, action_count, _ = candidates.shape
        flat_grid = grid.flatten(start_dim=1)[:, None, :].expand(-1, action_count, -1)
        flat_footprints = candidate_footprints.flatten(start_dim=2)
        combined = torch.cat((flat_grid, candidates, flat_footprints), dim=-1)
        return self.head(combined).squeeze(-1)
