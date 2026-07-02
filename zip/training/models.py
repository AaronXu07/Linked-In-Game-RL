"""Neural network models for Zip agents."""

from __future__ import annotations

import torch
from torch import nn


class ZipDQN(nn.Module):
    """Compact convolutional Q-network for grid observations."""

    def __init__(
        self,
        observation_shape: tuple[int, int, int],
        action_count: int = 4,
    ) -> None:
        super().__init__()
        channels, rows, cols = observation_shape
        if channels <= 0 or rows <= 0 or cols <= 0:
            raise ValueError("observation_shape must be positive")

        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat_size = self.features(torch.zeros(1, channels, rows, cols)).shape[1]
        self.head = nn.Sequential(
            nn.Linear(flat_size, 256),
            nn.ReLU(),
            nn.Linear(256, action_count),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(obs))
