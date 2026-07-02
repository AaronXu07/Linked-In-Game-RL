"""Replay buffer primitives for DQN training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Transition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    terminated: bool
    truncated: bool
    action_mask: np.ndarray
    next_action_mask: np.ndarray


@dataclass(frozen=True)
class TransitionBatch:
    obs: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    next_obs: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    action_mask: np.ndarray
    next_action_mask: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int, *, seed: int | None = None) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._storage: list[Transition] = []
        self._position = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._storage)

    def add(self, transition: Transition) -> None:
        stored = Transition(
            obs=np.asarray(transition.obs, dtype=np.float32).copy(),
            action=int(transition.action),
            reward=float(transition.reward),
            next_obs=np.asarray(transition.next_obs, dtype=np.float32).copy(),
            terminated=bool(transition.terminated),
            truncated=bool(transition.truncated),
            action_mask=np.asarray(transition.action_mask, dtype=np.bool_).copy(),
            next_action_mask=np.asarray(
                transition.next_action_mask,
                dtype=np.bool_,
            ).copy(),
        )
        if len(self._storage) < self.capacity:
            self._storage.append(stored)
        else:
            self._storage[self._position] = stored
        self._position = (self._position + 1) % self.capacity

    def sample(self, batch_size: int) -> TransitionBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if len(self._storage) < batch_size:
            raise ValueError("not enough transitions to sample a full batch")

        indices = self._rng.integers(0, len(self._storage), size=batch_size)
        transitions = [self._storage[int(index)] for index in indices]
        return TransitionBatch(
            obs=np.stack([transition.obs for transition in transitions]),
            action=np.asarray(
                [transition.action for transition in transitions],
                dtype=np.int64,
            ),
            reward=np.asarray(
                [transition.reward for transition in transitions],
                dtype=np.float32,
            ),
            next_obs=np.stack([transition.next_obs for transition in transitions]),
            terminated=np.asarray(
                [transition.terminated for transition in transitions],
                dtype=np.bool_,
            ),
            truncated=np.asarray(
                [transition.truncated for transition in transitions],
                dtype=np.bool_,
            ),
            action_mask=np.stack(
                [transition.action_mask for transition in transitions],
            ),
            next_action_mask=np.stack(
                [transition.next_action_mask for transition in transitions],
            ),
        )
