"""DQN agent wrapper for Patches candidate-action environments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from patches.simulation import Puzzle

from .env import PatchesEnv
from .models import FlatCandidateDQN, PatchesCandidateDQN
from .replay import ReplayBuffer, Transition, TransitionBatch

Callback = Callable[["PatchesAgent", str, dict[str, Any]], None]


@dataclass(frozen=True)
class AgentConfig:
    algorithm: str = "candidate_dqn"
    learning_rate: float = 1e-4
    replay_size: int = 100_000
    warmup_steps: int = 5_000
    batch_size: int = 128
    gamma: float = 0.99
    train_every: int = 4
    target_update_every: int = 1_000
    tau: float = 1.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 100_000
    double_dqn: bool = True
    max_grad_norm: float = 0.5
    seed: int = 0
    device: str = "auto"


class PatchesAgent:
    def __init__(
        self,
        env: PatchesEnv,
        config: AgentConfig,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        if config.algorithm not in {"candidate_dqn", "flat_candidate_dqn"}:
            raise ValueError("algorithm must be candidate_dqn or flat_candidate_dqn")
        _validate_config(config)

        self.env = env
        self.config = config
        self.device = select_device(config.device)
        self.rng = np.random.default_rng(config.seed)
        self.replay = ReplayBuffer(config.replay_size, seed=config.seed)
        self.total_steps = 0
        self.optimization_steps = 0
        self.checkpoint_metadata: dict[str, Any] = {}

        grid_shape = tuple(int(value) for value in _space_shape(env, "grid"))
        candidate_shape = _space_shape(env, "candidates")
        candidate_feature_count = int(candidate_shape[1])
        action_count = int(env.action_space.n)
        model_cls = (
            FlatCandidateDQN
            if config.algorithm == "flat_candidate_dqn"
            else PatchesCandidateDQN
        )
        self.online_net = model_cls(
            grid_shape,
            candidate_feature_count,
            action_count,
        ).to(self.device)
        self.target_net = model_cls(
            grid_shape,
            candidate_feature_count,
            action_count,
        ).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(
            self.online_net.parameters(),
            lr=config.learning_rate,
        )

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

    def epsilon(self) -> float:
        if self.config.epsilon_decay_steps <= 0:
            return self.config.epsilon_end
        fraction = min(1.0, self.total_steps / self.config.epsilon_decay_steps)
        return float(
            self.config.epsilon_start
            + fraction * (self.config.epsilon_end - self.config.epsilon_start)
        )

    def predict(
        self,
        observation: Mapping[str, np.ndarray],
        *,
        deterministic: bool = False,
        action_mask=None,
    ) -> int:
        mask = _normalize_action_mask(action_mask, int(self.env.action_space.n))
        legal_actions = np.flatnonzero(mask)
        if len(legal_actions) == 0:
            return 0

        if not deterministic and self.rng.random() < self.epsilon():
            return int(self.rng.choice(legal_actions))

        obs_tensors = _obs_to_tensors(observation, self.device, add_batch=True)
        with torch.no_grad():
            q_values = self.online_net(**obs_tensors).squeeze(0).detach().cpu().numpy()
        q_values = np.where(mask, q_values, -np.inf)
        return int(np.argmax(q_values))

    def train_step(self, batch: TransitionBatch) -> dict[str, float]:
        obs = _obs_to_tensors(batch.obs, self.device)
        next_obs = _obs_to_tensors(batch.next_obs, self.device)
        actions = torch.as_tensor(batch.action, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(batch.reward, dtype=torch.float32, device=self.device)
        terminated = torch.as_tensor(
            batch.terminated,
            dtype=torch.bool,
            device=self.device,
        )
        truncated = torch.as_tensor(batch.truncated, dtype=torch.bool, device=self.device)
        next_masks = torch.as_tensor(
            batch.next_action_mask,
            dtype=torch.bool,
            device=self.device,
        )

        q_values = self.online_net(**obs)
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_target_q = self.target_net(**next_obs)
            legal_next = next_masks.any(dim=1)
            if self.config.double_dqn:
                next_online_q = self.online_net(**next_obs)
                next_actions = next_online_q.masked_fill(~next_masks, -1e9).argmax(dim=1)
                next_values = next_target_q.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                next_values = next_target_q.masked_fill(~next_masks, -1e9).max(dim=1).values
            next_values = torch.where(
                legal_next,
                next_values,
                torch.zeros_like(next_values),
            )
            done = terminated | truncated
            targets = rewards + self.config.gamma * (~done).float() * next_values

        loss = F.smooth_l1_loss(q_sa, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(
                self.online_net.parameters(),
                self.config.max_grad_norm,
            )
        self.optimizer.step()
        self.optimization_steps += 1
        self._update_target_network()

        td_error = (q_sa.detach() - targets).abs()
        return {
            "loss": float(loss.detach().cpu().item()),
            "td_error": float(td_error.mean().cpu().item()),
            "q_mean": float(q_values.detach().mean().cpu().item()),
            "q_max": float(q_values.detach().max().cpu().item()),
        }

    def train(
        self,
        total_steps: int,
        callbacks: Sequence[Callback] | None = None,
    ) -> None:
        if total_steps <= 0:
            return
        callbacks = tuple(callbacks or ())
        target_step = self.total_steps + int(total_steps)
        obs, info = self.env.reset(seed=self.config.seed)
        action_mask = info["action_mask"]
        episode_reward = 0.0
        episode_length = 0

        _emit(callbacks, self, "train_start", {"target_step": target_step})
        while self.total_steps < target_step:
            action = self.predict(obs, deterministic=False, action_mask=action_mask)
            next_obs, reward, terminated, truncated, next_info = self.env.step(action)
            next_action_mask = next_info["action_mask"]
            self.replay.add(
                Transition(
                    obs=obs,
                    action=action,
                    reward=reward,
                    next_obs=next_obs,
                    terminated=terminated,
                    truncated=truncated,
                    action_mask=action_mask,
                    next_action_mask=next_action_mask,
                )
            )

            self.total_steps += 1
            episode_reward += reward
            episode_length += 1
            metrics: dict[str, float] = {
                "epsilon": self.epsilon(),
                "replay_size": float(len(self.replay)),
            }
            if (
                len(self.replay) >= self.config.warmup_steps
                and len(self.replay) >= self.config.batch_size
                and self.total_steps % self.config.train_every == 0
            ):
                batch = self.replay.sample(self.config.batch_size)
                metrics.update(self.train_step(batch))
            _emit(callbacks, self, "step", metrics)

            if terminated or truncated:
                episode_metrics = {
                    "reward": episode_reward,
                    "length": float(episode_length),
                    "success": float(next_info["success"]),
                    "terminated": float(terminated),
                    "truncated": float(truncated),
                    "invalid_action": float(next_info["invalid_action"]),
                    "dead_end": float(next_info["dead_end"]),
                    "placed_count": float(next_info["placed_count"]),
                    "covered_count": float(next_info["covered_count"]),
                }
                _emit(callbacks, self, "episode", episode_metrics)
                obs, info = self.env.reset()
                action_mask = info["action_mask"]
                episode_reward = 0.0
                episode_length = 0
            else:
                obs = next_obs
                action_mask = next_action_mask

        _emit(callbacks, self, "train_end", {"total_steps": self.total_steps})

    def train_parallel(
        self,
        envs: Sequence[PatchesEnv],
        total_steps: int,
        callbacks: Sequence[Callback] | None = None,
    ) -> None:
        if total_steps <= 0:
            return
        if not envs:
            raise ValueError("at least one environment is required")
        if len(envs) == 1:
            original_env = self.env
            self.env = envs[0]
            try:
                self.train(total_steps, callbacks=callbacks)
            finally:
                self.env = original_env
            return

        callbacks = tuple(callbacks or ())
        target_step = self.total_steps + int(total_steps)
        observations = []
        infos = []
        episode_rewards = []
        episode_lengths = []
        for index, env in enumerate(envs):
            obs, info = env.reset(seed=self.config.seed + index)
            observations.append(obs)
            infos.append(info)
            episode_rewards.append(0.0)
            episode_lengths.append(0)

        _emit(callbacks, self, "train_start", {"target_step": target_step})
        while self.total_steps < target_step:
            for env_index, env in enumerate(envs):
                if self.total_steps >= target_step:
                    break

                obs = observations[env_index]
                info = infos[env_index]
                action_mask = info["action_mask"]
                action = self.predict(obs, deterministic=False, action_mask=action_mask)
                next_obs, reward, terminated, truncated, next_info = env.step(action)
                next_action_mask = next_info["action_mask"]
                self.replay.add(
                    Transition(
                        obs=obs,
                        action=action,
                        reward=reward,
                        next_obs=next_obs,
                        terminated=terminated,
                        truncated=truncated,
                        action_mask=action_mask,
                        next_action_mask=next_action_mask,
                    )
                )

                self.total_steps += 1
                episode_rewards[env_index] += reward
                episode_lengths[env_index] += 1
                metrics: dict[str, float] = {
                    "epsilon": self.epsilon(),
                    "replay_size": float(len(self.replay)),
                    "parallel_envs": float(len(envs)),
                    "env_index": float(env_index),
                }
                if (
                    len(self.replay) >= self.config.warmup_steps
                    and len(self.replay) >= self.config.batch_size
                    and self.total_steps % self.config.train_every == 0
                ):
                    batch = self.replay.sample(self.config.batch_size)
                    metrics.update(self.train_step(batch))
                _emit(callbacks, self, "step", metrics)

                if terminated or truncated:
                    episode_metrics = {
                        "reward": episode_rewards[env_index],
                        "length": float(episode_lengths[env_index]),
                        "success": float(next_info["success"]),
                        "terminated": float(terminated),
                        "truncated": float(truncated),
                        "invalid_action": float(next_info["invalid_action"]),
                        "dead_end": float(next_info["dead_end"]),
                        "placed_count": float(next_info["placed_count"]),
                        "covered_count": float(next_info["covered_count"]),
                        "env_index": float(env_index),
                    }
                    original_env = self.env
                    self.env = env
                    try:
                        _emit(callbacks, self, "episode", episode_metrics)
                    finally:
                        self.env = original_env
                    observations[env_index], infos[env_index] = env.reset()
                    episode_rewards[env_index] = 0.0
                    episode_lengths[env_index] = 0
                else:
                    observations[env_index] = next_obs
                    infos[env_index] = next_info

        _emit(callbacks, self, "train_end", {"total_steps": self.total_steps})

    def evaluate(
        self,
        puzzles: Sequence[Puzzle],
        episodes: int,
        deterministic: bool = True,
    ) -> dict[str, float]:
        if not puzzles:
            raise ValueError("at least one evaluation puzzle is required")
        if episodes <= 0:
            raise ValueError("episodes must be positive")

        eval_env = PatchesEnv(
            puzzle=puzzles[0],
            max_rows=self.env.max_rows,
            max_cols=self.env.max_cols,
            max_actions=self.env.max_actions,
            reward_config=self.env.reward_config,
            invalid_action_mode=self.env.invalid_action_mode,
            episode_mode=self.env.episode_mode,
        )
        rewards: list[float] = []
        lengths: list[int] = []
        successes = invalid_actions = dead_ends = truncations = total_action_steps = 0
        covered_fractions: list[float] = []
        placed_fractions: list[float] = []

        for episode in range(episodes):
            puzzle = puzzles[episode % len(puzzles)]
            obs, info = eval_env.reset(options={"puzzle": puzzle})
            episode_reward = 0.0
            episode_length = 0
            while True:
                action = self.predict(
                    obs,
                    deterministic=deterministic,
                    action_mask=info["action_mask"],
                )
                obs, reward, terminated, truncated, info = eval_env.step(action)
                episode_reward += reward
                episode_length += 1
                total_action_steps += 1
                invalid_actions += int(info["invalid_action"])
                if terminated or truncated:
                    rewards.append(episode_reward)
                    lengths.append(episode_length)
                    successes += int(info["success"])
                    dead_ends += int(info["dead_end"])
                    truncations += int(truncated)
                    covered_fractions.append(
                        info["covered_count"] / max(1, info["total_cells"])
                    )
                    placed_fractions.append(
                        info["placed_count"] / max(1, info["clue_count"])
                    )
                    break

        return {
            "success_rate": successes / episodes,
            "mean_reward": float(np.mean(rewards)),
            "mean_episode_length": float(np.mean(lengths)),
            "invalid_action_rate": invalid_actions / max(1, total_action_steps),
            "dead_end_rate": dead_ends / episodes,
            "truncation_rate": truncations / episodes,
            "mean_covered_fraction": float(np.mean(covered_fractions)),
            "mean_placed_fraction": float(np.mean(placed_fractions)),
        }

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.config),
                "total_steps": self.total_steps,
                "optimization_steps": self.optimization_steps,
                "online_net": self.online_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "observation_shapes": {
                    key: tuple(int(value) for value in _space_shape(self.env, key))
                    for key in ("grid", "candidates", "candidate_footprints")
                },
                "action_count": int(self.env.action_space.n),
                "metadata": self.checkpoint_metadata,
            },
            path,
        )

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint.get("target_net", checkpoint["online_net"]))
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_steps = int(checkpoint.get("total_steps", 0))
        self.optimization_steps = int(checkpoint.get("optimization_steps", 0))
        self.checkpoint_metadata = dict(checkpoint.get("metadata", {}))

    def _update_target_network(self) -> None:
        if self.config.tau == 1.0:
            if self.optimization_steps % self.config.target_update_every == 0:
                self.target_net.load_state_dict(self.online_net.state_dict())
            return
        if not (0.0 < self.config.tau < 1.0):
            raise ValueError("tau must be in (0, 1] for target updates")
        with torch.no_grad():
            for target_param, online_param in zip(
                self.target_net.parameters(),
                self.online_net.parameters(),
            ):
                target_param.data.mul_(1.0 - self.config.tau)
                target_param.data.add_(self.config.tau * online_param.data)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be one of: auto, cpu, cuda, mps")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("cuda was requested but is not available")
    if requested == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise ValueError("mps was requested but is not available")
    return torch.device(requested)


def _obs_to_tensors(
    observation: Mapping[str, np.ndarray],
    device: torch.device,
    *,
    add_batch: bool = False,
) -> dict[str, torch.Tensor]:
    tensors = {}
    for key in ("grid", "candidates", "candidate_footprints"):
        array = np.asarray(observation[key], dtype=np.float32)
        if add_batch:
            array = array[None, ...]
        tensors[key] = torch.as_tensor(array, dtype=torch.float32, device=device)
    return tensors


def _space_shape(env: PatchesEnv, key: str) -> tuple[int, ...]:
    return tuple(int(value) for value in env.observation_space.spaces[key].shape)


def _validate_config(config: AgentConfig) -> None:
    positive_ints = {
        "replay_size": config.replay_size,
        "batch_size": config.batch_size,
        "train_every": config.train_every,
        "target_update_every": config.target_update_every,
    }
    for name, value in positive_ints.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if config.warmup_steps < 0:
        raise ValueError("warmup_steps cannot be negative")
    if not (0.0 <= config.gamma <= 1.0):
        raise ValueError("gamma must be between 0 and 1")
    if not (0.0 < config.tau <= 1.0):
        raise ValueError("tau must be in (0, 1]")
    if config.epsilon_decay_steps < 0:
        raise ValueError("epsilon_decay_steps cannot be negative")


def _normalize_action_mask(action_mask, action_count: int) -> np.ndarray:
    if action_mask is None:
        return np.ones(action_count, dtype=np.bool_)
    mask = np.asarray(action_mask, dtype=np.bool_)
    if mask.shape != (action_count,):
        raise ValueError(f"action_mask must have shape ({action_count},)")
    return mask


def _emit(
    callbacks: Sequence[Callback],
    agent: PatchesAgent,
    event: str,
    payload: dict[str, Any],
) -> None:
    for callback in callbacks:
        callback(agent, event, payload)
