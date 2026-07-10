"""Training callbacks for metrics, evaluation, and checkpoints."""

from __future__ import annotations

import json
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Sequence

from patches.simulation import Puzzle

Callback = Callable[[Any, str, dict[str, Any]], None]


class CallbackList:
    def __init__(self, callbacks: Sequence[Callback]) -> None:
        self.callbacks = tuple(callbacks)

    def __call__(self, agent, event: str, payload: dict[str, Any]) -> None:
        for callback in self.callbacks:
            callback(agent, event, payload)

    def close(self) -> None:
        for callback in self.callbacks:
            close = getattr(callback, "close", None)
            if close is not None:
                close()


class TensorBoardCallback:
    def __init__(self, log_dir: str | Path) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(str(log_dir))

    def __call__(self, agent, event: str, payload: dict[str, Any]) -> None:
        step = agent.total_steps
        if event == "step":
            for key in ("loss", "td_error", "q_mean", "q_max", "epsilon", "replay_size"):
                if key in payload:
                    self.writer.add_scalar(f"train/{key}", payload[key], step)
            curriculum_keys = {
                "curriculum_stage_index": "stage_index",
                "curriculum_grid_cells": "grid_cells",
                "curriculum_success_gate": "success_gate",
            }
            for key, name in curriculum_keys.items():
                if key in payload:
                    self.writer.add_scalar(f"curriculum/{name}", payload[key], step)
        elif event == "episode":
            for key in ("reward", "length", "success", "placed_count", "covered_count"):
                if key in payload:
                    self.writer.add_scalar(f"episode/{key}", payload[key], step)
        elif event == "eval":
            for key, value in payload.items():
                if isinstance(value, Real):
                    self.writer.add_scalar(f"eval/{key}", value, step)

    def close(self) -> None:
        self.writer.close()


class CheckpointCallback:
    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        every_steps: int,
        keep_latest_name: str = "latest.pt",
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.every_steps = int(every_steps)
        self.keep_latest_name = keep_latest_name
        self._last_saved_step = -1

    def __call__(self, agent, event: str, payload: dict[str, Any]) -> None:
        del payload
        if event == "step" and self.every_steps > 0:
            if agent.total_steps % self.every_steps == 0:
                self._save(agent)
        elif event == "train_end":
            self._save(agent)

    def _save(self, agent) -> None:
        if agent.total_steps == self._last_saved_step:
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        agent.save_checkpoint(self.checkpoint_dir / self.keep_latest_name)
        agent.save_checkpoint(self.checkpoint_dir / f"step_{agent.total_steps}.pt")
        self._last_saved_step = agent.total_steps


class BestCheckpointCallback:
    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        metric: str = "success_rate",
        mode: str = "max",
        filename: str = "best.pt",
    ) -> None:
        if mode not in {"max", "min"}:
            raise ValueError("mode must be 'max' or 'min'")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.metric = metric
        self.mode = mode
        self.filename = filename
        self.best_value: float | None = None

    def __call__(self, agent, event: str, payload: dict[str, Any]) -> None:
        if event != "eval" or self.metric not in payload:
            return
        value = float(payload[self.metric])
        if self.best_value is not None:
            if self.mode == "max" and value <= self.best_value:
                return
            if self.mode == "min" and value >= self.best_value:
                return
        self.best_value = value
        agent.checkpoint_metadata["best_checkpoint"] = {
            "metric": self.metric,
            "value": value,
            "step": agent.total_steps,
        }
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        agent.save_checkpoint(self.checkpoint_dir / self.filename)


class EpisodeJsonlCallback:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")

    def __call__(self, agent, event: str, payload: dict[str, Any]) -> None:
        if event != "episode":
            return
        env = agent.env
        puzzle = env.puzzle
        state = env.state
        record = {
            "global_step": agent.total_steps,
            "episode_id": env.episode_id,
            "difficulty": None if puzzle is None else puzzle.difficulty,
            "puzzle_seed": None if puzzle is None else puzzle.seed,
            "success": bool(payload["success"]),
            "episode_length": int(payload["length"]),
            "reward": float(payload["reward"]),
            "invalid_action": bool(payload["invalid_action"]),
            "dead_end": bool(payload["dead_end"]),
            "truncated": bool(payload["truncated"]),
            "placed_count": None if state is None else len(state.patches),
            "covered_count": None if state is None else state.covered_count,
            "total_cells": None if puzzle is None else puzzle.total_cells,
        }
        self._handle.write(json.dumps(record, sort_keys=True) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


class EvaluationCallback:
    def __init__(
        self,
        puzzles: Sequence[Puzzle],
        *,
        every_steps: int,
        episodes: int,
        callbacks: Sequence[Callback] | None = None,
    ) -> None:
        self.puzzles = tuple(puzzles)
        self.every_steps = int(every_steps)
        self.episodes = int(episodes)
        self.callbacks = tuple(callbacks or ())
        self._last_eval_step = -1

    def __call__(self, agent, event: str, payload: dict[str, Any]) -> None:
        del payload
        if event != "step" or self.every_steps <= 0:
            return
        if agent.total_steps == self._last_eval_step:
            return
        if agent.total_steps % self.every_steps != 0:
            return
        metrics = agent.evaluate(self.puzzles, self.episodes, deterministic=True)
        self._last_eval_step = agent.total_steps
        for callback in self.callbacks:
            callback(agent, "eval", metrics)
