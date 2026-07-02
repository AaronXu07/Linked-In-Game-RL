"""Gymnasium environment adapter for Zip puzzles."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from gymnasium import Env, spaces

from zip.simulation import (
    GameState,
    Puzzle,
    generate_puzzle,
    legal_moves,
    new_game,
)
from zip.simulation import step as simulator_step
from zip.simulation.config import get_difficulty_config
from zip.simulation.renderer import render_ansi, render_image
from zip.simulation.utils import Direction

from .observations import encode_observation, make_observation_space
from .rewards import RewardConfig, compute_reward

ACTION_TO_DIRECTION = {
    0: Direction.UP,
    1: Direction.DOWN,
    2: Direction.LEFT,
    3: Direction.RIGHT,
}
DIRECTION_TO_ACTION = {
    direction: action for action, direction in ACTION_TO_DIRECTION.items()
}
INVALID_ACTION_MODES = {"terminate", "penalize", "mask_required"}


class ZipEnv(Env):
    """A Gymnasium-compatible Zip puzzle environment."""

    metadata = {"render_modes": ["ansi", "rgb_array", "human"]}

    def __init__(
        self,
        difficulty: str = "super_easy",
        *,
        puzzle: Puzzle | None = None,
        puzzle_pool: Sequence[Puzzle] | None = None,
        puzzle_sampler: Any | None = None,
        max_rows: int | None = None,
        max_cols: int | None = None,
        reward_config: RewardConfig | None = None,
        invalid_action_mode: str = "terminate",
        render_mode: str | None = None,
        use_action_mask: bool = True,
    ) -> None:
        puzzle_sources = sum(
            source is not None for source in (puzzle, puzzle_pool, puzzle_sampler)
        )
        if puzzle_sources > 1:
            raise ValueError("pass only one of puzzle, puzzle_pool, or puzzle_sampler")
        if invalid_action_mode not in INVALID_ACTION_MODES:
            choices = ", ".join(sorted(INVALID_ACTION_MODES))
            raise ValueError(f"invalid_action_mode must be one of: {choices}")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            choices = ", ".join(self.metadata["render_modes"])
            raise ValueError(f"render_mode must be one of: {choices}")

        self.difficulty = difficulty
        self._fixed_puzzle = puzzle
        self._puzzle_pool = tuple(puzzle_pool or ())
        self._puzzle_sampler = puzzle_sampler
        self.reward_config = reward_config or RewardConfig()
        self.invalid_action_mode = invalid_action_mode
        self.render_mode = render_mode
        self.use_action_mask = use_action_mask

        inferred_rows, inferred_cols = _infer_observation_bounds(
            difficulty,
            puzzle=puzzle,
            puzzle_pool=self._puzzle_pool,
        )
        self.max_rows = inferred_rows if max_rows is None else int(max_rows)
        self.max_cols = inferred_cols if max_cols is None else int(max_cols)
        if self.max_rows <= 0 or self.max_cols <= 0:
            raise ValueError("max_rows and max_cols must be positive")

        self.observation_space = make_observation_space(self.max_rows, self.max_cols)
        self.action_space = spaces.Discrete(len(ACTION_TO_DIRECTION))

        self.puzzle: Puzzle | None = None
        self.state: GameState | None = None
        self.episode_id = 0
        self.last_action: int | None = None
        self.last_step_reason: str | None = None
        self.cumulative_reward = 0.0
        self.episode_step_count = 0
        self.max_steps = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}

        puzzle = self._select_puzzle(options)
        self._validate_puzzle_bounds(puzzle)
        self.puzzle = puzzle
        self.state = new_game(puzzle)
        self.episode_id += 1
        self.last_action = None
        self.last_step_reason = None
        self.cumulative_reward = 0.0
        self.episode_step_count = 0
        self.max_steps = int(options.get("max_steps", 2 * puzzle.total_cells))

        return self._observation(), self._info(
            invalid_action=False,
            invalid_reason=None,
            dead_end=self._dead_end(),
            truncated=False,
        )

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.puzzle is None or self.state is None:
            raise RuntimeError("call reset() before step()")
        if not self.action_space.contains(action):
            raise ValueError(f"{action!r} is not a valid action")

        action = int(action)
        direction = ACTION_TO_DIRECTION[action]
        previous_state = self.state
        self.last_action = action
        self.episode_step_count += 1

        result = simulator_step(self.puzzle, previous_state, direction)
        invalid_action = not result.valid
        if invalid_action and self.invalid_action_mode == "mask_required":
            raise ValueError(f"invalid action {action}: {result.reason}")

        if result.valid or invalid_action:
            self.state = result.state

        terminated = False
        if result.valid:
            terminated = self.state.success
        elif self.invalid_action_mode == "terminate":
            terminated = True

        dead_end = False
        if not terminated:
            dead_end = self._dead_end()
            if dead_end:
                terminated = True

        truncated = self.episode_step_count >= self.max_steps
        reward = compute_reward(
            self.puzzle,
            previous_state,
            result,
            dead_end=dead_end,
            truncated=truncated,
            config=self.reward_config,
        )
        self.cumulative_reward += reward
        self.last_step_reason = result.reason

        return self._observation(), reward, terminated, truncated, self._info(
            invalid_action=invalid_action,
            invalid_reason=result.reason if invalid_action else None,
            dead_end=dead_end,
            truncated=truncated,
        )

    def render(self):
        if self.puzzle is None or self.state is None:
            return None

        if self.render_mode in (None, "ansi"):
            return render_ansi(self.puzzle, self.state)
        if self.render_mode == "human":
            text = render_ansi(self.puzzle, self.state)
            print(text)
            return None
        if self.render_mode == "rgb_array":
            return _ppm_to_rgb_array(render_image(self.puzzle, self.state))
        raise ValueError(f"unsupported render mode {self.render_mode!r}")

    def close(self) -> None:
        close = getattr(self._puzzle_sampler, "close", None)
        if close is not None:
            close()

    def action_masks(self) -> np.ndarray:
        return self._action_mask()

    def _select_puzzle(self, options: dict[str, Any]) -> Puzzle:
        if "puzzle" in options:
            puzzle = options["puzzle"]
            if not isinstance(puzzle, Puzzle):
                raise TypeError("options['puzzle'] must be a Puzzle")
            return puzzle

        if self._fixed_puzzle is not None:
            return self._fixed_puzzle

        if self._puzzle_pool:
            index = int(self.np_random.integers(0, len(self._puzzle_pool)))
            return self._puzzle_pool[index]

        if self._puzzle_sampler is not None:
            sampler = self._puzzle_sampler
            if hasattr(sampler, "sample_puzzle"):
                puzzle = sampler.sample_puzzle()
            else:
                puzzle = sampler()
            if not isinstance(puzzle, Puzzle):
                raise TypeError("puzzle_sampler must return a Puzzle")
            return puzzle

        difficulty = str(options.get("difficulty", self.difficulty))
        puzzle_seed = options.get("seed")
        if puzzle_seed is None:
            puzzle_seed = int(self.np_random.integers(0, 2**63 - 1))
        return generate_puzzle(difficulty, seed=int(puzzle_seed))

    def _validate_puzzle_bounds(self, puzzle: Puzzle) -> None:
        if puzzle.rows > self.max_rows or puzzle.cols > self.max_cols:
            raise ValueError(
                "puzzle dimensions exceed environment bounds: "
                f"{puzzle.rows}x{puzzle.cols} > {self.max_rows}x{self.max_cols}"
            )

    def _observation(self) -> np.ndarray:
        if self.puzzle is None or self.state is None:
            raise RuntimeError("environment has no active puzzle")
        return encode_observation(
            self.puzzle,
            self.state,
            max_rows=self.max_rows,
            max_cols=self.max_cols,
        )

    def _action_mask(self) -> np.ndarray:
        mask = np.zeros(len(ACTION_TO_DIRECTION), dtype=np.bool_)
        if self.puzzle is None or self.state is None:
            return mask
        legal = set(legal_moves(self.puzzle, self.state))
        for action, direction in ACTION_TO_DIRECTION.items():
            mask[action] = direction in legal
        return mask

    def _dead_end(self) -> bool:
        if self.puzzle is None or self.state is None:
            return False
        return not self.state.success and len(legal_moves(self.puzzle, self.state)) == 0

    def _info(
        self,
        *,
        invalid_action: bool,
        invalid_reason: str | None,
        dead_end: bool,
        truncated: bool,
    ) -> dict[str, Any]:
        if self.puzzle is None or self.state is None:
            raise RuntimeError("environment has no active puzzle")
        mask = self._action_mask() if self.use_action_mask else np.ones(4, dtype=np.bool_)
        return {
            "action_mask": mask,
            "success": self.state.success,
            "invalid_action": invalid_action,
            "invalid_reason": invalid_reason,
            "dead_end": dead_end,
            "truncated": truncated,
            "episode_id": self.episode_id,
            "episode_step_count": self.episode_step_count,
            "path_step_count": self.state.step_count,
            "visited_count": len(self.state.visited),
            "total_cells": self.puzzle.total_cells,
            "current_pos": self.state.current_pos,
            "next_waypoint_index": self.state.next_waypoint_index,
            "waypoint_count": len(self.puzzle.waypoints),
            "puzzle_seed": self.puzzle.seed,
            "difficulty": self.puzzle.difficulty,
            "cumulative_reward": self.cumulative_reward,
        }


def _infer_observation_bounds(
    difficulty: str,
    *,
    puzzle: Puzzle | None,
    puzzle_pool: Sequence[Puzzle],
) -> tuple[int, int]:
    if puzzle is not None:
        return puzzle.rows, puzzle.cols
    if puzzle_pool:
        return max(p.rows for p in puzzle_pool), max(p.cols for p in puzzle_pool)
    config = get_difficulty_config(difficulty)
    return config.rows, config.cols


def _ppm_to_rgb_array(data: bytes) -> np.ndarray:
    header_end = data.find(b"\n255\n")
    if not data.startswith(b"P6\n") or header_end == -1:
        raise ValueError("expected binary PPM data")
    dimensions = data[3:header_end].strip().split()
    if len(dimensions) != 2:
        raise ValueError("invalid PPM dimensions")
    width, height = (int(value) for value in dimensions)
    body = data[header_end + len(b"\n255\n") :]
    return np.frombuffer(body, dtype=np.uint8).reshape((height, width, 3))
