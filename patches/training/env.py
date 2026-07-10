"""Gymnasium environment adapter for Patches puzzles."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from gymnasium import Env, spaces

from patches.simulation import GameState, Puzzle, StepResult, generate_puzzle, new_game
from patches.simulation import place as simulator_place
from patches.simulation.config import get_difficulty_config
from patches.simulation.renderer import render_ansi, render_image

from .actions import (
    EPISODE_MODES,
    PlacementAction,
    build_action_mask,
    infer_max_actions,
    legal_placement_actions,
    precompute_candidates,
)
from .observations import (
    CANDIDATE_FEATURE_NAMES,
    encode_observation,
    make_observation_space,
)
from .rewards import RewardConfig, compute_reward

INVALID_ACTION_MODES = {"terminate", "penalize", "mask_required"}


class PatchesEnv(Env):
    """A Gymnasium-compatible Patches placement environment."""

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
        max_actions: int | None = None,
        reward_config: RewardConfig | None = None,
        invalid_action_mode: str = "terminate",
        episode_mode: str = "commit_only",
        render_mode: str | None = None,
        use_action_mask: bool = True,
        use_solver_dead_end_check: bool = False,
        include_solution_hint: bool = False,
    ) -> None:
        puzzle_sources = sum(
            source is not None for source in (puzzle, puzzle_pool, puzzle_sampler)
        )
        if puzzle_sources > 1:
            raise ValueError("pass only one of puzzle, puzzle_pool, or puzzle_sampler")
        if invalid_action_mode not in INVALID_ACTION_MODES:
            choices = ", ".join(sorted(INVALID_ACTION_MODES))
            raise ValueError(f"invalid_action_mode must be one of: {choices}")
        if episode_mode not in EPISODE_MODES:
            choices = ", ".join(sorted(EPISODE_MODES))
            raise ValueError(f"episode_mode must be one of: {choices}")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            choices = ", ".join(self.metadata["render_modes"])
            raise ValueError(f"render_mode must be one of: {choices}")

        self.difficulty = difficulty
        self._fixed_puzzle = puzzle
        self._puzzle_pool = tuple(puzzle_pool or ())
        self._puzzle_sampler = puzzle_sampler
        self.reward_config = reward_config or RewardConfig()
        self.invalid_action_mode = invalid_action_mode
        self.episode_mode = episode_mode
        self.render_mode = render_mode
        self.use_action_mask = use_action_mask
        self.use_solver_dead_end_check = use_solver_dead_end_check
        self.include_solution_hint = include_solution_hint

        inferred_rows, inferred_cols = _infer_observation_bounds(
            difficulty,
            puzzle=puzzle,
            puzzle_pool=self._puzzle_pool,
        )
        self.max_rows = inferred_rows if max_rows is None else int(max_rows)
        self.max_cols = inferred_cols if max_cols is None else int(max_cols)
        if self.max_rows <= 0 or self.max_cols <= 0:
            raise ValueError("max_rows and max_cols must be positive")

        inferred_actions = infer_max_actions(
            puzzle,
            max_rows=self.max_rows,
            max_cols=self.max_cols,
        )
        self.max_actions = inferred_actions if max_actions is None else int(max_actions)
        if self.max_actions <= 0:
            raise ValueError("max_actions must be positive")

        self.observation_space = make_observation_space(
            self.max_rows,
            self.max_cols,
            self.max_actions,
            include_solution_hint=include_solution_hint,
        )
        self.action_space = spaces.Discrete(self.max_actions)
        self.candidate_feature_count = len(CANDIDATE_FEATURE_NAMES)

        self.puzzle: Puzzle | None = None
        self.state: GameState | None = None
        self.episode_id = 0
        self.last_action: int | None = None
        self.last_placement_action: PlacementAction | None = None
        self.last_step_reason: str | None = None
        self.cumulative_reward = 0.0
        self.episode_step_count = 0
        self.max_steps = 0
        self._base_candidates_by_clue: dict[int, tuple] = {}
        self._current_actions: list[PlacementAction] = []
        self._current_action_mask = np.zeros(self.max_actions, dtype=np.bool_)

    @property
    def current_actions(self) -> tuple[PlacementAction, ...]:
        return tuple(self._current_actions)

    def action_for_index(self, action: int) -> PlacementAction | None:
        index = int(action)
        if 0 <= index < len(self._current_actions):
            return self._current_actions[index]
        return None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}

        puzzle = self._select_puzzle(options)
        self._validate_puzzle_bounds(puzzle)
        self.puzzle = puzzle
        self.state = new_game(puzzle)
        self.episode_id += 1
        self.last_action = None
        self.last_placement_action = None
        self.last_step_reason = None
        self.cumulative_reward = 0.0
        self.episode_step_count = 0
        default_steps = len(puzzle.clues) if self.episode_mode == "commit_only" else 3 * len(puzzle.clues)
        self.max_steps = int(options.get("max_steps", default_steps))
        self._base_candidates_by_clue = precompute_candidates(puzzle)
        self._prepare_actions()

        return self._observation(), self._info(
            invalid_action=False,
            invalid_reason=None,
            dead_end=self._dead_end(),
            truncated=False,
        )

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self.puzzle is None or self.state is None:
            raise RuntimeError("call reset() before step()")
        if not self.action_space.contains(action):
            raise ValueError(f"{action!r} is not a valid action")

        action = int(action)
        previous_state = self.state
        self.last_action = action
        self.episode_step_count += 1

        placement_action: PlacementAction | None = None
        if action < len(self._current_actions) and self._current_action_mask[action]:
            placement_action = self._current_actions[action]

        if placement_action is None:
            result = StepResult(
                state=previous_state.clone(),
                valid=False,
                reason="action slot is not legal",
                solved=previous_state.success,
            )
        else:
            result = simulator_place(
                self.puzzle,
                previous_state,
                placement_action.clue_id,
                placement_action.rect,
            )

        invalid_action = not result.valid
        if invalid_action and self.invalid_action_mode == "mask_required":
            raise ValueError(f"invalid action {action}: {result.reason}")

        if result.valid or invalid_action:
            self.state = result.state
        self.last_placement_action = placement_action

        self._prepare_actions()
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

        truncated = (not terminated) and self.episode_step_count >= self.max_steps
        reward = compute_reward(
            self.puzzle,
            previous_state,
            placement_action,
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

    def _prepare_actions(self) -> None:
        if self.puzzle is None or self.state is None:
            self._current_actions = []
            self._current_action_mask = np.zeros(self.max_actions, dtype=np.bool_)
            return
        self._current_actions = legal_placement_actions(
            self.puzzle,
            self.state,
            base_candidates_by_clue=self._base_candidates_by_clue,
            episode_mode=self.episode_mode,
        )
        self._current_action_mask = build_action_mask(
            len(self._current_actions),
            self.max_actions,
        )

    def _observation(self) -> dict[str, np.ndarray]:
        if self.puzzle is None or self.state is None:
            raise RuntimeError("environment has no active puzzle")
        return encode_observation(
            self.puzzle,
            self.state,
            self._current_actions,
            max_rows=self.max_rows,
            max_cols=self.max_cols,
            max_actions=self.max_actions,
            include_solution_hint=self.include_solution_hint,
        )

    def _action_mask(self) -> np.ndarray:
        return self._current_action_mask.copy()

    def _dead_end(self) -> bool:
        if self.puzzle is None or self.state is None:
            return False
        if self.state.success:
            return False
        if len(self._current_actions) == 0:
            return True
        if self.use_solver_dead_end_check:
            return not self._solution_still_reachable()
        return False

    def _solution_still_reachable(self) -> bool:
        if self.puzzle is None or self.state is None or self.puzzle.solution is None:
            return True
        from patches.simulation import can_place

        for clue in self.puzzle.clues:
            solution_rect = self.puzzle.solution_rect(clue.id)
            if solution_rect is None:
                return True
            current = self.state.patches.get(clue.id)
            if current is not None:
                if self.episode_mode == "commit_only" and current != solution_rect:
                    return False
                continue
            if not can_place(self.puzzle, self.state, clue.id, solution_rect).valid:
                return False
        return True

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
        mask = self._action_mask()
        return {
            "action_mask": mask if self.use_action_mask else mask,
            "action_count": len(self._current_actions),
            "success": self.state.success,
            "invalid_action": invalid_action,
            "invalid_reason": invalid_reason,
            "dead_end": dead_end,
            "truncated": truncated,
            "episode_id": self.episode_id,
            "episode_step_count": self.episode_step_count,
            "simulator_step_count": self.state.step_count,
            "placed_count": len(self.state.patches),
            "clue_count": len(self.puzzle.clues),
            "covered_count": self.state.covered_count,
            "total_cells": self.puzzle.total_cells,
            "puzzle_seed": self.puzzle.seed,
            "difficulty": self.puzzle.difficulty,
            "cumulative_reward": self.cumulative_reward,
            "last_placement": self.last_placement_action,
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
