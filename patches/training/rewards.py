"""Reward calculation for Patches RL episodes."""

from __future__ import annotations

from dataclasses import dataclass

from patches.simulation import GameState, Puzzle, StepResult

from .actions import PlacementAction


@dataclass(frozen=True)
class RewardConfig:
    valid_placement: float = 0.05
    covered_cell: float = 0.02
    placed_clue: float = 0.05
    solve: float = 10.0
    invalid_action: float = -2.0
    dead_end: float = -2.0
    truncate: float = -1.0
    non_solution_placement: float = 0.0
    solution_placement: float = 0.0


def compute_reward(
    puzzle: Puzzle,
    previous_state: GameState,
    action: PlacementAction | None,
    result: StepResult,
    *,
    dead_end: bool,
    truncated: bool,
    config: RewardConfig,
) -> float:
    """Compute a transition reward without mutating simulator state."""

    if not result.valid:
        reward = config.invalid_action
    else:
        reward = config.valid_placement
        newly_covered = max(0, result.state.covered_count - previous_state.covered_count)
        reward += config.covered_cell * newly_covered
        if action is not None and action.clue_id not in previous_state.patches:
            reward += config.placed_clue
        if action is not None and puzzle.solution is not None:
            solution_rect = puzzle.solution_rect(action.clue_id)
            if solution_rect == action.rect:
                reward += config.solution_placement
            elif config.non_solution_placement:
                reward += config.non_solution_placement
        if result.state.success:
            reward += config.solve
        if dead_end:
            reward += config.dead_end

    if truncated:
        reward += config.truncate
    return float(reward)
