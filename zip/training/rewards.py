"""Reward calculation for Zip RL episodes."""

from __future__ import annotations

from dataclasses import dataclass

from zip.simulation import GameState, Puzzle, StepResult


@dataclass(frozen=True)
class RewardConfig:
    valid_step: float = 0.0
    new_cell: float = 0.02
    waypoint: float = 1.0
    solve: float = 10.0
    invalid_action: float = -2.0
    dead_end: float = -2.0
    truncate: float = -1.0


def compute_reward(
    puzzle: Puzzle,
    previous_state: GameState,
    result: StepResult,
    *,
    dead_end: bool,
    truncated: bool,
    config: RewardConfig,
) -> float:
    """Compute a transition reward without mutating simulator state."""

    del puzzle
    if not result.valid:
        reward = config.invalid_action
    else:
        reward = config.valid_step
        if len(result.state.visited) > len(previous_state.visited):
            reward += config.new_cell
        if result.state.next_waypoint_index > previous_state.next_waypoint_index:
            reward += config.waypoint
        if result.state.success:
            reward += config.solve
        if dead_end:
            reward += config.dead_end

    if truncated:
        reward += config.truncate

    return float(reward)
