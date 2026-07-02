"""Observation encoding for Zip reinforcement-learning environments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from zip.simulation import GameState, Puzzle, legal_moves
from zip.simulation.utils import Direction, add_direction, in_bounds

VALID_CELL = 0
VISITED = 1
CURRENT_HEAD = 2
WAYPOINT = 3
CURRENT_OR_PAST_WAYPOINT = 4
NEXT_WAYPOINT = 5
FUTURE_WAYPOINT = 6
WAYPOINT_NUMBER_NORMALIZED = 7
NEXT_WAYPOINT_NUMBER_NORMALIZED = 8
WALL_NORTH = 9
WALL_SOUTH = 10
WALL_WEST = 11
WALL_EAST = 12
LEGAL_MOVE_TARGET = 13
SOLUTION_PATH_HINT = 14

BASE_CHANNEL_NAMES = (
    "valid_cell",
    "visited",
    "current_head",
    "waypoint",
    "current_or_past_waypoint",
    "next_waypoint",
    "future_waypoint",
    "waypoint_number_normalized",
    "next_waypoint_number_normalized",
    "wall_north",
    "wall_south",
    "wall_west",
    "wall_east",
    "legal_move_target",
)
SOLUTION_HINT_CHANNEL_NAME = "solution_path_hint_optional"


@dataclass(frozen=True)
class ObservationConfig:
    """Grid-observation bounds and optional channels."""

    max_rows: int
    max_cols: int
    include_solution_hint: bool = False

    @property
    def channels(self) -> int:
        return len(channel_names(self.include_solution_hint))

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.channels, self.max_rows, self.max_cols


def channel_names(include_solution_hint: bool = False) -> tuple[str, ...]:
    if include_solution_hint:
        return (*BASE_CHANNEL_NAMES, SOLUTION_HINT_CHANNEL_NAME)
    return BASE_CHANNEL_NAMES


def observation_shape(
    max_rows: int,
    max_cols: int,
    *,
    include_solution_hint: bool = False,
) -> tuple[int, int, int]:
    return ObservationConfig(max_rows, max_cols, include_solution_hint).shape


def make_observation_space(
    max_rows: int,
    max_cols: int,
    *,
    include_solution_hint: bool = False,
):
    """Create a Gymnasium Box space for encoded observations."""

    from gymnasium import spaces

    return spaces.Box(
        low=0.0,
        high=1.0,
        shape=observation_shape(
            max_rows,
            max_cols,
            include_solution_hint=include_solution_hint,
        ),
        dtype=np.float32,
    )


def encode_observation(
    puzzle: Puzzle,
    state: GameState,
    *,
    max_rows: int | None = None,
    max_cols: int | None = None,
    include_solution_hint: bool = False,
) -> np.ndarray:
    """Encode a simulator state as a channel-first float32 grid."""

    max_rows = puzzle.rows if max_rows is None else int(max_rows)
    max_cols = puzzle.cols if max_cols is None else int(max_cols)
    _validate_bounds(puzzle, max_rows, max_cols)

    grid = np.zeros(
        observation_shape(
            max_rows,
            max_cols,
            include_solution_hint=include_solution_hint,
        ),
        dtype=np.float32,
    )
    valid_slice = (slice(0, puzzle.rows), slice(0, puzzle.cols))
    grid[VALID_CELL][valid_slice] = 1.0

    for row, col in state.visited:
        grid[VISITED, row, col] = 1.0

    head_row, head_col = state.current_pos
    grid[CURRENT_HEAD, head_row, head_col] = 1.0

    waypoint_denominator = max(1, len(puzzle.waypoints) - 1)
    next_index = state.next_waypoint_index
    next_value = min(next_index, len(puzzle.waypoints) - 1) / waypoint_denominator
    grid[NEXT_WAYPOINT_NUMBER_NORMALIZED][valid_slice] = float(next_value)

    for index, (row, col) in enumerate(puzzle.waypoints):
        grid[WAYPOINT, row, col] = 1.0
        grid[WAYPOINT_NUMBER_NORMALIZED, row, col] = index / waypoint_denominator
        if index < next_index:
            grid[CURRENT_OR_PAST_WAYPOINT, row, col] = 1.0
        elif index == next_index:
            grid[NEXT_WAYPOINT, row, col] = 1.0
        else:
            grid[FUTURE_WAYPOINT, row, col] = 1.0

    _encode_walls(puzzle, grid)
    _encode_legal_targets(puzzle, state, grid)

    if include_solution_hint and puzzle.solution is not None:
        for row, col in puzzle.solution:
            grid[SOLUTION_PATH_HINT, row, col] = 1.0

    return grid


def _validate_bounds(puzzle: Puzzle, max_rows: int, max_cols: int) -> None:
    if max_rows <= 0 or max_cols <= 0:
        raise ValueError("max_rows and max_cols must be positive")
    if puzzle.rows > max_rows or puzzle.cols > max_cols:
        raise ValueError(
            "puzzle dimensions exceed observation bounds: "
            f"{puzzle.rows}x{puzzle.cols} > {max_rows}x{max_cols}"
        )


def _encode_walls(puzzle: Puzzle, grid: np.ndarray) -> None:
    direction_channels = {
        Direction.UP: WALL_NORTH,
        Direction.DOWN: WALL_SOUTH,
        Direction.LEFT: WALL_WEST,
        Direction.RIGHT: WALL_EAST,
    }
    for row in range(puzzle.rows):
        for col in range(puzzle.cols):
            cell = (row, col)
            for direction, channel in direction_channels.items():
                destination = add_direction(cell, direction)
                if (
                    not in_bounds(puzzle.rows, puzzle.cols, destination)
                    or puzzle.has_wall(cell, destination)
                ):
                    grid[channel, row, col] = 1.0


def _encode_legal_targets(puzzle: Puzzle, state: GameState, grid: np.ndarray) -> None:
    for direction in legal_moves(puzzle, state):
        row, col = add_direction(state.current_pos, direction)
        grid[LEGAL_MOVE_TARGET, row, col] = 1.0
