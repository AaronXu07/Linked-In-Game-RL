"""Shared movement and solved-state rules.

Keep this module central. The simulator, solver, and UI all call through the
same transition checks so rule changes do not drift across the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AbstractSet

from .puzzle import Puzzle
from .state import GameState
from .utils import Coordinate, Direction, add_direction, are_adjacent, in_bounds


@dataclass(frozen=True)
class MoveValidation:
    valid: bool
    destination: Coordinate
    reason: str | None = None


def validate_destination(
    puzzle: Puzzle,
    *,
    current_pos: Coordinate,
    visited: AbstractSet[Coordinate],
    next_waypoint_index: int,
    destination: Coordinate,
    done: bool = False,
) -> MoveValidation:
    if done:
        return MoveValidation(False, destination, "game is already complete")
    if not in_bounds(puzzle.rows, puzzle.cols, destination):
        return MoveValidation(False, destination, "destination is out of bounds")
    if not are_adjacent(current_pos, destination):
        return MoveValidation(False, destination, "moves must be one cardinal step")
    if puzzle.has_wall(current_pos, destination):
        return MoveValidation(False, destination, "a wall blocks that move")
    if destination in visited:
        return MoveValidation(False, destination, "the path cannot revisit a cell")

    waypoint_index = puzzle.waypoint_index(destination)
    if waypoint_index is not None:
        if waypoint_index < next_waypoint_index:
            return MoveValidation(False, destination, "that waypoint was already visited")
        if waypoint_index > next_waypoint_index:
            return MoveValidation(False, destination, "waypoints must be visited in order")
        if waypoint_index == len(puzzle.waypoints) - 1 and len(visited) + 1 < puzzle.total_cells:
            return MoveValidation(False, destination, "the final waypoint must be the last cell")

    return MoveValidation(True, destination)


def validate_move(puzzle: Puzzle, state: GameState, direction: Direction) -> MoveValidation:
    destination = add_direction(state.current_pos, direction)
    return validate_destination(
        puzzle,
        current_pos=state.current_pos,
        visited=state.visited,
        next_waypoint_index=state.next_waypoint_index,
        destination=destination,
        done=state.done,
    )


def advance_waypoint_index(puzzle: Puzzle, next_waypoint_index: int, destination: Coordinate) -> int:
    waypoint_index = puzzle.waypoint_index(destination)
    if waypoint_index == next_waypoint_index:
        return next_waypoint_index + 1
    return next_waypoint_index


def is_solved(puzzle: Puzzle, state: GameState) -> bool:
    return (
        len(state.visited) == puzzle.total_cells
        and state.current_pos == puzzle.final_waypoint
        and state.next_waypoint_index >= len(puzzle.waypoints)
    )
