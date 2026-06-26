"""Standalone Zip puzzle solver and uniqueness checker."""

from __future__ import annotations

import time
from dataclasses import dataclass

from .puzzle import Puzzle
from .rules import advance_waypoint_index, validate_destination
from .utils import Coordinate, iter_grid, neighbors


@dataclass(frozen=True)
class SolverResult:
    solutions: list[list[Coordinate]]
    solution_count: int
    unique_solution: bool | None
    timed_out: bool
    elapsed_seconds: float


@dataclass
class _Frame:
    current: Coordinate
    visited: set[Coordinate]
    next_waypoint_index: int
    candidates: list[Coordinate]
    candidate_index: int = 0


def solve(
    puzzle: Puzzle,
    *,
    max_solutions: int = 2,
    timeout_seconds: float = 5.0,
    return_solutions: bool = True,
) -> SolverResult:
    """Find up to `max_solutions` valid Hamiltonian paths for a puzzle."""

    if max_solutions <= 0:
        raise ValueError("max_solutions must be positive")

    started_at = time.monotonic()
    deadline = started_at + timeout_seconds if timeout_seconds > 0 else None
    path = [puzzle.start]
    visited = {puzzle.start}
    stack = [
        _Frame(
            current=puzzle.start,
            visited=visited,
            next_waypoint_index=1,
            candidates=_ordered_candidates(puzzle, puzzle.start, visited, 1),
        )
    ]

    solution_count = 0
    solutions: list[list[Coordinate]] = []
    timed_out = False

    while stack:
        if deadline is not None and time.monotonic() >= deadline:
            timed_out = True
            break

        frame = stack[-1]
        if _is_complete(puzzle, frame, path):
            solution_count += 1
            if return_solutions:
                solutions.append(list(path))
            if solution_count >= max_solutions:
                break
            stack.pop()
            path.pop()
            continue

        if frame.candidate_index >= len(frame.candidates):
            stack.pop()
            path.pop()
            continue

        destination = frame.candidates[frame.candidate_index]
        frame.candidate_index += 1
        validation = validate_destination(
            puzzle,
            current_pos=frame.current,
            visited=frame.visited,
            next_waypoint_index=frame.next_waypoint_index,
            destination=destination,
        )
        if not validation.valid:
            continue

        next_visited = set(frame.visited)
        next_visited.add(destination)
        next_waypoint_index = advance_waypoint_index(
            puzzle, frame.next_waypoint_index, destination
        )
        if not _remaining_cells_connected(puzzle, destination, next_visited):
            continue

        path.append(destination)
        stack.append(
            _Frame(
                current=destination,
                visited=next_visited,
                next_waypoint_index=next_waypoint_index,
                candidates=_ordered_candidates(
                    puzzle, destination, next_visited, next_waypoint_index
                ),
            )
        )

    elapsed = time.monotonic() - started_at
    if timed_out:
        unique_solution = None
    elif solution_count == 1 and max_solutions > 1:
        unique_solution = True
    elif solution_count == 0:
        unique_solution = False
    elif solution_count > 1:
        unique_solution = False
    else:
        unique_solution = None

    return SolverResult(
        solutions=solutions,
        solution_count=solution_count,
        unique_solution=unique_solution,
        timed_out=timed_out,
        elapsed_seconds=elapsed,
    )


def find_solution(puzzle: Puzzle, *, timeout_seconds: float = 5.0) -> list[Coordinate] | None:
    result = solve(puzzle, max_solutions=1, timeout_seconds=timeout_seconds)
    return result.solutions[0] if result.solutions else None


def _is_complete(puzzle: Puzzle, frame: _Frame, path: list[Coordinate]) -> bool:
    return (
        len(path) == puzzle.total_cells
        and frame.current == puzzle.final_waypoint
        and frame.next_waypoint_index >= len(puzzle.waypoints)
    )


def _ordered_candidates(
    puzzle: Puzzle,
    current: Coordinate,
    visited: set[Coordinate],
    next_waypoint_index: int,
) -> list[Coordinate]:
    candidates = neighbors(puzzle.rows, puzzle.cols, current)
    candidates.sort(
        key=lambda cell: (
            _onward_count(puzzle, cell, visited | {cell}, next_waypoint_index),
            cell,
        )
    )
    return candidates


def _onward_count(
    puzzle: Puzzle,
    current: Coordinate,
    visited: set[Coordinate],
    next_waypoint_index: int,
) -> int:
    count = 0
    adjusted_next_waypoint = advance_waypoint_index(
        puzzle, next_waypoint_index, current
    )
    for candidate in neighbors(puzzle.rows, puzzle.cols, current):
        if validate_destination(
            puzzle,
            current_pos=current,
            visited=visited,
            next_waypoint_index=adjusted_next_waypoint,
            destination=candidate,
        ).valid:
            count += 1
    return count


def _remaining_cells_connected(
    puzzle: Puzzle, head: Coordinate, visited: set[Coordinate]
) -> bool:
    remaining = set(iter_grid(puzzle.rows, puzzle.cols)) - visited
    if not remaining:
        return True

    entry_cells = [
        cell
        for cell in neighbors(puzzle.rows, puzzle.cols, head)
        if cell in remaining and not puzzle.has_wall(head, cell)
    ]
    if not entry_cells:
        return False

    start = entry_cells[0]
    frontier = [start]
    seen = {start}
    while frontier:
        cell = frontier.pop()
        for candidate in neighbors(puzzle.rows, puzzle.cols, cell):
            if (
                candidate in remaining
                and candidate not in seen
                and not puzzle.has_wall(cell, candidate)
            ):
                seen.add(candidate)
                frontier.append(candidate)
    return seen == remaining
