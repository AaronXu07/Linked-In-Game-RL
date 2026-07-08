"""Standalone Patches puzzle solver and uniqueness checker.

Models the puzzle as an exact-cover problem: choose one rectangle per clue so
that every grid cell is covered exactly once. Uses a most-constrained-cell
heuristic and stops early once enough solutions are found.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass

from .puzzle import Puzzle
from .simulator import candidate_rects
from .utils import Rect, coordinate_to_index, iter_grid


@dataclass(frozen=True)
class SolverResult:
    solutions: list[list[Rect]]
    solution_count: int
    unique_solution: bool | None
    timed_out: bool
    elapsed_seconds: float


@dataclass(frozen=True)
class _Candidate:
    clue_id: int
    rect: Rect
    mask: int


def solve(
    puzzle: Puzzle,
    *,
    max_solutions: int = 2,
    timeout_seconds: float = 5.0,
    return_solutions: bool = True,
) -> SolverResult:
    """Find up to ``max_solutions`` valid tilings for a puzzle."""

    if max_solutions <= 0:
        raise ValueError("max_solutions must be positive")

    started_at = time.monotonic()
    deadline = started_at + timeout_seconds if timeout_seconds > 0 else None

    cols = puzzle.cols
    full_mask = (1 << (puzzle.rows * puzzle.cols)) - 1

    candidates: list[_Candidate] = []
    covers: dict[int, list[int]] = defaultdict(list)  # cell index -> candidate indices
    feasible = True
    for clue in puzzle.clues:
        clue_candidates = candidate_rects(puzzle, clue.id)
        if not clue_candidates:
            feasible = False
            break
        for rect in clue_candidates:
            mask = 0
            for cell in rect.cells():
                mask |= 1 << coordinate_to_index(cols, cell)
            index = len(candidates)
            candidates.append(_Candidate(clue.id, rect, mask))
            for cell in rect.cells():
                covers[coordinate_to_index(cols, cell)].append(index)

    solutions: list[list[Rect]] = []
    counter = {"count": 0}
    timed_out = {"value": False}

    all_cell_indices = [coordinate_to_index(cols, cell) for cell in iter_grid(puzzle.rows, puzzle.cols)]

    def choose_cell(covered: int, used_clues: frozenset[int]) -> tuple[int, list[int]] | None:
        best_cell = -1
        best_options: list[int] | None = None
        for cell_index in all_cell_indices:
            if covered & (1 << cell_index):
                continue
            options = [
                cand_index
                for cand_index in covers[cell_index]
                if candidates[cand_index].clue_id not in used_clues
                and not (candidates[cand_index].mask & covered)
            ]
            if best_options is None or len(options) < len(best_options):
                best_cell = cell_index
                best_options = options
                if len(options) <= 1:
                    break
        if best_options is None:
            return None
        return best_cell, best_options

    def recurse(covered: int, used_clues: frozenset[int], placed: list[Rect]) -> bool:
        """Return True when the search should stop (enough solutions / timeout)."""

        if deadline is not None and time.monotonic() >= deadline:
            timed_out["value"] = True
            return True

        if covered == full_mask:
            counter["count"] += 1
            if return_solutions:
                solutions.append(list(placed))
            return counter["count"] >= max_solutions

        chosen = choose_cell(covered, used_clues)
        if chosen is None:
            return False
        _, options = chosen
        for cand_index in options:
            candidate = candidates[cand_index]
            placed.append(candidate.rect)
            stop = recurse(
                covered | candidate.mask,
                used_clues | {candidate.clue_id},
                placed,
            )
            placed.pop()
            if stop:
                return True
        return False

    if feasible:
        recurse(0, frozenset(), [])

    elapsed = time.monotonic() - started_at
    solution_count = counter["count"]
    if timed_out["value"]:
        unique_solution = None
    elif solution_count == 0:
        unique_solution = False
    elif solution_count == 1 and max_solutions > 1:
        unique_solution = True
    else:
        unique_solution = False

    return SolverResult(
        solutions=solutions,
        solution_count=solution_count,
        unique_solution=unique_solution,
        timed_out=timed_out["value"],
        elapsed_seconds=elapsed,
    )


def find_solution(puzzle: Puzzle, *, timeout_seconds: float = 5.0) -> list[Rect] | None:
    result = solve(puzzle, max_solutions=1, timeout_seconds=timeout_seconds)
    return result.solutions[0] if result.solutions else None
