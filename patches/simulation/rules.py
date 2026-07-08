"""Shared placement and solved-state rules.

Keep this module central. The simulator, solver, and UI all call through the
same validity checks so rule changes do not drift across the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass

from .puzzle import Puzzle
from .state import GameState
from .utils import Rect, rect_in_bounds, shape_matches


@dataclass(frozen=True)
class PlacementValidation:
    valid: bool
    clue_id: int | None
    reason: str | None = None


def enclosed_clue_ids(puzzle: Puzzle, rect: Rect) -> list[int]:
    ids: list[int] = []
    for cell in rect.cells():
        clue = puzzle.clue_at(cell)
        if clue is not None:
            ids.append(clue.id)
    return ids


def validate_placement(
    puzzle: Puzzle,
    state: GameState,
    clue_id: int,
    rect: Rect,
) -> PlacementValidation:
    """Validate placing ``rect`` as the patch for ``clue_id``.

    A placement replaces any existing patch for the same clue, so overlap is
    only checked against *other* clues' patches.
    """

    if clue_id not in {clue.id for clue in puzzle.clues}:
        return PlacementValidation(False, None, f"no clue with id {clue_id}")

    clue = puzzle.clue(clue_id)

    if not rect_in_bounds(puzzle.rows, puzzle.cols, rect):
        return PlacementValidation(False, clue_id, "patch is out of bounds")

    enclosed = enclosed_clue_ids(puzzle, rect)
    if clue_id not in enclosed:
        return PlacementValidation(False, clue_id, "patch must contain its own clue")
    if len(enclosed) > 1:
        return PlacementValidation(False, clue_id, "patch must contain exactly one clue")

    if clue.number is not None and rect.area != clue.number:
        return PlacementValidation(
            False, clue_id, f"patch area {rect.area} must equal clue number {clue.number}"
        )
    if not shape_matches(clue.shape, rect.height, rect.width):
        return PlacementValidation(
            False, clue_id, f"patch shape must be {clue.shape.value}"
        )

    for cell in rect.cells():
        owner = state.assignment.get(cell)
        if owner is not None and owner != clue_id:
            return PlacementValidation(False, clue_id, "patch overlaps another patch")

    return PlacementValidation(True, clue_id)


def is_solved(puzzle: Puzzle, state: GameState) -> bool:
    return (
        len(state.patches) == len(puzzle.clues)
        and state.covered_count == puzzle.total_cells
    )
