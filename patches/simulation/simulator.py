"""Public simulator API for Patches puzzles."""

from __future__ import annotations

from .puzzle import Puzzle
from .rules import PlacementValidation, is_solved as rules_is_solved, validate_placement
from .state import GameState, StepResult
from .utils import Rect, rect_in_bounds, shape_matches


def new_game(puzzle: Puzzle) -> GameState:
    state = GameState(rows=puzzle.rows, cols=puzzle.cols)
    solved = rules_is_solved(puzzle, state)
    state.done = solved
    state.success = solved
    return state


def reset(puzzle: Puzzle) -> GameState:
    return new_game(puzzle)


def candidate_rects(puzzle: Puzzle, clue_id: int) -> list[Rect]:
    """All rectangles that are bounds/area/shape/single-clue valid for a clue.

    Occupancy of other patches is ignored; this is the placement search space
    used by hints and the solver.
    """

    clue = puzzle.clue(clue_id)
    if clue.number is not None:
        dimensions = _dimensions_for_number(clue.number, clue.shape)
    else:
        dimensions = _dimensions_for_shape(clue.shape, puzzle.rows, puzzle.cols)

    results: list[Rect] = []
    row, col = clue.pos
    for height, width in dimensions:
        # The clue cell must lie inside the rectangle.
        for top in range(row - height + 1, row + 1):
            for left in range(col - width + 1, col + 1):
                rect = Rect(top, left, height, width)
                if not rect_in_bounds(puzzle.rows, puzzle.cols, rect):
                    continue
                if _encloses_single_clue(puzzle, rect, clue_id):
                    results.append(rect)
    return results


def _dimensions_for_number(number: int, shape) -> list[tuple[int, int]]:
    dims: list[tuple[int, int]] = []
    for height in range(1, number + 1):
        if number % height != 0:
            continue
        width = number // height
        if shape_matches(shape, height, width):
            dims.append((height, width))
    return dims


def _dimensions_for_shape(shape, rows: int, cols: int) -> list[tuple[int, int]]:
    dims: list[tuple[int, int]] = []
    for height in range(1, rows + 1):
        for width in range(1, cols + 1):
            if shape_matches(shape, height, width):
                dims.append((height, width))
    return dims


def _encloses_single_clue(puzzle: Puzzle, rect: Rect, clue_id: int) -> bool:
    found_self = False
    for cell in rect.cells():
        clue = puzzle.clue_at(cell)
        if clue is None:
            continue
        if clue.id != clue_id:
            return False
        found_self = True
    return found_self


def can_place(puzzle: Puzzle, state: GameState, clue_id: int, rect: Rect) -> PlacementValidation:
    return validate_placement(puzzle, state, clue_id, rect)


def place(puzzle: Puzzle, state: GameState, clue_id: int, rect: Rect) -> StepResult:
    validation = validate_placement(puzzle, state, clue_id, rect)
    if not validation.valid:
        return StepResult(state=state.clone(), valid=False, reason=validation.reason, solved=state.success)

    next_state = state.clone()
    next_state.history.append(dict(state.patches))
    _remove_patch(next_state, clue_id)
    next_state.patches[clue_id] = rect
    for cell in rect.cells():
        next_state.assignment[cell] = clue_id
    next_state.step_count = state.step_count + 1

    solved = rules_is_solved(puzzle, next_state)
    next_state.done = solved
    next_state.success = solved
    return StepResult(state=next_state, valid=True, reason=None, solved=solved)


def clear_patch(state: GameState, clue_id: int) -> GameState:
    if clue_id not in state.patches:
        return state.clone()
    next_state = state.clone()
    next_state.history.append(dict(state.patches))
    _remove_patch(next_state, clue_id)
    next_state.step_count = state.step_count + 1
    next_state.done = False
    next_state.success = False
    return next_state


def undo(state: GameState) -> GameState:
    if not state.history:
        return state.clone()
    next_state = state.clone()
    previous_patches = next_state.history.pop()
    next_state.patches = dict(previous_patches)
    next_state.assignment = _rebuild_assignment(previous_patches)
    next_state.step_count = max(0, state.step_count - 1)
    next_state.done = False
    next_state.success = False
    return next_state


def is_solved(puzzle: Puzzle, state: GameState) -> bool:
    return rules_is_solved(puzzle, state)


def state_from_solution(puzzle: Puzzle, solution: list[Rect] | tuple[Rect, ...] | None = None) -> GameState:
    """Build a fully-placed state from a solution tiling (defaults to puzzle's)."""

    rects = solution if solution is not None else puzzle.solution
    if rects is None:
        raise ValueError("puzzle has no stored solution to build state from")

    state = new_game(puzzle)
    for rect in rects:
        enclosed = [puzzle.clue_at(cell) for cell in rect.cells() if puzzle.clue_at(cell) is not None]
        if len(enclosed) != 1:
            raise ValueError(f"solution rectangle must enclose exactly one clue: {rect!r}")
        result = place(puzzle, state, enclosed[0].id, rect)
        if not result.valid:
            raise ValueError(f"invalid solution rectangle {rect!r}: {result.reason}")
        state = result.state
    return state


def _remove_patch(state: GameState, clue_id: int) -> None:
    existing = state.patches.pop(clue_id, None)
    if existing is None:
        return
    for cell in existing.cells():
        if state.assignment.get(cell) == clue_id:
            del state.assignment[cell]


def _rebuild_assignment(patches: dict[int, Rect]) -> dict:
    assignment: dict = {}
    for clue_id, rect in patches.items():
        for cell in rect.cells():
            assignment[cell] = clue_id
    return assignment
