import pytest

from patches.simulation.puzzle import Clue, Puzzle
from patches.simulation.simulator import (
    candidate_rects,
    clear_patch,
    new_game,
    place,
    reset,
    state_from_solution,
    undo,
)
from patches.simulation.utils import Rect, ShapeType, rect_from_corners, shape_matches


def four_patch_puzzle() -> Puzzle:
    """A 4x4 board split into four 2x2 square patches."""

    clues = [
        Clue(id=0, pos=(0, 0), number=4, shape=ShapeType.SQUARE),
        Clue(id=1, pos=(0, 2), number=4, shape=ShapeType.SQUARE),
        Clue(id=2, pos=(2, 0), number=4, shape=ShapeType.SQUARE),
        Clue(id=3, pos=(2, 2), number=4, shape=ShapeType.SQUARE),
    ]
    solution = [
        Rect(0, 0, 2, 2),
        Rect(0, 2, 2, 2),
        Rect(2, 0, 2, 2),
        Rect(2, 2, 2, 2),
    ]
    return Puzzle(rows=4, cols=4, clues=clues, solution=solution)


def test_shape_matches_rules() -> None:
    assert shape_matches(ShapeType.SQUARE, 2, 2)
    assert not shape_matches(ShapeType.SQUARE, 1, 2)
    assert shape_matches(ShapeType.WIDE, 1, 3)
    assert not shape_matches(ShapeType.WIDE, 3, 1)
    assert shape_matches(ShapeType.TALL, 3, 1)
    assert not shape_matches(ShapeType.TALL, 1, 3)
    assert shape_matches(ShapeType.FREE, 1, 5)
    # single cell is a valid square
    assert shape_matches(ShapeType.SQUARE, 1, 1)


def test_rect_from_corners_normalizes() -> None:
    rect = rect_from_corners((3, 4), (1, 2))
    assert rect == Rect(1, 2, 3, 3)


def test_puzzle_rejects_area_mismatch() -> None:
    with pytest.raises(ValueError):
        Puzzle(rows=2, cols=2, clues=[Clue(0, (0, 0), 3, ShapeType.FREE)])


def test_candidate_rects_respects_shape_and_bounds() -> None:
    puzzle = four_patch_puzzle()
    rects = candidate_rects(puzzle, 0)
    # Only a 2x2 square containing (0,0) with no other clue fits.
    assert Rect(0, 0, 2, 2) in rects
    for rect in rects:
        assert rect.area == 4
        assert rect.height == rect.width


def test_place_valid_and_solve() -> None:
    puzzle = four_patch_puzzle()
    state = new_game(puzzle)

    for clue_id, rect in [
        (0, Rect(0, 0, 2, 2)),
        (1, Rect(0, 2, 2, 2)),
        (2, Rect(2, 0, 2, 2)),
    ]:
        result = place(puzzle, state, clue_id, rect)
        assert result.valid
        state = result.state
        assert not result.solved

    final = place(puzzle, state, 3, Rect(2, 2, 2, 2))
    assert final.valid
    assert final.solved
    assert final.state.success


def test_place_rejects_overlap_and_wrong_area() -> None:
    puzzle = four_patch_puzzle()
    state = place(puzzle, new_game(puzzle), 0, Rect(0, 0, 2, 2)).state

    overlap = place(puzzle, state, 1, Rect(0, 1, 2, 2))
    assert not overlap.valid
    assert "overlap" in overlap.reason

    wrong_area = place(puzzle, new_game(puzzle), 0, Rect(0, 0, 1, 2))
    assert not wrong_area.valid
    assert "area" in wrong_area.reason


def test_place_rejects_two_clues_in_one_patch() -> None:
    puzzle = four_patch_puzzle()
    # A 2x2 at (0,1) would swallow clue 0's neighbor region and clue 1.
    bad = place(puzzle, new_game(puzzle), 0, Rect(0, 1, 2, 2))
    assert not bad.valid


def test_undo_and_clear() -> None:
    puzzle = four_patch_puzzle()
    state = place(puzzle, new_game(puzzle), 0, Rect(0, 0, 2, 2)).state
    assert 0 in state.patches

    cleared = clear_patch(state, 0)
    assert 0 not in cleared.patches
    assert cleared.covered_count == 0

    restored = undo(cleared)
    assert 0 in restored.patches
    assert restored.covered_count == 4


def test_reset_clears_state() -> None:
    puzzle = four_patch_puzzle()
    state = place(puzzle, new_game(puzzle), 0, Rect(0, 0, 2, 2)).state
    fresh = reset(puzzle)
    assert fresh.covered_count == 0
    assert not fresh.patches


def test_state_from_solution_is_solved() -> None:
    puzzle = four_patch_puzzle()
    state = state_from_solution(puzzle)
    assert state.success
    assert state.covered_count == puzzle.total_cells


def test_numberless_clue_ignores_area_but_enforces_shape() -> None:
    # A single "wide" clue with no number can be any wide rectangle.
    puzzle = Puzzle(rows=2, cols=3, clues=[Clue(0, (0, 0), None, ShapeType.WIDE)])
    state = new_game(puzzle)

    # 2x3 is wide (width 3 > height 2) and covers the grid: valid and solved.
    result = place(puzzle, state, 0, Rect(0, 0, 2, 3))
    assert result.valid
    assert result.solved

    # A 2x2 square violates the wide shape even though a number is not required.
    bad = place(puzzle, new_game(puzzle), 0, Rect(0, 0, 2, 2))
    assert not bad.valid
    assert "shape" in bad.reason


def test_puzzle_allows_missing_numbers() -> None:
    # Numbers optional: sizes need not add up when a clue has no number.
    puzzle = Puzzle(
        rows=2,
        cols=2,
        clues=[
            Clue(0, (0, 0), 2, ShapeType.FREE),
            Clue(1, (1, 1), None, ShapeType.FREE),
        ],
    )
    assert puzzle.clue(1).number is None


def test_candidate_rects_numberless_offers_multiple_sizes() -> None:
    puzzle = Puzzle(rows=3, cols=3, clues=[Clue(0, (0, 0), None, ShapeType.FREE)])
    rects = candidate_rects(puzzle, 0)
    areas = {rect.area for rect in rects}
    # No number means many sizes are candidates (1x1 up through 3x3).
    assert len(areas) > 1
    assert all(rect.contains((0, 0)) for rect in rects)