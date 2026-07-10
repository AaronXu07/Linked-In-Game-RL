from patches.simulation.puzzle import Clue, Puzzle
from patches.simulation.simulator import place
from patches.simulation.utils import Rect, ShapeType
from patches.training.actions import (
    build_action_mask,
    legal_placement_actions,
    precompute_candidates,
)


def four_patch_puzzle() -> Puzzle:
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


def test_actions_include_legal_solution_rectangles() -> None:
    from patches.simulation import new_game

    puzzle = four_patch_puzzle()
    state = new_game(puzzle)

    actions = legal_placement_actions(puzzle, state)

    available = {(action.clue_id, action.rect) for action in actions}
    expected = {
        (clue.id, puzzle.solution_rect(clue.id))
        for clue in puzzle.clues
    }
    assert expected <= available


def test_actions_filter_placed_clues_in_commit_only_mode() -> None:
    from patches.simulation import new_game

    puzzle = four_patch_puzzle()
    state = new_game(puzzle)
    result = place(puzzle, state, 0, puzzle.solution_rect(0))
    assert result.valid

    actions = legal_placement_actions(
        puzzle,
        result.state,
        base_candidates_by_clue=precompute_candidates(puzzle),
    )

    assert {action.clue_id for action in actions} == {1, 2, 3}


def test_action_mask_overflow_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="exceeds max_actions"):
        build_action_mask(3, 2)
