import numpy as np

from patches.simulation import new_game
from patches.simulation.puzzle import Clue, Puzzle
from patches.simulation.utils import Rect, ShapeType
from patches.training.actions import legal_placement_actions
from patches.training.observations import (
    CLUE_CELL,
    COVERED_CELL,
    UNPLACED_CLUE,
    VALID_CELL,
    encode_observation,
)


def tiny_puzzle() -> Puzzle:
    clues = [
        Clue(id=0, pos=(0, 0), number=2, shape=ShapeType.WIDE),
        Clue(id=1, pos=(1, 0), number=2, shape=ShapeType.WIDE),
    ]
    solution = [Rect(0, 0, 1, 2), Rect(1, 0, 1, 2)]
    return Puzzle(rows=2, cols=2, clues=clues, solution=solution)


def test_grid_channels_and_padding_are_stable() -> None:
    puzzle = tiny_puzzle()
    state = new_game(puzzle)
    actions = legal_placement_actions(puzzle, state)

    obs = encode_observation(
        puzzle,
        state,
        actions,
        max_rows=3,
        max_cols=3,
        max_actions=4,
    )

    assert obs["grid"].dtype == np.float32
    assert obs["grid"][VALID_CELL, :2, :2].sum() == 4
    assert obs["grid"][VALID_CELL, 2, :].sum() == 0
    assert obs["grid"][COVERED_CELL].sum() == 0
    assert obs["grid"][CLUE_CELL].sum() == 2
    assert obs["grid"][UNPLACED_CLUE].sum() == 2
    assert obs["candidates"].shape[0] == 4
    assert obs["candidate_footprints"].shape == (4, 3, 3)
    assert obs["candidates"][len(actions) :].sum() == 0
    assert obs["candidate_footprints"][len(actions) :].sum() == 0


def test_candidate_footprints_match_rectangles() -> None:
    puzzle = tiny_puzzle()
    state = new_game(puzzle)
    actions = legal_placement_actions(puzzle, state)

    obs = encode_observation(
        puzzle,
        state,
        actions,
        max_rows=2,
        max_cols=2,
        max_actions=2,
    )

    for index, action in enumerate(actions):
        expected = np.zeros((2, 2), dtype=np.float32)
        for cell in action.rect.cells():
            expected[cell] = 1.0
        assert np.array_equal(obs["candidate_footprints"][index], expected)
