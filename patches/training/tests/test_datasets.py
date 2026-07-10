from patches.simulation.puzzle import Clue, Puzzle
from patches.simulation.utils import Rect, ShapeType
from patches.training.datasets import solution_examples


def tiny_puzzle() -> Puzzle:
    clues = [
        Clue(id=0, pos=(0, 0), number=2, shape=ShapeType.WIDE),
        Clue(id=1, pos=(1, 0), number=2, shape=ShapeType.WIDE),
    ]
    solution = [Rect(0, 0, 1, 2), Rect(1, 0, 1, 2)]
    return Puzzle(rows=2, cols=2, clues=clues, solution=solution)


def test_solution_examples_form_legal_trajectory() -> None:
    examples = solution_examples(tiny_puzzle(), max_actions=4)

    assert len(examples) == 2
    for example in examples:
        assert example.action_mask[example.action]
        assert example.observation["grid"].shape == (15, 2, 2)
