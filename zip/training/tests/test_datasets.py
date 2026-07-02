from zip.simulation.puzzle import Puzzle
from zip.training.datasets import solution_examples


SNAKE_2X2 = (
    (0, 0),
    (0, 1),
    (1, 1),
    (1, 0),
)


def tiny_puzzle() -> Puzzle:
    return Puzzle(
        rows=2,
        cols=2,
        waypoints=(SNAKE_2X2[0], SNAKE_2X2[-1]),
        walls=frozenset(),
        solution=SNAKE_2X2,
    )


def test_solution_examples_convert_path_to_actions() -> None:
    examples = solution_examples(tiny_puzzle())

    assert [example.action for example in examples] == [3, 1, 2]
    assert examples[0].observation.shape == (14, 2, 2)
