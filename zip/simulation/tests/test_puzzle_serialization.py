from zip.simulation.puzzle import Puzzle, puzzle_from_json, puzzle_to_json
from zip.simulation.utils import canonical_wall


def test_puzzle_serialization_round_trip() -> None:
    puzzle = Puzzle(
        rows=3,
        cols=3,
        waypoints=[(0, 0), (1, 1), (2, 2)],
        walls={canonical_wall((0, 0), (1, 0))},
        solution=[
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 2),
            (1, 1),
            (1, 0),
            (2, 0),
            (2, 1),
            (2, 2),
        ],
        difficulty="test",
        seed=123,
        unique_solution=True,
    )

    restored = puzzle_from_json(puzzle_to_json(puzzle))

    assert restored == puzzle
    assert restored.walls == frozenset({canonical_wall((1, 0), (0, 0))})


def test_partial_solution_json_is_allowed_for_external_debugging() -> None:
    puzzle = Puzzle(
        rows=3,
        cols=3,
        waypoints=[(0, 0), (2, 2)],
        solution=[(0, 0), (0, 1)],
    )

    assert puzzle.solution == ((0, 0), (0, 1))
