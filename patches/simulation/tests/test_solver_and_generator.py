from patches.simulation.config import DifficultyConfig
from patches.simulation.generator import generate_puzzle_with_report
from patches.simulation.puzzle import Clue, Puzzle
from patches.simulation.solver import find_solution, solve
from patches.simulation.utils import Rect, ShapeType, iter_grid


def four_patch_puzzle() -> Puzzle:
    clues = [
        Clue(id=0, pos=(0, 0), number=4, shape=ShapeType.SQUARE),
        Clue(id=1, pos=(0, 2), number=4, shape=ShapeType.SQUARE),
        Clue(id=2, pos=(2, 0), number=4, shape=ShapeType.SQUARE),
        Clue(id=3, pos=(2, 2), number=4, shape=ShapeType.SQUARE),
    ]
    return Puzzle(rows=4, cols=4, clues=clues)


def test_solver_finds_unique_solution() -> None:
    result = solve(four_patch_puzzle(), timeout_seconds=2.0)
    assert result.solution_count == 1
    assert result.unique_solution is True


def test_solver_detects_multiple_solutions() -> None:
    # A 2x2 grid with two area-2 FREE clues on opposite corners can tile two
    # ways: two horizontal rows, or two vertical columns.
    clues = [
        Clue(id=0, pos=(0, 0), number=2, shape=ShapeType.FREE),
        Clue(id=1, pos=(1, 1), number=2, shape=ShapeType.FREE),
    ]
    puzzle = Puzzle(rows=2, cols=2, clues=clues)
    result = solve(puzzle, max_solutions=2, timeout_seconds=2.0)
    assert result.solution_count == 2
    assert result.unique_solution is False


def test_solver_rejects_impossible_shape() -> None:
    # Area 3 as SQUARE is impossible (3 is not a perfect square).
    puzzle = Puzzle(rows=1, cols=3, clues=[Clue(0, (0, 0), 3, ShapeType.SQUARE)])
    result = solve(puzzle, timeout_seconds=1.0)
    assert result.solution_count == 0
    assert result.unique_solution is False


def test_find_solution_covers_every_cell() -> None:
    puzzle = four_patch_puzzle()
    solution = find_solution(puzzle, timeout_seconds=2.0)
    assert solution is not None
    covered = [cell for rect in solution for cell in rect.cells()]
    assert sorted(covered) == sorted(iter_grid(puzzle.rows, puzzle.cols))


def test_generator_super_easy_is_valid() -> None:
    result = generate_puzzle_with_report("super_easy", seed=7)
    puzzle = result.puzzle

    assert puzzle.solution is not None
    assert sum(clue.number for clue in puzzle.clues) == puzzle.total_cells

    covered = [cell for rect in puzzle.solution for cell in rect.cells()]
    assert sorted(covered) == sorted(iter_grid(puzzle.rows, puzzle.cols))
    # Every clue lies inside exactly one solution rectangle.
    for clue in puzzle.clues:
        assert puzzle.solution_rect(clue.id) is not None
    assert result.solution_count >= 1


def test_generator_medium_is_unique() -> None:
    result = generate_puzzle_with_report("medium", seed=11)
    assert result.unique_solution is True
    assert result.puzzle.unique_solution is True


def test_generator_respects_board_size() -> None:
    result = generate_puzzle_with_report("easy", seed=3)
    assert (result.puzzle.rows, result.puzzle.cols) == (6, 6)
    assert result.puzzle.difficulty == "easy"


def test_generator_unique_on_small_config() -> None:
    config = DifficultyConfig(
        name="tiny",
        rows=4,
        cols=4,
        min_patch_area=1,
        max_patch_area=6,
        free_shape_ratio=0.2,
        require_unique_solution=True,
        max_generation_attempts=120,
        generation_timeout_seconds=1.0,
        solver_timeout_seconds=1.0,
    )
    result = generate_puzzle_with_report(config, seed=5)
    assert result.unique_solution is True
