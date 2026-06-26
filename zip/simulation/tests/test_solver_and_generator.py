import random

from zip.simulation.config import DifficultyConfig
from zip.simulation.generator import generate_hamiltonian_path, generate_puzzle_with_report
from zip.simulation.puzzle import Puzzle
from zip.simulation.solver import solve
from zip.simulation.utils import canonical_wall, iter_grid, neighbors, solution_edges


SNAKE_3X3 = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 2),
    (1, 1),
    (1, 0),
    (2, 0),
    (2, 1),
    (2, 2),
)


def walls_for_only_path(path: tuple[tuple[int, int], ...], rows: int, cols: int) -> frozenset:
    path_edges = solution_edges(path)
    walls = set()
    for cell in iter_grid(rows, cols):
        for adjacent in neighbors(rows, cols, cell):
            if adjacent < cell:
                continue
            wall = canonical_wall(cell, adjacent)
            if wall not in path_edges:
                walls.add(wall)
    return frozenset(walls)


def longest_straight_run(path: list[tuple[int, int]]) -> int:
    longest = 0
    current = 0
    previous_delta = None
    for a, b in zip(path, path[1:]):
        delta = b[0] - a[0], b[1] - a[1]
        if delta == previous_delta:
            current += 1
        else:
            current = 1
            previous_delta = delta
        longest = max(longest, current)
    return longest


def test_solver_finds_known_unique_solution() -> None:
    puzzle = Puzzle(
        rows=3,
        cols=3,
        waypoints=(SNAKE_3X3[0], SNAKE_3X3[4], SNAKE_3X3[-1]),
        walls=walls_for_only_path(SNAKE_3X3, 3, 3),
        solution=SNAKE_3X3,
    )

    result = solve(puzzle, timeout_seconds=1.0)

    assert result.solution_count == 1
    assert result.unique_solution is True
    assert result.solutions[0] == list(SNAKE_3X3)


def test_solver_rejects_impossible_puzzle() -> None:
    puzzle = Puzzle(rows=2, cols=2, waypoints=((0, 0), (1, 1)))

    result = solve(puzzle, timeout_seconds=1.0)

    assert result.solution_count == 0
    assert result.unique_solution is False


def test_solver_stops_after_two_solutions_for_uniqueness() -> None:
    puzzle = Puzzle(rows=3, cols=3, waypoints=((0, 0), (2, 2)))

    result = solve(puzzle, max_solutions=2, timeout_seconds=2.0)

    assert result.solution_count == 2
    assert result.unique_solution is False
    assert len(result.solutions) == 2


def test_solver_timeout_reports_unknown_uniqueness() -> None:
    puzzle = Puzzle(rows=4, cols=4, waypoints=((0, 0), (3, 3)))

    result = solve(puzzle, max_solutions=2, timeout_seconds=0.000000001)

    assert result.timed_out
    assert result.unique_solution is None


def test_generator_outputs_valid_solvable_puzzle() -> None:
    result = generate_puzzle_with_report("super_easy", seed=123)
    puzzle = result.puzzle

    assert len(puzzle.solution or ()) == puzzle.total_cells
    assert len(set(puzzle.solution or ())) == puzzle.total_cells
    assert puzzle.waypoints[0] == puzzle.solution[0]
    assert puzzle.waypoints[-1] == puzzle.solution[-1]
    assert solution_edges(puzzle.solution or ()) & set(puzzle.walls) == set()
    assert result.solution_count >= 1


def test_large_generated_paths_avoid_board_length_sweeps() -> None:
    for size in (7, 8):
        for seed in range(3):
            path = generate_hamiltonian_path(
                size,
                size,
                rng=random.Random(seed),
                start_position_policy="corner",
            )

            assert len(path) == size * size
            assert len(set(path)) == size * size
            assert longest_straight_run(path) <= 4


def test_hard_and_expert_generation_uses_requested_board_sizes() -> None:
    expectations = {
        "hard": ((7, 7), 8),
        "expert": ((8, 8), 12),
    }

    for difficulty, (size, max_walls) in expectations.items():
        result = generate_puzzle_with_report(difficulty, seed=123)
        puzzle = result.puzzle

        assert (puzzle.rows, puzzle.cols) == size
        assert puzzle.difficulty == difficulty
        assert len(puzzle.walls) <= max_walls
        assert result.unique_solution is True
        assert puzzle.unique_solution is True


def test_generator_can_enforce_uniqueness_on_tiny_config() -> None:
    config = DifficultyConfig(
        name="tiny",
        rows=3,
        cols=3,
        waypoint_spacing=(2, 3),
        total_waypoints=None,
        wall_count=(4, 8),
        require_unique_solution=True,
        max_generation_attempts=80,
        generation_timeout_seconds=1.0,
        solver_timeout_seconds=1.0,
        allow_multi_solution=False,
        start_position_policy="corner",
    )

    result = generate_puzzle_with_report(config, seed=44)

    assert result.unique_solution is True
    assert result.puzzle.unique_solution is True
