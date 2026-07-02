from zip.simulation import new_game
from zip.simulation import step as simulator_step
from zip.simulation.puzzle import Puzzle
from zip.simulation.utils import Direction
from zip.training.rewards import RewardConfig, compute_reward


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


def snake_puzzle() -> Puzzle:
    return Puzzle(
        rows=3,
        cols=3,
        waypoints=((0, 0), (1, 1), (2, 2)),
        walls=frozenset(),
        solution=SNAKE_3X3,
    )


def test_valid_move_gets_new_cell_reward() -> None:
    puzzle = snake_puzzle()
    state = new_game(puzzle)
    result = simulator_step(puzzle, state, Direction.RIGHT)

    reward = compute_reward(
        puzzle,
        state,
        result,
        dead_end=False,
        truncated=False,
        config=RewardConfig(),
    )

    assert reward == 0.02


def test_waypoint_and_solve_rewards_are_added() -> None:
    puzzle = snake_puzzle()
    state = new_game(puzzle)
    for direction in (
        Direction.RIGHT,
        Direction.RIGHT,
        Direction.DOWN,
        Direction.LEFT,
    ):
        previous = state
        result = simulator_step(puzzle, state, direction)
        state = result.state

    waypoint_reward = compute_reward(
        puzzle,
        previous,
        result,
        dead_end=False,
        truncated=False,
        config=RewardConfig(),
    )
    assert waypoint_reward == 1.02

    for direction in (
        Direction.LEFT,
        Direction.DOWN,
        Direction.RIGHT,
        Direction.RIGHT,
    ):
        previous = state
        result = simulator_step(puzzle, state, direction)
        state = result.state

    solve_reward = compute_reward(
        puzzle,
        previous,
        result,
        dead_end=False,
        truncated=False,
        config=RewardConfig(),
    )
    assert solve_reward == 11.02


def test_invalid_dead_end_and_truncate_penalties() -> None:
    puzzle = snake_puzzle()
    state = new_game(puzzle)
    invalid = simulator_step(puzzle, state, Direction.UP)

    invalid_reward = compute_reward(
        puzzle,
        state,
        invalid,
        dead_end=False,
        truncated=True,
        config=RewardConfig(),
    )

    assert invalid_reward == -3.0

    valid = simulator_step(puzzle, state, Direction.RIGHT)
    dead_end_reward = compute_reward(
        puzzle,
        state,
        valid,
        dead_end=True,
        truncated=False,
        config=RewardConfig(),
    )

    assert dead_end_reward == -1.98
