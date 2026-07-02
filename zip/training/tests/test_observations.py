import numpy as np

from zip.simulation import legal_moves, new_game
from zip.simulation import step as simulator_step
from zip.simulation.puzzle import Puzzle
from zip.simulation.utils import Direction, add_direction, canonical_wall
from zip.training.observations import (
    CURRENT_HEAD,
    FUTURE_WAYPOINT,
    LEGAL_MOVE_TARGET,
    NEXT_WAYPOINT,
    VALID_CELL,
    VISITED,
    WALL_EAST,
    WALL_NORTH,
    WALL_SOUTH,
    WALL_WEST,
    encode_observation,
)


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


def snake_puzzle(**overrides: object) -> Puzzle:
    data = {
        "rows": 3,
        "cols": 3,
        "waypoints": ((0, 0), (1, 1), (2, 2)),
        "walls": frozenset(),
        "solution": SNAKE_3X3,
    }
    data.update(overrides)
    return Puzzle(**data)


def test_observation_shape_dtype_and_padding() -> None:
    puzzle = snake_puzzle()
    state = new_game(puzzle)

    obs = encode_observation(puzzle, state, max_rows=4, max_cols=5)

    assert obs.shape == (14, 4, 5)
    assert obs.dtype == np.float32
    assert obs[VALID_CELL, :3, :3].sum() == 9
    assert obs[:, 3, :].sum() == 0
    assert obs[:, :, 3:].sum() == 0


def test_current_visited_and_legal_targets_match_state() -> None:
    puzzle = snake_puzzle()
    state = simulator_step(puzzle, new_game(puzzle), Direction.RIGHT).state

    obs = encode_observation(puzzle, state)
    legal_targets = {
        add_direction(state.current_pos, direction)
        for direction in legal_moves(puzzle, state)
    }
    encoded_targets = {
        (row, col)
        for row in range(puzzle.rows)
        for col in range(puzzle.cols)
        if obs[LEGAL_MOVE_TARGET, row, col] == 1.0
    }

    assert obs[CURRENT_HEAD].sum() == 1.0
    assert obs[CURRENT_HEAD, 0, 1] == 1.0
    assert obs[VISITED].sum() == len(state.visited)
    assert encoded_targets == legal_targets


def test_wall_channels_include_boundaries_and_internal_walls() -> None:
    puzzle = snake_puzzle(
        walls={canonical_wall((1, 1), (1, 2))},
        solution=None,
    )
    state = new_game(puzzle)

    obs = encode_observation(puzzle, state)

    assert obs[WALL_NORTH, 0, 0] == 1.0
    assert obs[WALL_WEST, 0, 0] == 1.0
    assert obs[WALL_SOUTH, 2, 2] == 1.0
    assert obs[WALL_EAST, 2, 2] == 1.0
    assert obs[WALL_EAST, 1, 1] == 1.0
    assert obs[WALL_WEST, 1, 2] == 1.0


def test_waypoint_channels_change_after_reaching_next_waypoint() -> None:
    puzzle = snake_puzzle()
    state = new_game(puzzle)
    before = encode_observation(puzzle, state)

    state = simulator_step(puzzle, state, Direction.RIGHT).state
    state = simulator_step(puzzle, state, Direction.DOWN).state
    after = encode_observation(puzzle, state)

    assert before[NEXT_WAYPOINT, 1, 1] == 1.0
    assert before[FUTURE_WAYPOINT, 2, 2] == 1.0
    assert after[NEXT_WAYPOINT, 2, 2] == 1.0
    assert after[FUTURE_WAYPOINT].sum() == 0.0
