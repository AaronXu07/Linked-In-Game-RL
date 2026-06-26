from zip.simulation.puzzle import Puzzle
from zip.simulation.simulator import is_solved, new_game, reset, step, undo
from zip.simulation.utils import Direction, canonical_wall


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


def test_revisit_is_invalid_and_does_not_mutate_state() -> None:
    puzzle = snake_puzzle()
    state = new_game(puzzle)

    first = step(puzzle, state, Direction.RIGHT)
    assert first.valid
    second = step(puzzle, first.state, Direction.LEFT)

    assert not second.valid
    assert second.reason == "the path cannot revisit a cell"
    assert first.state.current_pos == (0, 1)


def test_walls_block_movement_bidirectionally() -> None:
    puzzle = snake_puzzle(walls={canonical_wall((0, 0), (0, 1))}, solution=None)
    state = new_game(puzzle)

    result = step(puzzle, state, Direction.RIGHT)

    assert not result.valid
    assert result.reason == "a wall blocks that move"


def test_waypoints_must_be_visited_in_order() -> None:
    puzzle = snake_puzzle(waypoints=((0, 0), (2, 0), (1, 1), (2, 2)), solution=None)
    state = new_game(puzzle)
    state = step(puzzle, state, Direction.DOWN).state

    result = step(puzzle, state, Direction.RIGHT)

    assert not result.valid
    assert result.reason == "waypoints must be visited in order"


def test_final_waypoint_must_be_last_cell() -> None:
    puzzle = snake_puzzle(waypoints=((0, 0), (2, 2)), solution=None)
    state = new_game(puzzle)
    state = step(puzzle, state, Direction.RIGHT).state
    state = step(puzzle, state, Direction.RIGHT).state
    state = step(puzzle, state, Direction.DOWN).state

    result = step(puzzle, state, Direction.DOWN)

    assert not result.valid
    assert result.reason == "the final waypoint must be the last cell"


def test_undo_restores_waypoint_progress() -> None:
    puzzle = snake_puzzle()
    state = new_game(puzzle)
    state = step(puzzle, state, Direction.RIGHT).state
    state = step(puzzle, state, Direction.DOWN).state

    assert state.current_pos == (1, 1)
    assert state.next_waypoint_index == 2

    undone = undo(state)

    assert undone.current_pos == (0, 1)
    assert undone.next_waypoint_index == 1
    assert undone.visited == {(0, 0), (0, 1)}


def test_reset_and_solution_completion() -> None:
    puzzle = snake_puzzle()
    state = new_game(puzzle)
    for direction in (
        Direction.RIGHT,
        Direction.RIGHT,
        Direction.DOWN,
        Direction.LEFT,
        Direction.LEFT,
        Direction.DOWN,
        Direction.RIGHT,
        Direction.RIGHT,
    ):
        result = step(puzzle, state, direction)
        assert result.valid, result.reason
        state = result.state

    assert is_solved(puzzle, state)
    assert state.success
    assert reset(puzzle).path == [(0, 0)]
