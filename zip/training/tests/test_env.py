import pytest

pytest.importorskip("gymnasium")

from zip.simulation.puzzle import Puzzle
from zip.simulation.utils import Direction
from zip.training.env import DIRECTION_TO_ACTION, ZipEnv


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


def solution_actions() -> list[int]:
    return [
        DIRECTION_TO_ACTION[Direction.between(current, destination)]
        for current, destination in zip(SNAKE_3X3, SNAKE_3X3[1:])
    ]


def test_reset_returns_observation_and_action_mask() -> None:
    env = ZipEnv(puzzle=snake_puzzle())

    obs, info = env.reset(seed=123)

    assert obs.shape == env.observation_space.shape
    assert info["episode_id"] == 1
    assert info["visited_count"] == 1
    assert info["action_mask"].shape == (4,)
    assert info["action_mask"][DIRECTION_TO_ACTION[Direction.UP]] == 0
    assert info["action_mask"][DIRECTION_TO_ACTION[Direction.RIGHT]] == 1


def test_invalid_action_terminates_by_default() -> None:
    env = ZipEnv(puzzle=snake_puzzle())
    env.reset(seed=123)

    _, reward, terminated, truncated, info = env.step(DIRECTION_TO_ACTION[Direction.UP])

    assert reward == -2.0
    assert terminated
    assert not truncated
    assert info["invalid_action"]
    assert info["invalid_reason"] == "destination is out of bounds"


def test_env_can_follow_solution_to_success() -> None:
    env = ZipEnv(puzzle=snake_puzzle())
    env.reset(seed=123)

    terminated = truncated = False
    info = {}
    for action in solution_actions():
        _, _, terminated, truncated, info = env.step(action)

    assert terminated
    assert not truncated
    assert info["success"]
    assert info["visited_count"] == info["total_cells"]


def test_penalize_mode_keeps_episode_alive_after_invalid_action() -> None:
    env = ZipEnv(puzzle=snake_puzzle(), invalid_action_mode="penalize")
    env.reset(seed=123)

    _, reward, terminated, truncated, info = env.step(DIRECTION_TO_ACTION[Direction.UP])

    assert reward == -2.0
    assert not terminated
    assert not truncated
    assert info["invalid_action"]
    assert info["current_pos"] == (0, 0)


def test_env_can_reset_from_puzzle_sampler() -> None:
    class Sampler:
        def __init__(self) -> None:
            self.closed = False

        def sample_puzzle(self) -> Puzzle:
            return snake_puzzle()

        def close(self) -> None:
            self.closed = True

    sampler = Sampler()
    env = ZipEnv(puzzle_sampler=sampler)

    obs, info = env.reset(seed=123)
    env.close()

    assert obs.shape == env.observation_space.shape
    assert info["total_cells"] == 9
    assert sampler.closed
