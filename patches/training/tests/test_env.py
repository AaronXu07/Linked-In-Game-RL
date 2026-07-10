import pytest

pytest.importorskip("gymnasium")

from patches.simulation.puzzle import Clue, Puzzle
from patches.simulation.utils import Rect, ShapeType
from patches.training.env import PatchesEnv


def four_patch_puzzle() -> Puzzle:
    clues = [
        Clue(id=0, pos=(0, 0), number=4, shape=ShapeType.SQUARE),
        Clue(id=1, pos=(0, 2), number=4, shape=ShapeType.SQUARE),
        Clue(id=2, pos=(2, 0), number=4, shape=ShapeType.SQUARE),
        Clue(id=3, pos=(2, 2), number=4, shape=ShapeType.SQUARE),
    ]
    solution = [
        Rect(0, 0, 2, 2),
        Rect(0, 2, 2, 2),
        Rect(2, 0, 2, 2),
        Rect(2, 2, 2, 2),
    ]
    return Puzzle(rows=4, cols=4, clues=clues, solution=solution)


def test_reset_returns_observation_and_action_mask() -> None:
    env = PatchesEnv(puzzle=four_patch_puzzle())

    obs, info = env.reset(seed=123)

    assert set(obs) == {"grid", "candidates", "candidate_footprints"}
    assert obs["grid"].shape == env.observation_space.spaces["grid"].shape
    assert info["episode_id"] == 1
    assert info["covered_count"] == 0
    assert info["action_mask"].shape == (env.max_actions,)
    assert info["action_mask"].sum() == info["action_count"]
    assert info["action_count"] == len(env.current_actions)


def test_invalid_padding_action_terminates_by_default() -> None:
    env = PatchesEnv(puzzle=four_patch_puzzle(), max_actions=12)
    _, info = env.reset(seed=123)
    invalid_action = int(info["action_count"])

    _, reward, terminated, truncated, step_info = env.step(invalid_action)

    assert reward == -2.0
    assert terminated
    assert not truncated
    assert step_info["invalid_action"]
    assert step_info["invalid_reason"] == "action slot is not legal"


def test_env_can_follow_solution_to_success() -> None:
    puzzle = four_patch_puzzle()
    env = PatchesEnv(puzzle=puzzle)
    _, info = env.reset(seed=123)

    terminated = truncated = False
    for clue_id in range(4):
        actions = env.current_actions
        action = next(
            index
            for index, candidate in enumerate(actions)
            if candidate.clue_id == clue_id and candidate.rect == puzzle.solution_rect(clue_id)
        )
        _, _, terminated, truncated, info = env.step(action)

    assert terminated
    assert not truncated
    assert info["success"]
    assert info["covered_count"] == info["total_cells"]


def test_penalize_mode_keeps_episode_alive_after_invalid_action() -> None:
    env = PatchesEnv(
        puzzle=four_patch_puzzle(),
        max_actions=12,
        invalid_action_mode="penalize",
    )
    _, info = env.reset(seed=123)
    invalid_action = int(info["action_count"])

    _, reward, terminated, truncated, step_info = env.step(invalid_action)

    assert reward == -2.0
    assert not terminated
    assert not truncated
    assert step_info["invalid_action"]
    assert step_info["covered_count"] == 0
