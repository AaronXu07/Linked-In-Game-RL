from patches.simulation import new_game
from patches.simulation import place as simulator_place
from patches.simulation.puzzle import Clue, Puzzle
from patches.simulation.state import StepResult
from patches.simulation.utils import Rect, ShapeType
from patches.training.actions import PlacementAction
from patches.training.rewards import RewardConfig, compute_reward


def tiny_puzzle() -> Puzzle:
    clues = [Clue(id=0, pos=(0, 0), number=4, shape=ShapeType.SQUARE)]
    solution = [Rect(0, 0, 2, 2)]
    return Puzzle(rows=2, cols=2, clues=clues, solution=solution)


def test_valid_and_solve_reward() -> None:
    puzzle = tiny_puzzle()
    state = new_game(puzzle)
    action = PlacementAction(0, puzzle.solution_rect(0))
    result = simulator_place(puzzle, state, action.clue_id, action.rect)

    reward = compute_reward(
        puzzle,
        state,
        action,
        result,
        dead_end=False,
        truncated=False,
        config=RewardConfig(),
    )

    assert reward == 0.05 + 4 * 0.02 + 0.05 + 10.0


def test_invalid_reward() -> None:
    puzzle = tiny_puzzle()
    state = new_game(puzzle)
    result = StepResult(state=state.clone(), valid=False, reason="bad", solved=False)

    reward = compute_reward(
        puzzle,
        state,
        None,
        result,
        dead_end=False,
        truncated=False,
        config=RewardConfig(),
    )

    assert reward == -2.0


def test_truncate_reward_adds_penalty() -> None:
    puzzle = tiny_puzzle()
    state = new_game(puzzle)
    result = StepResult(state=state.clone(), valid=False, reason="bad", solved=False)

    reward = compute_reward(
        puzzle,
        state,
        None,
        result,
        dead_end=False,
        truncated=True,
        config=RewardConfig(),
    )

    assert reward == -3.0
