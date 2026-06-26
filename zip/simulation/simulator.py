"""Public simulator API for Zip puzzles."""

from __future__ import annotations

from .puzzle import Puzzle
from .rules import advance_waypoint_index, is_solved as rules_is_solved, validate_move
from .state import GameState, StepResult
from .utils import Coordinate, Direction


def new_game(puzzle: Puzzle) -> GameState:
    start = puzzle.start
    next_waypoint_index = 1
    state = GameState(
        path=[start],
        visited={start},
        current_pos=start,
        next_waypoint_index=next_waypoint_index,
        done=False,
        success=False,
        step_count=0,
        waypoint_history=[next_waypoint_index],
    )
    solved = rules_is_solved(puzzle, state)
    state.done = solved
    state.success = solved
    return state


def reset(puzzle: Puzzle) -> GameState:
    return new_game(puzzle)


def legal_moves(puzzle: Puzzle, state: GameState) -> list[Direction]:
    return [direction for direction in Direction.all() if can_move(puzzle, state, direction)]


def can_move(puzzle: Puzzle, state: GameState, direction: Direction) -> bool:
    return validate_move(puzzle, state, direction).valid


def step(puzzle: Puzzle, state: GameState, direction: Direction) -> StepResult:
    validation = validate_move(puzzle, state, direction)
    if not validation.valid:
        return StepResult(state=state.clone(), valid=False, reason=validation.reason, solved=state.success)

    destination = validation.destination
    next_waypoint_index = advance_waypoint_index(puzzle, state.next_waypoint_index, destination)
    next_state = GameState(
        path=[*state.path, destination],
        visited={*state.visited, destination},
        current_pos=destination,
        next_waypoint_index=next_waypoint_index,
        done=False,
        success=False,
        step_count=state.step_count + 1,
        waypoint_history=[*state.waypoint_history, next_waypoint_index],
    )
    solved = rules_is_solved(puzzle, next_state)
    next_state.done = solved
    next_state.success = solved
    return StepResult(state=next_state, valid=True, reason=None, solved=solved)


def undo(state: GameState) -> GameState:
    if len(state.path) <= 1:
        return state.clone()

    path = state.path[:-1]
    history = state.waypoint_history[:-1] if len(state.waypoint_history) >= len(state.path) else []
    next_waypoint_index = history[-1] if history else state.next_waypoint_index
    return GameState(
        path=path,
        visited=set(path),
        current_pos=path[-1],
        next_waypoint_index=next_waypoint_index,
        done=False,
        success=False,
        step_count=max(0, state.step_count - 1),
        waypoint_history=history,
    )


def is_solved(puzzle: Puzzle, state: GameState) -> bool:
    return rules_is_solved(puzzle, state)


def state_from_path(puzzle: Puzzle, path: list[Coordinate] | tuple[Coordinate, ...]) -> GameState:
    if not path:
        raise ValueError("path cannot be empty")
    if path[0] != puzzle.start:
        raise ValueError("path must start at waypoint 1")

    state = new_game(puzzle)
    for current, destination in zip(path, path[1:]):
        direction = Direction.between(current, destination)
        result = step(puzzle, state, direction)
        if not result.valid:
            raise ValueError(f"invalid path step {current!r} -> {destination!r}: {result.reason}")
        state = result.state
    return state
