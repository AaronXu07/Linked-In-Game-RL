"""Small supervised datasets derived from simulator solution paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from zip.simulation import Puzzle, new_game
from zip.simulation import step as simulator_step
from zip.simulation.utils import Direction

from .observations import encode_observation

DATASET_DIRECTION_TO_ACTION = {
    Direction.UP: 0,
    Direction.DOWN: 1,
    Direction.LEFT: 2,
    Direction.RIGHT: 3,
}


@dataclass(frozen=True)
class ImitationExample:
    observation: np.ndarray
    action: int


def solution_examples(
    puzzle: Puzzle,
    *,
    max_rows: int | None = None,
    max_cols: int | None = None,
) -> list[ImitationExample]:
    """Return observation/action examples along a puzzle's stored solution."""

    if puzzle.solution is None:
        raise ValueError("puzzle does not include a solution path")
    state = new_game(puzzle)
    examples: list[ImitationExample] = []
    for current, destination in zip(puzzle.solution, puzzle.solution[1:]):
        direction = Direction.between(current, destination)
        examples.append(
            ImitationExample(
                observation=encode_observation(
                    puzzle,
                    state,
                    max_rows=max_rows,
                    max_cols=max_cols,
                ),
                action=DATASET_DIRECTION_TO_ACTION[direction],
            )
        )
        result = simulator_step(puzzle, state, direction)
        if not result.valid:
            raise ValueError(f"stored solution contains invalid move: {result.reason}")
        state = result.state
    return examples


def build_solution_dataset(
    puzzles: Iterable[Puzzle],
    *,
    max_rows: int | None = None,
    max_cols: int | None = None,
) -> list[ImitationExample]:
    examples: list[ImitationExample] = []
    for puzzle in puzzles:
        examples.extend(
            solution_examples(puzzle, max_rows=max_rows, max_cols=max_cols)
        )
    return examples
