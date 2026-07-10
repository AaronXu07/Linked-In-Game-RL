"""Small supervised datasets derived from Patches solution tilings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from patches.simulation import Puzzle, new_game
from patches.simulation import place as simulator_place

from .actions import (
    PlacementAction,
    build_action_mask,
    legal_placement_actions,
    precompute_candidates,
)
from .observations import encode_observation


@dataclass(frozen=True)
class ImitationExample:
    observation: dict[str, np.ndarray]
    action: int
    action_mask: np.ndarray


def solution_examples(
    puzzle: Puzzle,
    *,
    max_rows: int | None = None,
    max_cols: int | None = None,
    max_actions: int | None = None,
) -> list[ImitationExample]:
    """Return observation/action examples along a deterministic solution order."""

    if puzzle.solution is None:
        raise ValueError("puzzle does not include a solution tiling")

    max_rows = puzzle.rows if max_rows is None else int(max_rows)
    max_cols = puzzle.cols if max_cols is None else int(max_cols)
    base_candidates = precompute_candidates(puzzle)
    if max_actions is None:
        max_actions = sum(len(rects) for rects in base_candidates.values())

    state = new_game(puzzle)
    examples: list[ImitationExample] = []
    while not state.success:
        actions = legal_placement_actions(
            puzzle,
            state,
            base_candidates_by_clue=base_candidates,
        )
        action_index, action = _choose_solution_action(puzzle, actions)
        examples.append(
            ImitationExample(
                observation=encode_observation(
                    puzzle,
                    state,
                    actions,
                    max_rows=max_rows,
                    max_cols=max_cols,
                    max_actions=max_actions,
                ),
                action=action_index,
                action_mask=build_action_mask(len(actions), max_actions),
            )
        )
        result = simulator_place(puzzle, state, action.clue_id, action.rect)
        if not result.valid:
            raise ValueError(f"stored solution contains invalid placement: {result.reason}")
        state = result.state
    return examples


def build_solution_dataset(
    puzzles: Iterable[Puzzle],
    *,
    max_rows: int | None = None,
    max_cols: int | None = None,
    max_actions: int | None = None,
) -> list[ImitationExample]:
    examples: list[ImitationExample] = []
    for puzzle in puzzles:
        examples.extend(
            solution_examples(
                puzzle,
                max_rows=max_rows,
                max_cols=max_cols,
                max_actions=max_actions,
            )
        )
    return examples


def _choose_solution_action(
    puzzle: Puzzle,
    actions: list[PlacementAction],
) -> tuple[int, PlacementAction]:
    candidates: list[tuple[int, PlacementAction]] = []
    for index, action in enumerate(actions):
        if puzzle.solution_rect(action.clue_id) == action.rect:
            candidates.append((index, action))
    if not candidates:
        raise ValueError("no legal solution action found for current state")
    return min(
        candidates,
        key=lambda item: (
            sum(1 for action in actions if action.clue_id == item[1].clue_id),
            item[1].clue_id,
            item[1].rect.top,
            item[1].rect.left,
            item[1].rect.height,
            item[1].rect.width,
        ),
    )
