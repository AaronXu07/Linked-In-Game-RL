"""Candidate placement actions for Patches training environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from patches.simulation import Puzzle
from patches.simulation.simulator import can_place, candidate_rects
from patches.simulation.state import GameState
from patches.simulation.utils import Rect

EPISODE_MODES = {"commit_only", "revision"}


@dataclass(frozen=True)
class PlacementAction:
    """One dynamic action slot: place ``rect`` for ``clue_id``."""

    clue_id: int
    rect: Rect

    def label(self) -> str:
        return (
            f"clue {self.clue_id}: "
            f"r{self.rect.top} c{self.rect.left} "
            f"{self.rect.height}x{self.rect.width}"
        )


def precompute_candidates(puzzle: Puzzle) -> dict[int, tuple[Rect, ...]]:
    return {clue.id: tuple(candidate_rects(puzzle, clue.id)) for clue in puzzle.clues}


def legal_placement_actions(
    puzzle: Puzzle,
    state: GameState,
    *,
    base_candidates_by_clue: Mapping[int, tuple[Rect, ...]] | None = None,
    episode_mode: str = "commit_only",
) -> list[PlacementAction]:
    """Return all currently legal placement actions in deterministic order."""

    if episode_mode not in EPISODE_MODES:
        choices = ", ".join(sorted(EPISODE_MODES))
        raise ValueError(f"episode_mode must be one of: {choices}")

    base = base_candidates_by_clue or precompute_candidates(puzzle)
    legal_by_clue: dict[int, list[Rect]] = {}
    for clue in puzzle.clues:
        if episode_mode == "commit_only" and clue.id in state.patches:
            continue
        rects = [
            rect
            for rect in base.get(clue.id, ())
            if can_place(puzzle, state, clue.id, rect).valid
        ]
        if rects:
            rects.sort(key=_rect_sort_key)
            legal_by_clue[clue.id] = rects

    actions: list[PlacementAction] = []
    for clue_id, rects in sorted(
        legal_by_clue.items(),
        key=lambda item: (len(item[1]), item[0]),
    ):
        actions.extend(PlacementAction(clue_id, rect) for rect in rects)
    return actions


def build_action_mask(action_count: int, max_actions: int) -> np.ndarray:
    if action_count > max_actions:
        raise ValueError(
            f"current state has {action_count} legal actions, "
            f"which exceeds max_actions={max_actions}"
        )
    mask = np.zeros(max_actions, dtype=np.bool_)
    mask[:action_count] = True
    return mask


def infer_max_actions(
    puzzle: Puzzle | None = None,
    *,
    max_rows: int | None = None,
    max_cols: int | None = None,
    safety_factor: float = 1.5,
) -> int:
    """Infer a conservative padded action capacity.

    When a concrete puzzle is available this returns a puzzle-specific upper
    bound. Otherwise it scales with board area, which is intentionally roomy for
    generated numberless/free clues.
    """

    if puzzle is not None:
        total = sum(len(candidate_rects(puzzle, clue.id)) for clue in puzzle.clues)
        return max(1, int(total))
    rows = int(max_rows or 9)
    cols = int(max_cols or 9)
    area = rows * cols
    return max(32, int(area * max(rows, cols) * safety_factor))


def _rect_sort_key(rect: Rect) -> tuple[int, int, int, int]:
    return rect.top, rect.left, rect.height, rect.width
