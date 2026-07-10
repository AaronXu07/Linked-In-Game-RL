"""Dictionary observation encoding for Patches RL environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from patches.simulation import GameState, Puzzle
from patches.simulation.utils import Rect, ShapeType, iter_grid

from .actions import PlacementAction

VALID_CELL = 0
COVERED_CELL = 1
UNCOVERED_CELL = 2
CLUE_CELL = 3
UNPLACED_CLUE = 4
PLACED_CLUE = 5
CLUE_NUMBER_NORMALIZED = 6
CLUE_NUMBER_MISSING = 7
SHAPE_SQUARE = 8
SHAPE_WIDE = 9
SHAPE_TALL = 10
SHAPE_FREE = 11
ASSIGNMENT_OWNER_NORMALIZED = 12
PLACED_PATCH_BOUNDARY = 13
LEGAL_CANDIDATE_COVERAGE_COUNT_NORMALIZED = 14
SOLUTION_PATCH_OPTIONAL = 15

BASE_GRID_CHANNEL_NAMES = (
    "valid_cell",
    "covered_cell",
    "uncovered_cell",
    "clue_cell",
    "unplaced_clue",
    "placed_clue",
    "clue_number_normalized",
    "clue_number_missing",
    "shape_square",
    "shape_wide",
    "shape_tall",
    "shape_free",
    "assignment_owner_normalized",
    "placed_patch_boundary",
    "legal_candidate_coverage_count_normalized",
)
SOLUTION_GRID_CHANNEL_NAME = "solution_patch_optional"

CANDIDATE_FEATURE_NAMES = (
    "clue_id_normalized",
    "clue_row_normalized",
    "clue_col_normalized",
    "clue_has_number",
    "clue_number_normalized",
    "shape_square",
    "shape_wide",
    "shape_tall",
    "shape_free",
    "rect_top_normalized",
    "rect_left_normalized",
    "rect_bottom_normalized",
    "rect_right_normalized",
    "rect_height_normalized",
    "rect_width_normalized",
    "rect_area_normalized",
    "rect_contains_current_patch",
    "rect_matches_solution_optional",
    "new_covered_fraction",
    "overlap_with_current_same_clue_fraction",
)


@dataclass(frozen=True)
class ObservationConfig:
    max_rows: int
    max_cols: int
    max_actions: int
    include_solution_hint: bool = False

    @property
    def grid_channels(self) -> int:
        return len(grid_channel_names(self.include_solution_hint))

    @property
    def candidate_features(self) -> int:
        return len(CANDIDATE_FEATURE_NAMES)


def grid_channel_names(include_solution_hint: bool = False) -> tuple[str, ...]:
    if include_solution_hint:
        return (*BASE_GRID_CHANNEL_NAMES, SOLUTION_GRID_CHANNEL_NAME)
    return BASE_GRID_CHANNEL_NAMES


def observation_shapes(
    max_rows: int,
    max_cols: int,
    max_actions: int,
    *,
    include_solution_hint: bool = False,
) -> dict[str, tuple[int, ...]]:
    config = ObservationConfig(max_rows, max_cols, max_actions, include_solution_hint)
    return {
        "grid": (config.grid_channels, config.max_rows, config.max_cols),
        "candidates": (config.max_actions, config.candidate_features),
        "candidate_footprints": (config.max_actions, config.max_rows, config.max_cols),
    }


def make_observation_space(
    max_rows: int,
    max_cols: int,
    max_actions: int,
    *,
    include_solution_hint: bool = False,
):
    from gymnasium import spaces

    shapes = observation_shapes(
        max_rows,
        max_cols,
        max_actions,
        include_solution_hint=include_solution_hint,
    )
    return spaces.Dict(
        {
            key: spaces.Box(0.0, 1.0, shape=shape, dtype=np.float32)
            for key, shape in shapes.items()
        }
    )


def encode_observation(
    puzzle: Puzzle,
    state: GameState,
    actions: Sequence[PlacementAction],
    *,
    max_rows: int,
    max_cols: int,
    max_actions: int,
    include_solution_hint: bool = False,
) -> dict[str, np.ndarray]:
    """Encode the board plus current candidate slots as float32 arrays."""

    _validate_bounds(puzzle, max_rows, max_cols)
    if len(actions) > max_actions:
        raise ValueError(
            f"cannot encode {len(actions)} actions with max_actions={max_actions}"
        )

    config = ObservationConfig(max_rows, max_cols, max_actions, include_solution_hint)
    grid = np.zeros((config.grid_channels, max_rows, max_cols), dtype=np.float32)
    candidates = np.zeros((max_actions, config.candidate_features), dtype=np.float32)
    footprints = np.zeros((max_actions, max_rows, max_cols), dtype=np.float32)

    valid_slice = (slice(0, puzzle.rows), slice(0, puzzle.cols))
    grid[VALID_CELL][valid_slice] = 1.0
    grid[UNCOVERED_CELL][valid_slice] = 1.0

    max_clue_id = max(1, max(clue.id for clue in puzzle.clues))
    for cell, owner in state.assignment.items():
        row, col = cell
        grid[COVERED_CELL, row, col] = 1.0
        grid[UNCOVERED_CELL, row, col] = 0.0
        grid[ASSIGNMENT_OWNER_NORMALIZED, row, col] = owner / max_clue_id

    for clue in puzzle.clues:
        row, col = clue.pos
        grid[CLUE_CELL, row, col] = 1.0
        if clue.id in state.patches:
            grid[PLACED_CLUE, row, col] = 1.0
        else:
            grid[UNPLACED_CLUE, row, col] = 1.0
        if clue.number is None:
            grid[CLUE_NUMBER_MISSING, row, col] = 1.0
        else:
            grid[CLUE_NUMBER_NORMALIZED, row, col] = clue.number / puzzle.total_cells
        grid[_shape_channel(clue.shape), row, col] = 1.0

    for rect in state.patches.values():
        for row, col in rect.cells():
            if row in (rect.top, rect.bottom) or col in (rect.left, rect.right):
                grid[PLACED_PATCH_BOUNDARY, row, col] = 1.0

    _encode_candidate_coverage(puzzle, actions, grid)
    if include_solution_hint and puzzle.solution is not None:
        for rect in puzzle.solution:
            for row, col in rect.cells():
                grid[SOLUTION_PATCH_OPTIONAL, row, col] = 1.0

    for index, action in enumerate(actions):
        candidates[index] = _candidate_features(
            puzzle,
            state,
            action,
            include_solution_hint=include_solution_hint,
        )
        for row, col in action.rect.cells():
            footprints[index, row, col] = 1.0

    return {
        "grid": grid,
        "candidates": candidates,
        "candidate_footprints": footprints,
    }


def _validate_bounds(puzzle: Puzzle, max_rows: int, max_cols: int) -> None:
    if max_rows <= 0 or max_cols <= 0:
        raise ValueError("max_rows and max_cols must be positive")
    if puzzle.rows > max_rows or puzzle.cols > max_cols:
        raise ValueError(
            "puzzle dimensions exceed observation bounds: "
            f"{puzzle.rows}x{puzzle.cols} > {max_rows}x{max_cols}"
        )


def _shape_channel(shape: ShapeType) -> int:
    if shape is ShapeType.SQUARE:
        return SHAPE_SQUARE
    if shape is ShapeType.WIDE:
        return SHAPE_WIDE
    if shape is ShapeType.TALL:
        return SHAPE_TALL
    return SHAPE_FREE


def _encode_candidate_coverage(
    puzzle: Puzzle,
    actions: Sequence[PlacementAction],
    grid: np.ndarray,
) -> None:
    counts = np.zeros((puzzle.rows, puzzle.cols), dtype=np.float32)
    for action in actions:
        for row, col in action.rect.cells():
            counts[row, col] += 1.0
    maximum = float(counts.max())
    if maximum > 0:
        grid[
            LEGAL_CANDIDATE_COVERAGE_COUNT_NORMALIZED,
            : puzzle.rows,
            : puzzle.cols,
        ] = counts / maximum


def _candidate_features(
    puzzle: Puzzle,
    state: GameState,
    action: PlacementAction,
    *,
    include_solution_hint: bool,
) -> np.ndarray:
    clue = puzzle.clue(action.clue_id)
    rect = action.rect
    features = np.zeros(len(CANDIDATE_FEATURE_NAMES), dtype=np.float32)
    max_clue_id = max(1, max(item.id for item in puzzle.clues))
    rows_den = max(1, puzzle.rows - 1)
    cols_den = max(1, puzzle.cols - 1)

    features[0] = clue.id / max_clue_id
    features[1] = clue.pos[0] / rows_den
    features[2] = clue.pos[1] / cols_den
    features[3] = 1.0 if clue.number is not None else 0.0
    features[4] = 0.0 if clue.number is None else clue.number / puzzle.total_cells
    features[5 + _shape_feature_offset(clue.shape)] = 1.0
    features[9] = rect.top / rows_den
    features[10] = rect.left / cols_den
    features[11] = rect.bottom / rows_den
    features[12] = rect.right / cols_den
    features[13] = rect.height / puzzle.rows
    features[14] = rect.width / puzzle.cols
    features[15] = rect.area / puzzle.total_cells

    current = state.patches.get(clue.id)
    if current is not None:
        current_cells = set(current.cells())
        rect_cells = set(rect.cells())
        features[16] = 1.0 if current_cells <= rect_cells else 0.0
        features[19] = len(current_cells & rect_cells) / max(1, rect.area)

    solution_rect = puzzle.solution_rect(clue.id)
    if include_solution_hint and solution_rect is not None and solution_rect == rect:
        features[17] = 1.0

    new_cells = sum(1 for cell in rect.cells() if cell not in state.assignment)
    features[18] = new_cells / puzzle.total_cells
    return features


def _shape_feature_offset(shape: ShapeType) -> int:
    if shape is ShapeType.SQUARE:
        return 0
    if shape is ShapeType.WIDE:
        return 1
    if shape is ShapeType.TALL:
        return 2
    return 3
