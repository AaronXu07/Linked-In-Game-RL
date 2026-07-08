"""LinkedIn Patches-style puzzle simulator.

The package intentionally contains only simulation pieces: puzzle data,
rules, generation, solving, rendering, and (later) a local playable UI.
Training and reinforcement-learning code belong outside this package.

See ``patches/simulation_plan.md`` for the design.
"""

from .config import DIFFICULTIES, DifficultyConfig, UIConfig, get_difficulty_config
from .generator import (
    GenerationError,
    GenerationResult,
    generate_puzzle,
    generate_puzzle_with_report,
)
from .puzzle import Clue, Puzzle, load_puzzle, puzzle_from_json, puzzle_to_json, save_puzzle
from .rules import PlacementValidation, is_solved, validate_placement
from .simulator import (
    can_place,
    candidate_rects,
    clear_patch,
    new_game,
    place,
    reset,
    state_from_solution,
    undo,
)
from .solver import SolverResult, find_solution, solve
from .state import GameState, StepResult
from .utils import Coordinate, Rect, ShapeType, rect_from_corners

__all__ = [
    "DIFFICULTIES",
    "Clue",
    "Coordinate",
    "DifficultyConfig",
    "GameState",
    "GenerationError",
    "GenerationResult",
    "PlacementValidation",
    "Puzzle",
    "Rect",
    "ShapeType",
    "SolverResult",
    "StepResult",
    "UIConfig",
    "can_place",
    "candidate_rects",
    "clear_patch",
    "find_solution",
    "generate_puzzle",
    "generate_puzzle_with_report",
    "get_difficulty_config",
    "is_solved",
    "load_puzzle",
    "new_game",
    "place",
    "puzzle_from_json",
    "puzzle_to_json",
    "rect_from_corners",
    "reset",
    "save_puzzle",
    "solve",
    "state_from_solution",
    "undo",
    "validate_placement",
]
