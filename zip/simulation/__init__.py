"""LinkedIn Zip-style puzzle simulator.

The package intentionally contains only simulation pieces: puzzle data,
rules, generation, solving, rendering, and a local playable UI. Training and
reinforcement-learning code belong outside this package.
"""

from .config import DIFFICULTIES, DifficultyConfig, get_difficulty_config
from .generator import GenerationError, GenerationResult, generate_puzzle, generate_puzzle_with_report
from .puzzle import Puzzle, load_puzzle, puzzle_from_json, puzzle_to_json, save_puzzle
from .simulator import can_move, is_solved, legal_moves, new_game, reset, state_from_path, step, undo
from .solver import SolverResult, find_solution, solve
from .state import GameState, StepResult
from .utils import Direction

__all__ = [
    "DIFFICULTIES",
    "DifficultyConfig",
    "Direction",
    "GameState",
    "GenerationError",
    "GenerationResult",
    "Puzzle",
    "SolverResult",
    "StepResult",
    "can_move",
    "find_solution",
    "generate_puzzle",
    "generate_puzzle_with_report",
    "get_difficulty_config",
    "is_solved",
    "legal_moves",
    "load_puzzle",
    "new_game",
    "puzzle_from_json",
    "puzzle_to_json",
    "reset",
    "save_puzzle",
    "solve",
    "state_from_path",
    "step",
    "undo",
]
