"""Difficulty and UI configuration for Zip puzzles."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DifficultyConfig:
    """Generation knobs for a Zip puzzle difficulty."""

    name: str
    rows: int
    cols: int
    waypoint_spacing: tuple[int, int] | None
    total_waypoints: tuple[int, int] | None
    wall_count: tuple[int, int]
    require_unique_solution: bool
    max_generation_attempts: int = 40
    generation_timeout_seconds: float = 2.0
    solver_timeout_seconds: float = 2.0
    allow_multi_solution: bool = True
    start_position_policy: str = "random"
    waypoint_spacing_policy: str = "interval"
    wall_candidate_policy: str = "tempting"
    seed: int | None = None

    def normalized_name(self) -> str:
        return normalize_difficulty_name(self.name)


@dataclass(frozen=True)
class UIConfig:
    cell_size: int = 112
    min_cell_size: int = 72
    margin: int = 36
    panel_width: int = 320
    panel_gap: int = 34
    fps: int = 60
    playback_cells_per_second: float = 6.0
    path_animation_seconds: float = 0.16
    board_render_scale: int = 3


DIFFICULTIES: dict[str, DifficultyConfig] = {
    "super_easy": DifficultyConfig(
        name="super_easy",
        rows=4,
        cols=4,
        waypoint_spacing=(2, 3),
        total_waypoints=None,
        wall_count=(0, 0),
        require_unique_solution=False,
        allow_multi_solution=True,
        start_position_policy="random",
    ),
    "easy": DifficultyConfig(
        name="easy",
        rows=5,
        cols=5,
        waypoint_spacing=(4, 6),
        total_waypoints=None,
        wall_count=(2, 4),
        require_unique_solution=False,
        max_generation_attempts=60,
        allow_multi_solution=True,
        start_position_policy="edge",
    ),
    "medium": DifficultyConfig(
        name="medium",
        rows=6,
        cols=6,
        waypoint_spacing=(6, 8),
        total_waypoints=None,
        wall_count=(4, 8),
        require_unique_solution=True,
        max_generation_attempts=90,
        generation_timeout_seconds=3.0,
        solver_timeout_seconds=3.0,
        allow_multi_solution=False,
        start_position_policy="edge",
    ),
    "hard": DifficultyConfig(
        name="hard",
        rows=7,
        cols=7,
        waypoint_spacing=None,
        total_waypoints=(8, 11),
        wall_count=(4, 8),
        require_unique_solution=True,
        max_generation_attempts=120,
        generation_timeout_seconds=5.0,
        solver_timeout_seconds=3.0,
        allow_multi_solution=False,
        start_position_policy="corner",
        waypoint_spacing_policy="even_total",
        wall_candidate_policy="surgical",
    ),
    "expert": DifficultyConfig(
        name="expert",
        rows=8,
        cols=8,
        waypoint_spacing=None,
        total_waypoints=(10, 14),
        wall_count=(8, 12),
        require_unique_solution=True,
        max_generation_attempts=160,
        generation_timeout_seconds=8.0,
        solver_timeout_seconds=2.0,
        allow_multi_solution=False,
        start_position_policy="corner",
        waypoint_spacing_policy="even_total",
        wall_candidate_policy="surgical",
    ),
}


def normalize_difficulty_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def get_difficulty_config(name: str | DifficultyConfig) -> DifficultyConfig:
    if isinstance(name, DifficultyConfig):
        return name
    key = normalize_difficulty_name(name)
    try:
        return DIFFICULTIES[key]
    except KeyError as exc:
        choices = ", ".join(sorted(DIFFICULTIES))
        raise ValueError(f"unknown difficulty {name!r}; choose one of: {choices}") from exc
