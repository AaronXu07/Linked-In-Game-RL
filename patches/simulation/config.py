"""Difficulty and UI configuration for Patches puzzles."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DifficultyConfig:
    """Generation knobs for a Patches puzzle difficulty."""

    name: str
    rows: int
    cols: int
    min_patch_area: int
    max_patch_area: int
    free_shape_ratio: float
    require_unique_solution: bool
    number_ratio: float = 0.0
    hint_reduction: float = 1.0
    clue_position_policy: str = "corner"
    large_patch_bias: float = 2.0
    max_generation_attempts: int = 60
    generation_timeout_seconds: float = 3.0
    solver_timeout_seconds: float = 3.0
    seed: int | None = None

    def normalized_name(self) -> str:
        return normalize_difficulty_name(self.name)


@dataclass(frozen=True)
class UIConfig:
    cell_size: int = 112
    min_cell_size: int = 64
    margin: int = 36
    panel_width: int = 320
    panel_gap: int = 34
    fps: int = 60
    playback_patches_per_second: float = 3.0


DIFFICULTIES: dict[str, DifficultyConfig] = {
    "super_easy": DifficultyConfig(
        name="super_easy",
        rows=5,
        cols=5,
        min_patch_area=2,
        max_patch_area=6,
        free_shape_ratio=0.15,
        number_ratio=1.0,
        hint_reduction=0.0,
        require_unique_solution=False,
        large_patch_bias=3.0,
    ),
    "easy": DifficultyConfig(
        name="easy",
        rows=6,
        cols=6,
        min_patch_area=2,
        max_patch_area=7,
        free_shape_ratio=0.5,
        number_ratio=0.8,
        hint_reduction=0.35,
        require_unique_solution=False,
        large_patch_bias=5.0,
        max_generation_attempts=80,
    ),
    "medium": DifficultyConfig(
        name="medium",
        rows=7,
        cols=7,
        min_patch_area=3,
        max_patch_area=8,
        free_shape_ratio=1.0,
        number_ratio=0.35,
        hint_reduction=0.7,
        require_unique_solution=True,
        large_patch_bias=6.0,
        max_generation_attempts=140,
        generation_timeout_seconds=4.0,
        solver_timeout_seconds=4.0,
    ),
    "hard": DifficultyConfig(
        name="hard",
        rows=8,
        cols=8,
        min_patch_area=3,
        max_patch_area=11,
        free_shape_ratio=1.0,
        number_ratio=0.15,
        hint_reduction=0.9,
        require_unique_solution=True,
        large_patch_bias=7.0,
        max_generation_attempts=180,
        generation_timeout_seconds=6.0,
        solver_timeout_seconds=5.0,
    ),
    "expert": DifficultyConfig(
        name="expert",
        rows=9,
        cols=9,
        min_patch_area=4,
        max_patch_area=14,
        free_shape_ratio=1.0,
        number_ratio=0.0,
        hint_reduction=1.0,
        require_unique_solution=True,
        large_patch_bias=8.0,
        max_generation_attempts=240,
        generation_timeout_seconds=10.0,
        solver_timeout_seconds=6.0,
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
