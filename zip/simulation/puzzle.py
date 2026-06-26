"""Immutable puzzle definition and JSON serialization."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .utils import Coordinate, Wall, are_adjacent, canonical_wall, in_bounds, solution_edges


def _coerce_coordinate(value: Any) -> Coordinate:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"coordinate must be a length-2 sequence, got {value!r}")
    row, col = value
    return int(row), int(col)


def _coerce_wall(value: Any) -> Wall:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"wall must contain two coordinates, got {value!r}")
    return canonical_wall(_coerce_coordinate(value[0]), _coerce_coordinate(value[1]))


@dataclass(frozen=True)
class Puzzle:
    """Immutable Zip puzzle definition.

    `solution` may be omitted for externally-authored puzzles. Generated
    puzzles store a full Hamiltonian solution path.
    """

    rows: int
    cols: int
    waypoints: tuple[Coordinate, ...] | list[Coordinate]
    walls: frozenset[Wall] | set[Wall] | list[Wall] = field(default_factory=frozenset)
    solution: tuple[Coordinate, ...] | list[Coordinate] | None = None
    difficulty: str = "custom"
    seed: int | None = None
    unique_solution: bool | None = None
    _waypoint_lookup: dict[Coordinate, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        rows = int(self.rows)
        cols = int(self.cols)
        if rows <= 0 or cols <= 0:
            raise ValueError("rows and cols must be positive")

        waypoints = tuple(_coerce_coordinate(cell) for cell in self.waypoints)
        if not waypoints:
            raise ValueError("a puzzle needs at least one waypoint")
        if len(set(waypoints)) != len(waypoints):
            raise ValueError("waypoints must be unique")
        for cell in waypoints:
            if not in_bounds(rows, cols, cell):
                raise ValueError(f"waypoint out of bounds: {cell!r}")

        walls = frozenset(_coerce_wall(wall) for wall in self.walls)
        for a, b in walls:
            if not in_bounds(rows, cols, a) or not in_bounds(rows, cols, b):
                raise ValueError(f"wall out of bounds: {(a, b)!r}")

        solution = None
        if self.solution is not None:
            solution = tuple(_coerce_coordinate(cell) for cell in self.solution)
            self._validate_solution(rows, cols, waypoints, walls, solution)

        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "cols", cols)
        object.__setattr__(self, "waypoints", waypoints)
        object.__setattr__(self, "walls", walls)
        object.__setattr__(self, "solution", solution)
        object.__setattr__(self, "_waypoint_lookup", {cell: i for i, cell in enumerate(waypoints)})

    @staticmethod
    def _validate_solution(
        rows: int,
        cols: int,
        waypoints: tuple[Coordinate, ...],
        walls: frozenset[Wall],
        solution: tuple[Coordinate, ...],
    ) -> None:
        if not solution:
            raise ValueError("solution cannot be empty when provided")
        if solution[0] != waypoints[0]:
            raise ValueError("solution must start at waypoint 1")
        if len(set(solution)) != len(solution):
            raise ValueError("solution cannot revisit cells")
        for cell in solution:
            if not in_bounds(rows, cols, cell):
                raise ValueError(f"solution cell out of bounds: {cell!r}")
        for a, b in zip(solution, solution[1:]):
            if not are_adjacent(a, b):
                raise ValueError(f"solution contains a non-adjacent step: {a!r} -> {b!r}")
            if canonical_wall(a, b) in walls:
                raise ValueError(f"solution crosses a wall: {a!r} -> {b!r}")

        positions = {cell: i for i, cell in enumerate(solution)}
        seen_positions = [positions[cell] for cell in waypoints if cell in positions]
        if seen_positions != sorted(seen_positions):
            raise ValueError("waypoints in solution must appear in ascending order")

        if len(solution) == rows * cols:
            all_cells = {(row, col) for row in range(rows) for col in range(cols)}
            if set(solution) != all_cells:
                raise ValueError("full solution must visit every cell exactly once")
            missing = [cell for cell in waypoints if cell not in positions]
            if missing:
                raise ValueError(f"full solution is missing waypoints: {missing!r}")
            if solution[-1] != waypoints[-1]:
                raise ValueError("full solution must end at the final waypoint")
            blocked_solution_edges = solution_edges(solution) & walls
            if blocked_solution_edges:
                raise ValueError(f"walls block solution edges: {blocked_solution_edges!r}")

    @property
    def total_cells(self) -> int:
        return self.rows * self.cols

    @property
    def start(self) -> Coordinate:
        return self.waypoints[0]

    @property
    def final_waypoint(self) -> Coordinate:
        return self.waypoints[-1]

    def waypoint_index(self, cell: Coordinate) -> int | None:
        return self._waypoint_lookup.get(cell)

    def has_wall(self, a: Coordinate, b: Coordinate) -> bool:
        return canonical_wall(a, b) in self.walls

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "cols": self.cols,
            "waypoints": [list(cell) for cell in self.waypoints],
            "walls": [[list(a), list(b)] for a, b in sorted(self.walls)],
            "solution": None if self.solution is None else [list(cell) for cell in self.solution],
            "difficulty": self.difficulty,
            "seed": self.seed,
            "unique_solution": self.unique_solution,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Puzzle":
        return cls(
            rows=data["rows"],
            cols=data["cols"],
            waypoints=data["waypoints"],
            walls=data.get("walls", []),
            solution=data.get("solution"),
            difficulty=data.get("difficulty", "custom"),
            seed=data.get("seed"),
            unique_solution=data.get("unique_solution"),
        )


def puzzle_to_json(puzzle: Puzzle, *, indent: int = 2) -> str:
    return json.dumps(puzzle.to_dict(), indent=indent, sort_keys=True)


def puzzle_from_json(text: str) -> Puzzle:
    return Puzzle.from_dict(json.loads(text))


def save_puzzle(puzzle: Puzzle, path: str | Path) -> None:
    Path(path).write_text(puzzle_to_json(puzzle) + "\n", encoding="utf-8")


def load_puzzle(path: str | Path) -> Puzzle:
    return puzzle_from_json(Path(path).read_text(encoding="utf-8"))
