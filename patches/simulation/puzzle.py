"""Immutable Patches puzzle definition and JSON serialization."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .utils import (
    Coordinate,
    Rect,
    ShapeType,
    in_bounds,
    iter_grid,
    rect_in_bounds,
    shape_matches,
)


def _coerce_coordinate(value: Any) -> Coordinate:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"coordinate must be a length-2 sequence, got {value!r}")
    row, col = value
    return int(row), int(col)


@dataclass(frozen=True)
class Clue:
    """A single clue: its cell, shape constraint, and optional required size.

    ``number`` is optional. When it is ``None`` the patch may be any size (as
    long as its shape matches); when it is set, the patch area must equal it.
    """

    id: int
    pos: Coordinate
    number: int | None
    shape: ShapeType

    @property
    def has_number(self) -> bool:
        return self.number is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "pos": list(self.pos),
            "number": self.number,
            "shape": self.shape.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Clue":
        number = data.get("number")
        return cls(
            id=int(data["id"]),
            pos=_coerce_coordinate(data["pos"]),
            number=None if number is None else int(number),
            shape=ShapeType.from_value(data["shape"]),
        )


@dataclass(frozen=True)
class Puzzle:
    """Immutable Patches puzzle definition.

    ``solution`` may be omitted for externally-authored puzzles. Generated
    puzzles store the full rectangle tiling that solves the board.
    """

    rows: int
    cols: int
    clues: tuple[Clue, ...] | list[Clue]
    solution: tuple[Rect, ...] | list[Rect] | None = None
    difficulty: str = "custom"
    seed: int | None = None
    unique_solution: bool | None = None
    _clue_by_pos: dict[Coordinate, Clue] = field(init=False, repr=False, compare=False)
    _clue_by_id: dict[int, Clue] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        rows = int(self.rows)
        cols = int(self.cols)
        if rows <= 0 or cols <= 0:
            raise ValueError("rows and cols must be positive")

        clues = tuple(self.clues)
        if not clues:
            raise ValueError("a puzzle needs at least one clue")

        ids = [clue.id for clue in clues]
        if len(set(ids)) != len(ids):
            raise ValueError("clue ids must be unique")
        positions = [clue.pos for clue in clues]
        if len(set(positions)) != len(positions):
            raise ValueError("clue positions must be unique")
        for clue in clues:
            if not in_bounds(rows, cols, clue.pos):
                raise ValueError(f"clue out of bounds: {clue.pos!r}")
            if clue.number is not None and clue.number <= 0:
                raise ValueError(f"clue number must be positive: {clue!r}")

        # Numbers are optional, so the sizes only need to add up when every clue
        # specifies one.
        if all(clue.number is not None for clue in clues):
            total_area = sum(clue.number for clue in clues)
            if total_area != rows * cols:
                raise ValueError(
                    f"clue areas sum to {total_area} but grid has {rows * cols} cells"
                )

        solution = None
        if self.solution is not None:
            solution = tuple(self.solution)
            self._validate_solution(rows, cols, clues, solution)

        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "cols", cols)
        object.__setattr__(self, "clues", clues)
        object.__setattr__(self, "solution", solution)
        object.__setattr__(self, "_clue_by_pos", {clue.pos: clue for clue in clues})
        object.__setattr__(self, "_clue_by_id", {clue.id: clue for clue in clues})

    @staticmethod
    def _validate_solution(
        rows: int,
        cols: int,
        clues: tuple[Clue, ...],
        solution: tuple[Rect, ...],
    ) -> None:
        if len(solution) != len(clues):
            raise ValueError("solution must have one rectangle per clue")

        clue_by_pos = {clue.pos: clue for clue in clues}
        covered: dict[Coordinate, int] = {}
        matched_clue_ids: set[int] = set()

        for rect in solution:
            if not rect_in_bounds(rows, cols, rect):
                raise ValueError(f"solution rectangle out of bounds: {rect!r}")

            enclosed = [clue_by_pos[cell] for cell in rect.cells() if cell in clue_by_pos]
            if len(enclosed) != 1:
                raise ValueError(f"solution rectangle must enclose exactly one clue: {rect!r}")
            clue = enclosed[0]

            if clue.id in matched_clue_ids:
                raise ValueError(f"clue {clue.id} covered by more than one rectangle")
            matched_clue_ids.add(clue.id)

            if clue.number is not None and rect.area != clue.number:
                raise ValueError(f"rectangle area {rect.area} != clue number {clue.number}")
            if not shape_matches(clue.shape, rect.height, rect.width):
                raise ValueError(f"rectangle {rect!r} violates clue shape {clue.shape.value}")

            for cell in rect.cells():
                if cell in covered:
                    raise ValueError(f"solution rectangles overlap at {cell!r}")
                covered[cell] = clue.id

        all_cells = set(iter_grid(rows, cols))
        if set(covered) != all_cells:
            raise ValueError("solution must cover every cell exactly once")

    @property
    def total_cells(self) -> int:
        return self.rows * self.cols

    def clue_at(self, cell: Coordinate) -> Clue | None:
        return self._clue_by_pos.get(cell)

    def clue(self, clue_id: int) -> Clue:
        return self._clue_by_id[clue_id]

    def solution_rect(self, clue_id: int) -> Rect | None:
        if self.solution is None:
            return None
        target = self._clue_by_id[clue_id].pos
        for rect in self.solution:
            if rect.contains(target):
                return rect
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "cols": self.cols,
            "clues": [clue.to_dict() for clue in self.clues],
            "solution": None if self.solution is None else [rect.to_dict() for rect in self.solution],
            "difficulty": self.difficulty,
            "seed": self.seed,
            "unique_solution": self.unique_solution,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Puzzle":
        clues = [Clue.from_dict(item) for item in data["clues"]]
        solution = data.get("solution")
        rects = None if solution is None else [Rect.from_dict(item) for item in solution]
        return cls(
            rows=data["rows"],
            cols=data["cols"],
            clues=clues,
            solution=rects,
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
