"""Small coordinate helpers shared by the simulator."""

from __future__ import annotations

from enum import Enum
from typing import Iterable

Coordinate = tuple[int, int]
Wall = tuple[Coordinate, Coordinate]


class Direction(Enum):
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)

    @property
    def delta(self) -> Coordinate:
        return self.value

    @classmethod
    def all(cls) -> tuple["Direction", ...]:
        return (cls.UP, cls.DOWN, cls.LEFT, cls.RIGHT)

    @classmethod
    def from_delta(cls, delta: Coordinate) -> "Direction":
        for direction in cls:
            if direction.value == delta:
                return direction
        raise ValueError(f"{delta!r} is not a cardinal direction")

    @classmethod
    def between(cls, start: Coordinate, end: Coordinate) -> "Direction":
        return cls.from_delta((end[0] - start[0], end[1] - start[1]))


def add_direction(cell: Coordinate, direction: Direction) -> Coordinate:
    dr, dc = direction.delta
    return cell[0] + dr, cell[1] + dc


def in_bounds(rows: int, cols: int, cell: Coordinate) -> bool:
    row, col = cell
    return 0 <= row < rows and 0 <= col < cols


def manhattan_distance(a: Coordinate, b: Coordinate) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def are_adjacent(a: Coordinate, b: Coordinate) -> bool:
    return manhattan_distance(a, b) == 1


def canonical_wall(a: Coordinate, b: Coordinate) -> Wall:
    if not are_adjacent(a, b):
        raise ValueError(f"wall endpoints must be adjacent: {a!r}, {b!r}")
    first, second = sorted((a, b))
    return first, second


def neighbors(rows: int, cols: int, cell: Coordinate) -> list[Coordinate]:
    result: list[Coordinate] = []
    for direction in Direction.all():
        candidate = add_direction(cell, direction)
        if in_bounds(rows, cols, candidate):
            result.append(candidate)
    return result


def iter_grid(rows: int, cols: int) -> Iterable[Coordinate]:
    for row in range(rows):
        for col in range(cols):
            yield row, col


def coordinate_to_index(cols: int, cell: Coordinate) -> int:
    return cell[0] * cols + cell[1]


def index_to_coordinate(cols: int, index: int) -> Coordinate:
    return divmod(index, cols)


def solution_edges(path: Iterable[Coordinate]) -> set[Wall]:
    cells = list(path)
    return {canonical_wall(a, b) for a, b in zip(cells, cells[1:])}
