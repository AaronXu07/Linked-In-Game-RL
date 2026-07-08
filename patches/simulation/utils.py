"""Coordinate, rectangle, and shape helpers shared by the Patches simulator."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Iterator

Coordinate = tuple[int, int]


class ShapeType(Enum):
    """Shape constraint carried by a clue.

    The number on a clue sets the patch area; the shape sets the aspect
    constraint on the patch's ``width`` (``w``) and ``height`` (``h``):

    - ``SQUARE``: ``w == h`` (so the clue number must be a perfect square).
    - ``WIDE``:   ``w > h``.
    - ``TALL``:   ``h > w``.
    - ``FREE``:   any ``w x h`` whose area equals the number.
    """

    SQUARE = "square"
    WIDE = "wide"
    TALL = "tall"
    FREE = "free"

    @classmethod
    def from_value(cls, value: "str | ShapeType") -> "ShapeType":
        if isinstance(value, ShapeType):
            return value
        key = str(value).strip().lower()
        for shape in cls:
            if shape.value == key:
                return shape
        choices = ", ".join(shape.value for shape in cls)
        raise ValueError(f"unknown shape {value!r}; choose one of: {choices}")

    @property
    def glyph(self) -> str:
        return {"square": "S", "wide": "W", "tall": "T", "free": "F"}[self.value]


@dataclass(frozen=True)
class Rect:
    """Axis-aligned rectangle of grid cells.

    ``top``/``left`` are the inclusive top-left cell coordinates; ``height`` and
    ``width`` are cell counts (both >= 1).
    """

    top: int
    left: int
    height: int
    width: int

    def __post_init__(self) -> None:
        if self.height <= 0 or self.width <= 0:
            raise ValueError("rectangle height and width must be positive")

    @property
    def area(self) -> int:
        return self.height * self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height - 1

    @property
    def right(self) -> int:
        return self.left + self.width - 1

    def cells(self) -> Iterator[Coordinate]:
        for row in range(self.top, self.top + self.height):
            for col in range(self.left, self.left + self.width):
                yield row, col

    def contains(self, cell: Coordinate) -> bool:
        row, col = cell
        return self.top <= row <= self.bottom and self.left <= col <= self.right

    def overlaps(self, other: "Rect") -> bool:
        return not (
            self.right < other.left
            or other.right < self.left
            or self.bottom < other.top
            or other.bottom < self.top
        )

    def to_dict(self) -> dict[str, int]:
        return {"top": self.top, "left": self.left, "height": self.height, "width": self.width}

    @classmethod
    def from_dict(cls, data: dict[str, int]) -> "Rect":
        return cls(int(data["top"]), int(data["left"]), int(data["height"]), int(data["width"]))


def in_bounds(rows: int, cols: int, cell: Coordinate) -> bool:
    row, col = cell
    return 0 <= row < rows and 0 <= col < cols


def rect_in_bounds(rows: int, cols: int, rect: Rect) -> bool:
    return rect.top >= 0 and rect.left >= 0 and rect.bottom < rows and rect.right < cols


def rect_from_corners(a: Coordinate, b: Coordinate) -> Rect:
    """Build a rectangle spanning two (inclusive) corner cells.

    Useful for turning a UI drag from cell ``a`` to cell ``b`` into a ``Rect``.
    """

    top = min(a[0], b[0])
    left = min(a[1], b[1])
    height = abs(a[0] - b[0]) + 1
    width = abs(a[1] - b[1]) + 1
    return Rect(top, left, height, width)


def iter_grid(rows: int, cols: int) -> Iterable[Coordinate]:
    for row in range(rows):
        for col in range(cols):
            yield row, col


def coordinate_to_index(cols: int, cell: Coordinate) -> int:
    return cell[0] * cols + cell[1]


def index_to_coordinate(cols: int, index: int) -> Coordinate:
    return divmod(index, cols)


def shape_matches(shape: ShapeType, height: int, width: int) -> bool:
    if shape is ShapeType.SQUARE:
        return width == height
    if shape is ShapeType.WIDE:
        return width > height
    if shape is ShapeType.TALL:
        return height > width
    return True  # FREE


def shape_for_dimensions(height: int, width: int) -> ShapeType:
    """Return the tight (non-``FREE``) shape implied by a rectangle's dims."""

    if width == height:
        return ShapeType.SQUARE
    if width > height:
        return ShapeType.WIDE
    return ShapeType.TALL
