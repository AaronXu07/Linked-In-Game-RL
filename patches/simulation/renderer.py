"""ANSI and simple raster image rendering for Patches puzzles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .puzzle import Puzzle
from .state import GameState
from .utils import Coordinate, ShapeType, iter_grid


@dataclass(frozen=True)
class ImageConfig:
    cell_size: int = 56
    margin: int = 18
    border_thickness: int = 3


def render_ansi(puzzle: Puzzle, state: GameState | None = None) -> str:
    """Render the board as text.

    Clue cells show ``<number><shape-glyph>`` (e.g. ``4S``, ``6W``). Other cells
    show the owning clue id when covered, or ``.`` when empty.
    """

    assignment = state.assignment if state is not None else {}
    width = max(2, len(str(max((clue.number for clue in puzzle.clues), default=0))) + 1)

    lines: list[str] = []
    for row in range(puzzle.rows):
        cells: list[str] = []
        for col in range(puzzle.cols):
            cell = (row, col)
            clue = puzzle.clue_at(cell)
            if clue is not None:
                number_text = str(clue.number) if clue.number is not None else "-"
                token = f"{number_text}{clue.shape.glyph}"
            elif cell in assignment:
                token = f":{assignment[cell]}"
            else:
                token = "."
            cells.append(token.rjust(width))
        lines.append(" ".join(cells))

    if state is not None:
        status = f"covered {state.covered_count}/{puzzle.total_cells}"
        if state.success:
            status += " solved"
        lines.append(status)
    return "\n".join(lines)


def render_image(puzzle: Puzzle, state: GameState | None = None, config: ImageConfig | None = None) -> bytes:
    """Render a static PPM image (dependency-free)."""

    config = config or ImageConfig()
    width = config.margin * 2 + puzzle.cols * config.cell_size
    height = config.margin * 2 + puzzle.rows * config.cell_size
    image = _Raster(width, height, (250, 250, 247))
    assignment = state.assignment if state is not None else {}

    for row in range(puzzle.rows):
        for col in range(puzzle.cols):
            cell = (row, col)
            x0, y0, x1, y1 = _cell_rect(cell, config)
            owner = assignment.get(cell)
            fill = (255, 255, 255) if owner is None else _patch_color(owner)
            image.fill_rect(x0 + 1, y0 + 1, x1 - 1, y1 - 1, fill)
            image.stroke_rect(x0, y0, x1, y1, (210, 210, 204), 1)

    # Thicker borders between cells owned by different patches.
    for cell in iter_grid(puzzle.rows, puzzle.cols):
        _draw_patch_borders(image, puzzle, assignment, cell, config)

    for clue in puzzle.clues:
        center = _cell_center(clue.pos, config)
        radius = max(10, config.cell_size // 3)
        image.fill_circle(center[0], center[1], radius, (40, 40, 36))
        # Free clues carry no shape hint, so only the number is shown.
        number_offset = radius // 3 if clue.shape is not ShapeType.FREE else 0
        if clue.number is not None:
            image.draw_digits(
                str(clue.number),
                center[0],
                center[1] - number_offset,
                (255, 255, 255),
                scale=max(2, config.cell_size // 22),
            )
        if clue.shape is not ShapeType.FREE:
            image.draw_digits(
                _glyph_digits(clue.shape.glyph),
                center[0],
                center[1] + radius // 2,
                (250, 220, 150),
                scale=max(1, config.cell_size // 34),
            )

    return image.to_ppm()


def save_image(
    puzzle: Puzzle,
    state: GameState | None,
    path: str | Path,
    config: ImageConfig | None = None,
) -> None:
    Path(path).write_bytes(render_image(puzzle, state, config))


_SHAPE_DIGIT = {"S": "5", "W": "0", "T": "7", "F": "4"}


def _glyph_digits(glyph: str) -> str:
    # The tiny raster font only draws digits, so map shape glyphs to a digit.
    return _SHAPE_DIGIT.get(glyph, "")


def _patch_color(clue_id: int) -> tuple[int, int, int]:
    palette = [
        (156, 199, 234),
        (247, 202, 154),
        (176, 220, 176),
        (231, 176, 200),
        (206, 190, 235),
        (240, 224, 150),
        (170, 214, 214),
        (232, 184, 168),
        (196, 205, 165),
        (214, 189, 215),
    ]
    return palette[clue_id % len(palette)]


def _cell_rect(cell: Coordinate, config: ImageConfig) -> tuple[int, int, int, int]:
    row, col = cell
    x0 = config.margin + col * config.cell_size
    y0 = config.margin + row * config.cell_size
    return x0, y0, x0 + config.cell_size, y0 + config.cell_size


def _cell_center(cell: Coordinate, config: ImageConfig) -> tuple[int, int]:
    x0, y0, x1, y1 = _cell_rect(cell, config)
    return (x0 + x1) // 2, (y0 + y1) // 2


def _draw_patch_borders(
    image: "_Raster",
    puzzle: Puzzle,
    assignment: dict[Coordinate, int],
    cell: Coordinate,
    config: ImageConfig,
) -> None:
    row, col = cell
    owner = assignment.get(cell)
    x0, y0, x1, y1 = _cell_rect(cell, config)
    thickness = config.border_thickness
    dark = (60, 60, 56)

    right = (row, col + 1)
    if col == puzzle.cols - 1 or assignment.get(right) != owner:
        if owner is not None or assignment.get(right) is not None:
            image.fill_rect(x1 - thickness, y0, x1 + thickness, y1, dark)
    bottom = (row + 1, col)
    if row == puzzle.rows - 1 or assignment.get(bottom) != owner:
        if owner is not None or assignment.get(bottom) is not None:
            image.fill_rect(x0, y1 - thickness, x1, y1 + thickness, dark)


class _Raster:
    _DIGITS = {
        "0": ("111", "101", "101", "101", "111"),
        "1": ("010", "110", "010", "010", "111"),
        "2": ("111", "001", "111", "100", "111"),
        "3": ("111", "001", "111", "001", "111"),
        "4": ("101", "101", "111", "001", "001"),
        "5": ("111", "100", "111", "001", "111"),
        "6": ("111", "100", "111", "101", "111"),
        "7": ("111", "001", "001", "001", "001"),
        "8": ("111", "101", "111", "101", "111"),
        "9": ("111", "101", "111", "001", "111"),
    }

    def __init__(self, width: int, height: int, color: tuple[int, int, int]) -> None:
        self.width = width
        self.height = height
        self.pixels = [color] * (width * height)

    def fill_rect(self, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        x0 = max(0, min(self.width, x0))
        x1 = max(0, min(self.width, x1))
        y0 = max(0, min(self.height, y0))
        y1 = max(0, min(self.height, y1))
        for y in range(y0, y1):
            offset = y * self.width
            for x in range(x0, x1):
                self.pixels[offset + x] = color

    def stroke_rect(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        color: tuple[int, int, int],
        thickness: int,
    ) -> None:
        self.fill_rect(x0, y0, x1, y0 + thickness, color)
        self.fill_rect(x0, y1 - thickness, x1, y1, color)
        self.fill_rect(x0, y0, x0 + thickness, y1, color)
        self.fill_rect(x1 - thickness, y0, x1, y1, color)

    def fill_circle(self, cx: int, cy: int, radius: int, color: tuple[int, int, int]) -> None:
        radius_squared = radius * radius
        for y in range(cy - radius, cy + radius + 1):
            for x in range(cx - radius, cx + radius + 1):
                if 0 <= x < self.width and 0 <= y < self.height:
                    if (x - cx) ** 2 + (y - cy) ** 2 <= radius_squared:
                        self.pixels[y * self.width + x] = color

    def draw_digits(
        self,
        text: str,
        cx: int,
        cy: int,
        color: tuple[int, int, int],
        scale: int,
    ) -> None:
        glyphs = [self._DIGITS[digit] for digit in text if digit in self._DIGITS]
        if not glyphs:
            return
        width = len(glyphs) * 3 * scale + (len(glyphs) - 1) * scale
        height = 5 * scale
        x = cx - width // 2
        y = cy - height // 2
        for glyph in glyphs:
            for gy, row in enumerate(glyph):
                for gx, bit in enumerate(row):
                    if bit == "1":
                        self.fill_rect(
                            x + gx * scale,
                            y + gy * scale,
                            x + (gx + 1) * scale,
                            y + (gy + 1) * scale,
                            color,
                        )
            x += 4 * scale

    def to_ppm(self) -> bytes:
        header = f"P6\n{self.width} {self.height}\n255\n".encode("ascii")
        body = bytearray()
        for red, green, blue in self.pixels:
            body.extend((red, green, blue))
        return header + bytes(body)
