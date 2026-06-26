"""ANSI and simple raster image rendering for Zip puzzles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .puzzle import Puzzle
from .simulator import legal_moves
from .state import GameState
from .utils import Coordinate, Direction, add_direction


@dataclass(frozen=True)
class ImageConfig:
    cell_size: int = 48
    margin: int = 18
    wall_thickness: int = 5


def render_ansi(puzzle: Puzzle, state: GameState | None = None, *, show_legal: bool = True) -> str:
    legal_cells: set[Coordinate] = set()
    if state is not None and show_legal:
        legal_cells = {add_direction(state.current_pos, direction) for direction in legal_moves(puzzle, state)}

    lines: list[str] = []
    for row in range(puzzle.rows):
        cell_parts: list[str] = []
        for col in range(puzzle.cols):
            cell = (row, col)
            cell_parts.append(_ansi_cell(puzzle, state, cell, legal_cells))
            if col < puzzle.cols - 1:
                wall = "|" if puzzle.has_wall(cell, (row, col + 1)) else " "
                cell_parts.append(wall)
        lines.append("".join(cell_parts).rstrip())

        if row < puzzle.rows - 1:
            wall_parts: list[str] = []
            for col in range(puzzle.cols):
                cell = (row, col)
                wall_parts.append("---" if puzzle.has_wall(cell, (row + 1, col)) else "   ")
                if col < puzzle.cols - 1:
                    wall_parts.append(" ")
            lines.append("".join(wall_parts).rstrip())
    if state is not None:
        status = f"visited {len(state.visited)}/{puzzle.total_cells}"
        if state.success:
            status += " solved"
        lines.append(status)
    return "\n".join(lines)


def _ansi_cell(
    puzzle: Puzzle,
    state: GameState | None,
    cell: Coordinate,
    legal_cells: set[Coordinate],
) -> str:
    waypoint_index = puzzle.waypoint_index(cell)
    if state is not None and cell == state.current_pos:
        marker = "@"
    elif waypoint_index is not None:
        marker = str(waypoint_index + 1)
    elif state is not None and cell in state.visited:
        marker = "*"
    elif cell in legal_cells:
        marker = "+"
    else:
        marker = "."
    return marker.center(3)


def render_image(puzzle: Puzzle, state: GameState | None = None, config: ImageConfig | None = None) -> bytes:
    """Render a static PPM image.

    PPM keeps this renderer dependency-free. Most image viewers can open it,
    and tests/debugging can parse it trivially.
    """

    config = config or ImageConfig()
    width = config.margin * 2 + puzzle.cols * config.cell_size
    height = config.margin * 2 + puzzle.rows * config.cell_size
    image = _Raster(width, height, (250, 250, 247))
    legal_cells: set[Coordinate] = set()
    if state is not None:
        legal_cells = {add_direction(state.current_pos, direction) for direction in legal_moves(puzzle, state)}

    for row in range(puzzle.rows):
        for col in range(puzzle.cols):
            cell = (row, col)
            x0, y0, x1, y1 = _cell_rect(cell, config)
            fill = (255, 255, 255)
            if cell in legal_cells:
                fill = (220, 242, 224)
            if state is not None and cell in state.visited:
                fill = (210, 225, 248)
            if puzzle.waypoint_index(cell) is not None:
                fill = (249, 230, 159)
            if state is not None and cell == state.current_pos:
                fill = (249, 174, 94)
            image.fill_rect(x0 + 1, y0 + 1, x1 - 1, y1 - 1, fill)
            image.stroke_rect(x0, y0, x1, y1, (164, 164, 156), 1)

    if state is not None and len(state.path) > 1:
        for a, b in zip(state.path, state.path[1:]):
            image.draw_cardinal_line(_cell_center(a, config), _cell_center(b, config), (42, 102, 184), 7)

    for (a, b) in puzzle.walls:
        image.draw_cardinal_line(_wall_center(a, b, config), _wall_center(a, b, config), (20, 20, 20), 1)
        _draw_wall(image, a, b, config)

    for cell in puzzle.waypoints:
        label = str((puzzle.waypoint_index(cell) or 0) + 1)
        center = _cell_center(cell, config)
        image.fill_circle(center[0], center[1], max(10, config.cell_size // 5), (48, 48, 44))
        image.draw_digits(label, center[0], center[1], (255, 255, 255), scale=max(2, config.cell_size // 24))

    if state is not None:
        image.fill_circle(*_cell_center(state.current_pos, config), max(7, config.cell_size // 7), (25, 75, 150))

    return image.to_ppm()


def save_image(
    puzzle: Puzzle,
    state: GameState | None,
    path: str | Path,
    config: ImageConfig | None = None,
) -> None:
    Path(path).write_bytes(render_image(puzzle, state, config))


def _cell_rect(cell: Coordinate, config: ImageConfig) -> tuple[int, int, int, int]:
    row, col = cell
    x0 = config.margin + col * config.cell_size
    y0 = config.margin + row * config.cell_size
    return x0, y0, x0 + config.cell_size, y0 + config.cell_size


def _cell_center(cell: Coordinate, config: ImageConfig) -> tuple[int, int]:
    x0, y0, x1, y1 = _cell_rect(cell, config)
    return (x0 + x1) // 2, (y0 + y1) // 2


def _wall_center(a: Coordinate, b: Coordinate, config: ImageConfig) -> tuple[int, int]:
    ax, ay = _cell_center(a, config)
    bx, by = _cell_center(b, config)
    return (ax + bx) // 2, (ay + by) // 2


def _draw_wall(image: "_Raster", a: Coordinate, b: Coordinate, config: ImageConfig) -> None:
    x0, y0, x1, y1 = _cell_rect(a, config)
    thickness = config.wall_thickness
    if a[0] == b[0]:
        x = x1 if a[1] < b[1] else x0
        image.fill_rect(x - thickness // 2, y0, x + thickness // 2 + 1, y1, (22, 22, 20))
    else:
        y = y1 if a[0] < b[0] else y0
        image.fill_rect(x0, y - thickness // 2, x1, y + thickness // 2 + 1, (22, 22, 20))


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

    def draw_cardinal_line(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        color: tuple[int, int, int],
        thickness: int,
    ) -> None:
        x0, y0 = start
        x1, y1 = end
        half = thickness // 2
        if x0 == x1:
            self.fill_rect(x0 - half, min(y0, y1), x0 + half + 1, max(y0, y1) + 1, color)
        elif y0 == y1:
            self.fill_rect(min(x0, x1), y0 - half, max(x0, x1) + 1, y0 + half + 1, color)

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
