"""Mouse-playable Pygame UI for Patches puzzles.

Drag a rectangle around a clue to fill its patch; click a filled patch to clear
it. Each patch gets its own color, and the board is supersampled for crisp,
high-resolution rendering that mirrors the look of the LinkedIn game.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import DIFFICULTIES, UIConfig
from .generator import GenerationError, generate_puzzle
from .puzzle import Clue, Puzzle, load_puzzle, save_puzzle
from .rules import enclosed_clue_ids, validate_placement
from .simulator import clear_patch, new_game, place, reset, undo
from .state import GameState
from .utils import Coordinate, Rect, ShapeType, rect_from_corners


# Vibrant, fully-saturated palette. Each clue id maps to a stable color so a
# patch keeps the same color as it is drawn, cleared, and redrawn.
PATCH_PALETTE: tuple[tuple[int, int, int], ...] = (
    ( 30, 120, 220),  # vivid blue
    (255, 130,   0),  # vivid orange
    ( 30, 185,  60),  # vivid green
    (240,  50, 120),  # vivid pink
    (130,  60, 210),  # vivid purple
    (240, 200,   0),  # vivid yellow
    (  0, 185, 180),  # vivid teal
    (235,  60,  50),  # vivid red-coral
    (120, 175,   0),  # vivid lime-olive
    ( 70,  90, 220),  # vivid indigo
    (210,  60, 180),  # vivid magenta-orchid
    (  0, 160, 110),  # vivid emerald
)

BACKGROUND = (243, 242, 238)
BOARD_BG = (255, 255, 255)
GRID_DOT = (25, 25, 23)
BOARD_BORDER = (46, 46, 44)
VALID_PREVIEW = (86, 170, 96)
INVALID_PREVIEW = (211, 92, 84)
INK = (32, 32, 30)

# Higher supersample factor for crisper edges/text on modern high-DPI displays.
SUPERSAMPLE = 3


@dataclass(frozen=True)
class BoardLayout:
    rows: int
    cols: int
    cell_size: int
    origin_x: int
    origin_y: int

    @property
    def width(self) -> int:
        return self.cols * self.cell_size

    @property
    def height(self) -> int:
        return self.rows * self.cell_size

    def pixel_to_cell(self, x: int, y: int) -> Coordinate | None:
        local_x = x - self.origin_x
        local_y = y - self.origin_y
        if local_x < 0 or local_y < 0:
            return None
        col = local_x // self.cell_size
        row = local_y // self.cell_size
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return int(row), int(col)
        return None

    def clamp_to_cell(self, x: int, y: int) -> Coordinate:
        local_x = x - self.origin_x
        local_y = y - self.origin_y
        col = min(self.cols - 1, max(0, local_x // self.cell_size))
        row = min(self.rows - 1, max(0, local_y // self.cell_size))
        return int(row), int(col)

    def cell_rect(self, cell: Coordinate) -> tuple[int, int, int, int]:
        row, col = cell
        x = self.origin_x + col * self.cell_size
        y = self.origin_y + row * self.cell_size
        return x, y, self.cell_size, self.cell_size

    def cell_center(self, cell: Coordinate) -> tuple[int, int]:
        x, y, width, height = self.cell_rect(cell)
        return x + width // 2, y + height // 2

    def region_rect(self, rect: Rect) -> tuple[int, int, int, int]:
        x = self.origin_x + rect.left * self.cell_size
        y = self.origin_y + rect.top * self.cell_size
        return x, y, rect.width * self.cell_size, rect.height * self.cell_size


@dataclass
class Button:
    label: Callable[[], str] | str
    rect: tuple[int, int, int, int]
    action: Callable[[], None]

    def text(self) -> str:
        return self.label() if callable(self.label) else self.label

    def contains(self, pos: tuple[int, int]) -> bool:
        x, y = pos
        rx, ry, rw, rh = self.rect
        return rx <= x < rx + rw and ry <= y < ry + rh


class PatchesGameUI:
    def __init__(
        self,
        puzzle: Puzzle | None = None,
        *,
        difficulty: str = "easy",
        save_path: str | Path = "patches_saved_puzzle.json",
        config: UIConfig | None = None,
    ) -> None:
        self.config = config or UIConfig()
        self.difficulty_names = list(DIFFICULTIES)
        self.difficulty = difficulty if difficulty in DIFFICULTIES else "easy"
        self.save_path = Path(save_path)
        self.puzzle = puzzle or self._generate_current_difficulty()
        self.state = new_game(self.puzzle)
        self.layout = BoardLayout(
            self.puzzle.rows,
            self.puzzle.cols,
            self.config.cell_size,
            self.config.margin,
            self.config.margin,
        )
        self.buttons: list[Button] = []
        self.panel_metadata_y = self.config.margin

        self.dragging = False
        self.active_clue_id: int | None = None
        self.drag_anchor: Coordinate | None = None
        self.drag_cell: Coordinate | None = None
        # Farthest row/col offset (signed, relative to the anchor) reached so
        # far during the current drag. A drag only ever grows the patch: if
        # the cursor moves back toward the anchor, these offsets are left
        # alone instead of shrinking, so backtracking never "undoes" progress
        # made earlier in the same drag gesture.
        self.drag_row_offset = 0
        self.drag_col_offset = 0
        self.hover_cell: Coordinate | None = None

        self.feedback = "Drag a box around a clue to fill it"
        self.feedback_until = time.monotonic() + 4.0

        self.reveal_index = 0
        self.max_window_width: int | None = None
        self.max_window_height: int | None = None

    # ------------------------------------------------------------------ run
    def run(self) -> None:
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError(
                "Pygame is required for the human UI. Install pygame to use it."
            ) from exc

        pygame.init()
        self._fit_layout_to_display(pygame)
        screen = pygame.display.set_mode(self._screen_size())
        pygame.display.set_caption("Patches Game Simulator")
        clock = pygame.time.Clock()
        self._rebuild_buttons()

        running = True
        while running:
            clock.tick(self.config.fps)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self._handle_key(event.key, pygame)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_mouse_down(event.pos)
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    self._handle_mouse_up(event.pos)
                elif event.type == pygame.MOUSEMOTION:
                    self._handle_mouse_motion(event.pos)

            desired_size = self._screen_size()
            if screen.get_size() != desired_size:
                screen = pygame.display.set_mode(desired_size)
            self._draw(screen, pygame)
            pygame.display.flip()

        pygame.quit()

    def _screen_size(self) -> tuple[int, int]:
        width = (
            self.layout.origin_x
            + self.layout.width
            + self.config.panel_gap
            + self.config.panel_width
            + self.config.margin
        )
        board_height = self.layout.origin_y + self.layout.height + self.config.margin
        panel_height = self.panel_metadata_y + 360
        return width, max(board_height, panel_height, 620)

    def _fit_layout_to_display(self, pygame: object) -> None:
        display = pygame.display.Info()
        self.max_window_width = max(760, getattr(display, "current_w", 1280) - 80)
        self.max_window_height = max(620, getattr(display, "current_h", 900) - 110)
        self._fit_layout_to_window_limits()

    def _fit_layout_to_window_limits(self) -> None:
        max_width = self.max_window_width or 1280
        max_height = self.max_window_height or 900
        available_width = (
            max_width - self.config.margin * 2 - self.config.panel_gap - self.config.panel_width
        )
        available_height = max_height - self.config.margin * 2
        fitted_cell_size = min(
            self.config.cell_size,
            available_width // self.puzzle.cols,
            available_height // self.puzzle.rows,
        )
        fitted_cell_size = max(self.config.min_cell_size, int(fitted_cell_size))
        self.layout = BoardLayout(
            self.puzzle.rows,
            self.puzzle.cols,
            fitted_cell_size,
            self.config.margin,
            self.config.margin,
        )

    # ------------------------------------------------------------ puzzle mgmt
    def _generate_current_difficulty(self) -> Puzzle:
        try:
            return generate_puzzle(self.difficulty)
        except GenerationError:
            # Fallback keeps the UI usable if strict generation times out: a
            # simple 4x4 split into four 2x2 squares.
            clues = [
                Clue(id=0, pos=(0, 0), number=4, shape=ShapeType.SQUARE),
                Clue(id=1, pos=(0, 2), number=4, shape=ShapeType.SQUARE),
                Clue(id=2, pos=(2, 0), number=4, shape=ShapeType.SQUARE),
                Clue(id=3, pos=(2, 2), number=4, shape=ShapeType.SQUARE),
            ]
            solution = [Rect(0, 0, 2, 2), Rect(0, 2, 2, 2), Rect(2, 0, 2, 2), Rect(2, 2, 2, 2)]
            return Puzzle(
                rows=4,
                cols=4,
                clues=clues,
                solution=solution,
                difficulty="fallback",
                unique_solution=True,
            )

    def _set_puzzle(self, puzzle: Puzzle) -> None:
        self.puzzle = puzzle
        self.state = new_game(puzzle)
        self.reveal_index = 0
        self.dragging = False
        self.active_clue_id = None
        self.drag_anchor = None
        self.drag_cell = None
        self.drag_row_offset = 0
        self.drag_col_offset = 0
        self._fit_layout_to_window_limits()
        self._rebuild_buttons()

    def _rebuild_buttons(self) -> None:
        x = self.layout.origin_x + self.layout.width + self.config.panel_gap
        y = self.config.margin
        width = self.config.panel_width
        height = 40
        gap = 9

        specs: list[tuple[Callable[[], str] | str, Callable[[], None]]] = [
            (lambda: f"Difficulty: {self.difficulty}", self._cycle_difficulty),
            ("New Puzzle", self._new_puzzle),
            ("Undo", self._undo),
            ("Reset", self._reset),
            ("Clear All", self._clear_all),
            ("Save JSON", self._save),
            ("Load JSON", self._load),
        ]
        self.buttons = []
        for label, action in specs:
            self.buttons.append(Button(label, (x, y, width, height), action))
            y += height + gap

        half_gap = 10
        half_width = (width - half_gap) // 2
        y += 4
        self.buttons.append(Button("Reveal Step", (x, y, half_width, height), self._reveal_step))
        self.buttons.append(
            Button("Hide Step", (x + half_width + half_gap, y, half_width, height), self._reveal_back)
        )
        y += height + gap
        self.buttons.append(Button("Reveal All", (x, y, half_width, height), self._reveal_all))
        self.buttons.append(
            Button("Hide All", (x + half_width + half_gap, y, half_width, height), self._reset)
        )
        y += height + gap

        self.panel_metadata_y = y + 16

    # --------------------------------------------------------------- events
    def _handle_key(self, key: int, pygame: object) -> bool:
        if key == pygame.K_ESCAPE:
            return False
        if key == pygame.K_u:
            self._undo()
        elif key == pygame.K_r:
            self._reset()
        elif key == pygame.K_n:
            self._new_puzzle()
        elif key == pygame.K_c:
            self._clear_all()
        elif key == pygame.K_SPACE:
            self._reveal_step()
        return True

    def _handle_mouse_down(self, pos: tuple[int, int]) -> None:
        for button in self.buttons:
            if button.contains(pos):
                button.action()
                return
        cell = self.layout.pixel_to_cell(*pos)
        if cell is None:
            return
        clue = self.puzzle.clue_at(cell)
        if clue is not None:
            # A drag is anchored on the clue and grows toward the cursor. If
            # the mouse never moves away before release, this resolves as a
            # click instead (see `_handle_mouse_up`).
            self.dragging = True
            self.active_clue_id = clue.id
            self.drag_anchor = clue.pos
            self.drag_cell = cell
            self.drag_row_offset = 0
            self.drag_col_offset = 0
            self._drag_row_dir = 0
            self._drag_col_dir = 0
        else:
            # Pressing an empty/filled non-clue cell is a click (clear), not a drag.
            self.dragging = False
            self.active_clue_id = None
            self.drag_anchor = None
            self.drag_cell = None
            self.drag_row_offset = 0
            self.drag_col_offset = 0

    def _handle_mouse_motion(self, pos: tuple[int, int]) -> None:
        self.hover_cell = self.layout.pixel_to_cell(*pos)
        if self.dragging and self.drag_anchor is not None:
            cell = self.layout.clamp_to_cell(*pos)
            row_offset = cell[0] - self.drag_anchor[0]
            col_offset = cell[1] - self.drag_anchor[1]

            # Each axis locks onto the first non-zero direction the cursor
            # moves. Once a direction is locked, movement in the opposite
            # direction is ignored so the patch can only grow, never shrink.
            # The lock resets only when a new drag starts (mouse down).
            if not hasattr(self, "_drag_row_dir"):
                self._drag_row_dir = 0
                self._drag_col_dir = 0

            if self._drag_row_dir == 0 and row_offset != 0:
                self._drag_row_dir = 1 if row_offset > 0 else -1
            if self._drag_col_dir == 0 and col_offset != 0:
                self._drag_col_dir = 1 if col_offset > 0 else -1

            # Only apply the offset if it matches the locked direction;
            # movement against the locked direction is clamped to the current
            # farthest extent so the box never shrinks.
            if self._drag_row_dir == 0:
                # No direction locked yet — stay at anchor
                self.drag_row_offset = 0
            elif self._drag_row_dir > 0:
                self.drag_row_offset = max(self.drag_row_offset, max(row_offset, 0))
            else:
                self.drag_row_offset = min(self.drag_row_offset, min(row_offset, 0))

            if self._drag_col_dir == 0:
                self.drag_col_offset = 0
            elif self._drag_col_dir > 0:
                self.drag_col_offset = max(self.drag_col_offset, max(col_offset, 0))
            else:
                self.drag_col_offset = min(self.drag_col_offset, min(col_offset, 0))

            self.drag_cell = (
                self.drag_anchor[0] + self.drag_row_offset,
                self.drag_anchor[1] + self.drag_col_offset,
            )

    def _handle_mouse_up(self, pos: tuple[int, int]) -> None:
        if self.dragging and self.active_clue_id is not None:
            clue_id = self.active_clue_id
            clue = self.puzzle.clue(clue_id)
            end_cell = self.drag_cell or clue.pos
            self.dragging = False
            self.active_clue_id = None
            self.drag_anchor = None
            self.drag_cell = None
            self.drag_row_offset = 0
            self.drag_col_offset = 0
            self._drag_row_dir = 0
            self._drag_col_dir = 0

            rect = rect_from_corners(clue.pos, end_cell)
            if end_cell == clue.pos and rect.area == 1:
                # Pressed and released on the clue without ever dragging out
                # a box. If the clue already has a patch, this is a click to
                # clear it. Otherwise it either commits a size-1 patch or
                # nudges the user to drag.
                if clue_id in self.state.patches:
                    self.state = clear_patch(self.state, clue_id)
                    self.reveal_index = 0
                    self._show_feedback("Cleared patch")
                elif clue.number == 1:
                    self._place_clue(clue_id, rect)
                else:
                    self._show_feedback("Drag from the clue to size the patch")
            else:
                self._place_clue(clue_id, rect)
            return

        # Not a drag: a plain click clears whatever patch owns the cell.
        cell = self.layout.pixel_to_cell(*pos)
        if cell is not None:
            self._clear_at(cell)

    def _clear_at(self, cell: Coordinate) -> None:
        owner = self.state.assignment.get(cell)
        if owner is not None:
            self.state = clear_patch(self.state, owner)
            self.reveal_index = 0
            self._show_feedback("Cleared patch")

    def _place_clue(self, clue_id: int, rect: Rect) -> None:
        result = place(self.puzzle, self.state, clue_id, rect)
        if result.valid:
            self.state = result.state
            self.reveal_index = 0
            self._show_feedback("Solved!" if result.solved else "Patch placed")
        else:
            self._show_feedback(result.reason or "Invalid patch")

    # ------------------------------------------------------------- actions
    def _cycle_difficulty(self) -> None:
        index = self.difficulty_names.index(self.difficulty)
        self.difficulty = self.difficulty_names[(index + 1) % len(self.difficulty_names)]

    def _new_puzzle(self) -> None:
        self._set_puzzle(self._generate_current_difficulty())
        self._show_feedback("Generated new puzzle")

    def _undo(self) -> None:
        self.state = undo(self.state)
        self.reveal_index = 0

    def _reset(self) -> None:
        self.state = reset(self.puzzle)
        self.reveal_index = 0

    def _clear_all(self) -> None:
        self.state = reset(self.puzzle)
        self.reveal_index = 0
        self._show_feedback("Board cleared")

    def _save(self) -> None:
        save_puzzle(self.puzzle, self.save_path)
        self._show_feedback(f"Saved {self.save_path.name}")

    def _load(self) -> None:
        if not self.save_path.exists():
            self._show_feedback(f"No file at {self.save_path.name}")
            return
        self._set_puzzle(load_puzzle(self.save_path))
        self._show_feedback(f"Loaded {self.save_path.name}")

    def _reveal_step(self) -> None:
        if not self.puzzle.solution:
            self._show_feedback("No stored solution")
            return
        self.reveal_index = min(len(self.puzzle.solution), self.reveal_index + 1)
        self._apply_reveal()

    def _reveal_back(self) -> None:
        if not self.puzzle.solution:
            self._show_feedback("No stored solution")
            return
        self.reveal_index = max(0, self.reveal_index - 1)
        self._apply_reveal()

    def _reveal_all(self) -> None:
        if not self.puzzle.solution:
            self._show_feedback("No stored solution")
            return
        self.reveal_index = len(self.puzzle.solution)
        self._apply_reveal()

    def _apply_reveal(self) -> None:
        state = new_game(self.puzzle)
        for rect in (self.puzzle.solution or ())[: self.reveal_index]:
            enclosed = enclosed_clue_ids(self.puzzle, rect)
            if len(enclosed) != 1:
                continue
            result = place(self.puzzle, state, enclosed[0], rect)
            if result.valid:
                state = result.state
        self.state = state

    def _show_feedback(self, text: str) -> None:
        self.feedback = text
        self.feedback_until = time.monotonic() + 2.4

    # --------------------------------------------------------- preview state
    def _current_preview(self) -> tuple[Rect, int | None, bool] | None:
        """Return ``(rect, clue_id, valid)`` for the in-progress drag, if any."""

        if not self.dragging or self.active_clue_id is None or self.drag_cell is None:
            return None
        clue = self.puzzle.clue(self.active_clue_id)
        rect = rect_from_corners(clue.pos, self.drag_cell)
        valid = validate_placement(self.puzzle, self.state, self.active_clue_id, rect).valid
        return rect, self.active_clue_id, valid

    # --------------------------------------------------------------- drawing
    def _get_font(self, pygame: object, size: int, bold: bool = False) -> object:
        key = (size, bold)
        cache = getattr(self, "_font_cache", None)
        if cache is None:
            cache = {}
            self._font_cache = cache
        font = cache.get(key)
        if font is None:
            font = pygame.font.SysFont("arial", size, bold)
            cache[key] = font
        return font

    def _patch_color(self, clue_id: int) -> tuple[int, int, int]:
        return PATCH_PALETTE[clue_id % len(PATCH_PALETTE)]

    @staticmethod
    def _darken(color: tuple[int, int, int], factor: float = 0.78) -> tuple[int, int, int]:
        return tuple(max(0, min(255, int(channel * factor))) for channel in color)

    @staticmethod
    def _lighten(color: tuple[int, int, int], factor: float = 0.55) -> tuple[int, int, int]:
        return tuple(max(0, min(255, int(channel + (255 - channel) * factor))) for channel in color)

    def _round_rect_alpha(
        self,
        surface: object,
        pygame: object,
        rect: tuple[int, int, int, int],
        rgba: tuple[int, int, int, int],
        radius: int,
    ) -> None:
        x, y, w, h = rect
        if w <= 0 or h <= 0:
            return
        temp = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(temp, rgba, (0, 0, w, h), border_radius=radius)
        surface.blit(temp, (x, y))

    def _draw(self, screen: object, pygame: object) -> None:
        screen.fill(BACKGROUND)
        self._draw_board(screen, pygame)
        self._draw_panel(screen, pygame)
        if self.state.success:
            self._draw_solved_banner(screen, pygame)

    def _draw_board(self, screen: object, pygame: object) -> None:
        scale = SUPERSAMPLE
        padding = max(6, self.layout.cell_size // 2)
        surf_w = (self.layout.width + padding * 2) * scale
        surf_h = (self.layout.height + padding * 2) * scale
        surface = pygame.Surface((surf_w, surf_h), pygame.SRCALPHA)
        scaled_layout = BoardLayout(
            self.layout.rows,
            self.layout.cols,
            self.layout.cell_size * scale,
            padding * scale,
            padding * scale,
        )
        self._draw_board_content(surface, pygame, scaled_layout)

        # Every interior shape (cells, patches, previews) is drawn with square
        # corners. Rounding is applied exactly once here, as a mask over the
        # whole board rectangle, so only the outside of the board is rounded.
        board_radius = max(6, int(scaled_layout.cell_size * 0.14))
        board_rect = (
            scaled_layout.origin_x,
            scaled_layout.origin_y,
            scaled_layout.width,
            scaled_layout.height,
        )
        mask = pygame.Surface((surf_w, surf_h), pygame.SRCALPHA)
        pygame.draw.rect(mask, (255, 255, 255, 255), board_rect, border_radius=board_radius)
        surface.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

        # The border stroke is drawn after masking, directly with rounded
        # corners, so the board outline itself stays crisp and rounded.
        pygame.draw.rect(
            surface, BOARD_BORDER, board_rect, max(2, int(scaled_layout.cell_size * 0.045)),
            border_radius=board_radius,
        )

        target = (surf_w // scale, surf_h // scale)
        smoothed = pygame.transform.smoothscale(surface, target)
        screen.blit(smoothed, (self.layout.origin_x - padding, self.layout.origin_y - padding))

    def _draw_board_content(self, surface: object, pygame: object, layout: BoardLayout) -> None:
        cell = layout.cell_size
        board_rect = (layout.origin_x, layout.origin_y, layout.width, layout.height)

        # Square-cornered background; outer rounding is applied later as a
        # mask over the whole board, not per-cell.
        pygame.draw.rect(surface, BOARD_BG, board_rect)

        for clue_id, rect in self.state.patches.items():
            self._draw_patch(surface, pygame, layout, rect, self._patch_color(clue_id))

        preview = self._current_preview()
        if preview is not None:
            self._draw_preview(surface, pygame, layout, preview)

        self._draw_grid_lines(surface, pygame, layout)

        pill_font = self._get_font(pygame, max(13, int(cell * 0.2)), True)
        unit = max(10, int(cell * 0.46))
        token_font = self._get_font(pygame, max(13, int(unit * 0.74)), True)
        for clue in self.puzzle.clues:
            self._draw_clue(surface, pygame, layout, clue, unit, token_font)

        if preview is not None:
            self._draw_preview_label(surface, pygame, layout, preview, pill_font)

    def _draw_grid_lines(self, surface: object, pygame: object, layout: BoardLayout) -> None:
        """Separate every grid cell with a black dotted line.

        Lines are drawn on interior boundaries only; the outer edge of the
        board gets its own solid border, drawn separately in ``_draw_board``.
        """

        cell = layout.cell_size
        thickness = max(1, int(cell * 0.02))
        dash = max(3, int(cell * 0.09))
        gap = max(3, int(cell * 0.07))

        for col in range(1, self.puzzle.cols):
            x = layout.origin_x + col * cell
            self._draw_dotted_line(
                surface, pygame, (x, layout.origin_y), (x, layout.origin_y + layout.height),
                GRID_DOT, thickness, dash, gap,
            )
        for row in range(1, self.puzzle.rows):
            y = layout.origin_y + row * cell
            self._draw_dotted_line(
                surface, pygame, (layout.origin_x, y), (layout.origin_x + layout.width, y),
                GRID_DOT, thickness, dash, gap,
            )

    @staticmethod
    def _draw_dotted_line(
        surface: object,
        pygame: object,
        start: tuple[int, int],
        end: tuple[int, int],
        color: tuple[int, int, int],
        thickness: int,
        dash: int,
        gap: int,
    ) -> None:
        x0, y0 = start
        x1, y1 = end
        step = dash + gap
        if x0 == x1:
            y = y0
            while y < y1:
                seg = min(dash, y1 - y)
                pygame.draw.rect(surface, color, (x0 - thickness // 2, y, thickness, seg))
                y += step
        elif y0 == y1:
            x = x0
            while x < x1:
                seg = min(dash, x1 - x)
                pygame.draw.rect(surface, color, (x, y0 - thickness // 2, seg, thickness))
                x += step

    def _draw_patch(
        self,
        surface: object,
        pygame: object,
        layout: BoardLayout,
        rect: Rect,
        color: tuple[int, int, int],
    ) -> None:
        region = layout.region_rect(rect)
        # Light translucent fill with a saturated border, so the full-color clue
        # token reads clearly on top (matching the LinkedIn look). Corners stay
        # square; only the outer board edge is rounded.
        pygame.draw.rect(surface, self._lighten(color, 0.6), region)
        pygame.draw.rect(surface, color, region, max(3, int(layout.cell_size * 0.05)))

    def _draw_preview(
        self,
        surface: object,
        pygame: object,
        layout: BoardLayout,
        preview: tuple[Rect, int | None, bool],
    ) -> None:
        rect, clue_id, valid = preview
        region = layout.region_rect(rect)
        base = self._patch_color(clue_id) if clue_id is not None else INVALID_PREVIEW
        outline = VALID_PREVIEW if valid else INVALID_PREVIEW
        temp = pygame.Surface((region[2], region[3]), pygame.SRCALPHA)
        temp.fill((*base, 110))
        surface.blit(temp, (region[0], region[1]))
        pygame.draw.rect(surface, outline, region, max(3, int(layout.cell_size * 0.06)))

    def _draw_preview_label(
        self,
        surface: object,
        pygame: object,
        layout: BoardLayout,
        preview: tuple[Rect, int | None, bool],
        font: object,
    ) -> None:
        rect, clue_id, _ = preview
        if clue_id is not None:
            need = self.puzzle.clue(clue_id).number
            text = f"{rect.area}/{need}"
        else:
            text = str(rect.area)
        label = font.render(text, True, (255, 255, 255))
        cx, cy, w, h = layout.region_rect(rect)
        center = (cx + w // 2, cy + h // 2)
        pad_x, pad_y = max(6, layout.cell_size // 12), max(3, layout.cell_size // 24)
        pill = (
            center[0] - label.get_width() // 2 - pad_x,
            center[1] - label.get_height() // 2 - pad_y,
            label.get_width() + pad_x * 2,
            label.get_height() + pad_y * 2,
        )
        self._round_rect_alpha(surface, pygame, pill, (24, 24, 22, 205), pill[3] // 2)
        surface.blit(label, label.get_rect(center=center))

    def _draw_clue(
        self,
        surface: object,
        pygame: object,
        layout: BoardLayout,
        clue: Clue,
        unit: int,
        font: object,
    ) -> None:
        center = layout.cell_center(clue.pos)
        color = self._patch_color(clue.id)
        self._draw_shape_token(surface, pygame, center, clue.shape, unit, color, clue.number, font)

    @staticmethod
    def _token_dimensions(shape: ShapeType, unit: int) -> tuple[int, int]:
        """Return ``(width, height)`` of the clue token for a shape."""

        if shape is ShapeType.WIDE:
            return int(unit * 1.5), int(unit * 0.82)
        if shape is ShapeType.TALL:
            return int(unit * 0.82), int(unit * 1.5)
        return unit, unit  # SQUARE and FREE use a square silhouette

    def _draw_shape_token(
        self,
        surface: object,
        pygame: object,
        center: tuple[int, int],
        shape: ShapeType,
        unit: int,
        color: tuple[int, int, int],
        number: int | None,
        font: object | None,
    ) -> None:
        cx, cy = center
        w, h = self._token_dimensions(shape, unit)
        rect = (cx - w // 2, cy - h // 2, w, h)
        radius = max(2, int(min(w, h) * 0.26))

        if shape is ShapeType.FREE:
            # "Any of the above": a dashed square, faint fill, in the patch color.
            self._round_rect_alpha(surface, pygame, rect, (*color, 70), radius)
            self._draw_dashed_rect(
                surface, pygame, rect, color, max(2, unit // 8), max(3, unit // 4)
            )
            number_color = self._darken(color, 0.62)
        else:
            pygame.draw.rect(surface, color, rect, border_radius=radius)
            number_color = (255, 255, 255)

        if number is not None and font is not None:
            label = font.render(str(number), True, number_color)
            surface.blit(label, label.get_rect(center=center))

    @staticmethod
    def _draw_dashed_rect(
        surface: object,
        pygame: object,
        rect: tuple[int, int, int, int],
        color: tuple[int, int, int],
        thickness: int,
        dash: int,
    ) -> None:
        x, y, w, h = rect
        step = dash * 2
        px = x
        while px < x + w:
            seg = min(dash, x + w - px)
            pygame.draw.rect(surface, color, (px, y, seg, thickness))
            pygame.draw.rect(surface, color, (px, y + h - thickness, seg, thickness))
            px += step
        py = y
        while py < y + h:
            seg = min(dash, y + h - py)
            pygame.draw.rect(surface, color, (x, py, thickness, seg))
            pygame.draw.rect(surface, color, (x + w - thickness, py, thickness, seg))
            py += step

    def _draw_panel(self, screen: object, pygame: object) -> None:
        button_font = self._get_font(pygame, 16, True)
        label_font = self._get_font(pygame, 16)
        head_font = self._get_font(pygame, 20, True)
        mouse_pos = pygame.mouse.get_pos()

        panel_x = self.layout.origin_x + self.layout.width + self.config.panel_gap
        title = head_font.render("Patches", True, INK)
        screen.blit(title, (panel_x, self.config.margin - 26 if self.config.margin > 30 else 4))

        for button in self.buttons:
            hovered = button.contains(mouse_pos)
            fill = (24, 24, 22) if hovered else (255, 255, 255)
            text_color = (255, 255, 255) if hovered else INK
            pygame.draw.rect(screen, fill, button.rect, border_radius=9)
            pygame.draw.rect(screen, (36, 36, 34), button.rect, 1, border_radius=9)
            surface = button_font.render(button.text(), True, text_color)
            screen.blit(
                surface,
                surface.get_rect(center=(button.rect[0] + button.rect[2] // 2, button.rect[1] + button.rect[3] // 2)),
            )

        x = panel_x
        y = self.panel_metadata_y
        placed = len(self.state.patches)
        total_clues = len(self.puzzle.clues)
        metadata = [
            f"{self.puzzle.rows}x{self.puzzle.cols}  ({self.puzzle.difficulty})",
            f"seed: {self.puzzle.seed}",
            f"unique: {self.puzzle.unique_solution}",
            f"patches: {placed}/{total_clues}",
            f"covered: {self.state.covered_count}/{self.puzzle.total_cells}",
        ]
        for line in metadata:
            surface = label_font.render(line, True, (60, 60, 56))
            screen.blit(surface, (x, y))
            y += 24

        y += 12
        legend_title = self._get_font(pygame, 15, True)
        screen.blit(legend_title.render("Complete each shape to fill the grid", True, INK), (x, y))
        y += 28

        legend_font = self._get_font(pygame, 14)
        gray = (112, 112, 108)
        rows = [
            (ShapeType.SQUARE, "Square"),
            (ShapeType.TALL, "Tall rectangle"),
            (ShapeType.WIDE, "Wide rectangle"),
            (ShapeType.FREE, "Any of the above"),
        ]
        for shape, text in rows:
            self._draw_shape_token(screen, pygame, (x + 16, y + 15), shape, 16, gray, None, None)
            screen.blit(legend_font.render(text, True, (72, 72, 68)), (x + 42, y + 6))
            y += 30

        note_font = self._get_font(pygame, 13)
        notes = [
            "If a shape has a number, it must be that size.",
            "Press a clue and drag to size it. Click to clear.",
            "Keys: u undo  r reset  n new  c clear",
        ]
        y += 4
        for line in notes:
            screen.blit(note_font.render(line, True, (124, 124, 118)), (x, y))
            y += 21

        if self.feedback and time.monotonic() < self.feedback_until:
            surface = label_font.render(self.feedback, True, (188, 96, 40))
            screen.blit(surface, (x, y + 8))

    def _draw_solved_banner(self, screen: object, pygame: object) -> None:
        board_w = self.layout.width
        board_h = self.layout.height
        banner_w = min(board_w - 20, 360)
        banner_h = 96
        cx = self.layout.origin_x + board_w // 2
        cy = self.layout.origin_y + board_h // 2
        rect = (cx - banner_w // 2, cy - banner_h // 2, banner_w, banner_h)
        self._round_rect_alpha(screen, pygame, rect, (24, 24, 22, 225), 20)
        big = self._get_font(pygame, 46, True)
        small = self._get_font(pygame, 18)
        title = big.render("Solved!", True, (255, 255, 255))
        subtitle = small.render("Press n for a new puzzle", True, (214, 214, 210))
        screen.blit(title, title.get_rect(center=(cx, cy - 14)))
        screen.blit(subtitle, subtitle.get_rect(center=(cx, cy + 26)))


def run_ui(
    puzzle: Puzzle | None = None,
    *,
    difficulty: str = "easy",
    save_path: str | Path = "patches_saved_puzzle.json",
) -> None:
    PatchesGameUI(puzzle, difficulty=difficulty, save_path=save_path).run()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Play a LinkedIn Patches-style puzzle.")
    parser.add_argument("--difficulty", default="easy", choices=sorted(DIFFICULTIES))
    parser.add_argument("--puzzle", type=Path)
    parser.add_argument("--save-path", type=Path, default=Path("patches_saved_puzzle.json"))
    args = parser.parse_args(argv)

    puzzle = load_puzzle(args.puzzle) if args.puzzle else None
    run_ui(puzzle, difficulty=args.difficulty, save_path=args.save_path)


if __name__ == "__main__":
    main()
