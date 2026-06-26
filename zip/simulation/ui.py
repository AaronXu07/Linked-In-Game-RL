"""Mouse-playable Pygame UI for Zip puzzles."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import DIFFICULTIES, UIConfig
from .generator import GenerationError, generate_puzzle
from .puzzle import Puzzle, load_puzzle, save_puzzle
from .simulator import new_game, reset, state_from_path, step, undo
from .state import GameState
from .utils import Coordinate, Direction, are_adjacent


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

    def cell_rect(self, cell: Coordinate) -> tuple[int, int, int, int]:
        row, col = cell
        x = self.origin_x + col * self.cell_size
        y = self.origin_y + row * self.cell_size
        return x, y, self.cell_size, self.cell_size

    def cell_center(self, cell: Coordinate) -> tuple[int, int]:
        x, y, width, height = self.cell_rect(cell)
        return x + width // 2, y + height // 2


@dataclass
class Button:
    label: str
    rect: tuple[int, int, int, int]
    action: Callable[[], None]

    def contains(self, pos: tuple[int, int]) -> bool:
        x, y = pos
        rx, ry, rw, rh = self.rect
        return rx <= x < rx + rw and ry <= y < ry + rh


@dataclass
class PathMotion:
    from_path: list[Coordinate]
    to_path: list[Coordinate]
    elapsed: float = 0.0


class ZipGameUI:
    def __init__(
        self,
        puzzle: Puzzle | None = None,
        *,
        difficulty: str = "easy",
        save_path: str | Path = "zip_saved_puzzle.json",
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
        self.dragging = False
        self.hover_cell: Coordinate | None = None
        self.feedback = ""
        self.feedback_until = 0.0
        self.playback_active = False
        self.playback_index = 0
        self.playback_accumulator = 0.0
        self.playback_speed = self.config.playback_cells_per_second
        self.path_motion: PathMotion | None = None
        self.visual_path = list(self.state.path)
        self.max_window_width: int | None = None
        self.max_window_height: int | None = None

    def run(self) -> None:
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError("Pygame is required for the human UI. Install pygame to use it.") from exc

        pygame.init()
        self._fit_layout_to_display(pygame)
        screen = pygame.display.set_mode(self._screen_size())
        pygame.display.set_caption("Zip Game Simulator")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("arial", max(24, int(self.layout.cell_size * 0.42)), True)
        small_font = pygame.font.SysFont("arial", 15)
        font_cell_size = self.layout.cell_size
        self._rebuild_buttons()

        running = True
        while running:
            dt = clock.tick(self.config.fps) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self._handle_key(event.key, pygame)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_mouse_down(event.pos)
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    self.dragging = False
                elif event.type == pygame.MOUSEMOTION:
                    self._handle_mouse_motion(event.pos)

            self._update_playback(dt)
            self._update_path_motion(dt)
            desired_size = self._screen_size()
            if screen.get_size() != desired_size:
                screen = pygame.display.set_mode(desired_size)
            if font_cell_size != self.layout.cell_size:
                font = pygame.font.SysFont("arial", max(24, int(self.layout.cell_size * 0.42)), True)
                font_cell_size = self.layout.cell_size
            self._draw(screen, font, small_font, pygame)
            pygame.display.flip()

        pygame.quit()

    def _screen_size(self) -> tuple[int, int]:
        width = self.layout.origin_x + self.layout.width + self.config.panel_gap + self.config.panel_width + self.config.margin
        board_height = self.layout.origin_y + self.layout.height + self.config.margin
        return width, max(board_height, 640)

    def _fit_layout_to_display(self, pygame: object) -> None:
        display = pygame.display.Info()
        self.max_window_width = max(760, getattr(display, "current_w", 1280) - 80)
        self.max_window_height = max(640, getattr(display, "current_h", 900) - 110)
        self._fit_layout_to_window_limits()

    def _fit_layout_to_window_limits(self) -> None:
        max_width = self.max_window_width or 1280
        max_height = self.max_window_height or 900
        available_width = max_width - self.config.margin * 2 - self.config.panel_gap - self.config.panel_width
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
        self._snap_visual_path()

    def _generate_current_difficulty(self) -> Puzzle:
        try:
            return generate_puzzle(self.difficulty)
        except GenerationError:
            # Fallback keeps the UI usable even if a strict unique puzzle times out.
            solution = (
                (0, 0),
                (0, 1),
                (0, 2),
                (1, 2),
                (1, 1),
                (1, 0),
                (2, 0),
                (2, 1),
                (2, 2),
            )
            return Puzzle(
                rows=3,
                cols=3,
                waypoints=(solution[0], solution[4], solution[-1]),
                walls=frozenset(),
                solution=solution,
                difficulty="fallback",
                seed=None,
                unique_solution=None,
            )

    def _set_puzzle(self, puzzle: Puzzle) -> None:
        self.puzzle = puzzle
        self.state = new_game(puzzle)
        self._fit_layout_to_window_limits()
        self.playback_active = False
        self.playback_index = 0
        self._snap_visual_path()
        self._rebuild_buttons()

    def _rebuild_buttons(self) -> None:
        x = self.layout.origin_x + self.layout.width + self.config.panel_gap
        y = self.config.margin
        width = self.config.panel_width
        height = 38
        gap = 8

        specs: list[tuple[str, Callable[[], None]]] = [
            (f"Difficulty: {self.difficulty}", self._cycle_difficulty),
            ("New Puzzle", self._new_puzzle),
            ("Undo", self._undo),
            ("Reset", self._reset),
            ("Save JSON", self._save),
            ("Load JSON", self._load),
            ("Replay Solution", self._toggle_replay),
        ]
        self.buttons = []
        for label, action in specs:
            self.buttons.append(Button(label, (x, y, width, height), action))
            y += height + gap

        half_gap = 10
        half_width = (width - half_gap) // 2
        y += 4
        compact_rows: list[tuple[tuple[str, Callable[[], None]], tuple[str, Callable[[], None]]]] = [
            (("Step Back", self._playback_step_back), ("Step Forward", self._playback_step_forward)),
            (("Speed -", self._slower), ("Speed +", self._faster)),
        ]
        for left, right in compact_rows:
            self.buttons.append(Button(left[0], (x, y, half_width, height), left[1]))
            self.buttons.append(Button(right[0], (x + half_width + half_gap, y, half_width, height), right[1]))
            y += height + gap

        self.panel_metadata_y = y + 16

    def _handle_key(self, key: int, pygame: object) -> bool:
        if key == pygame.K_ESCAPE:
            return False
        if key == pygame.K_u:
            self._undo()
        elif key == pygame.K_r:
            self._reset()
        elif key == pygame.K_n:
            self._new_puzzle()
        elif key == pygame.K_SPACE:
            self._toggle_replay()
        return True

    def _handle_mouse_down(self, pos: tuple[int, int]) -> None:
        for button in self.buttons:
            if button.contains(pos):
                button.action()
                return
        cell = self.layout.pixel_to_cell(*pos)
        if cell is not None:
            self.dragging = True
            self._try_move_to_cell(cell)

    def _handle_mouse_motion(self, pos: tuple[int, int]) -> None:
        self.hover_cell = self.layout.pixel_to_cell(*pos)
        if self.dragging and self.hover_cell is not None:
            self._try_move_to_cell(self.hover_cell)

    def _try_move_to_cell(self, cell: Coordinate) -> None:
        if cell == self.state.current_pos:
            return
        if cell in self.state.path:
            index = self.state.path.index(cell)
            if index == len(self.state.path) - 2:
                self._go_back_to_path_index(index)
            return
        if not are_adjacent(self.state.current_pos, cell):
            self._show_feedback("Pick an adjacent cell")
            return
        direction = Direction.between(self.state.current_pos, cell)
        result = step(self.puzzle, self.state, direction)
        if result.valid:
            self._replace_state(result.state)
            self.playback_active = False
            if result.solved:
                self._show_feedback("Solved")
        else:
            self._show_feedback(result.reason or "Invalid move")

    def _go_back_to_path_index(self, index: int) -> None:
        self._replace_state(state_from_path(self.puzzle, self.state.path[: index + 1]))
        self.playback_active = False

    def _cycle_difficulty(self) -> None:
        index = self.difficulty_names.index(self.difficulty)
        self.difficulty = self.difficulty_names[(index + 1) % len(self.difficulty_names)]
        self._rebuild_buttons()

    def _new_puzzle(self) -> None:
        self._set_puzzle(self._generate_current_difficulty())
        self._show_feedback("Generated new puzzle")

    def _undo(self) -> None:
        self._replace_state(undo(self.state))
        self.playback_active = False

    def _reset(self) -> None:
        self._replace_state(reset(self.puzzle), animate=False)
        self.playback_active = False

    def _save(self) -> None:
        save_puzzle(self.puzzle, self.save_path)
        self._show_feedback(f"Saved {self.save_path}")

    def _load(self) -> None:
        if not self.save_path.exists():
            self._show_feedback(f"No file at {self.save_path}")
            return
        self._set_puzzle(load_puzzle(self.save_path))
        self._show_feedback(f"Loaded {self.save_path}")

    def _toggle_replay(self) -> None:
        if not self.puzzle.solution:
            self._show_feedback("No stored solution")
            return
        self.playback_active = not self.playback_active
        if self.playback_active and self.playback_index >= len(self.puzzle.solution) - 1:
            self.playback_index = 0
            self._set_playback_index(0)

    def _playback_step_back(self) -> None:
        if not self.puzzle.solution:
            self._show_feedback("No stored solution")
            return
        self.playback_active = False
        self._set_playback_index(max(0, self.playback_index - 1))

    def _playback_step_forward(self) -> None:
        if not self.puzzle.solution:
            self._show_feedback("No stored solution")
            return
        self.playback_active = False
        self._set_playback_index(min(len(self.puzzle.solution) - 1, self.playback_index + 1))

    def _set_playback_index(self, index: int) -> None:
        if not self.puzzle.solution:
            return
        self.playback_index = index
        self._replace_state(state_from_path(self.puzzle, self.puzzle.solution[: index + 1]))

    def _slower(self) -> None:
        self.playback_speed = max(1.0, self.playback_speed - 1.0)
        self._show_feedback(f"Speed {self.playback_speed:g}")

    def _faster(self) -> None:
        self.playback_speed = min(20.0, self.playback_speed + 1.0)
        self._show_feedback(f"Speed {self.playback_speed:g}")

    def _update_playback(self, dt: float) -> None:
        if not self.playback_active or not self.puzzle.solution:
            return
        self.playback_accumulator += dt * self.playback_speed
        while self.playback_accumulator >= 1.0 and self.playback_active:
            self.playback_accumulator -= 1.0
            if self.playback_index >= len(self.puzzle.solution) - 1:
                self.playback_active = False
                self._show_feedback("Playback complete")
                break
            self._set_playback_index(self.playback_index + 1)

    def _replace_state(self, state: GameState, *, animate: bool = True) -> None:
        old_path = list(self.state.path)
        self.state = state
        self._start_path_motion(old_path, list(state.path), animate=animate)

    def _start_path_motion(self, old_path: list[Coordinate], new_path: list[Coordinate], *, animate: bool) -> None:
        is_single_step = abs(len(new_path) - len(old_path)) == 1
        shares_prefix = (
            new_path[:-1] == old_path
            if len(new_path) > len(old_path)
            else old_path[:-1] == new_path
        )
        if animate and is_single_step and shares_prefix and old_path[-1] != new_path[-1]:
            self.path_motion = PathMotion(old_path, new_path)
        else:
            self.visual_path = list(new_path)
            self.path_motion = None

    def _update_path_motion(self, dt: float) -> None:
        if self.path_motion is None:
            return
        self.path_motion.elapsed += dt
        if self.path_motion.elapsed >= self.config.path_animation_seconds:
            self.visual_path = list(self.path_motion.to_path)
            self.path_motion = None

    def _snap_visual_path(self) -> None:
        self.visual_path = list(self.state.path)
        self.path_motion = None

    def _show_feedback(self, text: str) -> None:
        self.feedback = text
        self.feedback_until = time.monotonic() + 2.2

    def _draw(self, screen: object, font: object, small_font: object, pygame: object) -> None:
        screen.fill((244, 244, 241))
        self._draw_board(screen, font, pygame)
        self._draw_panel(screen, font, small_font, pygame)

    def _draw_board(self, screen: object, font: object, pygame: object) -> None:
        scale = max(1, self.config.board_render_scale)
        if scale == 1:
            self._draw_board_content(screen, font, pygame, self.layout)
            return

        padding = max(8, self.layout.cell_size // 2)
        surface_size = (
            (self.layout.width + padding * 2) * scale,
            (self.layout.height + padding * 2) * scale,
        )
        board_surface = pygame.Surface(surface_size, pygame.SRCALPHA)
        scaled_layout = BoardLayout(
            self.layout.rows,
            self.layout.cols,
            self.layout.cell_size * scale,
            padding * scale,
            padding * scale,
        )
        scaled_font = pygame.font.SysFont("arial", max(24, int(scaled_layout.cell_size * 0.42)), True)
        self._draw_board_content(board_surface, scaled_font, pygame, scaled_layout)
        target_size = (surface_size[0] // scale, surface_size[1] // scale)
        smoothed = pygame.transform.smoothscale(board_surface, target_size)
        screen.blit(smoothed, (self.layout.origin_x - padding, self.layout.origin_y - padding))

    def _draw_board_content(self, screen: object, font: object, pygame: object, layout: BoardLayout) -> None:
        board_rect = (layout.origin_x, layout.origin_y, layout.width, layout.height)
        board_radius = max(18, layout.cell_size // 4)
        pygame.draw.rect(screen, (255, 255, 255), board_rect, border_radius=board_radius)

        for cell in self.state.path:
            self._draw_path_cell_background(screen, pygame, cell, layout, board_radius)

        for row in range(1, self.puzzle.rows):
            y = layout.origin_y + row * layout.cell_size
            pygame.draw.line(screen, (28, 28, 28), (layout.origin_x, y), (layout.origin_x + layout.width, y), max(2, layout.cell_size // 56))
        for col in range(1, self.puzzle.cols):
            x = layout.origin_x + col * layout.cell_size
            pygame.draw.line(screen, (28, 28, 28), (x, layout.origin_y), (x, layout.origin_y + layout.height), max(2, layout.cell_size // 56))

        points = self._visual_path_points(layout)
        if len(points) > 1:
            self._draw_fluid_path(screen, pygame, points, layout)

        for a, b in self.puzzle.walls:
            self._draw_wall(screen, pygame, a, b, layout)

        for cell in self.puzzle.waypoints:
            center = layout.cell_center(cell)
            pygame.draw.circle(screen, (12, 12, 12), center, max(30, int(layout.cell_size * 0.42)))
            label = str((self.puzzle.waypoint_index(cell) or 0) + 1)
            surface = font.render(label, True, (255, 255, 255))
            screen.blit(surface, surface.get_rect(center=center))

        current_center = points[-1] if points else layout.cell_center(self.state.current_pos)
        if self.puzzle.waypoint_index(self.state.current_pos) is None:
            pygame.draw.circle(
                screen,
                self._path_color(self._path_fill()),
                (round(current_center[0]), round(current_center[1])),
                max(18, layout.cell_size // 5),
            )

        pygame.draw.rect(screen, (12, 12, 12), board_rect, max(4, layout.cell_size // 20), border_radius=board_radius)

    def _draw_path_cell_background(
        self,
        screen: object,
        pygame: object,
        cell: Coordinate,
        layout: BoardLayout,
        board_radius: int,
    ) -> None:
        row, col = cell
        pygame.draw.rect(
            screen,
            (219, 242, 255),
            layout.cell_rect(cell),
            0,
            border_top_left_radius=board_radius if row == 0 and col == 0 else 0,
            border_top_right_radius=board_radius if row == 0 and col == self.puzzle.cols - 1 else 0,
            border_bottom_left_radius=board_radius if row == self.puzzle.rows - 1 and col == 0 else 0,
            border_bottom_right_radius=board_radius if row == self.puzzle.rows - 1 and col == self.puzzle.cols - 1 else 0,
        )

    def _visual_path_points(self, layout: BoardLayout) -> list[tuple[float, float]]:
        if self.path_motion is None:
            return [layout.cell_center(cell) for cell in self.visual_path]

        motion = self.path_motion
        duration = max(0.001, self.config.path_animation_seconds)
        progress = min(1.0, motion.elapsed / duration)
        progress = progress * progress * (3.0 - 2.0 * progress)
        start = layout.cell_center(motion.from_path[-1])
        end = layout.cell_center(motion.to_path[-1])
        head = (
            start[0] + (end[0] - start[0]) * progress,
            start[1] + (end[1] - start[1]) * progress,
        )
        if len(motion.to_path) > len(motion.from_path):
            cells = motion.from_path
        else:
            cells = motion.to_path[:-1]
        return [layout.cell_center(cell) for cell in cells] + [head]

    def _draw_fluid_path(self, screen: object, pygame: object, points: list[tuple[float, float]], layout: BoardLayout) -> None:
        curve = self._rounded_path_points(points, layout)
        width = max(24, layout.cell_size // 3)
        int_points = [(round(x), round(y)) for x, y in curve]
        radius = width // 2
        path_fill = self._path_fill()
        for index, (start, end) in enumerate(zip(int_points, int_points[1:])):
            progress = (index / max(1, len(int_points) - 2)) * path_fill
            color = self._path_color(progress)
            pygame.draw.line(screen, color, start, end, width)
            pygame.draw.circle(screen, color, start, radius)
        pygame.draw.circle(screen, self._path_color(path_fill), int_points[-1], radius)

    def _path_fill(self) -> float:
        return min(1.0, max(0.0, (len(self.state.path) - 1) / max(1, self.puzzle.total_cells - 1)))

    def _path_color(self, progress: float) -> tuple[int, int, int]:
        light = (184, 231, 255)
        dark = (0, 82, 190)
        progress = min(1.0, max(0.0, progress))
        return tuple(round(light[channel] + (dark[channel] - light[channel]) * progress) for channel in range(3))

    def _rounded_path_points(self, points: list[tuple[float, float]], layout: BoardLayout) -> list[tuple[float, float]]:
        if len(points) <= 2:
            return points

        radius = layout.cell_size * 0.33
        rounded: list[tuple[float, float]] = [points[0]]
        for index in range(1, len(points) - 1):
            previous = points[index - 1]
            current = points[index]
            following = points[index + 1]
            incoming = (current[0] - previous[0], current[1] - previous[1])
            outgoing = (following[0] - current[0], following[1] - current[1])
            incoming_length = max(1.0, abs(incoming[0]) + abs(incoming[1]))
            outgoing_length = max(1.0, abs(outgoing[0]) + abs(outgoing[1]))
            turn_radius = min(radius, incoming_length * 0.48, outgoing_length * 0.48)
            before = (
                current[0] - incoming[0] / incoming_length * turn_radius,
                current[1] - incoming[1] / incoming_length * turn_radius,
            )
            after = (
                current[0] + outgoing[0] / outgoing_length * turn_radius,
                current[1] + outgoing[1] / outgoing_length * turn_radius,
            )
            rounded.append(before)
            if incoming != outgoing:
                for step_index in range(1, 7):
                    t = step_index / 6.0
                    one_minus_t = 1.0 - t
                    rounded.append(
                        (
                            one_minus_t * one_minus_t * before[0] + 2 * one_minus_t * t * current[0] + t * t * after[0],
                            one_minus_t * one_minus_t * before[1] + 2 * one_minus_t * t * current[1] + t * t * after[1],
                        )
                    )
            else:
                rounded.append(current)
        rounded.append(points[-1])
        return rounded

    def _draw_wall(self, screen: object, pygame: object, a: Coordinate, b: Coordinate, layout: BoardLayout) -> None:
        ax, ay, aw, ah = layout.cell_rect(a)
        thickness = max(12, layout.cell_size // 7)
        if a[0] == b[0]:
            x = ax + aw if a[1] < b[1] else ax
            rect = (x - thickness // 2, ay, thickness, ah)
        else:
            y = ay + ah if a[0] < b[0] else ay
            rect = (ax, y - thickness // 2, aw, thickness)
        pygame.draw.rect(screen, (14, 14, 14), rect)

    def _draw_panel(self, screen: object, font: object, small_font: object, pygame: object) -> None:
        mouse_pos = pygame.mouse.get_pos()
        for button in self.buttons:
            hovered = button.contains(mouse_pos)
            fill = (18, 18, 18) if hovered else (255, 255, 255)
            text = (255, 255, 255) if hovered else (20, 20, 20)
            pygame.draw.rect(screen, fill, button.rect, border_radius=7)
            pygame.draw.rect(screen, (18, 18, 18), button.rect, 1, border_radius=7)
            surface = small_font.render(button.label, True, text)
            screen.blit(surface, surface.get_rect(center=(button.rect[0] + button.rect[2] // 2, button.rect[1] + button.rect[3] // 2)))

        x = self.layout.origin_x + self.layout.width + self.config.panel_gap
        y = self.panel_metadata_y
        metadata = [
            f"{self.puzzle.rows}x{self.puzzle.cols} {self.puzzle.difficulty}",
            f"seed: {self.puzzle.seed}",
            f"unique: {self.puzzle.unique_solution}",
            f"visited: {len(self.state.visited)}/{self.puzzle.total_cells}",
            f"speed: {self.playback_speed:g}/s",
        ]
        for line in metadata:
            surface = small_font.render(line, True, (42, 42, 38))
            screen.blit(surface, (x, y))
            y += 22

        if self.feedback and time.monotonic() < self.feedback_until:
            surface = small_font.render(self.feedback, True, (20, 20, 20))
            screen.blit(surface, (x, y + 8))


def run_ui(
    puzzle: Puzzle | None = None,
    *,
    difficulty: str = "easy",
    save_path: str | Path = "zip_saved_puzzle.json",
) -> None:
    ZipGameUI(puzzle, difficulty=difficulty, save_path=save_path).run()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Play a LinkedIn Zip-style puzzle.")
    parser.add_argument("--difficulty", default="easy", choices=sorted(DIFFICULTIES))
    parser.add_argument("--puzzle", type=Path)
    parser.add_argument("--save-path", type=Path, default=Path("zip_saved_puzzle.json"))
    args = parser.parse_args(argv)

    puzzle = load_puzzle(args.puzzle) if args.puzzle else None
    run_ui(puzzle, difficulty=args.difficulty, save_path=args.save_path)


if __name__ == "__main__":
    main()
