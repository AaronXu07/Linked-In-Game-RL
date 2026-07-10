"""Real-time Pygame visual training for Patches DQN agents."""

from __future__ import annotations

import argparse
import math
import signal
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType
from typing import Any, Sequence

from patches.simulation.config import DIFFICULTIES, UIConfig
from patches.simulation.ui import BoardLayout, Button, PatchesGameUI

from .agent import AgentConfig, PatchesAgent
from .curriculum import (
    AsyncPuzzleBuffer,
    CurriculumManager,
    DifficultySampler,
    curriculum_from_name,
    evaluation_seeds,
    generate_evaluation_puzzles,
)
from .env import PatchesEnv
from .replay import Transition


class VisualTrainingInterrupted(Exception):
    """Raised when a handled stop signal asks visual training to save and exit."""


@dataclass
class VisualEnvState:
    env: PatchesEnv
    obs: Any
    info: dict[str, Any]
    episode_index: int = 1
    episode_reward: float = 0.0
    episode_length: int = 0
    last_reward: float = 0.0
    last_action_slot: int | None = None
    last_action_label: str = "-"
    last_terminal_reason: str = ""
    terminal_pause_remaining: float = 0.0
    recent_rewards: deque[float] = field(default_factory=lambda: deque(maxlen=50))
    recent_successes: deque[float] = field(default_factory=lambda: deque(maxlen=50))


class VisualTrainingUI(PatchesGameUI):
    """Train a DQN agent while rendering each environment step in Pygame."""

    def __init__(
        self,
        env: PatchesEnv | Sequence[PatchesEnv],
        agent: PatchesAgent,
        *,
        checkpoint_dir: str | Path = "checkpoints/patches_visual",
        checkpoint_every: int = 10_000,
        curriculum: CurriculumManager | None = None,
        curriculum_eval_every: int = 2_000,
        curriculum_eval_episodes: int = 20,
        steps_per_second: float = 4.0,
        max_frame_steps: int = 8,
        total_steps: int | None = None,
        config: UIConfig | None = None,
    ) -> None:
        envs = [env] if isinstance(env, PatchesEnv) else list(env)
        if not envs:
            raise ValueError("at least one visual training environment is required")
        self.envs = envs
        self.env = envs[0]
        self.agent = agent
        self.slots: list[VisualEnvState] = []
        for index, training_env in enumerate(self.envs):
            obs, info = training_env.reset(seed=agent.config.seed + index)
            if training_env.puzzle is None or training_env.state is None:
                raise RuntimeError("environment did not produce an initial puzzle")
            self.slots.append(VisualEnvState(env=training_env, obs=obs, info=info))

        super().__init__(
            self.env.puzzle,
            difficulty=self.env.puzzle.difficulty,
            save_path=Path("patches_visual_training_puzzle.json"),
            config=config,
        )
        self._sync_primary_aliases()

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_every = int(checkpoint_every)
        self.curriculum = curriculum
        self.curriculum_eval_every = int(curriculum_eval_every)
        self.curriculum_eval_episodes = int(curriculum_eval_episodes)
        self.steps_per_second = float(steps_per_second)
        self.max_frame_steps = max(1, int(max_frame_steps))
        self.total_steps_target = total_steps
        self.paused = False
        self.training_accumulator = 0.0
        self.terminal_pause_seconds = 0.7
        self.last_train_metrics: dict[str, float] = {}
        self.recent_rewards: deque[float] = deque(maxlen=50)
        self.recent_successes: deque[float] = deque(maxlen=50)
        self.last_checkpoint_path: Path | None = None
        self._last_autosave_step = self.agent.total_steps
        self._last_curriculum_eval_step = -1
        self._eval_puzzle_cache: dict[str, tuple] = {}
        self.last_curriculum_metrics: dict[str, float] = {}
        self.best_eval_score: float | None = None
        self.board_layouts: list[BoardLayout] = [self.layout]
        self.board_grid_width = self.layout.width
        self.board_grid_height = self.layout.height
        self.feedback = "Training"
        self.feedback_until = time.monotonic() + 2.4

    def run(self) -> None:
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError(
                "Pygame is required for visual training. Install pygame to use it."
            ) from exc

        pygame.init()
        self._fit_layout_to_display(pygame)
        screen = pygame.display.set_mode(self._screen_size())
        pygame.display.set_caption("Patches Visual Training")
        clock = pygame.time.Clock()
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
                elif event.type == pygame.MOUSEMOTION:
                    self._handle_mouse_motion(event.pos)

            self._update_training(dt)
            desired_size = self._screen_size()
            if screen.get_size() != desired_size:
                screen = pygame.display.set_mode(desired_size)
            self._draw(screen, pygame)
            pygame.display.flip()

        pygame.quit()

    def _handle_key(self, key: int, pygame: object) -> bool:
        if key == pygame.K_ESCAPE:
            return False
        if key == pygame.K_SPACE:
            self._toggle_pause()
        elif key == pygame.K_s:
            self._single_step()
        elif key == pygame.K_n:
            self._new_all_training_episodes()
        elif key in (pygame.K_PLUS, pygame.K_EQUALS):
            self._faster()
        elif key in (pygame.K_MINUS, pygame.K_UNDERSCORE):
            self._slower()
        return True

    def _handle_mouse_down(self, pos: tuple[int, int]) -> None:
        for button in self.buttons:
            if button.contains(pos):
                button.action()
                return

    def _handle_mouse_motion(self, pos: tuple[int, int]) -> None:
        self.hover_cell = self.layout.pixel_to_cell(*pos)

    def _sync_primary_aliases(self) -> None:
        slot = self.slots[0]
        self.env = slot.env
        if slot.env.puzzle is None or slot.env.state is None:
            raise RuntimeError("environment has no active state")
        self.puzzle = slot.env.puzzle
        self.state = slot.env.state
        self.pending_rects = dict(slot.env.state.patches)
        self.obs = slot.obs
        self.info = slot.info

    def _slot_puzzle_and_state(self, slot: VisualEnvState):
        if slot.env.puzzle is None or slot.env.state is None:
            raise RuntimeError("environment has no active state")
        return slot.env.puzzle, slot.env.state

    def _screen_size(self) -> tuple[int, int]:
        width = (
            self.config.margin
            + self.board_grid_width
            + self.config.panel_gap
            + self.config.panel_width
            + self.config.margin
        )
        height = self.config.margin + self.board_grid_height + self.config.margin
        max_width = self.max_window_width or width
        max_height = self.max_window_height or height
        return min(width, max_width), min(max(height, 640), max_height)

    def _fit_layout_to_window_limits(self) -> None:
        max_width = self.max_window_width or 1280
        max_height = self.max_window_height or 900
        count = max(1, len(self.slots))
        grid_cols = max(1, math.ceil(math.sqrt(count)))
        grid_rows = math.ceil(count / grid_cols)
        board_gap = max(14, self.config.margin // 2)

        available_width = (
            max_width
            - self.config.margin * 2
            - self.config.panel_gap
            - self.config.panel_width
            - board_gap * (grid_cols - 1)
        )
        available_height = (
            max_height
            - self.config.margin * 2
            - board_gap * (grid_rows - 1)
            - 24 * grid_rows
        )
        max_cols = max(slot.env.puzzle.cols for slot in self.slots if slot.env.puzzle is not None)
        max_rows = max(slot.env.puzzle.rows for slot in self.slots if slot.env.puzzle is not None)
        fitted_cell_size = min(
            self.config.cell_size,
            available_width // max(1, grid_cols * max_cols),
            available_height // max(1, grid_rows * max_rows),
        )
        fitted_cell_size = max(24, int(fitted_cell_size))

        layouts: list[BoardLayout] = []
        max_board_width = max_cols * fitted_cell_size
        max_board_height = max_rows * fitted_cell_size
        for index, slot in enumerate(self.slots):
            puzzle, _ = self._slot_puzzle_and_state(slot)
            grid_row = index // grid_cols
            grid_col = index % grid_cols
            board_width = puzzle.cols * fitted_cell_size
            board_height = puzzle.rows * fitted_cell_size
            origin_x = (
                self.config.margin
                + grid_col * (max_board_width + board_gap)
                + (max_board_width - board_width) // 2
            )
            origin_y = (
                self.config.margin
                + grid_row * (max_board_height + board_gap + 24)
                + 24
                + (max_board_height - board_height) // 2
            )
            layouts.append(
                BoardLayout(
                    puzzle.rows,
                    puzzle.cols,
                    fitted_cell_size,
                    origin_x,
                    origin_y,
                )
            )

        self.board_layouts = layouts
        self.board_grid_width = grid_cols * max_board_width + board_gap * (grid_cols - 1)
        self.board_grid_height = (
            grid_rows * max_board_height
            + board_gap * (grid_rows - 1)
            + 24 * grid_rows
        )
        self.layout = layouts[0]
        self._sync_primary_aliases()

    def _rebuild_buttons(self) -> None:
        x = self.config.margin + self.board_grid_width + self.config.panel_gap
        y = self.config.margin
        width = self.config.panel_width
        height = 38
        gap = 8
        pause_label = "Resume" if self.paused else "Pause"
        specs = [
            (pause_label, self._toggle_pause),
            ("Single Step", self._single_step),
            ("New Episodes", self._new_all_training_episodes),
            ("Save Checkpoint", self._save_checkpoint),
        ]
        self.buttons = []
        for label, action in specs:
            self.buttons.append(Button(label, (x, y, width, height), action))
            y += height + gap

        half_gap = 10
        half_width = (width - half_gap) // 2
        self.buttons.append(Button("Speed -", (x, y, half_width, height), self._slower))
        self.buttons.append(
            Button("Speed +", (x + half_width + half_gap, y, half_width, height), self._faster)
        )
        y += height + gap
        self.panel_metadata_y = y + 16

    def _update_training(self, dt: float) -> None:
        if self._target_reached() or self.paused:
            return

        for index, slot in enumerate(self.slots):
            if slot.terminal_pause_remaining <= 0.0:
                continue
            slot.terminal_pause_remaining -= dt
            if slot.terminal_pause_remaining <= 0.0:
                self._new_training_episode(index)

        self.training_accumulator += dt * self.steps_per_second
        steps_this_frame = 0
        while self.training_accumulator >= 1.0 and steps_this_frame < self.max_frame_steps:
            self.training_accumulator -= 1.0
            for slot_index, slot in enumerate(self.slots):
                if steps_this_frame >= self.max_frame_steps or self._target_reached():
                    break
                if slot.terminal_pause_remaining > 0.0:
                    continue
                self._train_one_step(slot_index)
                steps_this_frame += 1

    def _train_one_step(self, slot_index: int = 0) -> None:
        slot = self.slots[slot_index]
        action_mask = slot.info["action_mask"]
        action = self.agent.predict(slot.obs, deterministic=False, action_mask=action_mask)
        placement = slot.env.action_for_index(action)
        next_obs, reward, terminated, truncated, next_info = slot.env.step(action)
        next_action_mask = next_info["action_mask"]

        self.agent.replay.add(
            Transition(
                obs=slot.obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                terminated=terminated,
                truncated=truncated,
                action_mask=action_mask,
                next_action_mask=next_action_mask,
            )
        )
        self.agent.total_steps += 1
        slot.episode_reward += reward
        slot.episode_length += 1
        slot.last_reward = float(reward)
        slot.last_action_slot = action
        slot.last_action_label = "-" if placement is None else placement.label()
        self.last_train_metrics = {
            "epsilon": self.agent.epsilon(),
            "replay_size": float(len(self.agent.replay)),
            "parallel_envs": float(len(self.slots)),
        }

        if (
            len(self.agent.replay) >= self.agent.config.warmup_steps
            and len(self.agent.replay) >= self.agent.config.batch_size
            and self.agent.total_steps % self.agent.config.train_every == 0
        ):
            batch = self.agent.replay.sample(self.agent.config.batch_size)
            self.last_train_metrics.update(self.agent.train_step(batch))

        self._maybe_evaluate_curriculum()
        self._maybe_autosave()

        slot.obs = next_obs
        slot.info = next_info
        self._sync_from_env(slot_index)

        if terminated or truncated:
            success = bool(next_info["success"])
            self.recent_rewards.append(float(slot.episode_reward))
            self.recent_successes.append(1.0 if success else 0.0)
            slot.recent_rewards.append(float(slot.episode_reward))
            slot.recent_successes.append(1.0 if success else 0.0)
            slot.last_terminal_reason = _terminal_reason(next_info, truncated)
            self._show_feedback(
                f"Env {slot_index + 1}: "
                f"{'solved' if success else 'ended'} ({slot.last_terminal_reason})"
            )
            slot.terminal_pause_remaining = self.terminal_pause_seconds
        self._sync_primary_aliases()

    def _new_training_episode(self, slot_index: int = 0) -> None:
        slot = self.slots[slot_index]
        slot.obs, slot.info = slot.env.reset()
        slot.episode_index += 1
        slot.episode_reward = 0.0
        slot.episode_length = 0
        slot.last_reward = 0.0
        slot.last_action_slot = None
        slot.last_action_label = "-"
        slot.last_terminal_reason = ""
        slot.terminal_pause_remaining = 0.0
        self.training_accumulator = 0.0
        self._sync_from_env(slot_index)
        self._show_feedback(f"Env {slot_index + 1}: new episode")

    def _sync_from_env(self, slot_index: int = 0) -> None:
        slot = self.slots[slot_index]
        puzzle, _ = self._slot_puzzle_and_state(slot)
        layout_changed = (
            slot_index >= len(self.board_layouts)
            or self.board_layouts[slot_index].rows != puzzle.rows
            or self.board_layouts[slot_index].cols != puzzle.cols
        )
        if layout_changed:
            self._fit_layout_to_window_limits()
            self._rebuild_buttons()
        self._sync_primary_aliases()

    def _toggle_pause(self) -> None:
        self.paused = not self.paused
        self._rebuild_buttons()
        self._show_feedback("Paused" if self.paused else "Training")

    def _single_step(self) -> None:
        for index, slot in enumerate(self.slots):
            if slot.terminal_pause_remaining > 0.0:
                self._new_training_episode(index)
            else:
                self._train_one_step(index)

    def _new_all_training_episodes(self) -> None:
        for index in range(len(self.slots)):
            self._new_training_episode(index)

    def _slower(self) -> None:
        self.steps_per_second = max(0.5, self.steps_per_second / 1.5)
        self._show_feedback(f"Training speed {self.steps_per_second:g}/s")

    def _faster(self) -> None:
        self.steps_per_second = min(120.0, self.steps_per_second * 1.5)
        self._show_feedback(f"Training speed {self.steps_per_second:g}/s")

    def _save_checkpoint(self) -> None:
        path = self.save_progress()
        self._show_feedback(f"Saved {path}")

    def save_progress(self, *, name: str = "visual_latest.pt") -> Path:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / name
        self._store_curriculum_metadata()
        self.agent.save_checkpoint(path)
        self.last_checkpoint_path = path
        self._last_autosave_step = self.agent.total_steps
        return path

    def _maybe_autosave(self) -> None:
        if self.checkpoint_every <= 0:
            return
        if self.agent.total_steps == self._last_autosave_step:
            return
        if self.agent.total_steps % self.checkpoint_every != 0:
            return
        self.save_progress()

    def _maybe_evaluate_curriculum(self) -> None:
        if self.curriculum is None:
            return
        self._store_curriculum_metadata()
        if self.curriculum_eval_every <= 0 or self.curriculum_eval_episodes <= 0:
            return
        if self.agent.total_steps == self._last_curriculum_eval_step:
            return
        if self.agent.total_steps % self.curriculum_eval_every != 0:
            return
        stage_before = self.curriculum.current_stage
        difficulty = stage_before.target_difficulty
        puzzles = self._evaluation_puzzles(difficulty)
        metrics = self.agent.evaluate(
            puzzles,
            self.curriculum_eval_episodes,
            deterministic=True,
        )
        advanced = self.curriculum.update_from_evaluation(metrics)
        self._last_curriculum_eval_step = self.agent.total_steps
        self.last_curriculum_metrics = {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        }
        self._store_curriculum_metadata()
        score = float(metrics.get("success_rate", 0.0))
        if self.best_eval_score is None or score > self.best_eval_score:
            self.best_eval_score = score
            self.agent.checkpoint_metadata["best_checkpoint"] = {
                "metric": "success_rate",
                "value": score,
                "step": self.agent.total_steps,
            }
            self.save_progress(name="visual_best.pt")
        if advanced:
            self._show_feedback(f"Curriculum: {self.curriculum.current_stage.name}")

    def _evaluation_puzzles(self, difficulty: str) -> tuple:
        if difficulty not in self._eval_puzzle_cache:
            seeds = evaluation_seeds(difficulty, self.curriculum_eval_episodes)
            self._eval_puzzle_cache[difficulty] = tuple(
                generate_evaluation_puzzles(difficulty, seeds=seeds)
            )
        return self._eval_puzzle_cache[difficulty]

    def _store_curriculum_metadata(self) -> None:
        if self.curriculum is not None:
            self.agent.checkpoint_metadata["curriculum"] = self.curriculum.state_dict()

    def _target_reached(self) -> bool:
        return (
            self.total_steps_target is not None
            and self.agent.total_steps >= self.total_steps_target
        )

    def _draw(self, screen: object, pygame: object) -> None:
        from patches.simulation.ui import BACKGROUND

        screen.fill(BACKGROUND)
        self._draw_training_boards(screen, pygame)
        self._draw_panel(screen, pygame)

    def _draw_training_boards(self, screen: object, pygame: object) -> None:
        original = (self.puzzle, self.state, self.pending_rects, self.layout)
        small_font = self._get_font(pygame, 14)
        for index, (slot, layout) in enumerate(zip(self.slots, self.board_layouts)):
            puzzle, state = self._slot_puzzle_and_state(slot)
            self.puzzle = puzzle
            self.state = state
            self.pending_rects = dict(state.patches)
            self.layout = layout

            label = f"Env {index + 1}  {puzzle.rows}x{puzzle.cols}  {puzzle.difficulty}"
            if slot.last_terminal_reason:
                label += f"  {slot.last_terminal_reason}"
            surface = small_font.render(label, True, (30, 30, 28))
            screen.blit(surface, (layout.origin_x, max(4, layout.origin_y - 22)))
            self._draw_board(screen, pygame)

        self.puzzle, self.state, self.pending_rects, self.layout = original
        self._sync_primary_aliases()

    def _draw_panel(self, screen: object, pygame: object) -> None:
        button_font = self._get_font(pygame, 16, True)
        label_font = self._get_font(pygame, 15)
        head_font = self._get_font(pygame, 20, True)
        mouse_pos = pygame.mouse.get_pos()

        x = self.config.margin + self.board_grid_width + self.config.panel_gap
        title = head_font.render("Patches Training", True, (32, 32, 30))
        screen.blit(title, (x, self.config.margin - 26 if self.config.margin > 30 else 4))

        for button in self.buttons:
            hovered = button.contains(mouse_pos)
            fill = (24, 24, 22) if hovered else (255, 255, 255)
            text_color = (255, 255, 255) if hovered else (32, 32, 30)
            pygame.draw.rect(screen, fill, button.rect, border_radius=9)
            pygame.draw.rect(screen, (36, 36, 34), button.rect, 1, border_radius=9)
            surface = button_font.render(button.text(), True, text_color)
            screen.blit(
                surface,
                surface.get_rect(
                    center=(
                        button.rect[0] + button.rect[2] // 2,
                        button.rect[1] + button.rect[3] // 2,
                    )
                ),
            )

        y = self.panel_metadata_y
        primary_puzzle, primary_state = self._slot_puzzle_and_state(self.slots[0])
        lines = [
            f"envs: {len(self.slots)}",
            f"primary: {primary_puzzle.rows}x{primary_puzzle.cols} {primary_puzzle.difficulty}",
            f"seed: {primary_puzzle.seed}",
            f"patches: {len(primary_state.patches)}/{len(primary_puzzle.clues)}",
            f"covered: {primary_state.covered_count}/{primary_puzzle.total_cells}",
            "",
            *self._training_panel_lines(),
        ]
        for line in lines:
            if not line:
                y += 10
                continue
            surface = label_font.render(line, True, (40, 40, 36))
            screen.blit(surface, (x, y))
            y += 22

        if self.feedback and time.monotonic() < self.feedback_until:
            surface = label_font.render(self.feedback, True, (188, 96, 40))
            screen.blit(surface, (x, y + 8))

    def _training_panel_lines(self) -> list[str]:
        primary = self.slots[0]
        mean_reward = (
            sum(self.recent_rewards) / len(self.recent_rewards)
            if self.recent_rewards
            else 0.0
        )
        success_rate = (
            sum(self.recent_successes) / len(self.recent_successes)
            if self.recent_successes
            else 0.0
        )
        metrics = self.last_train_metrics
        lines = [
            "Training",
            f"mode: {'paused' if self.paused else 'running'}",
            f"steps: {self.agent.total_steps}",
            f"parallel envs: {len(self.slots)}",
            f"primary episode: {primary.episode_index}",
            f"primary len: {primary.episode_length}",
            f"primary reward: {primary.episode_reward:.2f}",
            f"primary action: {primary.last_action_label}",
            f"primary step reward: {primary.last_reward:.2f}",
            f"epsilon: {self.agent.epsilon():.3f}",
            f"replay: {len(self.agent.replay)}",
            f"updates: {self.agent.optimization_steps}",
            f"round speed: {self.steps_per_second:g}/s",
            f"sample speed: {self.steps_per_second * len(self.slots):g}/s",
            f"recent reward: {mean_reward:.2f}",
            f"recent success: {success_rate:.0%}",
        ]
        if self.curriculum is not None:
            stage = self.curriculum.current_stage
            lines.extend(
                [
                    f"curriculum: {stage.name}",
                    f"stage: {self.curriculum.stage_index + 1}/{len(self.curriculum.stages)}",
                ]
            )
            if self.curriculum.last_success_rate is not None:
                lines.append(f"gate eval: {self.curriculum.last_success_rate:.0%}")
        if "loss" in metrics:
            lines.extend(
                [
                    f"loss: {metrics['loss']:.4f}",
                    f"q mean: {metrics['q_mean']:.3f}",
                    f"q max: {metrics['q_max']:.3f}",
                ]
            )
        if primary.last_terminal_reason:
            lines.append(f"primary end: {primary.last_terminal_reason}")
        if self._target_reached():
            lines.append("target reached")
        return lines


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    curriculum = curriculum_from_name(
        args.curriculum,
        target_difficulty=args.difficulty,
        seed=args.seed,
    )
    puzzle_sampler = _build_puzzle_sampler(args, curriculum)
    env_count = _active_env_count(args)
    envs = [
        PatchesEnv(
            difficulty=args.difficulty,
            max_rows=args.max_rows,
            max_cols=args.max_cols,
            max_actions=args.max_actions,
            invalid_action_mode=args.invalid_action_mode,
            episode_mode=args.episode_mode,
            use_solver_dead_end_check=args.solver_dead_end_check,
            puzzle_sampler=puzzle_sampler,
        )
        for _ in range(env_count)
    ]
    config = AgentConfig(
        algorithm=args.algorithm,
        learning_rate=args.learning_rate,
        replay_size=args.replay_size,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        train_every=args.train_every,
        target_update_every=args.target_update_every,
        tau=args.tau,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        double_dqn=args.double_dqn,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        device=args.device,
    )
    agent = PatchesAgent(envs[0], config, checkpoint_path=args.checkpoint_path)
    if curriculum is not None:
        saved_curriculum = agent.checkpoint_metadata.get("curriculum")
        if isinstance(saved_curriculum, dict):
            curriculum.load_state_dict(saved_curriculum)
    ui = VisualTrainingUI(
        envs,
        agent,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        curriculum=curriculum,
        curriculum_eval_every=args.curriculum_eval_every,
        curriculum_eval_episodes=args.curriculum_eval_episodes,
        steps_per_second=args.steps_per_second,
        max_frame_steps=args.max_frame_steps,
        total_steps=args.total_steps,
    )
    try:
        with _interrupt_handler():
            ui.run()
    except (KeyboardInterrupt, VisualTrainingInterrupted):
        ui.save_progress(name="visual_interrupted.pt")
    finally:
        ui.save_progress()
        for env in envs:
            env.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--difficulty", choices=sorted(DIFFICULTIES), default="super_easy")
    parser.add_argument("--curriculum", default=None)
    parser.add_argument("--parallel-puzzles", type=int, default=0)
    parser.add_argument("--parallel-envs", type=int, default=0)
    parser.add_argument("--puzzle-buffer-size", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-cols", type=int, default=None)
    parser.add_argument("--max-actions", type=int, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--algorithm", choices=("candidate_dqn", "flat_candidate_dqn"), default="candidate_dqn")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--replay-size", type=int, default=100_000)
    parser.add_argument("--warmup-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--train-every", type=int, default=1)
    parser.add_argument("--target-update-every", type=int, default=1_000)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=50_000)
    parser.add_argument("--double-dqn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--checkpoint-dir", default="checkpoints/patches_visual")
    parser.add_argument("--checkpoint-every", type=int, default=10_000)
    parser.add_argument("--curriculum-eval-every", type=int, default=2_000)
    parser.add_argument("--curriculum-eval-episodes", type=int, default=20)
    parser.add_argument("--steps-per-second", type=float, default=4.0)
    parser.add_argument("--max-frame-steps", type=int, default=8)
    parser.add_argument(
        "--invalid-action-mode",
        choices=("terminate", "penalize", "mask_required"),
        default="terminate",
    )
    parser.add_argument(
        "--episode-mode",
        choices=("commit_only", "revision"),
        default="commit_only",
    )
    parser.add_argument("--solver-dead-end-check", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser.parse_args(argv)


def _active_env_count(args: argparse.Namespace) -> int:
    if args.parallel_envs > 0:
        return args.parallel_envs
    if args.parallel_puzzles > 1:
        return args.parallel_puzzles
    return 1


def _build_puzzle_sampler(
    args: argparse.Namespace,
    curriculum: CurriculumManager | None,
):
    sampler = curriculum
    if sampler is None and args.parallel_puzzles > 0:
        sampler = DifficultySampler((args.difficulty,), seed=args.seed)
    if sampler is None:
        return None
    if args.parallel_puzzles <= 0:
        return sampler
    return AsyncPuzzleBuffer(
        sampler,
        workers=args.parallel_puzzles,
        buffer_size=args.puzzle_buffer_size,
        seed=args.seed,
    )


@contextmanager
def _interrupt_handler() -> Iterator[None]:
    previous_handlers = {}

    def handle_signal(signum: int, frame: FrameType | None) -> None:
        del frame
        raise VisualTrainingInterrupted(f"received signal {signum}")

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, handle_signal)
    try:
        yield
    finally:
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)


def _terminal_reason(info: dict[str, Any], truncated: bool) -> str:
    if info["success"]:
        return "success"
    if truncated:
        return "truncated"
    if info["invalid_action"]:
        return str(info["invalid_reason"] or "invalid_action")
    if info["dead_end"]:
        return "dead_end"
    return "terminated"


if __name__ == "__main__":
    main()
