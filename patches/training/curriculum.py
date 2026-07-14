"""Difficulty schedules and puzzle sampling helpers for Patches training."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from random import Random
from typing import Iterable

from patches.simulation import Puzzle, generate_puzzle
from patches.simulation.config import DIFFICULTIES, get_difficulty_config

DIFFICULTY_ORDER = ("super_easy", "easy", "medium", "hard", "expert")

DEFAULT_EVALUATION_SEEDS: dict[str, range] = {
    "super_easy": range(11000, 11100),
    "easy": range(12000, 12100),
    "medium": range(13000, 13100),
    "hard": range(14000, 14100),
    "expert": range(15000, 15100),
}


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    difficulties: tuple[str, ...]
    min_success_rate: float | None = None
    eval_difficulty: str | None = None

    @property
    def target_difficulty(self) -> str:
        return self.eval_difficulty or self.difficulties[-1]


def build_default_curriculum(target_difficulty: str = "medium") -> tuple[CurriculumStage, ...]:
    """Build the default easier-to-harder Patches curriculum.

    The final stage has no success gate, so training stays there after reaching
    the requested target difficulty.
    """

    target = _normalize_difficulty(target_difficulty)
    target_index = DIFFICULTY_ORDER.index(target)
    if target_index == 0:
        return (CurriculumStage(target, (target,), eval_difficulty=target),)

    stages: list[CurriculumStage] = [
        CurriculumStage(
            "super_easy",
            ("super_easy",),
            min_success_rate=0.98,
            eval_difficulty="super_easy",
        )
    ]
    for index in range(1, target_index + 1):
        previous = DIFFICULTY_ORDER[index - 1]
        current = DIFFICULTY_ORDER[index]
        stages.append(
            CurriculumStage(
                f"{previous}_{current}_mix",
                (previous, current),
                min_success_rate=0.90 if index <= 1 else 0.80,
                eval_difficulty=current,
            )
        )
        stages.append(
            CurriculumStage(
                current,
                (current,),
                min_success_rate=None if index == target_index else 0.90,
                eval_difficulty=current,
            )
        )
    return tuple(stages)


class DifficultySampler:
    def __init__(
        self,
        difficulties: Iterable[str],
        *,
        seed: int | None = None,
    ) -> None:
        self.difficulties = tuple(difficulties)
        if not self.difficulties:
            raise ValueError("at least one difficulty is required")
        unknown = sorted(set(self.difficulties) - set(DIFFICULTIES))
        if unknown:
            raise ValueError(f"unknown difficulties: {', '.join(unknown)}")
        self.rng = Random(seed)
        self.version = 0

    def sample_difficulty(self) -> str:
        return self.rng.choice(self.difficulties)

    def sample_puzzle(self) -> Puzzle:
        difficulty = self.sample_difficulty()
        seed = self.rng.randrange(0, 2**63)
        return generate_puzzle(difficulty, seed=seed)


class CurriculumManager:
    """Stateful curriculum controller used by training and visual training."""

    # Number of consecutive evaluation passes required before promotion.
    DEFAULT_CONSECUTIVE_PASSES = 3

    def __init__(
        self,
        stages: Iterable[CurriculumStage],
        *,
        seed: int | None = None,
        stage_index: int = 0,
        consecutive_passes_required: int = DEFAULT_CONSECUTIVE_PASSES,
    ) -> None:
        self.stages = tuple(stages)
        if not self.stages:
            raise ValueError("curriculum needs at least one stage")
        for stage in self.stages:
            _validate_stage(stage)
        if stage_index < 0 or stage_index >= len(self.stages):
            raise ValueError("stage_index is out of range")
        self.stage_index = int(stage_index)
        self.rng = Random(seed)
        self.version = 0
        self.last_success_rate: float | None = None
        self.consecutive_passes_required = max(1, int(consecutive_passes_required))
        self._consecutive_passes = 0

    @property
    def current_stage(self) -> CurriculumStage:
        return self.stages[self.stage_index]

    @property
    def done(self) -> bool:
        return self.stage_index >= len(self.stages) - 1

    def sample_difficulty(self) -> str:
        return self.rng.choice(self.current_stage.difficulties)

    def sample_puzzle(self) -> Puzzle:
        difficulty = self.sample_difficulty()
        seed = self.rng.randrange(0, 2**63)
        return generate_puzzle(difficulty, seed=seed)

    def update_from_evaluation(self, metrics: dict[str, float]) -> bool:
        stage = self.current_stage
        self.last_success_rate = float(metrics.get("success_rate", 0.0))
        if self.done or stage.min_success_rate is None:
            self._consecutive_passes = 0
            return False
        if self.last_success_rate < stage.min_success_rate:
            # Failed the gate — reset the streak counter.
            self._consecutive_passes = 0
            return False
        # Passed the gate this evaluation — count toward consecutive requirement.
        self._consecutive_passes += 1
        if self._consecutive_passes < self.consecutive_passes_required:
            return False
        # Enough consecutive passes — promote to next stage.
        self._consecutive_passes = 0
        self.stage_index += 1
        self.version += 1
        return True

    def state_dict(self) -> dict[str, object]:
        return {
            "stage_index": self.stage_index,
            "stage_name": self.current_stage.name,
            "version": self.version,
            "last_success_rate": self.last_success_rate,
            "consecutive_passes": self._consecutive_passes,
            "consecutive_passes_required": self.consecutive_passes_required,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        stage_index = int(state.get("stage_index", self.stage_index))
        if stage_index < 0 or stage_index >= len(self.stages):
            return
        if stage_index != self.stage_index:
            self.stage_index = stage_index
            self.version += 1
            self._consecutive_passes = 0
        else:
            self._consecutive_passes = int(state.get("consecutive_passes", 0))
        self.last_success_rate = (
            None
            if state.get("last_success_rate") is None
            else float(state["last_success_rate"])
        )
        if "consecutive_passes_required" in state:
            self.consecutive_passes_required = max(1, int(state["consecutive_passes_required"]))


class CurriculumStateCallback:
    """Attach curriculum state to training metrics and checkpoints."""

    def __init__(self, manager: CurriculumManager) -> None:
        self.manager = manager

    def __call__(self, agent, event: str, payload: dict[str, object]) -> None:
        del event
        stage = self.manager.current_stage
        config = get_difficulty_config(stage.target_difficulty)
        payload["curriculum_stage_index"] = float(self.manager.stage_index)
        payload["curriculum_grid_cells"] = float(config.rows * config.cols)
        payload["curriculum_success_gate"] = (
            -1.0 if stage.min_success_rate is None else float(stage.min_success_rate)
        )
        agent.checkpoint_metadata["curriculum"] = self.manager.state_dict()


class CurriculumEvaluationCallback:
    """Evaluate the current curriculum gate and advance stages when ready."""

    def __init__(
        self,
        manager: CurriculumManager,
        *,
        every_steps: int,
        episodes: int,
        callbacks: Iterable[Callable] | None = None,
    ) -> None:
        self.manager = manager
        self.every_steps = int(every_steps)
        self.episodes = int(episodes)
        self.callbacks = tuple(callbacks or ())
        self._last_eval_step = -1
        self._puzzle_cache: dict[str, tuple[Puzzle, ...]] = {}

    def __call__(self, agent, event: str, payload: dict[str, object]) -> None:
        del payload
        if event != "step" or self.every_steps <= 0 or self.episodes <= 0:
            return
        if agent.total_steps == self._last_eval_step:
            return
        if agent.total_steps % self.every_steps != 0:
            return

        stage_before = self.manager.current_stage
        difficulty = stage_before.target_difficulty
        puzzles = self._evaluation_puzzles(difficulty)
        metrics = agent.evaluate(puzzles, self.episodes, deterministic=True)
        advanced = self.manager.update_from_evaluation(metrics)
        self._last_eval_step = agent.total_steps
        metrics.update(
            {
                "curriculum_stage_index": float(self.manager.stage_index),
                "curriculum_advanced": float(advanced),
                "curriculum_eval_difficulty": difficulty,
                "curriculum_stage": stage_before.name,
                "curriculum_consecutive_passes": float(self.manager._consecutive_passes),
                "curriculum_consecutive_required": float(self.manager.consecutive_passes_required),
            }
        )
        agent.checkpoint_metadata["curriculum"] = self.manager.state_dict()
        for callback in self.callbacks:
            callback(agent, "eval", metrics)

    def _evaluation_puzzles(self, difficulty: str) -> tuple[Puzzle, ...]:
        if difficulty not in self._puzzle_cache:
            seeds = evaluation_seeds(difficulty, self.episodes)
            self._puzzle_cache[difficulty] = tuple(
                generate_evaluation_puzzles(difficulty, seeds=seeds)
            )
        return self._puzzle_cache[difficulty]


class AsyncPuzzleBuffer:
    """Prefetch generated puzzles in worker processes."""

    def __init__(
        self,
        sampler,
        *,
        workers: int,
        buffer_size: int | None = None,
        seed: int | None = None,
    ) -> None:
        if workers <= 0:
            raise ValueError("workers must be positive")
        self.sampler = sampler
        self.workers = int(workers)
        self.buffer_size = max(self.workers, int(buffer_size or self.workers * 2))
        self.rng = Random(seed)
        self.version = _sampler_version(sampler)
        self._executor = ProcessPoolExecutor(max_workers=self.workers)
        self._pending: deque[tuple[int, Future[Puzzle]]] = deque()
        self._closed = False
        self._refill()

    def sample_puzzle(self) -> Puzzle:
        if self._closed:
            raise RuntimeError("cannot sample from a closed AsyncPuzzleBuffer")
        self._sync_version()
        while True:
            self._refill()
            for version, future in list(self._pending):
                if not future.done():
                    continue
                self._pending.remove((version, future))
                self._refill()
                try:
                    puzzle = future.result()
                except Exception:
                    continue
                if version == self.version:
                    return puzzle

            if self._pending:
                futures = [
                    future
                    for version, future in self._pending
                    if version == self.version
                ]
                if futures:
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)
                    future = next(iter(done))
                    self._remove_future(future)
                    self._refill()
                    try:
                        return future.result()
                    except Exception:
                        continue

            difficulty = self.sampler.sample_difficulty()
            seed = self.rng.randrange(0, 2**63)
            return generate_puzzle(difficulty, seed=seed)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for _, future in self._pending:
            future.cancel()
        self._pending.clear()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _sync_version(self) -> None:
        sampler_version = _sampler_version(self.sampler)
        if sampler_version == self.version:
            return
        self.version = sampler_version
        for _, future in self._pending:
            future.cancel()
        self._pending.clear()

    def _refill(self) -> None:
        while not self._closed and len(self._pending) < self.buffer_size:
            difficulty = self.sampler.sample_difficulty()
            seed = self.rng.randrange(0, 2**63)
            future = self._executor.submit(_generate_puzzle_job, difficulty, seed)
            self._pending.append((self.version, future))

    def _remove_future(self, future: Future[Puzzle]) -> None:
        for item in list(self._pending):
            if item[1] is future:
                self._pending.remove(item)
                return


def curriculum_from_name(
    name: str | None,
    *,
    target_difficulty: str,
    seed: int | None = None,
    consecutive_passes_required: int = CurriculumManager.DEFAULT_CONSECUTIVE_PASSES,
) -> CurriculumManager | None:
    if name is None or name.strip().lower() in {"", "none", "off", "false"}:
        return None
    normalized = name.strip().lower().replace("-", "_")
    if normalized not in {"default", "patches", "zip", "auto"}:
        raise ValueError("curriculum must be one of: none, default, patches, zip, auto")
    return CurriculumManager(
        build_default_curriculum(target_difficulty),
        seed=seed,
        consecutive_passes_required=consecutive_passes_required,
    )


def evaluation_seeds(difficulty: str, count: int | None = None) -> tuple[int, ...]:
    seeds = tuple(DEFAULT_EVALUATION_SEEDS[difficulty])
    if count is not None:
        return seeds[:count]
    return seeds


def generate_evaluation_puzzles(
    difficulty: str,
    *,
    seeds: Iterable[int] | None = None,
) -> list[Puzzle]:
    selected_seeds = tuple(seeds) if seeds is not None else evaluation_seeds(difficulty)
    return [generate_puzzle(difficulty, seed=seed) for seed in selected_seeds]


def _validate_stage(stage: CurriculumStage) -> None:
    if not stage.difficulties:
        raise ValueError("curriculum stages need at least one difficulty")
    for difficulty in (*stage.difficulties, stage.target_difficulty):
        _normalize_difficulty(difficulty)
    if stage.min_success_rate is not None and not (0.0 <= stage.min_success_rate <= 1.0):
        raise ValueError("min_success_rate must be between 0 and 1")


def _normalize_difficulty(difficulty: str) -> str:
    if difficulty not in DIFFICULTY_ORDER or difficulty not in DIFFICULTIES:
        choices = ", ".join(DIFFICULTY_ORDER)
        raise ValueError(f"unknown difficulty {difficulty!r}; choose one of: {choices}")
    return difficulty


def _sampler_version(sampler) -> int:
    return int(getattr(sampler, "version", 0))


DEFAULT_CURRICULUM = build_default_curriculum("medium")


def _generate_puzzle_job(difficulty: str, seed: int) -> Puzzle:
    return generate_puzzle(difficulty, seed=seed)
