"""Puzzle generation for LinkedIn Patches-style puzzles.

Generation is solution-first: partition the grid into rectangles, then derive a
clue for each rectangle. Shapes start tight and are relaxed to ``FREE`` per the
difficulty's ``free_shape_ratio``; if uniqueness is required and lost, ``FREE``
clues are tightened back until the board is unique again.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, replace

from .config import DifficultyConfig, get_difficulty_config
from .puzzle import Clue, Puzzle
from .solver import SolverResult, solve
from .utils import Coordinate, Rect, ShapeType, shape_for_dimensions


class GenerationError(RuntimeError):
    pass


@dataclass(frozen=True)
class GenerationResult:
    puzzle: Puzzle
    solution_count: int
    unique_solution: bool | None
    solver_timed_out: bool
    generation_seed: int
    generation_attempts: int
    solver_elapsed_seconds: float


def generate_puzzle(
    difficulty: str | DifficultyConfig = "medium",
    *,
    seed: int | None = None,
) -> Puzzle:
    return generate_puzzle_with_report(difficulty, seed=seed).puzzle


def generate_puzzle_with_report(
    difficulty: str | DifficultyConfig = "medium",
    *,
    seed: int | None = None,
) -> GenerationResult:
    config = get_difficulty_config(difficulty)
    base_seed = seed if seed is not None else config.seed
    if base_seed is None:
        base_seed = random.SystemRandom().randrange(0, 2**63)
    seed_source = random.Random(base_seed)

    last_report: SolverResult | None = None
    for attempt in range(1, config.max_generation_attempts + 1):
        attempt_seed = seed_source.randrange(0, 2**63)
        rng = random.Random(attempt_seed)

        tiling = _generate_tiling(config, rng)
        if tiling is None:
            continue

        # Start fully specified (every clue has its exact shape and size), then
        # drop as many hints as possible while keeping the solution unique. This
        # yields minimal, deducible hints and a proper difficulty gradient.
        clues = _full_clues(tiling, config, rng)
        puzzle = Puzzle(
            rows=config.rows,
            cols=config.cols,
            clues=clues,
            solution=tuple(tiling),
            difficulty=config.name,
            seed=attempt_seed,
            unique_solution=None,
        )
        report = solve(
            puzzle,
            max_solutions=2,
            timeout_seconds=config.solver_timeout_seconds,
            return_solutions=False,
        )
        last_report = report

        if report.unique_solution is True:
            clues, report = _reduce_hints(puzzle, tiling, config, rng)
        elif config.require_unique_solution:
            # Even the fully specified board is ambiguous; try another tiling.
            continue

        puzzle = Puzzle(
            rows=config.rows,
            cols=config.cols,
            clues=clues,
            solution=tuple(tiling),
            difficulty=config.name,
            seed=attempt_seed,
            unique_solution=report.unique_solution,
        )

        if config.require_unique_solution and report.unique_solution is not True:
            continue
        if report.solution_count == 0 and not report.timed_out:
            continue

        return GenerationResult(
            puzzle=puzzle,
            solution_count=report.solution_count,
            unique_solution=report.unique_solution,
            solver_timed_out=report.timed_out,
            generation_seed=attempt_seed,
            generation_attempts=attempt,
            solver_elapsed_seconds=report.elapsed_seconds,
        )

    detail = ""
    if last_report is not None:
        detail = (
            f"; last solver count={last_report.solution_count}, "
            f"unique={last_report.unique_solution}, timed_out={last_report.timed_out}"
        )
    raise GenerationError(
        f"could not generate a valid {config.name!r} puzzle after "
        f"{config.max_generation_attempts} attempts{detail}"
    )


def _generate_tiling(config: DifficultyConfig, rng: random.Random) -> list[Rect] | None:
    """Partition the grid into rectangles via backtracking.

    The top-left-most uncovered cell always becomes the top-left corner of the
    next rectangle, which guarantees full coverage with no overlaps.
    """

    rows, cols = config.rows, config.cols
    deadline = (
        time.monotonic() + config.generation_timeout_seconds
        if config.generation_timeout_seconds > 0
        else None
    )
    covered = [[False] * cols for _ in range(rows)]
    placed: list[Rect] = []

    def first_uncovered() -> Coordinate | None:
        for row in range(rows):
            for col in range(cols):
                if not covered[row][col]:
                    return row, col
        return None

    def options_at(top: int, left: int) -> list[Rect]:
        rects: list[Rect] = []
        max_height = rows - top
        max_width = cols - left
        for height in range(1, max_height + 1):
            # Stop early if the first column of this row-band is blocked.
            if covered[top + height - 1][left]:
                break
            for width in range(1, max_width + 1):
                area = height * width
                if area > config.max_patch_area:
                    break
                if covered[top + height - 1][left + width - 1]:
                    break
                if not _rect_free(covered, top, left, height, width):
                    continue
                if area < config.min_patch_area:
                    continue
                rects.append(Rect(top, left, height, width))
        # Prefer larger rectangles so puzzles end up with fewer, bigger patches.
        # The jitter keeps generation varied instead of always identical.
        bias = config.large_patch_bias
        rects.sort(key=lambda rect: -(rect.area + rng.random() * bias))
        return rects

    def recurse() -> bool:
        if deadline is not None and time.monotonic() >= deadline:
            return False
        cell = first_uncovered()
        if cell is None:
            return True
        top, left = cell
        for rect in options_at(top, left):
            _set_rect(covered, rect, True)
            placed.append(rect)
            if recurse():
                return True
            placed.pop()
            _set_rect(covered, rect, False)
        return False

    if recurse():
        return list(placed)
    return None


def _rect_free(covered: list[list[bool]], top: int, left: int, height: int, width: int) -> bool:
    for row in range(top, top + height):
        band = covered[row]
        for col in range(left, left + width):
            if band[col]:
                return False
    return True


def _set_rect(covered: list[list[bool]], rect: Rect, value: bool) -> None:
    for row in range(rect.top, rect.top + rect.height):
        band = covered[row]
        for col in range(rect.left, rect.left + rect.width):
            band[col] = value


def _full_clues(
    tiling: list[Rect],
    config: DifficultyConfig,
    rng: random.Random,
) -> list[Clue]:
    """Fully specify every clue with its exact shape and size."""

    clues: list[Clue] = []
    for clue_id, rect in enumerate(tiling):
        pos = _choose_clue_position(rect, config, rng)
        shape = shape_for_dimensions(rect.height, rect.width)
        clues.append(Clue(id=clue_id, pos=pos, number=rect.area, shape=shape))
    return clues


def _choose_clue_position(rect: Rect, config: DifficultyConfig, rng: random.Random) -> Coordinate:
    if config.clue_position_policy == "random":
        row = rng.randrange(rect.top, rect.top + rect.height)
        col = rng.randrange(rect.left, rect.left + rect.width)
        return row, col
    if config.clue_position_policy == "center":
        return rect.top + rect.height // 2, rect.left + rect.width // 2
    # "corner" (default): a clue at a rectangle corner lets the UI reproduce the
    # patch by dragging from the clue to the opposite corner.
    corners = [
        (rect.top, rect.left),
        (rect.top, rect.right),
        (rect.bottom, rect.left),
        (rect.bottom, rect.right),
    ]
    return rng.choice(corners)


def _reduce_hints(
    puzzle: Puzzle,
    tiling: list[Rect],
    config: DifficultyConfig,
    rng: random.Random,
) -> tuple[list[Clue], SolverResult]:
    """Drop clue hints while the solution stays unique.

    Starting from a fully specified (unique) board, each hint is a candidate to
    remove: a clue's size (``number`` -> ``None``) or its shape (tight ->
    ``FREE``). ``hint_reduction`` controls how aggressively hints are dropped, so
    harder difficulties end up with fewer hints.
    """

    clues = list(puzzle.clues)
    report = SolverResult(
        solutions=[], solution_count=1, unique_solution=True, timed_out=False, elapsed_seconds=0.0
    )
    if config.hint_reduction <= 0.0:
        return clues, report

    relaxations: list[tuple[int, str]] = []
    for clue in clues:
        relaxations.append((clue.id, "number"))
        relaxations.append((clue.id, "shape"))
    rng.shuffle(relaxations)

    trial_timeout = min(config.solver_timeout_seconds, 1.5)
    for clue_id, kind in relaxations:
        if rng.random() > config.hint_reduction:
            continue  # keep this hint (makes easier difficulties easier)
        trial = [_relax_clue(clue, kind) if clue.id == clue_id else clue for clue in clues]
        if trial == clues:
            continue  # nothing to drop (already relaxed)
        candidate_puzzle = Puzzle(
            rows=puzzle.rows,
            cols=puzzle.cols,
            clues=trial,
            solution=tuple(tiling),
            difficulty=puzzle.difficulty,
            seed=puzzle.seed,
            unique_solution=None,
        )
        trial_report = solve(
            candidate_puzzle,
            max_solutions=2,
            timeout_seconds=trial_timeout,
            return_solutions=False,
        )
        if trial_report.unique_solution is True:
            clues = trial
            report = trial_report

    return clues, report


def _relax_clue(clue: Clue, kind: str) -> Clue:
    if kind == "number":
        return replace(clue, number=None)
    return replace(clue, shape=ShapeType.FREE)
