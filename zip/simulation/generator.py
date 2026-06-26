"""Puzzle generation for LinkedIn Zip-style puzzles."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, replace

from .config import DifficultyConfig, get_difficulty_config
from .puzzle import Puzzle
from .solver import SolverResult, solve
from .utils import Coordinate, Wall, canonical_wall, iter_grid, neighbors, solution_edges


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


@dataclass
class _PathFrame:
    current: Coordinate
    candidates: list[Coordinate]
    candidate_index: int = 0


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
        try:
            solution = generate_hamiltonian_path(
                config.rows,
                config.cols,
                rng=rng,
                timeout_seconds=config.generation_timeout_seconds,
                start_position_policy=config.start_position_policy,
            )
        except GenerationError:
            continue

        waypoints = _choose_waypoints(solution, config, rng)
        walls = _choose_walls(solution, waypoints, config, rng)
        puzzle = Puzzle(
            rows=config.rows,
            cols=config.cols,
            waypoints=tuple(waypoints),
            walls=frozenset(walls),
            solution=tuple(solution),
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
        puzzle = replace(puzzle, unique_solution=report.unique_solution)

        if config.require_unique_solution and report.unique_solution is not True:
            continue
        if report.solution_count == 0 and not report.timed_out:
            continue
        if report.timed_out and config.require_unique_solution:
            continue
        if not config.allow_multi_solution and report.unique_solution is False:
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


def generate_hamiltonian_path(
    rows: int,
    cols: int,
    *,
    rng: random.Random | None = None,
    timeout_seconds: float = 2.0,
    start_position_policy: str = "random",
) -> list[Coordinate]:
    rng = rng or random.Random()
    deadline = time.monotonic() + timeout_seconds if timeout_seconds > 0 else None
    start = _choose_start(rows, cols, rng, start_position_policy)
    path = [start]
    visited = {start}
    stack = [
        _PathFrame(
            current=start,
            candidates=_ordered_path_candidates(rows, cols, path, visited, rng),
        )
    ]
    total_cells = rows * cols

    while stack:
        if deadline is not None and time.monotonic() >= deadline:
            raise GenerationError("Hamiltonian path generation timed out")
        if len(path) == total_cells:
            return _randomize_hamiltonian_path(list(path), rows, cols, rng)

        frame = stack[-1]
        if frame.candidate_index >= len(frame.candidates):
            stack.pop()
            removed = path.pop()
            visited.remove(removed)
            continue

        candidate = frame.candidates[frame.candidate_index]
        frame.candidate_index += 1
        if candidate in visited:
            continue
        visited.add(candidate)
        if not _unvisited_cells_connected(rows, cols, candidate, visited):
            visited.remove(candidate)
            continue

        path.append(candidate)
        stack.append(
            _PathFrame(
                current=candidate,
                candidates=_ordered_path_candidates(rows, cols, path, visited, rng),
            )
        )

    raise GenerationError("Hamiltonian path generation exhausted its search")


def _choose_start(
    rows: int,
    cols: int,
    rng: random.Random,
    policy: str,
) -> Coordinate:
    if policy == "corner":
        return rng.choice([(0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)])
    if policy == "edge":
        edge_cells = [
            cell
            for cell in iter_grid(rows, cols)
            if cell[0] in (0, rows - 1) or cell[1] in (0, cols - 1)
        ]
        return rng.choice(edge_cells)
    return rng.randrange(rows), rng.randrange(cols)


def _ordered_path_candidates(
    rows: int,
    cols: int,
    path: list[Coordinate],
    visited: set[Coordinate],
    rng: random.Random,
) -> list[Coordinate]:
    current = path[-1]
    candidates = [cell for cell in neighbors(rows, cols, current) if cell not in visited]
    rng.shuffle(candidates)
    candidates.sort(key=lambda cell: _unvisited_degree(rows, cols, cell, visited | {cell}))
    return candidates


def _randomize_hamiltonian_path(
    path: list[Coordinate],
    rows: int,
    cols: int,
    rng: random.Random,
) -> list[Coordinate]:
    if rows * cols < 25:
        return path

    best_path = list(path)
    best_score = _path_sweep_score(best_path)
    move_count = rows * cols * 20
    for _ in range(move_count):
        path = _backbite_path(path, rows, cols, rng)
        score = _path_sweep_score(path)
        if score < best_score:
            best_path = list(path)
            best_score = score

    return best_path


def _backbite_path(
    path: list[Coordinate],
    rows: int,
    cols: int,
    rng: random.Random,
) -> list[Coordinate]:
    positions = {cell: index for index, cell in enumerate(path)}
    endpoint = path[-1]
    options = [
        positions[cell]
        for cell in neighbors(rows, cols, endpoint)
        if cell in positions and positions[cell] < len(path) - 2
    ]
    if not options:
        return path
    index = rng.choice(options)
    return path[: index + 1] + list(reversed(path[index + 1 :]))


def _path_sweep_score(path: list[Coordinate]) -> tuple[int, int]:
    longest_run = 0
    turn_count = 0
    current_run = 0
    previous_delta: Coordinate | None = None

    for a, b in zip(path, path[1:]):
        delta = b[0] - a[0], b[1] - a[1]
        if delta == previous_delta:
            current_run += 1
        else:
            if previous_delta is not None:
                turn_count += 1
            current_run = 1
            previous_delta = delta
        longest_run = max(longest_run, current_run)

    return longest_run, -turn_count


def _unvisited_degree(
    rows: int,
    cols: int,
    cell: Coordinate,
    visited: set[Coordinate],
) -> int:
    return sum(1 for candidate in neighbors(rows, cols, cell) if candidate not in visited)


def _unvisited_cells_connected(
    rows: int,
    cols: int,
    head: Coordinate,
    visited: set[Coordinate],
) -> bool:
    remaining = set(iter_grid(rows, cols)) - visited
    if not remaining:
        return True

    entry_cells = [cell for cell in neighbors(rows, cols, head) if cell in remaining]
    if not entry_cells:
        return False

    start = entry_cells[0]
    frontier = [start]
    seen = {start}
    while frontier:
        cell = frontier.pop()
        for candidate in neighbors(rows, cols, cell):
            if candidate in remaining and candidate not in seen:
                seen.add(candidate)
                frontier.append(candidate)
    return seen == remaining


def _choose_waypoints(
    solution: list[Coordinate],
    config: DifficultyConfig,
    rng: random.Random,
) -> list[Coordinate]:
    total_cells = len(solution)
    if total_cells == 1:
        return [solution[0]]

    if config.total_waypoints is not None:
        minimum, maximum = config.total_waypoints
        total_waypoints = min(total_cells, rng.randint(minimum, maximum))
        total_waypoints = max(2, total_waypoints)
        indices = _even_waypoint_indices(total_cells, total_waypoints, rng)
    else:
        if config.waypoint_spacing is None:
            raise ValueError("config needs waypoint_spacing or total_waypoints")
        minimum, maximum = config.waypoint_spacing
        indices = [0]
        index = 0
        while True:
            index += rng.randint(minimum, maximum)
            if index >= total_cells - 1:
                break
            indices.append(index)
        indices.append(total_cells - 1)

    return [solution[index] for index in sorted(set(indices))]


def _even_waypoint_indices(
    total_cells: int,
    total_waypoints: int,
    rng: random.Random,
) -> list[int]:
    if total_waypoints <= 2:
        return [0, total_cells - 1]

    indices = [0]
    spacing = (total_cells - 1) / (total_waypoints - 1)
    previous = 0
    for position in range(1, total_waypoints - 1):
        ideal = round(position * spacing)
        radius = max(1, round(spacing * 0.3))
        remaining_internal = (total_waypoints - 1) - position
        low = max(previous + 1, ideal - radius)
        high = min(total_cells - 2 - remaining_internal, ideal + radius)
        if high < low:
            high = low
        chosen = rng.randint(low, high)
        indices.append(chosen)
        previous = chosen
    indices.append(total_cells - 1)
    return indices


def _choose_walls(
    solution: list[Coordinate],
    waypoints: list[Coordinate],
    config: DifficultyConfig,
    rng: random.Random,
) -> set[Wall]:
    if config.wall_candidate_policy == "surgical":
        return _choose_surgical_walls(solution, waypoints, config, rng)

    minimum, maximum = config.wall_count
    target = rng.randint(minimum, maximum)
    if target <= 0:
        return set()

    candidates = _wall_candidates(solution, config, rng, descending_gap=True)
    return {wall for _, _, wall in candidates[:target]}


def _choose_surgical_walls(
    solution: list[Coordinate],
    waypoints: list[Coordinate],
    config: DifficultyConfig,
    rng: random.Random,
) -> set[Wall]:
    minimum, maximum = config.wall_count
    target = rng.randint(minimum, maximum)
    if target <= 0:
        return set()

    positions = {cell: index for index, cell in enumerate(solution)}
    intended_edges = solution_edges(solution)
    walls: set[Wall] = set()
    candidates = _wall_candidates(solution, config, rng, descending_gap=True)
    repair_timeout_seconds = min(config.solver_timeout_seconds, 0.15)

    for _, _, wall in candidates[:minimum]:
        walls.add(wall)

    while len(walls) < target:
        puzzle = Puzzle(
            rows=config.rows,
            cols=config.cols,
            waypoints=tuple(waypoints),
            walls=frozenset(walls),
            solution=tuple(solution),
            difficulty=config.name,
            seed=None,
            unique_solution=None,
        )
        report = solve(
            puzzle,
            max_solutions=2,
            timeout_seconds=repair_timeout_seconds,
            return_solutions=True,
        )
        if report.unique_solution is True:
            break

        wall = _wall_from_alternate_solution(
            report.solutions,
            intended_edges,
            positions,
            walls,
            rng,
        )
        if wall is None:
            wall = _next_unused_wall(candidates, walls)
        if wall is None:
            break
        walls.add(wall)

    return walls


def _wall_candidates(
    solution: list[Coordinate],
    config: DifficultyConfig,
    rng: random.Random,
    *,
    descending_gap: bool,
) -> list[tuple[int, float, Wall]]:
    rows, cols = config.rows, config.cols
    positions = {cell: index for index, cell in enumerate(solution)}
    path_edges = solution_edges(solution)
    candidates: list[tuple[int, float, Wall]] = []
    for cell in iter_grid(rows, cols):
        for adjacent in neighbors(rows, cols, cell):
            if adjacent < cell:
                continue
            wall = canonical_wall(cell, adjacent)
            if wall in path_edges:
                continue
            gap = abs(positions[cell] - positions[adjacent])
            candidates.append((gap, rng.random(), wall))

    if descending_gap:
        candidates.sort(key=lambda item: (-item[0], item[1]))
    else:
        candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates


def _wall_from_alternate_solution(
    solutions: list[list[Coordinate]],
    intended_edges: set[Wall],
    positions: dict[Coordinate, int],
    existing_walls: set[Wall],
    rng: random.Random,
) -> Wall | None:
    options: list[tuple[int, float, Wall]] = []
    for solution in solutions:
        for wall in solution_edges(solution) - intended_edges - existing_walls:
            a, b = wall
            gap = abs(positions[a] - positions[b])
            options.append((gap, rng.random(), wall))

    if not options:
        return None

    options.sort(key=lambda item: (-item[0], item[1]))
    return options[0][2]


def _next_unused_wall(
    candidates: list[tuple[int, float, Wall]],
    existing_walls: set[Wall],
) -> Wall | None:
    for _, _, wall in candidates:
        if wall not in existing_walls:
            return wall
    return None
