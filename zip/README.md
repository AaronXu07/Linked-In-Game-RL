# Zip simulator

Standalone LinkedIn Zip-style simulator with no RL or training code.

Simulator code and tests live in `zip/simulation/` so future agent, model, and
training components can sit beside it cleanly.

## Setup

Run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

`pygame` is only needed for the playable window. The core simulator, generator,
solver, JSON handling, and ANSI renderer use the Python standard library.

## Play The Game

Run the mouse-playable UI from the repository root:

```bash
python3 -m zip.simulation.ui --difficulty easy
```

Available difficulties:

```text
super_easy, easy, medium, hard, expert
```

Examples:

```bash
python3 -m zip.simulation.ui --difficulty super_easy
python3 -m zip.simulation.ui --difficulty medium --save-path my_zip_puzzle.json
python3 -m zip.simulation.ui --puzzle my_zip_puzzle.json
```

### Controls

- Click a legal adjacent cell to extend the path.
- Drag through legal adjacent cells to draw the path quickly.
- Use `Undo` to remove the last move.
- Use `Reset` to restart the current puzzle.
- Use `Difficulty` to cycle puzzle difficulty.
- Use `New Puzzle` to generate another puzzle at the selected difficulty.
- Use `Save JSON` and `Load JSON` to persist the current puzzle.
- Use `Replay Solution` to watch the stored generated solution.
- Use `Step Back` and `Step Forward` to inspect solution playback manually.
- Use `Speed -` and `Speed +` to change playback speed.

Keyboard shortcuts:

- `u`: undo
- `r`: reset
- `n`: new puzzle
- `space`: pause/resume solution replay
- `escape`: close the window

## Game Rules

- Move up, down, left, or right only.
- Start on waypoint `1`.
- Visit numbered waypoints in strict ascending order.
- Do not step onto a future waypoint early.
- Do not revisit cells.
- Walls block movement between adjacent cells.
- Solve the puzzle by visiting every cell exactly once and ending on the final waypoint.

## Python API

```python
from zip.simulation import Direction, generate_puzzle, new_game, step

puzzle = generate_puzzle("easy", seed=7)
state = new_game(puzzle)
result = step(puzzle, state, Direction.RIGHT)
```

Useful entry points:

- `zip.simulation.Puzzle` for immutable puzzle definitions and JSON serialization
- `zip.simulation.new_game`, `step`, `undo`, `reset`, `legal_moves` for simulation
- `zip.simulation.generate_puzzle` for solution-first puzzle generation
- `zip.simulation.solve` for solvability and uniqueness checks
- `zip.simulation.renderer.render_ansi` and `render_image` for debug rendering
- `python3 -m zip.simulation.ui --difficulty easy` for the optional Pygame UI

## File Map

Top-level Zip files:

- `zip/README.md`: this guide.
- `zip/__init__.py`: game-level package wrapper that re-exports simulation APIs.
- `zip/simulation_plan.md`: simulator requirements and implementation plan.
- `zip/training_plan.md`: future RL/training plan; not implemented yet.

Simulation package:

- `zip/simulation/__init__.py`: public simulator exports.
- `zip/simulation/config.py`: difficulty settings and UI configuration.
- `zip/simulation/puzzle.py`: immutable `Puzzle` model plus JSON load/save helpers.
- `zip/simulation/state.py`: mutable `GameState` and `StepResult`.
- `zip/simulation/utils.py`: coordinates, directions, wall canonicalization, neighbors, and grid helpers.
- `zip/simulation/rules.py`: shared move legality and solved-state rules used by simulator, solver, and UI.
- `zip/simulation/simulator.py`: public gameplay API: new game, step, undo, reset, legal moves.
- `zip/simulation/generator.py`: solution-first puzzle generation, waypoint placement, and wall placement.
- `zip/simulation/solver.py`: standalone solver for solvability and uniqueness checks.
- `zip/simulation/renderer.py`: ANSI renderer and dependency-free static PPM image renderer.
- `zip/simulation/ui.py`: optional Pygame mouse-playable UI and playback tools.
- `zip/simulation/tests/`: simulator unit tests.

## Tests

Run from the repository root:

```bash
python3 -m pytest zip/simulation/tests -q
```
