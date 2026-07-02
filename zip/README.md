# Zip simulator

Standalone LinkedIn Zip-style simulator with an RL training layer beside it.

Simulator code and tests live in `zip/simulation/`; agent, model, environment,
and training components live in `zip/training/`.

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

## Training

The first training milestone implements a Gymnasium environment plus a compact
masked DQN pipeline:

```bash
python3 -m zip.training.train \
  --difficulty super_easy \
  --total-steps 200000 \
  --seed 0 \
  --log-dir runs/zip_super_easy \
  --checkpoint-dir checkpoints/zip_super_easy
```

Evaluate a checkpoint against held-out generated seeds:

```bash
python3 -m zip.training.evaluate \
  --checkpoint-path checkpoints/zip_super_easy/latest.pt \
  --difficulty super_easy \
  --episodes 100
```

The environment adapts the simulator directly: moves are applied through
`zip.simulation.step`, action masks come from `legal_moves`, and observations
are channel-first NumPy grids for PyTorch.

To watch training happen live in the Pygame UI:

```bash
python3 -m zip.training.visual_train \
  --difficulty super_easy \
  --steps-per-second 8 \
  --warmup-steps 64 \
  --batch-size 32 \
  --device cpu
```

The visual trainer uses the same board renderer as manual play. Use
Space/Pause to pause, `s` or **Single Step** to advance one training action,
and **Save Checkpoint** to write `checkpoints/zip_visual/visual_latest.pt`.

## File Map

Top-level Zip files:

- `zip/README.md`: this guide.
- `zip/__init__.py`: game-level package wrapper that re-exports simulation APIs.
- `zip/simulation_plan.md`: simulator requirements and implementation plan.
- `zip/training_plan.md`: RL/training implementation plan.

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

Training package:

- `zip/training/env.py`: Gymnasium environment adapter with action masks.
- `zip/training/observations.py`: channel-first grid observation encoder.
- `zip/training/rewards.py`: pure reward calculation.
- `zip/training/models.py`: compact convolutional DQN.
- `zip/training/replay.py`: replay buffer and transition batches.
- `zip/training/agent.py`: masked DQN action selection, optimization, evaluation, and checkpoints.
- `zip/training/train.py`: training CLI.
- `zip/training/evaluate.py`: checkpoint evaluation CLI.
- `zip/training/visual_train.py`: Pygame visual training CLI.
- `zip/training/tests/`: training-layer unit and smoke tests.

## Tests

Run from the repository root:

```bash
python3 -m pytest zip/simulation/tests -q
```

Run the training tests after installing the full requirements:

```bash
python3 -m pytest zip/training/tests -q
```
