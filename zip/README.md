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

The training layer implements a Gymnasium environment plus a compact masked DQN
pipeline. The agent learns from channel-first grid observations, uses action
masks from legal moves, stores transitions in replay, and periodically updates a
target network.

### Headless Training

Basic single-environment training:

```bash
python3 -m zip.training.train \
  --difficulty super_easy \
  --total-steps 200000 \
  --seed 0 \
  --log-dir runs/zip_super_easy \
  --checkpoint-dir checkpoints/zip_super_easy
```

Curriculum training up to a target difficulty:

```bash
python3 -m zip.training.train \
  --difficulty medium \
  --curriculum default \
  --total-steps 200000 \
  --checkpoint-dir checkpoints/zip_medium_curriculum \
  --log-dir runs/zip_medium_curriculum
```

`--difficulty` is the curriculum target. For example, `--difficulty medium
--curriculum default` starts on `super_easy`, mixes in `easy`, then works toward
`medium`.

Parallel experience collection with one shared DQN:

```bash
python3 -m zip.training.train \
  --difficulty medium \
  --curriculum default \
  --parallel-envs 4 \
  --parallel-puzzles 4 \
  --puzzle-buffer-size 16 \
  --checkpoint-dir checkpoints/zip_parallel
```

- `--parallel-envs 4`: runs four live training environments that all feed the
  same replay buffer and neural network.
- `--parallel-puzzles 4`: runs four background puzzle-generation workers so new
  boards are ready when environments reset.
- `--puzzle-buffer-size 16`: keeps extra generated puzzles queued.

This is not four separate agents competing. It is four active game streams
training one shared DQN policy.

Evaluate a checkpoint against held-out generated seeds:

```bash
python3 -m zip.training.evaluate \
  --checkpoint-path checkpoints/zip_super_easy/latest.pt \
  --difficulty super_easy \
  --episodes 100
```

Resume training from a checkpoint:

```bash
python3 -m zip.training.train \
  --difficulty medium \
  --curriculum default \
  --checkpoint-path checkpoints/zip_parallel/latest.pt \
  --checkpoint-dir checkpoints/zip_parallel
```

### Checkpoints

Training writes:

- `latest.pt`: latest checkpoint.
- `step_<N>.pt`: periodic numbered snapshots.
- `best.pt`: best evaluated checkpoint by held-out success rate.
- `interrupted.pt`: best-effort save when the process receives Ctrl-C/SIGTERM.

Visual training writes:

- `visual_latest.pt`: latest visual-training checkpoint.
- `visual_best.pt`: best visual-training checkpoint from curriculum evaluation.
- `visual_interrupted.pt`: best-effort save when visual training is interrupted.

`latest.pt` and `visual_latest.pt` include model weights, target network,
optimizer state, global step counters, and curriculum stage metadata.

### Loading Previous Models

Training does not automatically load the last checkpoint. A run starts from a
fresh randomly initialized network unless you pass `--checkpoint-path`.

Resume headless training from the latest checkpoint:

```bash
python3 -m zip.training.train \
  --difficulty medium \
  --curriculum default \
  --parallel-envs 4 \
  --parallel-puzzles 4 \
  --checkpoint-path checkpoints/zip_parallel/latest.pt \
  --checkpoint-dir checkpoints/zip_parallel
```

Resume visual training from the latest visual checkpoint:

```bash
python3 -m zip.training.visual_train \
  --difficulty medium \
  --curriculum default \
  --parallel-envs 4 \
  --parallel-puzzles 4 \
  --checkpoint-path checkpoints/zip_visual/visual_latest.pt \
  --checkpoint-dir checkpoints/zip_visual
```

Load the best saved model instead of the latest:

```bash
python3 -m zip.training.visual_train \
  --difficulty medium \
  --curriculum default \
  --parallel-envs 4 \
  --parallel-puzzles 4 \
  --checkpoint-path checkpoints/zip_visual/visual_best.pt
```

Evaluate a previously trained headless model:

```bash
python3 -m zip.training.evaluate \
  --checkpoint-path checkpoints/zip_parallel/best.pt \
  --difficulty medium \
  --episodes 100
```

Start completely from scratch by omitting `--checkpoint-path`. To remove old
visual checkpoints before starting fresh:

```bash
rm -rf checkpoints/zip_visual
```

Only delete `runs/...` directories if you also want to clear TensorBoard and
episode-history logs.

### How The Environment Connects

The environment adapts the simulator directly: moves are applied through
`zip.simulation.step`, action masks come from `legal_moves`, and observations
are channel-first NumPy grids for PyTorch.

Actions are:

```text
0 = up
1 = down
2 = left
3 = right
```

Default rewards are:

```text
+0.02  new cell
+1.00  next waypoint
+10.00 solve
-2.00  invalid action
-2.00  dead end
-1.00  truncation
```

### Watch Training

Single-board visual training:

```bash
python3 -m zip.training.visual_train \
  --difficulty super_easy \
  --steps-per-second 8 \
  --warmup-steps 64 \
  --batch-size 32 \
  --device cpu
```

Four live training boards, one shared DQN:

```bash
python3 -m zip.training.visual_train \
  --difficulty medium \
  --curriculum default \
  --parallel-envs 4 \
  --parallel-puzzles 4 \
  --checkpoint-every 1000
```

If `--parallel-envs` is omitted, visual training treats `--parallel-puzzles N`
with `N > 1` as a request to show `N` active training boards. Prefer
`--parallel-envs` when you want to be explicit.

The visual trainer uses the same board renderer as manual play. Multi-board
mode lays boards out in a compact grid and shrinks cell sizes as needed so the
window stays within the detected display bounds.

Visual controls:

- Space or **Pause**: pause/resume training.
- `s` or **Single Step**: advance each active environment once.
- `n` or **New Episodes**: reset all active environments.
- **Speed -** and **Speed +**: change visual training speed.
- **Save Checkpoint**: write `checkpoints/zip_visual/visual_latest.pt`.
- Escape: close the window.

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
