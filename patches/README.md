# Patches simulator and training

Standalone LinkedIn Patches-style rectangle-packing simulator.

Simulator code and tests live in `patches/simulation/`. The Gymnasium
environment, candidate-conditioned DQN agent, checkpoints, curriculum helpers,
evaluation CLI, and visual trainer live in `patches/training/`.

## Game Rules

- Partition the grid into non-overlapping rectangular patches.
- Each patch contains exactly one clue.
- Every clue specifies a shape the patch must take:
  - `square`: width equals height.
  - `wide`: width greater than height.
  - `tall`: height greater than width.
  - `free` ("any of the above"): any rectangle shape.
- A clue's number is optional. If a clue shows a number, its patch must be that
  exact size (cell count). Clues without a number can be any size, as long as
  the shape matches.
- Every cell must be covered exactly once, with no gaps or overlaps.
- The board is solved when every clue has a valid patch and the whole grid is
  covered.

Each clue is drawn as a small token in its patch's color: a solid square, wide,
or tall silhouette, or a dashed square for "any of the above". The number, when
present, is shown on the token.

## Difficulties

```text
super_easy (5x5), easy (6x6), medium (7x7), hard (8x8), expert (9x9)
```

`medium` and above are generated with a unique, logically deducible solution.

## Play The Game

Run the mouse-playable UI from the repository root:

```bash
python3 -m patches.simulation.ui --difficulty easy
```

Examples:

```bash
python3 -m patches.simulation.ui --difficulty medium --save-path my_patch_puzzle.json
python3 -m patches.simulation.ui --puzzle my_patch_puzzle.json
```

Each patch is drawn in its own color, and the board is supersampled for crisp,
high-resolution rendering.

### Controls

- Press a clue and drag outward to size its patch. The box is anchored on the
  clue, so it grows from the clue toward the cursor instead of flipping sides. A
  live preview shows the current area vs the required number and turns green
  when the placement is valid.
- Click a filled patch to clear it.
- Each clue is a colored token showing its required shape (square, tall, wide,
  or a dashed square for any). A number appears on the token only when the size
  is fixed. Harder difficulties reveal fewer hints, so most clues are "any"
  shape with no number and must be deduced.
- Use `Undo`, `Reset`, and `Clear All` to manage the board.
- Use `Difficulty` to cycle difficulty and `New Puzzle` to generate another.
- Use `Save JSON` / `Load JSON` to persist the current puzzle.
- Use `Reveal Step` / `Hide Step` / `Reveal All` to inspect the stored solution.

Keyboard shortcuts:

- `u`: undo
- `r`: reset
- `n`: new puzzle
- `c`: clear all
- `space`: reveal next solution patch
- `escape`: close the window

## Python API

```python
from patches.simulation import (
    Rect,
    generate_puzzle,
    new_game,
    place,
    candidate_rects,
    solve,
)

puzzle = generate_puzzle("medium", seed=7)
state = new_game(puzzle)

# Placement search space for a clue (bounds/area/shape/single-clue valid):
options = candidate_rects(puzzle, clue_id=0)

# Commit a patch for a clue:
result = place(puzzle, state, clue_id=0, rect=Rect(0, 0, 2, 3))
state = result.state
```

Useful entry points:

- `patches.simulation.Puzzle` / `Clue` for immutable definitions and JSON I/O
- `patches.simulation.new_game`, `place`, `clear_patch`, `undo`, `reset`
- `patches.simulation.candidate_rects`, `can_place` for placement queries
- `patches.simulation.generate_puzzle` for solution-first generation
- `patches.simulation.solve` / `find_solution` for solvability and uniqueness
- `patches.simulation.renderer.render_ansi` / `render_image` for debug rendering

## Training

Run headless candidate-DQN training:

```bash
python3 -m patches.training.train \
  --difficulty medium \
  --curriculum default \
  --parallel-envs 4 \
  --parallel-puzzles 4 \
  --checkpoint-dir checkpoints/patches_medium_curriculum
```

Watch live visual training with multiple boards:

```bash
python3 -m patches.training.visual_train \
  --difficulty medium \
  --curriculum default \
  --parallel-puzzles 4 \
  --checkpoint-every 1000
```

Resume a saved visual run:

```bash
python3 -m patches.training.visual_train \
  --difficulty medium \
  --curriculum default \
  --parallel-puzzles 4 \
  --checkpoint-path checkpoints/patches_visual/visual_latest.pt
```

Evaluate a saved checkpoint:

```bash
python3 -m patches.training.evaluate \
  --checkpoint-path checkpoints/patches_super_easy/latest.pt \
  --difficulty super_easy \
  --episodes 100
```

The visual trainer writes `visual_latest.pt`, `visual_best.pt`, and
`visual_interrupted.pt` under `checkpoints/patches_visual` by default. These
checkpoints include model weights, target network weights, optimizer state,
training counters, and curriculum metadata so training can continue after the
program closes.

`--curriculum default` uses the same shape as Zip: train on `super_easy`, then
mix each previous/current difficulty before moving to the current difficulty
alone. Held-out evaluation gates advance the active stage, and async puzzle
buffers discard stale prefetched boards when the curriculum stage changes.

## Tests

Run from the repository root:

```bash
python3 -m pytest patches/simulation/tests -q
python3 -m pytest patches/training/tests -q
```

## Status

Implemented: puzzle/clue model, shared rules, simulator API, exact-cover solver
with uniqueness checking, solution-first generator, ANSI/PPM renderers, and the
mouse-playable Pygame UI (`ui.py`). Training includes dynamic placement action
masks, dictionary observations, a candidate-conditioned DQN, replay buffer,
curriculum/evaluation/checkpoint callbacks, headless training, checkpoint
evaluation, and multi-board visual training.

`pygame` is only needed for the playable and visual training windows; the core
simulator, generator, solver, JSON handling, and ANSI renderer use the Python
standard library. Training additionally requires the dependencies in the root
`requirements.txt`.
