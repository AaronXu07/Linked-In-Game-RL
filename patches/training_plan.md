# Patches RL Training Plan

This plan defines the training layer for the completed Patches simulator. The
simulator remains the source of truth for puzzle data, placement rules, solved
state checks, generation, solving, rendering, and UI playback. The training
layer should adapt those pieces into Gymnasium-compatible environments and
PyTorch agents without duplicating game rules.

The Zip implementation in `zip/training/` is the main reference for project
shape: Gym environment, observations, reward module, replay buffer, DQN agent,
curriculum, callbacks, checkpoints, evaluation, and smoke tests. Patches should
reuse that structure, but not Zip's four-direction action model. Patches has a
dynamic set of clue-rectangle placement actions, so the model and environment
must make candidate generation and masking first-class.

---

## 1. Goals

Build four primary pieces:

- A Gymnasium environment that exposes Patches puzzles as placement episodes.
- A candidate-action layer that turns current legal clue/rectangle placements
  into padded, maskable action slots.
- A candidate-conditioned agent/model that scores legal placements instead of
  assuming fixed action semantics.
- A training pipeline with imitation pretraining, DQN fine-tuning, curriculum
  learning, logging, checkpointing, and held-out evaluation.

The first milestone should prove that an agent can solve fixed and generated
`super_easy` puzzles reliably. Later milestones should scale to `easy` and
`medium`, then improve search/generalization before `hard` and `expert`.

---

## 2. Simulator Contracts To Reuse

Use the existing simulator APIs directly:

- `Puzzle`: immutable puzzle definition with `rows`, `cols`, `clues`,
  optional `solution`, `difficulty`, `seed`, `unique_solution`, and
  `total_cells`.
- `Clue`: clue id, position, optional number, and shape constraint.
- `GameState`: mutable episode state with `patches`, `assignment`, `done`,
  `success`, and `step_count`.
- `Rect`: top/left/height/width rectangle helper with `area`, `cells()`, and
  `contains()`.
- `new_game(puzzle)`: creates the initial empty state.
- `candidate_rects(puzzle, clue_id)`: returns bounds/shape/area/single-clue
  valid rectangles for a clue, ignoring occupancy.
- `can_place(puzzle, state, clue_id, rect)`: validates a candidate placement
  against current occupancy.
- `place(puzzle, state, clue_id, rect)`: applies one placement and returns a
  `StepResult`.
- `clear_patch`, `undo`, and `reset`: useful for later interactive/revision
  environments, but not needed in the first RL baseline.
- `generate_puzzle(difficulty, seed=...)`: creates puzzles for training.
- `solve` / `find_solution`: verifies solvability and can support imitation
  datasets or optional dead-end checks.
- `render_ansi` / `render_image`: debug rendering and optional snapshots.

Important rule details:

- Every patch must be an axis-aligned rectangle.
- Every patch must contain exactly one clue.
- A numbered clue requires exact area; a numberless clue allows any area that
  satisfies the shape constraint.
- Shape validation is centralized through `shape_matches`.
- Patches cannot overlap.
- The puzzle is solved when every clue has a patch and every cell is covered.
- Invalid simulator placements return `StepResult(valid=False, state=clone, ...)`
  without mutating the original state.

The training layer must not fork these rules. Candidate validity should call
`candidate_rects`, `can_place`, or `place`.

---

## 3. Proposed File Layout

Create training modules beside the simulator:

```text
patches/training/
|-- __init__.py
|-- actions.py          # Candidate placement generation, sorting, encoding
|-- env.py              # Gymnasium PatchesEnv implementation
|-- observations.py     # Board/candidate observation encoders and spaces
|-- rewards.py          # Reward config and reward computation
|-- models.py           # Candidate-conditioned Torch Q-networks
|-- replay.py           # Replay buffer for dict observations and masks
|-- datasets.py         # Solver/solution-derived imitation examples
|-- curriculum.py       # Difficulty schedules and puzzle sampling/prefetch
|-- agent.py            # Agent wrapper around model/env interactions
|-- train.py            # DQN training entrypoint
|-- pretrain.py         # Behavior cloning entrypoint
|-- evaluate.py         # Held-out evaluation entrypoint
|-- callbacks.py        # Checkpoints, metrics, JSONL, TensorBoard
`-- tests/
    |-- test_actions.py
    |-- test_env.py
    |-- test_observations.py
    |-- test_rewards.py
    |-- test_datasets.py
    `-- test_training_smoke.py
```

Rationale:

- `patches/simulation/` stays dependency-light and usable without RL packages.
- `patches/training/` can depend on Gymnasium, NumPy, Torch, TensorBoard, and
  tqdm, matching the existing project dependencies.
- Action, observation, and reward code are isolated enough to test without
  launching a full training run.

---

## 4. Environment Design

Implement `PatchesEnv` in `patches/training/env.py`.

Required Gymnasium API:

- `reset(seed=None, options=None) -> (observation, info)`
- `step(action) -> (observation, reward, terminated, truncated, info)`
- `render()`
- `close()`
- `observation_space`
- `action_space`

Recommended constructor:

```python
class PatchesEnv(gymnasium.Env):
    metadata = {"render_modes": ["ansi", "rgb_array", "human"]}

    def __init__(
        self,
        difficulty: str = "super_easy",
        *,
        puzzle: Puzzle | None = None,
        puzzle_pool: Sequence[Puzzle] | None = None,
        puzzle_sampler: Any | None = None,
        max_rows: int | None = None,
        max_cols: int | None = None,
        max_actions: int | None = None,
        reward_config: RewardConfig | None = None,
        invalid_action_mode: str = "terminate",
        episode_mode: str = "commit_only",
        render_mode: str | None = None,
        use_action_mask: bool = True,
        use_solver_dead_end_check: bool = False,
    ) -> None:
        ...
```

Puzzle selection modes should mirror Zip:

- `puzzle`: always reset to the same puzzle.
- `puzzle_pool`: sample a fixed puzzle set for reproducible training.
- `puzzle_sampler`: curriculum or async puzzle source.
- `difficulty`: generate a new puzzle each reset.
- `options["puzzle"]`: one-off override for tests/evaluation.
- `options["difficulty"]`: one-off curriculum override.
- `options["seed"]`: optional puzzle seed override.

Environment-owned state:

- `self.puzzle: Puzzle`
- `self.state: GameState`
- `self.episode_id`
- `self.last_action`
- `self.last_step_reason`
- `self.cumulative_reward`
- `self.episode_step_count`
- `self.max_steps`
- `self._base_candidates_by_clue`
- `self._current_actions`
- `self._current_action_mask`

Use `self.np_random` from Gymnasium after calling `super().reset(seed=seed)`.
When generating a puzzle, derive an integer puzzle seed from `self.np_random`
and pass it to `generate_puzzle`.

---

## 5. Action Space

Do not copy Zip's `Discrete(4)` movement action model. Patches actions are
placements:

```python
@dataclass(frozen=True)
class PlacementAction:
    clue_id: int
    rect: Rect
```

Use a padded discrete slot space:

```python
action_space = gymnasium.spaces.Discrete(max_actions)
```

At each state, action index `i` maps to `current_actions[i]`. Slots beyond the
current action count are masked illegal. This preserves Gym compatibility while
allowing a dynamic action set.

### Candidate generation

Precompute occupancy-independent candidates on reset:

```python
base_candidates_by_clue = {
    clue.id: tuple(candidate_rects(puzzle, clue.id))
    for clue in puzzle.clues
}
```

For each state, build legal actions by filtering through `can_place`.

Default first baseline:

- `episode_mode="commit_only"`
- Only include clues that are not already placed.
- A wrong placement is permanent for that episode.
- Episodes are short, acyclic, and close to exact-cover search.

Later experimental mode:

- `episode_mode="revision"`
- Include replacements for already placed clues.
- Optionally include clear actions.
- Use a larger `max_steps` and stronger loop penalties.

### Deterministic ordering

Candidate order must be stable across identical states. Sort actions by:

1. Most constrained clue first: current legal candidate count ascending.
2. Clue id ascending.
3. Rectangle top, left, height, width.

This gives index slots a reproducible meaning within a state and makes solution
trajectories deterministic.

### max_actions

`max_actions` is the padded action capacity. Infer a conservative default from
the target difficulty, or pass it explicitly from the training CLI.

If a puzzle produces more legal actions than `max_actions`, do not silently
drop actions in the default implementation. Raise a clear error in tests and
training setup. Later, add a top-K mode only if needed, using solver heuristics
or most-constrained ordering.

### Action mask

Return a boolean mask of shape `(max_actions,)`:

- `True`: slot maps to a currently legal placement.
- `False`: slot is padding or illegal.

Include it in every `info` dict under `info["action_mask"]`, and expose:

```python
def action_masks(self) -> np.ndarray:
    return self._action_mask()
```

The agent must mask illegal actions during epsilon-greedy action selection and
when computing Bellman targets.

---

## 6. Step Flow

`step(action)` should:

1. Validate that `action` is in `action_space`.
2. Recompute or reuse the current candidate action list.
3. Treat masked/padding slots as invalid according to `invalid_action_mode`.
4. Convert the slot to `PlacementAction(clue_id, rect)`.
5. Increment `episode_step_count`.
6. Call simulator `place(self.puzzle, self.state, clue_id, rect)`.
7. If valid, update `self.state` with `result.state`.
8. Rebuild candidate actions and action mask for the new state.
9. Compute reward using the previous state, action, result, dead-end status,
   truncation status, and reward config.
10. Set `terminated=True` when solved, invalid, or dead-ended.
11. Set `truncated=True` when `episode_step_count >= max_steps`.
12. Return encoded observation plus `info`.

Do not manually mutate `GameState.patches`, `assignment`, or success flags in
the environment.

Default `max_steps`:

- `len(puzzle.clues)` for `commit_only`.
- `3 * len(puzzle.clues)` for `revision`.

Gymnasium semantics:

- `terminated=True`: natural terminal outcome, such as solved, invalid action,
  or dead end.
- `truncated=True`: time or training-limit cutoff.

Dead-end checks:

- Cheap default: no legal placement actions remain and the puzzle is not
  solved.
- Strong optional check: run a residual exact-cover solver from the partial
  state to detect globally unsolvable branches early. Keep this off by default
  because solver calls can dominate training time.

Invalid action modes:

- `"terminate"`: negative reward and end the episode.
- `"penalize"`: negative reward, keep the same state, continue.
- `"mask_required"`: raise an error if an invalid action arrives.

---

## 7. Observation Design

Use a Gymnasium dictionary observation rather than a plain grid tensor:

```python
spaces.Dict(
    {
        "grid": spaces.Box(0.0, 1.0, shape=(C, max_rows, max_cols), dtype=np.float32),
        "candidates": spaces.Box(0.0, 1.0, shape=(max_actions, F), dtype=np.float32),
        "candidate_footprints": spaces.Box(
            0.0,
            1.0,
            shape=(max_actions, max_rows, max_cols),
            dtype=np.float32,
        ),
    }
)
```

Keep the action mask in `info`, matching Zip, and store masks in replay.

Recommended grid channels:

```text
0  valid_cell
1  covered_cell
2  uncovered_cell
3  clue_cell
4  unplaced_clue
5  placed_clue
6  clue_number_normalized
7  clue_number_missing
8  shape_square
9  shape_wide
10 shape_tall
11 shape_free
12 assignment_owner_normalized
13 placed_patch_boundary
14 legal_candidate_coverage_count_normalized
15 solution_patch_optional
```

Channel details:

- `valid_cell`: `1` for cells inside the active puzzle, else `0`.
- `covered_cell`: `1` where `state.assignment` has an owner.
- `uncovered_cell`: `1` for valid cells without an owner.
- `clue_cell`: `1` at all clue positions.
- `unplaced_clue` / `placed_clue`: clue status markers.
- `clue_number_normalized`: clue number divided by total cells; `0` when
  number is missing.
- `clue_number_missing`: `1` for numberless clues.
- `shape_*`: one-hot clue shape at clue cells.
- `assignment_owner_normalized`: owner id normalized by clue count on covered
  cells.
- `placed_patch_boundary`: `1` on the border cells of placed rectangles.
- `legal_candidate_coverage_count_normalized`: optional heatmap of how many
  legal current candidates cover each cell, normalized by max count.
- `solution_patch_optional`: use only for imitation/debug ablations, not normal
  RL evaluation.

Recommended candidate features:

```text
clue_id_normalized
clue_row_normalized
clue_col_normalized
clue_has_number
clue_number_normalized
shape_square
shape_wide
shape_tall
shape_free
rect_top_normalized
rect_left_normalized
rect_bottom_normalized
rect_right_normalized
rect_height_normalized
rect_width_normalized
rect_area_normalized
rect_contains_current_patch
rect_matches_solution_optional
new_covered_fraction
overlap_with_current_same_clue_fraction
```

`candidate_footprints[i]` should be `1` for cells inside candidate action `i`.
This gives the model direct spatial information about what each action would
cover.

Observation tests should assert:

- Shape and dtype are stable across resets.
- Padding is zero except for explicitly documented channels.
- Covered/uncovered counts match `state.covered_count`.
- Clue channels match puzzle clues.
- Candidate features and footprints align with `current_actions`.
- Action masks match `can_place`.
- Replacing candidate generation order is deterministic for a fixed state.

---

## 8. Model Design

Zip's `ZipDQN` outputs one Q-value per fixed direction. That is not enough for
Patches because action slot `7` can mean a different clue/rectangle on every
state. Use a candidate-conditioned Q-network:

```python
class PatchesCandidateDQN(nn.Module):
    def forward(
        self,
        grid: torch.Tensor,
        candidates: torch.Tensor,
        candidate_footprints: torch.Tensor,
    ) -> torch.Tensor:
        # returns Q-values with shape (batch, max_actions)
        ...
```

Recommended architecture:

1. Encode the board grid with a compact CNN.
2. Produce both a global board embedding and spatial feature map.
3. Encode each candidate feature vector with a shared MLP.
4. Pool board feature-map values under each candidate footprint.
5. Concatenate global board embedding, candidate MLP embedding, and pooled
   footprint embedding.
6. Score each candidate with a shared MLP head to produce one scalar Q-value.
7. Mask illegal slots to `-inf` outside the model during action selection and
   target computation.

This design lets the model learn "what this rectangle does in this board"
instead of memorizing the meaning of an arbitrary action index.

Initial size:

- CNN channels: 32, 64, 64.
- Candidate MLP hidden size: 128.
- Combined head hidden size: 256.
- Output: one Q-value per candidate slot.

Add a simpler model only as a smoke baseline:

- `FlatCandidateDQN`: flatten grid and candidate tensors, score candidates with
  an MLP. Useful for tiny test puzzles, not expected to scale.

---

## 9. Reward Shaping

Implement reward calculation in `patches/training/rewards.py`.

Initial config:

```python
@dataclass(frozen=True)
class RewardConfig:
    valid_placement: float = 0.05
    covered_cell: float = 0.02
    placed_clue: float = 0.05
    solve: float = 10.0
    invalid_action: float = -2.0
    dead_end: float = -2.0
    truncate: float = -1.0
    non_solution_placement: float = 0.0
    solution_placement: float = 0.0
```

Recommended first reward:

- `+0.05` for a valid placement.
- `+0.02 * newly_covered_cells` for covering new cells.
- `+0.05` for placing a previously unplaced clue.
- `+10.0` for solving.
- `-2.0` for invalid action.
- `-2.0` for dead end.
- `-1.0` for truncation.

Keep oracle rewards off by default:

- `solution_placement`: optional positive reward when the placed rectangle
  equals `puzzle.solution_rect(clue_id)`.
- `non_solution_placement`: optional small penalty for a locally-valid but
  non-solution placement when a stored solution exists.

Oracle shaping can accelerate early experiments, but held-out evaluation should
also run with normal rewards and deterministic policy behavior.

Reward computation should stay pure and testable:

```python
def compute_reward(
    puzzle: Puzzle,
    previous_state: GameState,
    action: PlacementAction | None,
    result: StepResult,
    *,
    dead_end: bool,
    truncated: bool,
    config: RewardConfig,
) -> float:
    ...
```

Do not add a large per-step penalty initially. In `commit_only` mode, every
valid step commits useful structure, and failures are already punished through
invalid, dead-end, and truncation outcomes.

---

## 10. Imitation Data

Unlike Zip paths, Patches solutions are unordered rectangle tilings. Build
ordered demonstration trajectories from stored solutions.

Recommended policy for demonstrations:

1. Start from `new_game(puzzle)`.
2. Among unplaced clues, compute legal candidates.
3. Restrict to clues whose solution rectangle is still legal.
4. Choose the most constrained clue, using legal candidate count as the key.
5. Emit an example with current observation and the slot index for that clue's
   solution rectangle.
6. Apply `place`.
7. Repeat until solved.

This teaches both placement and a useful ordering heuristic.

Dataset module:

```python
@dataclass(frozen=True)
class ImitationExample:
    observation: dict[str, np.ndarray]
    action: int
    action_mask: np.ndarray
```

Functions:

- `solution_examples(puzzle, ...)`
- `build_solution_dataset(puzzles, ...)`
- `generate_solution_dataset(difficulty, seeds, ...)`

Pretraining:

- Use cross-entropy over masked candidate logits.
- Start with fixed-size `super_easy` generated puzzles.
- Verify overfit on one puzzle before scaling.
- Save checkpoints compatible with DQN fine-tuning.

This is the biggest planned improvement over copying Zip directly. Patches has
short episodes and sparse end rewards; behavior cloning from generated
solutions should dramatically reduce the cold-start problem.

---

## 11. Agent And Replay

Adapt Zip's agent structure, but update it for dictionary observations and
candidate-conditioned Q-values.

Agent responsibilities:

- Environment creation.
- Model creation/loading.
- Masked epsilon-greedy action selection.
- Replay buffer insertion.
- Double-DQN optimization.
- Target network updates.
- Optional behavior-cloning warm start.
- Evaluation episodes.
- Checkpoint loading and saving.

Suggested config:

```python
@dataclass(frozen=True)
class AgentConfig:
    algorithm: str = "candidate_dqn"
    learning_rate: float = 1e-4
    replay_size: int = 100_000
    warmup_steps: int = 5_000
    batch_size: int = 128
    gamma: float = 0.99
    train_every: int = 4
    target_update_every: int = 1_000
    tau: float = 1.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 100_000
    double_dqn: bool = True
    max_grad_norm: float = 0.5
    seed: int = 0
    device: str = "auto"
```

Replay transitions should store:

- `obs`: dictionary observation.
- `action`: integer slot.
- `reward`.
- `next_obs`: dictionary observation.
- `terminated`.
- `truncated`.
- `action_mask`.
- `next_action_mask`.

Bellman target:

- Compute `max_a Q(next_state, a)` only over `next_action_mask`.
- If no legal next action exists, use next value `0`.
- Treat `terminated | truncated` as done for bootstrapping.

Action selection:

- If exploring, sample uniformly among legal slots.
- If exploiting, mask illegal Q-values to `-inf` and pick `argmax`.
- If no legal slots remain, return `0`; the env should be terminal/dead-ended.

---

## 12. Curriculum

Port Zip's curriculum shape:

- Difficulty order: `super_easy`, `easy`, `medium`, `hard`, `expert`.
- Curriculum stages mix previous and current difficulties before switching to
  the current difficulty alone.
- Use fixed evaluation seeds per difficulty.
- Gate progression by held-out success rate.

Initial gates:

```text
super_easy: advance at 0.98 success
easy mix:   advance at 0.90 success
easy:       advance at 0.90 success
medium mix: advance at 0.80 success
medium:     final target for first serious run
```

Keep async puzzle prefetch from Zip. Patches generation and uniqueness checks
can be more expensive than environment stepping, especially for `medium+`.

Training CLI examples:

```bash
python3 -m patches.training.pretrain \
  --difficulty super_easy \
  --episodes 5000 \
  --checkpoint-dir checkpoints/patches_bc_super_easy

python3 -m patches.training.train \
  --difficulty super_easy \
  --checkpoint-path checkpoints/patches_bc_super_easy/latest.pt \
  --total-steps 200000 \
  --log-dir runs/patches_super_easy \
  --checkpoint-dir checkpoints/patches_super_easy

python3 -m patches.training.train \
  --difficulty medium \
  --curriculum default \
  --parallel-envs 4 \
  --parallel-puzzles 4 \
  --puzzle-buffer-size 16 \
  --checkpoint-dir checkpoints/patches_medium_curriculum
```

---

## 13. Evaluation

Evaluate deterministic policy on held-out generated seeds:

- Success rate.
- Mean episode length.
- Mean reward.
- Invalid action rate.
- Dead-end rate.
- Truncation rate.
- Mean legal action count.
- Mean placed clues at failure.
- Mean covered-cell fraction at failure.
- Per-difficulty breakdown.

Add optional debug artifacts:

- Episode JSONL with selected placements and invalid reasons.
- ANSI board snapshots for failed episodes.
- Rendered final-state PPM/PNG for a small sample.
- Confusion summary: which clue shapes or numberless clues cause failures.

Evaluation should not use oracle rewards or solution channels. Stored solutions
may be used only for measuring whether selected placements match ground truth.

---

## 14. Tests

Start with tests before long training runs:

- `test_actions.py`
  - Candidate action list includes all legal unplaced placements.
  - Occupancy filtering matches `can_place`.
  - Sorting is deterministic.
  - `max_actions` overflow raises a clear error.

- `test_env.py`
  - `reset` returns observation and info with a valid action mask.
  - `step` applies the selected candidate via simulator `place`.
  - Invalid masked actions terminate or raise according to mode.
  - A known four-patch puzzle can be solved by selecting solution candidates.
  - Dead-end and truncation behavior is correct.

- `test_observations.py`
  - Grid channels match puzzle/state.
  - Candidate features match actions.
  - Footprints match rectangles.
  - Padding is zeroed.

- `test_rewards.py`
  - Valid, invalid, solve, dead-end, and truncate rewards are correct.
  - Optional oracle rewards are disabled by default.

- `test_datasets.py`
  - Solution examples form a valid trajectory.
  - Most-constrained solution ordering is deterministic.
  - Dataset actions are legal under the emitted masks.

- `test_training_smoke.py`
  - Tiny model can run a few DQN optimization steps.
  - Behavior-cloning loss decreases on a tiny fixed dataset.
  - Checkpoint save/load preserves deterministic predictions.

---

## 15. Milestones

### Milestone 1: Environment foundation

- Add `patches/training/actions.py`, `env.py`, `observations.py`, `rewards.py`.
- Implement candidate action masks and dictionary observations.
- Add action/env/observation/reward tests.
- Prove a scripted solution can solve a fixed four-patch puzzle through
  `PatchesEnv`.

### Milestone 2: Imitation baseline

- Add `datasets.py`, `models.py`, and `pretrain.py`.
- Generate solution examples from stored generated puzzles.
- Train a candidate-conditioned model with masked cross-entropy.
- Overfit one fixed puzzle, then solve held-out `super_easy` seeds
  deterministically.

### Milestone 3: DQN fine-tuning

- Add dict replay buffer, `agent.py`, `train.py`, `evaluate.py`, callbacks.
- Load BC checkpoints into DQN.
- Run fixed-puzzle and generated `super_easy` smoke training.
- Track TensorBoard metrics and episode JSONL.

### Milestone 4: Curriculum and scale

- Port/adapt Zip curriculum, async puzzle buffer, best-checkpoint callback.
- Train through `easy` and target `medium`.
- Add failure artifact capture for debugging.
- Tune `max_actions`, reward weights, and model size.

### Milestone 5: Stronger search behavior

- Add optional residual solver dead-end detection for small boards.
- Experiment with oracle reward ablations, prioritized replay, and
  most-constrained action sampling.
- Consider revision mode only after commit-only reaches a plateau.

---

## 16. Key Differences From Zip

- Zip has four fixed cardinal actions; Patches has dynamic clue-rectangle
  candidate actions.
- Zip's Q-network can output one value per action index; Patches needs a shared
  candidate scorer.
- Zip's solution is an ordered path; Patches solutions are unordered tilings, so
  imitation data needs an ordering heuristic.
- Zip can terminate on movement dead ends cheaply; Patches may need optional
  solver-assisted dead-end detection to catch globally impossible partial
  tilings early.
- Patches should start with behavior cloning because generated puzzles already
  contain solution tilings and episodes have sparse terminal success rewards.

