# Zip Game RL Training Plan

This plan defines the training layer that should sit on top of the completed
Zip simulator. The simulator is the source of truth for puzzle data, movement
rules, solved-state checks, generation, solving, rendering, and UI playback.
The RL layer should adapt those pieces into Gymnasium-compatible training
interfaces without duplicating game logic.

---

## 1. Goals

Build three primary pieces:

- A Gymnasium environment that exposes Zip puzzles as RL episodes.
- An agent/training pipeline that interacts with the environment, logs
  metrics, checkpoints models, and evaluates held-out seeds.
- A neural Q-network suitable for grid observations, action masking, and
  curriculum learning.

The first milestone should prove that an agent can solve small generated
puzzles reliably. Later milestones can scale puzzle size, model complexity,
and curriculum difficulty.

---

## 2. Current Simulator Contracts To Reuse

Use the existing simulator APIs directly:

- `Puzzle`: immutable puzzle definition with `rows`, `cols`, `waypoints`,
  `walls`, `solution`, `difficulty`, `seed`, and `total_cells`.
- `GameState`: mutable episode state with `path`, `visited`, `current_pos`,
  `next_waypoint_index`, `done`, `success`, and `step_count`.
- `new_game(puzzle)`: creates the initial state at waypoint `1`.
- `step(puzzle, state, direction)`: applies one simulator move and returns a
  `StepResult`.
- `legal_moves(puzzle, state)`: returns legal `Direction` values.
- `generate_puzzle(difficulty, seed=...)`: creates puzzles for training.
- `load_puzzle(path)`: loads fixed JSON puzzles for regression and evaluation.
- `render_ansi` or `render_image`: debug and optional video output.

Important rule details:

- The episode starts on `puzzle.start`, which is waypoint `1`.
- Actions are cardinal moves only.
- Waypoints must be visited in ascending order.
- Future waypoints cannot be entered early.
- Visited cells cannot be revisited.
- Walls block movement.
- The final waypoint can only be entered after all other cells are visited.
- A solved puzzle has visited every cell exactly once and ends on the final
  waypoint.
- Invalid simulator moves return `StepResult(valid=False, state=clone, ...)`
  without mutating the original state.

The training layer must not fork these rules. If a training behavior needs to
ask whether a move is valid, call `legal_moves`, `can_move`, or `step`.

---

## 3. Dependency Additions

Add training dependencies separately from the simulator baseline:

- `gymnasium`: environment API.
- `numpy`: observation arrays and masks.
- `torch`: DQN model, replay-buffer tensors, and optimization.
- `tensorboard`: training metrics.
- `tqdm`: optional progress bars for dataset generation/evaluation.

Optional later dependencies:

- `wandb`: hosted experiment tracking.
- `opencv-python` or `imageio`: video encoding from rendered frames.

Keep `pygame` optional for interactive playback. Headless training should not
require the Pygame UI.

---

## 4. Proposed File Layout

Create training modules beside, not inside, the simulator package:

```text
zip/training/
|-- __init__.py
|-- env.py              # Gymnasium ZipEnv implementation
|-- observations.py     # Observation encoder and spaces
|-- rewards.py          # Reward config and reward computation
|-- curriculum.py       # Difficulty schedules and puzzle sampling
|-- agent.py            # Agent wrapper around model/env interactions
|-- models.py           # Torch DQN networks
|-- replay.py           # Replay buffer and transition batches
|-- train.py            # Training entrypoint
|-- evaluate.py         # Held-out evaluation entrypoint
|-- callbacks.py        # Checkpoints, metrics, action snapshots, videos
|-- datasets.py         # Solver-path dataset for imitation learning
`-- tests/
    |-- test_env.py
    |-- test_observations.py
    |-- test_rewards.py
    `-- test_training_smoke.py
```

Rationale:

- `zip/simulation/` stays dependency-light and usable without RL packages.
- `zip/training/` can depend on Gymnasium, NumPy, Torch, and TensorBoard.
- Observation and reward code are isolated enough to test without launching a
  full training run.

---

## 5. Gymnasium Environment

Implement `ZipEnv` in `zip/training/env.py`.

Required API:

- `reset(seed=None, options=None) -> (observation, info)`
- `step(action) -> (observation, reward, terminated, truncated, info)`
- `render()`
- `close()`
- `observation_space`
- `action_space`

Recommended constructor:

```python
class ZipEnv(gymnasium.Env):
    metadata = {"render_modes": ["ansi", "rgb_array", "human"]}

    def __init__(
        self,
        difficulty: str = "super_easy",
        *,
        puzzle: Puzzle | None = None,
        puzzle_pool: Sequence[Puzzle] | None = None,
        max_rows: int | None = None,
        max_cols: int | None = None,
        reward_config: RewardConfig | None = None,
        invalid_action_mode: str = "terminate",
        render_mode: str | None = None,
        use_action_mask: bool = True,
    ) -> None:
        ...
```

Puzzle selection modes:

- `puzzle`: always reset to the same puzzle.
- `puzzle_pool`: sample a fixed set of puzzles for reproducible training.
- `difficulty`: generate a new puzzle each reset using the environment RNG.
- `options["puzzle"]`: one-off override for tests/evaluation.
- `options["difficulty"]`: one-off curriculum override.
- `options["seed"]`: optional puzzle seed override.

Environment-owned state:

- `self.puzzle: Puzzle`
- `self.state: GameState`
- `self.rng`: seeded RNG derived from Gymnasium's `reset(seed=...)`
- `self.episode_id`
- `self.last_action`
- `self.last_step_reason`
- `self.cumulative_reward`
- `self.episode_step_count`
- `self.max_steps`

Use `self.np_random` from Gymnasium after calling `super().reset(seed=seed)`.
When generating a puzzle, derive an integer puzzle seed from `self.np_random`
and pass it to `generate_puzzle`.

### Action Space

Use a discrete four-action space:

```text
0 -> Direction.UP
1 -> Direction.DOWN
2 -> Direction.LEFT
3 -> Direction.RIGHT
```

Expose constants in `env.py`:

```python
ACTION_TO_DIRECTION = {
    0: Direction.UP,
    1: Direction.DOWN,
    2: Direction.LEFT,
    3: Direction.RIGHT,
}

DIRECTION_TO_ACTION = {direction: action for action, direction in ACTION_TO_DIRECTION.items()}
```

`action_space = gymnasium.spaces.Discrete(4)`.

Do not one-hot encode the Gym action. For DQN, the environment should receive
and return the selected action as an integer class index. The Q-network outputs
one scalar Q-value per action, so action selection is an `argmax` over the four
outputs. One-hot action tensors are only optional internal training helpers;
they should not be part of the public environment action space.

### Action Mask

Return a boolean mask of shape `(4,)`:

- `True`: action is available.
- `False`: action is illegal.

The mask should be based on `legal_moves(puzzle, state)` and the action mapping.
Include it in every `info` dict under `info["action_mask"]`.

Expose a convenience method for the DQN agent:

```python
def action_masks(self) -> np.ndarray:
    return self._action_mask()
```

The DQN code should store masks for both the current observation and the next
observation. The online action selector must avoid illegal actions, and the
Bellman target must compute `max_a Q(next_state, a)` over legal next actions
only.

Invalid move reasons should still be debuggable even when masks are used:

- Out of bounds.
- Wall.
- Revisit.
- Future waypoint.
- Already completed.
- Final waypoint too early.

### Step Flow

`step(action)` should:

1. Validate that `action` is in `action_space`.
2. Convert action to `Direction`.
3. Increment `episode_step_count`.
4. Call simulator `step(self.puzzle, self.state, direction)`.
5. If valid, update `self.state` with `result.state`.
6. Compute reward using the previous state, the action, the `StepResult`, and
   the new state.
7. Set `terminated=True` when solved, invalid, or dead-ended.
8. Set `truncated=True` when `episode_step_count >= max_steps`.
9. Return encoded observation plus `info`.

Do not manually mutate `GameState.path`, `visited`, or waypoint counters in the
environment.

### Termination And Truncation

Use Gymnasium semantics:

- `terminated=True`: natural terminal outcome.
- `truncated=True`: time or training-limit cutoff.

Terminate on:

- Puzzle solved successfully.
- Invalid action when `invalid_action_mode == "terminate"`.
- Dead end when there are no legal actions and the puzzle is not solved.

Alternative invalid-action modes for experiments:

- `"terminate"`: negative reward and end the episode.
- `"penalize"`: negative reward, keep same state, continue.
- `"mask_required"`: raise an error if an invalid action arrives.

Truncate on:

- `episode_step_count >= max_steps`.

Default `max_steps`:

- `2 * puzzle.total_cells`.

This keeps the requested training ceiling while dead-end detection should
usually terminate failed episodes earlier.

### Info Dict

Return this debugging data from `reset` and `step`:

```python
info = {
    "action_mask": mask,
    "success": self.state.success,
    "invalid_action": invalid_action,
    "invalid_reason": result.reason if invalid_action else None,
    "dead_end": dead_end,
    "truncated": truncated,
    "episode_id": self.episode_id,
    "episode_step_count": self.episode_step_count,
    "path_step_count": self.state.step_count,
    "visited_count": len(self.state.visited),
    "total_cells": self.puzzle.total_cells,
    "current_pos": self.state.current_pos,
    "next_waypoint_index": self.state.next_waypoint_index,
    "waypoint_count": len(self.puzzle.waypoints),
    "puzzle_seed": self.puzzle.seed,
    "difficulty": self.puzzle.difficulty,
    "cumulative_reward": self.cumulative_reward,
}
```

---

## 6. Observation Design

Implement observation encoding in `zip/training/observations.py`.

Use a channel-first tensor for PyTorch CNN compatibility:

```text
shape = (channels, max_rows, max_cols)
dtype = np.float32
```

For fixed-size training:

- `max_rows = puzzle.rows`
- `max_cols = puzzle.cols`

For mixed-size training:

- Set `max_rows` and `max_cols` to the largest curriculum size.
- Pad unused cells with zeros.
- Use an explicit valid-cell channel so the model can distinguish padding from
  real empty cells.

Recommended channels:

```text
0  valid_cell
1  visited
2  current_head
3  waypoint
4  current_or_past_waypoint
5  next_waypoint
6  future_waypoint
7  waypoint_number_normalized
8  next_waypoint_number_normalized
9  wall_north
10 wall_south
11 wall_west
12 wall_east
13 legal_move_target
14 solution_path_hint_optional
```

Channel details:

- `valid_cell`: `1` for cells inside `puzzle.rows x puzzle.cols`, else `0`.
- `visited`: `1` for cells in `state.visited`.
- `current_head`: `1` at `state.current_pos`.
- `waypoint`: `1` for any waypoint cell.
- `current_or_past_waypoint`: `1` for waypoint index `< state.next_waypoint_index`.
- `next_waypoint`: `1` for waypoint index `== state.next_waypoint_index`, if any.
- `future_waypoint`: `1` for waypoint index `> state.next_waypoint_index`.
- `waypoint_number_normalized`: waypoint index divided by `len(waypoints) - 1`.
- `next_waypoint_number_normalized`: fill all valid cells with the next waypoint
  index normalized by `len(waypoints) - 1`.
- `wall_*`: `1` if movement from that cell in that direction is blocked by a
  boundary or wall.
- `legal_move_target`: `1` for cells reachable by one legal action from the head.
- `solution_path_hint_optional`: use only for imitation/pretraining ablations;
  do not include this channel in the normal RL observation.

Optional scalar features:

```text
current_row_normalized
current_col_normalized
visited_count_normalized
step_count_normalized
next_waypoint_index_normalized
waypoint_count_normalized
rows_normalized
cols_normalized
```

If scalar features are used, prefer a dictionary observation space:

```python
spaces.Dict(
    {
        "grid": spaces.Box(0.0, 1.0, shape=(channels, max_rows, max_cols), dtype=np.float32),
        "features": spaces.Box(0.0, 1.0, shape=(feature_count,), dtype=np.float32),
    }
)
```

For the first implementation, use only the grid observation. Add scalar
features after the CNN baseline works.

Observation tests should assert:

- Shape and dtype are stable across resets.
- Padding is zero except for explicitly allowed channels.
- The current head appears in exactly one valid cell.
- `visited.sum()` matches `len(state.visited)`.
- Wall channels match `Puzzle.has_wall` and grid boundaries.
- Legal move target channel matches `legal_moves`.
- Encoded future waypoints change after a waypoint is reached.

---

## 7. Reward Design

Implement reward configuration in `zip/training/rewards.py`.

Initial reward constants:

```python
@dataclass(frozen=True)
class RewardConfig:
    valid_step: float = 0.0
    new_cell: float = 0.02
    waypoint: float = 1.0
    solve: float = 10.0
    invalid_action: float = -2.0
    dead_end: float = -2.0
    truncate: float = -1.0
```

Recommended first reward:

- `+0.02` for a valid move into a new cell.
- `+1.0` for reaching the next waypoint.
- `+10.0` for solving the puzzle.
- `-2.0` for invalid action if invalid actions are possible.
- `-2.0` for dead end.
- `-1.0` for truncation.

Avoid large per-step penalties initially. The agent cannot undo or revisit, so
failed exploration should mainly be punished by invalid-action, dead-end, and
truncation outcomes.

Reward computation should detect:

- `invalid_action`: `StepResult.valid is False`.
- `reached_waypoint`: new state's `next_waypoint_index` is greater than the
  previous state's index.
- `solved`: `new_state.success is True`.
- `dead_end`: no legal moves remain and the puzzle is not solved.
- `truncated`: max steps exceeded.

Keep reward calculation pure and testable:

```python
def compute_reward(
    puzzle: Puzzle,
    previous_state: GameState,
    result: StepResult,
    *,
    dead_end: bool,
    truncated: bool,
    config: RewardConfig,
) -> float:
    ...
```

---

## 8. Agent Layer

Implement the agent-facing interaction wrapper in `zip/training/agent.py`.

The agent layer should not contain game rules. It should coordinate:

- Environment creation.
- Model creation/loading.
- Action selection.
- Replay buffer insertion.
- DQN optimization steps.
- Target network updates.
- Optional action-mask handling.
- Rollout collection for debugging.
- Evaluation episodes.
- Checkpoint loading and saving.

Suggested classes/functions:

```python
@dataclass(frozen=True)
class AgentConfig:
    algorithm: str = "dqn"
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


class ZipAgent:
    def __init__(self, env, config: AgentConfig, checkpoint_path: str | None = None):
        ...

    def predict(self, observation, *, deterministic: bool = False, action_mask=None) -> int:
        ...

    def train_step(self, batch) -> dict:
        ...

    def train(self, total_steps: int, callbacks=None) -> None:
        ...

    def evaluate(self, puzzles: Sequence[Puzzle], episodes: int, deterministic: bool = True) -> dict:
        ...
```

Action selection:

- During training, use epsilon-greedy exploration over legal actions only.
- During evaluation, use masked greedy `argmax`.
- If all actions are masked, choose a deterministic fallback action and let the
  environment terminate as a dead end.
- Keep `invalid_action_mode="terminate"` so bugs in masking are visible.

Replay buffer transitions should store:

```python
Transition(
    obs=obs,
    action=action,
    reward=reward,
    next_obs=next_obs,
    terminated=terminated,
    truncated=truncated,
    action_mask=action_mask,
    next_action_mask=next_action_mask,
)
```

Optimization target:

- Use Huber loss.
- Use gradient clipping.
- Use `done = terminated or truncated` for the first baseline.
- For Double DQN, choose the next action with the online network and evaluate
  it with the target network.
- Mask invalid next actions before computing the bootstrap value.
- If `terminated=True`, target is just `reward`.

Target equation:

```text
q_sa = online_net(obs).gather(action)
target = reward + gamma * (1 - done) * max_legal_next_q
loss = smooth_l1_loss(q_sa, target)
```

Target network updates:

- Hard update every `target_update_every` steps when `tau == 1.0`.
- Soft update when `0.0 < tau < 1.0`.

Rollout debug records should include:

- Puzzle seed and difficulty.
- Step index.
- Observation summary.
- Action.
- Action mask.
- Q-values.
- Epsilon.
- Reward.
- Cumulative reward.
- Current position.
- Next waypoint index.
- Invalid reason.
- Terminal reason.

---

## 9. Model Architecture

Start with a compact convolutional DQN. Zip is a spatial planning problem, so a
CNN baseline is a better first model than a flat MLP.

### Baseline CNN Q-Network

Implement a PyTorch module in `zip/training/models.py`:

```text
input:  (channels, max_rows, max_cols)
conv1:  32 filters, 3x3, padding=1, ReLU
conv2:  64 filters, 3x3, padding=1, ReLU
conv3:  64 filters, 3x3, padding=1, ReLU
flatten
linear: 256 units, ReLU
output: 4 Q-values, one per action
```

Suggested class:

```python
class ZipDQN(nn.Module):
    def __init__(self, observation_shape: tuple[int, int, int], action_count: int = 4):
        ...

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Returns shape: (batch, action_count)
        ...
```

Use two copies:

- `online_net`: optimized every training step.
- `target_net`: used for Bellman targets.

Action masking should live outside the model:

```python
masked_q = q_values.masked_fill(~action_mask, -torch.inf)
action = masked_q.argmax(dim=1)
```

If a batch contains terminal next states, do not bootstrap from their
`next_action_mask`.

### Why This Is Enough For The First Pass

- Puzzles are small (`4x4` through `8x8` in current configs).
- Wall, waypoint, visited, and head information are naturally grid channels.
- Action masking reduces the need for the model to learn basic legality.
- DQN can learn from sparse terminal success once waypoint and solve rewards
  provide intermediate signal.

### Later Model Options

Add only after the baseline has been measured:

- Residual CNN blocks for harder puzzles.
- Dueling DQN heads.
- Prioritized replay.
- N-step returns.
- CoordConv channels for row/column position awareness.
- Graph neural network over cells and legal edges.
- Transformer over grid tokens plus path/history features.
- Recurrent Q-network if partial observability is introduced.
- Behavioral-cloning warm start using solver paths.

---

## 10. Training Pipeline

Implement `zip/training/train.py` as the main entrypoint.

Minimum CLI:

```bash
python3 -m zip.training.train \
  --difficulty super_easy \
  --total-steps 200000 \
  --seed 0 \
  --log-dir runs/zip_super_easy \
  --checkpoint-dir checkpoints/zip_super_easy
```

Important flags:

- `--difficulty`: one of `super_easy`, `easy`, `medium`, `hard`, `expert`.
- `--curriculum`: optional schedule name.
- `--max-rows`, `--max-cols`: fixed observation bounds.
- `--total-steps`.
- `--seed`.
- `--learning-rate`.
- `--replay-size`.
- `--warmup-steps`.
- `--batch-size`.
- `--train-every`.
- `--target-update-every`.
- `--epsilon-start`.
- `--epsilon-end`.
- `--epsilon-decay-steps`.
- `--double-dqn`.
- `--checkpoint-path`: resume training.
- `--eval-every`.
- `--eval-seeds`.
- `--save-video-every`: optional render snapshots.
- `--device`: `auto`, `cpu`, `cuda`, or `mps`.

Start with this curriculum:

1. `super_easy` only until held-out success rate is above 95%.
2. Mix `super_easy` and `easy`.
3. `easy` only.
4. Mix `easy` and `medium`.
5. Continue only after the agent remains stable on held-out seeds.

For each stage, keep a fixed evaluation seed list:

```text
super_easy: 1000-1099
easy:       2000-2099
medium:     3000-3099
hard:       4000-4099
expert:     5000-5099
```

Use separate training seeds and evaluation seeds. Do not evaluate only on
puzzles generated during training.

### Experience Collection

Start with one environment and a simple online collection loop:

1. Reset the environment.
2. Select an epsilon-greedy legal action.
3. Step the environment.
4. Store the transition in replay.
5. Optimize every `train_every` environment steps after `warmup_steps`.
6. Reset when the episode terminates or truncates.

Add multiple collector environments only after the single-environment loop is
stable. DQN does not need synchronous rollout batches, but parallel
collection can help once generation and training are working.

If puzzle generation becomes the bottleneck, pre-generate puzzle pools per
difficulty and sample from those in each environment.

### Logging

Log to TensorBoard:

- `episode/reward`
- `episode/length`
- `episode/success`
- `eval/success_rate`
- `eval/mean_reward`
- `eval/mean_episode_length`
- `eval/invalid_action_rate`
- `eval/dead_end_rate`
- `eval/truncation_rate`
- `eval/mean_visited_fraction`
- `train/loss`
- `train/td_error`
- `train/q_mean`
- `train/q_max`
- `train/epsilon`
- `train/replay_size`
- `curriculum/difficulty_id`
- `curriculum/grid_cells`

Also write periodic JSONL episode summaries:

```json
{
  "episode_id": 42,
  "difficulty": "easy",
  "puzzle_seed": 2007,
  "success": true,
  "episode_length": 24,
  "reward": 12.48,
  "invalid_action_count": 0,
  "dead_end": false,
  "truncated": false
}
```

---

## 11. Evaluation Pipeline

Implement `zip/training/evaluate.py`.

Evaluation should support:

- A checkpoint path.
- Difficulty or puzzle JSON path.
- Fixed seed ranges.
- Greedy and epsilon-greedy action modes.
- Action-mask enabled evaluation.
- Optional rollout export.
- Optional rendered image/video export.

Minimum CLI:

```bash
python3 -m zip.training.evaluate \
  --checkpoint checkpoints/zip_super_easy/best_model.zip \
  --difficulty super_easy \
  --seeds 1000:1099 \
  --deterministic \
  --out reports/super_easy_eval.json
```

Evaluation output:

```json
{
  "checkpoint": "...",
  "difficulty": "super_easy",
  "episodes": 100,
  "success_rate": 0.97,
  "mean_reward": 11.8,
  "mean_episode_length": 15.0,
  "invalid_action_rate": 0.0,
  "dead_end_rate": 0.03,
  "truncation_rate": 0.0,
  "failures": [
    {
      "seed": 1042,
      "reason": "dead_end",
      "visited_count": 14,
      "total_cells": 16
    }
  ]
}
```

Keep failed rollouts replayable by saving:

- Puzzle JSON.
- Action sequence.
- Reward sequence.
- Terminal reason.
- Optional Q-values.
- Optional action masks.

---

## 12. Imitation Learning Optional Warm Start

Because generated puzzles often include `puzzle.solution`, build an optional
solver-path dataset.

Dataset rows:

- Encoded observation before the move.
- Correct action from the solution path.
- Action mask.
- Difficulty.
- Puzzle seed.
- Step index.

Use cases:

- Supervised action pretraining before DQN fine-tuning.
- Model sanity checks.
- Overfit tests on tiny puzzle sets.
- Action-mask correctness tests.

For supervised action pretraining, treat the Q-values as action logits and use
masked cross-entropy against the solution action. Then switch to the normal DQN
loss for RL fine-tuning.

Do not include a solution hint channel in normal RL training unless explicitly
running an imitation or ablation experiment. The normal agent should solve from
the same puzzle information a human player sees.

---

## 13. Training Observer And UI Integration

The existing simulator UI can later gain an agent-observation mode. This should
consume exported rollouts instead of coupling the UI directly to the training
process.

Observer features:

- Load puzzle JSON plus action/reward trace.
- Play back an agent episode step by step.
- Pause/resume playback.
- Speed control.
- Step forward and backward.
- Compare agent path to stored/generated solution when available.
- Show action probabilities when exported.
- Show value estimate when exported.
- Show reward at each step.
- Show cumulative reward.
- Show success/failure reason.
- Show episode length.
- Show visited count.
- Show current curriculum difficulty.

The training process should emit lightweight snapshots:

- JSONL summaries for dashboards.
- Full rollout JSON for selected episodes.
- Optional `rgb_array` frame sequences for failed episodes.

---

## 14. Testing Plan

Add focused tests before running long training jobs.

Environment tests:

- `reset()` returns a valid observation and info dict.
- `step()` follows simulator `step()` exactly for all four actions.
- Invalid actions terminate or penalize according to `invalid_action_mode`.
- Solving with `puzzle.solution` produces `success=True`.
- `terminated` and `truncated` follow Gymnasium semantics.
- `info["action_mask"]` matches `legal_moves`.
- Dead-end detection triggers only when unsolved and no legal moves remain.
- Reset with the same seed produces the same puzzle seed and puzzle structure.

Observation tests:

- Shape, dtype, and value ranges match `observation_space`.
- Valid-cell mask handles mixed-size padding.
- Wall channels match boundaries and `Puzzle.walls`.
- Waypoint channels update after reaching a waypoint.
- Legal-move target channel matches the mask.

Reward tests:

- Valid move reward.
- Waypoint reward.
- Solve reward.
- Invalid action reward.
- Dead-end reward.
- Truncation reward.

Training smoke tests:

- Build model and environment.
- Run a tiny training job, such as 128 environment steps.
- Save and reload checkpoint.
- Run deterministic evaluation for a few fixed seeds.

Imitation tests:

- Convert a known solution path to actions.
- Replay generated labels through the environment.
- Confirm replay solves the puzzle.

---

## 15. First Implementation Milestone

Target a minimal but complete vertical slice:

1. Add `zip/training/` package.
2. Add dependencies for Gymnasium, NumPy, Torch, TensorBoard, and `tqdm`.
3. Implement `ZipEnv` with discrete actions, masks, reset, step, and ANSI
   render support.
4. Implement grid observation encoding.
5. Implement reward config and reward computation.
6. Add environment, observation, and reward tests.
7. Implement `ZipDQN`, replay buffer, optimizer, and target network updates.
8. Implement `train.py` using a PyTorch DQN loop.
9. Implement `evaluate.py` for fixed seed ranges.
10. Run a smoke training job on `super_easy`.
11. Save failed rollouts in a replayable JSON format.

Definition of done:

- Tests pass.
- The environment passes Gymnasium's environment checker or all known checker
  failures are documented.
- A random-policy rollout can run without crashes.
- A solver-path replay can solve generated puzzles.
- A short DQN run starts, logs metrics, checkpoints, and evaluates.

---

## 16. Open Design Decisions

These should be decided during the first implementation pass:

- Whether mixed-size training starts immediately or after fixed-size baselines.
- Whether scalar features are worth adding before the first DQN baseline.
- Whether puzzle generation should happen on reset or from pre-generated pools.
- Whether invalid actions should terminate, penalize, or be impossible via masks.
- Whether to train separate policies per difficulty or one curriculum policy.
- Whether to use generated solution paths for behavioral cloning before DQN.

Recommended defaults:

- Start fixed-size with `super_easy`.
- Use action masking.
- Terminate invalid actions.
- Generate puzzles on reset until generation becomes a bottleneck.
- Train one policy through a curriculum after the fixed-size baseline works.
- Add behavioral cloning only if DQN struggles on `super_easy`.
