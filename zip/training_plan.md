# Zip Game RL Training Plan

This file intentionally holds the RL-specific work separately from the simulator plan. Build the simulator first, then use it as the foundation for training.

---

## 1. Training Scope

The training layer should reuse the simulator's existing pieces:

- `Puzzle`
- `GameState`
- Shared movement rules
- Puzzle generator
- Solver
- Renderers
- UI playback tools

Do not fork or duplicate game rules for training. The RL environment should call the same simulator transition logic used by human play.

---

## 2. Gymnasium Environment

`ZipEnv` should implement the `gymnasium.Env` API:

- `reset(seed=None, options=None)`
- `step(action)`
- `render()`
- `close()`
- `observation_space`
- `action_space`

### Action Space

Discrete 4-action space:

- `0`: up
- `1`: down
- `2`: left
- `3`: right

Expose an action mask:

- Invalid because out of bounds
- Invalid because wall
- Invalid because revisit
- Invalid because future waypoint

Masking is strongly recommended for PPO-style algorithms that support it.

### Observation Space

Use a fixed-shape observation for RL compatibility.

For fixed-size training, use:

- `rows x cols x channels`

For mixed grid sizes, use:

- `max_rows x max_cols x channels`
- A valid-cell mask channel
- Padded cells filled with zero

Suggested channels:

- `is_valid_cell`
- `is_waypoint`
- `waypoint_number_normalized`
- `is_visited`
- `is_current_head`
- `next_waypoint_number_normalized`
- `wall_north`
- `wall_south`
- `wall_east`
- `wall_west`

Optional scalar features:

- Current row normalized
- Current col normalized
- Visited count normalized
- Next waypoint index normalized
- Total waypoint count normalized

---

## 3. Rewards and Termination

Initial reward function:

- Valid move: small neutral or small positive reward
- Step penalty: small negative reward only if needed
- Reaching next waypoint: positive reward
- Completing puzzle: large positive reward
- Invalid action without masking: negative reward and terminate
- Dead end: negative reward and terminate

Be careful with step penalties: every successful path has the same length, so a step penalty mainly punishes failed exploration, not inefficient successful solutions.

Terminate when:

1. Puzzle solved successfully
2. Invalid move is chosen and invalid moves are not masked
3. Dead end occurs
4. Max steps exceeded

Set max steps to `rows * cols - 1` when invalid revisits are impossible, or a larger debugging ceiling when experimenting.

Return useful debugging data in `info`:

- `success`
- `invalid_action`
- `dead_end`
- `next_waypoint_index`
- `visited_count`
- `action_mask`
- `puzzle_seed`
- `difficulty`

---

## 4. Training Pipeline

Start with a simple baseline before tuning:

1. Generate a small fixed curriculum of 4x4 and 5x5 puzzles.
2. Train with action masking.
3. Evaluate on held-out seeds.
4. Add harder puzzles only after the agent reliably solves easier ones.

Recommended algorithms:

- Maskable PPO as the first serious baseline
- Behavioral cloning or imitation learning from solver paths as an optional warm start
- Curriculum learning by grid size, waypoint density, and wall count

Training scripts should support:

- Configurable difficulty
- Fixed seed evaluation
- Checkpointing
- TensorBoard logging
- Video or image-array episode recording
- Evaluation on generated and saved puzzles

---

## 5. Training Observer

The simulator UI can later grow an agent-observation mode.

Observer features:

- Play back an agent episode step by step
- Pause/resume playback
- Speed control
- Step forward manually
- Compare agent path to oracle solution when available
- Show action probabilities when the model exposes them
- Show value estimate when available
- Show reward at each step
- Show cumulative reward
- Show success/failure reason
- Show episode length
- Show visited count
- Show current curriculum difficulty

Training dashboards should include:

- Success rate over time
- Mean episode reward
- Mean episode length
- Invalid action rate
- Dead-end rate
- Current difficulty distribution
- Recent puzzle seeds

The training process should emit snapshots or logs that the UI can read without slowing training too much.

---

## 6. Future Module Additions

When training starts, add these modules on top of the simulator:

```text
zip/simulation/
├── environment.py     # Gymnasium environment
├── callbacks.py       # Training logging, snapshots, and videos
├── train.py           # RL training entrypoint
└── evaluate.py        # Held-out puzzle evaluation
```

Keep all rule logic in the simulator layer. Training modules should adapt simulator state into RL observations and actions, not redefine the game.
