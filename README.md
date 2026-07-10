# Linked-In-Game-RL
Training an AI (through Reinforcement Learning) to master Linked In Games

## Quick Start

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Run headless Zip DQN training:

```bash
python3 -m zip.training.train \
  --difficulty medium \
  --curriculum default \
  --parallel-envs 4 \
  --parallel-puzzles 4
```

Watch live Zip training in Pygame:

```bash
python3 -m zip.training.visual_train \
  --difficulty medium \
  --curriculum default \
  --parallel-envs 4 \
  --parallel-puzzles 4 \
  --checkpoint-every 1000
```

Resume a previous Zip visual run:

```bash
python3 -m zip.training.visual_train \
  --difficulty medium \
  --curriculum default \
  --parallel-envs 4 \
  --parallel-puzzles 4 \
  --checkpoint-path checkpoints/zip_visual/visual_latest.pt
```

See `zip/README.md` for the simulator, Gymnasium environment, DQN training,
curriculum learning, parallel environments, checkpoints, and visualizer controls.

Run headless Patches DQN training:

```bash
python3 -m patches.training.train \
  --difficulty medium \
  --curriculum default \
  --parallel-envs 4 \
  --parallel-puzzles 4
```

Watch live Patches training with multiple boards:

```bash
python3 -m patches.training.visual_train \
  --difficulty medium \
  --curriculum default \
  --parallel-puzzles 4 \
  --checkpoint-every 1000
```

Resume a previous Patches visual run:

```bash
python3 -m patches.training.visual_train \
  --difficulty medium \
  --curriculum default \
  --parallel-puzzles 4 \
  --checkpoint-path checkpoints/patches_visual/visual_latest.pt
```

See `patches/README.md` for the Patches simulator, dynamic candidate-action
environment, candidate-conditioned DQN, checkpoints, and visualizer controls.
