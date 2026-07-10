import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("torch")

from patches.simulation.puzzle import Clue, Puzzle
from patches.simulation.utils import Rect, ShapeType
from patches.training.agent import AgentConfig, PatchesAgent
from patches.training.env import PatchesEnv
from patches.training.models import PatchesCandidateDQN
from patches.training.replay import ReplayBuffer, Transition


def tiny_puzzle() -> Puzzle:
    clues = [
        Clue(id=0, pos=(0, 0), number=2, shape=ShapeType.WIDE),
        Clue(id=1, pos=(1, 0), number=2, shape=ShapeType.WIDE),
    ]
    solution = [Rect(0, 0, 1, 2), Rect(1, 0, 1, 2)]
    return Puzzle(rows=2, cols=2, clues=clues, solution=solution)


def test_model_forward_shape() -> None:
    import torch

    model = PatchesCandidateDQN((15, 2, 2), candidate_feature_count=20, max_actions=4)
    output = model(
        torch.zeros(3, 15, 2, 2),
        torch.zeros(3, 4, 20),
        torch.zeros(3, 4, 2, 2),
    )

    assert tuple(output.shape) == (3, 4)


def test_replay_buffer_samples_transition_batch() -> None:
    replay = ReplayBuffer(4, seed=0)
    obs = {
        "grid": np.zeros((15, 2, 2), dtype=np.float32),
        "candidates": np.zeros((4, 20), dtype=np.float32),
        "candidate_footprints": np.zeros((4, 2, 2), dtype=np.float32),
    }
    mask = np.array([True, False, False, True])
    for action in range(4):
        replay.add(
            Transition(
                obs=obs,
                action=action % 4,
                reward=1.0,
                next_obs=obs,
                terminated=False,
                truncated=False,
                action_mask=mask,
                next_action_mask=mask,
            )
        )

    batch = replay.sample(2)

    assert batch.obs["grid"].shape == (2, 15, 2, 2)
    assert batch.action.shape == (2,)
    assert batch.action_mask.dtype == np.bool_


def test_agent_train_smoke() -> None:
    env = PatchesEnv(puzzle=tiny_puzzle(), max_actions=4)
    agent = PatchesAgent(
        env,
        AgentConfig(
            algorithm="flat_candidate_dqn",
            replay_size=32,
            warmup_steps=2,
            batch_size=2,
            train_every=1,
            target_update_every=2,
            epsilon_decay_steps=1,
            seed=7,
            device="cpu",
        ),
    )

    agent.train(6)

    assert agent.total_steps == 6
    assert len(agent.replay) >= 6


def test_agent_train_parallel_smoke() -> None:
    envs = [
        PatchesEnv(puzzle=tiny_puzzle(), max_actions=4),
        PatchesEnv(puzzle=tiny_puzzle(), max_actions=4),
    ]
    agent = PatchesAgent(
        envs[0],
        AgentConfig(
            algorithm="flat_candidate_dqn",
            replay_size=32,
            warmup_steps=2,
            batch_size=2,
            train_every=1,
            target_update_every=2,
            epsilon_decay_steps=1,
            seed=7,
            device="cpu",
        ),
    )

    agent.train_parallel(envs, 8)

    assert agent.total_steps == 8
    assert len(agent.replay) >= 8


def test_checkpoint_reload_preserves_prediction(tmp_path) -> None:
    env = PatchesEnv(puzzle=tiny_puzzle(), max_actions=4)
    config = AgentConfig(
        algorithm="flat_candidate_dqn",
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay_steps=1,
        seed=11,
        device="cpu",
    )
    agent = PatchesAgent(env, config)
    obs, info = env.reset(seed=11)
    before = agent.predict(obs, deterministic=True, action_mask=info["action_mask"])
    path = tmp_path / "latest.pt"
    agent.save_checkpoint(path)

    reloaded = PatchesAgent(env, config, checkpoint_path=path)
    after = reloaded.predict(obs, deterministic=True, action_mask=info["action_mask"])

    assert after == before
