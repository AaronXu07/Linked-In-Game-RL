import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("torch")

from zip.simulation.puzzle import Puzzle
from zip.training.agent import AgentConfig, ZipAgent
from zip.training.env import ZipEnv
from zip.training.models import ZipDQN
from zip.training.replay import ReplayBuffer, Transition


SNAKE_2X2 = (
    (0, 0),
    (0, 1),
    (1, 1),
    (1, 0),
)


def tiny_puzzle() -> Puzzle:
    return Puzzle(
        rows=2,
        cols=2,
        waypoints=(SNAKE_2X2[0], SNAKE_2X2[-1]),
        walls=frozenset(),
        solution=SNAKE_2X2,
    )


def test_model_forward_shape() -> None:
    import torch

    model = ZipDQN((14, 2, 2), action_count=4)
    output = model(torch.zeros(3, 14, 2, 2))

    assert tuple(output.shape) == (3, 4)


def test_replay_buffer_samples_transition_batch() -> None:
    replay = ReplayBuffer(4, seed=0)
    obs = np.zeros((14, 2, 2), dtype=np.float32)
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

    assert batch.obs.shape == (2, 14, 2, 2)
    assert batch.action.shape == (2,)
    assert batch.action_mask.dtype == np.bool_


def test_agent_train_smoke() -> None:
    env = ZipEnv(puzzle=tiny_puzzle())
    agent = ZipAgent(
        env,
        AgentConfig(
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
    envs = [ZipEnv(puzzle=tiny_puzzle()), ZipEnv(puzzle=tiny_puzzle())]
    agent = ZipAgent(
        envs[0],
        AgentConfig(
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
