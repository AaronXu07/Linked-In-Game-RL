"""Reinforcement-learning tools for the Zip simulator."""

from __future__ import annotations

__all__ = [
    "AgentConfig",
    "RewardConfig",
    "ZipAgent",
    "ZipDQN",
    "ZipEnv",
]


def __getattr__(name: str):
    if name == "RewardConfig":
        from .rewards import RewardConfig

        return RewardConfig
    if name == "ZipEnv":
        from .env import ZipEnv

        return ZipEnv
    if name in {"AgentConfig", "ZipAgent"}:
        from .agent import AgentConfig, ZipAgent

        return {"AgentConfig": AgentConfig, "ZipAgent": ZipAgent}[name]
    if name == "ZipDQN":
        from .models import ZipDQN

        return ZipDQN
    raise AttributeError(name)
