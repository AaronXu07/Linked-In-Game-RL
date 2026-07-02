"""Evaluate a trained Zip DQN checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from zip.simulation import Puzzle, load_puzzle
from zip.simulation.config import DIFFICULTIES

from .agent import AgentConfig, ZipAgent
from .curriculum import evaluation_seeds, generate_evaluation_puzzles
from .env import ZipEnv
from .train import parse_seed_spec


def main() -> None:
    args = parse_args()
    puzzles, seeds = load_evaluation_puzzles(args)
    max_rows = args.max_rows or max(puzzle.rows for puzzle in puzzles)
    max_cols = args.max_cols or max(puzzle.cols for puzzle in puzzles)
    env = ZipEnv(puzzle=puzzles[0], max_rows=max_rows, max_cols=max_cols)
    agent = ZipAgent(
        env,
        AgentConfig(seed=args.seed, device=args.device),
        checkpoint_path=args.checkpoint_path,
    )
    report = evaluate_detailed(
        agent,
        puzzles,
        episodes=args.episodes,
        seeds=seeds,
        checkpoint=args.checkpoint_path,
        difficulty=args.difficulty,
        deterministic=args.deterministic and not args.stochastic,
        rollout_dir=args.rollout_dir,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", "--checkpoint", required=True)
    parser.add_argument("--difficulty", choices=sorted(DIFFICULTIES), default="super_easy")
    parser.add_argument("--puzzle", default=None)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--eval-seeds", "--seeds", default="")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-cols", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--out", default=None)
    parser.add_argument("--rollout-dir", default=None)
    return parser.parse_args()


def load_evaluation_puzzles(args: argparse.Namespace) -> tuple[list[Puzzle], tuple[int | None, ...]]:
    if args.puzzle:
        puzzle = load_puzzle(args.puzzle)
        return [puzzle], (puzzle.seed,)

    seeds = parse_seed_spec(args.eval_seeds)
    if not seeds:
        seeds = evaluation_seeds(args.difficulty, args.episodes)
    puzzles = generate_evaluation_puzzles(args.difficulty, seeds=seeds)
    return puzzles, tuple(seeds)


def evaluate_detailed(
    agent: ZipAgent,
    puzzles: list[Puzzle],
    *,
    episodes: int,
    seeds: tuple[int | None, ...],
    checkpoint: str,
    difficulty: str,
    deterministic: bool,
    rollout_dir: str | None,
) -> dict[str, object]:
    eval_env = ZipEnv(
        puzzle=puzzles[0],
        max_rows=agent.env.max_rows,
        max_cols=agent.env.max_cols,
        reward_config=agent.env.reward_config,
        invalid_action_mode=agent.env.invalid_action_mode,
    )
    rollout_path = Path(rollout_dir) if rollout_dir else None
    if rollout_path is not None:
        rollout_path.mkdir(parents=True, exist_ok=True)

    rewards: list[float] = []
    lengths: list[int] = []
    failures: list[dict[str, object]] = []
    successes = invalid_actions = dead_ends = truncations = total_action_steps = 0
    visited_fractions: list[float] = []

    for episode in range(episodes):
        puzzle = puzzles[episode % len(puzzles)]
        seed = seeds[episode % len(seeds)] if seeds else puzzle.seed
        obs, info = eval_env.reset(options={"puzzle": puzzle})
        actions: list[int] = []
        step_rewards: list[float] = []
        action_masks: list[list[bool]] = []
        episode_reward = 0.0
        episode_length = 0

        while True:
            action_masks.append([bool(value) for value in info["action_mask"]])
            action = agent.predict(
                obs,
                deterministic=deterministic,
                action_mask=info["action_mask"],
            )
            obs, reward, terminated, truncated, info = eval_env.step(action)
            actions.append(action)
            step_rewards.append(float(reward))
            episode_reward += reward
            episode_length += 1
            total_action_steps += 1
            invalid_actions += int(info["invalid_action"])
            if terminated or truncated:
                rewards.append(episode_reward)
                lengths.append(episode_length)
                successes += int(info["success"])
                dead_ends += int(info["dead_end"])
                truncations += int(truncated)
                visited_fractions.append(info["visited_count"] / max(1, info["total_cells"]))
                terminal_reason = _terminal_reason(info, truncated)
                rollout = {
                    "episode": episode,
                    "seed": seed,
                    "difficulty": puzzle.difficulty,
                    "puzzle": puzzle.to_dict(),
                    "actions": actions,
                    "rewards": step_rewards,
                    "action_masks": action_masks,
                    "terminal_reason": terminal_reason,
                    "success": bool(info["success"]),
                    "visited_count": info["visited_count"],
                    "total_cells": info["total_cells"],
                }
                if not info["success"]:
                    failure = {
                        "episode": episode,
                        "seed": seed,
                        "reason": terminal_reason,
                        "visited_count": info["visited_count"],
                        "total_cells": info["total_cells"],
                    }
                    failures.append(failure)
                    if rollout_path is not None:
                        output = rollout_path / f"failure_{episode:04d}.json"
                        output.write_text(
                            json.dumps(rollout, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8",
                        )
                break

    return {
        "checkpoint": checkpoint,
        "difficulty": difficulty,
        "episodes": episodes,
        "success_rate": successes / episodes,
        "mean_reward": sum(rewards) / len(rewards),
        "mean_episode_length": sum(lengths) / len(lengths),
        "invalid_action_rate": invalid_actions / max(1, total_action_steps),
        "dead_end_rate": dead_ends / episodes,
        "truncation_rate": truncations / episodes,
        "mean_visited_fraction": sum(visited_fractions) / len(visited_fractions),
        "failures": failures,
    }


def _terminal_reason(info: dict[str, object], truncated: bool) -> str:
    if info["success"]:
        return "success"
    if truncated:
        return "truncated"
    if info["invalid_action"]:
        return str(info["invalid_reason"] or "invalid_action")
    if info["dead_end"]:
        return "dead_end"
    return "terminated"


if __name__ == "__main__":
    main()
