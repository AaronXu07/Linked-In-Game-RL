"""Evaluate a trained Patches DQN checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from patches.simulation import Puzzle, load_puzzle
from patches.simulation.config import DIFFICULTIES

from .agent import AgentConfig, PatchesAgent
from .curriculum import evaluation_seeds, generate_evaluation_puzzles
from .env import PatchesEnv
from .train import parse_seed_spec


def main() -> None:
    args = parse_args()
    puzzles, seeds = load_evaluation_puzzles(args)
    max_rows = args.max_rows or max(puzzle.rows for puzzle in puzzles)
    max_cols = args.max_cols or max(puzzle.cols for puzzle in puzzles)
    env = PatchesEnv(
        puzzle=puzzles[0],
        max_rows=max_rows,
        max_cols=max_cols,
        max_actions=args.max_actions,
    )
    agent = PatchesAgent(
        env,
        AgentConfig(seed=args.seed, device=args.device, algorithm=args.algorithm),
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
    parser.add_argument("--max-actions", type=int, default=None)
    parser.add_argument("--algorithm", choices=("candidate_dqn", "flat_candidate_dqn"), default="candidate_dqn")
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
    agent: PatchesAgent,
    puzzles: list[Puzzle],
    *,
    episodes: int,
    seeds: tuple[int | None, ...],
    checkpoint: str,
    difficulty: str,
    deterministic: bool,
    rollout_dir: str | None,
) -> dict[str, object]:
    eval_env = PatchesEnv(
        puzzle=puzzles[0],
        max_rows=agent.env.max_rows,
        max_cols=agent.env.max_cols,
        max_actions=agent.env.max_actions,
        reward_config=agent.env.reward_config,
        invalid_action_mode=agent.env.invalid_action_mode,
        episode_mode=agent.env.episode_mode,
    )
    rollout_path = Path(rollout_dir) if rollout_dir else None
    if rollout_path is not None:
        rollout_path.mkdir(parents=True, exist_ok=True)

    rewards: list[float] = []
    lengths: list[int] = []
    failures: list[dict[str, object]] = []
    successes = invalid_actions = dead_ends = truncations = total_action_steps = 0
    covered_fractions: list[float] = []
    placed_fractions: list[float] = []
    legal_action_counts: list[int] = []

    for episode in range(episodes):
        puzzle = puzzles[episode % len(puzzles)]
        seed = seeds[episode % len(seeds)] if seeds else puzzle.seed
        obs, info = eval_env.reset(options={"puzzle": puzzle})
        actions: list[dict[str, object] | None] = []
        step_rewards: list[float] = []
        action_masks: list[list[bool]] = []
        episode_reward = 0.0
        episode_length = 0

        while True:
            action_masks.append([bool(value) for value in info["action_mask"]])
            legal_action_counts.append(int(info["action_count"]))
            action = agent.predict(
                obs,
                deterministic=deterministic,
                action_mask=info["action_mask"],
            )
            selected = eval_env.action_for_index(action)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            actions.append(_action_record(selected))
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
                covered_fractions.append(info["covered_count"] / max(1, info["total_cells"]))
                placed_fractions.append(info["placed_count"] / max(1, info["clue_count"]))
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
                    "placed_count": info["placed_count"],
                    "covered_count": info["covered_count"],
                    "total_cells": info["total_cells"],
                }
                if not info["success"]:
                    failure = {
                        "episode": episode,
                        "seed": seed,
                        "reason": terminal_reason,
                        "placed_count": info["placed_count"],
                        "covered_count": info["covered_count"],
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
        "mean_covered_fraction": sum(covered_fractions) / len(covered_fractions),
        "mean_placed_fraction": sum(placed_fractions) / len(placed_fractions),
        "mean_legal_action_count": sum(legal_action_counts) / max(1, len(legal_action_counts)),
        "failures": failures,
    }


def _action_record(action) -> dict[str, object] | None:
    if action is None:
        return None
    rect = action.rect
    return {
        "clue_id": action.clue_id,
        "rect": {
            "top": rect.top,
            "left": rect.left,
            "height": rect.height,
            "width": rect.width,
        },
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
