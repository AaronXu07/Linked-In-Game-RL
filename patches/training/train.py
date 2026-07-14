"""Command-line DQN training entrypoint for Patches."""

from __future__ import annotations

import argparse
import signal
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import FrameType

from patches.simulation.config import DIFFICULTIES

from .agent import AgentConfig, PatchesAgent
from .callbacks import (
    BestCheckpointCallback,
    CallbackList,
    CheckpointCallback,
    EpisodeJsonlCallback,
    EvaluationCallback,
    TensorBoardCallback,
)
from .curriculum import (
    AsyncPuzzleBuffer,
    CurriculumEvaluationCallback,
    CurriculumManager,
    CurriculumStateCallback,
    DifficultySampler,
    curriculum_from_name,
    evaluation_seeds,
    generate_evaluation_puzzles,
)
from .env import PatchesEnv


class TrainingInterrupted(Exception):
    """Raised when a handled stop signal asks training to save and exit."""


def main() -> None:
    args = parse_args()
    curriculum = curriculum_from_name(
        args.curriculum,
        target_difficulty=args.difficulty,
        seed=args.seed,
    )
    puzzle_sampler = _build_puzzle_sampler(args, curriculum)

    envs = [
        PatchesEnv(
            difficulty=args.difficulty,
            max_rows=args.max_rows,
            max_cols=args.max_cols,
            max_actions=args.max_actions,
            puzzle_sampler=puzzle_sampler,
            invalid_action_mode=args.invalid_action_mode,
            episode_mode=args.episode_mode,
            use_solver_dead_end_check=args.solver_dead_end_check,
        )
        for _ in range(max(1, args.parallel_envs))
    ]
    env = envs[0]
    config = AgentConfig(
        algorithm=args.algorithm,
        learning_rate=args.learning_rate,
        replay_size=args.replay_size,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        train_every=args.train_every,
        target_update_every=args.target_update_every,
        tau=args.tau,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        double_dqn=args.double_dqn,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        device=args.device,
    )
    agent = PatchesAgent(env, config, checkpoint_path=args.checkpoint_path)
    if curriculum is not None:
        saved_curriculum = agent.checkpoint_metadata.get("curriculum")
        if isinstance(saved_curriculum, dict):
            curriculum.load_state_dict(saved_curriculum)

    callbacks = []
    if curriculum is not None:
        callbacks.append(CurriculumStateCallback(curriculum))
    tensorboard = TensorBoardCallback(args.log_dir) if args.log_dir else None
    if tensorboard is not None:
        callbacks.append(tensorboard)
    if args.checkpoint_dir:
        callbacks.append(
            CheckpointCallback(
                args.checkpoint_dir,
                every_steps=args.checkpoint_every,
            )
        )
    if args.episode_jsonl:
        callbacks.append(EpisodeJsonlCallback(args.episode_jsonl))
    if args.checkpoint_dir and args.eval_every > 0 and args.eval_episodes > 0:
        callbacks.append(BestCheckpointCallback(args.checkpoint_dir))
    if args.eval_every > 0 and args.eval_episodes > 0:
        if curriculum is not None:
            callbacks.append(
                CurriculumEvaluationCallback(
                    curriculum,
                    every_steps=args.eval_every,
                    episodes=args.eval_episodes,
                    callbacks=[callback for callback in callbacks if callback is not None],
                )
            )
        else:
            seeds = parse_seed_spec(args.eval_seeds)
            if not seeds:
                seeds = evaluation_seeds(args.difficulty, args.eval_episodes)
            puzzles = generate_evaluation_puzzles(args.difficulty, seeds=seeds)
            callbacks.append(
                EvaluationCallback(
                    puzzles,
                    every_steps=args.eval_every,
                    episodes=args.eval_episodes,
                    callbacks=[callback for callback in callbacks if callback is not None],
                )
            )

    callback_list = CallbackList(callbacks)
    try:
        with _interrupt_handler():
            if len(envs) > 1:
                agent.train_parallel(envs, args.total_steps, callbacks=[callback_list])
            else:
                agent.train(args.total_steps, callbacks=[callback_list])
    except (KeyboardInterrupt, TrainingInterrupted):
        _save_interrupted_checkpoint(agent, args.checkpoint_dir)
    finally:
        for env in envs:
            env.close()
        callback_list.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--difficulty", choices=sorted(DIFFICULTIES), default="super_easy")
    parser.add_argument("--curriculum", default=None)
    parser.add_argument("--parallel-puzzles", type=int, default=0)
    parser.add_argument("--parallel-envs", type=int, default=1)
    parser.add_argument("--puzzle-buffer-size", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-cols", type=int, default=None)
    parser.add_argument("--max-actions", type=int, default=None)
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--algorithm", choices=("candidate_dqn", "flat_candidate_dqn"), default="candidate_dqn")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--replay-size", type=int, default=100_000)
    parser.add_argument("--warmup-steps", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--train-every", type=int, default=4)
    parser.add_argument("--target-update-every", type=int, default=1_000)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=100_000)
    parser.add_argument("--double-dqn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--checkpoint-dir", default="checkpoints/patches_super_easy")
    parser.add_argument("--checkpoint-every", type=int, default=10_000)
    parser.add_argument("--eval-every", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--eval-seeds", default="")
    parser.add_argument("--log-dir", default="runs/patches_super_easy")
    parser.add_argument("--episode-jsonl", default=None)
    parser.add_argument(
        "--invalid-action-mode",
        choices=("terminate", "penalize", "mask_required"),
        default="terminate",
    )
    parser.add_argument(
        "--episode-mode",
        choices=("commit_only", "revision"),
        default="commit_only",
    )
    parser.add_argument("--solver-dead-end-check", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser.parse_args()


def _build_puzzle_sampler(
    args: argparse.Namespace,
    curriculum: CurriculumManager | None,
):
    sampler = curriculum
    if sampler is None and args.parallel_puzzles > 0:
        sampler = DifficultySampler((args.difficulty,), seed=args.seed)
    if sampler is None:
        return None
    if args.parallel_puzzles <= 0:
        return sampler
    return AsyncPuzzleBuffer(
        sampler,
        workers=args.parallel_puzzles,
        buffer_size=args.puzzle_buffer_size,
        seed=args.seed,
    )


@contextmanager
def _interrupt_handler() -> Iterator[None]:
    previous_handlers = {}

    def handle_signal(signum: int, frame: FrameType | None) -> None:
        del frame
        raise TrainingInterrupted(f"received signal {signum}")

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, handle_signal)
    try:
        yield
    finally:
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)


def _save_interrupted_checkpoint(agent: PatchesAgent, checkpoint_dir: str | Path | None) -> None:
    if not checkpoint_dir:
        return
    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    agent.save_checkpoint(directory / "latest.pt")
    agent.save_checkpoint(directory / "interrupted.pt")


def parse_seed_spec(spec: str) -> tuple[int, ...]:
    spec = spec.strip()
    if not spec:
        return ()
    seeds: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        delimiter = ":" if ":" in part else "-" if "-" in part else None
        if delimiter is not None:
            start_text, end_text = part.split(delimiter, 1)
            start = int(start_text)
            end = int(end_text)
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(part))
    return tuple(seeds)


if __name__ == "__main__":
    main()
