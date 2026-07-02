import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("torch")

from zip.training.visual_train import _active_env_count, parse_args


def test_visual_train_parse_args_defaults() -> None:
    args = parse_args([])

    assert args.difficulty == "super_easy"
    assert args.steps_per_second == 6.0
    assert args.warmup_steps == 256


def test_visual_train_parse_args_overrides() -> None:
    args = parse_args(
        [
            "--difficulty",
            "easy",
            "--steps-per-second",
            "12",
            "--total-steps",
            "100",
            "--curriculum",
            "default",
            "--parallel-puzzles",
            "2",
            "--parallel-envs",
            "3",
            "--checkpoint-every",
            "50",
            "--device",
            "cpu",
        ]
    )

    assert args.difficulty == "easy"
    assert args.steps_per_second == 12
    assert args.total_steps == 100
    assert args.curriculum == "default"
    assert args.parallel_puzzles == 2
    assert args.parallel_envs == 3
    assert args.checkpoint_every == 50
    assert args.device == "cpu"


def test_visual_train_parallel_puzzles_imply_visible_envs() -> None:
    args = parse_args(["--parallel-puzzles", "4"])

    assert _active_env_count(args) == 4


def test_visual_train_parallel_envs_override_parallel_puzzles() -> None:
    args = parse_args(["--parallel-puzzles", "4", "--parallel-envs", "2"])

    assert _active_env_count(args) == 2
