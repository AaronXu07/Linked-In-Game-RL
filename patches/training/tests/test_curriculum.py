from patches.training.curriculum import (
    CurriculumEvaluationCallback,
    CurriculumManager,
    CurriculumStage,
    CurriculumStateCallback,
    build_default_curriculum,
    curriculum_from_name,
)


def test_default_curriculum_targets_requested_difficulty() -> None:
    stages = build_default_curriculum("easy")

    assert [stage.name for stage in stages] == [
        "super_easy",
        "super_easy_easy_mix",
        "easy",
    ]
    assert stages[0].min_success_rate == 0.98
    assert stages[1].min_success_rate == 0.90
    assert stages[-1].min_success_rate is None


def test_default_curriculum_scales_to_medium_with_harder_gate() -> None:
    stages = build_default_curriculum("medium")

    assert [stage.name for stage in stages] == [
        "super_easy",
        "super_easy_easy_mix",
        "easy",
        "easy_medium_mix",
        "medium",
    ]
    assert stages[3].difficulties == ("easy", "medium")
    assert stages[3].min_success_rate == 0.80
    assert stages[-1].min_success_rate is None


def test_curriculum_advances_when_success_gate_is_met() -> None:
    manager = CurriculumManager(
        (
            CurriculumStage("first", ("super_easy",), min_success_rate=0.5),
            CurriculumStage("second", ("easy",)),
        ),
        seed=7,
        consecutive_passes_required=1,
    )

    assert not manager.update_from_evaluation({"success_rate": 0.49})
    assert manager.stage_index == 0
    assert manager.sample_difficulty() == "super_easy"

    assert manager.update_from_evaluation({"success_rate": 0.5})
    assert manager.stage_index == 1
    assert manager.current_stage.name == "second"
    assert manager.last_success_rate == 0.5
    assert manager.version == 1


def test_curriculum_requires_consecutive_passes() -> None:
    manager = CurriculumManager(
        (
            CurriculumStage("first", ("super_easy",), min_success_rate=0.5),
            CurriculumStage("second", ("easy",)),
        ),
        seed=7,
        consecutive_passes_required=3,
    )

    # Pass 1 — not enough yet.
    assert not manager.update_from_evaluation({"success_rate": 0.6})
    assert manager.stage_index == 0
    assert manager._consecutive_passes == 1

    # Pass 2 — still not enough.
    assert not manager.update_from_evaluation({"success_rate": 0.7})
    assert manager.stage_index == 0
    assert manager._consecutive_passes == 2

    # A failure resets the streak.
    assert not manager.update_from_evaluation({"success_rate": 0.4})
    assert manager.stage_index == 0
    assert manager._consecutive_passes == 0

    # Pass 1 again.
    assert not manager.update_from_evaluation({"success_rate": 0.5})
    assert manager._consecutive_passes == 1

    # Pass 2.
    assert not manager.update_from_evaluation({"success_rate": 0.8})
    assert manager._consecutive_passes == 2

    # Pass 3 — promotes!
    assert manager.update_from_evaluation({"success_rate": 0.9})
    assert manager.stage_index == 1
    assert manager._consecutive_passes == 0


def test_curriculum_can_restore_stage_state() -> None:
    manager = curriculum_from_name("default", target_difficulty="medium", seed=1)
    assert manager is not None

    manager.load_state_dict({"stage_index": 2, "last_success_rate": 0.8})

    assert manager.stage_index == 2
    assert manager.current_stage.name == "easy"
    assert manager.last_success_rate == 0.8


def test_zip_alias_is_accepted_for_parity() -> None:
    manager = curriculum_from_name("zip", target_difficulty="easy", seed=1)

    assert manager is not None
    assert [stage.name for stage in manager.stages] == [
        "super_easy",
        "super_easy_easy_mix",
        "easy",
    ]


def test_curriculum_state_callback_adds_training_metrics_and_metadata() -> None:
    manager = curriculum_from_name("default", target_difficulty="easy", seed=1)
    assert manager is not None
    callback = CurriculumStateCallback(manager)

    class FakeAgent:
        checkpoint_metadata: dict[str, object] = {}

    payload: dict[str, object] = {}
    agent = FakeAgent()
    callback(agent, "step", payload)

    assert payload["curriculum_stage_index"] == 0.0
    assert payload["curriculum_grid_cells"] == 25.0
    assert payload["curriculum_success_gate"] == 0.98
    assert agent.checkpoint_metadata["curriculum"] == manager.state_dict()


def test_curriculum_evaluation_callback_advances_and_emits_metrics() -> None:
    manager = CurriculumManager(
        (
            CurriculumStage(
                "first",
                ("super_easy",),
                min_success_rate=0.5,
                eval_difficulty="super_easy",
            ),
            CurriculumStage("second", ("easy",), eval_difficulty="easy"),
        ),
        seed=7,
        consecutive_passes_required=1,
    )
    events = []

    class FakeAgent:
        total_steps = 4

        def __init__(self) -> None:
            self.checkpoint_metadata: dict[str, object] = {}

        def evaluate(self, puzzles, episodes, deterministic=True):
            assert puzzles == ("cached-puzzle",)
            assert episodes == 3
            assert deterministic
            return {"success_rate": 0.75, "mean_reward": 1.0}

    agent = FakeAgent()
    callback = CurriculumEvaluationCallback(
        manager,
        every_steps=4,
        episodes=3,
        callbacks=[lambda _agent, event, payload: events.append((event, payload))],
    )
    callback._puzzle_cache["super_easy"] = ("cached-puzzle",)

    callback(agent, "step", {})

    assert manager.stage_index == 1
    assert agent.checkpoint_metadata["curriculum"] == manager.state_dict()
    assert events
    event, payload = events[0]
    assert event == "eval"
    assert payload["success_rate"] == 0.75
    assert payload["curriculum_advanced"] == 1.0
    assert payload["curriculum_eval_difficulty"] == "super_easy"
    assert payload["curriculum_stage"] == "first"
