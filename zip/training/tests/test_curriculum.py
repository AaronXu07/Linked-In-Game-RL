from zip.training.curriculum import (
    CurriculumManager,
    CurriculumStage,
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
    assert stages[-1].min_success_rate is None


def test_curriculum_advances_when_success_gate_is_met() -> None:
    manager = CurriculumManager(
        (
            CurriculumStage("first", ("super_easy",), min_success_rate=0.5),
            CurriculumStage("second", ("easy",)),
        ),
        seed=7,
    )

    assert not manager.update_from_evaluation({"success_rate": 0.49})
    assert manager.stage_index == 0

    assert manager.update_from_evaluation({"success_rate": 0.5})
    assert manager.stage_index == 1
    assert manager.current_stage.name == "second"


def test_curriculum_can_restore_stage_state() -> None:
    manager = curriculum_from_name("default", target_difficulty="medium", seed=1)
    assert manager is not None

    manager.load_state_dict({"stage_index": 2, "last_success_rate": 0.8})

    assert manager.stage_index == 2
    assert manager.last_success_rate == 0.8
