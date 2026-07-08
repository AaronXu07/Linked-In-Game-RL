from patches.simulation.generator import generate_puzzle
from patches.simulation.puzzle import Clue, Puzzle, puzzle_from_json, puzzle_to_json, save_puzzle, load_puzzle
from patches.simulation.renderer import render_ansi, render_image
from patches.simulation.simulator import new_game, state_from_solution
from patches.simulation.utils import Rect, ShapeType


def sample_puzzle() -> Puzzle:
    # 2x2 patches match SQUARE and FREE; using those keeps the stored solution valid.
    clues = [
        Clue(id=0, pos=(0, 0), number=4, shape=ShapeType.SQUARE),
        Clue(id=1, pos=(0, 2), number=4, shape=ShapeType.SQUARE),
        Clue(id=2, pos=(2, 0), number=4, shape=ShapeType.SQUARE),
        Clue(id=3, pos=(2, 2), number=4, shape=ShapeType.FREE),
    ]
    solution = [
        Rect(0, 0, 2, 2),
        Rect(0, 2, 2, 2),
        Rect(2, 0, 2, 2),
        Rect(2, 2, 2, 2),
    ]
    return Puzzle(rows=4, cols=4, clues=clues, solution=solution, difficulty="test", seed=1)


def test_json_round_trip() -> None:
    puzzle = sample_puzzle()
    restored = puzzle_from_json(puzzle_to_json(puzzle))
    assert restored.to_dict() == puzzle.to_dict()


def test_save_and_load(tmp_path) -> None:
    puzzle = sample_puzzle()
    path = tmp_path / "puzzle.json"
    save_puzzle(puzzle, path)
    loaded = load_puzzle(path)
    assert loaded.to_dict() == puzzle.to_dict()


def test_generated_puzzle_round_trips() -> None:
    puzzle = generate_puzzle("super_easy", seed=99)
    restored = puzzle_from_json(puzzle_to_json(puzzle))
    assert restored.to_dict() == puzzle.to_dict()


def test_render_ansi_contains_clue_tokens() -> None:
    puzzle = sample_puzzle()
    text = render_ansi(puzzle, new_game(puzzle))
    assert "4S" in text
    assert "covered 0/16" in text


def test_render_image_produces_ppm() -> None:
    puzzle = sample_puzzle()
    state = state_from_solution(puzzle)
    data = render_image(puzzle, state)
    assert data.startswith(b"P6\n")
