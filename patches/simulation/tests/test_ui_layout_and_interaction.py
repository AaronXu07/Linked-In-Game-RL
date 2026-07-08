from patches.simulation.puzzle import Clue, Puzzle
from patches.simulation.ui import BoardLayout, PatchesGameUI
from patches.simulation.utils import Rect, ShapeType


def four_patch_puzzle() -> Puzzle:
    clues = [
        Clue(id=0, pos=(0, 0), number=4, shape=ShapeType.SQUARE),
        Clue(id=1, pos=(0, 2), number=4, shape=ShapeType.SQUARE),
        Clue(id=2, pos=(2, 0), number=4, shape=ShapeType.SQUARE),
        Clue(id=3, pos=(2, 2), number=4, shape=ShapeType.SQUARE),
    ]
    solution = [Rect(0, 0, 2, 2), Rect(0, 2, 2, 2), Rect(2, 0, 2, 2), Rect(2, 2, 2, 2)]
    return Puzzle(rows=4, cols=4, clues=clues, solution=solution)


def test_board_layout_pixel_to_cell() -> None:
    layout = BoardLayout(rows=3, cols=4, cell_size=20, origin_x=10, origin_y=30)

    assert layout.pixel_to_cell(10, 30) == (0, 0)
    assert layout.pixel_to_cell(29, 49) == (0, 0)
    assert layout.pixel_to_cell(30, 50) == (1, 1)
    assert layout.pixel_to_cell(89, 89) == (2, 3)
    assert layout.pixel_to_cell(90, 89) is None
    assert layout.pixel_to_cell(9, 30) is None


def test_board_layout_clamp_and_region() -> None:
    layout = BoardLayout(rows=3, cols=4, cell_size=20, origin_x=10, origin_y=30)

    # Points outside the board clamp back to the nearest cell.
    assert layout.clamp_to_cell(-100, -100) == (0, 0)
    assert layout.clamp_to_cell(10_000, 10_000) == (2, 3)

    x, y, w, h = layout.region_rect(Rect(1, 1, 2, 2))
    assert (x, y, w, h) == (10 + 20, 30 + 20, 40, 40)


def test_place_clue_places_patch() -> None:
    ui = PatchesGameUI(four_patch_puzzle())
    ui._place_clue(0, Rect(0, 0, 2, 2))

    assert 0 in ui.state.patches
    assert ui.state.patches[0] == Rect(0, 0, 2, 2)
    assert ui.state.covered_count == 4


def test_place_clue_rejects_two_clues() -> None:
    ui = PatchesGameUI(four_patch_puzzle())
    # A box from clue 0 that also swallows clue 1 is invalid.
    ui._place_clue(0, Rect(0, 0, 2, 4))
    assert not ui.state.patches


def test_click_clears_filled_patch() -> None:
    ui = PatchesGameUI(four_patch_puzzle())
    ui._place_clue(0, Rect(0, 0, 2, 2))
    assert ui.state.covered_count == 4

    ui._clear_at((0, 1))  # a non-clue cell owned by patch 0
    assert 0 not in ui.state.patches
    assert ui.state.covered_count == 0


def test_reveal_all_solves_board() -> None:
    ui = PatchesGameUI(four_patch_puzzle())
    ui._reveal_all()
    assert ui.state.success
    assert ui.state.covered_count == ui.puzzle.total_cells

    ui._reveal_back()
    assert not ui.state.success
    assert len(ui.state.patches) == len(ui.puzzle.solution) - 1


def test_preview_reports_validity() -> None:
    ui = PatchesGameUI(four_patch_puzzle())
    ui.dragging = True
    ui.active_clue_id = 0
    ui.drag_anchor = (0, 0)
    ui.drag_cell = (1, 1)  # 2x2 square around clue 0
    rect, clue_id, valid = ui._current_preview()
    assert rect == Rect(0, 0, 2, 2)
    assert clue_id == 0
    assert valid

    ui.drag_cell = (0, 1)  # 1x2, wrong area for a 4-square clue
    _, clue_id, valid = ui._current_preview()
    assert clue_id == 0
    assert not valid


def test_drag_does_not_flip_when_anchored_on_clue() -> None:
    # Clue 3 sits at (2, 2), the top-left corner of its 2x2 patch.
    ui = PatchesGameUI(four_patch_puzzle())
    ui.dragging = True
    ui.active_clue_id = 3
    ui.drag_anchor = (2, 2)
    # Dragging toward the opposite corner grows the patch from the clue.
    ui.drag_cell = (3, 3)
    rect, _, valid = ui._current_preview()
    assert rect == Rect(2, 2, 2, 2)
    assert valid
