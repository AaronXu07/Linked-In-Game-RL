from zip.simulation.puzzle import Puzzle
from zip.simulation.renderer import render_ansi, render_image
from zip.simulation.simulator import new_game
from zip.simulation.ui import BoardLayout, ZipGameUI


def test_ansi_renderer_shows_waypoints_and_status() -> None:
    puzzle = Puzzle(rows=2, cols=2, waypoints=((0, 0), (1, 0)))
    state = new_game(puzzle)

    output = render_ansi(puzzle, state)

    assert "@" in output
    assert "2" in output
    assert "visited 1/4" in output


def test_image_renderer_returns_ppm_bytes() -> None:
    puzzle = Puzzle(rows=2, cols=2, waypoints=((0, 0), (1, 0)))
    image = render_image(puzzle, new_game(puzzle))

    assert image.startswith(b"P6\n")


def test_board_layout_pixel_to_cell() -> None:
    layout = BoardLayout(rows=3, cols=4, cell_size=20, origin_x=10, origin_y=30)

    assert layout.pixel_to_cell(10, 30) == (0, 0)
    assert layout.pixel_to_cell(29, 49) == (0, 0)
    assert layout.pixel_to_cell(30, 50) == (1, 1)
    assert layout.pixel_to_cell(89, 89) == (2, 3)
    assert layout.pixel_to_cell(90, 89) is None
    assert layout.pixel_to_cell(9, 30) is None
    assert layout.pixel_to_cell(10, 90) is None


def test_ui_dragging_back_to_tail_trims_path() -> None:
    puzzle = Puzzle(rows=2, cols=2, waypoints=((0, 0), (1, 0)))
    ui = ZipGameUI(puzzle)

    ui._try_move_to_cell((0, 1))
    ui._try_move_to_cell((1, 1))
    ui._try_move_to_cell((0, 1))

    assert ui.state.path == [(0, 0), (0, 1)]
    assert ui.state.current_pos == (0, 1)
    assert ui.state.visited == {(0, 0), (0, 1)}


def test_ui_dragging_over_older_tail_does_not_trim_whole_path() -> None:
    puzzle = Puzzle(rows=2, cols=2, waypoints=((0, 0), (1, 0)))
    ui = ZipGameUI(puzzle)

    ui._try_move_to_cell((0, 1))
    ui._try_move_to_cell((1, 1))
    ui._try_move_to_cell((0, 0))

    assert ui.state.path == [(0, 0), (0, 1), (1, 1)]
    assert ui.state.current_pos == (1, 1)


def test_ui_path_head_darkens_as_path_grows() -> None:
    puzzle = Puzzle(rows=3, cols=3, waypoints=((0, 0), (2, 2)))
    ui = ZipGameUI(puzzle)

    ui._try_move_to_cell((0, 1))
    short_head_color = ui._path_color(ui._path_fill())

    ui._try_move_to_cell((0, 2))
    ui._try_move_to_cell((1, 2))
    long_head_color = ui._path_color(ui._path_fill())

    assert long_head_color[0] < short_head_color[0]
    assert long_head_color[1] < short_head_color[1]
