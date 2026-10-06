import matplotlib

matplotlib.use("Agg")  # headless for tests -- no display needed

import numpy as np

from curve_guesser.board import DrawingBoard


def make_board():
    return DrawingBoard()


def test_eraser_starts_off():
    board = make_board()
    assert board.mode == "draw"


def test_toggle_eraser_switches_mode():
    board = make_board()
    board.toggle_eraser()
    assert board.mode == "erase"
    board.toggle_eraser()
    assert board.mode == "draw"


def test_erase_removes_whole_short_stroke():
    board = make_board()
    x = np.linspace(0, 0.1, 5)
    y = np.zeros(5)
    board.strokes = [(x, y)]
    board.erase_at(0.05, 0.0)
    assert board.strokes == []


def test_erase_leaves_untouched_strokes_alone():
    board = make_board()
    x = np.linspace(-5, -4, 20)
    y = np.zeros(20)
    board.strokes = [(x, y)]
    board.erase_at(5.0, 5.0)
    assert len(board.strokes) == 1
    np.testing.assert_allclose(board.strokes[0][0], x)


def test_erase_splits_stroke_into_two():
    board = make_board()
    x = np.linspace(-5, 5, 200)
    y = np.zeros(200)
    board.strokes = [(x, y)]
    board.erase_at(0.0, 0.0)
    assert len(board.strokes) == 2
    left, right = board.strokes
    assert left[0].max() < 0
    assert right[0].min() > 0


def test_erase_drops_fragments_too_short_to_be_a_stroke():
    board = make_board()
    x = np.array([0.0, board.eraser_radius * 0.5, 5.0, 5.1])
    y = np.array([0.0, 0.0, 0.0, 0.0])
    board.strokes = [(x, y)]
    board.erase_at(0.0, 0.0)
    assert len(board.strokes) == 1
    np.testing.assert_allclose(board.strokes[0][0], [5.0, 5.1])


def test_erase_at_updates_line_artists_count():
    board = make_board()
    x = np.linspace(-5, 5, 200)
    y = np.zeros(200)
    board.strokes = [(x, y)]
    board._redraw_strokes()
    assert len(board.line_artists) == 1
    board.erase_at(0.0, 0.0)
    assert len(board.line_artists) == 2
