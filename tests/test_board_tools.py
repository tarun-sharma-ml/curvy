import matplotlib

matplotlib.use("Agg")  # headless for tests -- no display needed

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from curve_guesser.board import DrawingBoard


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


class Sim:
    """Drives a DrawingBoard with fake mouse/keyboard events."""

    def __init__(self):
        self.b = DrawingBoard()

    def ev(self, x, y, button=1, key=None, dbl=False):
        return SimpleNamespace(inaxes=self.b.ax, button=button, xdata=x, ydata=y,
                               key=key, dblclick=dbl)

    def drag(self, pts, key=None):
        b = self.b
        b.on_press(self.ev(*pts[0], key=key))
        for p in pts[1:]:
            b.on_move(self.ev(*p, key=key))
        b.on_release(self.ev(*pts[-1], key=key))

    def click(self, x, y, **kw):
        self.b.on_press(self.ev(x, y, **kw))
        self.b.on_release(self.ev(x, y, **kw))

    def key(self, k):
        self.b.on_key(SimpleNamespace(key=k))


def test_pen_stroke_is_committed_and_guessed():
    s = Sim()
    t = np.linspace(0, 2 * np.pi, 60)
    s.drag(list(zip(3 + 2 * np.cos(t), -2 + 2 * np.sin(t))))
    assert len(s.b.strokes) == 1
    assert "CIRCLE" in s.b.info
    assert "CIRCLE" in s.b.ax.get_title()


def test_pen_scribble_is_not_called_an_excellent_match():
    s = Sim()
    rng = np.random.default_rng(0)
    x, y = rng.uniform(-3, 3, 60), rng.uniform(-3, 3, 60)
    order = np.argsort(np.arctan2(y, x))
    s.drag(list(zip(x[order], y[order])))
    assert "freeform" in s.b.info or "rough match" in s.b.info
    assert "excellent" not in s.b.info and "good match" not in s.b.info


def test_rectangle_makes_four_strokes_in_one_object():
    s = Sim()
    s.key("r")
    s.drag([(0, 0), (2, 1), (4, 3)])
    assert len(s.b.strokes) == 4
    assert len(set(s.b.stroke_gids)) == 1
    assert "Rectangle 4.00 x 3.00" in s.b.info


def test_shift_makes_square_and_circle():
    s = Sim()
    s.key("r")
    s.drag([(0, 0), (4, 2)], key="shift")
    assert "Square" in s.b.info
    s.key("v")
    s.drag([(0, 0), (6, 2)], key="shift")
    assert s.b.info.startswith("Circle")


def test_circle_tool_drags_from_centre():
    s = Sim()
    s.key("o")
    s.drag([(2, 1), (5, 5)])
    assert "(x - 2.00)^2 + (y - 1.00)^2 = 5.00^2" in s.b.info
    xs, ys = s.b.strokes[0]
    assert np.allclose(np.hypot(xs - 2, ys - 1), 5)


def test_line_tool_and_shift_constrain():
    s = Sim()
    s.key("l")
    s.drag([(0, 0), (8, 1)], key="shift")
    xs, ys = s.b.strokes[0]
    assert np.allclose(ys, 0)


def test_degenerate_drag_adds_nothing():
    s = Sim()
    s.key("r")
    s.drag([(1, 1), (1, 1)])
    assert s.b.strokes == []


def test_snap_to_grid_rounds_shape_corners():
    s = Sim()
    s.key("s")
    assert s.b.snap
    s.key("r")
    s.drag([(0.2, 0.3), (3.6, 2.4)])
    xs = np.concatenate([st[0] for st in s.b.strokes])
    ys = np.concatenate([st[1] for st in s.b.strokes])
    assert xs.min() == 0 and xs.max() == 4 and ys.min() == 0 and ys.max() == 2


def test_polygon_close_by_double_click():
    s = Sim()
    s.key("g")
    s.click(0, 0)
    s.click(4, 0)
    s.click(2, 3)
    s.click(2, 3, dbl=True)
    assert len(s.b.strokes) == 3
    assert s.b.info.startswith("Triangle")
    assert s.b.poly_pts == []


def test_polygon_close_by_clicking_first_point():
    s = Sim()
    s.key("g")
    for p in [(0, 0), (4, 0), (4, 4), (0, 4)]:
        s.click(*p)
    s.click(0.1, 0.1)
    assert len(s.b.strokes) == 4
    assert s.b.info.startswith("Quadrilateral")


def test_polygon_right_click_finishes_open_and_escape_cancels():
    s = Sim()
    s.key("g")
    s.click(0, 0)
    s.click(3, 0)
    s.click(3, 3)
    s.b.on_press(s.ev(3, 3, button=3))
    assert len(s.b.strokes) == 2 and s.b.info.startswith("Polyline")
    s.click(-5, -5)
    s.click(-4, -5)
    s.key("escape")
    assert s.b.poly_pts == [] and len(s.b.strokes) == 2


def test_undo_redo_roundtrip():
    s = Sim()
    s.key("r")
    s.drag([(0, 0), (4, 3)])
    s.key("l")
    s.drag([(5, 5), (8, 8)])
    assert len(s.b.strokes) == 5
    s.key("ctrl+z")
    assert len(s.b.strokes) == 4
    s.key("ctrl+z")
    assert len(s.b.strokes) == 0
    s.key("ctrl+y")
    assert len(s.b.strokes) == 4
    s.key("ctrl+y")
    assert len(s.b.strokes) == 5
    s.key("ctrl+y")  # nothing left to redo
    assert len(s.b.strokes) == 5


def test_new_action_clears_redo():
    s = Sim()
    s.key("l")
    s.drag([(0, 0), (4, 0)])
    s.key("ctrl+z")
    s.drag([(1, 1), (2, 2)])
    s.key("ctrl+y")
    assert len(s.b.strokes) == 1


def test_clear_is_undoable():
    s = Sim()
    s.key("r")
    s.drag([(0, 0), (4, 3)])
    s.key("c")
    assert s.b.strokes == []
    s.key("ctrl+z")
    assert len(s.b.strokes) == 4


def test_move_tool_moves_whole_object_and_undoes():
    s = Sim()
    s.key("r")
    s.drag([(0, 0), (4, 2)])
    before = [(x.copy(), y.copy()) for x, y in s.b.strokes]
    s.key("m")
    s.drag([(2, 0), (5, 3)])  # grab the bottom edge, drag by (3, 3)
    for (bx, by), (ax, ay) in zip(before, s.b.strokes):
        assert np.allclose(ax, bx + 3) and np.allclose(ay, by + 3)
    s.key("ctrl+z")
    for (bx, by), (ax, ay) in zip(before, s.b.strokes):
        assert np.allclose(ax, bx) and np.allclose(ay, by)


def test_move_tool_missing_the_object_does_nothing():
    s = Sim()
    s.key("l")
    s.drag([(0, 0), (4, 0)])
    s.key("m")
    s.drag([(8, 8), (9, 9)])
    assert s.b.selected_gid is None
    assert np.allclose(s.b.strokes[0][0][0], 0)


def test_delete_selected_object():
    s = Sim()
    s.key("r")
    s.drag([(0, 0), (4, 2)])
    s.key("m")
    s.click(2, 0)
    assert s.b.selected_gid is not None
    s.key("delete")
    assert s.b.strokes == []
    s.key("ctrl+z")
    assert len(s.b.strokes) == 4


def test_eraser_keeps_object_membership_and_is_one_undo_step():
    s = Sim()
    s.key("l")
    s.drag([(-8, 0), (8, 0)])
    gid = s.b.stroke_gids[0]
    s.key("e")
    assert s.b.mode == "erase"
    s.drag([(0, 0), (0.05, 0)])
    assert len(s.b.strokes) == 2
    assert s.b.stroke_gids == [gid, gid]
    s.key("ctrl+z")
    assert len(s.b.strokes) == 1


def test_eraser_that_hits_nothing_leaves_no_undo_step():
    s = Sim()
    s.key("l")
    s.drag([(-8, 0), (-6, 0)])
    n = len(s.b.undo_stack)
    s.key("e")
    s.click(8, 8)
    assert len(s.b.undo_stack) == n


def test_toggle_eraser_returns_to_previous_tool():
    s = Sim()
    s.key("r")
    s.key("e")
    assert s.b.tool == "erase"
    s.key("e")
    assert s.b.tool == "rect"


def test_switching_tool_cancels_half_finished_polygon():
    s = Sim()
    s.key("g")
    s.click(0, 0)
    s.click(2, 0)
    s.key("p")
    assert s.b.poly_pts == [] and s.b.strokes == []


def test_shapes_feed_the_fitting_pipeline():
    from curve_guesser.pieces import fit_stroke_pieces

    s = Sim()
    s.key("r")
    s.drag([(0, 0), (4, 2)])
    for x, y in s.b.strokes:
        pieces = fit_stroke_pieces(x, y, 1)
        assert pieces[0]["kind"] == "line"


def test_matplotlib_shortcuts_restored_on_close():
    import matplotlib as mpl

    before = list(mpl.rcParams["keymap.yscale"])
    b = DrawingBoard()
    assert "l" not in mpl.rcParams["keymap.yscale"]
    b.finish()
    assert mpl.rcParams["keymap.yscale"] == before or "l" in mpl.rcParams["keymap.yscale"]