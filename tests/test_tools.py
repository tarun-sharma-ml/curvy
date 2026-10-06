import numpy as np

from curve_guesser import tools as T


def test_snap_point_rounds_to_grid():
    assert T.snap_point(1.4, -2.6) == (1.0, -3.0)
    assert T.snap_point(0.5, 0.49) == (1.0, 0.0)
    assert T.snap_point(1.3, 2.7, step=0.5) == (1.5, 2.5)


def test_constrain_angle_snaps_to_15_degrees():
    x, y = T.constrain_angle((0, 0), (10, 1.0))  # ~5.7 deg -> 0 deg
    assert abs(y) < 1e-9 and x > 9
    x, y = T.constrain_angle((0, 0), (5, 5.5))   # ~47.7 deg -> 45 deg
    assert abs(x - y) < 1e-9


def test_constrain_angle_zero_length():
    assert T.constrain_angle((1, 1), (1, 1)) == (1, 1)


def test_box_from_drag_corner_and_negative_drag():
    assert T.box_from_drag((0, 0), (4, 2)) == (0, 0, 4, 2)
    assert T.box_from_drag((4, 2), (0, 0)) == (0, 0, 4, 2)


def test_box_from_drag_square_and_center():
    assert T.box_from_drag((0, 0), (4, 2), square=True) == (0, 0, 4, 4)
    assert T.box_from_drag((0, 0), (-4, 2), square=True) == (-4, 0, 0, 4)
    assert T.box_from_drag((1, 1), (3, 2), from_center=True) == (-1, 0, 3, 2)


def test_rectangle_strokes_are_four_straight_sides():
    strokes = T.rectangle_strokes(0, 0, 4, 2)
    assert len(strokes) == 4
    # consecutive sides connect end-to-start, and the last closes the loop
    for a, b in zip(strokes, strokes[1:] + strokes[:1]):
        assert (a[0][-1], a[1][-1]) == (b[0][0], b[1][0])


def test_ellipse_stroke_is_closed_and_on_curve():
    xs, ys = T.ellipse_stroke(1, 2, 3, 1.5)
    assert np.isclose(xs[0], xs[-1]) and np.isclose(ys[0], ys[-1])
    assert np.allclose(((xs - 1) / 3) ** 2 + ((ys - 2) / 1.5) ** 2, 1)


def test_ellipse_from_box():
    assert T.ellipse_from_box(0, 0, 6, 2) == (3, 1, 3, 1)


def test_polygon_strokes_closed_vs_open():
    tri = [(0, 0), (4, 0), (2, 3)]
    assert len(T.polygon_strokes(tri, closed=True)) == 3
    assert len(T.polygon_strokes(tri, closed=False)) == 2


def test_distance_to_stroke_perpendicular_and_endpoint():
    xs, ys = np.array([0.0, 10.0]), np.array([0.0, 0.0])
    assert np.isclose(T.distance_to_stroke(5, 3, xs, ys), 3)
    assert np.isclose(T.distance_to_stroke(13, 4, xs, ys), 5)  # past the end


def test_nearest_stroke_picks_closest_and_handles_empty():
    assert T.nearest_stroke([], 0, 0) is None
    a = (np.array([0.0, 1.0]), np.array([0.0, 0.0]))
    b = (np.array([0.0, 1.0]), np.array([5.0, 5.0]))
    idx, d = T.nearest_stroke([a, b], 0.5, 4.0)
    assert idx == 1 and np.isclose(d, 1.0)


def test_descriptions_fill_in_numbers():
    assert "y = 0.50x + 1.00" in T.describe_line((0, 1), (2, 2))
    assert "vertical" in T.describe_line((2, 0), (2, 5))
    assert T.describe_circle(0, 0, 3) == "Circle: x^2 + y^2 = 3.00^2"
    assert "(x - 2.00)^2 + (y + 1.00)^2 = 3.00^2" in T.describe_circle(2, -1, 3)
    assert "Ellipse" in T.describe_ellipse(0, 0, 4, 2)
    assert T.describe_ellipse(0, 0, 3, 3).startswith("Circle")
    assert T.describe_rectangle(0, 0, 2, 2).startswith("Square")
    assert T.describe_rectangle(0, 0, 4, 2).startswith("Rectangle")
    assert T.describe_polygon([(0, 0), (1, 0), (0, 1)]).startswith("Triangle")