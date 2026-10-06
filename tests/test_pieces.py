import numpy as np

from curve_guesser.pieces import (
    resample_by_arclength,
    split_stroke,
    fit_piece,
    fit_stroke_pieces,
    evaluate_piece,
    to_desmos_string,
    group_pieces,
    group_to_desmos,
    to_desmos_animated,
)


def test_resample_by_arclength_basic():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 0.0, 0.0])
    xu, yu = resample_by_arclength(x, y, 5)
    assert len(xu) == 5
    np.testing.assert_allclose(xu, [0, 0.5, 1.0, 1.5, 2.0])


def test_split_stroke_piece_count():
    x = np.linspace(0, 10, 200)
    y = np.zeros_like(x)
    pieces = split_stroke(x, y, num_pieces=4, points_per_piece=10)
    assert len(pieces) == 4
    for px, py in pieces:
        assert len(px) == 11  # points_per_piece + 1 shared boundary


def test_fit_piece_recovers_line():
    x = np.linspace(0, 5, 20)
    y = 2 * x + 1
    piece = fit_piece(x, y)
    assert piece["kind"] == "line"
    assert piece["error"] < 1e-3
    a, b = piece["params"]
    assert abs(a - 2) < 0.05
    assert abs(b - 1) < 0.05


def test_fit_piece_recovers_circle_arc():
    theta = np.linspace(0, np.pi / 2, 30)  # quarter circle, not line-like
    x = 3 * np.cos(theta) + 1
    y = 3 * np.sin(theta) - 2
    piece = fit_piece(x, y)
    assert piece["kind"] == "circle"
    assert piece["error"] < 1e-2
    h, k, r, _sign = piece["params"]
    assert abs(h - 1) < 0.1
    assert abs(k - (-2)) < 0.1
    assert abs(r - 3) < 0.1


def test_fit_piece_prefers_line_when_nearly_straight():
    x = np.linspace(0, 5, 20)
    y = 2 * x + 1 + 1e-4 * x**2
    piece = fit_piece(x, y)
    assert piece["kind"] == "line"


def test_fit_piece_handles_vertical_segment():
    y = np.linspace(0, 5, 20)
    x = np.full_like(y, 3.0) + 1e-6 * y
    piece = fit_piece(x, y)
    assert piece["orientation"] == "y"


def test_fit_stroke_pieces_covers_whole_stroke():
    x = np.linspace(0, 10, 300)
    y = np.sin(x)
    pieces = fit_stroke_pieces(x, y, num_pieces=6)
    assert len(pieces) == 6
    for p in pieces:
        assert p["error"] < 0.5  # loose sanity bound


def test_evaluate_piece_matches_domain():
    x = np.linspace(0, 5, 20)
    y = 2 * x + 1
    piece = fit_piece(x, y)
    px, py = evaluate_piece(piece, n=10)
    assert len(px) == 10


def test_to_desmos_string_line_format():
    x = np.linspace(0, 5, 20)
    y = 2 * x + 1
    piece = fit_piece(x, y)
    s = to_desmos_string(piece)
    assert s.startswith("y=")
    assert "\\le x\\le" in s


def test_to_desmos_string_circle_format():
    theta = np.linspace(0, np.pi / 2, 30)
    x = 3 * np.cos(theta)
    y = 3 * np.sin(theta)
    piece = fit_piece(x, y)
    s = to_desmos_string(piece)
    assert "\\sqrt" in s


def test_to_desmos_string_circle_uses_correct_center_signs():
    piece = {
        "kind": "circle",
        "orientation": "x",
        "params": (3.0, -2.0, 5.0, 1.0),
        "domain": (0.0, 5.0),
    }
    s = to_desmos_string(piece)
    assert "(x-3.0000)^2" in s
    assert "-2.0000+" in s


def test_to_desmos_string_negative_center_no_double_sign():
    theta = np.linspace(np.pi, 3 * np.pi / 2, 30)
    x = 3 * np.cos(theta) - 5
    y = 3 * np.sin(theta) - 5
    piece = fit_piece(x, y)
    s = to_desmos_string(piece)
    assert "--" not in s
    assert "-+" not in s
    assert "+-" not in s


def test_group_pieces_merges_monotonic_line_sequence():
    x = np.linspace(0, 10, 300)
    y = 2 * x + 1
    piece_list = fit_stroke_pieces(x, y, num_pieces=5)
    groups = group_pieces(piece_list)
    assert len(groups) == 1
    assert len(groups[0]["pieces"]) == 5


def test_group_pieces_splits_on_loopback():
    theta = np.linspace(0, 2 * np.pi, 300)
    x = np.cos(theta) * 5
    y = np.sin(theta) * 5
    piece_list = fit_stroke_pieces(x, y, num_pieces=8)
    groups = group_pieces(piece_list)
    assert len(groups) > 1


def test_group_to_desmos_multi_piece_uses_named_piecewise():
    x = np.linspace(0, 10, 300)
    y = 2 * x + 1
    piece_list = fit_stroke_pieces(x, y, num_pieces=4)
    groups = group_pieces(piece_list)
    eq = group_to_desmos(groups[0], "f_{1a}")
    assert eq.startswith("f_{1a}(x)=")
    assert "\\le x:" in eq or "x\\le" in eq
    assert eq.count(",") == len(groups[0]["pieces"]) - 1


def test_group_to_desmos_has_overall_domain_restriction():
    x = np.linspace(0, 10, 300)
    y = 2 * x + 1
    piece_list = fit_stroke_pieces(x, y, num_pieces=4)
    groups = group_pieces(piece_list)
    eq = group_to_desmos(groups[0], "f_{1a}")
    assert eq.count("\\left\\{") == 2
    assert eq.count("\\right\\}") == 2
    lo, hi = groups[0]["lo"], groups[0]["hi"]
    assert f"{lo:.4f}" in eq
    assert f"{hi:.4f}" in eq


def test_group_to_desmos_single_piece_falls_back_to_plain_equation():
    theta = np.linspace(0, np.pi / 2, 30)
    x = 3 * np.cos(theta)
    y = 3 * np.sin(theta)
    piece_list = [fit_piece(x, y)]
    groups = group_pieces(piece_list)
    eq = group_to_desmos(groups[0], "f_{1a}")
    assert eq.startswith("y=") or eq.startswith("x=")


def test_adjacent_pieces_meet_exactly_for_wavy_line():
    x = np.linspace(0, 10, 400)
    y = np.sin(x) + 0.1 * x
    pieces = fit_stroke_pieces(x, y, num_pieces=8)
    for i in range(len(pieces) - 1):
        p_a, p_b = pieces[i], pieces[i + 1]
        end_x, end_y = evaluate_piece(p_a, n=50)
        start_x, start_y = evaluate_piece(p_b, n=50)
        gap = np.hypot(end_x[-1] - start_x[0], end_y[-1] - start_y[0])
        assert gap < 1e-6, f"gap of {gap} between piece {i} and {i+1}"


def test_adjacent_pieces_nearly_meet_for_circular_shape():
    theta = np.linspace(0, 2 * np.pi, 300)
    x = np.cos(theta) * 5
    y = np.sin(theta) * 5
    pieces = fit_stroke_pieces(x, y, num_pieces=8)
    for i in range(len(pieces) - 1):
        p_a, p_b = pieces[i], pieces[i + 1]
        end_x, end_y = evaluate_piece(p_a, n=50)
        start_x, start_y = evaluate_piece(p_b, n=50)
        gap = np.hypot(end_x[-1] - start_x[0], end_y[-1] - start_y[0])
        assert gap < 0.2, f"gap of {gap} between piece {i} and {i+1}"
