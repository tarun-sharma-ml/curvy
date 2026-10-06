import numpy as np

from curve_guesser.pieces import fit_stroke_pieces, group_pieces, to_desmos_animated


def test_to_desmos_animated_returns_none_for_y_oriented_group():
    y = np.linspace(0, 5, 20)
    x = np.full_like(y, 3.0) + 1e-6 * y
    from curve_guesser.pieces import fit_piece
    piece_list = [fit_piece(x, y)]
    groups = group_pieces(piece_list)
    assert groups[0]["orientation"] == "y"
    result = to_desmos_animated(groups[0], "f_{0}")
    assert result is None


def test_to_desmos_animated_x_oriented_returns_three_expressions():
    x = np.linspace(0, 10, 300)
    y = 2 * x + 1
    piece_list = fit_stroke_pieces(x, y, num_pieces=4)
    groups = group_pieces(piece_list)
    result = to_desmos_animated(groups[0], "f_{0}")
    assert result is not None
    assert len(result) == 3
    reveal_def, tangent, point = result
    assert reveal_def.startswith("f_{0}(x)=")
    # must be triple-bounded: branch selector, slider lower/upper bound,
    # and the hard upper cap -- so it can never run unrestricted
    assert reveal_def.count("\\left\\{") == 3
    assert "x_{1}" in reveal_def
    assert tangent.startswith("y-f_{0}(x_{1})=f_{0}'(x_{1})")
    assert point == "(x_{1},f_{0}(x_{1}))"


def test_to_desmos_animated_never_defines_separate_unrestricted_function():
    # Regression test for the "wild curves everywhere" bug: there must be
    # exactly ONE named-function definition in the output, not a separate
    # unrestricted base function plus a restricted wrapper.
    x = np.linspace(0, 10, 300)
    y = np.sin(x)
    piece_list = fit_stroke_pieces(x, y, num_pieces=6)
    groups = group_pieces(piece_list)
    result = to_desmos_animated(groups[0], "f_{0}")
    function_definitions = [line for line in result if "(x)=" in line]
    assert len(function_definitions) == 1


def test_vertical_strokes_are_slider_gated_and_uniquely_named():
    # Regression test: vertical strokes must (a) each get a unique
    # function name, not all literally "v", and (b) be gated by the
    # shared slider so the canvas starts blank rather than showing every
    # vertical stroke immediately regardless of slider position.
    from curve_guesser.preview import build_animated_export

    x1 = np.linspace(0, 5, 100)
    y1 = np.sin(x1)
    y2 = np.linspace(0, 2, 50)
    x2 = np.full(50, 2.0) + 1e-6 * y2
    y3 = np.linspace(0, 2, 50)
    x3 = np.full(50, 4.0) + 1e-6 * y3

    fits = [
        fit_stroke_pieces(x1, y1, num_pieces=4),
        fit_stroke_pieces(x2, y2, num_pieces=2),
        fit_stroke_pieces(x3, y3, num_pieces=2),
    ]
    text = build_animated_export(fits)

    # unique names: v_{0}, v_{1}, not both just "v("
    assert "v_{0}(" in text
    assert "v_{1}(" in text
    assert text.count("v_{0}(") >= 1

    # each vertical stroke's line must carry an x_1-gate condition
    vertical_lines = [l for l in text.splitlines() if l.startswith("v_{")]
    assert len(vertical_lines) == 2
    for line in vertical_lines:
        assert "x_{1}\\ge" in line
