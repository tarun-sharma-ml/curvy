import numpy as np

from curve_guesser.shapes import guess_shape


def test_circle_is_identified():
    theta = np.linspace(0, 2 * np.pi, 200)
    x = 3 + 2 * np.cos(theta)
    y = -1 + 2 * np.sin(theta)
    g = guess_shape(x, y)
    assert g["kind"] == "circle"
    assert g["error"] < 1e-6
    assert abs(g["params"]["h"] - 3) < 1e-6
    assert abs(g["params"]["k"] - (-1)) < 1e-6
    assert abs(g["params"]["r"] - 2) < 1e-6


def test_line_is_identified():
    x = np.linspace(0, 5, 100)
    y = 2 * x + 1
    g = guess_shape(x, y)
    assert g["kind"] == "line"
    assert g["error"] < 1e-6


def test_vertical_parabola_is_identified_not_circle():
    # Regression test: a parabola's dependent axis can have a bigger
    # range than its independent one near the arms, which a naive
    # range-based orientation check gets wrong and misreports as a circle.
    x = np.linspace(-3, 3, 100)
    y = 0.5 * (x - 1) ** 2 + 2
    g = guess_shape(x, y)
    assert g["kind"] == "parabola"
    assert g["error"] < 1e-6
    assert abs(g["params"]["a"] - 0.5) < 1e-3
    assert abs(g["params"]["h"] - 1) < 1e-3
    assert abs(g["params"]["k"] - 2) < 1e-3


def test_horizontal_parabola_is_identified():
    y = np.linspace(-3, 3, 100)
    x = 0.7 * (y + 2) ** 2 - 1
    g = guess_shape(x, y)
    assert g["kind"] == "parabola"
    assert g["params"]["orientation"] == "y"
    assert g["error"] < 1e-6


def test_ellipse_is_identified():
    theta = np.linspace(0, 2 * np.pi, 200)
    x = 1 + 4 * np.cos(theta)
    y = -2 + 1.5 * np.sin(theta)
    g = guess_shape(x, y)
    assert g["kind"] == "ellipse"
    assert g["error"] < 1e-6
    assert abs(g["params"]["a"] - 4) < 1e-3
    assert abs(g["params"]["b"] - 1.5) < 1e-3


def test_hyperbola_is_identified_not_parabola():
    # Regression test: one branch of a hyperbola is locally similar to a
    # parabola, so the selection logic must actually compare the two
    # fairly rather than gating conics behind an overly strict threshold.
    t = np.linspace(-1.5, 1.5, 100)
    x = 0.5 + 2.0 * np.cosh(t)
    y = -1.0 + 1.0 * np.sinh(t)
    g = guess_shape(x, y)
    assert g["kind"] == "hyperbola"
    assert g["error"] < 1e-6


def test_noisy_circle_still_identified_with_higher_error():
    rng = np.random.default_rng(0)
    theta = np.linspace(0, 2 * np.pi, 150)
    x = 2 * np.cos(theta) + rng.normal(0, 0.05, 150)
    y = 2 * np.sin(theta) + rng.normal(0, 0.05, 150)
    g = guess_shape(x, y)
    assert g["kind"] == "circle"
    assert 0 < g["error"] < 0.05
    assert g["confidence"] in ("excellent match", "good match")


def test_random_scribble_gets_low_confidence():
    rng = np.random.default_rng(0)
    x = rng.uniform(-3, 3, 60)
    y = rng.uniform(-3, 3, 60)
    order = np.argsort(np.arctan2(y, x))
    x, y = x[order], y[order]
    g = guess_shape(x, y)
    assert g["confidence"] in ("rough match", "poor match -- doesn't resemble a simple shape well")


def test_too_few_points_returns_unknown_or_line_gracefully():
    g = guess_shape([0, 1], [0, 1])
    assert g["kind"] in ("line", "unknown")


def test_confidence_field_present_on_every_result():
    theta = np.linspace(0, 2 * np.pi, 50)
    x, y = np.cos(theta), np.sin(theta)
    g = guess_shape(x, y)
    assert "confidence" in g
