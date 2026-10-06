"""Whole-stroke shape classifier: look at an ENTIRE drawn stroke and guess
which single canonical curve it resembles overall -- circle, ellipse,
parabola, line, or hyperbola -- then fill in that curve's general equation
with numbers fitted to where you actually drew it.

This is deliberately different from `pieces.py`, which chops a stroke into
many small pieces to trace ANY shape exactly. This module instead asks one
question per whole stroke: "of the handful of named curves people learn in
school, which one does this look like, and with what parameters?"
"""

import numpy as np


def _choose_orientation(x, y):
    """Decide whether to treat x or y as the independent variable, based
    on monotonicity (can this axis be walked through without doubling
    back?) rather than raw range -- a parabola's dependent axis can have
    a bigger range than its independent one near the arms, which a plain
    range comparison gets wrong.
    """
    range_x, range_y = np.ptp(x), np.ptp(y)

    def monotonicity(v):
        d = np.diff(v)
        total_variation = np.sum(np.abs(d)) + 1e-12
        net_displacement = abs(v[-1] - v[0])
        return net_displacement / total_variation

    mono_x, mono_y = monotonicity(x), monotonicity(y)

    if range_x < 0.05 * max(range_y, 1e-9):
        return "y"
    if range_y < 0.05 * max(range_x, 1e-9):
        return "x"
    if mono_x >= 0.99 and mono_x >= mono_y - 0.01:
        return "x"
    if mono_y >= 0.99:
        return "y"
    return "x" if range_x >= range_y else "y"


def _normalized_rms(residual, scale):
    return float(np.sqrt(np.mean(residual**2)) / max(scale, 1e-9))


def _fit_line(x, y):
    """Best-fit line through the whole stroke, oriented to avoid a
    near-vertical blow-up, same idea as the per-piece orientation logic.
    """
    range_x, range_y = np.ptp(x), np.ptp(y)
    scale = max(range_x, range_y, 1e-9)

    if _choose_orientation(x, y) == "x":
        m, b = np.polyfit(x, y, 1)
        pred = m * x + b
        err = _normalized_rms(y - pred, scale)
        equation = f"y = {m:.4f}x {'+' if b >= 0 else '-'} {abs(b):.4f}"
        params = {"orientation": "x", "m": m, "b": b}
    else:
        m, b = np.polyfit(y, x, 1)
        pred = m * y + b
        err = _normalized_rms(x - pred, scale)
        equation = f"x = {m:.4f}y {'+' if b >= 0 else '-'} {abs(b):.4f}"
        params = {"orientation": "y", "m": m, "b": b}

    return {"kind": "line", "error": err, "equation": equation, "params": params}


def _fit_circle(x, y):
    """Algebraic (Kasa) circle fit."""
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x**2 + y**2)
    (D, E, F), *_ = np.linalg.lstsq(A, b, rcond=None)
    h, k = -D / 2, -E / 2
    r_sq = h**2 + k**2 - F
    if r_sq <= 0:
        raise ValueError("degenerate circle fit")
    r = np.sqrt(r_sq)

    dist = np.hypot(x - h, y - k)
    scale = max(np.ptp(x), np.ptp(y), 1e-9)
    err = _normalized_rms(dist - r, scale)

    h_str = f"x {'-' if h >= 0 else '+'} {abs(h):.4f}"
    k_str = f"y {'-' if k >= 0 else '+'} {abs(k):.4f}"
    equation = f"({h_str})^2 + ({k_str})^2 = {r:.4f}^2"
    return {
        "kind": "circle",
        "error": err,
        "equation": equation,
        "params": {"h": h, "k": k, "r": r},
    }


def _fit_parabola(x, y):
    """Best vertical (y=f(x)) or horizontal (x=f(y)) parabola, whichever
    orientation the stroke actually suggests.
    """
    range_x, range_y = np.ptp(x), np.ptp(y)

    def vertex_form(indep, dep):
        a, b, c = np.polyfit(indep, dep, 2)
        pred = a * indep**2 + b * indep + c
        h = -b / (2 * a)
        k = c - b**2 / (4 * a)
        return a, h, k, pred

    if _choose_orientation(x, y) == "x":
        a, h, k, pred = vertex_form(x, y)
        scale = max(range_y, 1e-9)
        err = _normalized_rms(y - pred, scale)
        h_str = f"x {'-' if h >= 0 else '+'} {abs(h):.4f}"
        k_sign = "+" if k >= 0 else "-"
        equation = f"y = {a:.4f}({h_str})^2 {k_sign} {abs(k):.4f}"
        params = {"orientation": "x", "a": a, "h": h, "k": k}
    else:
        a, h, k, pred = vertex_form(y, x)
        scale = max(range_x, 1e-9)
        err = _normalized_rms(x - pred, scale)
        h_str = f"y {'-' if h >= 0 else '+'} {abs(h):.4f}"
        k_sign = "+" if k >= 0 else "-"
        equation = f"x = {a:.4f}({h_str})^2 {k_sign} {abs(k):.4f}"
        params = {"orientation": "y", "a": a, "h": h, "k": k}

    return {"kind": "parabola", "error": err, "equation": equation, "params": params}


def _fit_axis_aligned_ellipse_or_hyperbola(x, y):
    """Fit the best axis-aligned conic A*x^2 + C*y^2 + D*x + E*y = 1 and
    classify it as an ellipse or a hyperbola from the signs of A and C.
    Axis-aligned only (no rotation term) -- matches the simple textbook
    forms the person asked for.
    """
    M = np.column_stack([x**2, y**2, x, y])
    target = np.ones_like(x)
    (A, C, D, E), *_ = np.linalg.lstsq(M, target, rcond=None)

    if abs(A) < 1e-9 or abs(C) < 1e-9:
        raise ValueError("degenerate conic (too close to a parabola/line)")

    h = -D / (2 * A)
    k = -E / (2 * C)
    rhs = D**2 / (4 * A) + E**2 / (4 * C) + 1  # +1 because target=1, not 0

    a_sq = rhs / A
    b_sq = rhs / C

    scale = max(np.ptp(x), np.ptp(y), 1e-9)
    pred = A * x**2 + C * y**2 + D * x + E * y
    err = _normalized_rms(pred - target, 1.0)  # algebraic residual, already ~O(1)

    h_str = f"x {'-' if h >= 0 else '+'} {abs(h):.4f}"
    k_str = f"y {'-' if k >= 0 else '+'} {abs(k):.4f}"

    if a_sq > 0 and b_sq > 0:
        a, b = np.sqrt(a_sq), np.sqrt(b_sq)
        equation = f"({h_str})^2/{a:.4f}^2 + ({k_str})^2/{b:.4f}^2 = 1"
        return {
            "kind": "ellipse",
            "error": err,
            "equation": equation,
            "params": {"h": h, "k": k, "a": a, "b": b},
        }
    elif a_sq > 0 and b_sq < 0:
        a, b = np.sqrt(a_sq), np.sqrt(-b_sq)
        equation = f"({h_str})^2/{a:.4f}^2 - ({k_str})^2/{b:.4f}^2 = 1"
        return {
            "kind": "hyperbola",
            "error": err,
            "equation": equation,
            "params": {"h": h, "k": k, "a": a, "b": b},
        }
    elif a_sq < 0 and b_sq > 0:
        a, b = np.sqrt(-a_sq), np.sqrt(b_sq)
        equation = f"({k_str})^2/{b:.4f}^2 - ({h_str})^2/{a:.4f}^2 = 1"
        return {
            "kind": "hyperbola",
            "error": err,
            "equation": equation,
            "params": {"h": h, "k": k, "a": a, "b": b},
        }
    else:
        raise ValueError("no real conic solution")


def guess_shape(x, y):
    """Look at a whole stroke and guess which canonical shape it best
    matches. Returns a dict: {kind, equation, error, params}.

    `error` is a normalized residual -- lower is better -- but it is NOT
    on a common scale across all candidate kinds (explicit y=f(x) fits vs.
    algebraic conic residuals aren't directly comparable), so candidates
    are ranked within comparable groups and a few simple preferences
    (favor the simpler shape -- line over parabola over circle/ellipse)
    break close ties, the same spirit as the per-piece fitter.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    candidates = []

    if len(x) >= 2 and len(y) >= 2:
        try:
            candidates.append(_fit_line(x, y))
        except Exception:
            pass

    if len(x) >= 3:
        try:
            candidates.append(_fit_parabola(x, y))
        except Exception:
            pass
        try:
            candidates.append(_fit_circle(x, y))
        except Exception:
            pass

    if len(x) >= 4:
        try:
            candidates.append(_fit_axis_aligned_ellipse_or_hyperbola(x, y))
        except Exception:
            pass

    if not candidates:
        return {
            "kind": "unknown",
            "error": float("inf"),
            "equation": "(not enough points to guess a shape)",
            "params": {},
            "confidence": "poor match -- doesn't resemble a simple shape well",
        }

    # All error metrics are normalized residuals on a comparable scale
    # (fraction of the stroke's own extent), so just pick the lowest
    # overall, then let a simpler shape win close ties.
    candidates.sort(key=lambda c: c["error"])
    best = candidates[0]

    complexity_rank = {"line": 0, "parabola": 1, "circle": 1, "ellipse": 2, "hyperbola": 2}
    for c in candidates:
        tol = max(best["error"] * 1.2, best["error"] + 1e-3)
        if c["error"] <= tol and complexity_rank[c["kind"]] < complexity_rank[best["kind"]]:
            best = c

    if best["error"] < 0.02:
        confidence = "excellent match"
    elif best["error"] < 0.08:
        confidence = "good match"
    elif best["error"] < 0.2:
        confidence = "rough match"
    else:
        confidence = "poor match -- doesn't resemble a simple shape well"
    best["confidence"] = confidence

    return best
