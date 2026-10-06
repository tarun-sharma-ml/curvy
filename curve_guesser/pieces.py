"""Piecewise curve fitting: break a stroke into small pieces and fit each
piece with whichever simple function -- line, parabola, or circular arc --
matches it best, restricted to that piece's own little domain.

This is the "recreate a drawing with a pile of restricted-domain equations"
technique: rather than one complicated equation per stroke, you get many
simple ones, each only drawn over a short stretch, e.g.

    y = 0.8342x + 1.204        {0.5 <= x <= 1.2}
    y = 3.1 + sqrt(2^2 - (x-1.2)^2)   {1.2 <= x <= 1.9}

stitched together they trace the original stroke.
"""

import numpy as np


def resample_by_arclength(x, y, num_points):
    """Resample (x, y) to `num_points` uniformly-spaced-by-arc-length points."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    seg_lengths = np.hypot(np.diff(x), np.diff(y))
    arc_length = np.concatenate([[0], np.cumsum(seg_lengths)])
    total = arc_length[-1]
    if total == 0:
        return np.full(num_points, x[0]), np.full(num_points, y[0])
    target = np.linspace(0, total, num_points)
    return np.interp(target, arc_length, x), np.interp(target, arc_length, y)


def split_stroke(x, y, num_pieces, points_per_piece=25):
    """Split a stroke into `num_pieces` consecutive pieces, evenly spaced by
    arc length. Adjacent pieces share their boundary point so the fitted
    curves connect with no visible gap.
    """
    xu, yu = resample_by_arclength(x, y, num_pieces * points_per_piece + 1)
    pieces = []
    n = points_per_piece
    for i in range(num_pieces):
        start = i * n
        end = start + n + 1  # +1 to include shared boundary point
        pieces.append((xu[start:end], yu[start:end]))
    return pieces


def _fit_line(indep, dep):
    a, b = np.polyfit(indep, dep, 1)
    pred = a * indep + b
    return {"kind": "line", "params": (a, b)}, pred


def _fit_quadratic(indep, dep):
    a, b, c = np.polyfit(indep, dep, 2)
    pred = a * indep**2 + b * indep + c
    return {"kind": "quadratic", "params": (a, b, c)}, pred


def _fit_circle(x, y):
    """Algebraic (Kasa) circle fit: minimizes algebraic distance to
    x^2 + y^2 + D*x + E*y + F = 0, giving center (h, k) and radius r.
    """
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x**2 + y**2)
    (D, E, F), *_ = np.linalg.lstsq(A, b, rcond=None)
    h, k = -D / 2, -E / 2
    r_sq = h**2 + k**2 - F
    if r_sq <= 0:
        raise ValueError("degenerate circle fit")
    r = np.sqrt(r_sq)
    return h, k, r


def _rms_error(actual, pred, scale):
    return np.sqrt(np.mean((actual - pred) ** 2)) / max(scale, 1e-9)

def _circle_pred(h, k, r, sign, orientation, v):
    """Evaluate a fitted circle at one independent-variable value v."""
    center, other = (h, k) if orientation == "x" else (k, h)
    inside = max(r**2 - (v - center) ** 2, 0.0)
    return other + sign * np.sqrt(inside)


def _enforce_endpoint_continuity(piece, x_seg, y_seg):
    """Adjust a fitted piece so its curve passes exactly through the piece's
    own start and end points.

    Pieces are fit independently by least squares, so without this, two
    neighboring pieces -- even though they share a boundary sample point
    in the input data -- don't generally agree on where their *fitted
    curves* land at that shared x (or y). Visually that's a gap.
    """
    orientation = piece["orientation"]
    if orientation == "x":
        v0, dep0 = x_seg[0], y_seg[0]
        v1, dep1 = x_seg[-1], y_seg[-1]
    else:
        v0, dep0 = y_seg[0], x_seg[0]
        v1, dep1 = y_seg[-1], x_seg[-1]

    if abs(v1 - v0) < 1e-9:
        return piece  # degenerate piece, leave as-is

    kind = piece["kind"]
    if kind == "line":
        a = (dep1 - dep0) / (v1 - v0)
        b = dep0 - a * v0
        piece["params"] = (a, b)
    elif kind == "quadratic":
        a, b, c = piece["params"]
        old0 = a * v0**2 + b * v0 + c
        old1 = a * v1**2 + b * v1 + c
        rhs0, rhs1 = dep0 - old0, dep1 - old1
        B = (rhs1 - rhs0) / (v1 - v0)
        A = rhs0 - B * v0
        piece["params"] = (a, b + B, c + A)
    elif kind == "circle":
        h, k, r, sign = piece["params"]
        pred0 = _circle_pred(h, k, r, sign, orientation, v0)
        pred1 = _circle_pred(h, k, r, sign, orientation, v1)
        delta = 0.5 * ((dep0 - pred0) + (dep1 - pred1))
        if orientation == "x":
            k = k + delta
        else:
            h = h + delta
        piece["params"] = (h, k, r, sign)

    scale = max(np.ptp(x_seg), np.ptp(y_seg), 1e-9)
    if orientation == "x":
        indep, dep = x_seg, y_seg
    else:
        indep, dep = y_seg, x_seg
    if kind == "line":
        a, b = piece["params"]
        pred = a * indep + b
    elif kind == "quadratic":
        a, b, c = piece["params"]
        pred = a * indep**2 + b * indep + c
    else:
        h, k, r, sign = piece["params"]
        pred = np.array([_circle_pred(h, k, r, sign, orientation, v) for v in indep])
    piece["error"] = _rms_error(dep, pred, scale)
    return piece

def fit_piece(x_seg, y_seg):
    """Fit a single piece with the best of {line, quadratic, circular arc}.

    Returns a dict describing the winning fit:
        kind: "line" | "quadratic" | "circle"
        orientation: "x" (y depends on x) or "y" (x depends on y)
        params: fit-specific coefficients
        domain: (lo, hi) bounds on the independent variable
        error: normalized RMS error
    """
    x_seg = np.asarray(x_seg, dtype=float)
    y_seg = np.asarray(y_seg, dtype=float)
    range_x = np.ptp(x_seg)
    range_y = np.ptp(y_seg)
    scale = max(range_x, range_y, 1e-9)

    def monotonicity(v):
        d = np.diff(v)
        total_variation = np.sum(np.abs(d)) + 1e-12
        net_displacement = abs(v[-1] - v[0])
        return net_displacement / total_variation

    mono_x = monotonicity(x_seg)
    mono_y = monotonicity(y_seg)

    if range_x < 0.05 * max(range_y, 1e-9):
        orientation = "y"
    elif range_y < 0.05 * max(range_x, 1e-9):
        orientation = "x"
    elif mono_x >= 0.99 and mono_x >= mono_y - 0.01:
        orientation = "x"
    elif mono_y >= 0.99:
        orientation = "y"
    elif range_x >= range_y:
        orientation = "x"
    else:
        orientation = "y"

    if orientation == "x":
        indep, dep = x_seg, y_seg
    else:
        indep, dep = y_seg, x_seg

    candidates = []

    try:
        fit, pred = _fit_line(indep, dep)
        fit["error"] = _rms_error(dep, pred, scale)
        candidates.append(fit)
    except Exception:
        pass

    if len(indep) >= 3:
        try:
            fit, pred = _fit_quadratic(indep, dep)
            fit["error"] = _rms_error(dep, pred, scale)
            candidates.append(fit)
        except Exception:
            pass

    if len(x_seg) >= 3:
        try:
            h, k, r = _fit_circle(x_seg, y_seg)
            if orientation == "x":
                inside = r**2 - (x_seg - h) ** 2
                sign = 1.0 if np.mean(y_seg - k) >= 0 else -1.0
                pred = k + sign * np.sqrt(np.clip(inside, 0, None))
                err = _rms_error(y_seg, pred, scale)
            else:
                inside = r**2 - (y_seg - k) ** 2
                sign = 1.0 if np.mean(x_seg - h) >= 0 else -1.0
                pred = h + sign * np.sqrt(np.clip(inside, 0, None))
                err = _rms_error(x_seg, pred, scale)
            candidates.append(
                {"kind": "circle", "params": (h, k, r, sign), "error": err}
            )
        except Exception:
            pass

    if not candidates:
        candidates = [{"kind": "line", "params": (0.0, float(np.mean(dep))), "error": 1.0}]

    candidates.sort(key=lambda c: c["error"])
    best = candidates[0]
    complexity_rank = {"line": 0, "quadratic": 1, "circle": 1}
    tolerance = max(best["error"] * 1.15, best["error"] + 1e-3)
    for c in candidates:
        if c["error"] <= tolerance and complexity_rank[c["kind"]] < complexity_rank[best["kind"]]:
            best = c

    best["orientation"] = orientation
    best["domain"] = (float(np.min(indep)), float(np.max(indep)))
    best["domain_dir"] = (float(indep[0]), float(indep[-1]))
    best = _enforce_endpoint_continuity(best, x_seg, y_seg)
    return best


def fit_stroke_pieces(x, y, num_pieces, points_per_piece=25):
    """Split a stroke into `num_pieces` and fit each. Returns a list of
    per-piece fit dicts (see `fit_piece`).
    """
    pieces = split_stroke(x, y, num_pieces, points_per_piece)
    return [fit_piece(px, py) for px, py in pieces]


def evaluate_piece(piece, n=40):
    """Sample points along a fitted piece for plotting."""
    lo, hi = piece.get("domain_dir", piece['domain'])
    t = np.linspace(lo, hi, n)
    kind = piece["kind"]

    if kind == "line":
        a, b = piece["params"]
        dep = a * t + b
    elif kind == "quadratic":
        a, b, c = piece["params"]
        dep = a * t**2 + b * t + c
    elif kind == "circle":
        h, k, r, sign = piece["params"]
        dep = np.array([_circle_pred(h, k, r, sign, piece["orientation"],v) for v in t])
    else:
        raise ValueError(f"unknown kind {kind}")

    if piece["orientation"] == "x":
        return t, dep
    else:
        return dep, t


def _piece_rhs(piece):
    """The right-hand-side expression for a piece, without the domain
    restriction or the 'y=' / 'x=' prefix. Used both for standalone
    equations and for branches inside a merged piecewise function.
    """
    kind = piece["kind"]
    orientation = piece["orientation"]
    var = "x" if orientation == "x" else "y"

    if kind == "line":
        a, b = piece["params"]
        return f"{a:.4f}{var}{'+' if b >= 0 else '-'}{abs(b):.4f}"
    elif kind == "quadratic":
        a, b, c = piece["params"]
        return (
            f"{a:.4f}{var}^2"
            f"{'+' if b >= 0 else '-'}{abs(b):.4f}{var}"
            f"{'+' if c >= 0 else '-'}{abs(c):.4f}"
        )
    elif kind == "circle":
        h, k, r, sign = piece["params"]
        center = h if orientation == "x" else k
        other_center = k if orientation == "x" else h
        sign_str = "+" if sign >= 0 else "-"
        center_str = f"+{abs(center):.4f}" if center < 0 else f"-{center:.4f}"
        return (
            f"{other_center:.4f}{sign_str}"
            f"\\sqrt{{{r:.4f}^2-({var}{center_str})^2}}"
        )
    else:
        raise ValueError(f"unknown kind {kind}")


def to_desmos_string(piece):
    """Format a fitted piece as a standalone Desmos-pasteable equation with
    a domain restriction, e.g. "y=1.2x+0.3\\left\\{0.1\\le x\\le0.9\\right\\}".
    """
    orientation = piece["orientation"]
    lo, hi = piece["domain"]
    var = "x" if orientation == "x" else "y"
    other = "y" if orientation == "x" else "x"
    rhs = _piece_rhs(piece)
    domain = f"\\left\\{{{lo:.4f}\\le {var}\\le{hi:.4f}\\right\\}}"
    return f"{other}={rhs}{domain}"


def group_pieces(piece_list):
    """Group consecutive same-orientation pieces whose domains extend
    outward (rather than doubling back / overlapping, as happens when a
    stroke loops around like a circle) into runs that can share one
    compact piecewise function.
    """
    groups = []
    current = None
    for piece in piece_list:
        lo, hi = piece["domain"]
        if current is None:
            current = {"orientation": piece["orientation"], "pieces": [piece], "lo": lo, "hi": hi}
            continue
        same_orientation = piece["orientation"] == current["orientation"]
        extends_right = lo >= current["hi"] - 1e-9
        extends_left = hi <= current["lo"] + 1e-9
        if same_orientation and (extends_right or extends_left):
            current["pieces"].append(piece)
            current["lo"] = min(current["lo"], lo)
            current["hi"] = max(current["hi"], hi)
        else:
            groups.append(current)
            current = {"orientation": piece["orientation"], "pieces": [piece], "lo": lo, "hi": hi}
    if current is not None:
        groups.append(current)
    return groups


def group_to_desmos(group, name):
    """Format a group of pieces as one compact named piecewise function.

    If the group has only one piece, falls back to a plain standalone
    equation (no point wrapping a single branch in a named function).
    """
    orientation = group["orientation"]
    var = "x" if orientation == "x" else "y"

    if len(group["pieces"]) == 1:
        return to_desmos_string(group["pieces"][0])

    pieces_sorted = sorted(group["pieces"], key=lambda p: p["domain"][0])
    branches = []
    for i, piece in enumerate(pieces_sorted):
        rhs = _piece_rhs(piece)
        if i < len(pieces_sorted) - 1:
            hi = piece["domain"][1]
            branches.append(f"{var}\\le{hi:.4f}:{rhs}")
        else:
            branches.append(rhs)
    body = ",".join(branches)
    lo, hi = group["lo"], group["hi"]
    outer_domain = f"\\left\\{{{lo:.4f}\\le {var}\\le{hi:.4f}\\right\\}}"
    return f"{name}({var})=\\left\\{{{body}\\right\\}}{outer_domain}"

def _group_body_expr(group, var):
    """Build just the RHS piecewise-branch expression for a group (no
    domain restriction) -- shared by group_to_desmos and the animated
    reveal builder below.
    """
    pieces_sorted = sorted(group["pieces"], key=lambda p: p["domain"][0])
    if len(pieces_sorted) == 1:
        return _piece_rhs(pieces_sorted[0])
    branches = []
    for i, piece in enumerate(pieces_sorted):
        rhs = _piece_rhs(piece)
        if i < len(pieces_sorted) - 1:
            hi_i = piece["domain"][1]
            branches.append(f"{var}\\le{hi_i:.4f}:{rhs}")
        else:
            branches.append(rhs)
    return "\\left\\{" + ",".join(branches) + "\\right\\}"


def to_desmos_animated(group, reveal_name, slider_var="x_{1}"):
    """Build a "pen drawing" reveal animation for one group: a shared
    slider that progressively extends the curve as it increases, with a
    tangent line and point marking the pen tip. Only x-oriented groups
    (y=f(x)) are supported; returns None for y-oriented groups.

    The piece's formula is inlined directly into the restricted reveal
    function rather than ALSO being defined as its own separate,
    unrestricted named function -- Desmos auto-plots EVERY top-level
    function definition, so a separate "base" function would itself get
    graphed fully unrestricted, blowing up to +-infinity outside its
    intended tiny range and drawing wild extra curves across the whole
    graph alongside the correctly-restricted one.
    """
    if group["orientation"] != "x":
        return None

    var = "x"
    lo, hi = group["lo"], group["hi"]
    body_expr = _group_body_expr(group, var)

    reveal_def = (
        f"{reveal_name}({var})={body_expr}"
        f"\\left\\{{{lo:.4f}\\le {var}\\le {slider_var}\\right\\}}"
        f"\\left\\{{{var}\\le{hi:.4f}\\right\\}}"
    )
    tangent = f"y-{reveal_name}({slider_var})={reveal_name}'({slider_var})({var}-{slider_var})"
    point = f"({slider_var},{reveal_name}({slider_var}))"
    return [reveal_def, tangent, point]
