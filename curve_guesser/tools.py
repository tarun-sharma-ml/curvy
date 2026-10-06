"""Geometry helpers behind the drawing board's illustrator-style tools.

Everything here is plain numpy -- no matplotlib -- so it can be tested
without a display. A "stroke" is an (xs, ys) pair of numpy arrays, exactly
what the rest of the pipeline (pieces.py, shapes.py, preview.py) expects, so
anything built with these helpers flows straight into the curve fitting.
Shapes with corners (rectangles, polygons) are returned as one stroke per
side so each side fits as a clean straight line instead of a rounded blob.
"""

import numpy as np

GRID_STEP = 1.0


# --------------------------------------------------------------------------
# snapping / constraints
# --------------------------------------------------------------------------
def snap_point(x, y, step=GRID_STEP):
    """Snap (x, y) to the nearest grid intersection."""
    return (float(np.floor(x / step + 0.5) * step),
            float(np.floor(y / step + 0.5) * step))


def constrain_angle(p0, p1, step_deg=15.0):
    """Rotate p1 about p0 so the p0->p1 direction is a multiple of step_deg
    (this is what holding Shift does for the line tool)."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = float(np.hypot(dx, dy))
    if length == 0:
        return p1
    step = np.radians(step_deg)
    ang = np.round(np.arctan2(dy, dx) / step) * step
    return (p0[0] + length * float(np.cos(ang)), p0[1] + length * float(np.sin(ang)))


# --------------------------------------------------------------------------
# building shapes as strokes
# --------------------------------------------------------------------------
def line_stroke(p0, p1, n=40):
    return np.linspace(p0[0], p1[0], n), np.linspace(p0[1], p1[1], n)


def box_from_drag(p0, p1, square=False, from_center=False):
    """Bounding box (xmin, ymin, xmax, ymax) for a drag from p0 to p1.

    square:      force equal width and height (Shift)
    from_center: p0 is the centre rather than a corner (Alt)
    """
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    if square:
        s = max(abs(dx), abs(dy))
        dx, dy = float(np.copysign(s, dx)), float(np.copysign(s, dy))
    if from_center:
        xa, xb = p0[0] - dx, p0[0] + dx
        ya, yb = p0[1] - dy, p0[1] + dy
    else:
        xa, xb = p0[0], p0[0] + dx
        ya, yb = p0[1], p0[1] + dy
    return min(xa, xb), min(ya, yb), max(xa, xb), max(ya, yb)


def rectangle_strokes(xmin, ymin, xmax, ymax, n=30):
    """Four strokes: bottom, right, top, left."""
    return [
        line_stroke((xmin, ymin), (xmax, ymin), n),
        line_stroke((xmax, ymin), (xmax, ymax), n),
        line_stroke((xmax, ymax), (xmin, ymax), n),
        line_stroke((xmin, ymax), (xmin, ymin), n),
    ]


def ellipse_from_box(xmin, ymin, xmax, ymax):
    return (xmin + xmax) / 2, (ymin + ymax) / 2, (xmax - xmin) / 2, (ymax - ymin) / 2


def ellipse_stroke(cx, cy, rx, ry, n=181):
    """A closed ellipse (the last point repeats the first)."""
    t = np.linspace(0, 2 * np.pi, n)
    return cx + rx * np.cos(t), cy + ry * np.sin(t)


def polygon_strokes(vertices, closed=True, n=30):
    """One stroke per edge of the polygon / polyline through `vertices`."""
    pts = list(vertices)
    edges = list(zip(pts[:-1], pts[1:]))
    if closed and len(pts) > 2:
        edges.append((pts[-1], pts[0]))
    return [line_stroke(a, b, n) for a, b in edges]


# --------------------------------------------------------------------------
# picking / moving
# --------------------------------------------------------------------------
def distance_to_stroke(px, py, xs, ys):
    """Shortest distance from the point (px, py) to the polyline (xs, ys)."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if len(xs) == 0:
        return float("inf")
    if len(xs) == 1:
        return float(np.hypot(px - xs[0], py - ys[0]))
    ax, ay, bx, by = xs[:-1], ys[:-1], xs[1:], ys[1:]
    abx, aby = bx - ax, by - ay
    len2 = abx ** 2 + aby ** 2
    safe = np.where(len2 > 0, len2, 1.0)
    t = np.clip(((px - ax) * abx + (py - ay) * aby) / safe, 0.0, 1.0)
    t = np.where(len2 > 0, t, 0.0)
    return float(np.hypot(px - (ax + t * abx), py - (ay + t * aby)).min())


def nearest_stroke(strokes, px, py):
    """(index, distance) of the stroke closest to (px, py), or None."""
    best = None
    for i, (xs, ys) in enumerate(strokes):
        d = distance_to_stroke(px, py, xs, ys)
        if best is None or d < best[1]:
            best = (i, d)
    return best


# --------------------------------------------------------------------------
# describing shapes (equations filled in from where they were drawn)
# --------------------------------------------------------------------------
def _sq(var, c):
    """'x^2' when centred on 0, otherwise '(x - 2.00)^2'."""
    if abs(c) < 0.005:
        return f"{var}^2"
    return f"({var} {'-' if c >= 0 else '+'} {abs(c):.2f})^2"


def describe_line(p0, p1):
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    if abs(dx) < 1e-9:
        return f"Line: x = {p0[0]:.2f}  (vertical)"
    m = dy / dx
    b = p0[1] - m * p0[0]
    return f"Line: y = {m:.2f}x {'+' if b >= 0 else '-'} {abs(b):.2f}"


def describe_rectangle(xmin, ymin, xmax, ymax):
    w, h = xmax - xmin, ymax - ymin
    kind = "Square" if abs(w - h) < 0.005 else "Rectangle"
    return (f"{kind} {w:.2f} x {h:.2f}:  {xmin:.2f} <= x <= {xmax:.2f},  "
            f"{ymin:.2f} <= y <= {ymax:.2f}")


def describe_circle(cx, cy, r):
    return f"Circle: {_sq('x', cx)} + {_sq('y', cy)} = {r:.2f}^2"


def describe_ellipse(cx, cy, rx, ry):
    if abs(rx - ry) < 0.005:
        return describe_circle(cx, cy, rx)
    return f"Ellipse: {_sq('x', cx)}/{rx:.2f}^2 + {_sq('y', cy)}/{ry:.2f}^2 = 1"


def describe_polygon(vertices, closed=True):
    n = len(vertices)
    if closed:
        names = {3: "Triangle", 4: "Quadrilateral", 5: "Pentagon", 6: "Hexagon"}
        name = names.get(n, f"{n}-sided polygon")
    else:
        name = f"Polyline ({n - 1} segments)"
    shown = ", ".join(f"({x:.1f}, {y:.1f})" for x, y in vertices[:6])
    more = ", ..." if n > 6 else ""
    return f"{name}: vertices {shown}{more}"