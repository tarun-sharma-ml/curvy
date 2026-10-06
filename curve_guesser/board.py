"""Interactive matplotlib drawing board with illustrator-style tools.

Tools (click the button on the left, or press the key):
    Pen (P)        freehand drawing
    Line (L)       drag; hold Shift to lock to 15 degree angles
    Rectangle (R)  drag corner to corner; Shift = square, Alt = from centre
    Circle (O)     drag from the centre outwards
    Ellipse (V)    drag the bounding box; Shift = circle, Alt = from centre
    Polygon (G)    click points; double-click / click the first point to
                   close, right-click to finish open, Esc to cancel
    Move (M)       drag any drawn object to reposition it (Delete removes it)
    Eraser (E)     rub out parts of strokes

Also: Undo (Ctrl+Z / U), Redo (Ctrl+Y), Clear (C), Snap-to-grid (S),
Fit Curve (Enter).
"""

import itertools
import textwrap

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Button

from . import tools as T
from .shapes import guess_shape

# (key, label, shortcut)
TOOLS = [
    ("pen", "Pen", "p"),
    ("line", "Line", "l"),
    ("rect", "Rectangle", "r"),
    ("circle", "Circle", "o"),
    ("ellipse", "Ellipse", "v"),
    ("polygon", "Polygon", "g"),
    ("move", "Move", "m"),
    ("erase", "Eraser", "e"),
]
SHORTCUTS = {sc: key for key, _, sc in TOOLS}
SHAPE_TOOLS = ("line", "rect", "circle", "ellipse")

HINTS = {
    "pen": "Pen: drag to draw freehand",
    "line": "Line: drag (Shift = 15 degree angles)",
    "rect": "Rectangle: drag corner to corner (Shift = square, Alt = from centre)",
    "circle": "Circle: drag from the centre outwards",
    "ellipse": "Ellipse: drag the box (Shift = circle, Alt = from centre)",
    "polygon": "Polygon: click points; double-click or click the first point to "
               "close, right-click to finish open, Esc cancels",
    "move": "Move: drag an object to move it (Delete removes it)",
    "erase": "ERASER: drag over strokes to remove them",
}

# Keys we use for shortcuts that matplotlib would otherwise grab for its own
# toolbar actions (pan, zoom, log-scale, save, ...). They're freed while the
# board is open and restored when it closes.
_CLAIMED_KEYS = set(SHORTCUTS) | {"s", "c", "u"}


def _free_shortcuts(keys):
    saved = {}
    for name in list(matplotlib.rcParams.keys()):
        if not name.startswith("keymap."):
            continue
        val = matplotlib.rcParams[name]
        if isinstance(val, list) and any(k in val for k in keys):
            saved[name] = list(val)
            matplotlib.rcParams[name] = [k for k in val if k not in keys]
    return saved


class DrawingBoard:
    """A grid + axes canvas for freehand strokes and drawn shapes, with an
    eraser, undo/redo, snap-to-grid and a move tool.

    `strokes` stays a flat list of (xs, ys) arrays -- the same thing the
    fitting pipeline has always consumed -- so shapes built with the tools
    (a rectangle is four strokes, a polygon one per edge) feed straight in.
    """

    def __init__(self, xlim=(-10, 10), ylim=(-10, 10), eraser_radius=None):
        self._saved_keymaps = _free_shortcuts(_CLAIMED_KEYS)

        self.fig, self.ax = plt.subplots(figsize=(9, 7.5))
        self.ax.set_position((0.20, 0.13, 0.77, 0.76))
        manager = self.fig.canvas.manager
        if manager is not None:
            manager.set_window_title(
                "Draw your curve  (pick a tool on the left | Enter / Fit Curve when done)"
            )
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.ax.axhline(0, color="black", linewidth=1)
        self.ax.axvline(0, color="black", linewidth=1)
        self.ax.grid(True, linestyle="--", alpha=0.4)
        self.ax.set_aspect("equal")

        span = min(xlim[1] - xlim[0], ylim[1] - ylim[0])
        # Eraser radius defaults to ~3% of the smaller axis span.
        self.eraser_radius = eraser_radius if eraser_radius is not None else 0.03 * span
        self.pick_tolerance = 0.03 * span      # how close a click must be to grab a stroke
        self.close_tolerance = 0.03 * span     # click this near the first polygon point to close

        # drawing state
        self.strokes = []          # list of (xs, ys) arrays, committed strokes
        self.stroke_gids = []      # parallel list: which object each stroke belongs to
        self._gids = itertools.count(1)
        self.selected_gid = None
        self.current_x = []
        self.current_y = []
        self.drawing = False
        self.erasing = False
        self.line_artists = []     # matplotlib artists for committed strokes
        self.eraser_cursor = None  # visual circle showing eraser reach
        self._live_line = None     # the stroke currently being drawn

        # tool state
        self.tool = "pen"
        self._prev_tool = "pen"
        self.snap = False
        self.info = ""             # last description shown under the hint
        self.poly_pts = []
        self._drag_start = None
        self._preview = None
        self._erase_snapshot = None
        self._erase_changed = False
        self._move_snapshot = None
        self._move_origin = None
        self._move_total = (0.0, 0.0)
        self._moved = False

        # undo / redo
        self.undo_stack = []
        self.redo_stack = []

        # events
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_move)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("close_event", self._on_close)

        self._build_buttons()
        self.done = False
        self._refresh_buttons()
        self._update_title()

    # -- UI construction --------------------------------------------------
    def _build_buttons(self):
        self.tool_buttons = {}
        for i, (key, label, shortcut) in enumerate(TOOLS):
            bax = self.fig.add_axes((0.02, 0.86 - i * 0.065, 0.15, 0.05))
            btn = Button(bax, f"{label} ({shortcut.upper()})")
            if key == "erase":
                btn.on_clicked(self.toggle_eraser)
            else:
                btn.on_clicked(lambda e, k=key: self.set_tool(k))
            self.tool_buttons[key] = btn
        self.eraser_button = self.tool_buttons["erase"]

        def bottom(x, w, label, cb):
            ax = self.fig.add_axes((x, 0.03, w, 0.05))
            btn = Button(ax, label)
            btn.on_clicked(cb)
            return btn

        self.undo_button = bottom(0.20, 0.10, "Undo", self.undo)
        self.redo_button = bottom(0.31, 0.10, "Redo", self.redo)
        self.clear_button = bottom(0.42, 0.10, "Clear", lambda e: self.clear())
        self.snap_button = bottom(0.53, 0.14, "Snap: off", self.toggle_snap)
        self.fit_button = bottom(0.76, 0.21, "Fit Curve", self.finish)

    def _refresh_buttons(self):
        for key, btn in self.tool_buttons.items():
            color = "#bcd6ff" if key == self.tool else "0.85"
            btn.color = color
            btn.hovercolor = color if key == self.tool else "0.95"
            btn.ax.set_facecolor(color)
        self.snap_button.label.set_text("Snap: on" if self.snap else "Snap: off")
        self.snap_button.color = "#bcd6ff" if self.snap else "0.85"
        self.snap_button.ax.set_facecolor(self.snap_button.color)

    def _update_title(self):
        lines = [HINTS[self.tool]]
        if self.info:
            lines.extend(textwrap.wrap(self.info, 78))
        self.ax.set_title("\n".join(lines), fontsize=9)

    def _on_close(self, event=None):
        for name, val in self._saved_keymaps.items():
            matplotlib.rcParams[name] = val
        self._saved_keymaps = {}

    # -- tool / mode handling ---------------------------------------------
    @property
    def mode(self):
        """'erase' while the eraser is active, otherwise 'draw'."""
        return "erase" if self.tool == "erase" else "draw"

    def set_tool(self, tool):
        if tool not in [k for k, _, _ in TOOLS]:
            return
        self._cancel_in_progress()
        if tool != "move":
            self._select(None)
        if tool != "erase" and self.eraser_cursor is not None:
            self.eraser_cursor.remove()
            self.eraser_cursor = None
        if tool == "erase" and self.tool != "erase":
            self._prev_tool = self.tool
        self.tool = tool
        self.info = ""
        self._refresh_buttons()
        self._update_title()
        self.fig.canvas.draw_idle()

    def toggle_eraser(self, event=None):
        self.set_tool(self._prev_tool if self.tool == "erase" else "erase")

    def toggle_snap(self, event=None):
        self.snap = not self.snap
        self._refresh_buttons()
        self.fig.canvas.draw_idle()

    # -- helpers -------------------------------------------------------------
    def _point(self, event):
        x, y = float(event.xdata), float(event.ydata)
        if self.snap:
            x, y = T.snap_point(x, y)
        return x, y

    @staticmethod
    def _modifiers(event):
        key = (getattr(event, "key", None) or "").lower()
        return "shift" in key, "alt" in key

    def _sync_gids(self):
        """Keep stroke_gids the same length as strokes (strokes can be
        assigned directly, e.g. by tests or other code)."""
        n = len(self.strokes)
        while len(self.stroke_gids) < n:
            self.stroke_gids.append(next(self._gids))
        del self.stroke_gids[n:]

    def _select(self, gid):
        if gid != self.selected_gid:
            self.selected_gid = gid
            self._redraw_strokes()

    # -- undo / redo ---------------------------------------------------------
    def _snapshot(self):
        self._sync_gids()
        return (list(self.strokes), list(self.stroke_gids))

    def _restore(self, snap):
        self.strokes = list(snap[0])
        self.stroke_gids = list(snap[1])
        self.selected_gid = None

    def _push_undo(self, snap=None):
        self.undo_stack.append(snap if snap is not None else self._snapshot())
        del self.undo_stack[:-100]
        self.redo_stack.clear()

    def undo(self, event=None):
        if not self.undo_stack:
            return
        self._cancel_in_progress()
        self.redo_stack.append(self._snapshot())
        self._restore(self.undo_stack.pop())
        self.info = ""
        self._redraw_strokes()
        self._update_title()

    def redo(self, event=None):
        if not self.redo_stack:
            return
        self._cancel_in_progress()
        self.undo_stack.append(self._snapshot())
        self._restore(self.redo_stack.pop())
        self.info = ""
        self._redraw_strokes()
        self._update_title()

    # -- committing things to the board -----------------------------------------
    def _commit(self, strokes, info=""):
        """Add one object (one or more strokes) to the drawing."""
        self._push_undo()
        gid = next(self._gids)
        for s in strokes:
            self.strokes.append((np.asarray(s[0], float), np.asarray(s[1], float)))
            self.stroke_gids.append(gid)
        self.info = info
        self._redraw_strokes()
        self._update_title()

    def _describe_pen_stroke(self, x, y):
        try:
            g = guess_shape(x, y)
        except Exception:
            return ""
        if g["kind"] == "unknown" or g["confidence"].startswith("poor"):
            return "Pen stroke: freeform (no single simple shape matches)"
        return (f"Pen stroke looks like a {g['kind'].upper()} "
                f"({g['confidence']}):  {g['equation']}")

    # -- mouse handling ---------------------------------------------------
    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        if self.tool == "polygon" and event.button == 3:
            self._finish_polygon(closed=False)
            return
        if event.button != 1 or event.xdata is None or event.ydata is None:
            return

        if self.tool == "erase":
            self.erasing = True
            self._erase_snapshot = self._snapshot()
            self._erase_changed = False
            self.erase_at(event.xdata, event.ydata)
        elif self.tool == "pen":
            self.drawing = True
            self.current_x = [event.xdata]
            self.current_y = [event.ydata]
        elif self.tool == "move":
            self._begin_move(event)
        elif self.tool == "polygon":
            self._polygon_click(event)
        elif self.tool in SHAPE_TOOLS:
            self._drag_start = self._point(event)

    def on_move(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        if self.tool == "erase":
            self._show_eraser_cursor(event.xdata, event.ydata)
            if self.erasing:
                self.erase_at(event.xdata, event.ydata)
        elif self.tool == "pen":
            if not self.drawing:
                return
            self.current_x.append(event.xdata)
            self.current_y.append(event.ydata)
            if self._live_line is None:
                (self._live_line,) = self.ax.plot([], [], color="tab:blue", linewidth=2)
            self._live_line.set_data(self.current_x, self.current_y)
            self.fig.canvas.draw_idle()
        elif self.tool == "move":
            if self._move_origin is not None:
                self._move_to(event)
        elif self.tool == "polygon":
            if self.poly_pts:
                self._update_poly_preview(self._point(event))
        elif self.tool in SHAPE_TOOLS:
            if self._drag_start is not None:
                built = self._build_shape(self._drag_start, self._point(event),
                                          *self._modifiers(event))
                if built is not None:
                    self._set_preview(built[0])

    def on_release(self, event):
        if self.tool == "erase":
            self.erasing = False
            if self._erase_changed and self._erase_snapshot is not None:
                self._push_undo(self._erase_snapshot)
                self.info = ""
                self._update_title()
            self._erase_snapshot = None
            self._erase_changed = False
            return

        if self.tool == "pen":
            if not self.drawing:
                return
            self.drawing = False
            x, y = list(self.current_x), list(self.current_y)
            self.current_x, self.current_y = [], []
            if self._live_line is not None:
                self._live_line.remove()
                self._live_line = None
            if len(x) > 1:
                xa, ya = np.array(x), np.array(y)
                self._commit([(xa, ya)], self._describe_pen_stroke(xa, ya))
            else:
                self.fig.canvas.draw_idle()
        elif self.tool == "move":
            self._end_move()
        elif self.tool in SHAPE_TOOLS:
            if self._drag_start is None:
                return
            start, self._drag_start = self._drag_start, None
            self._clear_preview()
            if event.xdata is None or event.ydata is None or event.inaxes != self.ax:
                return
            built = self._build_shape(start, self._point(event), *self._modifiers(event))
            if built is not None:
                self._commit(built[1], built[2])

    # -- keyboard ----------------------------------------------------------
    def on_key(self, event):
        key = event.key or ""
        if key == "c":
            self.clear()
        elif key == "e":
            self.toggle_eraser()
        elif key in SHORTCUTS:
            self.set_tool(SHORTCUTS[key])
        elif key == "s":
            self.toggle_snap()
        elif key in ("ctrl+z", "u"):
            self.undo()
        elif key in ("ctrl+y", "ctrl+shift+z", "ctrl+Z"):
            self.redo()
        elif key == "delete":
            self.delete_selected()
        elif key == "escape":
            self._cancel_in_progress()
            self._select(None)
        elif key == "enter":
            self.finish(event)

    # -- shape tools -----------------------------------------------------------
    def _build_shape(self, p0, p1, shift, alt):
        """Turn a drag p0 -> p1 into (preview_polyline, strokes, description),
        or None if the drag is degenerate."""
        if np.hypot(p1[0] - p0[0], p1[1] - p0[1]) < 1e-6:
            return None
        tool = self.tool
        if tool == "line":
            end = T.constrain_angle(p0, p1) if shift else p1
            strokes = [T.line_stroke(p0, end)]
            info = T.describe_line(p0, end)
        elif tool == "rect":
            box = T.box_from_drag(p0, p1, square=shift, from_center=alt)
            if box[2] - box[0] < 1e-6 or box[3] - box[1] < 1e-6:
                return None
            strokes = T.rectangle_strokes(*box)
            info = T.describe_rectangle(*box)
        elif tool == "circle":
            r = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
            strokes = [T.ellipse_stroke(p0[0], p0[1], r, r)]
            info = T.describe_circle(p0[0], p0[1], r)
        elif tool == "ellipse":
            box = T.box_from_drag(p0, p1, square=shift, from_center=alt)
            if box[2] - box[0] < 1e-6 or box[3] - box[1] < 1e-6:
                return None
            cx, cy, rx, ry = T.ellipse_from_box(*box)
            strokes = [T.ellipse_stroke(cx, cy, rx, ry)]
            info = T.describe_ellipse(cx, cy, rx, ry)
        else:
            return None
        return self._join(strokes), strokes, info

    @staticmethod
    def _join(strokes):
        """Concatenate strokes into one polyline with NaN gaps (for previews)."""
        xs, ys = [], []
        for sx, sy in strokes:
            xs.extend([*sx, np.nan])
            ys.extend([*sy, np.nan])
        return np.array(xs), np.array(ys)

    def _set_preview(self, xy):
        if self._preview is None:
            (self._preview,) = self.ax.plot(
                [], [], linestyle="--", color="tab:red", linewidth=1.5, zorder=6
            )
        self._preview.set_data(*xy)
        self.fig.canvas.draw_idle()

    def _clear_preview(self):
        if self._preview is not None:
            self._preview.remove()
            self._preview = None
            self.fig.canvas.draw_idle()

    def _cancel_in_progress(self):
        """Drop a half-finished polygon / shape drag (not committed strokes)."""
        self.poly_pts = []
        self._drag_start = None
        self._clear_preview()

    # -- polygon tool ------------------------------------------------------------
    def _polygon_click(self, event):
        pt = self._point(event)
        if getattr(event, "dblclick", False):
            if len(self.poly_pts) >= 3:
                self._finish_polygon(closed=True)
            return
        if len(self.poly_pts) >= 3:
            first = self.poly_pts[0]
            if np.hypot(pt[0] - first[0], pt[1] - first[1]) < self.close_tolerance:
                self._finish_polygon(closed=True)
                return
        if self.poly_pts and pt == self.poly_pts[-1]:
            return
        self.poly_pts.append(pt)
        self._update_poly_preview()

    def _update_poly_preview(self, cursor=None):
        pts = list(self.poly_pts)
        if cursor is not None:
            pts.append(cursor)
        if len(pts) >= 2:
            self._set_preview((np.array([p[0] for p in pts]),
                               np.array([p[1] for p in pts])))
        else:
            self._set_preview((np.array([pts[0][0]]), np.array([pts[0][1]])))

    def _finish_polygon(self, closed):
        pts = list(self.poly_pts)
        if (closed and len(pts) < 3) or (not closed and len(pts) < 2):
            return
        self.poly_pts = []
        self._clear_preview()
        self._commit(T.polygon_strokes(pts, closed=closed),
                     T.describe_polygon(pts, closed=closed))

    # -- move tool ------------------------------------------------------------------
    def _begin_move(self, event):
        self._sync_gids()
        hit = T.nearest_stroke(self.strokes, event.xdata, event.ydata)
        if hit is None or hit[1] > self.pick_tolerance:
            self._select(None)
            return
        self._select(self.stroke_gids[hit[0]])
        self._move_snapshot = self._snapshot()
        self._move_origin = (event.xdata, event.ydata)
        self._move_total = (0.0, 0.0)
        self._moved = False

    def _move_to(self, event):
        if self.selected_gid is None:
            return
        dx = event.xdata - self._move_origin[0]
        dy = event.ydata - self._move_origin[1]
        if self.snap:
            dx, dy = T.snap_point(dx, dy)
        step_x, step_y = dx - self._move_total[0], dy - self._move_total[1]
        if step_x == 0 and step_y == 0:
            return
        self._sync_gids()
        for i, gid in enumerate(self.stroke_gids):
            if gid == self.selected_gid:
                sx, sy = self.strokes[i]
                self.strokes[i] = (sx + step_x, sy + step_y)
        self._move_total = (dx, dy)
        self._moved = True
        self._redraw_strokes()

    def _end_move(self):
        if self._move_origin is not None and self._moved:
            self._push_undo(self._move_snapshot)
            self.info = (f"Moved by ({self._move_total[0]:+.2f}, "
                         f"{self._move_total[1]:+.2f})")
            self._update_title()
        self._move_origin = None
        self._move_snapshot = None
        self._moved = False

    def delete_selected(self):
        if self.selected_gid is None:
            return
        self._sync_gids()
        self._push_undo()
        keep = [i for i, g in enumerate(self.stroke_gids) if g != self.selected_gid]
        self.strokes = [self.strokes[i] for i in keep]
        self.stroke_gids = [self.stroke_gids[i] for i in keep]
        self.selected_gid = None
        self.info = "Deleted object"
        self._redraw_strokes()
        self._update_title()

    # -- eraser -----------------------------------------------------------
    def _show_eraser_cursor(self, x, y):
        if self.eraser_cursor is not None:
            self.eraser_cursor.remove()
        self.eraser_cursor = Circle(
            (x, y), self.eraser_radius, color="gray", alpha=0.3, zorder=10
        )
        self.ax.add_patch(self.eraser_cursor)
        self.fig.canvas.draw_idle()

    def erase_at(self, x, y):
        """Remove any drawn points within `eraser_radius` of (x, y).

        If erasing punches through the middle of a stroke, it's split
        into two separate strokes rather than being awkwardly reconnected
        across the gap. The pieces stay part of the same object, so the
        Move tool still moves them together.
        """
        self._sync_gids()
        new_strokes = []
        new_gids = []
        changed = False
        for (sx, sy), gid in zip(self.strokes, self.stroke_gids):
            dist = np.hypot(sx - x, sy - y)
            keep = dist >= self.eraser_radius
            if keep.all():
                new_strokes.append((sx, sy))
                new_gids.append(gid)
                continue
            changed = True
            # Split into contiguous runs of kept points; drop runs too
            # short to be a meaningful stroke.
            run_x, run_y = [], []
            for xi, yi, k in zip(sx, sy, keep):
                if k:
                    run_x.append(xi)
                    run_y.append(yi)
                else:
                    if len(run_x) > 1:
                        new_strokes.append((np.array(run_x), np.array(run_y)))
                        new_gids.append(gid)
                    run_x, run_y = [], []
            if len(run_x) > 1:
                new_strokes.append((np.array(run_x), np.array(run_y)))
                new_gids.append(gid)

        if changed:
            self.strokes = new_strokes
            self.stroke_gids = new_gids
            self._erase_changed = True
            self._redraw_strokes()

    def _redraw_strokes(self):
        self._sync_gids()
        for artist in self.line_artists:
            artist.remove()
        self.line_artists = []
        for (sx, sy), gid in zip(self.strokes, self.stroke_gids):
            selected = gid == self.selected_gid
            (artist,) = self.ax.plot(
                sx, sy,
                color="tab:orange" if selected else "tab:blue",
                linewidth=3 if selected else 2,
            )
            self.line_artists.append(artist)
        self.fig.canvas.draw_idle()

    # -- lifecycle ----------------------------------------------------------
    def clear(self):
        if self.strokes:
            self._push_undo()
        self.strokes = []
        self.stroke_gids = []
        self.selected_gid = None
        self._cancel_in_progress()
        self.info = ""
        self._redraw_strokes()
        self._update_title()

    def finish(self, event=None):
        if self.tool == "polygon" and len(self.poly_pts) >= 2:
            self._finish_polygon(closed=False)
        self.done = True
        self._on_close()
        plt.close(self.fig)

    def get_points(self):
        if not self.strokes:
            return np.array([]), np.array([])
        xs = np.concatenate([s[0] for s in self.strokes])
        ys = np.concatenate([s[1] for s in self.strokes])
        order = np.argsort(xs)
        return xs[order], ys[order]

    def run(self):
        """Show the board and block until the user finishes drawing."""
        plt.show()
        return self.get_points()