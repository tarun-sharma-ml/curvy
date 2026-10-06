"""Interactive matplotlib drawing board for sketching a curve."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Button


class DrawingBoard:
    """A grid + axes canvas that captures freehand mouse strokes, with an
    eraser tool for removing parts of what you've drawn.
    """

    def __init__(self, xlim=(-10, 10), ylim=(-10, 10), eraser_radius=None):
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        manager = self.fig.canvas.manager
        if manager is not None:
            manager.set_window_title(
                "Draw your curve  (drag=draw | 'e'=eraser | 'c' clear | Enter/Button = fit)"
            )
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.ax.axhline(0, color="black", linewidth=1)
        self.ax.axvline(0, color="black", linewidth=1)
        self.ax.grid(True, linestyle="--", alpha=0.4)
        self.ax.set_aspect("equal")

        # Eraser radius defaults to ~3% of the smaller axis span.
        span = min(xlim[1] - xlim[0], ylim[1] - ylim[0])
        self.eraser_radius = eraser_radius if eraser_radius is not None else 0.03 * span

        self.strokes = []  # list of (xs, ys) arrays, committed strokes
        self.current_x = []
        self.current_y = []
        self.drawing = False
        self.erasing = False
        self.mode = "draw"  # "draw" or "erase"
        self.line_artists = []  # matplotlib artists for committed strokes
        self.eraser_cursor = None  # visual circle showing eraser reach

        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_move)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        fit_ax = self.fig.add_axes((0.55, 0.01, 0.2, 0.05))
        self.fit_button = Button(fit_ax, "Fit Curve")
        self.fit_button.on_clicked(self.finish)

        eraser_ax = self.fig.add_axes((0.25, 0.01, 0.2, 0.05))
        self.eraser_button = Button(eraser_ax, "Eraser: off")
        self.eraser_button.on_clicked(self.toggle_eraser)

        self.done = False
        self._update_title()

    # -- mode handling ---------------------------------------------------
    def toggle_eraser(self, event=None):
        self.mode = "erase" if self.mode == "draw" else "draw"
        self.eraser_button.label.set_text(
            "Eraser: on" if self.mode == "erase" else "Eraser: off"
        )
        self._update_title()
        self.fig.canvas.draw_idle()

    def _update_title(self):
        if self.mode == "erase":
            self.ax.set_title("ERASER mode -- drag over strokes to remove them")
        else:
            self.ax.set_title(
                "Draw a curve with the mouse, then press Enter or click 'Fit Curve'"
            )

    # -- mouse handling ---------------------------------------------------
    def on_press(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        if self.mode == "erase":
            self.erasing = True
            self.erase_at(event.xdata, event.ydata)
        else:
            self.drawing = True
            self.current_x = [event.xdata]
            self.current_y = [event.ydata]

    def on_move(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        if self.mode == "erase":
            self._show_eraser_cursor(event.xdata, event.ydata)
            if self.erasing:
                self.erase_at(event.xdata, event.ydata)
            return

        if not self.drawing:
            return
        self.current_x.append(event.xdata)
        self.current_y.append(event.ydata)
        (artist,) = self.ax.plot(
            self.current_x, self.current_y, color="tab:blue", linewidth=2
        )
        self.line_artists.append(artist)
        self.fig.canvas.draw_idle()

    def on_release(self, event):
        if self.mode == "erase":
            self.erasing = False
            return
        if not self.drawing:
            return
        self.drawing = False
        if len(self.current_x) > 1:
            self.strokes.append((np.array(self.current_x), np.array(self.current_y)))
        self.current_x, self.current_y = [], []

    def on_key(self, event):
        if event.key == "c":
            self.clear()
        elif event.key == "e":
            self.toggle_eraser()
        elif event.key == "enter":
            self.finish(event)

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
        across the gap.
        """
        new_strokes = []
        changed = False
        for sx, sy in self.strokes:
            dist = np.hypot(sx - x, sy - y)
            keep = dist >= self.eraser_radius
            if keep.all():
                new_strokes.append((sx, sy))
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
                    run_x, run_y = [], []
            if len(run_x) > 1:
                new_strokes.append((np.array(run_x), np.array(run_y)))

        if changed:
            self.strokes = new_strokes
            self._redraw_strokes()

    def _redraw_strokes(self):
        for artist in self.line_artists:
            artist.remove()
        self.line_artists = []
        for sx, sy in self.strokes:
            (artist,) = self.ax.plot(sx, sy, color="tab:blue", linewidth=2)
            self.line_artists.append(artist)
        self.fig.canvas.draw_idle()

    # -- lifecycle ----------------------------------------------------------
    def clear(self):
        self.strokes = []
        for artist in self.line_artists:
            artist.remove()
        self.line_artists = []
        self.fig.canvas.draw_idle()

    def finish(self, event):
        self.done = True
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