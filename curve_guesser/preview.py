"""Interactive preview: pick how many pieces per stroke, see the fit update
live, then confirm to export the Desmos equations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

from .pieces import fit_stroke_pieces, evaluate_piece, group_pieces, group_to_desmos, to_desmos_animated, to_desmos_string
from .shapes import guess_shape

KIND_COLORS = {"line": "tab:blue", "quadratic": "tab:green", "circle": "tab:red"}


class Previewer:
    def __init__(self, strokes, initial_pieces=6, max_pieces=40):
        self.strokes = strokes
        self.max_pieces = max_pieces
        self.num_pieces = initial_pieces
        self.confirmed = False
        self.final_fits = None

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        manager = self.fig.canvas.manager
        if manager is not None:
            manager.set_window_title(
                "Preview: drag the slider, then click Confirm & Export"
            )
        plt.subplots_adjust(bottom=0.2)

        slider_ax = self.fig.add_axes((0.2, 0.1, 0.6, 0.03))
        self.slider = Slider(
            slider_ax, "Pieces per stroke", 1, max_pieces,
            valinit=initial_pieces, valstep=1,
        )
        self.slider.on_changed(self.on_slider_change)

        button_ax = self.fig.add_axes((0.4, 0.02, 0.25, 0.05))
        self.confirm_button = Button(button_ax, "Confirm & Export")
        self.confirm_button.on_clicked(self.on_confirm)

        self.redraw()

    def _compute_fits(self, num_pieces):
        fits = []
        for x, y in self.strokes:
            usable = max(1, min(num_pieces, len(x) // 3))
            fits.append(fit_stroke_pieces(x, y, usable))
        return fits

    def redraw(self):
        self.ax.clear()
        for x, y in self.strokes:
            self.ax.plot(x, y, color="lightgray", linewidth=4, zorder=1)

        fits = self._compute_fits(self.num_pieces)
        total_error = 0.0
        total_pieces = 0
        legend_kinds = set()
        for piece_list in fits:
            for piece in piece_list:
                px, py = evaluate_piece(piece)
                color = KIND_COLORS[piece["kind"]]
                label = piece["kind"] if piece["kind"] not in legend_kinds else None
                legend_kinds.add(piece["kind"])
                self.ax.plot(px, py, color=color, linewidth=2, zorder=2, label=label)
                total_error += piece["error"]
                total_pieces += 1

        avg_error = total_error / max(total_pieces, 1)
        self.ax.set_title(
            f"{total_pieces} pieces total  |  avg error: {avg_error:.4f}\n"
            "blue=line  green=parabola  red=circular arc"
        )
        self.ax.set_aspect("equal")
        self.ax.grid(True, linestyle="--", alpha=0.4)
        self.fig.canvas.draw_idle()
        self._last_fits = fits

    def on_slider_change(self, val):
        self.num_pieces = int(val)
        self.redraw()

    def on_confirm(self, event):
        self.confirmed = True
        self.final_fits = self._last_fits
        plt.close(self.fig)

    def run(self):
        plt.show()
        return self.final_fits


def build_desmos_export(final_fits, strokes=None):
    lines = []
    lines.append("=" * 70)
    lines.append("DESMOS EQUATIONS")
    lines.append("=" * 70)
    lines.append(
        "Whole-stroke guesses are optional canonical equations. The\n"
        "restricted piecewise equations below trace your drawing; paste\n"
        "each into Desmos as its own expression.\n"
    )
    if strokes is None:
        strokes = []
        for piece_list in final_fits:
            sampled = [evaluate_piece(piece, n=40) for piece in piece_list]
            if sampled:
                strokes.append((
                    np.concatenate([x for x, _y in sampled]),
                    np.concatenate([y for _x, y in sampled]),
                ))

    lines.append("-" * 70)
    lines.append("WHOLE-STROKE SHAPE GUESSES (optional; canonical equations)")
    lines.append("-" * 70)
    for s_idx, (x, y) in enumerate(strokes, start=1):
        guess = guess_shape(x, y)
        lines.append(
            f"# Stroke {s_idx}: {guess['kind']} ({guess['confidence']}, "
            f"error={guess['error']:.4f})"
        )
        if guess["kind"] != "unknown":
            lines.append(guess["equation"])
    lines.append("")

    total_equations = 0
    total_pieces = 0
    letters = "abcdefghijklmnopqrstuvwxyz"

    for s_idx, piece_list in enumerate(final_fits, start=1):
        groups = group_pieces(piece_list)
        lines.append("-" * 70)
        lines.append(f"STROKE {s_idx}  ({len(piece_list)} pieces, {len(groups)} equations)")
        lines.append("-" * 70)
        for g_idx, group in enumerate(groups):
            name = f"f_{{{s_idx}{letters[g_idx % len(letters)]}}}"
            eq = group_to_desmos(group, name)
            avg_err = sum(p["error"] for p in group["pieces"]) / len(group["pieces"])
            lines.append(f"# {len(group['pieces'])} piece(s), avg error={avg_err:.4f}")
            lines.append(eq)
            total_equations += 1
            total_pieces += len(group["pieces"])
        lines.append("")

    lines.append(f"Total: {total_pieces} pieces merged into {total_equations} equations")
    lines.append("")
    lines.append(build_animated_export(final_fits))
    return "\n".join(lines)


def build_animated_export(final_fits):
    lines = []
    lines.append("=" * 70)
    lines.append("ANIMATED (PEN-DRAWING) VERSION")
    lines.append("=" * 70)
    lines.append(
        "Paste these in instead of (or alongside) the equations above for\n"
        "a progressive drawing animation. Add the slider line first, then\n"
        "click its play button (or drag it) to watch the drawing appear.\n"
    )

    all_groups = []
    for piece_list in final_fits:
        all_groups.extend(group_pieces(piece_list))
    x_groups = [g for g in all_groups if g["orientation"] == "x"]
    y_groups = [g for g in all_groups if g["orientation"] == "y"]

    if not x_groups:
        lines.append("(No horizontal-orientation strokes to animate.)")
        return "\n".join(lines)

    global_lo = min(g["lo"] for g in x_groups)
    global_hi = max(g["hi"] for g in x_groups)
    slider_var = "x_{1}"
    lines.append(f"{slider_var}={global_lo:.4f}")
    lines.append(
        "# ^ the reveal slider -- click the circle next to it in Desmos\n"
        "#   and press play, or drag it from left to right by hand"
    )
    lines.append("")

    reveal_letters = "fghijklmnopqrstuvwxyz"
    for idx, group in enumerate(x_groups):
        reveal_name = f"{reveal_letters[idx % len(reveal_letters)]}_{{{idx}}}"
        exprs = to_desmos_animated(group, reveal_name)
        if exprs is not None:
            lines.extend(exprs)
        lines.append("")

    if y_groups:
        lines.append(
            "# Vertical strokes -- gated by the SAME slider (x_1): each one\n"
            "# stays hidden until x_1 reaches its x-position, then pops in\n"
            "# fully, so the canvas still starts blank overall."
        )
        for idx, group in enumerate(y_groups):
            xs_concat = []
            for p in group["pieces"]:
                px, _py = evaluate_piece(p, n=5)
                xs_concat.extend(px)
            approx_x = float(np.mean(xs_concat))
            gate = f"\\left\\{{{slider_var}\\ge{approx_x:.4f}\\right\\}}"

            name = f"v_{{{idx}}}"
            if len(group["pieces"]) == 1:
                base_eq = to_desmos_string(group["pieces"][0])
            else:
                base_eq = group_to_desmos(group, name)
            lines.append(base_eq + gate)
        lines.append("")

    return "\n".join(lines)
