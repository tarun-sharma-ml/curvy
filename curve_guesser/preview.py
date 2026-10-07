"""Interactive preview: start with automatic per-stroke fits, allow manual
piece-count overrides, then confirm to export the Desmos equations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

from .pieces import fit_stroke_pieces, fit_stroke_auto, evaluate_piece, group_pieces, group_to_desmos, to_desmos_animated, to_desmos_string
from .shapes import guess_shape

KIND_COLORS = {"line": "tab:blue", "quadratic": "tab:green", "circle": "tab:red"}


class Previewer:
    def __init__(self, strokes, initial_pieces=None, max_pieces=40):
        self.strokes = strokes
        self.max_pieces = max_pieces
        auto_results = [
            fit_stroke_auto(x, y, max_pieces=max_pieces)
            for x, y in strokes
        ]
        self.auto_piece_counts = [count for count, _fits in auto_results]
        self.piece_counts = (
            [max(1, min(initial_pieces, max_pieces))] * len(strokes)
            if initial_pieces is not None else list(self.auto_piece_counts)
        )
        self.selected_stroke = 1
        self.confirmed = False
        self.final_fits = None

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        manager = self.fig.canvas.manager
        if manager is not None:
            manager.set_window_title(
                "Preview: drag the slider, then click Confirm & Export"
            )
        plt.subplots_adjust(bottom=0.29)

        self.stroke_slider = None
        if len(strokes) > 1:
            stroke_ax = self.fig.add_axes((0.2, 0.2, 0.6, 0.025))
            self.stroke_slider = Slider(
                stroke_ax, f"Stroke (1-{len(strokes)})", 1, len(strokes),
                valinit=1, valstep=1,
            )
            self.stroke_slider.on_changed(self.on_stroke_change)
        else:
            self.fig.text(0.2, 0.2, "Stroke 1", ha="left", va="center")

        slider_ax = self.fig.add_axes((0.2, 0.12, 0.6, 0.025))
        self.slider = Slider(
            slider_ax, "Pieces for stroke 1", 1, max_pieces,
            valinit=self.piece_counts[0], valstep=1,
        )
        self.slider.on_changed(self.on_slider_change)

        button_ax = self.fig.add_axes((0.4, 0.035, 0.25, 0.05))
        self.confirm_button = Button(button_ax, "Confirm & Export")
        self.confirm_button.on_clicked(self.on_confirm)

        self._auto_fits = [fits for _count, fits in auto_results]
        self.redraw()

    def _compute_fits(self):
        fits = []
        for i, ((x, y), count) in enumerate(zip(self.strokes, self.piece_counts)):
            if count == self.auto_piece_counts[i]:
                fits.append(self._auto_fits[i])
            else:
                usable = max(1, min(count, len(x) // 3))
                fits.append(fit_stroke_pieces(x, y, usable))
        return fits

    def redraw(self):
        self.ax.clear()
        for x, y in self.strokes:
            self.ax.plot(x, y, color="lightgray", linewidth=4, zorder=1)

        fits = self._compute_fits()
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
        selected = self.selected_stroke - 1
        count = self.piece_counts[selected]
        automatic = self.auto_piece_counts[selected]
        self.ax.set_title(
            f"{total_pieces} pieces total  |  avg error: {avg_error:.4f}\n"
            f"Stroke {self.selected_stroke}: {count} pieces (automatic: {automatic})  "
            "|  blue=line  green=parabola  red=circular arc"
        )
        self.slider.label.set_text(f"Pieces for stroke {self.selected_stroke}")
        self.ax.set_aspect("equal")
        self.ax.grid(True, linestyle="--", alpha=0.4)
        self.fig.canvas.draw_idle()
        self._last_fits = fits

    def on_stroke_change(self, val):
        self.selected_stroke = int(val)
        self.slider.set_val(self.piece_counts[self.selected_stroke - 1])

    def on_slider_change(self, val):
        self.piece_counts[self.selected_stroke - 1] = int(val)
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
