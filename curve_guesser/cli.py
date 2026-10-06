"""Command-line entry point for Curve Guesser (piecewise / Desmos edition)."""

import os

from .board import DrawingBoard
from .preview import Previewer, build_desmos_export
from .shapes import guess_shape


def print_shape_guesses(strokes):
    """Look at each whole stroke and guess which canonical curve (line,
    circle, parabola, ellipse, hyperbola) it most resembles, filling in
    that curve's general equation with numbers fitted to where it was
    actually drawn.
    """
    print("\n" + "=" * 70)
    print("SHAPE GUESSES (whole-stroke)")
    print("=" * 70)
    for i, (x, y) in enumerate(strokes, start=1):
        g = guess_shape(x, y)
        print(f"Stroke {i}: looks like a {g['kind'].upper()} ({g['confidence']})")
        print(f"    {g['equation']}")
    print()


def main():
    board = DrawingBoard()
    board.run()

    if not board.strokes:
        print("Nothing was drawn. Please run again and draw something.")
        return

    print_shape_guesses(board.strokes)

    previewer = Previewer(board.strokes)
    final_fits = previewer.run()

    if not previewer.confirmed or final_fits is None:
        print("Preview closed without confirming -- nothing exported.")
        return

    export_text = build_desmos_export(final_fits, board.strokes)
    out_path = os.path.join(os.getcwd(), "desmos_equations.txt")
    with open(out_path, "w") as f:
        f.write(export_text)

    print(export_text)
    print(f"\n(Saved to {out_path})")


if __name__ == "__main__":
    main()
