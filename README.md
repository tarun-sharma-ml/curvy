# Curve Guesser 📈✏️

Draw **anything** — your name, a doodle, a circle, a parabola — on an
interactive board. Curve Guesser gives you two things back:

1. **A whole-shape guess** for each stroke: if it looks like a line,
   circle, parabola, ellipse, or hyperbola, you get that curve's general
   equation with the numbers filled in (e.g. `(x-3)^2+(y+1)^2=2^2`).
2. **A piecewise reconstruction** that recreates the drawing exactly,
   stitched together from small restricted-domain lines, parabolas, and
   circular arcs — the same technique behind hand-built Desmos art — with
   an optional animated "pen drawing" reveal version.

## How it works

1. A matplotlib window opens with a grid and x/y axes.
2. Draw with the mouse — release and click again to start a new stroke.
   Toggle the **Eraser** button (or press `e`) to remove parts of what
   you've drawn; erasing through the middle of a stroke splits it into
   two rather than leaving a jump.
3. Press **Enter** or click **Fit Curve** when done.
4. The console immediately prints a **shape guess** for each stroke —
   which canonical curve it resembles and that curve's equation, tagged
   with a confidence level (excellent / good / rough / poor match).
5. A **live preview** window opens with a per-stroke automatic piece count.
   Select a stroke and adjust its piece-count slider to trade off detail
   vs. equation count.
6. Click **Confirm & Export**. Whole-stroke canonical guesses and the
   final piecewise equations print to the console and save to
   `desmos_equations.txt`, including a second, animated "pen drawing"
   version driven by a shared Desmos slider.

## Installation

```bash
git clone https://github.com/<your-username>/curve-guesser.git
cd curve-guesser
pip install -r requirements.txt
```

Or install as a package (adds a `curve-guesser` command):

```bash
pip install .
```

## Usage

```bash
python -m curve_guesser.cli
```

or, if installed as a package:

```bash
curve-guesser
```

### Browser version (no installation)

Open [`web/index.html`](./web/index.html) directly in a modern browser.
Draw one or more strokes, click **Fit drawing**, select a stroke to adjust
its piece count, then download the Desmos expressions. The browser version
is self-contained and does not need a server, package install, or network
connection. It exports static equations; animated reveal exports are
available through the Python application.

**Drawing tools** (left-hand palette, or press the key):

| Tool | Key | How it works |
|---|---|---|
| Pen | `p` | Freehand. After each stroke the board shows which canonical shape it looks like, with the equation filled in |
| Line | `l` | Drag. Hold **Shift** to lock to 15° angles |
| Rectangle | `r` | Drag corner to corner. **Shift** = square, **Alt** = from centre |
| Circle | `o` | Drag from the centre outwards |
| Ellipse | `v` | Drag the bounding box. **Shift** = circle, **Alt** = from centre |
| Polygon | `g` | Click points; double-click or click the first point to close, right-click to finish open, `Esc` to cancel |
| Move | `m` | Drag any drawn object to reposition it; `Delete` removes it |
| Eraser | `e` | Drag over strokes to rub them out (press again to return to your previous tool) |

**Other controls:**
- **Undo / Redo**: buttons, `Ctrl+Z` / `u`, and `Ctrl+Y`
- **Snap** button or `s`: snap shape corners and moves to the grid
- `c`: clear the board (undoable)
- `Enter` or **Fit Curve**: move to shape guesses + the piece-count preview

Shapes made with the tools are ordinary strokes, so they flow straight into
the fitting and Desmos export. Rectangles and polygons are added one stroke
per side so each side fits as an exact straight line.

**In the preview window:**
- Each stroke starts with the fewest pieces that reach the automatic fit
  error target; use the stroke selector and slider to override that count
- Click **Confirm & Export** when happy with it

## Shape guessing

Each stroke is checked as a *whole* against canonical curves — a
line, a circle, a parabola (either orientation), a rotated ellipse, and a
rotated hyperbola — and whichever fits best (with a
preference for the simplest shape on close calls) is reported with its
general equation and a confidence label:

```
Stroke 1: looks like a CIRCLE (excellent match)
    (x - 3.0000)^2 + (y + 1.0000)^2 = 2.0000^2
```

This is independent of the piecewise reconstruction below it — it's
meant for when you've drawn something that genuinely *is* one of these
basic shapes and want the textbook equation for it, not a 20-piece
reconstruction.

## Using the piecewise output in Desmos

`desmos_equations.txt` first lists optional whole-stroke shape guesses,
then contains named, restricted-domain equations like:

```
f_{1a}(x)=\left\{x\le0.5472:...,...\right\}\left\{0.0000\le x\le0.9540\right\}
```

Paste each line into Desmos as its own expression — named functions plot
automatically, no extra `y=f(x)` line needed. A second section in the
same file gives an **animated version**: paste the `x_1=...` slider line
first, then each stroke's reveal function/tangent-line/point group, and
press the slider's play button to watch the drawing appear stroke by
stroke, left to right.

## Project structure

```
curve-guesser/
├── curve_guesser/
│   ├── __init__.py
│   ├── board.py      # interactive drawing board + tool palette
│   ├── tools.py       # geometry behind the shape tools
│   ├── shapes.py      # whole-stroke canonical shape guesser
│   ├── pieces.py       # split a stroke into pieces, fit line/parabola/arc
│   ├── preview.py      # live slider preview + Desmos export (static + animated)
│   └── cli.py            # entry point
├── web/
│   └── index.html        # self-contained browser version
├── tests/
│   ├── test_board.py
│   ├── test_board_tools.py
│   ├── test_tools.py
│   ├── test_pieces.py
│   ├── test_shapes.py
│   └── test_animated_export.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Running tests

```bash
pip install -e ".[dev]"
pytest
```

## Notes & limitations

- The piecewise fitter: more pieces = tighter match but more equations.
  It automatically chooses a count per stroke using a normalized error
  target; the preview lets you override each stroke independently.
  Adjacent pieces are forced to share exact endpoints (no gaps), with an
  exact solve for line/quadratic pieces and an averaged-translation
  correction for circular arcs.
- The animated reveal only drives horizontal (y=f(x)) strokes directly
  with the shared slider; vertical (x=f(y)) strokes are gated to appear
  once the slider passes their x-position, so the canvas still starts
  blank overall.
- Conic fits support rotation, but the browser edition uses a lightweight
  algebraic fit and can be less stable on very short, noisy, or partial
  strokes than the Python fitter.

## License

MIT — see [LICENSE](LICENSE).