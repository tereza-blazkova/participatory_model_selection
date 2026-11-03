"""
THE style of every paper figure: canvas, type sizes, font, method colours.

Every draw script gets it through utils.py, and the pipeline's own plot modules
through src/paper_style.py, which loads this file. Nothing else sets rcParams.
It imports matplotlib only, so figures/ still re-renders with no pipeline present.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

FONTS = Path(__file__).resolve().parent.parent / "fonts"

# ── Canvas size ───────────────────────────────────────────────────────────────
#
# matplotlib bakes font sizes into the PDF as absolute points, so a figure's
# printed type size is its designed size times how much \includegraphics
# scales it:
#
#     printed_pt = designed_pt x (printed width / canvas width)
#
# Which means the only way several figures print at the same type size is for
# all of them to have the same canvas-to-printed ratio. They did not: canvases
# were picked per figure (13, 9, 8.6, 8, 5 in) and the same 9 pt label printed
# anywhere between 4.1 pt and 8.6 pt across the paper.
#
# One ratio now, set by the figure that needs the most canvas per point of type.
# That is dropout_timing: (when measured) five panels side by side, each with its own four-line
# legend above it, and the legends are wider than the panels. Measured, by
# rendering it and looking:
#
#     13.0 in  clean (what it was drawn at)
#     11.0 in  clean                     <- the floor
#     10.5 in  x tick labels collide
#     10.0 in  tick labels run together
#      9.5 in  axes collapse to strips
#
# model_comparison, the next densest, is clean down to 9.5 in. So dropout_timing
# sets the size of the type in every figure in the paper: shrink its legend —
# one shared legend, or shorter labels — and CANVAS_SCALE can come down and
# every figure's type grows.

TEXTWIDTH_IN = 430.00462 / 72.27      # \textwidth of the paper
CANVAS_SCALE = 11.0 / TEXTWIDTH_IN    # canvas = printed width x this  (~1.85)


def canvas(aspect: float, width_fraction: float = 1.0) -> list[float]:
    """
    Canvas size in inches for a figure the paper prints at
    `width_fraction` x \textwidth, keeping `aspect` (height / width).

    Every figure goes through here, so changing CANVAS_SCALE changes the whole
    paper's figure type size at once and keeps it uniform.
    """
    width = TEXTWIDTH_IN * width_fraction * CANVAS_SCALE
    return [round(width, 4), round(width * aspect, 4)]


# ── Type sizes ────────────────────────────────────────────────────────────────
# One size per kind of text, for every figure, in the points it PRINTS at in
# the paper. Every draw script takes its sizes from FONT by role and never
# types a number. FONT is the same table on the
# canvas, where sizes are CANVAS_SCALE times larger because LaTeX shrinks the
# canvas back to \textwidth.

FONT_PRINTED = {
    "tick":   6.0,     # tick labels
    "legend": 6.0,     # legend entries
    "label":  6.5,     # axis labels
    "title":  6.5,     # panel titles
    "panel":  8.0,     # the bold panel letter
    "annot":  5.5,     # numbers on bars, cells and points; notes
    "side":   4.5,     # the rotated Dropout / Non-dropout names beside a stacked
                       #   bar's dashed line: the dropout band can be too short
                       #   for "Dropout" at the annot size
}
FONT = {role: round(pt * CANVAS_SCALE, 2) for role, pt in FONT_PRINTED.items()}


# ── Style ─────────────────────────────────────────────────────────────────────

_RCPARAMS = {
    "font.family":       "sans-serif",
    "font.size":         FONT["tick"],
    "axes.titlesize":    FONT["title"],
    "axes.labelsize":    FONT["label"],
    "xtick.labelsize":   FONT["tick"],
    "ytick.labelsize":   FONT["tick"],
    "legend.fontsize":   FONT["legend"],
    "legend.title_fontsize": FONT["legend"],
    "figure.titlesize":  FONT["label"],
    "figure.labelsize":  FONT["label"],
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth":   1.8,
    "lines.markersize":  6,
    "legend.frameon":    False,
    "legend.framealpha": 0,
    "legend.edgecolor":  "none",
    "pdf.fonttype":      42,   # real glyphs in the vector PDF, not Type-3 outlines
}


def apply_style() -> None:
    """
    The house style, applied once per script.

    Helvetica is licensed and not distributed here: put Helvetica.ttf and
    Helvetica-Bold.ttf in figures/fonts/ if you have them. A missing font falls back with a printed warning rather than
    silently shipping a different face — matplotlib's font lookup never fails, so
    without the warning the substitution is invisible until the PDF is in the
    paper.
    """
    plt.style.use("seaborn-v0_8-white")
    ## The fallback chain is not a free choice: matplotlib scores a FontProperties
    ## across the WHOLE family list, so an extra name in it can change which face
    ## is selected even when the first name resolves. An earlier version of this
    ## function carried one more family than src/plotting.py, and every glyph in
    ## every figure moved by a fraction of a pixel. These two lists must agree.
    sans = ["Arial", "Helvetica Neue", "DejaVu Sans"]
    ttf = FONTS / "Helvetica.ttf"
    if ttf.exists():
        try:
            fm.fontManager.addfont(str(ttf))
            sans = ["Helvetica", "Arial", "DejaVu Sans"]
        except Exception as exc:                                    # noqa: BLE001
            print(f"  ! Helvetica not loaded ({exc}) — falling back to {sans[0]}")
    else:
        print(f"  ! no {ttf} — falling back to {sans[0]}")
    ## The bold face is its own file. Without it every fontweight="bold" (the
    ## panel letters, the cell values) silently renders in the regular weight.
    bold = FONTS / "Helvetica-Bold.ttf"
    if ttf.exists() and bold.exists():
        fm.fontManager.addfont(str(bold))
    elif ttf.exists():
        print(f"  ! no {bold} — bold text will render in the regular weight")
    mpl.rcParams.update(_RCPARAMS | {"font.sans-serif": sans})



# ── Method colours ────────────────────────────────────────────────────────────
# One colour per method, the same in every figure: the palette of the model
# comparison figure. The Participatory model is bluish green and Max ML the NN
# pink in every figure, the FA frontiers included.

ML_MODELS = ["fasterrisk", "logistic_regression", "xgboost", "mlp"]
METHOD_COLOR = {
    "credits_policy":       "#D55E00",
    "operational_rule":     "#0072B2",
    "fasterrisk":           "#56C4B4",
    "logistic_regression":  "#3A8FA8",
    "xgboost":              "#5A5FA6",
    "mlp":                  "#C4547A",
    "participatory":        "#009E73",
    "max_ml":               "#C4547A",   # Max ML is the NN, so it keeps the NN colour
}
METHOD_MARKER = {"fasterrisk": "o", "logistic_regression": "s", "xgboost": "^",
                 "mlp": "v"}
