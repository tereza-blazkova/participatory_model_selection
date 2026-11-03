"""
House style and I/O for the paper figures. Nothing else decides how a figure looks.

A draw script chooses *what* to draw; font, sizes, palette, markers, output paths
and the save call all live here, so no two figures can disagree on house style.

This module imports pandas and matplotlib and NOTHING from src/, models/ or
paths/. Neither does any draw_*.py. That wall is the point of the layout: a draw
script reads figures/input/ and cannot reach the analysis frame, the per-student
predictions, or any number the disclosure rule suppressed — not by policy, but
because the code to reach them is not imported. It also means figures/ re-renders
on a machine that has the paper but not the pipeline.

Paths are derived from this file, never from paths/paths.py, so the whole
figures/ tree can be copied elsewhere and still render.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties

FIGURES = Path(__file__).resolve().parent.parent
INPUT   = FIGURES / "input"
OUTPUT  = FIGURES / "output"
FONTS   = FIGURES / "fonts"

# ── Canvas and style ──────────────────────────────────────────────────────────
# In paper_style.py, the one module that sets rcParams; re-exported here.

from paper_style import (                                            # noqa: E402,F401
    CANVAS_SCALE, FONT, FONT_PRINTED, METHOD_COLOR, METHOD_MARKER, ML_MODELS, TEXTWIDTH_IN,
    apply_style, canvas,
)


# ── Palette ───────────────────────────────────────────────────────────────────
# The four ML models, in the order every figure shows them: 4-element parallel lists.
# src/plotting.py keeps the same lists: the pipeline's own diagnostic plots use
# the same palette and the two must not drift.

MODEL_COLORS = ["#56C4B4", "#3A8FA8", "#5A5FA6", "#C4547A"]
MARKERS      = ["o", "s", "^", "v"]
RULE_COLOR   = "#D55E00"   # vermillion (Wong) — the credits-policy baseline

# Outcome breakdown: dropout is orange, non-dropout sky blue (both Wong), each
# in a full and a light shade for flagged vs not flagged.
DROPOUT_COLOR    = "#E69F00"   # flagged dropouts (TP)
NONDROPOUT_COLOR = "#56B4E9"   # flagged non-dropouts (FP)
DROPOUT_LIGHT    = "#F8E3B3"   # unflagged dropouts (FN)
NONDROPOUT_LIGHT = "#CCE9F8"   # unflagged non-dropouts (TN)
SUPPRESSED_BLOCK = "0.88"      # a same-outcome pair merged because one is small

# One colour per thing a figure can plot, keyed by what it IS rather than where
# it sits, so a series keeps its colour across every panel and every figure.
# The prepared input names the key; this maps it to a hex.
UV_COLOR      = "#56B4E9"   # sky blue (Wong) — FasterRisk on the counsellors' variables
FACULTY_COLOR = {"A": "#0072B2", "B": "#FFC107"}   # never the faculty codes
HIGHLIGHT_EDGE = "#000000"  # the border marking the Participatory and Max ML bars

# Dropout-timing layers: the reference curve, and the two rules read against it.
ALL_COLOR       = "gray"
WORKSHOP_COLOR  = "#009E73"   # bluish green (Wong) — the Participatory model
SUBGROUP_COLORS = ["#5C3D9E", "#FFC107"]   # the two halves of a binary axis

# Performance-by-period: one colour and marker per metric, held across all four
# panels so a metric is recognisable wherever it appears.
METRIC_COLOR = {
    "auc":       "#000000",
    "accuracy":  "#E69F00",
    "recall":    "#CC79A7",
    "precision": "#009E73",
    "tnr":       "#0072B2",
}
METRIC_MARKER = {"auc": "o", "accuracy": "s", "recall": "D", "precision": "^", "tnr": "v"}

# Fairness-accuracy frontier: each object on the plane has one colour.
FA_COLOR = {
    "unconstrained": "#0072B2",   # blue      — the frontier over all algorithms
    "permissible":   "#D55E00",   # vermilion — the workshop-permissible one
    "band":          "#E69F00",   # orange    — the gap between them
    "cloud":         "#999999",   # grey      — the candidate algorithms
    "credits":       "#000000",   # black     — the credits policy as deployed
    "participatory": "#009E73",   # bluish green — as in the model comparison
    "max_ml":        "#C4547A",   # the NN pink — as in the model comparison
}
FA_MARKER = {"credits": "D", "participatory": "^", "max_ml": "o"}

# The FasterRisk scorecard: which way a term moves the risk.
SCORE_POS_COLOR = "#56C4B4"   # teal — raises dropout risk
SCORE_NEG_COLOR = "#C4547A"   # pink — lowers it

SERIES_COLOR = {
    "all":                  ALL_COLOR,
    "max_ml":               MODEL_COLORS[3],   # Max ML is the NN, so it keeps the NN colour
    "credits_policy_60":    RULE_COLOR,
    "participatory":        WORKSHOP_COLOR,
    "subgroup0":            SUBGROUP_COLORS[0],
    "subgroup1":            SUBGROUP_COLORS[1],
    "credits_policy":       RULE_COLOR,
    ## One colour for both faculties: the panel already says which faculty.
    "operational_rule":     FACULTY_COLOR["A"],
    "fasterrisk_used_vars": UV_COLOR,
    "fasterrisk":           MODEL_COLORS[0],
    "logistic_regression":  MODEL_COLORS[1],
    "xgboost":              MODEL_COLORS[2],
    "mlp":                  MODEL_COLORS[3],
}
MODEL_MARKER = {"fasterrisk": "o", "logistic_regression": "s", "xgboost": "^",
                "mlp": "v"}

# Annotation, not a quantity: the selection ring is the one black mark on the
# panel, so it cannot be read as a fifth series.
SELECTED_COLOR = "#333333"
BAND_COLOR     = "#EFEFEF"   # the never-permissible band
BAND_TEXT      = "#6E6E6E"
FADE           = 0.30        # hard-blocked cells, shown for reference only
STRUCK_HATCH   = "#C4C4C4"   # 45-degree hatching over a model no rule lets through
DEMAND_DASH    = (0, (1, 1.8))
NEUTRAL        = "#3F3F3F"   # legend keys that stand for any model


# ── I/O ───────────────────────────────────────────────────────────────────────

def read_input(stem: str) -> tuple[pd.DataFrame, dict]:
    """
    One figure's prepared input: the drawn values (.csv) and its scalars (.json).

    Refuses rather than autoscaling from whatever is there. An input file is a run
    product of the pipeline, so a fresh checkout has none and the
    fix is always the same: run the matching prepare script.

    `text_columns` in the .json names the columns that are LABELS, not numbers,
    and they are read as strings. A column of labels like "0.99", "1.00" is all
    digits, so pandas would type it float and the figure would print "1.0"
    where the label reads "1.00".
    """
    csv, meta = INPUT / f"{stem}.csv", INPUT / f"{stem}.json"
    missing = [p.name for p in (csv, meta) if not p.exists()]
    if missing:
        raise SystemExit(
            f"missing figures/input/{', '.join(missing)}\n"
            f"  build it with the matching figures/scripts/prepare_*.py, on a machine\n"
            f"  with a finished results/ — see figures/README.md"
        )
    spec = json.loads(meta.read_text())
    frame = pd.read_csv(csv, dtype={c: str for c in spec.get("text_columns", [])},
                        keep_default_na=True)
    for col in spec.get("text_columns", []):
        if col in frame.columns:
            frame[col] = frame[col].fillna("")
    return frame, spec


def text_fits(ax, text: str, *, width_data: float, fontsize: float,
              height_data: float | None = None, linespacing: float = 1.3,
              pad_pt: float = 1.0) -> bool:
    """
    Would `text` fit inside a bar `width_data` wide — and, with `height_data`,
    a segment that tall — on this axes?

    A segment name in a stacked bar needs horizontal room, not just a tall
    enough segment: a long label can be wider than a narrow bar. Measuring the
    rendered string against the bar answers that directly, and keeps answering
    it when a canvas changes.

    Height is measured the same way as width, on the text as it would be drawn,
    so a segment shows the fullest label it has room for.

    The axes transform has to be final when this is called, so the caller draws
    the figure and freezes the layout first.
    """
    renderer = ax.figure.canvas.get_renderer()
    probe = ax.text(0, 0, text, fontsize=fontsize, linespacing=linespacing,
                    ha="center", va="center")
    box = probe.get_window_extent(renderer)
    probe.remove()
    pad = pad_pt * ax.figure.dpi / 72
    (x0, y0), (x1, y1) = ax.transData.transform([(0.0, 0.0), (width_data, height_data or 0.0)])
    if box.width + 2 * pad > abs(x1 - x0):
        return False
    return height_data is None or box.height + 2 * pad <= abs(y1 - y0)


def wrap_label(label: str) -> str:
    """`label` broken at its last hyphen or space, so a long name can stay on a
    narrow bar instead of being dropped for its share alone."""
    for sep in ("-", " "):
        head, found, tail = label.rpartition(sep)
        if found:
            return f"{head}{sep if sep == '-' else ''}\n{tail}"
    return label


def band_labels(ax, x: float, line_y: float, upper: str, lower: str,
                pad_pt: float = 2.0, **kwargs) -> list:
    """
    The rotated names of the two blocks a stacked bar's dashed line separates,
    each kept entirely on its own side of the line.

    Each name is centred in its band when it fits there. When it does not — the
    dropout band is often a tenth of the axis — it is anchored `pad_pt` off the
    line and runs away from it, past the axis edge if need be, rather than
    across the line. Measured on the drawn text, so the layout must be final.
    """
    renderer = ax.figure.canvas.get_renderer()
    pad = pad_pt * ax.figure.dpi / 72
    line_disp = ax.transData.transform((x, line_y))[1]
    to_data = ax.transData.inverted()
    style = dict(ha="center", va="center", fontsize=FONT["side"], color="#555555",
                 rotation=90, zorder=5)
    style.update(kwargs)
    texts = []
    for text, y_mid, side in ((upper, (1 + line_y) / 2, +1), (lower, line_y / 2, -1)):
        t = ax.text(x, y_mid, text, **style)
        box = t.get_window_extent(renderer)
        crosses = (box.y0 < line_disp + pad) if side > 0 else (box.y1 > line_disp - pad)
        if crosses:
            t.set_va("bottom" if side > 0 else "top")
            t.set_y(to_data.transform((0, line_disp + side * pad))[1])
        texts.append(t)
    return texts


def panel_label(ax, label: str, xoff: float = -0.10, yoff: float = 1.10,
                fontsize: float | None = None) -> None:
    """The bold lowercase letter at a panel's top-left."""
    ax.text(xoff, yoff, label, transform=ax.transAxes,
            fontsize=fontsize or FONT["panel"], fontweight="bold", va="top")


def note(ax, text: str | None, loc: str = "above", **kwargs):
    """
    The suppression note, placed where it cannot sit on data and kept out of the
    layout engine so a long note never shrinks the axes. `text` arrives in the
    prepared input as a finished sentence, and None means nothing was withheld.
    """
    if not text:
        return None
    xy = {"above":       (1.0, 1.01, "right", "bottom"),
          "lower right": (0.99, 0.01, "right", "bottom"),
          "upper right": (0.99, 0.99, "right", "top"),
          "lower left":  (0.01, 0.01, "left", "bottom"),
          "upper left":  (0.01, 0.99, "left", "top")}[loc]
    style = dict(transform=ax.transAxes, ha=xy[2], va=xy[3], fontsize=FONT["annot"], color="0.35")
    style.update(kwargs)
    t = ax.text(xy[0], xy[1], text, **style)
    t.set_in_layout(False)
    return t


## Internal model / feature-set keys, which must never be printed on a figure:
## the paper says "Neural network" and "(d) + LMS", not `mlp` and `d_plus_lms`. A copy of
## src/display.INTERNAL_KEYS (this module imports nothing from src/), and the
## two must list the same keys. The paper writes "enrollment", never "enrolment".
INTERNAL_KEYS = (
    "mlp", "a_credits", "b_credits_performance", "c_plus_enrolment",
    "d_plus_lms", "e_plus_sensitive", "b_plus_sensitive",
    "logistic_regression", "credits_policy", "ects_behind_ge_60",
)
_LEAK = re.compile(r"(?<![A-Za-z])(" + "|".join(INTERNAL_KEYS) + r")(?![A-Za-z])"
                   r"|[Ee]nrolment")


def leaked_names(fig) -> list[str]:
    """Every drawn text (labels, ticks, legends, titles, notes) that prints an internal key."""
    return sorted({t.get_text() for t in fig.findobj(mpl.text.Text)
                   if t.get_visible() and _LEAK.search(t.get_text())})


def text_overlaps(fig, min_pt: float = 1.0) -> list[tuple[str, str]]:
    """
    Pairs of drawn texts whose boxes overlap by more than `min_pt` in both
    directions: tick labels, legend entries, labels on bars and cells, notes.
    A slanted label is measured by its upright box, so it can be reported where
    the glyphs themselves clear each other — the warning is for looking at,
    not a verdict.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pad = min_pt * fig.dpi / 72
    ## Tick labels of ticks outside the axis range are never drawn, though the
    ## objects exist; leave them out, and count each text object once.
    undrawn = set()
    for ax in fig.get_axes():
        for axis, (lo, hi) in ((ax.xaxis, sorted(ax.get_xlim())), (ax.yaxis, sorted(ax.get_ylim()))):
            span = (hi - lo) * 1e-6
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not (lo - span <= tick.get_loc() <= hi + span):
                    undrawn |= {id(tick.label1), id(tick.label2)}
    seen, texts = set(), []
    for t in fig.findobj(mpl.text.Text):
        if (id(t) in seen or id(t) in undrawn or not t.get_visible()
                or not t.get_text().strip()):
            continue
        seen.add(id(t))
        texts.append(t)
    boxes = [(t, t.get_window_extent(renderer)) for t in texts]
    hits = []
    for i, (a, ba) in enumerate(boxes):
        for b, bb in boxes[i + 1:]:
            w = min(ba.x1, bb.x1) - max(ba.x0, bb.x0)
            h = min(ba.y1, bb.y1) - max(ba.y0, bb.y0)
            if w > pad and h > pad:
                hits.append((a.get_text().replace("\n", " "), b.get_text().replace("\n", " ")))
    return hits


def save(fig, stem: str, png: bool = False) -> Path:
    """
    Write the panel to figures/output/. PDF is the deliverable; PNG is for review.

    Refuses a figure that prints an internal key — checked on the text the
    figure actually draws, so it holds however the label reached the axes.
    """
    leaks = leaked_names(fig)
    if leaks:
        plt.close(fig)
        raise SystemExit(f"REFUSING to save {stem}: it prints internal names {leaks}. "
                         "Map them to display names (src/display.py) in the prepare step.")
    ## Overlapping text only warns: the figure is still written, for a look.
    for a, b in text_overlaps(fig)[:10]:
        print(f"  ! {stem}: text overlaps — {a[:40]!r} / {b[:40]!r}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pdf = OUTPUT / f"{stem}.pdf"
    fig.savefig(pdf)
    print(f"Saved -> {pdf}")
    if png:
        fig.savefig(OUTPUT / f"{stem}.png", dpi=300)
        print(f"Saved -> {OUTPUT / f'{stem}.png'}")
    plt.close(fig)
    return pdf
