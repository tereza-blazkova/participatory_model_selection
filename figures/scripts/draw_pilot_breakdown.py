"""
The October 2023 pilot's outcome breakdown: what each rule or model actually did
to the enrollments it saw, as one stacked bar per method. Faculty A only: (a) each
rule at its real threshold, (b) the rules recalibrated to the ML threshold rule.

    bottom block  non-dropouts — not flagged (light) then over-flagged (full)
    top block     dropouts — detected (full) then missed (light)
    dashed line   the faculty's dropout rate, so the two blocks have a fixed edge
    grey block    a same-outcome pair merged because one of them is small; its
                  height is the known margin, so the block itself says nothing
    tick label    each bar's recall and flag share, or the marker when withheld

Two figures share this script — the rules at their real thresholds, and the same
rules recalibrated to the ML models' own threshold rule:

    python figures/scripts/draw_pilot_breakdown.py --stem pilot_breakdown_faculty_a [--png]
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

import utils

_BAR_W = 0.7

## Segment order is the stacking order, bottom to top, within each outcome pair.
_NONDROPOUT = [("frac_tn", utils.NONDROPOUT_LIGHT, "#444444", "Not flagged"),
               ("frac_fp", utils.NONDROPOUT_COLOR, "white",   "Over-flagged")]
_DROPOUT    = [("frac_tp", utils.DROPOUT_COLOR,    "white",   "Detected"),
               ("frac_fn", utils.DROPOUT_LIGHT,    "#444444", "Missed")]


def _draw_bar(ax, x, row, marker) -> list:
    """One method's stacked bar: two outcome pairs, each either split or merged.

    Returns the segment labels for a second pass — whether a segment's NAME fits
    inside its bar depends on the axes' final width, which the layout engine has
    not decided yet.
    """
    pending = []
    bottom = 0.0
    for block_col, segments in (("block_nondropout", _NONDROPOUT),
                                ("block_dropout",    _DROPOUT)):
        block = row[block_col]
        if block == block:          # the pair was merged — draw the block, not its parts
            ax.bar(x, block, width=_BAR_W, bottom=bottom, color=utils.SUPPRESSED_BLOCK,
                   edgecolor="white", linewidth=0.6)
            ax.text(x, bottom + block / 2, marker, ha="center", va="center",
                    fontsize=utils.FONT["annot"], color="0.35")
            bottom += block
            continue
        for col, colour, txt_colour, label in segments:
            frac = row[col]
            if frac != frac:
                continue
            ax.bar(x, frac, width=_BAR_W, bottom=bottom, color=colour,
                   edgecolor="white", linewidth=0.6)
            ## Every segment is offered for a label; _label_segments measures
            ## what fits, so no share-of-axis cut-off decides it.
            pending.append((x, bottom + frac / 2, label, frac, txt_colour))
            bottom += frac
    return pending


def _label_segments(ax, pending) -> None:
    """
    Name and share inside each segment, at the widest form that fits the bar.

    In order: the name on one line above its share; the name wrapped at its
    hyphen; the share alone. The name is what identifies the segment, so it is
    only given up when even the wrapped form overruns the bar.
    Every form is measured against the segment's width AND height as drawn;
    a segment too small even for its share is left unlabelled.
    """
    for x, y, label, frac, colour in pending:
        candidates = []
        if label is not None:
            candidates.append((f"{label}\n{frac:.1%}", utils.FONT["annot"]))
            candidates.append((f"{utils.wrap_label(label)}\n{frac:.1%}", utils.FONT["annot"]))
        candidates.append((f"{frac:.1%}", utils.FONT["annot"]))
        for text, size in candidates:
            if not utils.text_fits(ax, text, width_data=_BAR_W, fontsize=size,
                                   height_data=frac, linespacing=1.3):
                continue
            ax.text(x, y, text, ha="center", va="center", fontsize=size,
                    color=colour, linespacing=1.3)
            break


def _draw_panel(ax, panel, meta, line, *, first: bool) -> list:
    panel = panel.sort_values("bar")
    xs = np.arange(len(panel))
    pending = []
    for x, row in zip(xs, panel.to_dict("records")):
        if not row["empty"]:
            pending += _draw_bar(ax, x, row, meta["marker"])

    ax.axhline(line["y"], color="#333333", linewidth=1.2, linestyle="--", zorder=5)
    ## The side labels naming the two blocks are placed in draw(), once the
    ## layout is final: keeping each on its side of the line means measuring it.

    ax.set_xticks(xs)
    ax.set_xticklabels(panel["tick"].tolist(), fontsize=utils.FONT["tick"], rotation=0, ha="center",
                       multialignment="center")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylabel(meta["ylabel"] if first else "", labelpad=6)
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.8, len(panel) - 0.5)
    if line.get("xlabel"):
        ## How the panel's methods are thresholded, under the tick labels.
        ax.set_xlabel(line["xlabel"], labelpad=6)
    ax.spines[["top", "right"]].set_visible(False)
    if not first:
        ax.tick_params(left=False, labelleft=False)
        ax.spines["left"].set_visible(False)
    utils.panel_label(ax, line["letter"])
    return pending


def draw(stem: str, png: bool = False):
    utils.apply_style()
    df, meta = utils.read_input(stem)
    lines = {ln["panel"]: ln for ln in meta["dropout_lines"]}

    fig, axes = plt.subplots(1, len(lines), figsize=meta["figsize"],
                             constrained_layout=True)
    pending = []
    for ax, (panel_no, panel) in zip(np.atleast_1d(axes), df.groupby("panel", sort=True)):
        panel_no = int(panel_no)
        pending.append((ax, _draw_panel(ax, panel, meta, lines[panel_no],
                                        first=(panel_no == 0))))
        ## Beside the block it explains, not once for the whole figure.
        if panel_no in meta["note_panels"]:
            utils.note(ax, meta["note"])

    ## Freeze the layout, then label: a segment name is kept or dropped by
    ## measuring it against its bar, which needs the axes' final width.
    fig.canvas.draw()
    fig.set_layout_engine(None)
    for ax, jobs in pending:
        _label_segments(ax, jobs)
    _side_labels(fig, np.atleast_1d(axes), lines[min(lines)]["y"], meta["side_labels"],
                 n_bars=len(df[df["panel"] == min(lines)]))
    return utils.save(fig, stem, png=png)


def _side_labels(fig, axes, line_y: float, names: dict, n_bars: int,
                 pad_pt: float = 2.0) -> None:
    """
    Name the two blocks the dashed line separates, once, level, in the gap
    between the panels — so the reader does not have to infer them from the
    colours. Standing up beside each panel's first bar, they were cramped into
    a narrow dropout band.

    Both panels show the same faculty, so the line sits at the same height in
    each. Each name is centred in its band when it fits there; otherwise it is
    held `pad_pt` off the line on its own side.
    """
    left, right = axes[0], axes[-1]
    x = (left.transData.transform((n_bars - 1 + _BAR_W / 2, 0))[0]
         + right.transData.transform((-_BAR_W / 2, 0))[0]) / 2
    renderer = fig.canvas.get_renderer()
    pad = pad_pt * fig.dpi / 72
    line = left.transData.transform((0, line_y))[1]
    to_fig = fig.transFigure.inverted()
    for text, y_mid, side in ((names["upper"], (1 + line_y) / 2, +1),
                              (names["lower"], line_y / 2, -1)):
        y = left.transData.transform((0, y_mid))[1]
        t = fig.text(*to_fig.transform((x, y)), text, ha="center", va="center",
                     fontsize=utils.FONT["side"], color="#555555")
        box = t.get_window_extent(renderer)
        if (box.y0 < line + pad) if side > 0 else (box.y1 > line - pad):
            t.set_va("bottom" if side > 0 else "top")
            t.set_y(to_fig.transform((x, line + side * pad))[1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ## Faculty A: (a) the rules at their real thresholds, (b) recalibrated.
    ap.add_argument("--stem", default="pilot_breakdown_faculty_a",
                    choices=["pilot_breakdown_faculty_a"])
    ap.add_argument("--png", action="store_true")
    args = ap.parse_args()
    draw(args.stem, png=args.png)


if __name__ == "__main__":
    main()
