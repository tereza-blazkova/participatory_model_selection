"""
The model-comparison figure: the paper's headline composite.

    a  activation-level ROC AUC for every model x feature set, against the credits
       policy; whiskers are bootstrap intervals, a black outline marks the
       Participatory model and a dashed one Max ML
    b  what each rule actually does to the cohort, as shares of enrollments
    c  precision-recall curves, with dashed iso-budget lines joining the methods
       at the same contact rate — so the panel reads as "at this budget, who
       does better" rather than "at each method's own threshold"
    d  ABROCA per fairness axis per rule: how far apart the two groups' ROC
       curves lie (unsigned)

The model_comparison_2021_2023 variant adds

    e  the October 2023 pilot at Faculty A: activation-level ROC AUC of the
       credits policy, the operational rule-based model, FasterRisk on the
       advisor variables, the Participatory model and Max ML

under a (d) cut to the height of its three rows: axis names for rows, column
names horizontal, two decimals, no colorbar. Which one is drawn follows from
the input (a `panel_e` block in its .json).

    python figures/scripts/draw_model_comparison.py [--stem model_comparison_2021_2023] [--png]
"""
from __future__ import annotations

import argparse

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

import utils

_BAR_W = 0.185  # four bars fill 74% of a feature-set group
_B_BAR_W = 0.72  # panel (b)'s stacked bars, as a share of their slot
_BUDGET_GREY = ["#777777", "#999999", "#444444"]

## Panel b's stacking order, bottom to top, within each outcome pair.
_NONDROPOUT = [("frac_tn", utils.NONDROPOUT_LIGHT, "#444444", "Not flagged"),
               ("frac_fp", utils.NONDROPOUT_COLOR, "white",   "Over-flagged")]
_DROPOUT    = [("frac_tp", utils.DROPOUT_COLOR,    "white",   "Detected"),
               ("frac_fn", utils.DROPOUT_LIGHT,    "#444444", "Missed")]

_CURVE_STYLE = {
    "max_ml":         dict(lw=1.8, ls="-",  zorder=4, label="Max ML"),
    "participatory":  dict(lw=1.8, ls="--", zorder=4, label="Participatory"),
    "credits_policy": dict(lw=1.4, ls=":",  zorder=3, label="Credits policy"),
}
_OP_STYLE = {
    "max_ml":        dict(marker="o", s=60,  zorder=6),
    "participatory": dict(marker="*", s=110, zorder=6),
    "credits_60":    dict(marker="P", s=100, zorder=7),
}


def _panel_a(ax, panel, spec, meta):
    for row in panel.itertuples():
        colour = utils.SERIES_COLOR[row.series]
        ## Both outlines are separate rectangles, so a cell that is both
        ## Participatory and Max ML can show both.
        ax.bar(row.x, row.y, width=_BAR_W, color=colour, alpha=0.9,
               edgecolor="white", linewidth=0.4, zorder=3 if row.highlight else 2)
        if row.highlight in ("participatory", "max_ml"):
            ax.add_patch(mpl.patches.Rectangle(
                ## From the axis floor, unclipped: the bars start at 0, below the
                ## axis, so the bar's own edge lost its bottom line to clipping
                ## and the Participatory outline ended short at the floor; and the
                ## last bar ends at the panel's right edge, where clipping cut the
                ## Max ML border to half width.
                (row.x - _BAR_W / 2, spec["ylim"][0]), _BAR_W, row.y - spec["ylim"][0],
                fill=False, edgecolor=utils.HIGHLIGHT_EDGE, linewidth=2.5,
                linestyle="-" if row.highlight == "participatory" else (0, (4, 2)),
                zorder=4, clip_on=False))
        if row.ci_low == row.ci_low:
            cap = _BAR_W * 0.35
            kw = dict(color="#222222", lw=1.2, zorder=5, alpha=0.85,
                      solid_capstyle="butt")
            ax.plot([row.x, row.x], [row.ci_low, row.ci_high], **kw)
            for y in (row.ci_low, row.ci_high):
                ax.plot([row.x - cap, row.x + cap], [y, y], **kw)
        ax.text(row.x, row.label_y + 0.003, row.value_label, ha="center",
                va="bottom", fontsize=utils.FONT["annot"])

    ax.set_xticks([t["x"] for t in spec["ticks"]])
    ax.set_xticklabels([t["label"] for t in spec["ticks"]])
    for tick, t in zip(ax.get_xticklabels(), spec["ticks"]):
        tick.set_rotation(t["rotation"])
        tick.set_ha(t["ha"])
    ax.set_xlim(*spec["xlim"])
    ax.set_ylim(*spec["ylim"])
    ## Every 0.1 from the floor up.
    ax.set_yticks(np.arange(spec["ylim"][0], spec["ylim"][1] + 1e-9, 0.1).round(1))
    ax.set_ylabel(spec["ylabel"], labelpad=6)

    handles = []
    for entry in spec["legend"]:
        colour = utils.SERIES_COLOR[entry["series"]]
        hl = entry.get("highlight")
        handles.append(mpatches.Patch(
            facecolor=colour, alpha=0.9, label=entry["label"],
            **(dict(edgecolor=utils.HIGHLIGHT_EDGE, linewidth=2.0,
                    linestyle="-" if hl == "participatory" else (0, (4, 2))) if hl else {})))
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=len(handles), frameon=False, fontsize=utils.FONT["legend"],
              ## Tighter than the defaults, as in (c): at the default spacing a
              ## legend wider than the axes ("Neural network") made the layout
              ## shift (a) right, and (b), which starts where (a) does,
              ## narrowed until its tick labels ran together.
              handlelength=1.6, columnspacing=1.0)
    ax.spines[["top", "right"]].set_visible(False)
    ## The axis line over the bars and the Max ML outline, not under them.
    ax.spines["bottom"].set_zorder(6)


def _panel_b(ax, panel, spec, meta):
    """Draw the bars; return the segment labels for a second pass.

    The labels cannot be placed yet: whether a segment's NAME fits inside its bar
    depends on the axes' final width, and the layout engine has not run.
    """
    line = spec["dropout_line"]
    pending = []
    for row in panel.itertuples():
        if row.empty:
            continue
        bottom = 0.0
        for block_col, segments in (("block_nondropout", _NONDROPOUT),
                                    ("block_dropout", _DROPOUT)):
            block = getattr(row, block_col)
            if block == block:
                ax.bar(row.x, block, width=_B_BAR_W, bottom=bottom,
                       color=utils.SUPPRESSED_BLOCK, edgecolor="white", linewidth=0.6)
                ax.text(row.x, bottom + block / 2, meta["marker"], ha="center",
                        va="center", fontsize=utils.FONT["annot"], color="0.35")
                bottom += block
                continue
            for col, colour, txt_colour, label in segments:
                frac = getattr(row, col)
                if frac != frac:
                    continue
                ax.bar(row.x, frac, width=_B_BAR_W, bottom=bottom, color=colour,
                       edgecolor="white", linewidth=0.6)
                ## Every segment is offered for a label; _label_segments measures
                ## what fits, so no share-of-axis cut-off decides it.
                pending.append((row.x, bottom + frac / 2, label, frac, txt_colour))
                bottom += frac

    ax.axhline(line, color="#333333", linewidth=1.2, linestyle="--", zorder=5)
    ## The Dropout / Non-dropout side labels are placed in draw(), once (b) has
    ## its final size: keeping each on its side of the line means measuring it.
    ## Not square: it is placed by hand with a wide box (see draw()).
    ax.set_xticks(panel["x"].tolist())
    ax.set_xticklabels(panel["tick"].tolist(), fontsize=utils.FONT["tick"], rotation=0,
                       ha="center", multialignment="center")
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
    ax.set_ylabel(spec["ylabel"], labelpad=6)
    ax.set_ylim(0, 1)
    ax.set_xlim(*spec["xlim"])
    ax.spines[["top", "right"]].set_visible(False)
    return pending


def _label_segments(ax, pending, bar_width=None) -> None:
    """
    Name and share inside each segment, at the widest form that fits the bar.

    In order: the name on one line above its share; the name wrapped at its
    hyphen; the share alone. The name is what identifies the segment — panel b
    has no legend — so it is only given up when even the wrapped form overruns
    the bar.
    Every form is measured against the segment's width AND height as drawn;
    a segment too small even for its share is left unlabelled.
    """
    bar_width = _B_BAR_W if bar_width is None else bar_width
    for x, y, label, frac, colour in pending:
        candidates = []
        if label is not None:
            candidates.append((f"{label}\n{frac:.1%}", utils.FONT["annot"]))
            candidates.append((f"{utils.wrap_label(label)}\n{frac:.1%}", utils.FONT["annot"]))
        candidates.append((f"{frac:.1%}", utils.FONT["annot"]))
        for text, size in candidates:
            if not utils.text_fits(ax, text, width_data=bar_width, fontsize=size,
                                   height_data=frac, linespacing=1.3):
                continue
            ax.text(x, y, text, ha="center", va="center", fontsize=size,
                    color=colour, linespacing=1.3)
            break


def _panel_c(ax, panel, spec, meta):
    ## The Participatory model keeps its own model's colour, as its bar has in (a).
    colour = dict(utils.SERIES_COLOR)
    if spec["participatory_model"]:
        colour["participatory"] = utils.SERIES_COLOR[spec["participatory_model"]]
    curves = []
    for series, style in _CURVE_STYLE.items():
        pts = panel[(panel["kind"] == "curve") & (panel["series"] == series)]
        if not len(pts):
            continue
        pts = pts.sort_values("order")
        kw = dict(style)
        curves += ax.plot(pts["x"], pts["y"], color=colour[series], **kw)

    ## Iso-budget lines first, so the curves and their operating points sit on top.
    for series, group in panel[panel["kind"] == "budget"].groupby("series"):
        group = group.sort_values("order")
        grey = _BUDGET_GREY[int(series) % len(_BUDGET_GREY)]
        ax.plot(group["x"], group["y"], color=grey, lw=0.9, linestyle="--",
                marker="x", markersize=5, zorder=2, alpha=0.75)
    budget_labels = []
    for row in panel[panel["kind"] == "budget_label"].itertuples():
        grey = _BUDGET_GREY[int(row.series) % len(_BUDGET_GREY)]
        budget_labels.append(ax.text(
            row.x, row.y, row.value_label, ha="center", va="center",
            fontsize=utils.FONT["annot"], color=grey, zorder=4,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=2)))

    for row in panel[panel["kind"] == "operating_point"].itertuples():
        ax.scatter(row.x, row.y, color=colour[row.series],
                   edgecolors="white", linewidths=0.8, **_OP_STYLE[row.style])

    handles, labels = ax.get_legend_handles_labels()
    hmap = dict(zip(labels, handles))
    ## Upper right, stacked, each name before its line, at the annotation size
    ## (the smallest the figure uses). The lines stand just past the plot's
    ## right edge, which moves the names off a Max ML curve that stays near
    ## precision 1 into mid recall.
    ax.legend(handles=[hmap[l] for l in spec["legend_order"][1:]],
              labels=spec["legend_order"][1:], loc="upper right",
              bbox_to_anchor=(1.07, 1.0), frameon=False,
              ncol=1, markerfirst=False, fontsize=utils.FONT["annot"],
              handlelength=1.4, handletextpad=0.5, labelspacing=0.3,
              borderaxespad=0, borderpad=0)
    ax.set_xlabel(spec["xlabel"], labelpad=6)
    ax.set_ylabel(spec["ylabel"], labelpad=6)
    ax.set_xlim(*spec["lim"])
    ax.set_ylim(*spec["lim"])
    ## The same ticks on both axes, and one zero at the corner, on the x axis.
    ticks = np.arange(0, spec["lim"][1] + 1e-9, 0.2)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
        lambda v, _: "" if v == 0 else f"{v:.1f}"))
    ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(False)
    return budget_labels, curves


def _separate_labels(ax, texts, curves=(), avoid=(), gap_pt: float = 4.0,
                     step_pt: float = 2.0, reach_pt: float = 40.0):
    """
    Move budget labels off each other and off the precision-recall curves.

    Measured on the drawn boxes, after the layout is frozen: only the renderer
    knows how tall a label is (the credits-policy one is two lines), and where
    the lines run depends on the data — two lines can run close together, and
    their labels with them. Labels are taken bottom to top; each stays where it
    is if it is clear, and otherwise moves to the nearest position within
    `reach_pt` that clears every curve, every label already placed, anything in
    `avoid` (the legend) and the axes' edges. If none does, it keeps its place.
    """
    if not texts:
        return
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    px = fig.dpi / 72
    gap = gap_pt * px
    to_data = ax.transData.inverted()
    frame = ax.get_window_extent(renderer)

    ## Each curve as dense points on screen, so a box test catches a segment
    ## that crosses the box between two of the curve's own vertices.
    curve_pts = []
    for line in curves:
        xy = line.get_transform().transform(line.get_xydata())
        xy = xy[np.isfinite(xy).all(axis=1)]
        for a, b in zip(xy[:-1], xy[1:]):
            n = max(2, int(np.hypot(*(b - a)) / px) + 1)
            curve_pts.append(np.linspace(a, b, n))
    curve_pts = np.vstack(curve_pts) if curve_pts else np.empty((0, 2))

    def box(t):
        ## The text's own extent plus the white box's pad (stored as a fraction
        ## of the font size); the patch itself only follows set_position at
        ## the next draw.
        return t.get_window_extent(renderer).padded(
            t.get_bbox_patch().get_boxstyle().pad * t.get_fontsize() * px)

    def clear(b, placed):
        if b.x0 < frame.x0 or b.x1 > frame.x1 or b.y0 < frame.y0 or b.y1 > frame.y1:
            return False
        if any(b.padded(gap / 2).overlaps(o) for o in placed):
            return False
        inside = ((curve_pts[:, 0] >= b.x0) & (curve_pts[:, 0] <= b.x1)
                  & (curve_pts[:, 1] >= b.y0) & (curve_pts[:, 1] <= b.y1))
        return not inside.any()

    ## Candidate offsets, nearest first.
    steps = np.arange(-reach_pt, reach_pt + step_pt / 2, step_pt) * px
    offsets = sorted(((dx, dy) for dx in steps for dy in steps),
                     key=lambda d: np.hypot(*d))

    ## Anything else drawn in the axes the labels must stay off (the legend).
    placed = [a.get_window_extent(renderer) for a in avoid if a is not None]
    for t in sorted(texts, key=lambda t: t.get_position()[1]):
        x0, y0 = ax.transData.transform(t.get_position())
        b0 = box(t)
        for dx, dy in offsets:
            if clear(b0.translated(dx, dy), placed):
                t.set_position(to_data.transform((x0 + dx, y0 + dy)))
                break
        placed.append(box(t))


def _panel_d(ax, panel, spec, meta, compact=False):
    n_rows = len(spec["row_labels"])
    n_cols = len(spec["col_labels"])
    vmax = spec["vmax"]
    grid = np.full((n_rows, n_cols), np.nan)
    for row in panel.itertuples():
        grid[int(row.row), int(row.col)] = row.y

    ## Sequential from zero, because ABROCA is unsigned: white is no gap
    ## between the two groups' ROC curves.
    cmap = LinearSegmentedColormap.from_list(
        "abroca_d", ["#ffffff", utils.DROPOUT_LIGHT, utils.DROPOUT_COLOR])
    im = ax.imshow(grid, cmap=cmap, aspect="auto", vmin=0, vmax=vmax)

    for row in panel.itertuples():
        if row.y == row.y:
            ## Dark cells take white; elsewhere the style's text colour (None),
            ## the same as the row and column labels.
            colour = "white" if row.y > 0.6 * vmax else None
            ax.text(int(row.col), int(row.row), f"{row.y:.2f}" if compact else f"{row.y:.3f}",
                    ha="center",
                    va="center", fontsize=utils.FONT["annot"], color=colour)
        else:
            text = meta["marker"] if row.hidden else meta["not_applicable"]
            ax.text(int(row.col), int(row.row), text, ha="center", va="center",
                    fontsize=utils.FONT["annot"], color="gray")

    ax.set_xticks(range(n_cols))
    if compact:
        ax.set_xticklabels(spec["col_labels"], fontsize=utils.FONT["tick"])
    else:
        ## Slanted at 30 degrees: the columns are narrower than their names.
        ax.set_xticklabels(spec["col_labels"], fontsize=utils.FONT["tick"], rotation=30,
                           ha="right", rotation_mode="anchor")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(spec["row_labels"], fontsize=utils.FONT["tick"])
    ax.spines[["top", "right"]].set_visible(False)
    return im


def _panel_e(ax, panel, spec, meta):
    panel = panel.sort_values("x")
    for row in panel.itertuples():
        x = row.x
        if row.hidden:
            ax.text(x, spec["ylim"][0] + 0.02, meta["marker"], ha="center",
                    va="bottom", fontsize=utils.FONT["annot"], color="gray")
            continue
        hl = row.highlight if isinstance(row.highlight, str) else None
        ax.bar(x, row.y, width=0.7, color=utils.SERIES_COLOR[row.series], alpha=0.9,
               hatch="////" if row.series == "fasterrisk_used_vars" else None,
               edgecolor="white", linewidth=0.4, zorder=2)
        if hl:
            ## The Participatory and Max ML outlines of (a), from the axis floor.
            ax.add_patch(mpl.patches.Rectangle(
                (x - 0.35, spec["ylim"][0]), 0.7, row.y - spec["ylim"][0],
                fill=False, edgecolor=utils.HIGHLIGHT_EDGE, linewidth=1.8,
                linestyle="-" if hl == "participatory" else (0, (4, 2)),
                zorder=4, clip_on=False))
        top = row.y
        if row.ci_low == row.ci_low:
            kw = dict(color="#222222", lw=1.0, zorder=5, alpha=0.85,
                      solid_capstyle="butt")
            ax.plot([x, x], [row.ci_low, row.ci_high], **kw)
            for y in (row.ci_low, row.ci_high):
                ax.plot([x - 0.12, x + 0.12], [y, y], **kw)
            top = row.ci_high
        ax.text(x, top + 0.01, row.value_label, ha="center", va="bottom",
                fontsize=utils.FONT["annot"])
    ## Slanted at 30 degrees, as (a)'s: five names in a narrow panel.
    ax.set_xticks(panel["x"].tolist())
    ax.set_xticklabels(panel["tick"].tolist(), fontsize=utils.FONT["tick"], rotation=30,
                       ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.6, len(panel) - 0.4)
    ax.set_ylim(*spec["ylim"])
    ax.set_yticks(np.arange(round(spec["ylim"][0], 1), spec["ylim"][1] + 1e-9, 0.2).round(1))
    ax.set_ylabel(spec["ylabel"], labelpad=4)
    ## Top left, where the two rule-based bars leave room; at the size of (b)'s
    ## Dropout / Non-dropout names, framed in the operational rule's blue.
    ax.text(0.03, 0.98, spec["note"], transform=ax.transAxes, ha="left", va="top",
            fontsize=utils.FONT["side"], color="#555555",
            bbox=dict(boxstyle="square,pad=0.4", facecolor="white",
                      edgecolor=utils.SERIES_COLOR["operational_rule"], linewidth=0.8))
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["bottom"].set_zorder(6)


def draw(stem: str = "model_comparison", png: bool = False):
    utils.apply_style()
    df, meta = utils.read_input(stem)

    fig = plt.figure(figsize=meta["figsize"], constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=meta["height_ratios"], hspace=0.0)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b, ax_c, ax_d = (fig.add_subplot(gs[1, i]) for i in range(3))

    letter_a = ax_a.text(meta["panel_a"]["letter_x"], 1.10, meta["panel_a"]["letter"],
                         transform=ax_a.transAxes, fontsize=utils.FONT["panel"],
                         fontweight="bold", va="top")

    _panel_a(ax_a, df[df["panel"] == "a"], meta["panel_a"], meta)
    pending_b = _panel_b(ax_b, df[df["panel"] == "b"].sort_values("x"),
                         meta["panel_b"], meta)
    budget_labels, curves_c = _panel_c(ax_c, df[df["panel"] == "c"], meta["panel_c"], meta)
    has_e = "panel_e" in meta
    im = _panel_d(ax_d, df[df["panel"] == "d"], meta["panel_d"], meta, compact=has_e)
    ax_e = None
    if has_e:
        ax_e = fig.add_axes([0, 0, 0.1, 0.1])
        _panel_e(ax_e, df[df["panel"] == "e"], meta["panel_e"], meta)

    ## (a) keeps the place the layout engine gives it; the bottom row is then
    ## placed by hand under it, and the figure cut to what the two need.
    fig.canvas.draw()
    fig.set_layout_engine(None)
    renderer = fig.canvas.get_renderer()
    dpi = fig.dpi
    ## The bottom row, placed by hand under (a): left to the layout engine, the
    ## square panels came out ~2 in wide in ~3.7 in columns and (b)'s three
    ## multi-line names ran together. (b) gets a wide box, (c) stays square,
    ## (d) takes the rest with its colorbar. In inches, then figure fractions.
    W, H = fig.get_size_inches()
    pa = ax_a.get_position()
    a_x0, a_y0, a_w, a_h = pa.x0 * W, pa.y0 * H, pa.width * W, pa.height * H
    a_above = H - (a_y0 + a_h)                     # (a)'s legend
    h = min(2.3, a_y0 - 2.25)
    ## (b) starts where (a) does, so the two read as one left column.
    ## With (e): (d) only as tall as its rows, (e) under it in the height (d)'s
    ## column names and (e)'s letter leave.
    d_h = 0.16 * len(meta["panel_d"]["row_labels"])
    def place(y0):
        if not has_e:
            for ax, x_in, w_in in ((ax_b, a_x0, 3.75 - a_x0), (ax_c, 4.55, h),
                                   (ax_d, 8.75, 1.5)):   # (d)'s row labels need ~1.6 in
                ax.set_position([x_in / W, y0 / H, w_in / W, h / H])
            return
        for ax, x_in, w_in in ((ax_b, a_x0, 3.75 - a_x0), (ax_c, 4.55, h)):
            ax.set_position([x_in / W, y0 / H, w_in / W, h / H])
        ax_d.set_position([8.15 / W, (y0 + h - d_h) / H, 2.1 / W, d_h / H])
        ax_e.set_position([8.15 / W, y0 / H, 2.5 / W, (h - d_h - 0.65) / H])
    bottom_row = [ax for ax in (ax_b, ax_c, ax_d, ax_e) if ax is not None]
    place(1.05)
    fig.canvas.draw()
    ## Measured, not guessed: the bottom row's lowest text (b's names) sets the
    ## bottom margin, (a)'s slanted ticks and the panel letters the gap between rows.
    letter_up = 0.10 * a_h                         # letter tops above their axes, as (a)'s
    tick_drop = a_y0 - ax_a.get_tightbbox(renderer).y0 / dpi
    lowest = min(ax.get_tightbbox(renderer).y0 / dpi for ax in bottom_row)
    y0 = 1.05 - lowest + 0.04
    a_y0 = y0 + h + tick_drop + letter_up + 0.12
    H = a_y0 + a_h + a_above
    fig.set_size_inches(W, H)
    ax_a.set_position([a_x0 / W, a_y0 / H, a_w / W, a_h / H])
    place(y0)
    fig.canvas.draw()

    ## Now that the panels have their final size, measure what fits.
    _label_segments(ax_b, pending_b, bar_width=_B_BAR_W - 0.06)   # a margin to the edge
    spec_b = meta["panel_b"]
    utils.band_labels(ax_b, -0.80, spec_b["dropout_line"],
                      spec_b["side_labels"]["upper"], spec_b["side_labels"]["lower"])
    _separate_labels(ax_c, budget_labels, curves_c, avoid=[ax_c.get_legend()])

    if has_e:
        ## No colorbar: (d) is too short for one, and its cells carry the values.
        ## Its name stands above it instead, and (d) runs to the right edge.
        right = W - 0.12
        pd_ = ax_d.get_position()
        ax_d.set_position([pd_.x0, pd_.y0, right / W - pd_.x0, pd_.height])
        ax_d.set_title(meta["panel_d"]["cbar_label"], fontsize=utils.FONT["label"], pad=3)
    else:
        pb = ax_b.get_position()
        pd_ = ax_d.get_position()
        cax = fig.add_axes([pd_.x0 + pd_.width + 0.007, pb.y0, 0.013, pb.height])
        cbar = fig.colorbar(im, cax=cax, orientation="vertical")
        cbar.outline.set_visible(False)
        cbar.set_label(meta["panel_d"]["cbar_label"], fontsize=utils.FONT["label"], labelpad=6)
        ## Only the two ends: the cells carry the values.
        vmax = meta["panel_d"]["vmax"]
        cbar.set_ticks([0, vmax], labels=["0.00", f"{vmax:.2f}"])
        cbar.ax.tick_params(labelsize=utils.FONT["tick"])

        ## The right edge everything ends at: where the colorbar's label ends when
        ## it stands beside the ticks. Inside the canvas width, so this figure
        ## prints at the same type size as the others.
        fig.canvas.draw()
        right = cax.get_tightbbox(renderer).x1 / dpi
        ## The label then moves between the two end values, into the tick column,
        ## and (d) takes the width it frees.
        cbar.set_label("")
        ticks = [t.get_window_extent(renderer) for t in cax.get_yticklabels()]
        centre = cax.transAxes.inverted().transform(
            ((min(t.x0 for t in ticks) + max(t.x1 for t in ticks)) / 2, 0))[0]
        cax.text(centre, 0.5, meta["panel_d"]["cbar_label"], transform=cax.transAxes,
                 rotation=90, ha="center", va="center", fontsize=utils.FONT["label"])
        fig.canvas.draw()
        grow = right - cax.get_tightbbox(renderer).x1 / dpi
        ax_d.set_position([pd_.x0, pd_.y0, pd_.width + grow / W, pd_.height])
        pc = cax.get_position()
        cax.set_position([pc.x0 + grow / W, pc.y0, pc.width, pc.height])

    ## (a) runs to the same right edge, so its last bar lines up with the end
    ## of everything under it.
    ax_a.set_position([a_x0 / W, a_y0 / H, (right - a_x0) / W, a_h / H])
    if has_e:
        pe = ax_e.get_position()
        ax_e.set_position([pe.x0, pe.y0, right / W - pe.x0, pe.height])

    ## (a)'s and (b)'s y labels on one vertical line, at whichever of the two
    ## sits further left by default, so neither runs into its tick labels.
    fig.canvas.draw()
    target = min(ax.yaxis.label.get_window_extent(renderer).x0 for ax in (ax_a, ax_b))
    for ax in (ax_a, ax_b):
        x = -0.1
        for _ in range(2):
            ax.yaxis.set_label_coords(x, 0.5)
            fig.canvas.draw()
            x += ((target - ax.yaxis.label.get_window_extent(renderer).x0)
                  / ax.get_window_extent(renderer).width)

    ## The other letters beside their panels, as (a)'s, at the same height above
    ## their top: as far left of the axes as (a)'s, or further where a tick label
    ## reaches the top (b's 100%, c's 1.0), so the letter clears it.
    fig.canvas.draw()
    dx = letter_a.get_window_extent(renderer).x0 / dpi - a_x0
    for ax, key in ((ax_b, "panel_b"), (ax_c, "panel_c"), (ax_d, "panel_d"),
                    *([(ax_e, "panel_e")] if has_e else [])):
        p = ax.get_position()
        letter = fig.text((p.x0 * W + dx) / W, (p.y1 * H + letter_up) / H,
                          meta[key]["letter"], fontsize=utils.FONT["panel"],
                          fontweight="bold", va="top")
        box = letter.get_window_extent(renderer)
        ticks = [t.get_window_extent(renderer) for t in ax.get_yticklabels()
                 if t.get_text() and t.get_window_extent(renderer).y1 > box.y0]
        if ticks:
            right = min(t.x0 for t in ticks) - 0.04 * dpi
            if box.x1 > right:
                letter.set_x((box.x0 - (box.x1 - right)) / dpi / W)
        ## (d) and (e) sit a little further out, clear of their row labels.
        if key in ("panel_d", "panel_e"):
            letter.set_x(letter.get_position()[0] - 0.15 / W)

    return utils.save(fig, stem, png=png)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stem", default="model_comparison",
                    help="model_comparison, or model_comparison_2021_2023 for the "
                         "variant with the pilot panel")
    ap.add_argument("--png", action="store_true")
    args = ap.parse_args()
    draw(stem=args.stem, png=args.png)


if __name__ == "__main__":
    main()
