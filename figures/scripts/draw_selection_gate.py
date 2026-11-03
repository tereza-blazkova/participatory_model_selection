"""
The selection-gate figure: the workshop selection rules applied to every model
and feature set, one panel per model. The paper prints this.

Reading the figure
    solid line     the enrollment-level recall the combination achieved
    dotted line    the recall the workshop rules demanded of it, which is the
                   higher of the two floors (data-input and explainability)
    filled marker  permissible — achieved cleared demanded
    open marker    rejected — achieved fell below demanded
    faded          hard-blocked by a workshop rule, never put to the gate
    hatched panel  a model blocked on every feature set (the neural network): struck out
    ring           the selected Participatory model
    horizontal     the credits-policy baseline (>=60 ECTS behind)

This script reads figures/input/ and nothing else. It does not know what a
feature set is, which recall rests on how many students, or what the disclosure
threshold is: the ladder arrives as x positions and tick labels, a suppressed
value arrives as NaN — which matplotlib drops from the line and the markers by
itself — and the note arrives as a finished sentence. Every quantity it draws is
one the paper is allowed to print.

    python figures/scripts/draw_selection_gate.py [--year 2021] [--png]
"""
from __future__ import annotations

import argparse

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea

import utils

## Gap between the credits-policy baseline and the "never permissible" label.
_BAND_LABEL_GAP_PT = 2.0
## Gap between the baseline and its own label, under the line.
_BASELINE_LABEL_GAP_PT = 2.0
## In the feature-set line: between a step and its gain, and between steps.
_XLABEL_GAIN_GAP_PT = 3.0
_XLABEL_STEP_GAP_PT = 16.0


def _draw_panel(ax, panel, colour, marker) -> None:
    """
    One model's two lines: what it achieved, and what was demanded of it.

    The achieved line is drawn in two segments so the hard-blocked feature sets
    fade out, the faded one starting a point early so the line stays connected
    across the boundary.
    """
    x        = panel["x"].to_numpy()
    recall   = panel["recall"].to_numpy(dtype=float)
    demand   = panel["demand"].to_numpy(dtype=float)
    evaluated = panel["evaluated"].to_numpy(dtype=bool)

    fade_from = int(np.argmin(evaluated)) if not evaluated.all() else len(x)
    if fade_from == 0:
        ax.plot(x, recall, color=colour, linewidth=2.4, alpha=utils.FADE, zorder=2)
    else:
        ax.plot(x[:fade_from], recall[:fade_from], color=colour, linewidth=2.4, zorder=2)
        if fade_from < len(x):
            ax.plot(x[fade_from - 1:], recall[fade_from - 1:], color=colour,
                    linewidth=2.4, alpha=utils.FADE, zorder=2)

    ax.plot(x, demand, color=colour, linestyle=utils.DEMAND_DASH, linewidth=1.8,
            marker="o", markersize=5, markeredgewidth=0, zorder=3)

    ## Markers carry the verdict. A cell the rules never reached is filled but
    ## faded, so "open" always and only means "evaluated and rejected". A cell
    ## with no recall to draw gets no marker either — its shape would state the
    ## verdict on a value that is not shown.
    for row in panel.itertuples():
        if row.recall != row.recall:
            continue
        filled = row.permissible or not row.evaluated
        ax.plot([row.x], [row.recall], marker=marker, markersize=8.0,
                markerfacecolor=colour if filled else "white", markeredgecolor=colour,
                markeredgewidth=1.8, linestyle="none",
                alpha=1.0 if row.evaluated else utils.FADE, zorder=4)
        if row.selected:
            ax.plot([row.x], [row.recall], marker="o", markersize=15,
                    markerfacecolor="none", markeredgecolor=utils.SELECTED_COLOR,
                    markeredgewidth=2.0, zorder=5)


def _strike_out(ax) -> None:
    """
    Hatch a whole panel at 45 degrees: the model is hard-blocked on every
    feature set (the neural network), so no recall it shows could have been selected. Behind the
    data, above the never-permissible band; the series keeps its own colour
    (the Max ML pink), faded as every blocked cell is.
    """
    ax.add_patch(mpatches.Rectangle(
        (0, 0), 1, 1, transform=ax.transAxes, fill=False, hatch="////",
        edgecolor=utils.STRUCK_HATCH, linewidth=0, zorder=0.5))


def _decorate(ax, ticks, meta, baseline, *, ylabel: bool, baseline_label: bool) -> None:
    """Shared furniture: the blocked band, the baseline, the axes."""
    band = meta["band"]
    if band["from_x"] is not None:
        ## The blocked sets are always the tail of the ladder, so one span covers them.
        ax.axvspan(band["from_x"] - 0.5, len(ticks) - 0.5, color=utils.BAND_COLOR,
                   zorder=0, linewidth=0)
        ## Just above the credits-policy baseline, so the label reads against the
        ## reference every blocked cell is measured by. A fixed offset in points,
        ## not in recall, keeps the gap the same on any axis range. With the
        ## baseline withheld there is nothing to sit on: the bottom of the band.
        x = (band["from_x"] + len(ticks) - 1) / 2
        if baseline == baseline:
            xy, trans = (x, baseline), mtransforms.offset_copy(
                ax.transData, fig=ax.figure, y=_BAND_LABEL_GAP_PT, units="points")
        else:
            xy, trans = (x, 0.015), ax.get_xaxis_transform()
        ax.text(*xy, band["label"].replace(" ", "\n"), transform=trans,
                ha="center", va="bottom", fontsize=utils.FONT["annot"],
                color=utils.BAND_TEXT)

    ## The baseline arrives as a row of the CSV like any other value, so it is
    ## suppressed by the same guard and is absent — not zero — when withheld.
    if baseline == baseline:
        ax.axhline(baseline, color=utils.RULE_COLOR,
                   linestyle=(0, (1, 1.6)), linewidth=1.2, zorder=1)
        ## Named once, in the first panel; the line reads the same in the others.
        ## Below it: the demands that rise off it are all above, so this is the
        ## one clear side. Ending at the blocked band, so it stays off the grey.
        if baseline_label:
            end = len(ticks) if band["from_x"] is None else band["from_x"]
            ax.annotate("Credits policy", xy=(end - 0.55, baseline),
                        xytext=(0, -_BASELINE_LABEL_GAP_PT), textcoords="offset points",
                        ha="right", va="top", fontsize=utils.FONT["annot"],
                        color=utils.RULE_COLOR)

    ax.set_xticks(range(len(ticks)))
    ax.set_xticklabels(ticks)
    ax.set_xlim(-0.5, len(ticks) - 0.5)
    ax.set_ylim(*meta["ylim"])          # one range for all four panels, set in prepare
    if ylabel:
        ax.set_ylabel(meta["ylabel"], labelpad=6)
    ax.spines[["top", "right"]].set_visible(False)


def _legend(fig) -> None:
    """One row above the panels: what the lines and marks mean. Which model is
    which is said inside each panel, under its name."""
    key_handles = [
        Line2D([], [], color=utils.NEUTRAL, linewidth=2.4, marker="o", markersize=8.0,
               label="Recall achieved"),
        Line2D([], [], color=utils.NEUTRAL, linewidth=1.8, linestyle=utils.DEMAND_DASH,
               marker="o", markersize=5, markeredgewidth=0, label="Recall demanded"),
        Line2D([], [], color=utils.NEUTRAL, marker="o", linestyle="none", markersize=8.0,
               label="Permissible"),
        Line2D([], [], color=utils.NEUTRAL, marker="o", linestyle="none", markersize=8.0,
               markerfacecolor="white", markeredgewidth=1.8, label="Rejected"),
        Line2D([], [], color=utils.SELECTED_COLOR, marker="o", linestyle="none",
               markersize=11, markerfacecolor="none", markeredgewidth=2.0,
               label="Selected configuration"),
    ]
    fig.legend(handles=key_handles, loc="upper center", bbox_to_anchor=(0.5, 1.08),
               ncol=len(key_handles), frameon=False, fontsize=utils.FONT["legend"],
               handlelength=1.8, columnspacing=1.4)


def _feature_set_line(fig, xlabel) -> None:
    """
    The line under the panels, with each step's gain smaller and grey so the
    ladder reads first. One text cannot mix sizes, so the pieces are packed on a
    shared baseline. A blank supxlabel keeps the room for them in the layout, and
    the pieces sit where the layout then puts it.
    """
    label = fig.supxlabel(" ", fontsize=utils.FONT["label"])
    ## A little air between the tick labels and the feature-set line under them.
    fig.get_layout_engine().set(h_pad=8 / 72)
    fig.draw_without_rendering()

    def piece(text, **style):
        return TextArea(text, textprops={"fontsize": utils.FONT["label"], **style})

    def step(s):
        parts = [piece(s["name"])]
        if s["gain"]:
            parts.append(piece(s["gain"], fontsize=utils.FONT["annot"], color=utils.BAND_TEXT))
        return HPacker(children=parts, align="baseline", pad=0, sep=_XLABEL_GAIN_GAP_PT)

    line = HPacker(children=[piece(xlabel["title"])] + [step(s) for s in xlabel["steps"]],
                   align="baseline", pad=0, sep=_XLABEL_STEP_GAP_PT)
    fig.add_artist(AnchoredOffsetbox(
        loc="lower center", child=line, pad=0, borderpad=0, frameon=False,
        bbox_to_anchor=(0.5, label.get_position()[1]), bbox_transform=fig.transFigure))


def draw(stem: str, png: bool = False):
    utils.apply_style()
    df, meta = utils.read_input(stem)
    ## The CSV serves the figure and its table twin, so it carries one row per
    ## cell either of them shows. The panels are the `panel` rows; the credits-
    ## policy baseline is a row of its own.
    panels   = df[df["role"] == "panel"]
    base     = df[df["role"] == "baseline"]["recall"]
    baseline = float(base.iloc[0]) if len(base) else float("nan")
    ticks    = panels.sort_values("x").drop_duplicates("x")["x_tick"].tolist()
    ## Models the rules block on every feature set (the neural network): hatched.
    struck   = frozenset(m for m, g in panels.groupby("model") if g["model_blocked"].all())
    tiers    = {p["model"]: p["tier"] for p in meta["panels"]}

    ## One row: the panels are read across, model by model, against a single
    ## recall axis. sharey enforces the range prepare computed.
    n_panels = panels["panel"].nunique()
    fig, axes = plt.subplots(1, n_panels, figsize=meta["figsize"], sharey=True,
                             constrained_layout=True)
    for ax, (panel_no, panel) in zip(axes.flat, panels.groupby("panel", sort=True)):
        panel_no = int(panel_no)
        panel  = panel.sort_values("x")
        model  = panel["model"].iloc[0]
        if model in struck:
            _strike_out(ax)
        _draw_panel(ax, panel, utils.SERIES_COLOR[model], utils.MODEL_MARKER[model])
        ## Every panel sits on the bottom row; only the first carries the recall
        ## label, and sharey hides the other three's tick labels.
        _decorate(ax, ticks, meta, baseline, ylabel=(panel_no == 0),
                  baseline_label=(panel_no == 0))
        ## The offset is in axes fractions, so a panel a quarter as wide needs
        ## roughly four times the fraction to keep the letter off the name.
        ax.text(0.03, 0.965, panel["panel_letter"].iloc[0], transform=ax.transAxes,
                fontsize=utils.FONT["panel"], fontweight="bold", va="top")
        ## The model's name, and under it the gain the explainability rule
        ## demanded of it (or that it was never allowed).
        ax.text(0.115, 0.958, f"{panel['panel_title'].iloc[0]}\n({tiers[model]})",
                transform=ax.transAxes, fontsize=utils.FONT["annot"], va="top",
                linespacing=1.3)

    if meta["note"]:
        ## One note for the whole figure, on the last panel drawn. Kept out of the
        ## layout engine so it cannot shrink the axes.
        ## Outside the panels, on its own line under the feature-set line, at
        ## the right: inside, every corner holds the band's label or the data.
        t = fig.text(0.995, -0.01, meta["note"], ha="right", va="top",
                     fontsize=utils.FONT["annot"], color="0.35")
        t.set_in_layout(True)   # inside the saved crop, below the feature-set line

    _feature_set_line(fig, meta["xlabel"])
    _legend(fig)
    return utils.save(fig, stem, png=png)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year", type=int, default=2021)
    ap.add_argument("--ruleset", default="endline")
    ap.add_argument("--png", action="store_true", help="also write a 300 dpi PNG for review")
    args = ap.parse_args()
    stem = f"selection_gate_{args.year}" + ("" if args.ruleset == "endline" else f"_{args.ruleset}")
    draw(stem, png=args.png)


if __name__ == "__main__":
    main()
