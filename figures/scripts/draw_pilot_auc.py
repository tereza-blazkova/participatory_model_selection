"""
Pilot ROC AUC by faculty: how well each rule and each model ranks this faculty's
dropouts against its non-dropouts, one panel per faculty.

    left block   the three standalone baselines — credits policy, the faculty's
                 operational rule-based model, FasterRisk on the counsellors' nine variables
    right grid   the four ML models across the five feature sets
    black edge   the Participatory model and Max ML
    whiskers     bootstrap confidence interval, where the run produced one
    marker       a bar resting on too few students; it is not drawn, because a
                 vector PDF keeps the height of anything that is

    python figures/scripts/draw_pilot_auc.py --stem pilot_auc_faculty_a [--png]
"""
from __future__ import annotations

import argparse

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

import utils

_BAR_W = 0.185   # as in the model comparison
_LABEL_GAP_PT = 2.5     # number above the whisker cap


def _ci_whisker(ax, x, lo, hi) -> None:
    """Flat-capped interval, drawn above the bar it belongs to."""
    cap = _BAR_W * 0.35
    kw = dict(color="#222222", lw=1.2, zorder=5, alpha=0.85, solid_capstyle="butt")
    ax.plot([x, x], [lo, hi], **kw)
    for y in (lo, hi):
        ax.plot([x - cap, x + cap], [y, y], **kw)


def _draw_panel(ax, panel, spec, meta) -> None:
    for row in panel.itertuples():
        if row.hidden:
            ## The marker goes where the bar would have started, so the reader
            ## can see which position was withheld without seeing its height.
            ax.text(row.x, spec["hidden_y"], meta["marker"], ha="center", va="bottom",
                    fontsize=utils.FONT["annot"], color="0.35")
            continue
        ## Solid outline: the Participatory model. Max ML gets a dashed outline
        ## drawn as its own rectangle, as its legend entry shows. The advisor-
        ## variable FasterRisk bar is hatched.
        solid = row.highlight == "participatory"
        ax.bar(row.x, row.value, width=_BAR_W, color=utils.SERIES_COLOR[row.series],
               hatch="////" if row.series == "fasterrisk_used_vars" else None,
               alpha=0.9, edgecolor=utils.HIGHLIGHT_EDGE if solid else "white",
               linewidth=2.5 if solid else 0.4,
               zorder=3 if row.highlight in ("participatory", "max_ml") else 2)
        if row.highlight == "max_ml":
            ax.add_patch(mpatches.Rectangle(
                (row.x - _BAR_W / 2, spec["ylim"][0]),
                _BAR_W, row.value - spec["ylim"][0], fill=False,
                edgecolor=utils.HIGHLIGHT_EDGE, linewidth=2.5, linestyle=(0, (4, 2)),
                zorder=4, clip_on=False))   # the bar ends at the panel edge
        top = row.label_y
        if row.ci_low == row.ci_low:
            _ci_whisker(ax, row.x, row.ci_low, row.ci_high)
            top = max(top, row.ci_high)
        ## A fixed gap in points above the whisker cap, not in AUC units: the
        ## same 0.008 AUC is ~1.5 pt on a 0.5-1.0 axis, and the cap then
        ## touched the number.
        ax.annotate(row.value_label, (row.x, top), xytext=(0, _LABEL_GAP_PT),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=utils.FONT["annot"])

    ax.set_xticks([t["x"] for t in spec["ticks"]])
    ax.set_xticklabels([t["label"] for t in spec["ticks"]], fontsize=utils.FONT["tick"])
    for tick, t in zip(ax.get_xticklabels(), spec["ticks"]):
        tick.set_rotation(t["rotation"])
        tick.set_ha(t["ha"])
    ax.set_xlim(*spec["xlim"])
    ax.set_ylim(*spec["ylim"])
    if spec.get("chance_line") is not None:
        ## The floor dropped below chance for a rule that ranks worse than it;
        ## the line keeps 0.5 findable.
        ax.axhline(spec["chance_line"], color="0.6", lw=0.8, ls="--", zorder=1)
    ax.spines[["top", "right"]].set_visible(False)
    ## Level with the top of the axes, left of the y ticks: clear of the legend
    ## above panel a and of panel a's tick labels above panel b.
    utils.panel_label(ax, spec["letter"], xoff=-0.05, yoff=1.04)


def _legend(fig, meta) -> None:
    handles = []
    for entry in meta["legend"]:
        colour = utils.SERIES_COLOR[entry["series"]]
        if entry.get("highlight"):
            handles.append(mpatches.Patch(facecolor=colour, edgecolor=utils.HIGHLIGHT_EDGE,
                                          linewidth=2.0, linestyle=(0, (4, 2)),
                                          alpha=0.9, label=entry["label"]))
        else:
            handles.append(mpatches.Patch(facecolor=colour, alpha=0.9, label=entry["label"],
                                          hatch="////" if entry.get("hatch") else None,
                                          edgecolor="white" if entry.get("hatch") else None))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=4, frameon=False, fontsize=utils.FONT["legend"], columnspacing=1.6)


def draw(stem: str = "pilot_auc_faculty_a", png: bool = False):
    utils.apply_style()
    df, meta = utils.read_input(stem)

    fig, axes = plt.subplots(len(meta["panels"]), 1, figsize=meta["figsize"],
                             constrained_layout=True)
    for ax, spec in zip(np.atleast_1d(axes), meta["panels"]):
        _draw_panel(ax, df[df["panel"] == spec["panel"]], spec, meta)
    utils.note(np.atleast_1d(axes)[-1], meta["note"])

    ## The legend sits inside the figure boundary, so constrained_layout is
    ## turned off and the top margin reserved by hand — otherwise savefig's
    ## tight bbox expands the canvas around it and the width stops matching the
    ## column it was designed for.
    fig.set_layout_engine(None)
    fig.subplots_adjust(**meta["subplots_adjust"])
    ## One label for the shared axis: a label per panel ran longer than the
    ## panel and collided with the letter of the one below.
    fig.supylabel(meta["ylabel"], x=0.005, fontsize=utils.FONT["label"])
    _legend(fig, meta)
    return utils.save(fig, stem, png=png)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ## One faculty per figure: Faculty A in the main text, Faculty B in the SI.
    ap.add_argument("--stem", default="pilot_auc_faculty_a",
                    choices=["pilot_auc_faculty_a", "pilot_auc_faculty_b"])
    ap.add_argument("--png", action="store_true")
    args = ap.parse_args()
    draw(args.stem, png=args.png)


if __name__ == "__main__":
    main()
