"""
The FA frontier decomposed: what each rung of the ladder buys.

    row 1   data ladder  — one model, feature sets a through e
    row 2   model ladder — one feature set, FasterRisk through NN
    columns sex, nationality, entry age — as in the FA frontier figure

The two rows answer different questions. Adding data changes the information
structure the frontier is computed over; changing model shrinks the set of
permissible algorithms. `e_plus_sensitive` appears twice in the data row —
solid without group-aware decision rules, dashed with them — because more
covariates and group-specific thresholds are different effects and the figure
must not blur them.

Filled endpoints are the two group-optimal points; a hollow one is f_X, drawn
only where the frontier is skewed. The data ladder is nested and we pick the
algorithm, so a crossing there is estimation noise; the model classes are not
nested, so a crossing in the model row is a finding.

    python figures/scripts/draw_fa_frontier_ladders.py [--year 2021] [--png]
"""
from __future__ import annotations

import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import utils


def _ladder_style(meta, ladder):
    """Colours and markers for one ladder's series (0 = data, 1 = model), in order."""
    specs = [s for s in meta["series"] if s["row"] == ladder]
    if ladder == 1:
        return ([utils.SERIES_COLOR[s["colour"]] for s in specs],
                [utils.MARKERS[i] for i in range(len(specs))], specs)
    ramp = meta["data_ramp"]
    colours = [mpl.colormaps[ramp["cmap"]](v)
               for v in np.linspace(ramp["start"], ramp["stop"], ramp["n"])]
    if ramp["repeat_last"]:
        ## The group-aware variant is the same feature set drawn again, so it
        ## keeps that feature set's colour and is told apart by the dash.
        colours.append(colours[-1])
    return colours, meta["data_markers"], specs


def draw(stem: str = "fa_frontier_ladders_2021", png: bool = False):
    utils.apply_style()
    df, meta = utils.read_input(stem)
    lo, hi = meta["lim"]

    n_cols = max(p["col"] for p in meta["panels"]) + 1
    fig, axes = plt.subplots(2, n_cols, figsize=meta["figsize"], constrained_layout=True)

    for spec in meta["panels"]:
        ax = axes[spec["row"], spec["col"]]
        colours, markers, series = _ladder_style(meta, spec["row"])
        panel = df[(df["row"] == spec["row"]) & (df["col"] == spec["col"])]

        for s, colour, marker in zip(series, colours, markers):
            rung = panel[panel["series"] == s["series"]]

            path = rung[rung["kind"] == "path"].sort_values("order")
            ax.plot(path["x"], path["y"], color=colour, lw=1.8,
                    ls="--" if s["group_aware"] else "-", zorder=3)

            ends = rung[rung["kind"] == "endpoint"].sort_values("order")
            ax.plot(ends["x"], ends["y"], marker, color=colour, ms=5,
                    mec="white", mew=0.7, ls="none", zorder=4)

            fair = rung[rung["kind"] == "fair"]
            if len(fair):
                ax.plot(fair["x"], fair["y"], marker, color=colour, ms=5,
                        mfc="white", mec=colour, mew=1.1, ls="none", zorder=4)

        ax.axline((lo, lo), (hi, hi), color=utils.FA_COLOR["cloud"], lw=0.9,
                  ls=":", zorder=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1, decimals=0))
            axis.set_major_locator(mpl.ticker.MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel(spec["xlabel"], labelpad=6)
        ax.set_ylabel(spec["ylabel"], labelpad=6)
        ax.text(-0.16, 1.06, spec["letter"], transform=ax.transAxes,
                fontsize=utils.FONT["panel"], fontweight="bold", va="top")

    ## Each ladder's legend sits to the right of its row, never inside the axes:
    ## the frontiers sit in a different corner in every panel, so no in-axes
    ## position is safe. Attached to the row's last axes so constrained_layout
    ## makes room for it.
    for row, title in enumerate(meta["legend_titles"]):
        colours, markers, series = _ladder_style(meta, row)
        handles = [Line2D([], [], color=c, lw=1.8, marker=m, ms=5,
                          ls="--" if s["group_aware"] else "-", label=s["label"])
                   for s, c, m in zip(series, colours, markers)]
        leg = axes[row, -1].legend(handles=handles, loc="center left",
                                   bbox_to_anchor=(1.04, 0.5), frameon=False,
                                   fontsize=utils.FONT["legend"], handlelength=1.8,
                                   labelspacing=0.35, title=title)
        leg.get_title().set_fontsize(utils.FONT["legend"])
    return utils.save(fig, stem, png=png)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year", type=int, default=2021)
    ap.add_argument("--png", action="store_true")
    args = ap.parse_args()
    draw(f"fa_frontier_ladders_{args.year}", png=args.png)


if __name__ == "__main__":
    main()
