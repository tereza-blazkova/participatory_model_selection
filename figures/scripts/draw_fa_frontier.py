"""
The fairness-accuracy frontier: what pairs of group error rates are reachable at
all, and what the workshop's participation rules cost inside that set.

    grey cloud    every candidate (model, feature set, threshold)
    solid line    the frontier over all algorithms
    dashed line   the frontier over the workshop-permissible ones
    shaded band   the gap between them — the cost of the participation rules
    dotted 45°    equal error rates for both groups
    markers       the rules as actually deployed

All panels share one range on both axes, so the 45° line is a true diagonal and
a disparity that looks twice as wide is twice as wide — across panels too.

    python figures/scripts/draw_fa_frontier.py [--year 2021] [--png]
"""
from __future__ import annotations

import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

import utils


def _path(panel, kind):
    p = panel[panel["kind"] == kind].sort_values("order")
    return p[["x", "y"]].to_numpy()


def draw(stem: str = "fa_frontier_2021", png: bool = False):
    utils.apply_style()
    df, meta = utils.read_input(stem)
    lo, hi = meta["lim"]

    fig, axes = plt.subplots(1, len(meta["panels"]), figsize=meta["figsize"],
                             constrained_layout=True)
    for ax, spec in zip(np.atleast_1d(axes), meta["panels"]):
        panel = df[df["panel"] == spec["panel"]]
        fa, fp = _path(panel, "frontier_all"), _path(panel, "frontier_perm")

        ## The band is the region between the two frontiers: close the polygon
        ## by walking one out and the other back.
        ax.add_patch(Polygon(np.vstack([fa, fp[::-1]]), closed=True,
                             facecolor=utils.FA_COLOR["band"], alpha=0.18,
                             lw=0, zorder=1))

        cloud = _path(panel, "cloud")
        ## Rasterised: thousands of markers as vectors make a PDF that no
        ## reader can open, and the cloud carries no detail worth keeping sharp.
        ax.scatter(cloud[:, 0], cloud[:, 1], s=1.5, color=utils.FA_COLOR["cloud"],
                   alpha=0.30, linewidths=0, rasterized=True, zorder=2)

        ax.plot(fp[:, 0], fp[:, 1], color=utils.FA_COLOR["permissible"],
                lw=2.2, ls="--", zorder=4)
        ax.plot(fa[:, 0], fa[:, 1], color=utils.FA_COLOR["unconstrained"],
                lw=2.2, zorder=5)

        for row in panel[panel["kind"] == "operating_point"].itertuples():
            ax.plot(row.x, row.y, utils.FA_MARKER[row.style], ms=7,
                    color=utils.FA_COLOR[row.style], mec="white", mew=0.9, zorder=8)

        ## 45° equal-error reference. Identical limits on both axes make it a
        ## true diagonal.
        ax.axline((lo, lo), (hi, hi), color=utils.FA_COLOR["cloud"], lw=0.9,
                  ls=":", alpha=0.9, zorder=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")

        for row in panel[panel["kind"] == "named"].itertuples():
            ax.plot(row.x, row.y, "o", ms=5, color=utils.FA_COLOR["unconstrained"],
                    mec="white", mew=0.8, zorder=7)
            ## The offset was resolved when the input was built, so a label
            ## never lands on the frontier line or on its neighbour.
            ax.annotate(row.label, xy=(row.x, row.y), xytext=(row.dx, row.dy),
                        textcoords="offset points", fontsize=utils.FONT["annot"],
                        color=utils.FA_COLOR["unconstrained"], zorder=7)

        ax.set_xlabel(spec["xlabel"], labelpad=6)
        ax.set_ylabel(spec["ylabel"], labelpad=6)
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1, decimals=0))
            axis.set_major_locator(mpl.ticker.MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(-0.2, 1.1, spec["letter"], transform=ax.transAxes, fontsize=utils.FONT["panel"],
                fontweight="bold", va="top")

    c = utils.FA_COLOR
    handles = [
        Line2D([], [], color=c["unconstrained"], lw=2.2, label="FA frontier — all algorithms"),
        Line2D([], [], color=c["permissible"], lw=2.2, ls="--",
               label="FA frontier — workshop-permissible"),
        Line2D([], [], color=c["band"], lw=7, alpha=0.35,
               label="Cost of the participation constraints"),
        Line2D([], [], color=c["cloud"], marker=".", ls="none", ms=4,
               label=r"Candidate (model, feature set, $\tau$)"),
        Line2D([], [], color=c["cloud"], ls=":", lw=0.9, label="Equal error rates"),
        Line2D([], [], color=c["credits"], marker=utils.FA_MARKER["credits"], ls="none",
               ms=6, label=meta["legend"]["credits"]),
        Line2D([], [], color=c["participatory"], marker=utils.FA_MARKER["participatory"],
               ls="none", ms=6, label=meta["legend"]["participatory"]),
        Line2D([], [], color=c["max_ml"], marker=utils.FA_MARKER["max_ml"], ls="none",
               ms=6, label=meta["legend"]["max_ml"]),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.13),
               ncol=3, frameon=False, fontsize=utils.FONT["legend"], handlelength=1.8, columnspacing=1.4)
    return utils.save(fig, stem, png=png)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year", type=int, default=2021)
    ap.add_argument("--png", action="store_true")
    args = ap.parse_args()
    draw(f"fa_frontier_{args.year}", png=args.png)


if __name__ == "__main__":
    main()
