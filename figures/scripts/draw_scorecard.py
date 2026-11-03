"""
The FasterRisk integer scorecard: the whole model, as the points each condition
adds to a student's risk score.

Teal raises the risk, pink lowers it. Each label states the cut the term was
binarised at, so the card can be applied by hand.

    python figures/scripts/draw_scorecard.py [--png]
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

import utils


def draw(stem: str = "scorecard_used_vars", png: bool = False):
    utils.apply_style()
    df, meta = utils.read_input(stem)

    fig, ax = plt.subplots(figsize=meta["figsize"], constrained_layout=True)
    y = df["y"].to_numpy()
    colours = [utils.SCORE_POS_COLOR if s > 0 else utils.SCORE_NEG_COLOR for s in df["score"]]
    ax.barh(y, df["score"], color=colours, alpha=0.88, edgecolor="white",
            linewidth=0.4, height=0.6)
    ax.axvline(0, color="#333333", linewidth=0.8)

    ## Each score sits just past the end of its own bar, on the side the bar
    ## points, so the numbers never overlap the zero line.
    x_abs_max = max(abs(s) for s in df["score"])
    for row in df.itertuples():
        forward = row.score >= 0
        ax.text(row.score + 0.08 * x_abs_max * (1 if forward else -1), row.y,
                row.score_label, va="center", ha="left" if forward else "right",
                fontsize=utils.FONT["annot"], fontweight="bold")

    ax.set_xlim(-x_abs_max * 1.35, x_abs_max * 1.35)
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"].tolist(), fontsize=utils.FONT["tick"])
    ax.invert_yaxis()
    ## A FIGURE-level label, not an axis one. The feature names are long enough
    ## to push the axes into the right half of the canvas, and an axis label is
    ## centred on the axes — so it could run off the right edge of the figure.
    fig.supxlabel(meta["xlabel"], fontsize=utils.FONT["label"])

    ## Bottom-left, inside the axes, and out of the layout so it cannot shrink
    ## the plot area. The scores are sorted by magnitude, so the space that looks
    ## free on the right of the short bottom bars is exactly where their numbers go.
    note = ax.text(0.01, 0.01, meta["intercept_note"], transform=ax.transAxes,
                   ha="left", va="bottom", fontsize=utils.FONT["annot"], color="gray")
    note.set_in_layout(False)
    ax.spines[["top", "right"]].set_visible(False)

    return utils.save(fig, stem, png=png)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--png", action="store_true")
    draw(png=ap.parse_args().png)


if __name__ == "__main__":
    main()
