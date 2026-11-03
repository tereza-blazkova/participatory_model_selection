"""
ABROCA for the October 2023 pilot: how far apart two groups' ROC curves lie under
each decision rule, one heatmap per faculty.

Rows are the three fairness axes, columns the rules and models. The colour scale
is shared across both panels — a cell means the same thing wherever it sits.

A cell is blank-with-a-marker when either group is too small for the pair of ROC
curves the measure is built from, and "n/a" when the rule was never run here.
The two are different things and the figure says which.

    python figures/scripts/draw_pilot_abroca.py [--png]
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import utils


def draw(stem: str = "pilot_abroca", png: bool = False):
    utils.apply_style()
    df, meta = utils.read_input(stem)

    n_rows = int(df["row"].max()) + 1
    n_cols = int(df["col"].max()) + 1
    vmax   = meta["vmax"]
    cmap   = LinearSegmentedColormap.from_list("abroca", meta["cmap"])

    fig, axes = plt.subplots(1, len(meta["panels"]), figsize=meta["figsize"],
                             constrained_layout=True)
    ims = []
    for ax, spec in zip(np.atleast_1d(axes), meta["panels"]):
        panel = df[df["panel"] == spec["panel"]]
        grid = np.full((n_rows, n_cols), np.nan)
        for row in panel.itertuples():
            grid[row.row, row.col] = row.value
        ims.append(ax.imshow(grid, cmap=cmap, aspect="auto", vmin=0, vmax=vmax))

        for row in panel.itertuples():
            if row.value == row.value:
                ## White on the dark end of the scale; elsewhere the style's
                ## text colour (None), the same as the row and column labels.
                colour = "white" if row.value > 0.6 * vmax else None
                ax.text(row.col, row.row, f"{row.value:.3f}", ha="center", va="center",
                        fontsize=utils.FONT["annot"], color=colour, fontweight="bold")
            else:
                text = meta["marker"] if row.hidden else meta["not_applicable"]
                ax.text(row.col, row.row, text, ha="center", va="center",
                        fontsize=utils.FONT["annot"], color="gray")

        cols = panel.sort_values("col").drop_duplicates("col")["col_label"].tolist()
        rows = panel.sort_values("row").drop_duplicates("row")["row_label"].tolist()
        ax.set_xticks(range(n_cols))
        ## One line each, slanted as in the model comparison's ABROCA panel:
        ## upright, the names had to wrap and still ran together.
        ax.set_xticklabels([" ".join(c.split()) for c in cols], fontsize=utils.FONT["tick"],
                           rotation=30, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(rows if spec["ylabels"] else [""] * n_rows, fontsize=utils.FONT["tick"])
        utils.panel_label(ax, spec["letter"])
        ax.spines[["top", "right"]].set_visible(False)

    cbar = fig.colorbar(ims[0], ax=np.atleast_1d(axes).tolist(), fraction=0.015, pad=0.02)
    cbar.outline.set_visible(False)
    ## Only the two ends, as in the model comparison: the cells carry the values.
    cbar.set_ticks([0, vmax], labels=["0.00", f"{vmax:.2f}"])
    cbar.ax.tick_params(labelsize=utils.FONT["tick"])
    ## Its name between them, in the tick column, so it takes no width of its own.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ticks = [t.get_window_extent(renderer) for t in cbar.ax.get_yticklabels()]
    centre = cbar.ax.transAxes.inverted().transform(
        ((min(t.x0 for t in ticks) + max(t.x1 for t in ticks)) / 2, 0))[0]
    cbar.ax.text(centre, 0.5, meta["cbar_label"], transform=cbar.ax.transAxes,
                 rotation=90, ha="center", va="center", fontsize=utils.FONT["label"])
    return utils.save(fig, stem, png=png)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--png", action="store_true")
    draw(png=ap.parse_args().png)


if __name__ == "__main__":
    main()
