"""
When dropouts leave, and how much of it each rule had already caught.

Four panels — overall, then sex, nationality and entry age. In each, the
faint curve is every dropout in that group, and the dashed and dotted ones are
the share the credits policy and the Participatory model had flagged by that
month. A curve's plateau is its recall, marked with a faint horizontal line.
The dropout_timing_maxml variant adds Max ML as a dash-dotted curve.

The curves rise once per half-year rather than once per student, and a half-year
resting on too few dropouts is merged into the next, so no step locates one
student's departure. A layer whose recall would rest on too few keeps its legend
entry and loses its curve.

    python figures/scripts/draw_dropout_timing.py [--stem dropout_timing_maxml] [--png]
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

import utils

## Line style per layer kind. "background" is the group's own all-dropout curve,
## drawn faint and unlabelled — the thing the other two are read against.
_STYLE = {
    "solid":      dict(linestyle="-", linewidth=1.6, alpha=0.40),
    "background": dict(linestyle="-", linewidth=1.2, alpha=0.18),
    "dashed":     dict(linestyle="--", linewidth=1.8),
    "dotted":     dict(linestyle=":",  linewidth=1.8),
    ## A tighter dash-dot than matplotlib's "-.", so a full dash-dot cycle
    ## fits in the short legend handle; "-." shows there as a plain dash.
    "dashdot":    dict(linestyle=(0, (3, 1.2, 0.8, 1.2)), linewidth=1.6),
}
_HLINE = {"solid": None,
          "background": None,
          "dashed": dict(linestyle="--", linewidth=0.7, alpha=0.35),
          "dotted": dict(linestyle=":",  linewidth=0.7, alpha=0.35),
          "dashdot": dict(linestyle=(0, (3, 1.2, 0.8, 1.2)), linewidth=0.7, alpha=0.35)}
## Panel a's recall lines are a touch heavier than the subgroup panels'.
_HLINE_SUB = {"dashed": dict(linestyle="--", linewidth=0.6, alpha=0.30),
              "dotted": dict(linestyle=":",  linewidth=0.6, alpha=0.30),
              "dashdot": dict(linestyle=(0, (3, 1.2, 0.8, 1.2)), linewidth=0.6, alpha=0.30)}


def _compact(label):
    """
    "Non-Danish, Participatory (NN%)" rather than "Non-Danish — Participatory
    (NN%)": at the paper's type size each legend is as wide as its panel, and
    the spaced dash ran one legend into the next.
    """
    return " ".join(label.replace(" — ", ", ").split()) if label else label


def draw(stem: str = "dropout_timing", png: bool = False):
    utils.apply_style()
    df, meta = utils.read_input(stem)

    ## Spacing set by hand, not by constrained_layout: the legends above the
    ## panels are wider than the panels, and a layout engine that makes room
    ## for them pushes the panels apart into narrow strips with wide gaps.
    ## The top margin is reserved for them in the JSON (`subplots_adjust`).
    fig, axes = plt.subplots(1, len(meta["panels"]), figsize=meta["figsize"],
                             sharey=True)
    fig.subplots_adjust(**meta["subplots_adjust"])
    for spec in meta["layers"]:
        ax     = axes[spec["panel"]]
        colour = utils.SERIES_COLOR[spec["colour"]]
        style  = _STYLE[spec["style"]]

        if not spec["drawn"]:
            ## Legend entry with no curve behind it — the group is named, its
            ## timing is not shown.
            ax.plot([], [], color=colour, linestyle=style["linestyle"],
                    linewidth=style["linewidth"], label=_compact(spec["label"]))
            continue

        points = df[df["layer"] == spec["layer"]].sort_values("x")
        x, y = points["x"].to_numpy(), points["y"].to_numpy()
        ax.step(x, y, where="post", color=colour, zorder=spec["zorder"],
                label=_compact(spec["label"]) or None, **style)
        if spec["fill_alpha"] is not None:
            ## Its own z, not the curve's minus one: in the subgroup panels both
            ## fills sit under every curve so the lines stay readable where the
            ## shaded areas overlap.
            ax.fill_between(x, y, step="post", color=colour,
                            alpha=spec["fill_alpha"], zorder=spec["fill_zorder"])
        if spec["hline"] is not None:
            hl = (_HLINE if spec["panel"] == 0 else _HLINE_SUB)[spec["style"]]
            ax.axhline(spec["hline"], color=colour, **hl)

    for spec in meta["panels"]:
        ax = axes[spec["panel"]]
        if spec["ylabel"]:
            ax.set_ylabel(meta["ylabel"], labelpad=6)
        if ax.get_legend_handles_labels()[0]:
            ## One legend per panel, above it and left-aligned to its own y
            ## axis, so each panel's entries stay with the panel.
            ax.legend(loc=meta["legend"]["loc"],
                      bbox_to_anchor=tuple(meta["legend"]["bbox_to_anchor"]),
                      ncol=1, frameon=False, fontsize=utils.FONT["legend"],
                      handlelength=1.2, handletextpad=0.4,
                      labelspacing=meta["legend"]["labelspacing"])
        ## Inside the axes, top-left, where every curve is still at zero: above
        ## the panel it sat in the previous panel's legend.
        ax.text(0.03, 0.97, spec["letter"], transform=ax.transAxes,
                fontsize=utils.FONT["panel"], fontweight="bold", va="top")

    for ax in axes:
        ax.set_xlabel(meta["xlabel"], labelpad=6)
        ax.set_xlim(*meta["xlim"])
        ax.set_ylim(*meta["ylim"])
        ax.set_xticks(meta["xticks"])
        ax.grid(axis="both", linestyle=":", linewidth=0.5, color="gray", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)

    return utils.save(fig, stem, png=png)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stem", default="dropout_timing",
                    choices=["dropout_timing", "dropout_timing_maxml"])
    ap.add_argument("--png", action="store_true")
    args = ap.parse_args()
    draw(stem=args.stem, png=args.png)


if __name__ == "__main__":
    main()
