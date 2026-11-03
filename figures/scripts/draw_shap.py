"""
What the model reads: the SHAP layered violin, one row per feature.

Each row is a stack of filled bands — one per value-quantile of that feature —
so the figure shows the shape of the distribution rather than where any one
student sits in it. That is also all this script has: the band outlines and
their colours, never the values behind them.

The paper prints the two models side by side in one landscape figure
(shap_participatory_maxml: a Participatory, b Max ML); each model also has a
figure of its own.

    python figures/scripts/draw_shap.py --stem shap_participatory_maxml [--png]
    python figures/scripts/draw_shap.py --stem shap_mlp_e_plus_sensitive [--png]
"""
from __future__ import annotations

import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt

import utils

_NAME_SIZE = utils.FONT["tick"]     # feature names and ticks
_LABEL_SIZE = utils.FONT["label"]


def _fit_xlabel(ax, box, pad_pt: float = 2.0) -> None:
    """
    Keep the x-axis label inside its own panel.

    The label is centred under the axes, and the axes start where the longest
    feature name ends — so a long name narrows the axes, and a centred label
    can run off the panel's right edge. Shift it
    left until its right end is inside the panel; if it is wider than the
    whole panel, wrap it before the parenthesis as well.
    """
    fig = ax.figure
    fig.canvas.draw()                       # the label follows the new axes box
    renderer = fig.canvas.get_renderer()
    pad = pad_pt * fig.dpi / 72
    label = ax.xaxis.label
    ext = label.get_window_extent(renderer)
    if ext.x1 <= box.x1 - pad:
        return
    if ext.width > box.width - 2 * pad and " (" in label.get_text():
        label.set_text(label.get_text().replace(" (", "\n(", 1))
        label.set_multialignment("center")
        fig.canvas.draw()
        ext = label.get_window_extent(renderer)
        if ext.x1 <= box.x1 - pad:
            return
    ## Label coords are in axes fractions: x is the label's centre, y its top
    ## (va="top"), kept where the default spacing put it.
    ax_box = ax.get_window_extent(renderer)
    x = (box.x1 - pad - ext.width / 2 - ax_box.x0) / ax_box.width
    y = (ext.y1 - ax_box.y0) / ax_box.height
    ax.xaxis.set_label_coords(x, y)


def _draw_panel(container, df, meta, letter: str | None = None,
                name_w: float | None = None) -> float:
    """
    One violin plot into a figure or subfigure, laid out to fit its names.

    `name_w` (inches) overrides the width left for the feature names, so panels
    side by side can share one; returns the width this panel's own names need.
    """
    ax = container.add_axes(meta["axes_position"])

    ## Bands first, in the order shap drew them, so the stacking reads the same.
    for band, colour in enumerate(meta["band_colors"]):
        pts = df[df["band"] == band].sort_values("order")
        if not len(pts):
            continue
        ax.fill(pts["x"], pts["y"], facecolor=colour, edgecolor="none", zorder=2)

    ## Then whatever straight lines shap put on top — the zero reference, which
    ## spans the panel rather than living in the data.
    for j, line in enumerate(meta["lines"]):
        style = dict(color=line["color"], linewidth=line["linewidth"],
                     linestyle=line["linestyle"], zorder=3)
        if line["kind"] == "axvline":
            ax.axvline(line["at"], **style)
            continue
        pts = df[df["band"] == -1 - j].sort_values("order")
        ax.plot(pts["x"], pts["y"], **style)

    ax.set_xlim(*meta["xlim"])
    ax.set_ylim(*meta["ylim"])
    ax.set_yticks(meta["yticks"])
    ax.set_yticklabels(meta["yticklabels"], fontsize=_NAME_SIZE)
    ax.tick_params(axis="x", labelsize=_NAME_SIZE)
    ax.set_xlabel(meta["xlabel"], fontsize=_LABEL_SIZE)
    ax.set_title(meta["title"], fontsize=utils.FONT["title"], pad=8)
    ax.spines[["top", "right"]].set_visible(False)

    ## The axes start where the longest feature name ends and stop short of the
    ## colour key, measured in this container's own width.
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    box = container.bbox
    w_in, h_in = box.width / fig.dpi, box.height / fig.dpi
    own_w = max(t.get_window_extent(renderer).width for t in ax.get_yticklabels()) / fig.dpi
    left = ((own_w if name_w is None else name_w) + 0.15) / w_in
    bottom = max(meta["axes_position"][1], 0.75 / h_in)
    top = min(meta["axes_position"][1] + meta["axes_position"][3], 1 - 0.45 / h_in)
    right = 0.86 if meta["colorbar"] is not None else 0.98
    ax.set_position([left, bottom, right - left, top - bottom])
    _fit_xlabel(ax, box)
    if letter:
        container.text(0.01, top + 0.25 / h_in, letter, ha="left", va="bottom",
                       fontsize=utils.FONT["panel"], fontweight="bold")

    cb = meta["colorbar"]
    if cb is not None:
        ## shap's colour key: the feature's own value range, low to high. A
        ## real Colorbar, rasterised as shap draws it (vector quads leave seams).
        cax = container.add_axes([right + 0.025, bottom, 0.022, top - bottom])
        norm = mpl.colors.Normalize(vmin=cb["clim"][0], vmax=cb["clim"][1])
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cb["cmap"])
        bar = container.colorbar(sm, cax=cax, ticks=cb["yticks"])
        bar.solids.set_rasterized(cb["rasterized"])
        bar.outline.set_visible(False)
        cax.set_yticklabels(cb["yticklabels"], fontsize=_NAME_SIZE)
        cax.tick_params(length=0)
        container.text(0.975, (bottom + top) / 2, cb["ylabel"], rotation=cb["label_rotation"],
                       ha="center", va="center", fontsize=_LABEL_SIZE)

    ## meta["note"] (merged values, skipped features) belongs in the caption,
    ## not in the axes — the figure carries no text of its own about it.
    return own_w


def draw(stem: str, png: bool = False):
    utils.apply_style()
    df, meta = utils.read_input(stem)
    fig = plt.figure(figsize=meta["figsize"])
    if "panels" in meta:
        ## Side by side: one subfigure per model. Drawn twice: once to measure
        ## each panel's feature names, then with the longer of them left on both,
        ## so the two plots come out the same width.
        def panels(name_w=None):
            subs = fig.subfigures(1, len(meta["panels"]), wspace=0.0)
            return [_draw_panel(sub, df[df["panel"] == i], spec, letter="abcd"[i],
                                name_w=name_w)
                    for i, (sub, spec) in enumerate(zip(subs, meta["panels"]))]
        name_w = max(panels())
        plt.close(fig)
        fig = plt.figure(figsize=meta["figsize"])
        panels(name_w)
    else:
        _draw_panel(fig, df, meta)
    return utils.save(fig, stem, png=png)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stem", default=None,
                    help="one prepared input; default: every shap_*.json in figures/input/")
    ap.add_argument("--png", action="store_true")
    args = ap.parse_args()
    ## Which models get a SHAP figure is decided when the input is prepared (Max
    ## ML, and whatever the workshop rules selected), so by default every
    ## prepared input is drawn rather than a list that assumes the selection.
    stems = [args.stem] if args.stem else sorted(p.stem for p in utils.INPUT.glob("shap_*.json"))
    if not stems:
        raise SystemExit("No SHAP input in figures/input/ — run prepare_shap.py first.")
    for stem in stems:
        draw(stem, png=args.png)


if __name__ == "__main__":
    main()
