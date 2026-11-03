"""
How the Participatory model performs as a cohort is observed for longer.

    a       five metrics against EWS activation, with 95% bootstrap bands
    b, c, d the same five as a DIFFERENCE between two groups, against a zero line

A point whose confusion cells rest on too few students is absent, and its band
with it — a band drawn around a missing point would outline where it sat.

    python figures/scripts/draw_performance_by_period.py [--png]
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

import utils


def draw(stem: str = "performance_by_period", png: bool = False):
    utils.apply_style()
    df, meta = utils.read_input(stem)

    fig = plt.figure(figsize=meta["figsize"], constrained_layout=True)
    gs = fig.add_gridspec(1, len(meta["width_ratios"]), width_ratios=meta["width_ratios"])
    handles, labels = [], []

    for spec in meta["panels"]:
        ax = fig.add_subplot(gs[spec["index"]])
        ax.text(-0.12, 1.10, spec["letter"], transform=ax.transAxes,
                fontsize=utils.FONT["panel"], fontweight="bold", va="top")
        if not spec["visible"]:
            ax.set_visible(False)
            continue

        panel = df[df["panel"] == spec["index"]]
        for metric in meta["metrics"]:
            series = panel[panel["metric"] == metric].sort_values("activation")
            colour = utils.METRIC_COLOR[metric]
            line, = ax.plot(series["activation"], series["y"], color=colour,
                            marker=utils.METRIC_MARKER[metric], markersize=5,
                            linewidth=1.8, markeredgewidth=0,
                            label=meta["metric_labels"][metric])
            ax.fill_between(series["activation"], series["lo"], series["hi"],
                            color=colour, alpha=0.15)
            if spec["kind"] == "overall":
                handles.append(line)
                labels.append(meta["metric_labels"][metric])

        if spec["zero_line"]:
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=1)
        if spec["note"]:
            utils.note(ax, meta["note"])
        ax.set_xlabel(meta["xlabel"], labelpad=6)
        ax.set_ylabel(spec["ylabel"], labelpad=6)
        ax.set_xticks(meta["xticks"])
        ax.spines[["top", "right"]].set_visible(False)

    ## The three difference panels share one y range, so a gap reads the same
    ## size in each; the overall panel is on its own scale (scores, not gaps).
    diffs = [ax for ax, spec in zip(fig.axes, meta["panels"])
             if spec["visible"] and spec["kind"] != "overall"]
    if diffs:
        lo = min(ax.get_ylim()[0] for ax in diffs)
        hi = max(ax.get_ylim()[1] for ax in diffs)
        for ax in diffs:
            ax.set_ylim(lo, hi)

    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.14),
               ncol=len(meta["metrics"]), frameon=False, fontsize=utils.FONT["legend"], handlelength=1.5)
    return utils.save(fig, stem, png=png)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--png", action="store_true")
    draw(png=ap.parse_args().png)


if __name__ == "__main__":
    main()
