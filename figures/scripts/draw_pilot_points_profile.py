"""
The pilot's points score against what happened: dropout rate by score (top) and
how many enrollments hold each score (bottom), one column per faculty.

    dashed line  the score at which the faculty actually started contacting
                 (Faculty A only: Faculty B's threshold is not known, and its
                 input carries no line)
    wide bar     several scores merged, because one of them rested on too few
                 students; nothing is hidden, so the shown bins still add up to
                 the faculty total

    python figures/scripts/draw_pilot_points_profile.py [--png]
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

import utils


def draw(stem: str = "pilot_points_profile_faculty_a", png: bool = False):
    utils.apply_style()
    df, meta = utils.read_input(stem)

    row = meta.get("layout") == "row"
    if row:
        ## One faculty: dropout rate (a) and count (b) side by side.
        fig, axes = plt.subplots(1, 2, figsize=meta["figsize"], constrained_layout=True)
        pairs = [(meta["panels"][0], axes[0], axes[1])]
    else:
        fig, axes = plt.subplots(2, len(meta["panels"]), figsize=meta["figsize"],
                                 constrained_layout=True, sharex=True)
        pairs = [(spec, axes[0, spec["panel"]], axes[1, spec["panel"]])
                 for spec in meta["panels"]]
    for spec, ax_rate, ax_count in pairs:
        panel  = df[df["panel"] == spec["panel"]].sort_values("mid")
        colour = utils.FACULTY_COLOR[spec["faculty"]]

        ## Rate: an empty bin has no rate, so matplotlib breaks the line there.
        ax_rate.plot(panel["mid"], panel["rate"], color=colour, marker="o", ms=4, lw=1.8)
        ax_rate.set_ylim(*meta["rate_ylim"])
        ax_rate.set_ylabel(meta["ylabel_rate"])
        ax_rate.spines[["top", "right"]].set_visible(False)
        utils.panel_label(ax_rate, "a" if row else spec["letter"])
        if row:
            utils.panel_label(ax_count, "b")
            ax_rate.set_xlabel(meta["xlabel"])
            ax_rate.set_xticks(meta["xticks"])
            ax_rate.tick_params(axis="x", labelsize=utils.FONT["tick"])

        n_max = max(panel["n"].max(), 1)
        ax_count.bar(panel["mid"], panel["n"], color=colour, alpha=0.85,
                     width=panel["width"])
        for row in panel.itertuples():
            if row.count_label:
                ax_count.text(row.mid, row.n + n_max * 0.01, row.count_label,
                              ha="center", va="bottom", fontsize=utils.FONT["annot"])
        ax_count.set_ylim(0, n_max * 1.2)   # room for the count labels and the note
        ax_count.set_ylabel(meta["ylabel_count"])
        ax_count.set_xlabel(meta["xlabel"])
        ax_count.set_xticks(meta["xticks"])
        ax_count.tick_params(axis="x", labelsize=utils.FONT["tick"])
        ax_count.spines[["top", "right"]].set_visible(False)

        if spec["threshold_x"] is not None:
            for ax in (ax_rate, ax_count):
                ax.axvline(spec["threshold_x"], color=colour, linestyle="--", lw=1.4)
        if spec["panel"] in meta["merged_note_panels"]:
            utils.note(ax_count, meta["note"])

    return utils.save(fig, stem, png=png)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stem", default="pilot_points_profile_faculty_a",
                    choices=["pilot_points_profile_faculty_a", "pilot_points_profile_faculty_b"])
    ap.add_argument("--png", action="store_true")
    args = ap.parse_args()
    draw(args.stem, png=args.png)


if __name__ == "__main__":
    main()
