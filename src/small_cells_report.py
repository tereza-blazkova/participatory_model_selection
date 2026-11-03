"""
The small-group cases the pilot figures cannot resolve on their own.

Three checks in the pilot figures guard a disclosure risk that suppressing one
cell does not fix, because two published figures together could give a hidden
group back by subtraction:

  contact_threshold  the pilot threshold recovered from the contact list is the
                     lowest score among contacted students; printing it states
                     that group's value
  points_bin         a whole stretch of the points profile between two published
                     thresholds still holds a small group after merging
  threshold_band     the outcome-breakdown figure draws a rule at the pilot's
                     own threshold and the recalibrated figure the same rule at
                     tau; differencing the two bars gives the students between

A case is reported on the console when it occurs and the run carries on. No
file is written: the decision it calls for (redraw, merge, drop the figure, or
accept it in the caption) belongs to whoever publishes the figures.
"""
from __future__ import annotations

_KINDS = {
    "contact_threshold": "Pilot contact threshold",
    "points_bin":        "Points-profile bin",
    "threshold_band":    "Pilot vs. recalibrated threshold",
}


def record(kind: str, where: str, what: str, counts: dict, artefacts: list[str]) -> None:
    """
    Report one unresolved small group on the console.

    kind      one of _KINDS
    where     the population it happened in (faculty, panel, threshold)
    what      one sentence: which group is exposed, and by what
    counts    {label: n} — the group sizes that failed the rule
    artefacts the figures that together carry the risk
    """
    smallest = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print(f"  SMALL CELLS: {_KINDS.get(kind, kind)} — {where}: {what} ({smallest})",
          flush=True)
    print(f"    concerns {', '.join(sorted(artefacts))}; the figure is still written.",
          flush=True)
