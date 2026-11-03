"""
Distil the fairness-accuracy frontier (Liang, Lu, Mu & Okumura, JPE 2024): what
pairs of group error rates any algorithm here can reach, and what the workshop's
participation rules cost within that set.

    figures/input/fa_frontier_{year}.csv    every drawn point, by kind
    figures/input/fa_frontier_{year}.json   axes, labels, legend

and, because both come from the same expensive build, also
results/fairness/fa_frontier_summary_{year}.csv — every number the caption might
quote, deliberately kept out of the axes.

src/fa_frontier.py owns the computation: the candidate sweep over (model,
feature set, tau), the convex hull, the Pareto arc and its extension to f_X, the
group-skew bootstrap. Its threshold sweep already drops any tau whose confusion
cells rest on too few students (`_disclosable_taus`), so the cloud that reaches
this file has passed disclosure control before it is thinned for drawing.

    python figures/scripts/prepare_fa_frontier.py [--year 2021] [--n-boot 1000]
                                                  [--zoom] [--full-range]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from paths.paths import RESULTS_PATH                                   # noqa: E402
from prepare_utils import write_input                                  # noqa: E402
from src.config import MAIN_TEST_YEAR, MAX_ML_FS, MAX_ML_MODEL         # noqa: E402
from src.data import load_analysis_df                                  # noqa: E402
from utils import canvas                                               # noqa: E402
from src.fa_frontier import (
    _FS_LABELS,
    _MODEL_LABELS,
    _SUBGROUPS,
    _axis_limits,
    build_subgroup_data,
    label_offsets,
)

SCRIPT = "prepare_fa_frontier.py"

## The cloud is drawn rasterised and thinned; thinning HERE means the file holds
## exactly the points the figure shows, not a superset of them.
_MAX_CLOUD = 4000

## Each deployed rule's marker, in the order build_subgroup_data returns them.
_OP_STYLE = ["credits", "participatory", "max_ml"]


def prepare(panels: dict, year: int, full_range: bool, zoom: bool):
    if full_range:
        P = np.vstack([d["cloud"] for d in panels.values()])
        lo, hi = float(np.nanmin(P)), float(np.nanmax(P))
    else:
        lo, hi = _axis_limits(list(panels.values()), include_ops=not zoom)

    rows, specs = [], []
    for idx, (name, d) in enumerate(panels.items()):
        spec = d["spec"]
        fa, fp = d["frontier_all"], d["frontier_perm"]

        def add(kind, pts, **extra):
            for order, (x, y) in enumerate(np.atleast_2d(pts)):
                rows.append({"panel": idx, "subgroup": name, "kind": kind,
                             "order": order, "x": x, "y": y,
                             "label": extra.get("label", ""),
                             "style": extra.get("style", ""),
                             "dx": extra.get("dx", np.nan),
                             "dy": extra.get("dy", np.nan)})

        cloud = d["cloud"]
        step = max(1, len(cloud) // _MAX_CLOUD)
        add("cloud", cloud[::step])
        add("frontier_all", fa["path"])
        add("frontier_perm", fp["path"])

        ## The two frontiers bound the cost-of-constraints band, so the band is
        ## not stored: the draw script closes the polygon between them.
        ##
        ## Named points on the unconstrained frontier. When one algorithm is
        ## optimal for BOTH groups the two coincide and take a joint label, as
        ## in their Figure 10.
        if np.linalg.norm(fa["p1"] - fa["p0"]) < 1e-9:
            named = [(fa["p1"], f"{spec['sym1']} = {spec['sym0']}")]
        else:
            named = [(fa["p1"], spec["sym1"]), (fa["p0"], spec["sym0"])]
        if fa["skewed"]:
            named.append((fa["f"], r"$f_X$" if fa["f_exact"] else r"$\tilde{f}_X$"))

        ## Label placement is collision-avoidance geometry over points the
        ## figure already draws, so it is resolved once, here, rather than
        ## reimplemented in the draw script.
        offs = label_offsets(named, hi - lo, away_from=fa["hull"].mean(axis=0),
                             lim=(lo, hi),
                             avoid=[np.asarray(e, float) for e in d["ops"].values()])
        for (pt, sym), (dx, dy) in zip(named, offs):
            add("named", pt, label=sym, dx=dx, dy=dy)

        for (label, e), style in zip(d["ops"].items(), _OP_STYLE):
            if np.isfinite(e).all():
                add("operating_point", np.atleast_2d(e), label=label, style=style)
            else:
                print(f"  ! {label} has no finite error pair in the {name} panel")

        ## A point outside the frame is reported, never silently clipped.
        for label, e in d["ops"].items():
            if np.isfinite(e).all() and (min(e) < lo or max(e) > hi):
                print(f"  ! {label} is outside the plotted range in the {name} panel "
                      f"({e[0]:.1%}, {e[1]:.1%}) — not drawn")

        specs.append({"panel": idx, "subgroup": name, "letter": "abcdefgh"[idx],
                      "xlabel": f"{spec['lbl1']} error rate (%)",
                      "ylabel": f"{spec['lbl0']} error rate (%)"})

    p_model, p_fs = next(iter(panels.values()))["participatory"]
    meta = {
        ## One row, one square panel per fairness axis, legend above.
        "figsize": canvas(4.3 / 11.0),
        ## Unrounded, unlike the data. An axis limit is layout, not a
        ## measurement: it reveals nothing a drawn point does not, and rounding
        ## it shifts every one of the 4,000-odd cloud markers by a fraction of a
        ## pixel, which is visible on marks this small.
        "lim": [lo, hi],
        "panels": specs,
        "legend": {
            "participatory": f"Participatory ({_MODEL_LABELS.get(p_model, p_model)}, "
                             f"{_FS_LABELS.get(p_fs, p_fs)})",
            "max_ml": f"Max ML ({_MODEL_LABELS.get(MAX_ML_MODEL, MAX_ML_MODEL)}, "
                      f"{_FS_LABELS.get(MAX_ML_FS, MAX_ML_FS)})",
            "credits": "Credits policy (≥60 ECTS)",
        },
        "text_columns": ["subgroup", "kind", "label", "style"],
    }
    return pd.DataFrame(rows), meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year", type=int, default=MAIN_TEST_YEAR)
    ap.add_argument("--n-boot", type=int, default=1000,
                    help="bootstrap draws for the group-skew test (0 to skip)")
    ap.add_argument("--full-range", action="store_true",
                    help="show the whole candidate cloud instead of framing the frontier")
    ap.add_argument("--zoom", action="store_true",
                    help="frame on the frontiers alone, excluding the operating points")
    args = ap.parse_args()

    ep = pd.read_csv(RESULTS_PATH / "enrollment_predictions.csv")
    results_df = pd.read_csv(RESULTS_PATH / "evaluation_results.csv")
    panels = {name: build_subgroup_data(ep, load_analysis_df(), results_df,
                                        args.year, name, n_boot=args.n_boot)
              for name in _SUBGROUPS}

    df, meta = prepare(panels, args.year, args.full_range, args.zoom)
    write_input(f"fa_frontier_{args.year}", df, meta,
                source="results/enrollment_predictions.csv + evaluation_results.csv",
                repo=ROOT, script=SCRIPT)

    for name, d in panels.items():
        fa = d["frontier_all"]
        if fa["skewed"]:
            favoured = (d["spec"]["lbl1"] if fa["skew_side"] == "group1"
                        else d["spec"]["lbl0"])
            print(f"  {name:12s} group-skewed towards {favoured} "
                  f"(bootstrap p = {d['p_skew']:.3f}) — the FA frontier extends "
                  f"beyond the group-optimal points")
        else:
            print(f"  {name:12s} group-balanced — equal error rates are reachable "
                  f"on the Pareto boundary, no fairness-accuracy tradeoff")


if __name__ == "__main__":
    main()
