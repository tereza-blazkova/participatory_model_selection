"""
Distil the FA-frontier ladders: the same frontier, decomposed once by how much
data the model sees and once by how flexible the model is.

    figures/input/fa_frontier_ladders_{year}.csv    every drawn path and marker
    figures/input/fa_frontier_ladders_{year}.json   axes, series, legends

The two rows are not the same exercise, and the figure exists to keep them
apart: the data ladder changes the information structure X, while the model
ladder shrinks the set of permissible algorithms. `e_plus_sensitive` is drawn
twice — solid without group-aware rules, dashed with them — to separate the
effect of more covariates from their Prop. 2 (group-specific thresholds).

src/fa_frontier_ladders.py owns the computation and writes the summary CSV;
this script writes what the figure draws.

    python figures/scripts/prepare_fa_frontier_ladders.py [--year 2021]
                                                          [--model-ladder-fs ...]
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
from src.config import MAIN_TEST_YEAR                                  # noqa: E402
from utils import canvas                                               # noqa: E402
from src.fa_frontier_ladders import (                                  # noqa: E402
    _DATA_LADDER_MODEL, _DEFAULT_MODEL_LADDER_FS, _FS_LABELS, _MODEL_LABELS,
    _axis_limits, _model_colors, build_ladders,
)

SCRIPT = "prepare_fa_frontier_ladders.py"


def prepare(ladders: dict, model_ladder_fs: str):
    """
    Layout: one ROW per ladder (data on top, model below), one COLUMN per
    fairness axis (sex, nationality, entry age) — the same three columns, in the
    same order, as the FA frontier figure, so a column there can be read straight
    down into its decomposition here. Each row carries its own legend.
    """
    lo, hi = _axis_limits(ladders)
    models, _, _ = _model_colors()
    n_attr = len(ladders)

    rows, panels, series_specs = [], [], []
    for col, (name, d) in enumerate(ladders.items()):
        spec = d["spec"]
        for row, key in enumerate(("data", "model")):
            panels.append({
                "row": row, "col": col, "letter": "abcdefgh"[row * n_attr + col],
                "subgroup": name,
                "xlabel": f"{spec['lbl1']} error rate (%)",
                "ylabel": f"{spec['lbl0']} error rate (%)",
            })
            for i, s in enumerate(d[key]):
                fr = s["frontier"]
                for kind, pts in (("path", fr["path"]),
                                  ## The two group-optimal endpoints carry the
                                  ## marker, so a series is identifiable without
                                  ## relying on colour alone.
                                  ("endpoint", np.vstack([fr["p1"], fr["p0"]])),
                                  ## f_X, hollow, only where the frontier is skewed.
                                  ("fair", np.atleast_2d(fr["f"]) if fr["skewed"] else None)):
                    if pts is None:
                        continue
                    for order, (x, y) in enumerate(np.atleast_2d(pts)):
                        rows.append({"row": row, "col": col, "series": i,
                                     "kind": kind, "order": order, "x": x, "y": y})
                if col == 0:
                    series_specs.append({
                        "row": row, "series": i, "label": s["label"],
                        ## Dashed marks the variant that may use group identity
                        ## in the decision rule itself — their Prop. 2, not an
                        ## effect of more covariates; the figure must not blur them.
                        "group_aware": bool(s["group_aware"]),
                        ## The model ladder reuses the repo-wide model colours so
                        ## it matches every other model comparison; the data
                        ## ladder is an ORDERED ladder and gets a sequential ramp.
                        "colour": models[i] if row == 1 else None,
                    })

    n_data = max(len(d["data"]) for d in ladders.values())
    meta = {
        ## Two rows of square panels plus a legend column on the right.
        "figsize": canvas(6.4 / 11.0),
        "lim": [lo, hi],
        "panels": panels,
        "series": series_specs,
        ## Stop short of viridis's pale yellow end — too low-contrast on white.
        ## The last rung reuses the previous colour: it is the same feature set,
        ## drawn a second time with group-aware rules.
        "data_ramp": {"cmap": "viridis", "start": 0.05, "stop": 0.72,
                      "n": n_data - 1, "repeat_last": True},
        "data_markers": ["o", "s", "^", "v", "D", "D"][:n_data],
        "legend_titles": [
            f"Data ladder\n{_MODEL_LABELS[_DATA_LADDER_MODEL].lower()}, by feature set",
            f"Model ladder\n{_FS_LABELS.get(model_ladder_fs, model_ladder_fs)}, by model",
        ],
        "text_columns": ["kind"],
    }
    return pd.DataFrame(rows), meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year", type=int, default=MAIN_TEST_YEAR)
    ap.add_argument("--model-ladder-fs", default=_DEFAULT_MODEL_LADDER_FS)
    args = ap.parse_args()

    ep = pd.read_csv(RESULTS_PATH / "enrollment_predictions.csv")
    ladders = build_ladders(ep, args.year, args.model_ladder_fs)
    df, meta = prepare(ladders, args.model_ladder_fs)
    write_input(f"fa_frontier_ladders_{args.year}", df, meta,
                source="results/enrollment_predictions.csv", repo=ROOT, script=SCRIPT)


if __name__ == "__main__":
    main()
