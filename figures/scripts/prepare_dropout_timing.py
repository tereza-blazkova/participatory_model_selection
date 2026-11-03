"""
Distil the dropout-timing figure: when dropouts leave, and how much of that the
two decision rules had already caught.

    figures/input/dropout_timing.csv    one row per step vertex of each curve
    figures/input/dropout_timing.json   the layers, the panels, the axes

src/plotting.py's dropout_timing_layers() owns everything that reads a student
record — the half-year binning, the merging of a bin holding too few dropouts,
and the rule that drops a whole layer whose recall would rest on too few. This
script calls it and writes the curves out.

A CDF with one step per dropout would state each student's dropout month, which
is why the curves rise once per half-year and why a thin half-year is merged
forward. What lands here has already passed through that.

The cohorts are the test cohorts without 2023 (config.DROPOUT_TIMING_YEARS),
pooled; they are recorded in the JSON (`start_years`) for the caption. Every
dropout drawn must have a Participatory prediction, or the run stops.

`--max-ml` writes the dropout_timing_maxml variant instead: the same panels
with a Max ML layer (the NN on (e) + Sensitive) beside the credits policy and
the Participatory model.

    python figures/scripts/prepare_dropout_timing.py [--max-ml] [--start-years 2019 2020 ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from paths.paths import RESULTS_PATH                                   # noqa: E402
from prepare_utils import write_input                                  # noqa: E402
from src.data import load_analysis_df                                  # noqa: E402
from src.plotting import dropout_timing_layers                         # noqa: E402
from utils import canvas                                               # noqa: E402


SCRIPT = "prepare_dropout_timing.py"
_LEGEND_ROW_IN = 0.17      # one 9 pt legend row at labelspacing 0.3, in inches


def prepare(raw_df, preds_df, start_years=None,
            include_max_ml: bool = False) -> tuple[pd.DataFrame, dict]:
    content = dropout_timing_layers(raw_df, preds_df, start_years=start_years,
                                    include_max_ml=include_max_ml)

    rows, specs = [], []
    for i, layer in enumerate(content["layers"]):
        for x, y in zip(layer["x"], layer["y"]):
            rows.append({"layer": i, "panel": layer["panel"], "x": x, "y": y})
        specs.append({
            "layer": i, "panel": layer["panel"],
            ## What it is, not what colour it is: the draw script maps these.
            "colour": layer["colour"], "style": layer["style"],
            "label": layer["label"], "hline": layer["hline"],
            "drawn": layer["drawn"], "fill_alpha": layer["fill_alpha"],
            "zorder": layer["zorder"], "fill_zorder": layer["fill_zorder"],
        })

    max_months = content["max_months"]

    ## The canvas grows with the legend, not the axes: each legend row past
    ## four (a Max ML layer adds one per group) adds its height to the top
    ## margin and lifts the panel letters with it, so the panels print at the
    ## same size in both variants.
    width, base_h = canvas(4.6 / 13)
    n_rows = max(sum(1 for l in specs if l["panel"] == p and l["label"])
                 for p in {l["panel"] for l in specs})
    extra_in = max(0, n_rows - 4) * _LEGEND_ROW_IN
    height = base_h + extra_in
    bottom_in, axes_in = 0.14 * base_h, 0.48 * base_h
    panels = [{**p, "letter_y": round(p["letter_y"] + extra_in / axes_in, 3)}
              for p in content["panels"]]

    meta = {
        "figsize": [width, round(height, 4)],
        "xlabel": "Months of enrollment",
        "ylabel": "Cumulative % of dropout enrollments",
        "xlim": [0, max_months],
        "ylim": [0, 109],
        ## The usual gap between panels; the top margin holds the legends,
        ## which sit above each panel and are wider than it.
        "subplots_adjust": {"left": 0.06, "right": 0.985,
                            "bottom": round(bottom_in / height, 4),
                            "top": round((bottom_in + axes_in) / height, 4),
                            "wspace": 0.12},
        "xticks": list(range(0, int(max_months) + 1, 12)),
        "start_years": [int(y) for y in content["start_years"]],
        "panels": panels,
        "layers": specs,
        "legend": {"loc": "lower left", "bbox_to_anchor": [-0.02, 1.02],
                   "handlelength": 1.4, "labelspacing": 0.3},
        "text_columns": [],
    }
    return pd.DataFrame(rows), meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start-years", type=int, nargs="*", default=None)
    ap.add_argument("--max-ml", action="store_true",
                    help="add the Max ML layer; writes dropout_timing_maxml")
    args = ap.parse_args()

    preds = pd.read_csv(RESULTS_PATH / "enrollment_predictions.csv")
    df, meta = prepare(load_analysis_df(), preds, args.start_years, args.max_ml)
    write_input("dropout_timing_maxml" if args.max_ml else "dropout_timing", df, meta,
                source="data + results/enrollment_predictions.csv",
                repo=ROOT, script=SCRIPT)


if __name__ == "__main__":
    main()
