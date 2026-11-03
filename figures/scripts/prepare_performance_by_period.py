"""
Distil the performance-by-period figure — how the Participatory model performs as
a cohort is observed for longer, overall and as a difference between groups.

    figures/input/performance_by_period.csv   one row per panel x metric x period
    figures/input/performance_by_period.json  panels, metrics, axes, note

src/plotting.py's performance_by_period_series() owns everything that reads a
prediction: the per-period threshold, the five metrics, the 500-draw bootstrap,
and the rule that drops a point whose confusion cells rest on too few students —
a hidden point takes its confidence band with it, so the band can never outline
where the point would have been.

Because the bootstrap runs here, restyling the figure no longer re-runs it.

    python figures/scripts/prepare_performance_by_period.py [--n-boot 500] [--max-activation 6]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from paths.paths import RESULTS_PATH                                   # noqa: E402
from prepare_utils import write_input                                  # noqa: E402
from src import disclosure                                             # noqa: E402
from src.config import MAIN_TEST_YEAR                                  # noqa: E402
from src.plotting import _workshop_underlying_name, performance_by_period_series  # noqa: E402
from src.workshop_rules import _RULE_BASED_MODELS  # noqa: E402
from utils import canvas                                               # noqa: E402


SCRIPT = "prepare_performance_by_period.py"

## The last EWS activation drawn (the sixth, about three years in).
MAX_ACTIVATION = 6


def prepare(row_preds, train_pos_rate, model, feature_set, test_year, n_boot,
            max_activation=MAX_ACTIVATION):
    content = performance_by_period_series(
        row_preds, train_pos_rate, model=model, feature_set=feature_set,
        test_year=test_year, n_boot=n_boot, max_activation=max_activation)

    rows, panels = [], []
    for i, panel in enumerate(content["panels"]):
        if panel is None:
            ## A subgroup column the predictions do not carry: the panel is
            ## reserved but drawn empty, rather than the others sliding over.
            panels.append({"index": i, "letter": chr(ord("a") + i), "visible": False})
            continue
        panels.append({
            "index": i, "letter": panel["letter"], "visible": True,
            "kind": panel["kind"], "title": panel["title"],
            ## Panel a reads as levels, b-d as differences from zero.
            "ylabel": "Metric Score" if panel["kind"] == "overall" else "Metric Difference",
            "zero_line": panel["kind"] == "difference",
            "note": (panel["title"] or "a") in content["hidden_panels"],
        })
        for metric, series in panel["series"].items():
            for act, y, lo, hi in zip(content["activations"], series["y"],
                                      series["lo"], series["hi"]):
                rows.append({"panel": i, "metric": metric, "activation": act,
                             "y": y, "lo": lo, "hi": hi})

    meta = {
        "figsize": canvas(3.25 / 13),
        ## Panel a carries five series across every period, so it gets half again
        ## the width of the difference panels.
        "width_ratios": [1, 1, 1, 1],   # all four panels the same width
        "metrics": content["metrics"],
        "metric_labels": content["metric_labels"],
        "xlabel": "EWS activation",
        "xticks": content["activations"],
        "panels": panels,
        "note": disclosure.note_text() if content["hidden_panels"] else None,
        "text_columns": ["metric"],
    }
    return pd.DataFrame(rows), meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-year", type=int, default=MAIN_TEST_YEAR)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--max-activation", type=int, default=MAX_ACTIVATION,
                    help=f"last EWS activation drawn (default {MAX_ACTIVATION})")
    args = ap.parse_args()

    row_preds = pd.read_csv(RESULTS_PATH / "row_predictions.csv")
    train_pos_rate = json.loads(
        (RESULTS_PATH / "models" / "train_metadata.json").read_text())["train_pos_rate"]

    ## The workshop config stores a DISPLAY label ("workshop_model"), which never
    ## appears in row_predictions.csv — that file holds estimator names. Resolve
    ## the underlying one, as every workshop-aware figure does.
    model, feature_set = _workshop_underlying_name()
    if not (model and feature_set):
        raise SystemExit("No workshop selection found — run src/workshop_evaluation.py first.")
    if model in _RULE_BASED_MODELS:
        raise SystemExit("REFUSING to write: no combination is permissible, so the Credits "
                         "policy is retained, and row_predictions.csv holds trained models "
                         "only: there is no Participatory series to draw.")

    df, meta = prepare(row_preds, train_pos_rate, model, feature_set,
                       args.test_year, args.n_boot, max_activation=args.max_activation)
    write_input("performance_by_period", df, meta,
                source="results/row_predictions.csv", repo=ROOT, script=SCRIPT)


if __name__ == "__main__":
    main()
