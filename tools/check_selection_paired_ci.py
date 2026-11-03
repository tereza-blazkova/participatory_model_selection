"""
Paired bootstrap CI for the recall gain that decides the model class.

The selection compares point estimates, and XGBoost on (b) misses its required
gain over logistic regression by under a point. Both models are resampled on the
same enrollments, so the interval is on the difference itself. The result is
typed into the paper's main text and SI, so nothing regenerates it: re-run this
and update the prose after any re-tune.

Reports the threshold held fixed and re-estimated in each resample. The paper
quotes the second, which is closer to how the selection works.

Check the printed recalls against the paper before using the interval.

Usage:  python -m tools.check_selection_paired_ci
"""
import json

import numpy as np
import pandas as pd

from paths.paths import RESULTS_PATH
from src.config import MAIN_TEST_YEAR, RANDOM_SEED
from src.metrics import conservative_threshold

FEATURE_SET = "b_credits_performance"
N_BOOT = 1000
REQUIRED_GAIN = 0.05


def _recall(flagged: np.ndarray, y: np.ndarray) -> float:
    return (flagged & y).sum() / y.sum()


def main() -> None:
    p = pd.read_csv(RESULTS_PATH / "enrollment_predictions.csv")
    p = p[(p["test_year"] == MAIN_TEST_YEAR) & (p["feature_set"] == FEATURE_SET)]
    meta = json.loads((RESULTS_PATH / "models" / "train_metadata.json").read_text())
    target = meta["train_pos_rate"]

    prob = p.pivot(index="enrollment", columns="model", values="max_prob")
    flag = p.pivot(index="enrollment", columns="model", values="ever_flagged").astype(bool)
    y = (p.drop_duplicates("enrollment").set_index("enrollment")["dropped_out"]
         .reindex(prob.index).astype(bool).values)
    x_prob, lr_prob = prob["xgboost"].values, prob["logistic_regression"].values
    x_flag, lr_flag = flag["xgboost"].values, flag["logistic_regression"].values

    print("Enrollments:", len(y), "| dropouts:", int(y.sum()))
    print(f"Recall XGBoost {_recall(x_flag, y):.3f} | LR {_recall(lr_flag, y):.3f} "
          f"| difference {100 * (_recall(x_flag, y) - _recall(lr_flag, y)):.1f} pp")

    rng = np.random.default_rng(RANDOM_SEED)
    fixed, reestimated = [], []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        yy = y[i]
        fixed.append(_recall(x_flag[i], yy) - _recall(lr_flag[i], yy))
        a, b = x_prob[i], lr_prob[i]
        reestimated.append(_recall(a >= conservative_threshold(a, target), yy)
                           - _recall(b >= conservative_threshold(b, target), yy))

    for name, draws in [("threshold held fixed", fixed), ("threshold re-estimated", reestimated)]:
        d = np.array(draws)
        lo, hi = np.percentile(d, [2.5, 97.5]) * 100
        print(f"{name}: 95% CI [{lo:.1f}, {hi:.1f}] pp | share of resamples reaching "
              f"{REQUIRED_GAIN * 100:.0f} pp: {(d >= REQUIRED_GAIN).mean():.3f}")


if __name__ == "__main__":
    main()
