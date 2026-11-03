"""
The paper's selections, reproduced from the recalls the paper prints. It needs no
student data and no trained model: the recalls below are copied from the selection
tables, and the selection code is the same code the pipeline runs.

It checks the 2021 and 2019 selections under the endline rules, the same cohorts
under the baseline (pre-deliberation) rules, and every threshold variant in the
robustness table. It exits with an error if any selection differs from the paper.

The recalls are printed to three decimals, so a gain computed here can differ from
the paper's by a rounding step. It cannot change a selection: no gain sits that
close to its required gain.

Usage:  python -m examples.reproduce_paper_selection
"""
import sys
from dataclasses import replace

import pandas as pd

from src.workshop_rules import (
    BASELINE_RULES, ENDLINE_RULES, LARGE_IMPROVEMENT, MODERATE_IMPROVEMENT, NEVER,
    NO_THRESHOLD, SMALL_IMPROVEMENT, select_grid_config,
)

MODELS = ["fasterrisk", "logistic_regression", "xgboost", "mlp"]

## Recall per feature set, models in the order above: the selection tables.
RECALLS_2021 = {
    "a_credits":             [0.425, 0.442, 0.412, 0.421],
    "b_credits_performance": [0.546, 0.623, 0.666, 0.655],
    "c_plus_enrolment":      [0.535, 0.684, 0.735, 0.721],
    "d_plus_lms":            [0.471, 0.684, 0.732, 0.720],
    "e_plus_sensitive":      [0.474, 0.687, 0.735, 0.719],
}
CREDITS_POLICY_2021 = 0.235

RECALLS_2019 = {
    "a_credits":             [0.412, 0.428, 0.430, 0.435],
    "b_credits_performance": [0.405, 0.568, 0.623, 0.604],
    "c_plus_enrolment":      [0.532, 0.644, 0.697, 0.685],
    "d_plus_lms":            [0.455, 0.647, 0.683, 0.680],
    "e_plus_sensitive":      [0.477, 0.647, 0.689, 0.680],
}
CREDITS_POLICY_2019 = 0.307

LR_B  = ("logistic_regression", "b_credits_performance")
XGB_B = ("xgboost", "b_credits_performance")
XGB_C = ("xgboost", "c_plus_enrolment")

## The robustness table, row by row: which threshold moves, its value in the
## paper, and what the selection becomes. Variants are (label, new value, expected).
FS, EXPL = "fs_gain_required", "expl_gain_required"
ROBUSTNESS = [
    ("Adding performance data", FS, "b_credits_performance", SMALL_IMPROVEMENT, [
        ("0 pp, one level looser", NO_THRESHOLD, LR_B),
        ("5 pp, one level stricter", MODERATE_IMPROVEMENT, LR_B)]),
    ("Adding enrollment data", FS, "c_plus_enrolment", LARGE_IMPROVEMENT, [
        ("5 pp, one level looser", MODERATE_IMPROVEMENT, XGB_C),
        ("never, one level stricter", NEVER, LR_B)]),
    ("Adding LMS data", FS, "d_plus_lms", NEVER, [
        ("10 pp, one level looser", LARGE_IMPROVEMENT, LR_B)]),
    ("FasterRisk over the credits policy", EXPL, "high", MODERATE_IMPROVEMENT, [
        ("1 pp, one level looser", SMALL_IMPROVEMENT, LR_B),
        ("10 pp, one level stricter", LARGE_IMPROVEMENT, LR_B)]),
    ("Logistic regression over FasterRisk", EXPL, "moderate", SMALL_IMPROVEMENT, [
        ("0 pp, one level looser", NO_THRESHOLD, LR_B),
        ("5 pp, one level stricter", MODERATE_IMPROVEMENT, LR_B)]),
    ("XGBoost over logistic regression", EXPL, "low", MODERATE_IMPROVEMENT, [
        ("1 pp, one level looser", SMALL_IMPROVEMENT, XGB_B),
        ("10 pp, one level stricter", LARGE_IMPROVEMENT, LR_B)]),
    ("Neural network over XGBoost", EXPL, "very_low", NEVER, [
        ("10 pp, one level looser", LARGE_IMPROVEMENT, LR_B)]),
]
## The numeric moves the table prints as -5, -2, +2 and +5 pp. Rows whose paper
## value is "never" have none. Anything not listed here keeps the main selection.
SHIFT_EXCEPTIONS = {
    ("c_plus_enrolment", -0.05): XGB_C,
    ("low", -0.05): XGB_B,
    ("low", -0.02): XGB_B,
}


def results_table(recalls: dict, credits_policy: float, year: int) -> pd.DataFrame:
    rows = [{"model": model, "feature_set": fs, "test_year": year,
             "enr_recall": recall, "enr_precision": float("nan")}
            for fs, values in recalls.items() for model, recall in zip(MODELS, values)]
    rows.append({"model": "credits_policy", "feature_set": "ects_behind_ge_60",
                 "test_year": year, "enr_recall": credits_policy,
                 "enr_precision": float("nan")})
    return pd.DataFrame(rows)


def selected(results: pd.DataFrame, year: int, rules) -> tuple:
    model, feature_set, _ = select_grid_config(results, test_year=year, rules=rules)
    return model, feature_set


def moved(field: str, key: str, value: float):
    """The endline rules with one threshold moved."""
    gains = dict(getattr(ENDLINE_RULES, field))
    gains[key] = value
    return replace(ENDLINE_RULES, name="variant", **{field: gains})


def main() -> int:
    r2021 = results_table(RECALLS_2021, CREDITS_POLICY_2021, 2021)
    r2019 = results_table(RECALLS_2019, CREDITS_POLICY_2019, 2019)

    checks = [
        ("2021, endline rules", selected(r2021, 2021, ENDLINE_RULES), LR_B),
        ("2019, endline rules", selected(r2019, 2019, ENDLINE_RULES), XGB_B),
        ("2021, baseline rules", selected(r2021, 2021, BASELINE_RULES), LR_B),
        ("2019, baseline rules", selected(r2019, 2019, BASELINE_RULES), LR_B),
    ]
    for row, field, key, value, variants in ROBUSTNESS:
        for label, new, expected in variants:
            checks.append((f"{row}: {label}",
                           selected(r2021, 2021, moved(field, key, new)), expected))
        if value != NEVER:
            for shift in (-0.05, -0.02, 0.02, 0.05):
                expected = SHIFT_EXCEPTIONS.get((key, shift), LR_B)
                checks.append((f"{row}: {shift * 100:+.0f} pp",
                               selected(r2021, 2021, moved(field, key, value + shift)),
                               expected))

    failures = 0
    for label, got, expected in checks:
        ok = got == expected
        failures += not ok
        print(f"  {'ok ' if ok else 'DIFFERENT'}  {label:62s} {got[0]} / {got[1]}"
              + ("" if ok else f"  (paper: {expected[0]} / {expected[1]})"))

    ## The margins behind the two closest calls in the 2021 selection.
    b, c = RECALLS_2021["b_credits_performance"], RECALLS_2021["c_plus_enrolment"]
    print(f"\n  XGBoost (b) over logistic regression (b): "
          f"{(b[2] - b[1]) * 100:+.1f} pp, required {ENDLINE_RULES.expl_gain_required['low'] * 100:.0f} pp")
    print(f"  XGBoost (c) over logistic regression (b): {(c[2] - b[1]) * 100:+.1f} pp")

    print(f"\n{len(checks) - failures} of {len(checks)} selections match the paper.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
