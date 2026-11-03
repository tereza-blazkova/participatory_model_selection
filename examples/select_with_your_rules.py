"""
The selection test on its own: a table of recalls and a set of survey rules go
in, a selected model comes out. It needs no student data and no trained model.

Every recall below is made up for this example. None comes from the paper.

Usage:  python -m examples.select_with_your_rules
"""
import pandas as pd

from src.workshop_rules import (
    LARGE_IMPROVEMENT, MODERATE_IMPROVEMENT, NEVER, NO_THRESHOLD,
    SMALL_IMPROVEMENT, SurveyRuleSet, select_grid_config,
)

YEAR = 2021

## Recall of each model on each feature set, measured at the same contact budget.
## With your own data, this is what evaluating your candidate models produces.
RECALLS = {
    "a_credits":             {"fasterrisk": 0.50, "logistic_regression": 0.52, "xgboost": 0.53, "mlp": 0.52},
    "b_credits_performance": {"fasterrisk": 0.55, "logistic_regression": 0.60, "xgboost": 0.63, "mlp": 0.61},
    "c_plus_enrolment":      {"fasterrisk": 0.56, "logistic_regression": 0.64, "xgboost": 0.71, "mlp": 0.66},
    "d_plus_lms":            {"fasterrisk": 0.57, "logistic_regression": 0.66, "xgboost": 0.73, "mlp": 0.69},
    "e_plus_sensitive":      {"fasterrisk": 0.57, "logistic_regression": 0.67, "xgboost": 0.74, "mlp": 0.70},
}
CREDITS_POLICY_RECALL = 0.40


def results_table() -> pd.DataFrame:
    rows = [{"model": model, "feature_set": fs, "test_year": YEAR,
             "enr_recall": recall, "enr_precision": float("nan")}
            for fs, by_model in RECALLS.items() for model, recall in by_model.items()]
    ## The rule-based reference the explainability walk starts from.
    rows.append({"model": "credits_policy", "feature_set": "ects_behind_ge_60",
                 "test_year": YEAR, "enr_recall": CREDITS_POLICY_RECALL,
                 "enr_precision": float("nan")})
    return pd.DataFrame(rows)


## Two groups answering the same survey differently. Each number is the median
## answer: the recall gain a step must bring before the group accepts it.
CAUTIOUS = SurveyRuleSet(
    name="cautious",
    label="no LMS data, no neural network, large gains for enrollment data",
    fs_gain_required={
        "b_credits_performance": SMALL_IMPROVEMENT,
        "c_plus_enrolment":      LARGE_IMPROVEMENT,
        "d_plus_lms":            NEVER,
        "e_plus_sensitive":      NEVER,
    },
    expl_gain_required={
        "high":     MODERATE_IMPROVEMENT,
        "moderate": SMALL_IMPROVEMENT,
        "low":      MODERATE_IMPROVEMENT,
        "very_low": NEVER,
    },
)

PERMISSIVE = SurveyRuleSet(
    name="permissive",
    label="LMS data and a less explainable model for modest gains",
    fs_gain_required={
        "b_credits_performance": NO_THRESHOLD,
        "c_plus_enrolment":      SMALL_IMPROVEMENT,
        "d_plus_lms":            SMALL_IMPROVEMENT,
        "e_plus_sensitive":      NEVER,
    },
    expl_gain_required={
        "high":     SMALL_IMPROVEMENT,
        "moderate": SMALL_IMPROVEMENT,
        "low":      SMALL_IMPROVEMENT,
        "very_low": MODERATE_IMPROVEMENT,
    },
)


def main() -> None:
    results = results_table()
    for rules in (CAUTIOUS, PERMISSIVE):
        model, feature_set, candidates = select_grid_config(results, test_year=YEAR, rules=rules)
        print(f"\n{rules.name}: {rules.label}")
        print(f"  selected: {model} on {feature_set}")
        print(candidates[["model", "feature_set", "recall", "data_permissible",
                          "expl_permissible", "permissible"]].to_string(index=False))


if __name__ == "__main__":
    main()
