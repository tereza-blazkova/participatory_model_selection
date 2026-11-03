"""
Workshop-grounded model: train, evaluate, and compare against all other models.

Reads evaluation_results.csv and enrollment_predictions.csv produced by
python src/evaluation.py, runs the grid selection (the algorithm the paper
describes — there is no other), evaluates the selected model, and saves:

    results/models/workshop_config.json   the Participatory model; read ONLY via
                                          workshop_rules.participatory_config()
    results/workshop_results.csv          its metrics per test year
    results/workshop_candidates.csv       every (model, feature set) and its verdicts

Rule sets other than endline write the same names with `_{ruleset}` appended.

Run from project root:
    python src/workshop_evaluation.py                   # endline rules
    python src/workshop_evaluation.py --ruleset both    # + baseline, and the comparison
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import joblib

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from paths.paths import DATA_PATH_PARQUET, RESULTS_PATH
from src.config import (DATA_CONFIGS, MAIN_TEST_YEAR, SELECTION_YEAR, TARGET,
                        TEST_YEARS, TRAIN_YEARS)
from src.metrics import credits_policy_metrics, enrollment_level_metrics
from src.workshop_rules import (
    _RULE_BASED_MODELS,
    CANDIDATES_FILE,
    CONFIG_FILE,
    DEFAULT_RULES,
    DELIBERATION_FILE,
    EXPLAINABILITY_LEVEL,
    RESULTS_FILE,
    RULE_SETS,
    SELECTION_ALGORITHM,
    WorkshopConfig,
    get_rules,
    ruleset_file,
    run_workshop_selection,
)
from src.data import load_analysis_df

_MODELS_DIR = RESULTS_PATH / "models"


def _load_model_and_metadata(model_name: str, feature_set: str):
    """Load saved model pkl and train metadata. Raises if not found."""
    model_path = _MODELS_DIR / f"{model_name}__{feature_set}.joblib"
    meta_path  = _MODELS_DIR / "train_metadata.json"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Saved model not found: {model_path}\n"
            "Run  python src/evaluation.py  first."
        )
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Train metadata not found: {meta_path}\n"
            "Run  python src/evaluation.py  first."
        )
    model = joblib.load(model_path)
    with open(meta_path) as fh:
        meta = json.load(fh)
    return model, meta


def _evaluate_retained_credits_policy(
    df: pd.DataFrame,
    feature_set: str,
    label: str,
) -> pd.DataFrame:
    """No combination was permissible: the policy has no saved model, only its rule."""
    rows = []
    for yr in TEST_YEARS:
        test_df = df[df["start_year"] == yr].copy()
        if len(test_df) == 0:
            continue
        rows.append({
            "model":       label,
            "feature_set": feature_set,
            "test_year":   yr,
            "is_2023":     yr == 2023,
            **credits_policy_metrics(test_df, target=TARGET),
        })
    return pd.DataFrame(rows)


def evaluate_workshop_model(
    df: pd.DataFrame,
    model_name: str,
    feature_set: str,
    label: str = "workshop_model",
) -> pd.DataFrame:
    """Load saved model, evaluate on each TEST_YEAR. Returns results rows."""
    if model_name in _RULE_BASED_MODELS:
        return _evaluate_retained_credits_policy(df, feature_set, label)

    model, meta    = _load_model_and_metadata(model_name, feature_set)
    train_pos_rate = meta["train_pos_rate"]
    available      = meta["available_features"][feature_set]

    print(f"  Loaded: {model_name} + {feature_set}")

    rows = []
    for yr in TEST_YEARS:
        test_df = df[df["start_year"] == yr].copy()
        if len(test_df) == 0:
            continue
        y_prob  = model.predict_proba(test_df[available])[:, 1]
        metrics = enrollment_level_metrics(test_df, y_prob, train_pos_rate, target=TARGET)
        rows.append({
            "model":       label,
            "feature_set": feature_set,
            "test_year":   yr,
            "is_2023":     yr == 2023,
            **metrics,
        })
    return pd.DataFrame(rows)


def _print_comparison(combined: pd.DataFrame, test_year: int, workshop_fs: str) -> None:
    """Print a side-by-side comparison table for a single test year."""
    yr = combined[combined["test_year"] == test_year].copy()

    # All models on the workshop feature set + credits_policy
    subset = yr[
        (yr["feature_set"] == workshop_fs) |
        (yr["model"] == "credits_policy") |
        (yr["model"].astype(str).str.startswith("workshop_model"))
    ].copy()

    cols = ["model", "feature_set", "enr_recall", "enr_precision", "early_flag_rate"]

    print(f"\n{'='*72}")
    print(f"Comparison — test year {test_year}  |  feature set: {workshop_fs}")
    print(f"{'='*72}")
    print(subset[cols].sort_values("enr_recall", ascending=False).to_string(index=False))


def run_for_ruleset(
    df: pd.DataFrame,
    results_df: pd.DataFrame,
    preds_df: Optional[pd.DataFrame],
    rules_name: str,
    selection_year: int = SELECTION_YEAR,
) -> tuple[WorkshopConfig, pd.DataFrame]:
    """
    Select the workshop model for one survey rule set on `selection_year`,
    evaluate it on every test year, and save. Only endline on SELECTION_YEAR
    writes the unsuffixed files that are the Participatory model.
    """
    rules  = get_rules(rules_name)
    config = run_workshop_selection(results_df, preds_df, df,
                                    test_year=selection_year, rules=rules)

    label = Path(ruleset_file("workshop_model.x", rules, selection_year)).stem

    print(f"Loading workshop model ({rules.name} rules) from saved joblib ...")
    workshop_results = evaluate_workshop_model(df, config.model, config.feature_set, label=label)
    workshop_results.insert(0, "ruleset", rules.name)

    RESULTS_PATH.mkdir(exist_ok=True)

    # Only the endline (post-deliberation) rules write the unsuffixed names, which
    # everything downstream reads as "the Participatory model".
    out_path = RESULTS_PATH / ruleset_file(RESULTS_FILE, rules, selection_year)
    workshop_results.to_csv(out_path, index=False)
    print(f"Workshop results saved → {out_path}")

    sens_model, sens_fs = config.sensitivity
    config_data = {
        "selection":   SELECTION_ALGORITHM,
        "ruleset":     rules.name,
        "test_year":   config.test_year,
        "model":       config.model,
        "feature_set": config.feature_set,
        # Not used by anything downstream: the pick with the data rule judged on
        # one reference model for every model, kept so the choice can be reported.
        "sensitivity_single_data_reference": {"model": sens_model, "feature_set": sens_fs},
    }
    config_path = RESULTS_PATH / "models" / ruleset_file(CONFIG_FILE, rules, selection_year)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as fh:
        json.dump(config_data, fh, indent=2)
    print(f"Workshop config saved  → {config_path}")

    if config.candidates_df is not None:
        cand_path = RESULTS_PATH / ruleset_file(CANDIDATES_FILE, rules, selection_year)
        config.candidates_df.to_csv(cand_path, index=False)
        print(f"Candidates table saved → {cand_path}")

    return config, workshop_results


def _deliberation_comparison(
    runs: list[tuple[WorkshopConfig, pd.DataFrame]],
    test_year: int,
) -> pd.DataFrame:
    """One row per rule set: selected configuration + headline metrics + fairness gaps."""
    metric_cols = [
        "n_enrollments", "n_dropout_enrollments", "n_flagged_enrollments",
        "enr_flag_rate", "tau", "enr_recall", "enr_precision", "enr_accuracy",
        "early_flag_rate", "roc_auc",
    ]
    rows = []
    for config, results in runs:
        rules = get_rules(config.rules)
        yr    = results[results["test_year"] == test_year]
        row = {
            "ruleset":        rules.name,
            "ruleset_label":  rules.label,
            "test_year":      test_year,
            "model":          config.model,
            "feature_set":    config.feature_set,
            "explainability": EXPLAINABILITY_LEVEL.get(config.model, "?"),
            "sensitivity_model":       config.sensitivity[0],
            "sensitivity_feature_set": config.sensitivity[1],
        }
        row.update({c: (yr.iloc[0][c] if len(yr) else float("nan")) for c in metric_cols})
        for col, gap in config.fairness_gaps.items():
            row[f"recall_gap_{col}"] = gap
        rows.append(row)
    return pd.DataFrame(rows)


def _print_deliberation_comparison(comparison: pd.DataFrame, test_year: int) -> None:
    """Print the before/after-deliberation table and the endline − baseline deltas."""
    print(f"\n{'='*72}")
    print(f"Deliberation effect — workshop model before vs after  (test year {test_year})")
    print(f"{'='*72}")

    show = ["ruleset", "model", "feature_set", "explainability",
            "enr_flag_rate", "enr_recall", "enr_precision", "roc_auc"]
    print(comparison[[c for c in show if c in comparison.columns]].to_string(index=False))

    idx = comparison.set_index("ruleset")
    if not {"baseline", DEFAULT_RULES.name}.issubset(idx.index):
        return

    before, after = idx.loc["baseline"], idx.loc[DEFAULT_RULES.name]
    print(f"\n  Pre-deliberation  : {before['model']} + {before['feature_set']}")
    print(f"  Post-deliberation : {after['model']} + {after['feature_set']}")
    print("\n  Change attributable to deliberation (post − pre):")
    delta_cols = [c for c in idx.columns
                  if c.startswith(("enr_", "roc_auc", "early_flag", "recall_gap_"))]
    for c in delta_cols:
        b, a = before.get(c), after.get(c)
        if pd.isna(b) or pd.isna(a):
            continue
        print(f"    {c:<28}  {b:>7.3f} → {a:>7.3f}   ({a - b:+.3f})")


def _print_stability(workshop_results: pd.DataFrame) -> None:
    """Print workshop model recall across all test years."""
    pivot = (
        workshop_results[["test_year", "enr_recall", "enr_precision"]]
        .set_index("test_year")
    )
    print(f"\n{'='*72}")
    print("Workshop model — stability across test years  (2023 = incomplete cohort)")
    print(f"{'='*72}")
    print(pivot.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ruleset",
        choices=sorted(RULE_SETS) + ["both"],
        default=DEFAULT_RULES.name,
        help=(
            "Which workshop survey administration supplies the selection rules. "
            "endline (default) = post-deliberation, the Participatory model reported "
            "in the paper; baseline = pre-deliberation medians; both = run endline and "
            "baseline and write the before/after comparison."
        ),
    )
    parser.add_argument(
        "--selection-year", type=int, default=SELECTION_YEAR, dest="selection_year",
        help=(
            f"Cohort the rules are applied to (default {SELECTION_YEAR}, the main "
            "text). Any other year writes `_sel{year}` files — the SI robustness "
            "version — and never the stored Participatory model."
        ),
    )
    args = parser.parse_args()

    ruleset_names = (["baseline", DEFAULT_RULES.name] if args.ruleset == "both"
                     else [args.ruleset])

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"Loading data from {DATA_PATH_PARQUET} ...")
    df = load_analysis_df()
    print(f"  {len(df):,} rows | {df['enrollment'].nunique():,} enrollments | "
          f"start_year {df['start_year'].min()}–{df['start_year'].max()}")

    results_path = RESULTS_PATH / "evaluation_results.csv"
    preds_path   = RESULTS_PATH / "enrollment_predictions.csv"

    if not results_path.exists():
        print(f"\nERROR: {results_path} not found.")
        print("Run  python src/evaluation.py  first.")
        sys.exit(1)

    results_df = pd.read_csv(results_path)
    preds_df   = pd.read_csv(preds_path) if preds_path.exists() else None
    if preds_df is None:
        print(f"WARNING: {preds_path} not found — fairness assessment will be skipped.")

    # ── Selection + evaluation, once per rule set ────────────────────────────
    runs: list[tuple[WorkshopConfig, pd.DataFrame]] = []
    for name in ruleset_names:
        config, workshop_results = run_for_ruleset(
            df, results_df, preds_df, name, selection_year=args.selection_year
        )
        runs.append((config, workshop_results))

        # ── Print comparison ──────────────────────────────────────────────────
        combined = pd.concat([results_df, workshop_results], ignore_index=True)
        _print_comparison(combined, MAIN_TEST_YEAR, config.feature_set)
        _print_stability(workshop_results)

        # ── Fairness table for workshop config ────────────────────────────────
        if config.fairness_df is not None and len(config.fairness_df) > 0:
            print(f"\n{'='*72}")
            print(f"Fairness — workshop model, {config.rules} rules "
                  f"({config.model} + {config.feature_set})")
            print(f"{'='*72}")
            for col, grp in config.fairness_df.groupby("subgroup_col"):
                gap = config.fairness_gaps.get(col, float("nan"))
                flag = "⚠ " if gap > 0.03 else "  "
                print(f"  {flag}{col}  (gap = {gap:.1%})")
                for _, row in grp.iterrows():
                    print(f"      {row['group_label']:<18}  recall = {row['recall']:.3f}  "
                          f"n_dropout = {row['n_dropout']:,}")

    # ── Before/after deliberation ────────────────────────────────────────────
    if len(runs) > 1 and args.selection_year == SELECTION_YEAR:
        comparison = _deliberation_comparison(runs, MAIN_TEST_YEAR)
        cmp_path = RESULTS_PATH / DELIBERATION_FILE
        comparison.to_csv(cmp_path, index=False)
        _print_deliberation_comparison(comparison, MAIN_TEST_YEAR)
        print(f"\nDeliberation comparison saved → {cmp_path}")

    print("\nDone.")
    for config, _ in runs:
        print(f"  {config.rules:<12} → {config.model} + {config.feature_set}"
              f"  ({EXPLAINABILITY_LEVEL.get(config.model, '?')} explainability)")