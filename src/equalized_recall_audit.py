"""
Equalized-recall threshold audit for the Participatory model.

Deliberative-workshop participants said they would accept identifying 1-3
fewer students per 100 (of those experiencing academic difficulties) in
exchange for equal model performance across demographic subgroups. This
module tests whether the recall cost of actually equalizing recall across a
subgroup — via group-specific decision thresholds, holding the Participatory
model's predicted probabilities fixed (no retraining) — falls inside that
tolerance band.

This is deliberately NOT src/fa_frontier.py's frontier sweep: that sweep uses
a single *global* threshold and reports per-group misclassification error
(1 - accuracy) as it varies, so it never assigns different thresholds to
different groups. Group-specific thresholding lives in src/debias.py
(`equal_opportunity_thresholds`, Hardt et al. 2016) — this module reuses that
directly and adds the piece debias.py doesn't compute: the overall (pooled)
enrollment-level recall that results once each group is thresholded at its
own equal-opportunity τ, compared against the model's current single-
threshold recall.

Run for all three subgroup axes reported elsewhere (sex, nationality, entry
age relative to cohort) — not just one, since each answers a materially different fairness
question.

Outputs → results/fairness/equalized_recall_audit.csv

Usage:
    python src/equalized_recall_audit.py
    python src/equalized_recall_audit.py --model logistic_regression --feature-set b_credits_performance --year 2021
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from paths.paths import RESULTS_PATH
from src import paper_style
from src import disclosure
from src.config import MAIN_TEST_YEAR, RANDOM_SEED
from src.debias import (
    apply_group_thresholds,
    budgeted_equal_opportunity_thresholds,
    equal_opportunity_thresholds,
)
from src.fairness_audit import _BINARY_SUBGROUPS, _resample, _resolve_participatory_model

## The paper style, shared with every figure (src/paper_style.py).
paper_style.apply_style()

_COLORS = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#D55E00", "#CC79A7"]
_PLOT_DIR = RESULTS_PATH / "plots" / "fairness"

# The Participatory model is whatever the workshop selection rules chose — it
# is resolved via workshop_rules.participatory_config() at call time, never
# hardcoded (see _resolve_participatory_model's docstring in fairness_audit.py).
# The pair below is only a fallback for when no selection config exists at all;
# it is not assumed to be what the selection rules choose.

_FAIRNESS_DIR = RESULTS_PATH / "fairness"


def _participatory_model(
    model: str | None, feature_set: str | None
) -> tuple[str, str]:
    """
    Resolve which (model, feature_set) to audit.

    Explicit --model/--feature-set always win; otherwise the workshop selection
    config decides. Warns when an explicit choice disagrees with the selection,
    and stops when neither is available.
    """
    try:
        resolved_model, resolved_fs = _resolve_participatory_model()
    except FileNotFoundError:
        resolved_model, resolved_fs = None, None

    if model is None:
        model = resolved_model
    if feature_set is None:
        feature_set = resolved_fs

    if model is None or feature_set is None:
        raise SystemExit(
            "No workshop selection found — run src/workshop_evaluation.py first, "
            "or pass --model and --feature-set."
        )
    if resolved_model is None:
        print(f"Auditing {model}/{feature_set} (no workshop selection to compare with).")
    elif (resolved_model, resolved_fs) != (model, feature_set):
        print(
            f"NOTE: auditing {model}/{feature_set}, but the workshop selection "
            f"resolves the Participatory model to {resolved_model}/{resolved_fs}. "
            "Drop --model/--feature-set to audit the selected model instead."
        )
    else:
        print(f"Participatory model (from workshop selection): {model}/{feature_set}")

    return model, feature_set


def _print_available_combinations(preds_df: pd.DataFrame, test_year: int) -> None:
    """List the model/feature_set pairs that do exist, so the miss is diagnosable."""
    avail = (
        preds_df[preds_df["test_year"] == test_year][["model", "feature_set"]]
        .drop_duplicates()
        .sort_values(["model", "feature_set"])
    )
    if avail.empty:
        print(f"  enrollment_predictions.csv has no rows at all for test_year={test_year}.")
        return
    print(f"  Available for test_year={test_year}:")
    for _, r in avail.iterrows():
        print(f"    {r['model']}/{r['feature_set']}")


def _train_pos_rate() -> float | None:
    """
    Training enrollment-level dropout rate — the intervention budget the global
    threshold is calibrated to. Returns None if the metadata is missing, in
    which case the audit falls back to the unconstrained specification and says
    so rather than silently inventing a budget.
    """
    meta_path = RESULTS_PATH / "models" / "train_metadata.json"
    if not meta_path.exists():
        return None
    with open(meta_path, encoding="utf-8") as f:
        return float(json.load(f)["train_pos_rate"])


def _resolve_budget(budget_rate: float | None, unconstrained: bool) -> float | None:
    """Budget to price equalization against; None = unconstrained."""
    if unconstrained:
        print("  Unconstrained (Hardt et al.) specification — flag rate not capped.")
        return None
    if budget_rate is None:
        budget_rate = _train_pos_rate()
    if budget_rate is None:
        print("  WARNING: results/models/train_metadata.json not found — no capacity "
              "budget available, falling back to the unconstrained specification.")
        return None
    print(f"  Capacity budget: flagged share capped at train_pos_rate = {budget_rate:.4f}")
    return budget_rate


def _pooled_recall(enr: pd.DataFrame, flag_col: str = "ever_flagged") -> float:
    do = enr["dropped_out"]
    fl = enr[flag_col]
    tp = int(((do == 1) & (fl == 1)).sum())
    fn = int(((do == 1) & (fl == 0)).sum())
    return tp / (tp + fn) if (tp + fn) > 0 else np.nan


def _error_rate(enr: pd.DataFrame, flag_col: str = "ever_flagged") -> float:
    """1 - accuracy, matching src/fa_frontier.py's error-rate definition."""
    if len(enr) == 0:
        return np.nan
    return float((enr[flag_col] != enr["dropped_out"]).mean())


def equalize_recall(
    subset: pd.DataFrame,
    subgroup_col: str,
    reference_val,
    n_boot: int = 1000,
    seed: int = RANDOM_SEED,
    budget_rate: float | None = None,
) -> dict:
    """
    Set per-group equal-opportunity thresholds for `subgroup_col`, then report:
      - the model's current single-threshold pooled recall
      - the pooled recall once every group uses its own equalizing threshold
      - the difference (equalized - single), in recall points and pp
      - the flagged share before and after, so the cost of equalization is
        visible in contacts as well as in recall
      - the residual recall gap between groups at the equalized thresholds
        (zero only up to one student per group: thresholds cannot split students)
      - the error-rate gap between groups at the single threshold AND at the
        equalized thresholds. Error is not targeted, so the gap is not expected
        to shrink — moving each group's threshold to equalize recall changes its
        false positives, and the gap often widens.
      - 95% bootstrap CIs (resampling enrollments, decision rules held fixed)
        on the equalized recall and on the recall difference

    `budget_rate` (default: the training enrollment-level dropout rate, passed
    in by the caller) caps the pooled flagged share, so equalization is priced
    against a fixed intervention budget — the same capacity constraint the
    single global threshold is calibrated under. This is the specification the
    workshop tolerance is defined against; see
    `budgeted_equal_opportunity_thresholds` in src/debias.py. Pass
    `budget_rate=None` for the unconstrained Hardt et al. (2016) version, which
    is retained as a diagnostic: it shows how far the global τ sits from the
    recall frontier, but its "gains" are bought with unbudgeted extra contacts.
    """
    if budget_rate is None:
        thresholds = equal_opportunity_thresholds(subset, subgroup_col, reference_val)
        target_recall = np.nan
    else:
        thresholds, target_recall = budgeted_equal_opportunity_thresholds(
            subset, subgroup_col, budget_rate
        )
    debiased = apply_group_thresholds(subset, thresholds, subgroup_col)

    single_recall    = _pooled_recall(subset, "ever_flagged")
    equalized_recall = _pooled_recall(debiased, "ever_flagged")
    diff             = equalized_recall - single_recall

    non_ref  = [g for g in subset[subgroup_col].dropna().unique() if g != reference_val]
    comp_val = non_ref[0] if non_ref else None

    ref_after  = debiased[debiased[subgroup_col] == reference_val]
    comp_after = debiased[debiased[subgroup_col] == comp_val] if comp_val is not None else debiased.iloc[0:0]
    recall_gap_after = _pooled_recall(comp_after) - _pooled_recall(ref_after)
    error_gap_after  = _error_rate(comp_after) - _error_rate(ref_after)

    ref_before  = subset[subset[subgroup_col] == reference_val]
    comp_before = subset[subset[subgroup_col] == comp_val] if comp_val is not None else subset.iloc[0:0]
    error_gap_before = _error_rate(comp_before) - _error_rate(ref_before)

    # Bootstrap: decision rules (global tau already baked into "ever_flagged";
    # group taus already baked into "ever_flagged_eq") are held fixed — only
    # the sample is resampled, matching the convention in
    # src/fairness_audit.py's bootstrap_differences / src/debias.py's
    # evaluate_strategy (which resample already-thresholded frames rather
    # than re-fitting thresholds per resample).
    combined = subset.copy()
    combined["ever_flagged_eq"] = debiased["ever_flagged"].values

    rng = np.random.default_rng(seed)
    boot_equalized, boot_diff = [], []
    for _ in range(n_boot):
        s = _resample(combined, rng)
        r_single = _pooled_recall(s, "ever_flagged")
        r_eq     = _pooled_recall(s, "ever_flagged_eq")
        boot_equalized.append(r_eq)
        boot_diff.append(r_eq - r_single)

    boot_equalized = [v for v in boot_equalized if not np.isnan(v)]
    boot_diff      = [v for v in boot_diff if not np.isnan(v)]

    # ── disclosure control (src/disclosure.py) ───────────────────────────────
    # Each printed number is checked against every count it rests on.
    do_       = combined["dropped_out"] == 1
    single_fl = combined["ever_flagged"] == 1
    eq_fl     = combined["ever_flagged_eq"] == 1
    n_all, n_pos = len(combined), int(do_.sum())

    def _pooled_hidden(flag) -> bool:
        return disclosure.operating_point_hidden(n_all, n_pos, int(flag.sum()),
                                                 int((flag & do_).sum()))["recall"]

    newly_caught = int((do_ & eq_fl & ~single_fl).sum())
    no_longer    = int((do_ & single_fl & ~eq_fl).sum())
    hide_single  = _pooled_hidden(single_fl)
    hide_eq      = _pooled_hidden(eq_fl)

    def _errors_hidden(g_df) -> bool:
        wrong = int((g_df["ever_flagged"] != g_df["dropped_out"]).sum())
        return disclosure.rate_is_small(wrong, len(g_df))

    return {
        "hidden_single_recall":    hide_single,
        "hidden_equalized_recall": hide_eq,
        "hidden_diff":             (hide_single or hide_eq
                                    or disclosure.is_small(newly_caught, no_longer)),
        "hidden_flag_rate_after":  disclosure.rate_is_small(int(eq_fl.sum()), n_all),
        "hidden_error_rate_gap":   _errors_hidden(ref_after) or _errors_hidden(comp_after),
        "hidden_error_rate_gap_before": _errors_hidden(ref_before) or _errors_hidden(comp_before),
        "subgroup":                subgroup_col,
        "reference":               reference_val,
        "comparison":              comp_val,
        "n":                       len(subset),
        "single_threshold_recall": single_recall,
        "equalized_recall":        equalized_recall,
        "budget_rate":             budget_rate if budget_rate is not None else np.nan,
        "target_recall":           target_recall,
        "flag_rate_before":        float(subset["ever_flagged"].mean()),
        "flag_rate_after":         float(debiased["ever_flagged"].mean()),
        "n_flagged_before":        int(subset["ever_flagged"].sum()),
        "n_flagged_after":         int(debiased["ever_flagged"].sum()),
        "equalized_recall_lo":     float(np.percentile(boot_equalized, 2.5))  if boot_equalized else np.nan,
        "equalized_recall_hi":     float(np.percentile(boot_equalized, 97.5)) if boot_equalized else np.nan,
        "diff_pp":                 diff * 100,
        "diff_pp_lo":              float(np.percentile(boot_diff, 2.5))  * 100 if boot_diff else np.nan,
        "diff_pp_hi":              float(np.percentile(boot_diff, 97.5)) * 100 if boot_diff else np.nan,
        "recall_gap_after":        recall_gap_after,
        "error_rate_gap_before":   error_gap_before,
        "error_rate_gap_after":    error_gap_after,
        "threshold_reference":     thresholds.get(reference_val, np.nan),
        "threshold_comparison":    thresholds.get(comp_val, np.nan) if comp_val is not None else np.nan,
    }


def run_equalized_recall_audit(
    model: str | None     = None,
    feature_set: str | None = None,
    test_year: int        = MAIN_TEST_YEAR,
    subgroup_cols: list[str] | None = None,
    n_boot: int           = 1000,
    save: bool            = True,
    budget_rate: float | None = None,
    unconstrained: bool   = False,
) -> pd.DataFrame:
    """
    Run the equalized-recall audit for every subgroup axis in `subgroup_cols`
    (default: all three binary axes — sex, nationality, entry age relative to cohort).
    `model`/`feature_set` default to the workshop-selected Participatory model.

    By default equalization is priced against the training enrollment-level
    dropout rate as a capacity budget, matching the global threshold's
    calibration — so the reported recall difference is a genuine cost paid at
    constant contacts, which is what the workshop's tolerance was expressed
    against. `unconstrained=True` reverts to Hardt et al. (2016) without a
    budget (diagnostic only).

    Saves equalized_recall_audit.csv → results/fairness/.
    """
    model, feature_set = _participatory_model(model, feature_set)

    enr_preds_path = RESULTS_PATH / "enrollment_predictions.csv"
    if not enr_preds_path.exists():
        print(f"ERROR: {enr_preds_path} not found — run evaluation.py first.")
        return pd.DataFrame()

    preds_df = pd.read_csv(enr_preds_path)
    subset = preds_df[
        (preds_df["test_year"]   == test_year) &
        (preds_df["model"]       == model) &
        (preds_df["feature_set"] == feature_set)
    ].copy()
    if len(subset) == 0:
        print(f"ERROR: no predictions for {model}/{feature_set}, test_year={test_year}.")
        _print_available_combinations(preds_df, test_year)
        return pd.DataFrame()

    if subgroup_cols is None:
        subgroup_cols = list(_BINARY_SUBGROUPS.keys())

    budget = _resolve_budget(budget_rate, unconstrained)

    rows = []
    for col in subgroup_cols:
        if col not in subset.columns:
            print(f"  Skipping {col} — not in predictions.")
            continue
        ref_val, ref_label, comp_val, comp_label = _BINARY_SUBGROUPS[col]
        print(f"  Equalizing recall on {col} ({comp_label} vs {ref_label}) ...")
        # Only enrollments with a value on this axis. apply_group_thresholds
        # re-flags each group at its own threshold and leaves everyone else
        # unflagged, so unknown citizenship (danish_nationality NaN) kept in the
        # frame was silently unflagged after equalization — lowering the
        # equalized recall and flagged share for a reason that has nothing to do
        # with equalizing. Same convention as fairness_audit._all_subgroup_diffs.
        axis = subset[subset[col].notna()]
        result = equalize_recall(axis, col, ref_val, n_boot=n_boot,
                                 budget_rate=budget)
        result["comparison_label"] = f"{comp_label} vs {ref_label}"
        result["model"]            = model
        result["feature_set"]      = feature_set
        result["test_year"]        = test_year
        rows.append(result)

    result_df = pd.DataFrame(rows)
    if save and not result_df.empty:
        _FAIRNESS_DIR.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(_FAIRNESS_DIR / "equalized_recall_audit.csv", index=False)
        print(f"Saved → {_FAIRNESS_DIR / 'equalized_recall_audit.csv'}")


    return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EWS equalized-recall threshold audit")
    parser.add_argument("--model",       default=None,
                        help="Default: the workshop-selected Participatory model")
    parser.add_argument("--feature-set", default=None, dest="feature_set",
                        help="Default: the workshop-selected feature set")
    parser.add_argument("--year",        type=int, default=MAIN_TEST_YEAR)
    parser.add_argument("--subgroup",    nargs="+", default=None, dest="subgroup_cols",
                        help="Subgroup columns (default: all binary subgroups)")
    parser.add_argument("--n-boot",      type=int, default=1000, dest="n_boot")
    parser.add_argument("--unconstrained", action="store_true",
                        help="Drop the capacity budget (plain Hardt et al. 2016). "
                             "Diagnostic only: recall 'gains' are then bought with "
                             "unbudgeted extra contacts.")
    parser.add_argument("--budget-rate", type=float, default=None, dest="budget_rate",
                        help="Override the flagged-share cap (default: train_pos_rate "
                             "from results/models/train_metadata.json)")
    args = parser.parse_args()

    df = run_equalized_recall_audit(
        model=args.model,
        feature_set=args.feature_set,
        test_year=args.year,
        subgroup_cols=args.subgroup_cols,
        n_boot=args.n_boot,
        budget_rate=args.budget_rate,
        unconstrained=args.unconstrained,
    )
    if df.empty:
        # Nothing is written when the result is empty, so exiting 0 here would
        # let a pipeline run "succeed" with no audit output at all.
        sys.exit(
            "ERROR: equalized-recall audit produced no rows — nothing written. "
            "See the message above for which model/feature_set was requested."
        )
    print("\n" + df.to_string(index=False))
