import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _enrollment_auc_from_arrays(
    enr_ids: np.ndarray,
    y_prob: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """
    Enrollment-level AUC from numpy arrays.
    Intended for use inside CV objective functions in model files.
    Groups rows by enrollment, takes max_prob per enrollment, computes AUC.
    Returns 0.5 if only one class is present in the val fold.
    """
    tmp = pd.DataFrame({"enrollment": enr_ids, "y_prob": y_prob, "y": y_true})
    enr = tmp.groupby("enrollment").agg(
        max_prob=("y_prob", "max"),
        dropped_out=("y", "max"),
    )
    try:
        return float(roc_auc_score(enr["dropped_out"], enr["max_prob"]))
    except Exception:
        return 0.5

## Carried onto every predictions frame. `older_for_cohort` is the fairness
## age axis (src/data.py); `undergraduate` stays for the descriptive tables and
## the level commentary, though it is no longer a fairness axis.
_SUBGROUP_COLS = [
    "female", "danish_nationality", "older_for_cohort",
    "undergraduate", "nationality", "age_at_study_start",
]


def _attach_subgroup_cols(
    enr: pd.DataFrame, d: pd.DataFrame, extra_cols: list[str] | None = None
) -> pd.DataFrame:
    cols = [c for c in _SUBGROUP_COLS if c in d.columns]
    if extra_cols:
        cols += [c for c in extra_cols if c in d.columns and c not in cols]
    if cols:
        sg = d.groupby("enrollment")[cols].first().reset_index()
        enr = enr.merge(sg, on="enrollment", how="left")
    return enr


def conservative_threshold(scores, train_positive_rate: float) -> float:
    """
    Smallest achievable τ whose flagged share does not exceed the target rate.

    τ is restricted to the observed score values (no quantile interpolation)
    and chosen as the *largest flagged set that still fits within the budget*
    of `train_positive_rate * n` — i.e. flag_rate ≤ target, never above.

    Rationale (capacity constraint). The target flagged share is an
    intervention budget: with counsellor capacity for the training dropout
    share of enrollments, a threshold may fall short of that budget but must
    not exceed it. Choosing the *nearest* achievable share instead is well
    behaved for continuous scores but not for coarse ones. An integer scorecard
    (FasterRisk) yields coarse scores, so many enrollments are tied on the same
    score; because flagging is "≥ τ", the nearest-share rule can pull in a whole
    tied block and overshoot the budget, inflating recall and depressing precision
    purely as an artifact of score granularity rather than of ranking quality.
    Rounding down removes that artifact and biases in a single, declared
    direction for every model.

    The rule is applied uniformly to all models. For continuous scores (one
    distinct value per enrollment) it is indistinguishable from a standard
    quantile threshold — it moves the flagged count by at most one enrollment.
    It binds only where the score is coarse.

    Only exception: if even the top score block already exceeds the budget,
    τ is that top score (the smallest flagged set achievable) and the
    realised flag rate does exceed the target. Report realised flag rates
    alongside recall/precision so this stays visible.
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if n == 0:
        return float("inf")
    # Whole enrollments. The epsilon absorbs float error in the product: 0.29 * 100
    # is 28.999999999999996, which would otherwise put 29 flagged over budget.
    target_n = np.floor(train_positive_rate * n + 1e-9)
    sorted_scores = np.sort(scores)
    candidates = np.unique(sorted_scores)
    # candidates ascending → n_flagged descending
    n_flagged = n - np.searchsorted(sorted_scores, candidates, side="left")
    within_budget = n_flagged <= target_n
    if not within_budget.any():
        return float(candidates[-1])
    # first True = smallest τ meeting the budget = largest flagged set within it
    return float(candidates[int(within_budget.argmax())])


def _enrollment_quantile_threshold(
    d: pd.DataFrame, train_positive_rate: float
) -> float:
    """
    Threshold calibrated on the enrollment-level distribution of max-prob.

    Picks τ so that the fraction of enrollments with max(y_prob) ≥ τ is as
    close as possible to — but not above — the training enrollment-level
    dropout rate. This aligns threshold calibration with the unit of
    evaluation (enrollment), so "the share of enrollments we flag" matches
    "the share that drop out in training". Follows the spirit of Yu et al.
    (10.1145/3430895.3460139) while operating at the enrollment level rather
    than the EWS-row level.

    See `conservative_threshold` for the rounding rule and why it rounds down.

    Expects `d` to have columns `enrollment` and `y_prob`.
    """
    enr_max_prob = d.groupby("enrollment")["y_prob"].max()
    return conservative_threshold(enr_max_prob.to_numpy(), train_positive_rate)


def _early_flag_rate(d: pd.DataFrame, flag_col: str, flagged_dropout_ids) -> float:
    """
    Of the dropout enrollments that were flagged, the share flagged at least one
    EWS activation before the last activation at which the student was seen.

    NOT `first_flag_date < status_change_date`. Rows exist only where
    `ews_date < status_change_date` — an enrollment ending exactly on an EWS date is
    excluded from that date — so *every* flag precedes the exit by
    construction, and that comparison would be 1 for any input.

    A student is only ever scored at an activation, so "one EWS period of
    warning" can only mean the flag was not raised at the last activation there
    was: `first_flag_date < max(ews_date)` over the enrollment's own rows. An
    enrollment flagged for the first time at its final activation contributes 0 —
    correctly, since no further activation existed to act on the flag.

    NaN when no dropout enrollment was flagged at all.
    """
    if len(flagged_dropout_ids) == 0:
        return np.nan
    first_flag = (d[d[flag_col] == 1]
                  .groupby("enrollment")["ews_date"]
                  .min()
                  .rename("first_flag_date"))
    last_ews = (d.groupby("enrollment")["ews_date"]
                .max()
                .rename("last_ews_date"))
    timing = (pd.DataFrame({"enrollment": pd.Series(flagged_dropout_ids).values})
              .merge(first_flag, on="enrollment", how="left")
              .merge(last_ews,   on="enrollment", how="left"))
    return float((timing["first_flag_date"] < timing["last_ews_date"]).mean())


def enrollment_predictions_df(
    df_subset: pd.DataFrame,
    y_prob,
    train_positive_rate: float,
    target: str = "dropout",
) -> pd.DataFrame:
    """Per-enrollment predictions with subgroup columns, for subgroup recall analysis."""
    d = df_subset.copy()
    d["y_prob"] = y_prob
    threshold = _enrollment_quantile_threshold(d, train_positive_rate)
    d["y_pred"] = (d["y_prob"] >= threshold).astype(int)
    enr = d.groupby("enrollment").agg(
        ever_flagged=("y_pred", "max"),
        max_prob=("y_prob", "max"),
        dropped_out=(target, "max"),
    ).reset_index()
    flagged_rows = d[d["y_pred"] == 1]
    if len(flagged_rows) > 0 and "ews_date" in d.columns:
        first_flag = (
            flagged_rows.groupby("enrollment")["ews_date"].min()
            .rename("first_flag_date")
        )
        enr = enr.merge(first_flag, on="enrollment", how="left")
    else:
        enr["first_flag_date"] = pd.NaT
    return _attach_subgroup_cols(enr, d, extra_cols=["person_id"])


def enrollment_pr_curve_data(
    enr_preds: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Enrollment-level precision-recall curve from per-enrollment max probabilities.

    enr_preds must have columns max_prob and dropped_out (one row per enrollment).
    Returns (recall_arr, precision_arr) — arrays of equal length for plotting.

    A point is dropped when any cell of its confusion matrix (TP, FP, FN, TN)
    holds 1..MIN_CELL_SIZE-1 students — statistical disclosure control, see
    src/disclosure.py. The top of the ranking (few flagged) and its bottom (few
    left unflagged) are where that happens.

    sklearn's (recall 0, precision 1) sentinel is NOT returned: it is no rule's
    result, and with the top-of-ranking points suppressed a drawn curve joined
    it straight to the first disclosable point — a stretch nothing produced. Each
    curve now starts at its first real point.
    """
    from src.disclosure import confusion_is_small
    from sklearn.metrics import precision_recall_curve
    y_true  = enr_preds["dropped_out"].values
    y_score = enr_preds["max_prob"].values
    n = len(enr_preds)
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # thresholds are ascending; for each threshold t, n_flagged = (y_score >= t).sum()
    # is descending.  Use sorted scores + searchsorted to compute efficiently.
    sorted_scores = np.sort(y_score)
    n_flagged = n - np.searchsorted(sorted_scores, thresholds, side="left")
    n_pos = int(y_true.sum())
    tp = np.rint(recall[:-1] * n_pos)
    fp = n_flagged - tp
    fn = n_pos - tp
    tn = (n - n_pos) - fp
    # precision/recall carry one extra sentinel element (recall=0, precision=1)
    # at the end, with no threshold behind it; it is dropped with the rest.
    keep = ~confusion_is_small(tp, fp, fn, tn)
    return recall[:-1][keep], precision[:-1][keep]


def credits_policy_predictions_df(
    df_subset: pd.DataFrame,
    ects_threshold: int = 60,
    target: str = "dropout",
) -> pd.DataFrame:
    """
    Per-enrollment credits_policy predictions with subgroup columns.

    max_prob is the per-enrollment max of ects_behind — a continuous risk
    score (higher = more behind = higher risk), of which the >=60 ECTS legal
    threshold is just one operating point. ever_flagged stays the binary
    >=60 ECTS rule. Same scoring as `_raw_credits_enr()` in plotting.py.
    """
    d = df_subset.copy()
    d["y_pred"] = (d["ects_behind"] >= ects_threshold).astype(int)
    enr = d.groupby("enrollment").agg(
        ever_flagged=("y_pred", "max"),
        max_prob=("ects_behind", "max"),
        dropped_out=(target, "max"),
    ).reset_index()
    return _attach_subgroup_cols(enr, d, extra_cols=["person_id"])


def row_predictions_df(
    df_subset: pd.DataFrame,
    y_prob,
    model: str,
    feature_set: str,
    test_year: int,
    target: str = "dropout",
) -> pd.DataFrame:
    """
    Per-row (enrollment × ews_date) predictions for temporal fairness analysis.
    Saves y_prob at each EWS activation alongside subgroup columns.
    Used by fairness_audit.run_over_periods().
    """
    d = df_subset.copy()
    d["y_prob"] = y_prob
    d["model"] = model
    d["feature_set"] = feature_set
    d["test_year"] = test_year

    keep = ["enrollment", "ews_date", "period", "ews_month", "y_prob", target,
            "model", "feature_set", "test_year"]
    for col in _SUBGROUP_COLS + ["person_id"]:
        if col in d.columns and col not in keep:
            keep.append(col)
    return d[[c for c in keep if c in d.columns]].copy()


def enrollment_level_metrics(df_subset, y_prob, train_positive_rate, target="dropout"):
    d = df_subset.copy()
    d["y_prob"] = y_prob
    threshold = _enrollment_quantile_threshold(d, train_positive_rate)
    d["y_pred"] = (d["y_prob"] >= threshold).astype(int)

    # ── per enrollment: ever flagged, max prob, dropped out ───────────────────
    enr = d.groupby("enrollment").agg(
        ever_flagged = ("y_pred", "max"),
        max_prob     = ("y_prob", "max"),
        dropped_out  = (target,   "max"),
    ).reset_index()

    # ── 1. enrollment-level recall ─────────────────────────────────────────────
    dropout_enrollments = enr[enr["dropped_out"] == 1]
    recall = (dropout_enrollments["ever_flagged"] == 1).mean()

    # ── 2. enrollment-level precision ──────────────────────────────────────────
    flagged_enrollments = enr[enr["ever_flagged"] == 1]
    precision = (flagged_enrollments["dropped_out"] == 1).mean() if len(flagged_enrollments) > 0 else 0

    # ── 3. early flag rate ─────────────────────────────────────────────────────
    flagged_dropout_ids = enr[
        (enr["dropped_out"] == 1) & (enr["ever_flagged"] == 1)
    ]["enrollment"]
    early_flag_rate = _early_flag_rate(d, "y_pred", flagged_dropout_ids)

    # ── 4. enrollment-level ROC AUC (max prob per enrollment) ─────────────────
    try:
        auc = roc_auc_score(enr["dropped_out"], enr["max_prob"])
    except Exception:
        auc = np.nan

    # ── 5. enrollment-level accuracy ──────────────────────────────────────────
    tp = int(((enr["ever_flagged"] == 1) & (enr["dropped_out"] == 1)).sum())
    tn = int(((enr["ever_flagged"] == 0) & (enr["dropped_out"] == 0)).sum())
    accuracy = (tp + tn) / len(enr) if len(enr) > 0 else np.nan

    n_enr = enr["enrollment"].nunique()
    n_flagged = int(flagged_enrollments["enrollment"].nunique())
    return {
        "n_enrollments":         n_enr,
        "n_dropout_enrollments": int(dropout_enrollments["enrollment"].nunique()),
        "n_flagged_enrollments": n_flagged,
        # exact TP, so disclosure control can rebuild the confusion matrix without
        # going through the 3-decimal recall (src/disclosure.py)
        "n_tp_enrollments":      tp,
        # realised flag rate — must be reported alongside recall/precision because
        # τ rounds down (see conservative_threshold); a model whose scores are
        # coarse can flag materially less than the calibration target, making its
        # recall a lower bound rather than a like-for-like comparison.
        "enr_flag_rate":         round(n_flagged / n_enr, 4) if n_enr else np.nan,
        "tau":                   round(float(threshold), 6),
        "enr_recall":            round(recall, 3),
        "enr_precision":         round(precision, 3),
        "enr_accuracy":          round(accuracy, 3) if not np.isnan(accuracy) else np.nan,
        "early_flag_rate":       round(early_flag_rate, 3) if not np.isnan(early_flag_rate) else np.nan,
        "roc_auc":               round(auc, 3) if not np.isnan(auc) else np.nan,
    }


def row_level_metrics(df_subset, y_prob, train_positive_rate, target="dropout"):
    """
    EWS-activation-level recall, precision, and AUC.

    Uses the same enrollment-level threshold as enrollment_level_metrics (calibrated
    so the fraction of enrollments with max_prob >= tau is as close as possible to,
    and never above, the training dropout rate — conservative_threshold),
    but applies it to individual (enrollment, ews_date) rows rather than aggregating to
    the enrollment level.  Intended as a model-diagnostic counterpart to
    enrollment_level_metrics — not the headline metric.
    """
    d = df_subset.copy()
    d["y_prob"] = y_prob
    threshold = _enrollment_quantile_threshold(d, train_positive_rate)
    d["y_pred"] = (d["y_prob"] >= threshold).astype(int)

    n_rows = len(d)
    n_dropout_rows = int((d[target] == 1).sum())

    dropout_rows = d[d[target] == 1]
    row_recall = float((dropout_rows["y_pred"] == 1).mean()) if len(dropout_rows) > 0 else 0.0

    flagged_rows = d[d["y_pred"] == 1]
    row_precision = float((flagged_rows[target] == 1).mean()) if len(flagged_rows) > 0 else 0.0

    try:
        row_auc = roc_auc_score(d[target], d["y_prob"])
    except Exception:
        row_auc = np.nan

    row_tp = int(((d["y_pred"] == 1) & (d[target] == 1)).sum())
    row_tn = int(((d["y_pred"] == 0) & (d[target] == 0)).sum())
    row_accuracy = (row_tp + row_tn) / n_rows if n_rows > 0 else np.nan

    return {
        "n_rows":          n_rows,
        "n_dropout_rows":  n_dropout_rows,
        "row_recall":      round(row_recall,    3),
        "row_precision":   round(row_precision, 3),
        "row_accuracy":    round(row_accuracy,  3) if not np.isnan(row_accuracy) else np.nan,
        "row_roc_auc":     round(row_auc, 3) if not np.isnan(row_auc) else np.nan,
    }


def credits_policy_row_level_metrics(df_subset, ects_threshold=60, target="dropout"):
    """Row-level counterpart of credits_policy_metrics."""
    d = df_subset.copy()
    d["y_pred"] = (d["ects_behind"] >= ects_threshold).astype(int)

    n_rows = len(d)
    n_dropout_rows = int((d[target] == 1).sum())

    dropout_rows = d[d[target] == 1]
    row_recall = float((dropout_rows["y_pred"] == 1).mean()) if len(dropout_rows) > 0 else 0.0

    flagged_rows = d[d["y_pred"] == 1]
    row_precision = float((flagged_rows[target] == 1).mean()) if len(flagged_rows) > 0 else 0.0

    try:
        row_auc = roc_auc_score(d[target], d["ects_behind"])
    except Exception:
        row_auc = np.nan

    cp_tp = int(((d["y_pred"] == 1) & (d[target] == 1)).sum())
    cp_tn = int(((d["y_pred"] == 0) & (d[target] == 0)).sum())
    cp_row_accuracy = (cp_tp + cp_tn) / n_rows if n_rows > 0 else np.nan

    return {
        "n_rows":          n_rows,
        "n_dropout_rows":  n_dropout_rows,
        "row_recall":      round(row_recall,    3),
        "row_precision":   round(row_precision, 3),
        "row_accuracy":    round(cp_row_accuracy, 3) if not np.isnan(cp_row_accuracy) else np.nan,
        "row_roc_auc":     round(row_auc, 3) if not np.isnan(row_auc) else np.nan,
    }


def credits_policy_metrics(df_subset, ects_threshold=60, target="dropout"):
    """
    Enrollment-level metrics for the credits policy:
    flag any enrollment that was ever >= ects_threshold behind.
    """
    d = df_subset.copy()
    d["flagged"] = (d["ects_behind"] >= ects_threshold).astype(int)

    enr = d.groupby("enrollment").agg(
        ever_flagged = ("flagged",      "max"),
        max_behind   = ("ects_behind",  "max"),
        dropped_out  = (target,         "max"),
    ).reset_index()

    dropout_enrollments = enr[enr["dropped_out"] == 1]
    flagged_enrollments = enr[enr["ever_flagged"] == 1]

    recall    = (dropout_enrollments["ever_flagged"] == 1).mean()
    precision = (flagged_enrollments["dropped_out"] == 1).mean() if len(flagged_enrollments) > 0 else 0

    # early flag rate — see _early_flag_rate for what "early" can mean here
    flagged_dropout_ids = enr[
        (enr["dropped_out"] == 1) & (enr["ever_flagged"] == 1)
    ]["enrollment"]
    early_flag_rate = _early_flag_rate(d, "flagged", flagged_dropout_ids)

    # roc_auc using max ects_behind per enrollment as score
    valid = enr[["dropped_out", "max_behind"]].dropna()
    try:
        auc = roc_auc_score(valid["dropped_out"], valid["max_behind"])
    except Exception:
        auc = np.nan

    cp_tp = int(((enr["ever_flagged"] == 1) & (enr["dropped_out"] == 1)).sum())
    cp_tn = int(((enr["ever_flagged"] == 0) & (enr["dropped_out"] == 0)).sum())
    accuracy = (cp_tp + cp_tn) / len(enr) if len(enr) > 0 else np.nan

    n_enr = enr["enrollment"].nunique()
    n_flagged = int(flagged_enrollments["enrollment"].nunique())
    return {
        "n_enrollments":         n_enr,
        "n_dropout_enrollments": int(dropout_enrollments["enrollment"].nunique()),
        "n_flagged_enrollments": n_flagged,
        "n_tp_enrollments":      cp_tp,
        # the AUC ranks only enrollments with an ects_behind value
        "n_auc_pos":             int((valid["dropped_out"] == 1).sum()),
        "n_auc_neg":             int((valid["dropped_out"] == 0).sum()),
        # the credits policy is not τ-calibrated — this is the share the fixed
        # national-law rule happens to flag, reported for comparability
        "enr_flag_rate":         round(n_flagged / n_enr, 4) if n_enr else np.nan,
        "tau":                   float(ects_threshold),
        "enr_recall":            round(recall, 3),
        "enr_precision":         round(precision, 3),
        "enr_accuracy":          round(accuracy, 3) if not np.isnan(accuracy) else np.nan,
        "early_flag_rate":       round(early_flag_rate, 3) if not np.isnan(early_flag_rate) else np.nan,
        "roc_auc":               round(auc, 3) if not np.isnan(auc) else np.nan,
    }

## ABROCA grid resolution — the default of Gardner et al.'s own implementation.
ABROCA_N_GRID = 10_000
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")  # renamed in NumPy 2.0


def roc_on_grid(y_true, score, fpr_grid) -> tuple[np.ndarray, float]:
    """One group's ROC curve interpolated onto `fpr_grid`, and its exact AUC."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, score)
    return np.interp(fpr_grid, fpr, tpr), float(_trapz(tpr, fpr))


def abroca(tpr_ref, tpr_comp, fpr_grid) -> float:
    """
    Absolute Between-ROC Area (Gardner, Brooks & Baker, LAK 2019):
    the integral over FPR of |ROC_comp - ROC_ref|, for two ROC curves already
    on a common grid (roc_on_grid).

    Unsigned, and NOT the AUC difference. Integrating the signed gap gives
    AUC_comp - AUC_ref exactly: where the curves cross, the stretches where
    each group is ahead cancel. Which group ranks better overall is AUC_comp - AUC_ref;
    ABROCA says only how far apart the curves lie.
    """
    return float(_trapz(np.abs(np.asarray(tpr_comp) - np.asarray(tpr_ref)), fpr_grid))


# ── Fixed contact budget ─────────────────────────────────────────────────────

def recall_precision_at_budget(scores, dropped_out, k: int) -> dict:
    """
    Recall and precision when exactly `k` enrollments are contacted, highest
    score first — one score and one outcome per enrollment.

    A score that ties at the cutoff cannot be split by rank, so the places left
    over at the cutoff are spread evenly over the tied enrollments and the
    result is an expectation:

        expected TP = dropouts above the cutoff
                      + dropouts at the cutoff x (places left / enrollments at the cutoff)

    For a continuous score nothing ties and this is plain top-k. A missing score
    ranks last. `tp_low` / `tp_high` are the whole-student bounds of the
    expected TP, for disclosure control.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(dropped_out, dtype=int)
    n = len(s)
    if not 0 < k <= n:
        raise ValueError(f"budget {k} outside 1..{n}")
    s = np.where(np.isnan(s), -np.inf, s)
    cutoff = np.sort(s)[::-1][k - 1]
    above, at = s > cutoff, s == cutoff
    left = k - int(above.sum())
    n_at = int(at.sum())
    tp_above = int(y[above].sum())
    tp_at = int(y[at].sum())
    exp_tp = tp_above + tp_at * left / n_at
    n_pos = int(y.sum())
    return {
        "budget": k, "n_enrollments": n, "n_dropout": n_pos,
        "expected_tp": exp_tp,
        "recall": exp_tp / n_pos if n_pos else np.nan,
        "precision": exp_tp / k,
        "n_at_cutoff": n_at, "places_at_cutoff": left,
        "tie_split": left < n_at,
        "tp_low": int(np.floor(exp_tp)), "tp_high": int(np.ceil(exp_tp)),
    }


# ── Bootstrap intervals ──────────────────────────────────────────────────────

# A bootstrap draw in which a metric is undefined (recall with no dropouts in the
# resampled group, AUC with one class) cannot be placed in the distribution.
# Above this share of such draws the interval is withheld rather than computed
# on the draws that happened to be defined.
MAX_UNDEFINED_DRAWS = 0.05


def percentile_ci(draws) -> tuple[float, float]:
    """
    95% percentile interval over ALL bootstrap draws, or (NaN, NaN) if more than
    MAX_UNDEFINED_DRAWS of them are undefined.

    Draws must be computed without disclosure control: suppressing a metric
    inside a draw discards exactly the draws whose counts swing low — the tails —
    and an interval over the survivors is too narrow. Suppress at output instead:
    show the interval only next to a shown point estimate.
    """
    a = np.asarray(draws, dtype=float)
    if a.size == 0:
        return np.nan, np.nan
    ok = ~np.isnan(a)
    if ok.mean() < 1 - MAX_UNDEFINED_DRAWS:
        return np.nan, np.nan
    return float(np.percentile(a[ok], 2.5)), float(np.percentile(a[ok], 97.5))

