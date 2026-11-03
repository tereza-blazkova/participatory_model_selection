"""
Evaluation on the October 2023 EWS snapshot (pilot_snapshot), faculty by faculty.

FAC_A and FAC_B piloted counsellor outreach in October 2023; every result is
computed per faculty, on that faculty's enrollments. evaluate_faculty() scores
each trained model and the rule-based baselines — the credits policy and the
pilot's own contact list — on one faculty. The rest recovers the pilot's
contact threshold, merges sparse points-profile bins, and runs the
small-group checks the pilot figures rely on
(figures/scripts/prepare_pilot.py).
"""
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")  # renamed in NumPy 2.0
import pandas as pd
from sklearn.metrics import roc_auc_score as _roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from paths.paths import HELVETICA, RESULTS_PATH
from src import paper_style
from src import disclosure, small_cells_report
from src.display import MODEL_LABELS
from src.config import MIN_CELL_SIZE, TARGET
from src.metrics import (
    ABROCA_N_GRID,
    _attach_subgroup_cols,
    abroca,
    credits_policy_metrics,
    enrollment_level_metrics,
    enrollment_predictions_df,
    roc_on_grid,
    row_level_metrics,
)
from src.features import load_model_checked

## Imported here for figures/scripts/prepare_pilot.py, which takes them from
## this module rather than from where they are defined.
from paths.paths import PILOT_SNAPSHOT_PARQUET                                   # noqa: F401
from src.config import USED_VARS                                           # noqa: F401
from src.fairness_audit import _BINARY_SUBGROUPS as _FAIRNESS_BINARY_SUBGROUPS  # noqa: F401

# ── Style ─────────────────────────────────────────────────────────────────────

## The paper style, shared with every figure (src/paper_style.py).
paper_style.apply_style()

# ── Constants ─────────────────────────────────────────────────────────────────

_COLORS = [paper_style.METHOD_COLOR[m] for m in ["fasterrisk", "logistic_regression", "xgboost", "mlp"]]  # one colour per method (paper_style)
_DROPOUT_COLOR    = "#E69F00"
_NONDROPOUT_COLOR = "#56B4E9"
_DROPOUT_LIGHT    = "#F8E3B3"
_NONDROPOUT_LIGHT = "#CCE9F8"
_MARKERS        = [paper_style.METHOD_MARKER[m] for m in ["fasterrisk", "logistic_regression", "xgboost", "mlp"]]
_RULE_COLOR     = "#D55E00"   # vermillion — credits_policy; consistent with plotting.py
_WORKSHOP_COLOR = "#009E73"
_REF_LW         = 2.2

FACULTIES            = ["FAC_A", "FAC_B"]
_FACULTY_CONTACT     = {"FAC_A": "contact_fac_a", "FAC_B": "contact_fac_b"}
_FACULTY_COLOR       = {"FAC_A": "#0072B2",       "FAC_B": "#FFC107"}
_FACULTY_PANEL_LABEL = {"FAC_A": "a",             "FAC_B": "b"}
_FACULTY_DISPLAY     = {"FAC_A": "Faculty A",     "FAC_B": "Faculty B"}

_ML_MODELS    = ["fasterrisk", "logistic_regression", "xgboost", "mlp"]
_MODEL_DISPLAY = {
    "fasterrisk":          "FasterRisk",
    "logistic_regression": "Logistic Regression",
    "xgboost":             "XGBoost",
    "mlp":                 MODEL_LABELS["mlp"],   # one name, in src/display.py
}
_CONFIG_ORDER = ["a_credits", "b_credits_performance", "c_plus_enrolment",
                 "d_plus_lms", "e_plus_sensitive"]

_METRIC_COLS = {
    "enrollment": {
        "enr_recall":    "enr_recall",
        "enr_precision": "enr_precision",
        "enr_accuracy":  "enr_accuracy",
        "roc_auc":       "roc_auc",
    },
    "ews_row": {
        "enr_recall":    "row_recall",
        "enr_precision": "row_precision",
        "enr_accuracy":  "row_accuracy",
        "roc_auc":       "row_roc_auc",
    },
}
_METRIC_LABELS = {
    "enr_recall":    "Dropout detection rate",
    "enr_precision": "Share of flags that were dropouts",
    "enr_accuracy":  "Enrollment-level accuracy",
    "roc_auc":       "ROC AUC",
    "row_recall":    "Dropout detection rate (activation level)",
    "row_precision": "Share of flags that were dropouts (activation level)",
    "row_accuracy":  "Accuracy (activation level)",
    "row_roc_auc":   "ROC AUC (activation level)",
}
_LEVEL_SUFFIX = {
    "enrollment": "(enrollment-level)",
    "ews_row":    "(activation level — model diagnostic)",
}

_MODELS_DIR      = RESULTS_PATH / "models"


# ── Evaluation ────────────────────────────────────────────────────────────────

def _contact_recall(pilot_snapshot: pd.DataFrame, col: str, score_col: str | None = None) -> dict:
    """Enrollment-level recall/precision treating col == 1 as 'flagged'.

    score_col: continuous ranking score for ROC AUC (e.g. 'points_total').
               Falls back to the binary flag if None or not present in pilot_snapshot.
    """
    enr = pilot_snapshot.groupby("enrollment").agg(
        ever_flagged=(col,    "max"),
        dropped_out =(TARGET, "max"),
    ).reset_index()
    if score_col is not None and score_col in pilot_snapshot.columns:
        scores = pilot_snapshot.groupby("enrollment")[score_col].max().reindex(enr["enrollment"]).values
    else:
        scores = enr["ever_flagged"].values
    dropout_enr = enr[enr["dropped_out"] == 1]
    flagged_enr = enr[enr["ever_flagged"] == 1]
    recall    = float((dropout_enr["ever_flagged"] == 1).mean()) if len(dropout_enr) > 0 else 0.0
    precision = float((flagged_enr["dropped_out"]  == 1).mean()) if len(flagged_enr)  > 0 else 0.0
    tp = int(((enr["ever_flagged"] == 1) & (enr["dropped_out"] == 1)).sum())
    tn = int(((enr["ever_flagged"] == 0) & (enr["dropped_out"] == 0)).sum())
    accuracy = (tp + tn) / len(enr) if len(enr) > 0 else float("nan")
    try:
        auc = float(_roc_auc_score(enr["dropped_out"], scores))
    except Exception:
        auc = float("nan")
    return {
        "enr_recall":    round(recall,    3),
        "enr_precision": round(precision, 3),
        "enr_accuracy":  round(accuracy,  3),
        "roc_auc":       round(auc,       3),
        "n_flagged":     int(flagged_enr["enrollment"].nunique()),
        "n_dropout":     int(dropout_enr["enrollment"].nunique()),
        # for disclosure control (src/disclosure.py): the full confusion matrix
        "n_enrollments": int(len(enr)),
        "n_tp":          tp,
    }


def _load_metadata() -> dict:
    meta_path = _MODELS_DIR / "train_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Train metadata not found at {meta_path}.\n"
            "Run  python src/evaluation.py  first."
        )
    with open(meta_path) as fh:
        return json.load(fh)


def evaluate_faculty(
    tab_subset: pd.DataFrame,
    contact_col: str,
    available_features: dict,
    train_pos_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict, pd.DataFrame]:
    """
    Evaluate all saved models on one faculty's subset of pilot_snapshot.

    Returns
    -------
    enr_df, row_df, rule_metrics, contact_metrics, preds_df
    """
    enr_rows, row_rows, preds_rows = [], [], []

    for config_name in _CONFIG_ORDER:
        available = available_features[config_name]
        tab_X     = tab_subset[available]

        for model_name in _ML_MODELS:
            model_path = _MODELS_DIR / f"{model_name}__{config_name}.joblib"
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Saved model not found: {model_path}\n"
                    "Run  python src/evaluation.py  first."
                )
            model  = load_model_checked(model_path, available)
            y_prob = model.predict_proba(tab_X)[:, 1]

            enr_m = enrollment_level_metrics(tab_subset, y_prob, train_pos_rate, target=TARGET)
            row_m = row_level_metrics(tab_subset, y_prob, train_pos_rate, target=TARGET)
            enr_rows.append({"model": model_name, "feature_set": config_name, **enr_m})
            row_rows.append({"model": model_name, "feature_set": config_name, **row_m})

            preds = enrollment_predictions_df(tab_subset, y_prob, train_pos_rate, target=TARGET)
            preds["model"]       = model_name
            preds["feature_set"] = config_name
            preds["test_year"]   = 2023
            preds_rows.append(preds)

    enr_df       = pd.DataFrame(enr_rows)
    row_df       = pd.DataFrame(row_rows)
    preds_df     = pd.concat(preds_rows, ignore_index=True)
    rule_metrics = credits_policy_metrics(tab_subset, target=TARGET)
    contact_m    = _contact_recall(tab_subset, col=contact_col, score_col="points_total")

    return enr_df, row_df, rule_metrics, contact_m, preds_df


# ── Fairness audit ────────────────────────────────────────────────────────────

def _flagged_enr(
    subset: pd.DataFrame,
    flag_col: str,
    score_col: str,
    target: str = TARGET,
) -> pd.DataFrame:
    """
    Per-enrollment ever_flagged / max_prob / dropped_out frame, with subgroup
    + person_id columns, aggregated per enrollment (max within enrollment).

    flag_col: binary decision column (e.g. the >=60 ECTS credits_policy
        flag, or the recorded contact_fac_b/contact_fac_a outreach flag).
    score_col: continuous ranking score, always aggregated into max_prob so
        AUC can be computed downstream (e.g. by fairness_audit._compute_metrics).
        For credits_policy this is 'ects_behind'; for the contact pilot this is
        'points_total', the score behind the binary contact flag.
    """
    enr = subset.groupby("enrollment").agg(
        ever_flagged=(flag_col, "max"),
        max_prob=(score_col, "max"),
        dropped_out=(target, "max"),
    ).reset_index()
    return _attach_subgroup_cols(enr, subset, extra_cols=["person_id"])


# ── ABROCA: ROC curve gap by demographic subgroup ─────────────────────────────

def _compute_abroca(
    preds: pd.DataFrame,
    group_col: str,
    ref_val,
    comp_val,
    n_grid: int = ABROCA_N_GRID,
) -> tuple:
    """
    ABROCA between two groups and their ROC curves on a common FPR grid.

    preds must have columns: dropped_out, max_prob, <group_col>.
    Returns (fpr_grid, tpr_ref, tpr_comp, abroca, auc_ref, auc_comp).
    `abroca` is unsigned (src/metrics.abroca); the direction of the gap is
    auc_comp - auc_ref. tpr_ref / tpr_comp are None if a group has fewer than
    MIN_CELL_SIZE enrollments or only one outcome class.
    """
    fpr_grid = np.linspace(0, 1, n_grid)

    def _interp_roc(df):
        # A ROC curve over 1..MIN_CELL_SIZE-1 dropouts (or non-dropouts) is those students'
        # ranks: not computed, see _abroca_hidden / src/disclosure.py.
        n_pos = int((df["dropped_out"] == 1).sum())
        if (len(df) < MIN_CELL_SIZE or df["dropped_out"].nunique() < 2
                or disclosure.is_small(n_pos, len(df) - n_pos)):
            return None, np.nan
        return roc_on_grid(df["dropped_out"], df["max_prob"], fpr_grid)

    tpr_ref,  auc_ref  = _interp_roc(preds[preds[group_col] == ref_val])
    tpr_comp, auc_comp = _interp_roc(preds[preds[group_col] == comp_val])
    value = (
        abroca(tpr_ref, tpr_comp, fpr_grid)
        if tpr_ref is not None and tpr_comp is not None
        else np.nan
    )
    return fpr_grid, tpr_ref, tpr_comp, value, auc_ref, auc_comp


def _abroca_hidden(preds: pd.DataFrame, group_col: str, ref_val, comp_val) -> bool:
    """True if either group's ROC curve would rest on 1..MIN_CELL_SIZE-1 students."""
    return disclosure.roc_hidden(preds, group_col, (ref_val, comp_val))


# ── Panel-drawing helpers ─────────────────────────────────────────────────────

_WS_BORDER_COLOR = "#000000"


# ── 2-panel plot functions ────────────────────────────────────────────────────

def _operational_model_threshold(subset: pd.DataFrame, faculty: str) -> int | None:
    """
    Empirical points_total threshold actually used for outreach at `faculty`:
    the lowest score observed among enrollments that were contacted
    (contact_fac_a / contact_fac_b == 1). Returns None if nobody was contacted.
    """
    contacted = subset.loc[subset[_FACULTY_CONTACT[faculty]] == 1, "points_total"]
    return int(contacted.min()) if len(contacted) else None


def _check_threshold_disclosable(subset: pd.DataFrame, faculty: str, thresh: int | None) -> None:
    """
    The empirical threshold is the lowest score among contacted students. If
    only 1..MIN_CELL_SIZE-1 contacted students sit at that score, printing it
    (in a table, or as a line in a figure) states one small group's value.
    Record the case rather than print it — the figure is still drawn, and the
    case is reported on the console. See src/disclosure.py.
    """
    if thresh is None:
        return
    at_min = int(((subset[_FACULTY_CONTACT[faculty]] == 1) & (subset["points_total"] == thresh)).sum())
    if disclosure.is_small(at_min):
        small_cells_report.record(
            kind="contact_threshold", where=faculty,
            what=(f"the empirical contact threshold of {thresh} points is the lowest score "
                  f"among contacted students, so printing it states that group's value"),
            counts={"contacted students at that score": at_min},
            artefacts=["descriptive/pilot_snapshot_recalibration.tex",
                       "figures/output/pilot_points_profile.pdf"],
        )


def _merge_points_bins(subset: pd.DataFrame, boundaries: list[int]) -> list[dict]:
    """
    points_total bins (0-15) under disclosure control.

    A bin whose students, dropouts or non-dropouts number 1..MIN_CELL_SIZE-1
    is merged with its smaller neighbour until every bin passes — nothing is
    hidden, so nothing can be recovered by subtracting the shown bins from the
    faculty total. Merges never cross a threshold in `boundaries` (a bin starts
    at each): other outputs publish counts at or above those thresholds, and a
    merged bin that straddled one could be split back apart by subtraction.
    Stops with an error if a whole stretch between thresholds still fails.
    Empty bins are kept as they are, with no rate.
    """
    g = subset.groupby("points_total")[TARGET].agg(["count", "sum"])
    bins = [{"lo": p, "hi": p, "n": int(g["count"].get(p, 0)), "k": int(g["sum"].get(p, 0))}
            for p in range(16)]

    def bad(b):
        ## An accepted bin is one no merge can rescue — it has been recorded and
        ## is left as it is. Without this the loop would pick it again forever.
        return not b.get("accepted") and disclosure.is_small(b["n"], b["k"], b["n"] - b["k"])

    def seg(b):   # which stretch between thresholds a bin sits in
        return sum(b["lo"] >= t for t in boundaries)

    while True:
        failing = [i for i, b in enumerate(bins) if bad(b)]
        if not failing:
            return bins
        i = failing[0]
        mates = [j for j in (i - 1, i + 1) if 0 <= j < len(bins) and seg(bins[j]) == seg(bins[i])]
        if not mates:
            b = bins[i]
            small_cells_report.record(
                kind="points_bin", where=f"points {b['lo']}--{b['hi']}",
                what=("a whole stretch between two published thresholds holds a small group, "
                      "and merging across a threshold could be undone by subtraction, so the "
                      "bin is drawn as it stands"),
                counts={"students": b["n"], "dropouts": b["k"],
                        "non-dropouts": b["n"] - b["k"]},
                artefacts=["figures/output/pilot_points_profile.pdf"],
            )
            bins[i] = {**b, "accepted": True}
            continue
        j = min(mates, key=lambda m: (bins[m]["n"] == 0, bins[m]["n"]))
        a, b = sorted((i, j))
        bins[a:b + 1] = [{"lo": bins[a]["lo"], "hi": bins[b]["hi"],
                          "n": bins[a]["n"] + bins[b]["n"], "k": bins[a]["k"] + bins[b]["k"]}]


# ── Enrollment breakdown ──────────────────────────────────────────────────────

def _breakdown_from_preds(sub: pd.DataFrame) -> dict:
    n = len(sub)
    if n == 0:
        return {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "n": 0, "dropout_rate": 0.0}
    tp = int(((sub["ever_flagged"] == 1) & (sub["dropped_out"] == 1)).sum())
    fp = int(((sub["ever_flagged"] == 1) & (sub["dropped_out"] == 0)).sum())
    fn = int(((sub["ever_flagged"] == 0) & (sub["dropped_out"] == 1)).sum())
    tn = int(((sub["ever_flagged"] == 0) & (sub["dropped_out"] == 0)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": n,
            "dropout_rate": (tp + fn) / n}


def _breakdown_exact(n_total: int, n_dropout: int, n_flagged: int, n_tp: int) -> dict:
    """TP/FP/FN/TN from exact counts (the metric dicts carry n_tp since disclosure control)."""
    fp = n_flagged - n_tp
    return {"tp": n_tp, "fp": fp, "fn": n_dropout - n_tp, "tn": (n_total - n_dropout) - fp,
            "n": n_total, "dropout_rate": n_dropout / n_total if n_total > 0 else 0.0}


def operational_model_real_breakdown(subset: pd.DataFrame, faculty: str) -> dict:
    """
    The operational rule-based model at its real threshold: the students the
    pilot actually contacted (the faculty's contact flag), as TP/FP/FN/TN.

    THE one computation — the pilot breakdown figure and
    pilot_snapshot_recalibration.tex both call it. The table used to rebuild the
    operating point as `points_total >= lowest score among the contacted`,
    which also flags every uncontacted student at or above that score, so the
    table and the figure disagreed. The contacted set is what the pilot did,
    so it is the operating point both report.
    """
    m = _contact_recall(subset, col=_FACULTY_CONTACT[faculty], score_col="points_total")
    return _breakdown_exact(m["n_enrollments"], m["n_dropout"], m["n_flagged"], m["n_tp"])


def _check_threshold_band(subset: pd.DataFrame, real_flag, recal_preds: pd.DataFrame, what: str) -> None:
    """
    The outcome-breakdown figure shows a rule at its real threshold and the
    recalibrated figure the same rule at τ. Both flag `score >= threshold`, so
    the flagged sets are nested, and subtracting the two bars' cells gives the
    dropouts and non-dropouts between the thresholds. Record the case if either
    is 1..MIN_CELL_SIZE-1; the figures are still drawn, and the case is
    reported on the console.
    """
    recal = recal_preds.set_index("enrollment")["ever_flagged"].reindex(subset["enrollment"]).fillna(0)
    band = np.asarray(real_flag, dtype=bool) != (recal.to_numpy() == 1)
    do = subset[TARGET].to_numpy() == 1
    n_do, n_nd = int((band & do).sum()), int((band & ~do).sum())
    if disclosure.is_small(n_do, n_nd):
        small_cells_report.record(
            kind="threshold_band", where=what,
            what=("the outcome-breakdown figure draws this rule at its real threshold and the "
                  "recalibrated figure at tau; the flagged sets are nested, so differencing the "
                  "two bars gives the students between the thresholds"),
            counts={"dropouts in the band": n_do, "non-dropouts in the band": n_nd},
            artefacts=["figures/output/pilot_breakdown_faculty_a.pdf"],
        )
