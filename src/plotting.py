"""
Figure content for the paper figures that are prepared from the pipeline's
predictions: the series behind the performance-by-period and dropout-timing
figures, the outcome breakdowns, operating points and ABROCA values behind the
model-comparison figure, and the shared palette and style. The figures
themselves are drawn by figures/scripts/draw_*.py from their prepared input.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")  # renamed in NumPy 2.0
import pandas as pd

from paths.paths import HELVETICA, RESULTS_PATH
from src import paper_style
from src import disclosure
from src.display import MODEL_LABELS
from src.config import (
    DROPOUT_TIMING_YEARS, MAIN_TEST_YEAR, MAX_ML_FS, MAX_ML_MODEL, MIN_CELL_SIZE, RANDOM_SEED,
)
from src.data import derive_danish_nationality
from src.metrics import ABROCA_N_GRID, abroca, conservative_threshold, percentile_ci, roc_on_grid

_MIN_CELL = MIN_CELL_SIZE   # local alias used in SDC guards throughout this file

# ── Style ─────────────────────────────────────────────────────────────────────

## The paper style, shared with every figure (src/paper_style.py).
paper_style.apply_style()

# ── Palette ───────────────────────────────────────────────────────────────────

_COLORS = [paper_style.METHOD_COLOR[m] for m in ["fasterrisk", "logistic_regression", "xgboost", "mlp"]]  # one colour per method (paper_style)
_RULE_COLOR = "#D55E00"   # vermillion — credits_policy; matches _ISO_COLOR
_MARKERS    = [paper_style.METHOD_MARKER[m] for m in ["fasterrisk", "logistic_regression", "xgboost", "mlp"]]

# Breakdown chart: dropout = orange (Wong), non-dropout = sky blue (Wong)
_DROPOUT_COLOR    = "#E69F00"   # flagged dropouts (TP)
_NONDROPOUT_COLOR = "#56B4E9"   # flagged non-dropouts (FP)
_DROPOUT_LIGHT    = "#F8E3B3"   # unflagged dropouts (FN) — orange at ~30 % alpha on white
_NONDROPOUT_LIGHT = "#CCE9F8"   # unflagged non-dropouts (TN) — sky blue at ~30 % alpha on white
_WORKSHOP_COLOR   = "#009E73"   # bluish green (Wong) — workshop model
_ISO_COLOR        = "#D55E00"   # vermillion (Wong) — credits-policy curve

_CONFIG_ORDER = ["a_credits", "b_credits_performance", "c_plus_enrolment",
                 "d_plus_lms", "e_plus_sensitive"]
_ML_MODELS    = ["fasterrisk", "logistic_regression", "xgboost", "mlp"]

# ABROCA feature × model grid: columns ordered by model complexity
_ABROCA_GRID_COLS = ["fasterrisk", "logistic_regression", "xgboost", "mlp"]

# Pairwise and triple intersectional comparisons for plot_abroca_heatmap_intersectional.
# Each entry: (row_label, ref_conditions, comp_conditions).
# ref/comp are {col: val} dicts; both must be satisfied simultaneously.
# ABROCA is unsigned (src/metrics.abroca): how far apart the two ROC curves lie.
#
# Structure: 3 concordant pairwise (both attrs same direction: ref=ref×ref, comp=comp×comp),
#            3 discordant pairwise (attrs cross-cut: ref=ref×comp, comp=comp×ref),
#            1 triple concordant (all three attrs differ in same direction).
# "Comparison" and "reference" are the binary axes' own: Female/Male,
# Non-Danish/Danish, Older/Younger (fairness_audit._BINARY_SUBGROUPS), so the
# labels read in the same direction as every other output.
_N_CONCORDANT_PAIRWISE = 3
_N_DISCORDANT_PAIRWISE = 3
_N_TRIPLE              = 1
_INTERSECT_ROW_SPECS: list[tuple[str, dict, dict]] = [
    # ── concordant pairwise (compounded) ─────────────────────────────────────
    (
        "Female×Non-Danish vs Male×Danish",
        {"female": 0, "danish_nationality": 1},   # ref : Male × Danish
        {"female": 1, "danish_nationality": 0},   # comp: Female × Non-Danish
    ),
    (
        "Female×Older vs Male×Younger",
        {"female": 0, "older_for_cohort": 0},      # ref : Male × Younger
        {"female": 1, "older_for_cohort": 1},      # comp: Female × Older
    ),
    (
        "Non-Danish×Older vs Danish×Younger",
        {"danish_nationality": 1, "older_for_cohort": 0},  # ref : Danish × Younger
        {"danish_nationality": 0, "older_for_cohort": 1},  # comp: Non-Danish × Older
    ),
    # ── discordant pairwise (cross-cutting) ──────────────────────────────────
    (
        "Female×Danish vs Male×Non-Danish",
        {"female": 0, "danish_nationality": 0},   # ref : Male × Non-Danish
        {"female": 1, "danish_nationality": 1},   # comp: Female × Danish
    ),
    (
        "Female×Younger vs Male×Older",
        {"female": 0, "older_for_cohort": 1},      # ref : Male × Older
        {"female": 1, "older_for_cohort": 0},      # comp: Female × Younger
    ),
    (
        "Non-Danish×Younger vs Danish×Older",
        {"danish_nationality": 1, "older_for_cohort": 1},  # ref : Danish × Older
        {"danish_nationality": 0, "older_for_cohort": 0},  # comp: Non-Danish × Younger
    ),
    # ── triple concordant ─────────────────────────────────────────────────────
    (
        "Female×Non-Danish×Older vs Male×Danish×Younger",
        {"female": 0, "danish_nationality": 1, "older_for_cohort": 0},  # ref : Male×D×Younger
        {"female": 1, "danish_nationality": 0, "older_for_cohort": 1},  # comp: Female×ND×Older
    ),
]
_MODEL_DISPLAY = {
    "fasterrisk":          "FasterRisk",
    "logistic_regression": "Logistic Regression",
    "xgboost":             "XGBoost",
    "mlp":                 MODEL_LABELS["mlp"],   # one name, in src/display.py
}
_ABROCA_GRID_COL_LABELS = _MODEL_DISPLAY
_ABROCA_GRID_ROW_LABELS = {
    "a_credits":             "a — Credits only",
    "b_credits_performance": "b — + Performance",
    "c_plus_enrolment":      "c — + Enrollment",
    "d_plus_lms":            "d — + LMS",
    "e_plus_sensitive":      "e — + Sensitive",
}

# Metric column names and display labels for each level
_METRIC_COLS = {
    "enrollment": {
        "recall":    "enr_recall",
        "precision": "enr_precision",
        "accuracy":  "enr_accuracy",
        "roc_auc":   "roc_auc",
    },
    "ews_row": {
        "recall":    "row_recall",
        "precision": "row_precision",
        "accuracy":  "row_accuracy",
        "roc_auc":   "row_roc_auc",
    },
}

_METRIC_LABELS = {
    # enrollment-level
    "enr_recall":    "Dropout detection rate",
    "enr_precision": "Share of flags that were dropouts",
    "enr_accuracy":  "Enrollment-level accuracy score",
    "roc_auc":       "Enrollment-level ROC AUC",
    # ews-row-level
    "row_recall":    "Dropout detection rate (activation level)",
    "row_precision": "Share of flags that were dropouts (activation level)",
    "row_accuracy":  "Accuracy score (activation level)",
    "row_roc_auc":   "ROC AUC (activation level)",
}

_LEVEL_SUFFIX = {
    "enrollment": "(enrollment-level)",
    "ews_row":    "(activation level — model diagnostic)",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

_CORNERS = [
    (0.02, 0.97, "left",  "top"),
    (0.98, 0.97, "right", "top"),
    (0.02, 0.03, "left",  "bottom"),
    (0.98, 0.03, "right", "bottom"),
]


# ── Main comparison bar chart ─────────────────────────────────────────────────

_WS_BORDER_COLOR = "#000000"   # black — workshop bar border


# ── Enrollment outcome breakdown (100% stacked bars) ─────────────────────────

def _enr_breakdown(sub: pd.DataFrame) -> dict:
    """Confusion matrix counts from an enrollment-level predictions subset."""
    n = len(sub)
    if n == 0:
        return {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "n": 0, "dropout_rate": 0.0}
    tp = int(((sub["ever_flagged"] == 1) & (sub["dropped_out"] == 1)).sum())
    fp = int(((sub["ever_flagged"] == 1) & (sub["dropped_out"] == 0)).sum())
    fn = int(((sub["ever_flagged"] == 0) & (sub["dropped_out"] == 1)).sum())
    tn = int(((sub["ever_flagged"] == 0) & (sub["dropped_out"] == 0)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": n,
            "dropout_rate": (tp + fn) / n}


def _ws_breakdown(ws_row: pd.Series) -> dict:
    """
    Confusion matrix from a workshop_results aggregate row, from its exact
    counts. Never rebuild TP from enr_precision: it is rounded to 3 decimals,
    so precision × n_flagged can be off by one.
    """
    if "n_tp_enrollments" not in ws_row.index or pd.isna(ws_row["n_tp_enrollments"]):
        raise ValueError("workshop results row has no n_tp_enrollments — re-run "
                         "src/workshop_evaluation.py; TP is not rebuilt from a rounded precision.")
    n         = int(ws_row["n_enrollments"])
    n_dropout = int(ws_row["n_dropout_enrollments"])
    n_flagged = int(ws_row["n_flagged_enrollments"])
    tp        = int(ws_row["n_tp_enrollments"])
    fp = n_flagged - tp
    fn = n_dropout - tp
    tn = (n - n_dropout) - fp
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": n,
            "dropout_rate": n_dropout / n}


# ── PR curve with fixed contact-rate budget lines ────────────────────────────


def _raw_credits_enr(
    raw_df: pd.DataFrame,
    test_year: int,
) -> pd.DataFrame:
    """
    Enrollment-level credits-policy scoring using raw max(ects_behind).

    Returns enr_df with max_prob = max(ects_behind) per enrollment and
    dropped_out. The ≥60 ECTS operating point is located by thresholding
    max_prob >= 60 directly — no training data or model fitting needed.
    """
    test = raw_df[raw_df["start_year"] == test_year].copy()
    return (
        test.groupby("enrollment")
        .agg(max_prob=("ects_behind", "max"), dropped_out=("dropout", "max"))
        .reset_index()
    )


def _workshop_underlying_name() -> tuple[str | None, str | None]:
    """Return (underlying_model, feature_set) of the Participatory model, or (None, None)."""
    from src.workshop_rules import participatory_config
    return participatory_config(RESULTS_PATH)


def _load_workshop_underlying(preds_df: pd.DataFrame, test_year: int) -> pd.DataFrame | None:
    """
    Return per-enrollment predictions for the workshop model's underlying ML model,
    identified via workshop_rules.participatory_config().
    Returns None if config not found or predictions unavailable.
    """
    underlying_model, underlying_fs = _workshop_underlying_name()
    if underlying_model and underlying_fs:
        sub = preds_df[
            (preds_df["test_year"] == test_year) &
            (preds_df["model"] == underlying_model) &
            (preds_df["feature_set"] == underlying_fs)
        ]
        if len(sub) > 0 and "max_prob" in sub.columns and not sub["max_prob"].isna().all():
            return sub, underlying_model, underlying_fs
    return None, None, None


def _point_at_budget(enr_df: pd.DataFrame, flag_rate: float) -> tuple[float, float]:
    """(recall, precision) when exactly flag_rate × n enrollments are flagged.

    Ties in max_prob at the cutoff are broken with a fixed random seed so
    exactly n_target enrollments are flagged, regardless of tie clusters
    (e.g. coarse integer risk scores). Without this, two models nominally
    compared "at the same flag rate" can end up with different actual
    flagged counts, which can make recall and precision appear to trade
    off against each other in a way that's not otherwise possible when
    the flagged count and population are truly identical.
    """
    n = len(enr_df)
    n_dropout = int((enr_df["dropped_out"] == 1).sum())
    if n_dropout == 0 or "max_prob" not in enr_df.columns:
        return 0.0, 0.0
    n_target = max(1, round(flag_rate * n))
    rng      = np.random.RandomState(RANDOM_SEED)
    tiebreak = rng.random(n)
    ranked = pd.DataFrame({
        "max_prob":    enr_df["max_prob"].to_numpy(),
        "dropped_out": enr_df["dropped_out"].to_numpy(),
        "tiebreak":    tiebreak,
    }).sort_values(["max_prob", "tiebreak"], ascending=[False, False])
    top = ranked.iloc[:n_target]
    tp  = int((top["dropped_out"] == 1).sum())
    # Disclosure control: an operating point is a whole confusion matrix.
    if disclosure.operating_point_hidden(n, n_dropout, n_target, tp)["recall"]:
        return np.nan, np.nan
    return tp / n_dropout, tp / n_target


def _op_point_hidden(c: dict) -> bool:
    """True if a drawn operating point / breakdown bar rests on a cell of fewer than MIN_CELL_SIZE students."""
    return bool(disclosure.confusion_is_small(c["tp"], c["fp"], c["fn"], c["tn"]).any())


# ── PR curves + operating-point table ────────────────────────────────────────

_RULE_MARKERS   = ["P", "*", "X", "h"]
_RULE_COLORS    = ["#D55E00", "#FFC107", "#0072B2", "#009E73"]   # credits_policy, contact_fac_b, contact_fac_a, spare


# ── Model × feature-set heatmap ──────────────────────────────────────────────

_FS_XLABELS = ["credits", "+perf.", "+enroll.", "+LMS", "+sensitive"]

_MODEL_YLABELS = [
    "FasterRisk\n(very high)",
    "Logistic Regression\n(high)",
    "XGBoost\n(moderate)",
    "Neural network\n(low)",
]

_HEATMAP_METRICS = {
    "enrollment": {
        "enr_recall": {"title": "Dropout detection rate",   "vmin": 0.0, "vmax": 1.0},
        "roc_auc":    {"title": "Enrollment-level ROC AUC", "vmin": 0.5, "vmax": 1.0},
    },
    "ews_row": {
        "row_recall":  {"title": "Dropout detection rate (activation level)",   "vmin": 0.0, "vmax": 1.0},
        "row_roc_auc": {"title": "ROC AUC (activation level)",                  "vmin": 0.5, "vmax": 1.0},
    },
}


# ── Subgroup recall (enrollment-level only) ───────────────────────────────────

_SUBGROUP_COLS    = ["female", "danish_nationality", "older_for_cohort"]
_SUBGROUP_DISPLAY = {
    "female":             ("Male",          "Female"),
    "danish_nationality": ("Non-Danish",    "Danish"),
    "older_for_cohort":   ("Younger",       "Older"),
}
_SUBGROUP_COLORS  = ["#5C3D9E", "#E69F00"]


# ── FasterRisk scorecard ─────────────────────────────────────────────────────

_SCORE_POS_COLOR = "#56C4B4"   # teal  — risk-increasing features
_SCORE_NEG_COLOR = "#C4547A"   # pink  — risk-decreasing features


def _clean_feature_name(name: str) -> str:
    """Strip ColumnTransformer prefixes (num__, cat__) for display."""
    for prefix in ("num__", "cat__"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


_SCORECARD_ROW_H = 0.32   # inches per scorecard row; see plot_scorecard


# ── Publication figures ───────────────────────────────────────────────────────

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

    def _interp(df):
        # A ROC curve over too few dropouts (or non-dropouts) is those students'
        # ranks — not computed (disclosure control, src/disclosure.py).
        n_pos = int((df["dropped_out"] == 1).sum())
        if (len(df) < _MIN_CELL or df["dropped_out"].nunique() < 2
                or disclosure.is_small(n_pos, len(df) - n_pos)):
            return None, np.nan
        return roc_on_grid(df["dropped_out"], df["max_prob"], fpr_grid)

    tpr_ref,  auc_ref  = _interp(preds[preds[group_col] == ref_val])
    tpr_comp, auc_comp = _interp(preds[preds[group_col] == comp_val])
    value = (
        abroca(tpr_ref, tpr_comp, fpr_grid)
        if tpr_ref is not None and tpr_comp is not None
        else np.nan
    )
    return fpr_grid, tpr_ref, tpr_comp, value, auc_ref, auc_comp


# ── Performance-by-period figure ─────────────────────────────────────────────────────

## (column, reference value, comparison value, panel title) for the
## performance-by-period difference panels. Same convention as every fairness
## artefact (fairness_audit._BINARY_SUBGROUPS): comparison − reference, the
## reference group second.
_SUBGROUP_SPECS = [
    ("female",             0, 1, "Sex (Female − Male)"),
    ("danish_nationality", 1, 0, "Origin (Non-Danish − Danish)"),
    ("older_for_cohort",   0, 1, "Entry age (Older − Younger)"),
]


def performance_by_period_series(
    row_preds: pd.DataFrame,
    train_pos_rate: float,
    model: str,
    feature_set: str,
    test_year: int = MAIN_TEST_YEAR,
    n_boot: int = 500,
    max_activation: int | None = None,
) -> dict:
    """
    The performance-by-period figure's content — a replication of Blazkova et al.
    (2025) for the Participatory model, as series rather than a drawing.

    (a) Overall model performance: AUC, Accuracy, Recall, Precision, TNR by EWS period.
    (b) Female − Male metric differences.
    (c) Non-Danish − Danish metric differences.
    (d) Older − Younger (entry age relative to the level-and-cohort median).
    Differences are comparison − reference with the reference group second, as
    in every fairness artefact (fairness_audit._BINARY_SUBGROUPS).

    X-axis: the EWS activation itself, 1, 2, 3, ... — exact, where a nominal
    "months of enrollment" would put activation 1 at month 0 although students
    have been enrolled about six weeks (and spring starters longer).
    Period 0 is the "never evaluated" sentinel and is filtered out at load time;
    the guard below is belt-and-braces.
    CIs: 95% bootstrap over enrollments within each period.

    max_activation drops every later activation BEFORE anything is computed, so
    hidden_panels describes only the points that are drawn. Late periods hold
    fewer students, so cells there are more likely to be small.
    τ is set per period, so the cut does not move any point that is kept.

    Separated from the drawing so the figure can be rendered from the series
    alone (figures/scripts/draw_performance_by_period.py). Everything that reads
    a prediction happens here, including the bootstrap — which is why restyling
    the figure no longer re-runs 500 resamples per period.

    Returns {"activations", "metrics", "panels", "hidden_panels"}, where a panel holds
    one {y, lo, hi} per metric and a hidden point has already taken its band with
    it.
    """
    from sklearn.metrics import roc_auc_score as _roc_auc

    _FIG2_METRICS = ["auc", "accuracy", "recall", "precision", "tnr"]
    _FIG2_LABELS  = {
        "auc": "AUC", "accuracy": "Accuracy Score",
        "recall": "Recall", "precision": "Precision", "tnr": "TNR",
    }
    _FIG2_COLORS = {
        "auc":       "#000000",
        "accuracy":  "#E69F00",
        "recall":    "#CC79A7",
        "precision": "#009E73",
        "tnr":       "#0072B2",
    }
    _FIG2_MARKERS = ["o", "s", "D", "^", "v"]

    sub = row_preds[
        (row_preds["model"]       == model) &
        (row_preds["feature_set"] == feature_set) &
        (row_preds["test_year"]   == test_year)
    ].copy()
    if len(sub) == 0:
        ## Raise rather than return None. An empty filter (e.g. a display label
        ## such as "workshop_model" passed instead of an estimator name) would
        ## otherwise print a single easily missed line while the previous PDF
        ## sat on disk looking current. A figure that cannot be drawn has to stop
        ## the run; it must never leave yesterday's file in place unremarked.
        ## The message lists what IS available so the cause is readable without
        ## opening row_predictions.csv.
        available = (
            row_preds.loc[row_preds["test_year"] == test_year,
                          ["model", "feature_set"]]
            .drop_duplicates()
            .to_string(index=False)
        )
        raise ValueError(
            f"performance_by_period_series: no rows for {model} | {feature_set} | {test_year}. "
            f"row_predictions.csv holds these combinations for {test_year}:\n{available}"
        )

    target_col = "dropout" if "dropout" in sub.columns else "dropped_out"
    periods    = sorted(sub["period"].unique())
    periods    = [p for p in periods if p >= 1]       # exclude the period-0 sentinel
    if max_activation is not None:
        periods = [p for p in periods if p <= max_activation]
    sg_cols    = [col for col, *_ in _SUBGROUP_SPECS if col in sub.columns]

    def _agg(df: pd.DataFrame) -> pd.DataFrame:
        agg = {"max_prob": ("y_prob", "max"), "dropped_out": (target_col, "max")}
        for c in sg_cols:
            agg[c] = (c, "first")
        enr = df.groupby("enrollment").agg(**agg).reset_index()
        tau = conservative_threshold(enr["max_prob"].to_numpy(), train_pos_rate)
        enr["ever_flagged"] = (enr["max_prob"] >= tau).astype(int)
        return enr

    hidden_panels: set[str] = set()   # panels where disclosure control hid a point

    def _metrics(enr: pd.DataFrame, panel: str | None = None, raw: bool = False) -> dict:
        """
        Metrics for one group at one period. Disclosure control: AUC needs at
        least MIN_CELL_SIZE dropouts and non-dropouts; recall/precision/TNR/
        accuracy need every confusion cell to be 0 or >= MIN_CELL_SIZE (with the
        group's size they pin each other down). A hidden metric is NaN. See
        src/disclosure.py.

        `raw=True` skips disclosure control, for bootstrap draws only — they are
        never drawn, and suppressing inside a draw drops the draws whose counts
        swing low, which narrows the band exactly where few students remain.
        """
        if len(enr) == 0 or (not raw and len(enr) < _MIN_CELL):  # SDC: suppress small cells
            if len(enr) > 0 and panel:
                hidden_panels.add(panel)
            return {m: np.nan for m in _FIG2_METRICS}
        do, fl = enr["dropped_out"], enr["ever_flagged"]
        tp = int(((do == 1) & (fl == 1)).sum())
        fp = int(((do == 0) & (fl == 1)).sum())
        fn = int(((do == 1) & (fl == 0)).sum())
        tn = int(((do == 0) & (fl == 0)).sum())
        n  = len(enr)
        if do.nunique() < 2:          # AUC needs both classes; a draw may hold one
            auc = np.nan
        else:
            try:
                auc = float(_roc_auc(do, enr["max_prob"]))
            except ValueError:
                auc = np.nan
        out = {
            "auc":       auc,
            "accuracy":  (tp + tn) / n      if n > 0            else np.nan,
            "recall":    tp / (tp + fn)      if (tp + fn) > 0    else np.nan,
            "precision": tp / (tp + fp)      if (tp + fp) > 0    else np.nan,
            "tnr":       tn / (tn + fp)      if (tn + fp) > 0    else np.nan,
        }
        if raw:
            return out
        h = disclosure.operating_point_hidden(n, tp + fn, tp + fp, tp)
        for m in out:
            if h["auc"] if m == "auc" else h["recall"]:
                out[m] = np.nan
                if panel:
                    hidden_panels.add(panel)
        return out

    period_enrs = {p: _agg(sub[sub["period"] == p]) for p in periods}
    rng = np.random.default_rng(42)

    def _bootstrap(enr: pd.DataFrame) -> tuple[dict, dict]:
        lo, hi = {m: [] for m in _FIG2_METRICS}, {m: [] for m in _FIG2_METRICS}
        boot = {m: [] for m in _FIG2_METRICS}
        for _ in range(n_boot):
            s = enr.sample(n=len(enr), replace=True,
                           random_state=int(rng.integers(1_000_000)))
            for m, v in _metrics(s, raw=True).items():
                boot[m].append(v)
        for m in _FIG2_METRICS:
            lo[m], hi[m] = percentile_ci(boot[m])
        return lo, hi

    # ── Overall metrics ────────────────────────────────────────────────────────
    overall    = {m: [] for m in _FIG2_METRICS}
    overall_lo = {m: [] for m in _FIG2_METRICS}
    overall_hi = {m: [] for m in _FIG2_METRICS}
    for p in periods:
        enr = period_enrs[p]
        point = _metrics(enr, panel="a")
        lo, hi = _bootstrap(enr)
        for m in _FIG2_METRICS:
            overall[m].append(point[m])
            # a hidden point takes its band with it
            overall_lo[m].append(lo[m] if not np.isnan(point[m]) else np.nan)
            overall_hi[m].append(hi[m] if not np.isnan(point[m]) else np.nan)

    # ── Subgroup differences ───────────────────────────────────────────────────
    subgroup_results = []
    for col, ref_val, comp_val, title in _SUBGROUP_SPECS:
        if col not in sub.columns:
            subgroup_results.append(None)
            continue
        diffs    = {m: [] for m in _FIG2_METRICS}
        diffs_lo = {m: [] for m in _FIG2_METRICS}
        diffs_hi = {m: [] for m in _FIG2_METRICS}
        for p in periods:
            enr      = period_enrs[p]
            ref_m    = _metrics(enr[enr[col] == ref_val],  panel=title)
            comp_m   = _metrics(enr[enr[col] == comp_val], panel=title)
            for m in _FIG2_METRICS:
                d = comp_m[m] - ref_m[m] if not (np.isnan(comp_m[m]) or np.isnan(ref_m[m])) else np.nan
                diffs[m].append(d)
            # bootstrap diff CIs
            boot_d = {m: [] for m in _FIG2_METRICS}
            for _ in range(n_boot):
                s = enr.sample(n=len(enr), replace=True,
                               random_state=int(rng.integers(1_000_000)))
                bv_r = _metrics(s[s[col] == ref_val], raw=True)
                bv_c = _metrics(s[s[col] == comp_val], raw=True)
                for m in _FIG2_METRICS:
                    boot_d[m].append(bv_c[m] - bv_r[m])
            for m in _FIG2_METRICS:
                lo, hi = percentile_ci(boot_d[m])
                shown = not np.isnan(diffs[m][-1])     # a hidden point takes its band with it
                diffs_lo[m].append(lo if shown else np.nan)
                diffs_hi[m].append(hi if shown else np.nan)
        subgroup_results.append({"title": title, "diffs": diffs,
                                 "lo": diffs_lo, "hi": diffs_hi})

    return {
        "activations":   [int(p) for p in periods],
        "metrics":       _FIG2_METRICS,
        "metric_labels": _FIG2_LABELS,
        "panels": (
            [{"letter": "a", "title": None, "kind": "overall",
              "series": {m: {"y": overall[m], "lo": overall_lo[m], "hi": overall_hi[m]}
                         for m in _FIG2_METRICS}}]
            + [None if sg is None else
               {"letter": chr(ord("b") + i), "title": sg["title"], "kind": "difference",
                "series": {m: {"y": sg["diffs"][m], "lo": sg["lo"][m], "hi": sg["hi"][m]}
                           for m in _FIG2_METRICS}}
               for i, sg in enumerate(subgroup_results)]
        ),
        "hidden_panels": sorted(hidden_panels),
    }


def dropout_timing_layers(
    raw_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    ects_threshold: int = 60,
    target: str = "dropout",
    max_months: float = 60,
    start_years: list | None = None,
    include_max_ml: bool = False,
) -> dict:
    """
    The dropout-timing figure's content: four panels of cumulative-dropout
    curves, already binned and suppressed, with no student-level anything left.

    Panels: (a) overall  (b) sex  (c) nationality  (d) entry age. Level is not
    a panel: it is not a fairness axis.
    Three layers per group — all dropouts, those the credits policy ever caught,
    those the Participatory model ever flagged — each as a step curve, a recall
    level, and the legend entry that names it. `include_max_ml` adds a fourth,
    those Max ML (MAX_ML_MODEL on MAX_ML_FS) ever flagged — the
    dropout_timing_maxml variant of the figure.

    Separated from the drawing so the figure can be rendered from the curves
    alone (figures/scripts/draw_dropout_timing.py). Everything that reads a
    student record happens here.

    `start_years` defaults to DROPOUT_TIMING_YEARS, never to every cohort: the
    Participatory model scores only test cohorts, so a training-cohort dropout in
    the denominator would count as missed by construction, while the credits
    policy, computed from ects_behind, could still catch it.

    Returns {"panels": [...], "layers": [...], "max_months": ..., "n_dropout": ...,
    "start_years": [...]},
    where a layer carries `x`/`y` arrays, `drawn` (False for one that may not be
    shown but still needs its legend entry), and `hline` (its recall).
    """
    d = raw_df.copy()
    d["ews_date"]    = pd.to_datetime(d["ews_date"])
    d["status_change_date"] = pd.to_datetime(d["status_change_date"])

    ws_model, ws_fs = _workshop_underlying_name()
    if ws_model is None or ws_fs is None:
        raise ValueError("Workshop model not found — run workshop selection first.")

    ws_preds = preds_df[
        (preds_df["model"] == ws_model) & (preds_df["feature_set"] == ws_fs)
    ]
    start_years = sorted(DROPOUT_TIMING_YEARS if start_years is None else start_years)
    ws_preds = ws_preds[ws_preds["test_year"].isin(start_years)]
    d = d[d["start_year"].isin(start_years)]

    ## Each flagging model layer: (colour key in panel a, line style, legend
    ## name in panel a, legend name in the subgroup panels, its predictions).
    scored = [("participatory", "dotted", "Participatory", "Participatory",
               ws_model, ws_fs, ws_preds)]
    if include_max_ml:
        mx_preds = preds_df[(preds_df["model"] == MAX_ML_MODEL)
                            & (preds_df["feature_set"] == MAX_ML_FS)
                            & (preds_df["test_year"].isin(start_years))]
        scored.append(("max_ml", "dashdot", "Max ML", "Max ML",
                       MAX_ML_MODEL, MAX_ML_FS, mx_preds))
    flagged_by = {key: set(p.loc[p["ever_flagged"] == 1, "enrollment"].unique())
                  for key, *_rest, p in scored}

    ## Age relative to the level-and-cohort median, derived once in
    ## src/data.py. It replaces a pooled median computed here: pooled, the
    ## split mostly reproduces BA vs MA, because a MA student starts after a
    ## three-year bachelor.
    d = derive_danish_nationality(d)

    def _enroll_start(enr_df: pd.DataFrame) -> pd.Series:
        if "study_start" in enr_df.columns:
            return pd.to_datetime(enr_df["study_start"])
        return (pd.to_datetime(enr_df["ews_date"])
                - pd.to_timedelta(enr_df["days_observed"], unit="D"))

    # Disclosure control: a CDF with one step per dropout shows each student's
    # dropout month. The curves instead rise once per half-year (the EWS
    # rhythm), and a half-year holding 1..MIN_CELL_SIZE-1 dropouts is merged
    # into the next one, so every visible rise is at least MIN_CELL_SIZE
    # students. See src/disclosure.py.
    _BIN_MONTHS = 6

    def _binned_cdf(months: np.ndarray, n_total: int) -> tuple:
        edges = list(np.arange(0, max_months + _BIN_MONTHS, _BIN_MONTHS, dtype=float))
        counts = list(np.histogram(np.clip(months, 0, edges[-1] - 1e-9), bins=edges)[0])
        i = 0
        while i < len(counts):                    # bin i spans [edges[i], edges[i+1])
            if disclosure.is_small(counts[i]) and len(counts) > 1:
                if i + 1 < len(counts):           # merge into the next half-year
                    counts[i + 1] += counts[i]
                    del counts[i], edges[i + 1]
                else:                             # the last one into the previous
                    counts[i - 1] += counts[i]
                    del counts[i], edges[i]
                continue
            i += 1
        y = np.concatenate([[0.0], np.cumsum(counts) / n_total * 100])
        return np.array(edges), y

    def _months(enr_df: pd.DataFrame) -> np.ndarray:
        return ((pd.to_datetime(enr_df["status_change_date"]) - _enroll_start(enr_df))
                .dt.days.div(30.44).values)

    def _all_cdf(enr_df: pd.DataFrame, n_total: int) -> tuple:
        return _binned_cdf(_months(enr_df), n_total)

    def _flag_cdf(enr_df: pd.DataFrame, flagged_ids: set, n_total: int) -> tuple:
        flagged_enr = enr_df[enr_df["enrollment"].isin(flagged_ids)]
        months = _months(flagged_enr)
        n_flag = len(months)
        recall = n_flag / n_total * 100 if n_total > 0 else 0.0
        x, y = _binned_cdf(months, n_total)
        return x, y, recall, n_flag

    def _layer_shown(n_flag: int, n_total: int) -> bool:
        """A layer and its recall rest on the caught AND the missed dropouts."""
        return n_flag >= MIN_CELL_SIZE and not disclosure.rate_is_small(n_flag, n_total)

    dropout_ids = set(d.loc[d[target] == 1, "enrollment"].unique())
    n_dropout   = len(dropout_ids)
    if n_dropout == 0:
        raise ValueError("No dropout enrollments found for the selected years.")

    ## Every layer must describe the same students. A dropout a model never
    ## scored would count as missed by it while the credits policy could still
    ## catch it — a training cohort, or a test year missing from a partial
    ## evaluation run. Refuse rather than draw a lower recall.
    for _key, _style, name, _short, model, fs, preds in scored:
        unscored = dropout_ids - set(preds["enrollment"])
        if unscored:
            years = sorted(d.loc[d["enrollment"].isin(unscored), "start_year"].unique())
            raise ValueError(
                f"{len(unscored)} dropout enrollments (start years {years}) have no "
                f"{model} / {fs} prediction, so the {name} layer would count "
                "them as missed. Pass start_years covering scored cohorts only, or "
                "re-run the evaluation for the missing years."
            )

    layers: list[dict] = []

    def add(panel, colour, style, label, x, y, hline, drawn=True, alpha=None,
            zorder=2, fill_zorder=None):
        ## The fill is not simply one below its curve: in the subgroup panels
        ## both fills sit UNDER every curve, so the two groups' curves stay
        ## readable where their shaded areas overlap. Carried explicitly rather
        ## than derived, because that difference is the whole point.
        layers.append({"panel": panel, "colour": colour, "style": style, "label": label,
                       "x": np.asarray(x), "y": np.asarray(y), "hline": hline,
                       "drawn": drawn, "fill_alpha": alpha, "zorder": zorder,
                       "fill_zorder": fill_zorder})

    # ── Panel a: overall ──────────────────────────────────────────────────────
    enr_all = d[d["enrollment"].isin(dropout_ids)].drop_duplicates("enrollment").copy()
    cp_flagged_all = set(
        d[(d["enrollment"].isin(dropout_ids)) & (d["ects_behind"] >= ects_threshold)]
        ["enrollment"].unique()
    )

    x_all, y_all                = _all_cdf(enr_all, n_dropout)
    x_cp, y_cp, cp_recall, cp_n = _flag_cdf(enr_all, cp_flagged_all, n_dropout)

    add(0, "all", "solid", f"all dropouts  (n={n_dropout:,})", x_all, y_all,
        None, alpha=0.08, zorder=2, fill_zorder=1)
    if _layer_shown(cp_n, n_dropout):
        add(0, "credits_policy", "dashed", f"Credits policy  (recall {cp_recall:.0f}%)",
            x_cp, y_cp, cp_recall, alpha=0.15, zorder=4, fill_zorder=3)
    for k, (key, style, name, _short, *_rest) in enumerate(scored):
        x, y, rec, n = _flag_cdf(enr_all, flagged_by[key] & dropout_ids, n_dropout)
        if _layer_shown(n, n_dropout):
            add(0, key, style, f"{name}  (recall {rec:.0f}%)",
                x, y, rec, alpha=0.15, zorder=5 + k, fill_zorder=4 + k)

    # ── Panels b–d: subgroups ─────────────────────────────────────────────────
    subgroups = [
        ("b", "Sex",         "female",             {0: "Male",       1: "Female"}),
        ("c", "Nationality", "danish_nationality", {0: "Non-Danish", 1: "Danish"}),
        ("d", "Entry age",   "older_for_cohort",   {0: "Younger",    1: "Older"}),
    ]
    ## One height for every panel letter, although panel a's legend is one line
    ## shorter, so the letters sit on one line across the figure.
    _LETTER_Y = 1.40
    panels = [{"panel": 0, "letter": "a", "letter_y": _LETTER_Y, "ylabel": True}]

    for p_idx, (panel_label, _title, col, labels) in enumerate(subgroups, start=1):
        panels.append({"panel": p_idx, "letter": panel_label, "letter_y": _LETTER_Y,
                       "ylabel": False})
        if col not in d.columns:
            continue

        for g_idx, (grp_val, grp_label) in enumerate(labels.items()):
            grp_dropout_ids = set(
                d.loc[(d[col] == grp_val) & (d[target] == 1), "enrollment"].unique()
            )
            n_grp = len(grp_dropout_ids)
            if n_grp < MIN_CELL_SIZE:
                continue

            grp_enr    = d[d["enrollment"].isin(grp_dropout_ids)].drop_duplicates("enrollment").copy()
            cp_grp_ids = set(
                d[(d["enrollment"].isin(grp_dropout_ids)) & (d["ects_behind"] >= ects_threshold)]
                ["enrollment"].unique()
            )

            x_all, y_all = _all_cdf(grp_enr, n_grp)
            colour = f"subgroup{g_idx}"

            ## The all-dropout curve is the faint background the other two are
            ## read against, and carries no legend entry of its own.
            add(p_idx, colour, "background", "", x_all, y_all, None, zorder=2)

            ## (style, ids flagged, legend name, curve z, fill z). Every fill
            ## sits below every curve (z >= 3).
            grp_layers = [("dashed", cp_grp_ids, "credits", 3, 1)]
            grp_layers += [(style, flagged_by[key] & grp_dropout_ids, short, 4 + k, 2)
                           for k, (key, style, _n, short, *_r) in enumerate(scored)]
            for style, ids, name, z, fz in grp_layers:
                x, y, rec, n_flag = _flag_cdf(grp_enr, ids, n_grp)
                if _layer_shown(n_flag, n_grp):
                    add(p_idx, colour, style, f"{grp_label} — {name}  ({rec:.0f}%)",
                        x, y, rec, alpha=0.10, zorder=z, fill_zorder=fz)
                else:
                    ## Legend entry only: the layer may not be drawn, but the
                    ## reader still has to be told the group exists.
                    add(p_idx, colour, style,
                        f"{grp_label} — {name}  (n{disclosure.SUPPRESSED})",
                        [], [], None, drawn=False)

    return {"panels": panels, "layers": layers, "max_months": max_months,
            "n_dropout": n_dropout, "start_years": start_years}
