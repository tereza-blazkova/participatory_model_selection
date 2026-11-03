"""
What happens to the rule-based baselines (credits policy, operational
rule-based model) if they are thresholded the same way as the ML models?

Background: on the October 2023 snapshot (pilot_snapshot), the ML models are
calibrated with `_enrollment_quantile_threshold` in src/metrics.py — the
score threshold is picked so the flagged share of enrollments comes as close
as possible to, without exceeding, train_pos_rate (the training
enrollment-level dropout rate, treated as an intervention budget). The two
rule-based baselines instead use their own real-world thresholds:
  - credits policy:            fixed national-law rule, ects_behind >= 60
  - operational rule-based model: the students the pilot actually contacted
                                   (operational_model_real_breakdown() in
                                   src/pilot_snapshot_comparison.py — the same
                                   computation as the pilot breakdown figure);
                                   tau is the lowest points_total among them
                                   (_operational_model_threshold())

Faculty A only. Faculty B's outreach threshold cannot be recovered reliably
from who was contacted, so the paper reports only threshold-free measures for
Faculty B (AUC, ABROCA), and this table — which is about thresholds — has no
Faculty B rows.

This script has two parts:

  1. Threshold comparison — for Faculty A, computes what threshold
     `_enrollment_quantile_threshold` would pick for ects_behind and for
     points_total (matching train_pos_rate), and compares it to the real
     threshold each rule actually uses, along with the resulting flag rate /
     recall / precision under both.

  2. Recalibrated enrollment breakdown — regenerates the pilot_snapshot enrollment
     breakdown figure with credits policy and the operational model recomputed
     under the ML-style calibrated threshold instead of their real one
     (Participatory and Max ML are unchanged, since they already use this
     calibration). This isolates how much of the ML-vs-baseline gap is due
     to discriminative power vs. differing threshold philosophy.

Run from project root (after python src/evaluation.py and, optionally,
python -m src.workshop_evaluation):
    python src/check_rule_based_recalibration.py

Writes:
    results/descriptive/pilot_snapshot_recalibration.tex
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from paths.paths import PROJECT_ROOT, RESULTS_PATH, PILOT_SNAPSHOT_PARQUET
from src.workshop_rules import _RULE_BASED_MODELS, participatory_config
from src.config import TARGET
from src.metrics import _enrollment_quantile_threshold, enrollment_predictions_df
from src.pilot_snapshot_comparison import (
    FACULTIES,
    _FACULTY_CONTACT,
    _FACULTY_DISPLAY,
    _breakdown_from_preds,
    _load_metadata,
    _check_threshold_disclosable,
    _operational_model_threshold,
    evaluate_faculty,
    operational_model_real_breakdown,
)
from src import disclosure
from src.data import load_pilot_snapshot

OUT_DIR = RESULTS_PATH / "descriptive"
OUT_DIR.mkdir(parents=True, exist_ok=True)

## The faculties this table covers: Faculty A alone (see the module docstring).
TABLE_FACULTIES = [f for f in FACULTIES if _FACULTY_DISPLAY[f] == "Faculty A"]


def _fmt(x, decimals: int = 3) -> str:
    return "---" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{decimals}f}"


def _recall_precision_flagrate(preds: pd.DataFrame) -> dict:
    return _stats_from_counts(_breakdown_from_preds(preds))


def _stats_from_counts(c: dict) -> dict:
    n = c["n"]
    recall    = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) > 0 else float("nan")
    precision = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) > 0 else float("nan")
    flag_rate = (c["tp"] + c["fp"]) / n if n > 0 else float("nan")
    h = disclosure.operating_point_hidden(n, c["tp"] + c["fn"], c["tp"] + c["fp"], c["tp"])
    return {"recall": recall, "precision": precision, "flag_rate": flag_rate,
            "hidden": {"recall": h["recall"], "precision": h["precision"],
                       "flag_rate": h["flag_rate"]}}


def _cell(stats: dict, metric: str) -> str:
    """One metric of one operating point, or the marker when it rests on fewer than MIN_CELL_SIZE students."""
    if stats.get("hidden", {}).get(metric):
        return disclosure.SUPPRESSED_TEX
    return _fmt(stats[metric])


# ── Part 1: threshold comparison ──────────────────────────────────────────────

def threshold_comparison_rows(pilot_snapshot: pd.DataFrame, train_pos_rate: float) -> list[dict]:
    rows = []
    for faculty in TABLE_FACULTIES:
        subset = pilot_snapshot[pilot_snapshot["faculty_code"] == faculty].copy()

        # ── Credits policy: real (fixed, national law) vs ML-style calibrated ──
        real_thresh = 60
        real_preds  = subset.copy()
        real_preds["y_pred"] = (real_preds["ects_behind"] >= real_thresh).astype(int)
        real_stats  = _recall_precision_flagrate(
            real_preds.assign(
                ever_flagged=real_preds["y_pred"], dropped_out=real_preds[TARGET]
            )
        )

        calib_thresh = _enrollment_quantile_threshold(
            subset.rename(columns={"ects_behind": "y_prob"}), train_pos_rate
        )
        calib_preds  = enrollment_predictions_df(
            subset, subset["ects_behind"].values, train_pos_rate, target=TARGET
        )
        calib_stats  = _recall_precision_flagrate(calib_preds)

        rows.append({
            "faculty": faculty, "rule": "Credits policy",
            "real_thresh": real_thresh, "real": real_stats,
            "calib_thresh": calib_thresh, "calib": calib_stats,
        })

        # ── Operational model: real (the students contacted) vs ML-style calibrated ──
        ## The operating point is who was contacted, not points_total >= tau:
        ## that rebuild also flags uncontacted students at or above tau.
        real_thresh = _operational_model_threshold(subset, faculty)
        _check_threshold_disclosable(subset, faculty, real_thresh)
        if real_thresh is not None:
            real_stats = _stats_from_counts(operational_model_real_breakdown(subset, faculty))
        else:
            real_stats = {"recall": float("nan"), "precision": float("nan"), "flag_rate": float("nan"),
                          "hidden": {}}

        calib_thresh = _enrollment_quantile_threshold(
            subset.rename(columns={"points_total": "y_prob"}), train_pos_rate
        )
        calib_preds  = enrollment_predictions_df(
            subset, subset["points_total"].values, train_pos_rate, target=TARGET
        )
        calib_stats  = _recall_precision_flagrate(calib_preds)

        rows.append({
            "faculty": faculty, "rule": "Operational model",
            "real_thresh": real_thresh, "real": real_stats,
            "calib_thresh": calib_thresh, "calib": calib_stats,
        })
    return rows


def threshold_comparison_table(rows: list[dict]) -> list[str]:
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r" & \multicolumn{4}{c}{Real threshold} & \multicolumn{4}{c}{ML-style calibrated threshold} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
        r"Rule & $\tau$ & Flag rate & Recall & Precision & $\tau$ & Flag rate & Recall & Precision \\",
        r"\midrule",
    ]
    for row in rows:
        real   = row["real"]
        calib  = row["calib"]
        rt     = "---" if row["real_thresh"] is None else str(row["real_thresh"])
        ct     = _fmt(row["calib_thresh"], 1)
        lines.append(
            f"{row['rule']} & {rt} & {_cell(real, 'flag_rate')} & "
            f"{_cell(real, 'recall')} & {_cell(real, 'precision')} & {ct} & "
            f"{_cell(calib, 'flag_rate')} & {_cell(calib, 'recall')} & {_cell(calib, 'precision')} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{\textbf{Real vs.\ ML-style calibrated thresholds for the rule-based baselines, "
        r"Faculty A, October 2023 snapshot.} ``Real threshold'' is what each rule actually uses in "
        r"practice: a fixed cutoff of $\geq 60$ ECTS credits behind (national law) for the credits policy; "
        r"for the operational rule-based model, the students it actually contacted, with $\tau$ "
        r"the lowest total points among them. ``ML-style calibrated threshold'' instead applies "
        r"the enrollment-level calibration to the same score "
        r"(ECTS credits behind or total points), picking $\tau$ so the flagged share matches the "
        r"training enrollment-level dropout rate --- exactly as done for the ML models.}",
        r"\label{tab:pilot_snapshot_threshold_comparison}",
        r"\end{table}",
    ]
    return lines


# ── Part 2: recalibrated enrollment breakdown ─────────────────────────────────

def recalibrated_breakdown_rows(
    pilot_snapshot: pd.DataFrame, faculty_data: dict, train_pos_rate: float,
    ws_model: str | None, ws_fs: str | None,
) -> list[dict]:
    from src.config import MAX_ML_FS, MAX_ML_MODEL, USED_VARS
    import joblib
    from src.pilot_snapshot_comparison import _MODELS_DIR

    rows = []
    fr_path = _MODELS_DIR / "fasterrisk__used_vars.joblib"
    for faculty in TABLE_FACULTIES:
        subset   = pilot_snapshot[pilot_snapshot["faculty_code"] == faculty].copy()
        preds_df = faculty_data[faculty]["preds_df"]

        cp_preds = enrollment_predictions_df(
            subset, subset["ects_behind"].values, train_pos_rate, target=TARGET
        )
        op_preds = enrollment_predictions_df(
            subset, subset["points_total"].values, train_pos_rate, target=TARGET
        )
        bars = [
            ("Credits policy (recalibrated)",    _recall_precision_flagrate(cp_preds)),
            ("Operational model (recalibrated)", _recall_precision_flagrate(op_preds)),
        ]

        if fr_path.exists():
            fr_model = joblib.load(fr_path)
            y_prob   = fr_model.predict_proba(subset[USED_VARS])[:, 1]
            fr_enr   = enrollment_predictions_df(subset, y_prob, train_pos_rate, target=TARGET)
            bars.append((r"FasterRisk (\texttt{used\_vars})", _recall_precision_flagrate(fr_enr)))

        ## A retained credits policy is the first bar already, and has no row in
        ## the per-faculty predictions, which hold trained models only.
        if ws_model is not None and ws_fs is not None and ws_model not in _RULE_BASED_MODELS:
            ws_preds = preds_df[
                (preds_df["model"] == ws_model) & (preds_df["feature_set"] == ws_fs)
            ]
            bars.append(("Participatory", _recall_precision_flagrate(ws_preds)))

        best_preds = preds_df[
            (preds_df["model"] == MAX_ML_MODEL) & (preds_df["feature_set"] == MAX_ML_FS)
        ]
        bars.append(("Max ML", _recall_precision_flagrate(best_preds)))

        for label, stats in bars:
            rows.append({"faculty": faculty, "model": label, **stats})
    return rows


def recalibrated_breakdown_table(rows: list[dict]) -> list[str]:
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Model & Flag rate & Recall & Precision \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['model']} & {_cell(row, 'flag_rate')} & "
            f"{_cell(row, 'recall')} & {_cell(row, 'precision')} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{\textbf{Recalibrated enrollment-level breakdown, Faculty A, October 2023 snapshot.} "
        r"Credits policy and the operational rule-based model are thresholded with the same "
        r"enrollment-level calibration rule as the ML models (flagged share matches the "
        r"training enrollment-level dropout rate), rather than their real-world thresholds "
        r"(Table~\ref{tab:pilot_snapshot_threshold_comparison}). See "
        r"Figure~\ref{fig:pilot_snapshot_recalibrated_breakdown} for the corresponding stacked bar chart.}",
        r"\label{tab:pilot_snapshot_recalibrated_breakdown}",
        r"\end{table}",
    ]
    return lines


def main() -> None:
    if not PILOT_SNAPSHOT_PARQUET.exists():
        print(f"ERROR: {PILOT_SNAPSHOT_PARQUET} not found. Run python data/generate_pilot_snapshot.py first.")
        sys.exit(1)

    pilot_snapshot = load_pilot_snapshot()

    meta               = _load_metadata()
    train_pos_rate     = meta["train_pos_rate"]
    available_features = meta["available_features"]
    print(f"  train_pos_rate = {train_pos_rate:.4f}")

    print("\n[1/2] Threshold comparison (real vs. ML-style calibrated) ...")
    thresh_rows = threshold_comparison_rows(pilot_snapshot, train_pos_rate)
    for row in thresh_rows:
        print(f"  {row['faculty']:5s} {row['rule']:18s}  "
              f"real tau={row['real_thresh']}  recall={_fmt(row['real']['recall'])}  "
              f"| calibrated tau={_fmt(row['calib_thresh'], 1)}  recall={_fmt(row['calib']['recall'])}")
    tex_threshold = threshold_comparison_table(thresh_rows)

    print("\n[2/2] Recalibrated enrollment breakdown ...")
    faculty_data: dict = {}
    for faculty in TABLE_FACULTIES:
        contact_col = _FACULTY_CONTACT[faculty]
        subset      = pilot_snapshot[pilot_snapshot["faculty_code"] == faculty].copy()
        _, _, _, _, preds_df = evaluate_faculty(
            subset, contact_col, available_features, train_pos_rate,
        )
        faculty_data[faculty] = {"preds_df": preds_df}

    ws_model, ws_fs = participatory_config(RESULTS_PATH)
    if ws_model is not None:
        print(f"  Workshop model: {ws_model} + {ws_fs}")
    else:
        print("  workshop_config.json not found — skipping Participatory bar.")

    breakdown_rows = recalibrated_breakdown_rows(pilot_snapshot, faculty_data, train_pos_rate, ws_model, ws_fs)
    tex_breakdown  = recalibrated_breakdown_table(breakdown_rows)

    intro = (
        r"% Auto-generated by check_rule_based_recalibration.py — do not edit manually."
        "\n"
        r"% Compares the rule-based baselines' real thresholds against an ML-style "
        r"calibrated threshold, and shows the resulting enrollment breakdown on the "
        r"October 2023 snapshot (Faculty A) when all methods are calibrated the same way."
    )
    content = "\n\n".join([intro, "\n".join(tex_threshold), "\n".join(tex_breakdown)]) + "\n"

    out = OUT_DIR / "pilot_snapshot_recalibration.tex"
    out.write_text(content)
    print(f"\nSaved: {out}")



if __name__ == "__main__":
    main()
