"""
Generate LaTeX-friendly Markdown results tables.

Table 1 — 2021 test year: all models × all feature sets,
  enrollment-level and row-level metrics, with 95% bootstrap CIs where available.

Table 2 — pilot_snapshot October 2023 snapshot: same metrics, split by FAC_A and FAC_B faculty.

Run from project root:
    python src/make_results_tables.py
Writes: results/results_tables.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from paths.paths import RESULTS_PATH, PILOT_SNAPSHOT_PARQUET
from src.config import CONTACTS_RECORDED_FACULTIES, DATA_CONFIGS, TARGET
from src.metrics import (
    credits_policy_metrics,
    enrollment_level_metrics,
    row_level_metrics,
)
from src.pilot_snapshot_comparison import (
    FACULTIES,
    _CONFIG_ORDER,
    _FACULTY_DISPLAY,
    _FACULTY_CONTACT,
    _ML_MODELS,
    _contact_recall,
    _load_metadata,
)
from src import disclosure
from src.data import load_pilot_snapshot
from src.features import load_model_checked

_TABLE_ML_MODELS = ["fasterrisk", "logistic_regression", "xgboost", "mlp"]

# ── Display names ──────────────────────────────────────────────────────────────

## Shared with every caption that names a model or feature set.
from src.display import FEATURE_SET_LABELS, MODEL_LABELS  # noqa: E402

FEATURE_SET_ORDER_MAIN = [
    "ects_behind_ge_60",
    "a_credits",
    "b_credits_performance",
    "c_plus_enrolment",
    "d_plus_lms",
    "e_plus_sensitive",
    "used_vars",
]

# Table 2 uses the same ladder as pilot_snapshot_comparison, plus used_vars
_TABLE2_CONFIG_ORDER = list(_CONFIG_ORDER) + ["used_vars"]

# ── Helpers ────────────────────────────────────────────────────────────────────

def _fmt(val: float, ci_lo: float | None = None, ci_hi: float | None = None) -> str:
    """Format a metric value with optional CI as 'x.xxx [lo, hi]'."""
    if np.isnan(val):
        return "—"
    s = f"{val:.3f}"
    if ci_lo is not None and ci_hi is not None and not (np.isnan(ci_lo) or np.isnan(ci_hi)):
        s += f" [{ci_lo:.3f}, {ci_hi:.3f}]"
    return s


def _flag_rate(r) -> float:
    """
    Realised share of enrollments flagged.

    Reported alongside recall/precision because τ rounds down (capacity
    constraint — see conservative_threshold in src/metrics.py): a model with
    coarse scores can flag materially less than the calibration target, so its
    recall is a lower bound and not directly comparable to a model sitting on
    target. Prefers the `enr_flag_rate` column; falls back to the counts so
    tables can still be built from results CSVs written before it existed.
    """
    val = getattr(r, "enr_flag_rate", np.nan)
    if val is not None and not (isinstance(val, float) and np.isnan(val)):
        return float(val)
    n = getattr(r, "n_enrollments", np.nan)
    k = getattr(r, "n_flagged_enrollments", np.nan)
    if np.isnan(n) or np.isnan(k) or n == 0:
        return np.nan
    return float(k) / float(n)


def _get_ci(cis: pd.DataFrame, model: str, feature_set: str, metric: str
            ) -> tuple[float | None, float | None]:
    row = cis[
        (cis["model"] == model) &
        (cis["feature_set"] == feature_set) &
        (cis["metric"] == metric)
    ]
    if row.empty:
        return None, None
    return float(row["ci_low"].iloc[0]), float(row["ci_high"].iloc[0])


def _get_ci_tab(cis: pd.DataFrame, model: str, feature_set: str,
                faculty: str, metric: str) -> tuple[float | None, float | None]:
    row = cis[
        (cis["model"] == model) &
        (cis["feature_set"] == feature_set) &
        (cis["faculty"] == faculty) &
        (cis["metric"] == metric)
    ]
    if row.empty:
        return None, None
    return float(row["ci_low"].iloc[0]), float(row["ci_high"].iloc[0])


# ── Table 1: 2021 main evaluation ─────────────────────────────────────────────

def build_table1(df2021: pd.DataFrame, cis: pd.DataFrame) -> str:
    """Return markdown text for the 2021 results table."""
    lines: list[str] = []
    lines.append("## Table 1: Evaluation results — test year 2021")
    lines.append("")
    lines.append(
        "Enrollment-level metrics use max predicted probability per enrollment; "
        "threshold τ calibrated to come as close as possible to the training "
        "dropout rate without exceeding it. Flag rate is the realized share of "
        "enrollments flagged — where it falls short of the target, recall is a "
        "lower bound (coarse scores cannot hit the target exactly). "
        "Row-level metrics are per (enrollment, EWS date). "
        "95\\% bootstrap CIs in brackets (1000 iterations)."
    )
    lines.append("")

    # Header
    header = (
        "| Model | Feature set "
        "| Flag rate | Recall | Precision | AUC "
        "| Row recall | Row precision | Row AUC |"
    )
    sep = (
        "|:---|:---|"
        ":---|:---|:---|:---|"
        ":---|:---|:---|"
    )
    lines.append(header)
    lines.append(sep)

    # Group by feature set in canonical order
    for fs in FEATURE_SET_ORDER_MAIN:
        fs_rows = df2021[df2021["feature_set"] == fs]
        if fs_rows.empty:
            continue

        fs_label = FEATURE_SET_LABELS.get(fs, fs)

        # credits_policy row (only in ects_behind_ge_60 group)
        if fs == "ects_behind_ge_60":
            r = fs_rows[fs_rows["model"] == "credits_policy"]
            if not r.empty:
                r = r.iloc[0]
                lines.append(
                    f"| {MODEL_LABELS['credits_policy']} | {fs_label} "
                    f"| {_fmt(_flag_rate(r))} "
                    f"| {_fmt(r.enr_recall)} "
                    f"| {_fmt(r.enr_precision)} "
                    f"| {_fmt(r.roc_auc)} "
                    f"| {_fmt(r.row_recall)} "
                    f"| {_fmt(r.row_precision)} "
                    f"| {_fmt(r.row_roc_auc)} |"
                )
            continue  # no ML models for this feature set

        # ML model rows within this feature set
        first = True
        for model in _TABLE_ML_MODELS:
            r = fs_rows[fs_rows["model"] == model]
            if r.empty:
                continue
            r = r.iloc[0]

            lo_recall, hi_recall   = _get_ci(cis, model, fs, "enr_recall")
            lo_prec,   hi_prec     = _get_ci(cis, model, fs, "enr_precision")
            lo_auc,    hi_auc      = _get_ci(cis, model, fs, "roc_auc")
            lo_rauc,   hi_rauc     = _get_ci(cis, model, fs, "row_roc_auc")

            lines.append(
                f"| {MODEL_LABELS.get(model, model)} | {fs_label if first else ''} "
                f"| {_fmt(_flag_rate(r))} "
                f"| {_fmt(r.enr_recall, lo_recall, hi_recall)} "
                f"| {_fmt(r.enr_precision, lo_prec, hi_prec)} "
                f"| {_fmt(r.roc_auc, lo_auc, hi_auc)} "
                f"| {_fmt(r.row_recall)} "
                f"| {_fmt(r.row_precision)} "
                f"| {_fmt(r.row_roc_auc, lo_rauc, hi_rauc)} |"
            )
            first = False

    return "\n".join(lines)


# ── Table 2: pilot_snapshot by faculty ───────────────────────────────────────────────

def _run_faculty_eval(
    pilot_snapshot: pd.DataFrame,
    available_features: dict,
    train_pos_rate: float,
) -> dict:
    """Run evaluate_faculty for both FAC_A and FAC_B; return dict of results."""
    models_dir = RESULTS_PATH / "models"
    faculty_data: dict = {}

    for faculty in FACULTIES:
        contact_col = _FACULTY_CONTACT[faculty]
        subset = pilot_snapshot[pilot_snapshot["faculty_code"] == faculty].copy()

        enr_rows, row_rows = [], []
        for config_name in _TABLE2_CONFIG_ORDER:
            # Use DATA_CONFIGS as the authoritative feature list; available_features
            # from train_metadata.json can be stale if USED_VARS was updated after
            # the last evaluation run.
            available = DATA_CONFIGS[config_name]
            tab_X = subset[available]
            for model_name in _TABLE_ML_MODELS:
                model_path = models_dir / f"{model_name}__{config_name}.joblib"
                if not model_path.exists():
                    continue
                model = load_model_checked(model_path, available)
                y_prob = model.predict_proba(tab_X)[:, 1]

                enr_m = enrollment_level_metrics(subset, y_prob, train_pos_rate, target=TARGET)
                row_m = row_level_metrics(subset, y_prob, train_pos_rate, target=TARGET)
                enr_rows.append({"model": model_name, "feature_set": config_name, **enr_m})
                row_rows.append({"model": model_name, "feature_set": config_name, **row_m})

        faculty_data[faculty] = {
            "enr_df":      pd.DataFrame(enr_rows),
            "row_df":      pd.DataFrame(row_rows),
            "rule_m":      credits_policy_metrics(subset, target=TARGET),
            "contact_m":   _contact_recall(subset, col=contact_col, score_col="points_total"),
            "dropout_rate": float(subset[TARGET].mean()),
            "n_enrollments": int(subset["enrollment"].nunique()),
        }

    return faculty_data


def _pilot_snapshot_faculty_block(
    faculty: str,
    enr_df: pd.DataFrame,
    row_df: pd.DataFrame,
    rule_m: dict,
    contact_m: dict,
    cis_tab: pd.DataFrame,
) -> list[str]:
    """Return markdown rows for one faculty panel in the pilot_snapshot table."""
    lines: list[str] = []

    # Merge enr and row metrics
    # credits_policy row (no CI — no continuous score)
    lines.append(
        f"| {MODEL_LABELS['credits_policy']} | Rule: ≥60 ECTS behind "
        f"| {_fmt(rule_m['enr_recall'])} "
        f"| {_fmt(rule_m['enr_precision'])} "
        f"| {_fmt(rule_m['roc_auc'])} |"
    )

    # contact pilot row (no CI — AUC uses points_total as the continuous score)
    contact_name = f"contact_{faculty.lower()}"
    contacts_known = faculty in CONTACTS_RECORDED_FACULTIES
    lines.append(
        f"| {MODEL_LABELS[contact_name]} | Faculty pilot "
        f"| {_fmt(contact_m['enr_recall']) if contacts_known else '---'} "
        f"| {_fmt(contact_m['enr_precision']) if contacts_known else '---'} "
        f"| {_fmt(contact_m['roc_auc'])} |"
    )

    # ML model rows
    for fs in _TABLE2_CONFIG_ORDER:
        first = True
        for model in _TABLE_ML_MODELS:
            r = enr_df[(enr_df["model"] == model) & (enr_df["feature_set"] == fs)]
            if r.empty:
                continue
            r = r.iloc[0]

            lo_rec,  hi_rec  = _get_ci_tab(cis_tab, model, fs, faculty, "enr_recall")
            lo_prec, hi_prec = _get_ci_tab(cis_tab, model, fs, faculty, "enr_precision")
            lo_auc,  hi_auc  = _get_ci_tab(cis_tab, model, fs, faculty, "roc_auc")

            lines.append(
                f"| {MODEL_LABELS.get(model, model)} | {FEATURE_SET_LABELS.get(fs, fs) if first else ''} "
                f"| {_fmt(r.enr_recall,    lo_rec,  hi_rec)} "
                f"| {_fmt(r.enr_precision, lo_prec, hi_prec)} "
                f"| {_fmt(r.roc_auc,       lo_auc,  hi_auc)} |"
            )
            first = False

    return lines


def build_table2(faculty_data: dict, cis_tab: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("## Table 2: Pilot snapshot results — October 2023 snapshot")
    lines.append("")
    lines.append(
        "Evaluation on the October 2023 EWS activation snapshot (single October activation date), "
        "split by faculty. FAC_A and FAC_B piloted counsellor outreach; other faculties are excluded. "
        "Metrics defined identically to Table 1; τ from training dropout rate. "
        "Because the pilot snapshot contains exactly one EWS date per enrollment, "
        "row-level recall and precision are identical to the enrollment-level values and are omitted. "
        "95\\% bootstrap CIs on recall, precision and AUC (1000 iterations, τ re-estimated in each resample)."
    )
    lines.append("")

    header = (
        "| Model | Feature set "
        "| Recall | Precision | AUC |"
    )
    sep = (
        "|:---|:---|"
        ":---|:---|:---|"
    )

    for faculty in FACULTIES:
        fd = faculty_data[faculty]
        lines.append(f"### {faculty} faculty (n = {fd['n_enrollments']:,} enrollments; dropout rate = {fd['dropout_rate']:.3f})")
        lines.append("")
        lines.append(header)
        lines.append(sep)
        lines.extend(
            _pilot_snapshot_faculty_block(
                faculty,
                fd["enr_df"],
                fd["row_df"],
                fd["rule_m"],
                fd["contact_m"],
                cis_tab,
            )
        )
        lines.append("")

    return "\n".join(lines)


# ── LaTeX table helpers ────────────────────────────────────────────────────────

def _hidden(r) -> dict:
    """Which of this row's metrics rest on fewer than MIN_CELL_SIZE students."""
    return disclosure.results_row_hidden(r)


def _tex_cell(val, ci_lo=None, ci_hi=None, hide: bool = False) -> str:
    """Point estimate with optional 95% CI in scriptsize font; the marker when hidden."""
    if hide:
        return disclosure.SUPPRESSED_TEX
    try:
        if val is None or np.isnan(float(val)):
            return "---"
    except (TypeError, ValueError):
        return "---"
    s = f"{float(val):.3f}"
    if ci_lo is not None:
        try:
            if not np.isnan(float(ci_lo)):
                s += r" {\scriptsize[" + f"{float(ci_lo):.2f},\\,{float(ci_hi):.2f}" + r"]}"
        except (TypeError, ValueError):
            pass
    return s


# ── LaTeX Table 1: 2021 results ────────────────────────────────────────────────

def build_table1_tex(df2021: pd.DataFrame, cis: pd.DataFrame) -> str:
    col_spec = "llccccccc"
    header_row = (
        r"\textbf{Model} & \textbf{Feature set} "
        r"& \textbf{Flag rate} & \textbf{Recall} & \textbf{Precision} & \textbf{Enr. AUC} "
        r"& \textbf{Act. recall} & \textbf{Act. prec.} & \textbf{Act. AUC} \\"
    )

    body: list[str] = []

    for fs in FEATURE_SET_ORDER_MAIN:
        fs_rows = df2021[df2021["feature_set"] == fs]
        if fs_rows.empty:
            continue
        fs_label = FEATURE_SET_LABELS.get(fs, fs)

        if fs == "ects_behind_ge_60":
            r = fs_rows[fs_rows["model"] == "credits_policy"]
            if not r.empty:
                r = r.iloc[0]
                h = _hidden(r)
                body.append(
                    f"{MODEL_LABELS['credits_policy']} & Rule: $\\geq$60 ECTS behind "
                    f"& {_tex_cell(_flag_rate(r), hide=h['flag_rate'])} "
                    f"& {_tex_cell(r.enr_recall, hide=h['recall'])} "
                    f"& {_tex_cell(r.enr_precision, hide=h['precision'])} "
                    f"& {_tex_cell(r.roc_auc, hide=h['auc'])} "
                    f"& {_tex_cell(r.row_recall, hide=h['recall'])} "
                    f"& {_tex_cell(r.row_precision, hide=h['precision'])} "
                    f"& {_tex_cell(r.row_roc_auc, hide=h['auc'])} \\\\"
                )
            body.append(r"\hline")
            continue

        first = True
        for model in _ML_MODELS:
            r = fs_rows[fs_rows["model"] == model]
            if r.empty:
                continue
            r = r.iloc[0]
            h = _hidden(r)
            lo_rec, hi_rec   = _get_ci(cis, model, fs, "enr_recall")
            lo_prec, hi_prec = _get_ci(cis, model, fs, "enr_precision")
            lo_auc, hi_auc   = _get_ci(cis, model, fs, "roc_auc")
            lo_rauc, hi_rauc = _get_ci(cis, model, fs, "row_roc_auc")
            body.append(
                f"{MODEL_LABELS.get(model, model)} & {fs_label if first else ''} "
                f"& {_tex_cell(_flag_rate(r), hide=h['flag_rate'])} "
                f"& {_tex_cell(r.enr_recall,   lo_rec,  hi_rec, hide=h['recall'])} "
                f"& {_tex_cell(r.enr_precision, lo_prec, hi_prec, hide=h['precision'])} "
                f"& {_tex_cell(r.roc_auc,       lo_auc,  hi_auc, hide=h['auc'])} "
                f"& {_tex_cell(r.row_recall, hide=h['recall'])} "
                f"& {_tex_cell(r.row_precision, hide=h['precision'])} "
                f"& {_tex_cell(r.row_roc_auc,   lo_rauc, hi_rauc, hide=h['auc'])} \\\\"
            )
            first = False
        body.append(r"\hline")

    caption = (
        r"\caption{\textbf{Enrollment-level and activation-level evaluation results, "
        r"test year 2021.} "
        r"Enrollment-level metrics use the maximum predicted probability per "
        r"enrollment; threshold $\tau$ is calibrated so the share of enrollments "
        r"flagged comes as close as possible to the training enrollment-level "
        r"dropout rate without exceeding it. Flag rate is the realized share of "
        r"enrollments flagged: models whose scores take only a few distinct "
        r"values cannot hit the target exactly and fall short of it, so their "
        r"recall is a lower bound and is not directly comparable to that of a "
        r"model sitting on target. "
        r"Activation-level metrics are computed per (enrollment, EWS-date) "
        r"observation using the same $\tau$. "
        r"95\% bootstrap confidence intervals (1\,000 iterations, resampling "
        r"enrollments with $\tau$ held fixed) in brackets where available. "
        + r"Feature sets are cumulative: (a)~credits; (b)~+~performance; "
        r"(c)~+~enrollment; (d)~+~LMS; (e)~+~sensitive attributes.}"
    )

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\footnotesize",
        # 8 columns + inline CIs don't fit \linewidth at \footnotesize alone;
        # resizebox here only applies a modest ~0.78x shrink (verified), not
        # the uncontrolled/extreme shrink from the old \small+resizebox+\tiny
        # combination this replaced.
        r"\resizebox{\linewidth}{!}{%",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\hline",
        header_row,
        r"\hline",
    ] + body + [
        r"\end{tabular}%",
        r"}",
        caption,
        r"\label{tab:results_2021}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


# ── LaTeX Table 2: pilot_snapshot by faculty ────────────────────────────────────────

def build_table2_tex(faculty_data: dict, cis_tab: pd.DataFrame) -> str:
    col_spec = "llccc"
    header_row = (
        r"\textbf{Model} & \textbf{Feature set} "
        r"& \textbf{Recall} & \textbf{Precision} & \textbf{AUC} \\"
    )

    body: list[str] = []

    for faculty in FACULTIES:
        fd = faculty_data[faculty]
        enr_df    = fd["enr_df"]
        rule_m    = fd["rule_m"]
        contact_m = fd["contact_m"]
        n_enr     = fd["n_enrollments"]
        dr        = fd["dropout_rate"]
        n_do      = int(round(dr * n_enr))
        dr_txt    = (disclosure.SUPPRESSED_TEX if disclosure.rate_is_small(n_do, n_enr)
                     else f"{dr:.3f}")
        h_rule    = _hidden(pd.Series(rule_m))
        # the contact pilot's counts, under the standard names
        h_contact = _hidden(pd.Series({
            "n_enrollments":         contact_m["n_enrollments"],
            "n_dropout_enrollments": contact_m["n_dropout"],
            "n_flagged_enrollments": contact_m["n_flagged"],
            "n_tp_enrollments":      contact_m["n_tp"],
        }))

        body.append(
            r"\multicolumn{5}{l}{\textbf{"
            + _FACULTY_DISPLAY[faculty]
            + r"} ($n = "
            + f"{n_enr:,}"
            + r"$ enrollments; dropout rate "
            + dr_txt
            + r")} \\"
        )
        body.append(r"\hline")

        # credits_policy (no CI — no continuous score)
        body.append(
            f"{MODEL_LABELS['credits_policy']} & Rule: $\\geq$60 ECTS behind "
            f"& {_tex_cell(rule_m['enr_recall'], hide=h_rule['recall'])} "
            f"& {_tex_cell(rule_m['enr_precision'], hide=h_rule['precision'])} "
            f"& {_tex_cell(rule_m['roc_auc'], hide=h_rule['auc'])} \\\\"
        )

        # contact pilot (no CI — AUC uses points_total as the continuous score)
        contact_name = f"contact_{faculty.lower()}"
        contacts_known = faculty in CONTACTS_RECORDED_FACULTIES
        op_recall = (_tex_cell(contact_m['enr_recall'], hide=h_contact['recall'])
                     if contacts_known else "---")
        op_precision = (_tex_cell(contact_m['enr_precision'], hide=h_contact['precision'])
                        if contacts_known else "---")
        body.append(
            f"{MODEL_LABELS[contact_name]} & Faculty pilot "
            f"& {op_recall} "
            f"& {op_precision} "
            f"& {_tex_cell(contact_m['roc_auc'], hide=h_contact['auc'])} \\\\"
        )
        body.append(r"\hline")

        for fs in _TABLE2_CONFIG_ORDER:
            first = True
            for model in _ML_MODELS:
                r = enr_df[(enr_df["model"] == model) & (enr_df["feature_set"] == fs)]
                if r.empty:
                    continue
                r = r.iloc[0]
                h = _hidden(r)
                lo_rec,  hi_rec  = _get_ci_tab(cis_tab, model, fs, faculty, "enr_recall")
                lo_prec, hi_prec = _get_ci_tab(cis_tab, model, fs, faculty, "enr_precision")
                lo_auc,  hi_auc  = _get_ci_tab(cis_tab, model, fs, faculty, "roc_auc")
                body.append(
                    f"{MODEL_LABELS.get(model, model)} & {FEATURE_SET_LABELS.get(fs, fs) if first else ''} "
                    f"& {_tex_cell(r.enr_recall,    lo_rec,  hi_rec,  hide=h['recall'])} "
                    f"& {_tex_cell(r.enr_precision, lo_prec, hi_prec, hide=h['precision'])} "
                    f"& {_tex_cell(r.roc_auc,       lo_auc,  hi_auc,  hide=h['auc'])} \\\\"
                )
                first = False
            body.append(r"\hline")

        body.append(r"\\[-4pt]")

    caption = (
        r"\caption{\textbf{Evaluation results on the October 2023 EWS snapshot, "
        r"by faculty.} "
        r"The snapshot contains exactly one EWS observation per enrollment, so "
        r"enrollment-level and activation-level metrics coincide. "
        r"$\tau$ is calibrated to the training enrollment-level dropout rate. "
        r"95\% bootstrap CIs (1\,000 iterations, with $\tau$ re-estimated in each "
        r"resample) in brackets for ML models; "
        r"the credits policy and the operational rule-based model are shown as "
        r"point estimates without bootstrap CIs. "
        r"The operational rule-based model was deployed at Faculties A and B only; "
        r"it is not known which students Faculty B contacted, so its recall and "
        r"precision there are not reported.}"
    )

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\footnotesize",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\hline",
        header_row,
        r"\hline",
    ] + body + [
        r"\end{tabular}",
        caption,
        r"\label{tab:results_pilot_snapshot}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


# ── Main ───────────────────────────────────────────────────────────────────────

DESCRIPTIVE_DIR = RESULTS_PATH / "descriptive"


def main() -> None:
    print("Loading data ...")
    df  = pd.read_csv(RESULTS_PATH / "evaluation_results.csv")
    cis = pd.read_csv(RESULTS_PATH / "bootstrap_cis.csv")

    cis_tab_path = RESULTS_PATH / "bootstrap_cis_pilot_snapshot.csv"
    has_tab_cis  = cis_tab_path.exists()
    cis_tab      = pd.read_csv(cis_tab_path) if has_tab_cis else pd.DataFrame()
    if not has_tab_cis:
        print("WARNING: bootstrap_cis_pilot_snapshot.csv not found — Table 2 CIs will be omitted.")

    df2021 = df[df["test_year"] == 2021].copy()
    cis2021 = cis[cis["test_year"] == 2021].copy()

    print("Building Table 1 (2021) ...")
    t1_md  = build_table1(df2021, cis2021)
    t1_tex = build_table1_tex(df2021, cis2021)

    print("Loading pilot_snapshot and running faculty evaluation ...")
    pilot_snapshot = load_pilot_snapshot()
    meta = _load_metadata()
    faculty_data = _run_faculty_eval(
        pilot_snapshot, meta["available_features"], meta["train_pos_rate"]
    )

    print("Building Table 2 (pilot_snapshot) ...")
    t2_md  = build_table2(faculty_data, cis_tab)
    t2_tex = build_table2_tex(faculty_data, cis_tab)

    DESCRIPTIVE_DIR.mkdir(parents=True, exist_ok=True)

    (DESCRIPTIVE_DIR / "results_table1.tex").write_text(t1_tex)
    (DESCRIPTIVE_DIR / "results_table2.tex").write_text(t2_tex)

    print(f"Saved → {DESCRIPTIVE_DIR / 'results_table1.tex'}")
    print(f"Saved → {DESCRIPTIVE_DIR / 'results_table2.tex'}")


if __name__ == "__main__":
    main()
