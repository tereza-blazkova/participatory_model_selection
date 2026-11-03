"""
Distil the October 2023 pilot figures' inputs.

Five paper figures rest on the same expensive step — scoring every saved model on
the pilot snapshot, faculty by faculty (`evaluate_faculty`) — so one prepare
script builds that once and writes all of their inputs:

    figures/input/pilot_breakdown_faculty_a.csv     Faculty A: real thresholds | the rules at tau
    figures/input/pilot_auc_faculty_{a,b}.csv       ROC AUC, one faculty each
    figures/input/pilot_points_profile.csv          dropout rate and count by points score
    figures/input/pilot_abroca.csv                  the ABROCA heatmap
    figures/input/scorecard_used_vars.csv           the FasterRisk integer scorecard

Each with a .json of its scalars. The draw scripts are one per figure, as always;
it is only the reaching-into-the-pipeline half that is shared.

What the breakdown inputs may contain
    Shares, never counts. Each bar is a 2x2 table whose margins are on the figure
    (the dashed dropout line, the flag share under the bar), so a cell gives away
    the others; where one holds too few students it is merged with its
    same-outcome partner into one grey block, exactly as the figure draws it, and
    that bar's recall and flag share are withheld from the tick label. The
    partner-merging is _render_breakdown_panel's rule, reproduced here so the
    numbers never reach the drawing at all.

    python figures/scripts/prepare_pilot.py [--figure breakdown|auc|points|abroca|scorecard|all]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from paths.paths import RESULTS_PATH                                      # noqa: E402
from prepare_utils import workshop_selection, write_input                 # noqa: E402
from utils import canvas                                                  # noqa: E402
from src import disclosure                                                # noqa: E402
from src.data import load_pilot_snapshot                                        # noqa: E402
from src.metrics import enrollment_predictions_df                         # noqa: E402
from src.pilot_snapshot_comparison import (                                     # noqa: E402
    FACULTIES, PILOT_SNAPSHOT_PARQUET, TARGET, USED_VARS,
    _CONFIG_ORDER, _FACULTY_CONTACT, _FACULTY_DISPLAY, _FACULTY_PANEL_LABEL,
    _METRIC_LABELS, _ML_MODELS, _MODELS_DIR,
    _breakdown_exact, _breakdown_from_preds, _check_threshold_band,
    _load_metadata, evaluate_faculty, operational_model_real_breakdown,
)
from src.config import CONTACTS_RECORDED_FACULTIES, MAX_ML_FS, MAX_ML_MODEL  # noqa: E402
from src.display import feature_labels, feature_set_label                # noqa: E402
from src.plotting import _MODEL_DISPLAY                                   # noqa: E402

SCRIPT = "prepare_pilot.py"


# ── The shared expensive step ─────────────────────────────────────────────────

def load_pilot() -> dict:
    """pilot_snapshot, every model scored on each faculty, and the workshop choice."""
    import joblib

    pilot_snapshot = load_pilot_snapshot(PILOT_SNAPSHOT_PARQUET)
    meta     = _load_metadata()
    train_pos_rate, available = meta["train_pos_rate"], meta["available_features"]

    _, ws_model, ws_fs = workshop_selection(RESULTS_PATH)

    faculty_data = {}
    for faculty in FACULTIES:
        subset = pilot_snapshot[pilot_snapshot["faculty_code"] == faculty].copy()
        enr_df, row_df, rule_m, contact_m, preds = evaluate_faculty(
            subset, _FACULTY_CONTACT[faculty], available, train_pos_rate)
        faculty_data[faculty] = {
            "subset": subset, "enr_df": enr_df, "row_df": row_df,
            "rule_m": rule_m, "contact_m": contact_m, "preds_df": preds,
            "n_total": subset["enrollment"].nunique(),
            "n_contacted": (int(subset[_FACULTY_CONTACT[faculty]].sum())
                            if faculty in CONTACTS_RECORDED_FACULTIES else None),
        }

    fr_path = _MODELS_DIR / "fasterrisk__used_vars.joblib"
    fr = joblib.load(fr_path) if fr_path.exists() else None
    uv_auc = {}
    if fr is not None:
        from src.used_vars_scorecard import _used_vars_roc_auc
        uv_auc = {f: _used_vars_roc_auc(fr, faculty_data[f]["subset"]) for f in FACULTIES}

    ## The pilot snapshot bootstrap CIs, if the run produced them. Drawn as whiskers.
    cis_path = RESULTS_PATH / "bootstrap_cis_pilot_snapshot.csv"
    cis = pd.read_csv(cis_path) if cis_path.exists() else pd.DataFrame()

    return {
        "pilot_snapshot": pilot_snapshot, "faculty_data": faculty_data,
        "train_pos_rate": train_pos_rate,
        "ws_model": ws_model, "ws_fs": ws_fs,
        "fasterrisk_used_vars": fr, "uv_auc": uv_auc, "cis": cis,
        "available_features": available,
    }


# ── Outcome breakdown (both variants) ─────────────────────────────────────────

def _bar_row(panel: int, faculty: str, i: int, label: str, counts: dict) -> dict:
    """
    One stacked bar, as shares, with the suppression already applied.

    _render_breakdown_panel's rule: the four cells are two same-outcome pairs
    (TN+FP non-dropout, TP+FN dropout), and a pair holding too few students is
    drawn as one grey block of their combined height. The block is the known
    margin — the dropout line is on the figure — so it discloses nothing, while
    either cell alone would. That bar's recall and flag share go with them.
    """
    n = counts["n"]
    hide_non = disclosure.is_small(counts["tn"], counts["fp"])
    hide_do  = disclosure.is_small(counts["tp"], counts["fn"])
    recall = counts["tp"] / (counts["tp"] + counts["fn"]) if (counts["tp"] + counts["fn"]) else 0.0
    flag_r = (counts["tp"] + counts["fp"]) / n if n else 0.0
    tick = (label + f"\nrecall {disclosure.SUPPRESSED}\nflags {disclosure.SUPPRESSED}"
            if hide_non or hide_do else
            label + f"\nrecall {recall:.0%}\nflags {flag_r:.0%}")

    share = (lambda k: counts[k] / n) if n else (lambda k: np.nan)
    row = {
        "panel": panel, "panel_letter": _FACULTY_PANEL_LABEL[faculty], "faculty": faculty,
        "bar": i, "label": label, "tick": tick, "empty": n == 0,
        "frac_tn": np.nan, "frac_fp": np.nan, "frac_tp": np.nan, "frac_fn": np.nan,
        "block_nondropout": np.nan, "block_dropout": np.nan,
        "hide_nondropout": hide_non, "hide_dropout": hide_do,
    }
    if n == 0:
        return row
    if hide_non:
        row["block_nondropout"] = share("tn") + share("fp")
    else:
        row["frac_tn"], row["frac_fp"] = share("tn"), share("fp")
    if hide_do:
        row["block_dropout"] = share("tp") + share("fn")
    else:
        row["frac_tp"], row["frac_fn"] = share("tp"), share("fn")
    return row


def _breakdown_bars(ctx: dict, faculty: str, recalibrated: bool) -> list[tuple[str, dict]]:
    """The bars of one panel, in drawing order — the two variants differ only here."""
    data   = ctx["faculty_data"][faculty]
    subset, preds_df = data["subset"], data["preds_df"]
    rate   = ctx["train_pos_rate"]

    if recalibrated:
        ## Both rules put through the ML models' own threshold rule, so the
        ## comparison isolates discriminative power from threshold philosophy.
        cp_preds = enrollment_predictions_df(subset, subset["ects_behind"].values, rate, target=TARGET)
        op_preds = enrollment_predictions_df(subset, subset["points_total"].values, rate, target=TARGET)
        _check_threshold_band(subset, subset["ects_behind"] >= 60, cp_preds,
                              f"{faculty} credits policy")
        _check_threshold_band(subset, subset[_FACULTY_CONTACT[faculty]] == 1, op_preds,
                              f"{faculty} operational model")
        ## The panel's own line under the ticks says these are recalibrated.
        bars = [("Credits\npolicy",                     _breakdown_from_preds(cp_preds)),
                ("Operational\nrule-based\nmodel",     _breakdown_from_preds(op_preds))]
    else:
        rule_m = data["rule_m"]
        ## Exact counts: rebuilding TP from a 3-decimal precision can be off by
        ## one, which is exactly the margin that decides suppression.
        bars = [
            ("Credits\npolicy", _breakdown_exact(
                rule_m["n_enrollments"], rule_m["n_dropout_enrollments"],
                rule_m["n_flagged_enrollments"], rule_m["n_tp_enrollments"])),
            ## The students actually contacted — the same computation as
            ## pilot_snapshot_recalibration.tex's real-threshold row.
            ("Operational\nrule-based\nmodel", operational_model_real_breakdown(subset, faculty)),
        ]

    if ctx["fasterrisk_used_vars"] is not None:
        y_prob = ctx["fasterrisk_used_vars"].predict_proba(subset[USED_VARS])[:, 1]
        fr_enr = enrollment_predictions_df(subset, y_prob, rate, target=TARGET)
        bars.append(("FasterRisk\n(advisor\nvariables)", _breakdown_from_preds(fr_enr)))

    if ctx["ws_model"] and ctx["ws_fs"]:
        ws = preds_df[(preds_df["model"] == ctx["ws_model"])
                      & (preds_df["feature_set"] == ctx["ws_fs"])]
        ## Empty when the credits policy was retained: these predictions hold
        ## trained models only, and it has its own bar above.
        if len(ws):
            bars.append(("Participatory", _breakdown_from_preds(ws)))

    best = preds_df[(preds_df["model"] == MAX_ML_MODEL)
                    & (preds_df["feature_set"] == MAX_ML_FS)]
    bars.append(("Max ML", _breakdown_from_preds(best)))
    return bars


def prepare_breakdown(ctx: dict, recalibrated: bool,
                      faculties: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """One panel per faculty in `faculties` (default both); one faculty fills the width."""
    faculties = list(FACULTIES) if faculties is None else faculties
    single = len(faculties) == 1
    rows, lines = [], []
    for panel, faculty in enumerate(faculties):
        bars = _breakdown_bars(ctx, faculty, recalibrated)
        for i, (label, counts) in enumerate(bars):
            rows.append(_bar_row(panel, faculty, i, label, counts))
        ## The dashed line is the population's dropout rate, read off the first
        ## bar, which covers the whole faculty whatever the rule.
        lines.append({"panel": panel, "faculty": faculty,
                      "y": round(1.0 - bars[0][1]["dropout_rate"], 4),
                      "letter": "" if single else _FACULTY_PANEL_LABEL[faculty]})

    df = pd.DataFrame(rows)
    ## The note goes on each panel that withheld something, as the original did —
    ## it explains that panel's grey block, so it belongs beside it.
    hidden_panels = sorted(
        int(p) for p, g in df.groupby("panel")
        if bool(g["hide_nondropout"].any() or g["hide_dropout"].any())
    )
    meta = {
        "figsize": canvas(3.4 / 13) if single else canvas(4.5 / 13),
        "ylabel": "Share of enrollments",
        "dropout_lines": lines,
        "side_labels": {"upper": "Dropout", "lower": "Non-dropout"},
        ## The marker travels as data too, so figures/ never spells the threshold.
        "marker": disclosure.SUPPRESSED,
        "note": disclosure.note_text() if hidden_panels else None,
        "note_panels": hidden_panels,
        "text_columns": ["label", "tick", "faculty", "panel_letter"],
    }
    return df, meta



def prepare_breakdown_faculty(ctx: dict, faculty: str) -> tuple[pd.DataFrame, dict]:
    """
    One faculty, two panels side by side: (a) each rule at its real threshold,
    (b) the rules recalibrated to the ML models' threshold rule.
    """
    rows, lines = [], []
    for panel, recalibrated in enumerate((False, True)):
        bars = _breakdown_bars(ctx, faculty, recalibrated)
        for i, (label, counts) in enumerate(bars):
            rows.append(_bar_row(panel, faculty, i, label, counts))
        lines.append({"panel": panel, "faculty": faculty,
                      "y": round(1.0 - bars[0][1]["dropout_rate"], 4),
                      "letter": "ab"[panel],
                      ## Under the tick labels: what each panel's thresholds are.
                      "xlabel": ("Rules at their real thresholds, models at the calibrated threshold"
                                 if not recalibrated else
                                 "Rules recalibrated to the calibrated threshold, like the models")})
    df = pd.DataFrame(rows)
    df["panel_letter"] = df["panel"].map({0: "a", 1: "b"})
    hidden_panels = sorted(
        int(p) for p, g in df.groupby("panel")
        if bool(g["hide_nondropout"].any() or g["hide_dropout"].any())
    )
    meta = {
        "figsize": canvas(4.5 / 13),
        "ylabel": "Share of enrollments",
        "dropout_lines": lines,
        "side_labels": {"upper": "Dropout", "lower": "Non-dropout"},
        "marker": disclosure.SUPPRESSED,
        "note": disclosure.note_text() if hidden_panels else None,
        "note_panels": hidden_panels,
        "text_columns": ["label", "tick", "faculty", "panel_letter"],
    }
    return df, meta


# ── ROC AUC by faculty ────────────────────────────────────────────────────────

_AUC_COL  = "row_roc_auc"
_BAR_W    = 0.185   # as in the model comparison
_GROUP_GAP = 1.0 - len(_ML_MODELS) * _BAR_W


def _auc_positions() -> dict:
    """
    Where each bar sits on the x axis.

    Left to right: credits policy | operational rule-based model | FasterRisk used_vars | then
    the ML grid, five feature sets each holding one bar per model. The three
    standalone bars are pushed left of the grid by a gap apiece so they read as
    a separate block rather than a sixth feature set.
    """
    n_models = len(_ML_MODELS)
    first_left = -(n_models - 1) / 2 * _BAR_W - _BAR_W / 2
    ## The same white space between the standalone bars as between the model
    ## groups, as in the model comparison.
    x_uv = first_left - _GROUP_GAP - _BAR_W / 2
    x_ct = x_uv - _GROUP_GAP - _BAR_W
    x_cp = x_ct - _GROUP_GAP - _BAR_W
    return {"credits_policy": x_cp, "contact": x_ct, "fasterrisk_used_vars": x_uv,
            "grid": np.arange(len(_CONFIG_ORDER))}


def prepare_auc(ctx: dict, faculties: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Activation-level ROC AUC on the snapshot, one panel per faculty in
    `faculties` (default both). The main text prints Faculty A alone, the SI
    Faculty B alone; a single panel carries no panel letter.
    """
    faculties = list(FACULTIES) if faculties is None else faculties
    single = len(faculties) == 1
    pos  = _auc_positions()
    rows, panels = [], []

    for panel, faculty in enumerate(faculties):
        data = ctx["faculty_data"][faculty]
        rule_m, contact_m = data["rule_m"], data["contact_m"]
        cis = ctx["cis"]
        cis = (cis[(cis["metric"] == "roc_auc") & (cis["faculty"] == faculty)]
               if len(cis) else cis)

        ## Every bar ranks this faculty's dropouts against its non-dropouts, so
        ## all of them rest on those two counts; the credits-policy AUC rests in
        ## addition on the enrollments that have an ects_behind value at all.
        n_pos = int(contact_m.get("n_dropout", 0))
        n_all = int(contact_m.get("n_enrollments", data["n_total"]))
        all_hidden = disclosure.is_small(n_pos, n_all - n_pos)
        cp_hidden  = all_hidden or disclosure.is_small(rule_m.get("n_auc_pos"),
                                                       rule_m.get("n_auc_neg"))

        def ci_for(model, fs):
            if not len(cis):
                return (np.nan, np.nan)
            hit = cis[(cis["model"] == model) & (cis["feature_set"] == fs)]
            return ((float(hit.iloc[0]["ci_low"]), float(hit.iloc[0]["ci_high"]))
                    if len(hit) == 1 else (np.nan, np.nan))

        def add(series, x, value, label, hidden, model=None, fs=None, highlight=""):
            lo, hi = ci_for(model, fs) if model else (np.nan, np.nan)
            if hidden:
                value = lo = hi = np.nan
            rows.append({
                "panel": panel,
                "panel_letter": "" if single else _FACULTY_PANEL_LABEL[faculty],
                "faculty": faculty, "series": series, "x": round(float(x), 6),
                "value": value, "ci_low": lo, "ci_high": hi,
                "value_label": "" if hidden else f"{value:.2f}",
                ## The printed number sits above the whisker wherever one is
                ## drawn, on every bar. The standalone FasterRisk bar used to
                ## print on its own top, inside its interval, and the whisker
                ## struck through the number.
                "label_y": np.nan if hidden else (hi if hi == hi else value),
                "tick_label": label, "highlight": highlight, "hidden": hidden,
            })

        ## Standalone bars. A missing metric means the rule does not apply here,
        ## which is not the same as a withheld one, so it contributes no bar.
        if not pd.isna(rule_m.get("roc_auc", np.nan)):
            add("credits_policy", pos["credits_policy"], float(rule_m["roc_auc"]),
                "Credits\npolicy", cp_hidden)
        if not pd.isna(contact_m.get("roc_auc", np.nan)):
            add("operational_rule", pos["contact"], float(contact_m["roc_auc"]),
                "Operational\nrule-based\nmodel", all_hidden)
        uv = ctx["uv_auc"].get(faculty, np.nan)
        if not np.isnan(uv):
            add("fasterrisk_used_vars", pos["fasterrisk_used_vars"], float(uv),
                "FasterRisk\n(advisor\nvariables)", all_hidden,
                model="fasterrisk", fs="used_vars")

        ## The ML grid.
        row_df = data["row_df"]
        for i, model in enumerate(_ML_MODELS):
            vals = (row_df[row_df["model"] == model]
                    .set_index("feature_set")[_AUC_COL].reindex(_CONFIG_ORDER))
            offset = (i - (len(_ML_MODELS) - 1) / 2) * _BAR_W
            for j, (fs, v) in enumerate(zip(_CONFIG_ORDER, vals)):
                if pd.isna(v):
                    continue
                ## Which outline the bar gets: solid for the Participatory
                ## model, dashed for Max ML (as in the legend).
                highlight = ("max_ml" if (model == MAX_ML_MODEL and fs == MAX_ML_FS)
                             else "participatory" if (model == ctx["ws_model"]
                                                      and fs == ctx["ws_fs"]) else "")
                add(model, pos["grid"][j] + offset, float(v), fs, all_hidden,
                    model=model, fs=fs, highlight=highlight)

        xs_sa = [r["x"] for r in rows if r["panel"] == panel and r["series"] not in _ML_MODELS]
        panels.append({
            "panel": panel, "letter": "" if single else _FACULTY_PANEL_LABEL[faculty],
            "faculty": faculty,
            ## As in the model comparison: the last bar ends at the right edge,
            ## and a slim margin on the left — wider bars, room for the values.
            "xlim": [round(min(xs_sa, default=0.0) - _BAR_W / 2 - 0.05, 6),
                     round(len(_CONFIG_ORDER) - 1 + len(_ML_MODELS) / 2 * _BAR_W, 6)],
            ## Two kinds of tick: the standalone bars are named under themselves,
            ## the grid under its feature set, and only the latter is angled.
            ## At the groups' spacing the standalone names no longer fit upright
            ## side by side, so they slant like the feature-set names.
            "ticks": ([{"x": r["x"], "label": r["tick_label"].replace("\n", " "),
                        "rotation": 25, "ha": "right"}
                       for r in rows if r["panel"] == panel and r["series"] not in _ML_MODELS]
                      + [{"x": float(pos["grid"][j]), "label": feature_set_label(fs),
                          "rotation": 25, "ha": "right"}
                         for j, fs in enumerate(_CONFIG_ORDER)]),
        })

    ## One y-range for both panels, so a bar means the same height in each. The
    ## floor is chance (0.5) unless something drawn sits below it — a rule can
    ## rank worse than chance, and a bar under the axis floor is invisible while
    ## its printed value lands on the tick labels.
    drawn = [r for r in rows if not r["hidden"]]
    vals  = [r["value"] for r in drawn]
    vals += [r[k] for r in drawn for k in ("ci_low", "ci_high") if r[k] == r[k]]
    ## Headroom for the value labels above the bars, as in the model comparison.
    ymax  = min(1.06, round((max(vals) + 0.06) * 20) / 20) if vals else 1.06
    ymin  = min(0.5, np.floor((min(vals) - 0.02) * 20) / 20) if vals else 0.5
    for spec in panels:
        spec["ylim"] = [round(float(ymin), 6), round(float(ymax), 6)]
        spec["hidden_y"] = round(float(ymin) + 0.02, 6)
        spec["chance_line"] = None   # not drawn

    df = pd.DataFrame(rows)
    meta = {
        "figsize": canvas(3.4 / 13) if single else canvas(5.5 / 13),
        "subplots_adjust": ({"top": 0.74, "bottom": 0.16, "left": 0.07, "right": 0.99}
                            if single else
                            {"top": 0.85, "bottom": 0.08, "left": 0.07,
                             "right": 0.99, "hspace": 0.45}),
        "ylabel": "Activation-level ROC AUC",
        "panels": panels,
        "legend": ([{"series": "credits_policy", "label": "Credits policy"}]
                   + [{"series": "operational_rule", "label": "Operational rule-based model"}]
                   + [{"series": "fasterrisk_used_vars",
                       "label": "FasterRisk (advisor variables)", "hatch": True}]
                   + [{"series": m, "label": _MODEL_DISPLAY.get(m, m)} for m in _ML_MODELS]
                   + [{"series": MAX_ML_MODEL, "label": "Max ML", "highlight": True}]),
        "marker": disclosure.SUPPRESSED,
        "note": disclosure.note_text() if bool(df["hidden"].any()) else None,
        "text_columns": ["value_label", "tick_label", "series", "faculty", "panel_letter",
                         "highlight"],
    }
    return df, meta



# ── Points profile ────────────────────────────────────────────────────────────

def prepare_points_profile(ctx: dict, faculties: list[str] | None = None,
                           show_threshold: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    Dropout rate and enrollment count by the pilot's points score, per faculty.

    `show_threshold=False` leaves out the dashed outreach-threshold line. Faculty
    B's figure is drawn that way: its threshold cannot be recovered reliably
    from who was contacted, so the paper reports only threshold-free measures
    (AUC, ABROCA) for Faculty B and prints no Faculty B threshold anywhere.

    The score bins are merged, not suppressed: a bin resting on too few students,
    dropouts or non-dropouts is joined to a neighbour until every bin passes, and
    merges never cross a published threshold. Nothing is hidden, so nothing can
    be recovered by subtracting the drawn bins from the faculty total — which is
    why this input may carry counts, and why the figure prints them.
    """
    from src.metrics import _enrollment_quantile_threshold
    from src.pilot_snapshot_comparison import (
        _check_threshold_disclosable, _merge_points_bins, _operational_model_threshold,
    )

    faculties = list(FACULTIES) if faculties is None else faculties
    single = len(faculties) == 1
    rows, panels = [], []
    for col, faculty in enumerate(faculties):
        subset = ctx["faculty_data"][faculty]["subset"]

        ## The threshold the pilot actually used, recovered as the lowest score
        ## among contacted students — an empirical fact, not a calibration.
        thresh = _operational_model_threshold(subset, faculty)
        if show_threshold:
            _check_threshold_disclosable(subset, faculty, thresh)
        ## Bins never merge across either threshold, drawn or not, so the
        ## panels are the same with or without the line.
        tau_cal = _enrollment_quantile_threshold(
            subset.rename(columns={"points_total": "y_prob"}), ctx["train_pos_rate"])
        boundaries = sorted({int(np.ceil(t)) for t in (thresh, tau_cal)
                             if t is not None and np.isfinite(t) and 0 < t <= 15})
        bins = _merge_points_bins(subset, boundaries)

        for b in bins:
            rows.append({
                "panel": col, "faculty": faculty,
                "lo": b["lo"], "hi": b["hi"],
                "mid": (b["lo"] + b["hi"]) / 2,
                ## A merged bin is drawn wider, which is how the figure says a
                ## merge happened without naming the bins it joined.
                "width": b["hi"] - b["lo"] + 0.8,
                "n": b["n"],
                "rate": b["k"] / b["n"] if b["n"] else np.nan,
                "count_label": str(b["n"]) if b["n"] else "",
            })
        panels.append({
            "panel": col, "faculty": faculty, "letter": _FACULTY_PANEL_LABEL[faculty],
            ## The line marks the edge of the first contacted score, so it sits
            ## half a point below it. None only when nobody at the faculty was
            ## contacted.
            "threshold_x": None if (thresh is None or not show_threshold) else thresh - 0.5,
            "merged": any(b["hi"] > b["lo"] for b in bins),
        })

    df = pd.DataFrame(rows)
    merged_panels = [p["panel"] for p in panels if p["merged"]]
    return df, {
        ## One faculty: dropout rate and count side by side across the width.
        "figsize": canvas(3.6 / 13) if single else canvas(8.67 / 13),
        "layout": "row" if single else "grid",
        "xticks": list(range(16)),
        "xlabel": "Total points",
        "ylabel_rate": "Dropout rate",
        "ylabel_count": "N",
        "rate_ylim": [0, 1.05],
        "panels": panels,
        "merged_note_panels": merged_panels,
        "note": (f"wide bars: scores merged ({disclosure.SUPPRESSED} students)"
                 if merged_panels else None),
        "text_columns": ["count_label", "faculty"],
    }



# ── ABROCA heatmap ────────────────────────────────────────────────────────────

_ABROCA_COLS = ["Credits\npolicy", "Operational\nrule-based\nmodel", "FasterRisk\n(advisor\nvariables)",
                "Participatory", "Max ML"]


def prepare_abroca(ctx: dict) -> tuple[pd.DataFrame, dict]:
    """
    ABROCA for each fairness axis against each decision rule, per faculty.

    An ABROCA rests on each group's dropouts AND non-dropouts, since it is the
    area between their two ROC curves — so a cell whose either group is small is
    not computed into the figure at all, and the marker takes its place. The
    empty cells that are simply not applicable say "n/a" instead: a rule that was
    never run on this faculty is not a small group.
    """
    from src.pilot_snapshot_comparison import (
        _FAIRNESS_BINARY_SUBGROUPS, _abroca_hidden, _compute_abroca, _flagged_enr,
    )

    subgroups = list(_FAIRNESS_BINARY_SUBGROUPS)
    rows = []
    for panel, faculty in enumerate(FACULTIES):
        data   = ctx["faculty_data"][faculty]
        subset = data["subset"].copy()
        preds  = data["preds_df"]

        ## Both rules enter as the CONTINUOUS score behind them, not as their
        ## thresholded flag: outreach was decided from the points scorecard, and
        ## the binary contact column is only its operating point.
        subset["_cp_flag"] = (subset["ects_behind"] >= 60).astype(int)
        col_enrs = [_flagged_enr(subset, "_cp_flag", "ects_behind"),
                    _flagged_enr(subset, _FACULTY_CONTACT[faculty], "points_total")]

        if ctx["fasterrisk_used_vars"] is not None:
            y_prob = ctx["fasterrisk_used_vars"].predict_proba(subset[USED_VARS])[:, 1]
            col_enrs.append(enrollment_predictions_df(subset, y_prob,
                                                      ctx["train_pos_rate"], target=TARGET))
        else:
            col_enrs.append(None)

        for model, fs in ((ctx["ws_model"], ctx["ws_fs"]), (MAX_ML_MODEL, MAX_ML_FS)):
            hit = (preds[(preds["model"] == model) & (preds["feature_set"] == fs)].copy()
                   if model and fs and len(preds) and "model" in preds.columns
                   else None)
            col_enrs.append(hit if hit is not None and len(hit) else None)

        for r, sg in enumerate(subgroups):
            ref_val, ref_label, comp_val, comp_label = _FAIRNESS_BINARY_SUBGROUPS[sg]
            for c, enr in enumerate(col_enrs):
                value, hidden, applicable = np.nan, False, True
                if enr is None or sg not in getattr(enr, "columns", []):
                    applicable = False
                else:
                    hidden = bool(_abroca_hidden(enr, sg, ref_val, comp_val))
                    if not hidden:
                        value = _compute_abroca(enr, sg, ref_val, comp_val)[3]
                rows.append({
                    "panel": panel, "faculty": faculty,
                    "row": r, "col": c, "subgroup": sg,
                    "row_label": f"{comp_label} vs {ref_label}",
                    "col_label": _ABROCA_COLS[c].format(faculty=_FACULTY_DISPLAY[faculty]),
                    "value": value, "hidden": hidden, "applicable": applicable,
                })

    df = pd.DataFrame(rows)
    vals = df.loc[~df["hidden"], "value"].dropna()
    vmax = float(vals.max()) if len(vals) and vals.max() > 0 else 1.0
    return df, {
        "figsize": canvas(4 / 13),
        "vmax": round(vmax, 6),
        "cmap": ["#ffffff", "#D55E00"],
        "cbar_label": "ABROCA",
        "panels": [{"panel": i, "faculty": f, "letter": _FACULTY_PANEL_LABEL[f],
                    "ylabels": i == 0}
                   for i, f in enumerate(FACULTIES)],
        "marker": disclosure.SUPPRESSED,
        "not_applicable": "n/a",
        "text_columns": ["subgroup", "row_label", "col_label", "faculty"],
    }



# ── FasterRisk scorecard ──────────────────────────────────────────────────────

_SCORECARD_ROW_H = 0.32   # inches per row; the figure is the only thing that
                          # knows how many rows it has, so it sets its own height


def prepare_scorecard(ctx: dict) -> tuple[pd.DataFrame, dict]:
    """
    The integer risk score the counsellors' nine variables buy.

    Model coefficients, not student counts, so nothing here is suppressed. The
    row labels carry each term's cut, which is a median of the whole training
    population.

    The saved model is LOADED, never retrained: Step 1 is its only writer, so
    every artefact describes the model the results table and its CIs describe.
    """
    from src.plotting import _clean_feature_name

    pipeline = ctx["fasterrisk_used_vars"]
    if pipeline is None:
        raise SystemExit("no results/models/fasterrisk__used_vars.joblib — "
                         "run src/evaluation.py first")
    clf = pipeline["clf"]
    try:
        raw_names = list(pipeline["prep"].get_feature_names_out())
    except Exception:                                               # noqa: BLE001
        raw_names = [f"feature_{i}" for i in range(len(clf.coef_))]
    names = [_clean_feature_name(n) for n in raw_names]
    binarizer = pipeline.named_steps.get("binarize")
    if hasattr(binarizer, "cut_labels"):
        names = binarizer.cut_labels(names)

    ## Printed names, not column names; the cut stays in the label.
    names = feature_labels(names)
    card = clf.get_scorecard(names)
    if card.empty:
        raise SystemExit("scorecard is empty — the model has no non-zero coefficients")

    df = pd.DataFrame({
        "y": range(len(card)),
        "label": card["feature"].tolist(),
        "score": [int(s) for s in card["score"]],
        ## A typographic minus, as on the axis ticks.
        "score_label": [str(int(s)).replace("-", "\u2212") for s in card["score"]],
    })
    ## FasterRisk is sparse, so the scorecard holds only its non-zero terms and
    ## the height follows the rows that are actually drawn — sizing off the
    ## coefficient vector buys empty space for bars that do not exist.
    n = len(df)
    height = max(2.6, _SCORECARD_ROW_H * n + 1.0)
    return df, {
        ## Printed at 0.8\\textwidth. The height follows the row count, so the
        ## aspect ratio is computed from the design height rather than fixed.
        "figsize": canvas(height / 7, width_fraction=0.8),
        "xlabel": "Integer score  (positive = higher dropout risk)",
        "intercept_note": f"intercept = {float(card['intercept'].iloc[0]):+.0f}".replace("-", "\u2212"),
        "text_columns": ["label", "score_label"],
    }


# ── Driver ────────────────────────────────────────────────────────────────────

def _faculty(letter: str) -> str:
    """The pipeline's code for the paper's Faculty `letter` — never spelled out here."""
    return next(f for f in FACULTIES if _FACULTY_DISPLAY[f] == f"Faculty {letter}")


_FIGURES = {
    ## Faculty A only: (a) real thresholds, (b) recalibrated, side by side.
    "breakdown_faculty_a":    ("pilot_breakdown_faculty_a",
                               lambda c: prepare_breakdown_faculty(c, _faculty("A"))),
    "points_faculty_a":       ("pilot_points_profile_faculty_a",
                               lambda c: prepare_points_profile(c, [_faculty("A")])),
    ## No threshold line for Faculty B: its outreach threshold is not known.
    "points_faculty_b":       ("pilot_points_profile_faculty_b",
                               lambda c: prepare_points_profile(c, [_faculty("B")],
                                                                show_threshold=False)),
    ## Main text: Faculty A alone. SI: Faculty B alone.
    "auc_faculty_a":          ("pilot_auc_faculty_a", lambda c: prepare_auc(c, [_faculty("A")])),
    "auc_faculty_b":          ("pilot_auc_faculty_b", lambda c: prepare_auc(c, [_faculty("B")])),
    "abroca":                 ("pilot_abroca", prepare_abroca),
    "scorecard":              ("scorecard_used_vars", prepare_scorecard),
}


## Every prepared file says A / B (the paper's Faculty A / B).
FACULTY_KEY = {f: _FACULTY_DISPLAY[f].removeprefix("Faculty ") for f in FACULTIES}


def anonymise(df: pd.DataFrame, meta: dict) -> tuple[pd.DataFrame, dict]:
    """Every faculty code in a prepared input replaced by its letter, then checked."""
    import json
    import re

    df = df.copy()
    if "faculty" in df.columns:
        df["faculty"] = df["faculty"].map(lambda f: FACULTY_KEY.get(f, f))

    def walk(obj):
        if isinstance(obj, dict):
            return {k: (FACULTY_KEY.get(v, v) if k == "faculty" else walk(v))
                    for k, v in obj.items()}
        if isinstance(obj, list):
            return [walk(v) for v in obj]
        return obj

    meta = walk(meta)
    codes = re.compile(r"\b(" + "|".join(FACULTY_KEY) + r")\b")
    text = df.to_csv(index=False) + json.dumps(meta, default=str)
    if codes.search(text):
        raise SystemExit(f"REFUSING to write: a faculty code survives in the input "
                         f"({sorted(set(codes.findall(text)))})")
    return df, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--figure", default="all", choices=["all", *_FIGURES])
    args = ap.parse_args()

    wanted = list(_FIGURES) if args.figure == "all" else [args.figure]
    ctx = load_pilot()
    for name in wanted:
        stem, fn = _FIGURES[name]
        df, meta = anonymise(*fn(ctx))
        write_input(stem, df, meta, source="data/pilot_snapshot.parquet + results/models/",
                    repo=ROOT, script=SCRIPT)


if __name__ == "__main__":
    main()
