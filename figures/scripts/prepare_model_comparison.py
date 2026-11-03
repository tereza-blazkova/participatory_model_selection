"""
Distil the model-comparison figure — the paper's headline four-panel composite —
and its five-panel variant with the October 2023 pilot added.

    figures/input/model_comparison.csv    every drawn mark, by panel and kind
    figures/input/model_comparison.json   axes, ticks, legends, labels
    figures/input/model_comparison_2021_2023.{csv,json}
                                          the same, plus panel e; (d) compacted

    a  activation-level ROC AUC for every model x feature set, against the credits policy
    b  what each of the three decision rules does to the cohort, as shares
    c  precision-recall curves with iso-contact-rate budget lines
    d  ABROCA per fairness axis per rule

Four panels, four kinds of disclosure risk, all resolved here:
  a  a bar rests on the year's dropouts and non-dropouts
  b  each bar is a 2x2 table whose margins are on the figure, so a small cell is
     merged with its same-outcome partner into one grey block
  c  every point of a PR curve is an operating point; enrollment_pr_curve_data
     drops the ones whose confusion cells are too small, and an operating-point
     marker is withheld on the same rule
  d  an ABROCA rests on each group's dropouts and non-dropouts
  e  (variant only) taken from figures/input/pilot_auc_faculty_a.csv, where the
     pilot figure's suppression was already applied — so prepare_pilot.py must
     run first, and the two figures withhold the same bars

    python figures/scripts/prepare_model_comparison.py [--test-year 2021]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from paths.paths import RESULTS_PATH                                   # noqa: E402
from prepare_utils import workshop_selection, write_input              # noqa: E402
from src import disclosure                                             # noqa: E402
from src.config import MAIN_TEST_YEAR, MAX_ML_FS, MAX_ML_MODEL         # noqa: E402
from src.display import feature_set_label                                # noqa: E402
from src.data import load_analysis_df                                  # noqa: E402
from src.fairness_audit import _BINARY_SUBGROUPS                       # noqa: E402
from src.metrics import _attach_subgroup_cols, enrollment_pr_curve_data  # noqa: E402
from utils import canvas                                               # noqa: E402
from src.plotting import (                                             # noqa: E402
    _CONFIG_ORDER, _MODEL_DISPLAY, _ML_MODELS, _compute_abroca, _enr_breakdown,
    _load_workshop_underlying, _op_point_hidden, _point_at_budget,
    _raw_credits_enr, _workshop_underlying_name,
)

SCRIPT = "prepare_model_comparison.py"

_BAR_W = 0.185  # four bars fill 74% of a feature-set group
## (d)'s rows in the 2021-2023 variant: the axis, not the pair, so the column
## names can stand horizontally. The pairs go in the caption.
_AXIS_NAME = {"female": "Sex", "danish_nationality": "Nationality",
              "older_for_cohort": "Entry age"}


def _rows_frame(rows: list[dict]) -> pd.DataFrame:
    """Long table; every panel contributes its own kinds, unused columns stay empty."""
    return pd.DataFrame(rows)


def prepare(results_df, preds_df, workshop_df, raw_df, test_year, budget_rates):
    rows: list[dict] = []

    def add(panel, kind, **kw):
        rows.append({"panel": panel, "kind": kind, **kw})

    yr_preds = preds_df[preds_df["test_year"] == test_year]
    ml_res = results_df[(results_df["test_year"] == test_year)
                        & (results_df["model"].isin(_ML_MODELS))]
    if len(ml_res) == 0:
        raise SystemExit(f"no ML results for test_year={test_year}")

    best_enr = yr_preds[(yr_preds["model"] == MAX_ML_MODEL)
                        & (yr_preds["feature_set"] == MAX_ML_FS)]
    if "max_prob" not in best_enr.columns or best_enr["max_prob"].isna().all():
        raise SystemExit("max_prob missing — re-run src/evaluation.py")

    cp = _enr_breakdown(yr_preds[yr_preds["model"] == "credits_policy"])
    cp_flag_rate = (cp["tp"] + cp["fp"]) / cp["n"] if cp["n"] else 0.0
    ws_enr, ws_model, ws_fs = _load_workshop_underlying(preds_df, test_year)

    # ── (a) activation-level ROC AUC ────────────────────────────────────────────────
    cis = pd.DataFrame()
    cis_path = RESULTS_PATH / "bootstrap_cis.csv"
    if cis_path.exists():
        allc = pd.read_csv(cis_path)
        if "metric" in allc.columns:
            cis = allc[(allc["metric"] == "row_roc_auc")
                       & (allc["test_year"] == test_year)].reset_index(drop=True)

    n_models = len(_ML_MODELS)
    x_pos = np.arange(len(_CONFIG_ORDER))
    group_gap = 1.0 - n_models * _BAR_W
    first_left = -(n_models - 1) / 2 * _BAR_W - _BAR_W / 2
    x_cp = first_left - group_gap - _BAR_W / 2

    ## The workshop config stores a display label; resolve the estimator behind it.
    ws_model_a, ws_fs_a = ws_model, ws_fs
    if ws_model_a is None and workshop_df is not None and len(workshop_df):
        yr = workshop_df[workshop_df["test_year"] == test_year]
        if len(yr):
            m, f = _workshop_underlying_name()
            ws_model_a = m or yr["model"].values[0]
            ws_fs_a = f or yr["feature_set"].values[0]

    def ci_for(model, fs):
        if not len(cis):
            return np.nan, np.nan
        hit = cis[(cis["model"] == model) & (cis["feature_set"] == fs)]
        return ((float(hit.iloc[0]["ci_low"]), float(hit.iloc[0]["ci_high"]))
                if len(hit) == 1 else (np.nan, np.nan))

    vals_a = []
    rule_a = results_df[(results_df["test_year"] == test_year)
                        & (results_df["model"] == "credits_policy")]
    if len(rule_a) and "row_roc_auc" in rule_a.columns and not pd.isna(rule_a["row_roc_auc"].values[0]):
        v = float(rule_a["row_roc_auc"].values[0])
        add("a", "bar", series="credits_policy", x=x_cp, y=v, ci_low=np.nan, ci_high=np.nan,
            value_label=f"{v:.2f}", label_y=v, highlight="")
        vals_a.append(v)

    for i, model in enumerate(_ML_MODELS):
        series = (ml_res[ml_res["model"] == model]
                  .set_index("feature_set")["row_roc_auc"].reindex(_CONFIG_ORDER))
        offset = (i - (n_models - 1) / 2) * _BAR_W
        for j, (fs, v) in enumerate(zip(_CONFIG_ORDER, series)):
            if pd.isna(v):
                continue
            v = float(v)
            is_ws = (ws_model_a is not None and model == ws_model_a and fs == ws_fs_a)
            is_max = (model == MAX_ML_MODEL and fs == MAX_ML_FS)
            lo, hi = ci_for(model, fs)
            vals_a.append(v)
            if hi == hi:
                vals_a += [lo, hi]
            add("a", "bar", series=model, x=float(x_pos[j] + offset), y=v,
                ci_low=lo, ci_high=hi, value_label=f"{v:.2f}",
                ## The printed number clears the whisker where there is one.
                label_y=hi if hi == hi else v,
                ## Max ML wins the dashed outline when a cell is both.
                highlight="max_ml" if is_max else ("participatory" if is_ws else ""))

    ymax_a = min(1.05, round((max(vals_a) + 0.06) * 20) / 20) if vals_a else 1.05
    last_right = x_pos[-1] + (n_models - 1) / 2 * _BAR_W + _BAR_W / 2

    # ── (b) outcome breakdown ────────────────────────────────────────────────
    ## All three bars from the per-student predictions, the source panel c
    ## draws from too, so the two panels cannot disagree. (TP is never rebuilt
    ## from a 3-decimal precision, which can be a student off.)
    bars_b = [("Credits policy\n(>=60 ECTS behind)", cp)]
    if ws_enr is not None:
        bars_b.append(("Participatory", _enr_breakdown(ws_enr)))
    bars_b.append(("Max ML", _enr_breakdown(best_enr)))

    for xi, (label, c) in enumerate(bars_b):
        n = c["n"]
        hide_non = disclosure.is_small(c["tn"], c["fp"])
        hide_do = disclosure.is_small(c["tp"], c["fn"])
        recall = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
        flag_r = (c["tp"] + c["fp"]) / n if n else 0.0
        ## Two words go on two lines: three bars share the panel.
        base = label.split("\n")[0].replace(" ", "\n", 1)
        tick = (base + f"\nrecall {disclosure.SUPPRESSED}\nflags {disclosure.SUPPRESSED}"
                if hide_non or hide_do else
                base + f"\nrecall {recall:.0%}\nflags {flag_r:.0%}")
        r = {"x": float(xi), "tick": tick, "empty": n == 0,
             "frac_tn": np.nan, "frac_fp": np.nan, "frac_tp": np.nan, "frac_fn": np.nan,
             "block_nondropout": np.nan, "block_dropout": np.nan,
             "hide_nondropout": hide_non, "hide_dropout": hide_do}
        if n:
            share = {k: c[k] / n for k in ("tn", "fp", "tp", "fn")}
            if hide_non:
                r["block_nondropout"] = share["tn"] + share["fp"]
            else:
                r["frac_tn"], r["frac_fp"] = share["tn"], share["fp"]
            if hide_do:
                r["block_dropout"] = share["tp"] + share["fn"]
            else:
                r["frac_tp"], r["frac_fn"] = share["tp"], share["fn"]
        add("b", "bar", **r)

    # ── (c) PR curves ────────────────────────────────────────────────────────
    def curve(series, enr, style):
        rec, prec = enrollment_pr_curve_data(enr)
        for order, (x, y) in enumerate(zip(rec, prec)):
            add("c", "curve", series=series, order=order, x=float(x), y=float(y))
        c = _enr_breakdown(enr)
        if not _op_point_hidden(c):
            add("c", "operating_point", series=series,
                x=c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0,
                y=c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0,
                style=style)

    curve("max_ml", best_enr, "max_ml")
    if ws_enr is not None:
        curve("participatory", ws_enr, "participatory")

    iso_enr = None
    if raw_df is not None:
        iso_enr = _raw_credits_enr(raw_df, test_year)
        rec, prec = enrollment_pr_curve_data(iso_enr)
        for order, (x, y) in enumerate(zip(rec, prec)):
            add("c", "curve", series="credits_policy", order=order, x=float(x), y=float(y))
        ## The legal threshold as an operating point on its own curve.
        flagged = iso_enr["max_prob"] >= 60
        n, nd = len(iso_enr), int((iso_enr["dropped_out"] == 1).sum())
        tp = int((flagged & (iso_enr["dropped_out"] == 1)).sum())
        fp = int((flagged & (iso_enr["dropped_out"] == 0)).sum())
        if not disclosure.operating_point_hidden(n, nd, tp + fp, tp)["recall"]:
            add("c", "operating_point", series="credits_policy_60",
                x=tp / nd if nd else 0.0, y=tp / (tp + fp) if (tp + fp) else 0.0,
                style="credits_60",
                value_label=f">=60 ECTS threshold  (flags {(tp + fp) / n:.0%})")

    ## Iso-budget lines: the same contact rate across every method, so the panel
    ## can be read as "at this budget, who does better" rather than "at each
    ## method's own threshold".
    avail = [e for e in (best_enr, ws_enr, iso_enr) if e is not None]
    budgets = list(budget_rates) + [cp_flag_rate]
    labels_b = [f"{r:.0%} flagged" for r in budget_rates] + ["Credits policy\nbudget"]
    label_pts = []
    for i, (rate, blabel) in enumerate(zip(budgets, labels_b)):
        pts = [p for p in (_point_at_budget(e, rate) for e in avail) if np.isfinite(p[0])]
        if not pts:
            continue
        for order, (x, y) in enumerate(pts):
            add("c", "budget", series=str(i), order=order, x=float(x), y=float(y))
        label_pts.append([float(np.mean([p[0] for p in pts])),
                          float(np.mean([p[1] for p in pts])), blabel, i])

    ## Each label sits at the mean of its line's points. Overlaps are resolved in
    ## draw_model_comparison.py, which measures the labels and separates them; a
    ## fixed gap here could not know how large the text is.
    for lx, ly, blabel, i in label_pts:
        add("c", "budget_label", series=str(i), x=lx, y=ly, value_label=blabel)

    # ── (d) ABROCA heatmap ───────────────────────────────────────────────────
    iso_d = None
    if raw_df is not None:
        iso_d = _attach_subgroup_cols(_raw_credits_enr(raw_df, test_year),
                                      raw_df[raw_df["start_year"] == test_year])
    col_defs = [("Credits policy", iso_d),
                ("Participatory" if ws_model else "Participatory (n/a)", ws_enr),
                ("Max ML", best_enr)]
    sg_keys = list(_BINARY_SUBGROUPS)
    for r, sg in enumerate(sg_keys):
        ref_v, ref_l, comp_v, comp_l = _BINARY_SUBGROUPS[sg]
        for c_i, (col_label, enr) in enumerate(col_defs):
            value, hidden, applicable = np.nan, False, True
            if enr is None or sg not in getattr(enr, "columns", []):
                applicable = False
            else:
                hidden = bool(disclosure.roc_hidden(enr, sg, (ref_v, comp_v)))
                if not hidden:
                    ## Unsigned (src/metrics.abroca): how far apart the two
                    ## groups' ROC curves lie, not which group ranks better.
                    value = _compute_abroca(enr, sg, ref_v, comp_v)[3]
            add("d", "cell", row=r, col=c_i, y=value, hidden=hidden,
                applicable=applicable,
                row_label=f"{comp_l} vs {ref_l}", tick=col_label)

    df = _rows_frame(rows)
    d_vals = df.loc[(df["panel"] == "d") & ~df["hidden"].fillna(False), "y"].dropna()
    vmax_d = float(np.nanmax(d_vals)) if len(d_vals) else 1.0

    meta = {
        "figsize": canvas(8.0 / 13),
        "height_ratios": [4.0, 7],
        "panel_a": {
            "letter": "a", "letter_x": -0.035,
            "ylabel": "Activation-level ROC AUC",
            "ylim": [0.5, round(float(ymax_a), 6)],
            ## The last bar ends at the right edge of the panel, and so of the figure.
            "xlim": [x_cp - _BAR_W / 2 - 0.12, float(last_right)],
            "ticks": ([{"x": x_cp, "label": "credits\npolicy", "rotation": 0, "ha": "center"}]
                      + [{"x": float(x_pos[j]), "label": feature_set_label(fs),
                          "rotation": 0, "ha": "center"}
                         for j, fs in enumerate(_CONFIG_ORDER)]),
            "legend": ([{"series": "credits_policy", "label": "Credits policy"}]
                       + [{"series": m, "label": _MODEL_DISPLAY.get(m, m)} for m in _ML_MODELS]
                       + ([{"series": ws_model_a, "label": "Participatory",
                            "highlight": "participatory"}] if ws_model_a else [])
                       + [{"series": MAX_ML_MODEL, "label": "Max ML", "highlight": "max_ml"}]),
        },
        "panel_b": {
            "letter": "b", "ylabel": "Share of enrollments",
            "dropout_line": round(1.0 - cp["dropout_rate"], 6),
            "side_labels": {"upper": "Dropout", "lower": "Non-dropout"},
            "xlim": [-0.95, len(bars_b) - 0.5],
        },
        "panel_c": {
            "letter": "c", "xlabel": "Enrollment recall",
            "ylabel": "Enrollment precision", "lim": [0, 1.02],
            "legend_order": ["", "Participatory", "Credits policy", "Max ML"],
            ## The Participatory curve is drawn in its model's colour, as its
            ## bar is in (a); null when there is no Participatory model.
            "participatory_model": ws_model_a,
        },
        "panel_d": {
            "letter": "d", "vmax": round(vmax_d, 6), "cbar_label": "ABROCA",
            "row_labels": [f"{_BINARY_SUBGROUPS[s][3]} vs {_BINARY_SUBGROUPS[s][1]}"
                           for s in sg_keys],
            "col_labels": [lbl for lbl, _ in col_defs],
        },
        "marker": disclosure.SUPPRESSED,
        "not_applicable": "n/a",
        "text_columns": ["panel", "kind", "series", "style", "value_label", "tick",
                         "row_label"],
    }
    return df, meta


def with_pilot_panel(df: pd.DataFrame, meta: dict, pilot_csv: Path):
    """
    The 2021-2023 variant: (d) compacted to sit above a new panel e, the
    October 2023 pilot at Faculty A — credits policy, operational rule-based
    model, FasterRisk on the advisor variables, Participatory, Max ML.
    """
    if not pilot_csv.exists():
        raise SystemExit(f"{pilot_csv} missing — run prepare_pilot.py first")
    pilot = pd.read_csv(pilot_csv, dtype={"value_label": str, "highlight": str})
    pilot_ylim = json.loads(pilot_csv.with_suffix(".json").read_text())["panels"][0]["ylim"]
    pick = [("Credits policy", pilot["series"] == "credits_policy"),
            ("Operational rule (OR)", pilot["series"] == "operational_rule"),
            ("FasterRisk (OR vars)", pilot["series"] == "fasterrisk_used_vars"),
            ("Participatory", pilot["highlight"] == "participatory"),
            ("Max ML", pilot["highlight"] == "max_ml")]
    rows = []
    for i, (tick, mask) in enumerate(pick):
        sel = pilot[mask]
        if len(sel) != 1:
            raise SystemExit(f"pilot AUC input: expected one row for {tick!r}, got {len(sel)}"
                             " (there is no Participatory row if the Credits policy was retained)")
        p = sel.iloc[0]
        hidden = str(p["hidden"]) == "True"
        rows.append({"panel": "e", "kind": "bar", "series": p["series"], "x": float(i),
                     "y": np.nan if hidden else float(p["value"]),
                     "ci_low": np.nan if hidden else p["ci_low"],
                     "ci_high": np.nan if hidden else p["ci_high"],
                     "value_label": None if hidden else p["value_label"],
                     "highlight": p["highlight"] if isinstance(p["highlight"], str) else None,
                     "hidden": hidden, "tick": tick})
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)

    meta = json.loads(json.dumps(meta))   # a deep copy; the base figure keeps its own
    meta["panel_d"]["row_labels"] = [_AXIS_NAME[s] for s in _BINARY_SUBGROUPS]
    ## Wrapped: horizontal, "Credits policy" ran into "Participatory".
    meta["panel_d"]["col_labels"] = [lbl.replace("Credits policy", "Credits\npolicy")
                                     for lbl in meta["panel_d"]["col_labels"]]
    meta["panel_e"] = {
        "letter": "e", "ylabel": "Pilot AUC", "ylim": pilot_ylim,
        "note": "vs. operational\nmodel (2023,\npartial year)",
    }
    return df, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-year", type=int, default=MAIN_TEST_YEAR)
    args = ap.parse_args()

    results_df = pd.read_csv(RESULTS_PATH / "evaluation_results.csv")
    preds_df = pd.read_csv(RESULTS_PATH / "enrollment_predictions.csv")
    workshop_df, _, _ = workshop_selection(RESULTS_PATH)

    df, meta = prepare(results_df, preds_df, workshop_df, load_analysis_df(),
                       args.test_year, (0.10, 0.20))
    write_input("model_comparison", df, meta,
                source="results/evaluation_results.csv + enrollment_predictions.csv",
                repo=ROOT, script=SCRIPT)
    df_e, meta_e = with_pilot_panel(df, meta, ROOT / "figures" / "input" / "pilot_auc_faculty_a.csv")
    write_input("model_comparison_2021_2023", df_e, meta_e,
                source="results/evaluation_results.csv + enrollment_predictions.csv"
                       " + figures/input/pilot_auc_faculty_a.csv",
                repo=ROOT, script=SCRIPT)


if __name__ == "__main__":
    main()
