"""
Re-reads the models on bachelor's enrollments only, as a robustness check on
the master's credit features.

The models are trained and evaluated on bachelor's and master's enrollments
together, and that is what the paper reports. Master's credit features may be
incomplete, so this checks how the reported numbers change when only
bachelor's enrollments are read.

This re-reads the SAME predictions on the bachelor's subset. It does not retrain
and it does not recalibrate: `ever_flagged` in enrollment_predictions.csv is the
deployed operating point, so every number here is "how the system as calibrated
actually behaves on bachelor's enrollments".

It is NOT the same as building the system for bachelors. That would retrain on
BA only, which changes the training dropout rate and therefore tau, and could
change the workshop selection — a different study, and a second set of numbers
to keep straight.

Metrics come from fairness_audit._compute_metrics, so they are the same
definitions and the same disclosure control as every other fairness output.

Writes results/descriptive/bachelor_only.tex — a table and the sentences that
go with it.

Usage:  python src/bachelor_only_audit.py
        python src/bachelor_only_audit.py --year 2021
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from paths.paths import RESULTS_PATH
from src import disclosure
from src.config import MAIN_TEST_YEAR, MAX_ML_FS, MAX_ML_MODEL
from src.fairness_audit import _compute_metrics, _resolve_participatory_model

OUT_PATH = RESULTS_PATH / "descriptive" / "bachelor_only.tex"

## (column value, label). `undergraduate` is 1 for BA, 0 for MA.
_POPULATIONS = [
    (None, "All enrollments"),
    (1,    "Bachelor's only"),
    (0,    "Master's only"),
]


def _methods(preds: pd.DataFrame) -> list[tuple[str, str, str]]:
    """(model, feature_set, label) for the three the paper compares."""
    ws_model, ws_fs = _resolve_participatory_model()
    ## The credits policy carries its rule name in feature_set
    ## ("ects_behind_ge_60"), not a feature set, so it is read off the frame
    ## rather than spelled here — hardcoding "" silently dropped it.
    cp_fs = preds.loc[preds["model"] == "credits_policy", "feature_set"]
    out = [("credits_policy", cp_fs.iloc[0] if len(cp_fs) else "", "Credits policy"),
           (ws_model, ws_fs, "Participatory"),
           (MAX_ML_MODEL, MAX_ML_FS, "Max ML")]
    available = set(zip(preds["model"], preds["feature_set"].fillna("")))
    return [m for m in out if (m[0], m[1]) in available]


def _rows(preds: pd.DataFrame, test_year: int) -> pd.DataFrame:
    records = []
    for model, fs, label in _methods(preds):
        sub = preds[(preds["model"] == model)
                    & (preds["feature_set"].fillna("") == fs)
                    & (preds["test_year"] == test_year)]
        for value, pop_label in _POPULATIONS:
            enr = sub if value is None else sub[sub["undergraduate"] == value]
            m = _compute_metrics(enr)
            records.append({
                "method": label, "population": pop_label,
                "n_enrollments": len(enr),
                "n_dropouts": int((enr["dropped_out"] == 1).sum()),
                **m,
            })
    return pd.DataFrame(records)


def _delta(df: pd.DataFrame, method: str, metric: str) -> float:
    """Bachelor's minus all, for one method — NaN if either side is suppressed."""
    sel = df[df["method"] == method].set_index("population")
    if not {"Bachelor's only", "All enrollments"} <= set(sel.index):
        return np.nan
    return sel.loc["Bachelor's only", metric] - sel.loc["All enrollments", metric]


def _moves(d: float) -> str:
    """The verb for a signed change; an exact zero neither rises nor falls."""
    if d == 0:
        return "is unchanged"
    return "rises" if d > 0 else "falls"


def _sentences(df: pd.DataFrame, test_year: int) -> str:
    """
    The one or two sentences that go under the table.

    Written from the numbers rather than around them: the claim the paper needs
    to make is about the SIZE of the shift, so the shift is what the sentence
    states, and its direction word is chosen from the sign.
    """
    parts = []
    for method in ("Participatory", "Max ML"):
        d_rec = _delta(df, method, "recall")
        d_auc = _delta(df, method, "auc")
        if np.isnan(d_rec) and np.isnan(d_auc):
            continue
        ## Each metric states its own direction. Sharing one verb read as "both
        ## fall" when recall and AUC move opposite ways.
        bits = []
        if not np.isnan(d_rec):
            bits.append(f"recall {_moves(d_rec)} by {abs(d_rec) * 100:.1f}~pp")
        if not np.isnan(d_auc):
            bits.append(f"AUC {_moves(d_auc)} by {abs(d_auc):.3f}")
        parts.append(f"for the {method} model, {' while '.join(bits)}")

    if not parts:
        return ("Every comparison in this table rests on too few students to show, so no "
                "bachelor's-only claim can be made from it.")

    return (
        "Restricting the evaluation to bachelor's enrollments, relative to the full "
        f"{test_year} test cohort: " + "; ".join(parts) + ". "
        "The models are unchanged: they are trained on both levels and scored at the "
        "threshold calibrated for the full cohort, so this isolates the effect "
        "of the population the metrics are read over, not the effect of building "
        "the system for bachelors alone."
    )


def _tex(df: pd.DataFrame, test_year: int) -> str:
    head = [
        r"% Generated by src/bachelor_only_audit.py — do not edit by hand.",
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        (r"Method & Population & Enrollments & Dropouts & Recall & Precision & Enr. AUC \\"),
        r"\midrule",
    ]
    lines = []
    for method in df["method"].unique():
        block = df[df["method"] == method]
        for i, (_, r) in enumerate(block.iterrows()):
            lines.append(
                f"{method if i == 0 else ''} & {r['population']} & "
                f"{disclosure.fmt_count(r['n_enrollments'])} & "
                f"{disclosure.fmt_count(r['n_dropouts'])} & "
                f"{disclosure.fmt(r['recall'], spec='{:.3f}') if not r['hidden_recall'] else disclosure.SUPPRESSED_TEX} & "
                f"{disclosure.fmt(r['precision'], spec='{:.3f}') if not r['hidden_precision'] else disclosure.SUPPRESSED_TEX} & "
                f"{disclosure.fmt(r['auc'], spec='{:.3f}') if not r['hidden_auc'] else disclosure.SUPPRESSED_TEX} \\\\"
            )
        lines.append(r"\addlinespace")
    caption = (
        r"\caption{\textbf{The same models re-read on the bachelor's subset, "
        + str(test_year) + r" test cohort.} "
        r"Master's ECTS credits behind are overstated by 30--60~ECTS because the "
        r"master's thesis is absent from the student-records extract, so credit coverage is "
        r"53\% of the nominal 120~ECTS against 85\% for bachelor's. Each model is "
        r"trained on both levels and scored at the threshold calibrated for the full "
        r"cohort; only the population the metrics are read over changes, so the table "
        r"isolates the effect of that population on the reported numbers rather than "
        r"asking what a bachelor's-only system would do.}"
    )
    ## The prose is emitted as prose, not as a comment: one \input gives the
    ## paragraph and the table together, so the sentence cannot drift from the
    ## numbers it describes.
    return "\n".join(
        [r"% Generated by src/bachelor_only_audit.py — do not edit by hand.",
         _sentences(df, test_year), ""]
        + head[1:] + lines
        + [r"\bottomrule", r"\end{tabular}", caption,
           r"\label{tab:bachelor_only}", r"\end{table}"]
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, default=MAIN_TEST_YEAR)
    args = ap.parse_args()

    path = RESULTS_PATH / "enrollment_predictions.csv"
    if not path.exists():
        print(f"{path} not found — run src/evaluation.py first.")
        return 1
    preds = pd.read_csv(path)
    if "undergraduate" not in preds.columns:
        print("enrollment_predictions.csv has no `undergraduate` column — "
              "re-run src/evaluation.py so metrics._SUBGROUP_COLS is applied.")
        return 1

    df = _rows(preds, args.year)
    if df.empty:
        print(f"No predictions for test_year={args.year}.")
        return 1

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(_tex(df, args.year) + "\n")

    print(f"\nBachelor's-only audit, {args.year}\n")
    show = df[["method", "population", "n_enrollments", "n_dropouts",
               "recall", "precision", "auc"]]
    print(show.to_string(index=False, na_rep="--",
                         float_format=lambda v: f"{v:.3f}"))
    print("\n" + _sentences(df, args.year))
    print(f"\nSaved: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
