"""
The October 2023 pilot at its real contact budget, Faculty A (main-text table).

The faculty contacted a fixed number of students in the pilot. This asks what
each method would have achieved with exactly that many contacts: each ranks the
faculty's enrollments by its own score and contacts the top k, where k is the
number of enrollments the pilot actually contacted (contact flag in the
snapshot — read from the data, never written in here).

    credits policy                    ects_behind
    operational rule-based model      the points score the faculty used
    FasterRisk (advisor variable set) its predicted probability on used_vars
    Participatory model               the stored workshop selection
    Max ML                            MAX_ML_MODEL on MAX_ML_FS

Ties at the cutoff — the points score is an integer, FasterRisk's score coarse —
are split in expectation (metrics.recall_precision_at_budget), so recall and
precision are expected values. ROC AUC stays as the threshold-free comparison;
on the snapshot (one activation per enrollment) it is the activation-level AUC
the pilot AUC figure draws.

Faculty B is reported in the SI for AUC only (the pilot AUC figure), not here.
The faculty is never named: "Faculty A" throughout.

Disclosure: recall and precision rest on all four confusion cells; they are
withheld when either whole-student bound of the expected TP would be
(operating_point_hidden). An AUC rests on the dropouts and non-dropouts it ranks.

    python src/make_pilot_budget_table.py
Writes results/descriptive/pilot_budget_table.tex
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from paths.paths import RESULTS_PATH, PILOT_SNAPSHOT_PARQUET
from src import disclosure
from src.config import MAX_ML_FS, MAX_ML_MODEL, TARGET, USED_VARS
from src.data import load_pilot_snapshot
from src.display import feature_set_label, model_label
from src.features import load_model_checked
from src.metrics import recall_precision_at_budget
from src.pilot_snapshot_comparison import (
    FACULTIES, _FACULTY_CONTACT, _FACULTY_DISPLAY, _load_metadata,
)
from src.workshop_rules import _RULE_BASED_MODELS, participatory_config

OUT = RESULTS_PATH / "descriptive" / "pilot_budget_table.tex"
_MODELS_DIR = RESULTS_PATH / "models"
FACULTY_LETTER = "A"


def _faculty(letter: str) -> str:
    return next(f for f in FACULTIES if _FACULTY_DISPLAY[f] == f"Faculty {letter}")


def _score(path_stem: str, features: list[str], subset: pd.DataFrame) -> np.ndarray:
    model = load_model_checked(_MODELS_DIR / f"{path_stem}.joblib", features)
    return model.predict_proba(subset[features])[:, 1]


def method_scores(subset: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
    """(row label, one score per enrollment), in the table's order."""
    available = _load_metadata()["available_features"]
    ws_model, ws_fs = participatory_config(RESULTS_PATH)
    if ws_model is None:
        raise FileNotFoundError("no stored Participatory model — run src/workshop_evaluation.py")
    if ws_model in _RULE_BASED_MODELS:
        raise ValueError("no combination was permissible, so the Credits policy is retained "
                         "and there is no Participatory model to score: this table has no "
                         "row for that case. Decide how the paper reports it first.")
    return [
        ("Credits policy", subset["ects_behind"].to_numpy(dtype=float)),
        ("Operational rule-based model", subset["points_total"].to_numpy(dtype=float)),
        ("FasterRisk (advisor variable set)",
         _score("fasterrisk__used_vars", available.get("used_vars", USED_VARS), subset)),
        (f"Participatory model ({model_label(ws_model)}, {feature_set_label(ws_fs)})",
         _score(f"{ws_model}__{ws_fs}", available[ws_fs], subset)),
        (f"Max ML ({model_label(MAX_ML_MODEL)}, {feature_set_label(MAX_ML_FS)})",
         _score(f"{MAX_ML_MODEL}__{MAX_ML_FS}", available[MAX_ML_FS], subset)),
    ]


def budget_rows(subset: pd.DataFrame, k: int,
                methods: list[tuple[str, np.ndarray]]) -> pd.DataFrame:
    y = subset[TARGET].to_numpy(dtype=int)
    rows = []
    for label, s in methods:
        m = recall_precision_at_budget(s, y, k)
        n, n_pos = m["n_enrollments"], m["n_dropout"]
        hidden = any(
            disclosure.operating_point_hidden(n, n_pos, k, tp)[metric]
            for tp in {m["tp_low"], m["tp_high"]} for metric in ("recall", "precision"))
        valid = ~np.isnan(s)
        y_v = y[valid]
        n_auc_pos, n_auc_neg = int(y_v.sum()), int((y_v == 0).sum())
        auc = (float(roc_auc_score(y_v, s[valid]))
               if n_auc_pos and n_auc_neg else np.nan)
        rows.append({"method": label, **m, "hidden": hidden, "auc": auc,
                     "auc_hidden": disclosure.is_small(n_auc_pos, n_auc_neg)})
    return pd.DataFrame(rows)


def build_table(rows: pd.DataFrame, k: int) -> str:
    X = disclosure.SUPPRESSED_TEX
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        rf"Method & Recall at {k:,} contacts & Precision at {k:,} contacts & ROC AUC \\",
        r"\midrule",
    ]
    for r in rows.itertuples():
        rec = X if r.hidden else f"{r.recall:.3f}"
        pre = X if r.hidden else f"{r.precision:.3f}"
        auc = X if r.auc_hidden else ("--" if np.isnan(r.auc) else f"{r.auc:.3f}")
        lines.append(f"{r.method} & {rec} & {pre} & {auc} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{\textbf{The October 2023 pilot at its real contact budget, Faculty "
        + FACULTY_LETTER + r".} Each method contacts the " + f"{k:,}" + r" enrollments "
        r"it ranks highest --- the number the faculty contacted in the pilot --- and "
        r"recall and precision are enrollment-level at that budget. Where a method's "
        r"score ties at the cutoff, the remaining contacts are spread evenly over the "
        r"tied enrollments, so recall and precision are expected values. ROC AUC is "
        r"threshold-free and is the activation-level AUC of the snapshot, which holds "
        r"one activation per enrollment.}",
        r"\label{tab:pilot_budget}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    tab = load_pilot_snapshot(PILOT_SNAPSHOT_PARQUET)
    faculty = _faculty(FACULTY_LETTER)
    subset = tab[tab["faculty_code"] == faculty].reset_index(drop=True)
    if subset["enrollment"].duplicated().any():
        raise ValueError("the snapshot holds more than one activation per enrollment; "
                         "the budget table assumes one")
    k = int(subset[_FACULTY_CONTACT[faculty]].sum())
    rows = budget_rows(subset, k, method_scores(subset))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build_table(rows, k))
    print(f"Faculty {FACULTY_LETTER}: budget {k} contacts")
    print(rows[["method", "recall", "precision", "auc", "tie_split", "hidden"]]
          .to_string(index=False))
    print(f"Saved → {OUT}")


if __name__ == "__main__":
    main()
