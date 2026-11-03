"""
Distil results/evaluation_results.csv into the selection-gate figure's input.

This is the half that is allowed to reach into the pipeline. It runs the workshop
selection for one test year, reconstructs what each rule demanded of each cell,
applies statistical disclosure control, and writes the result as a small table of
DRAWN VALUES:

    figures/input/selection_gate_{year}.csv    one row per (model, feature set)
    figures/input/selection_gate_{year}.json   the figure's scalars + provenance

The logic is not reimplemented here. src/plot_selection_gate.py owns it and
cross-checks itself against select_grid_config(), so this script calls that code
and serialises the answer.

BOTH artefacts read the result: the figure (figures/scripts/draw_selection_gate.py)
and its table twin, tab:selection_gate (src/generate_selection_table.py). That is
the point — the suppression decision is made once, here, so the pair cannot
disagree about which cells are too small to show: the figure and the table read
this one file, so they withhold the same cells.

The table shows two things the figure does not, so the CSV carries them: the
enrollment-level precision of each cell, and the credits-policy baseline as a row
of its own (`role`). It also splits the verdict into its two halves
(`data_permissible` / `expl_permissible`) and says which rule blocked a cell.
Those are facts; the words "Pass", "Fail" and "Blocked (LMS)" stay in the table
generator, which owns its own LaTeX vocabulary.

What the CSV may contain
    The `*_shown` values only — NaN wherever the disclosure rule forbids drawing.
    Verdicts (evaluated / permissible / selected) ARE carried, because the table
    twin prints Pass/Fail/Yes/No for every cell it evaluates whether or not the
    recall behind it is suppressed; withholding them here would hide something
    the paper already states. Recall, precision and demand are the disclosive
    quantities, and _assert_only_shown_values() re-checks every one of them
    against disclosure.results_row_hidden before the file is written.

    A vector PDF already carries the exact geometry of everything it draws, so
    this CSV is the same information in a readable form — never more.

    python figures/scripts/prepare_selection_gate.py [--test-year 2021]
                                                     [--ruleset endline]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from paths.paths import RESULTS_PATH                                   # noqa: E402
from prepare_utils import source_label, write_input                    # noqa: E402
from src import disclosure                                             # noqa: E402
from src.config import MAIN_TEST_YEAR, SELECTION_YEAR                  # noqa: E402
from src.plot_selection_gate import (                                  # noqa: E402
    _FS_LABEL, _MODEL_LABEL, _cell_demands, _cross_check,
    _expl_threshold_label, _hidden_recalls, _ylim,
)
from src.plotting import _ML_MODELS                                    # noqa: E402
from utils import canvas                                               # noqa: E402

from src.workshop_rules import (                                      # noqa: E402
    _FS_ORDER, _RULE_BASED_MODELS, NEVER, SurveyRuleSet, get_rules, participatory_config,
    select_grid_config, stored_selection,
)


def _fs_gain_label(fs: str, rules: SurveyRuleSet) -> str:
    """The recall gain this data step costs, as it reads on the x-axis line."""
    if fs == _FS_ORDER[0]:
        return ""   # the starting point: there is no step to pay for
    thr = rules.fs_gain_required.get(fs, NEVER)
    return "(never)" if thr is NEVER else f"(+{thr * 100:g} pp)"


def _xlabel(rules: SurveyRuleSet) -> dict:
    """
    The line under the panels: the ladder, and what each step of it costs. In
    pieces rather than one string, so the draw script can set the gains apart.
    """
    return {"title": "Feature set (cumulative):",
            "steps": [{"name": _FS_LABEL[fs].replace("\n", " "),
                       "gain": _fs_gain_label(fs, rules)} for fs in _FS_ORDER]}


def _assert_only_shown_values(df: pd.DataFrame) -> None:
    """
    The guard the whole layout rests on: nothing suppressed may be in the file.

    A draw script cannot leak what it cannot see, but only if `prepare` never
    writes it. Checked as a post-condition on the frame about to be written
    rather than trusted from the flags that built it, so a future column that
    happens to carry a hidden recall is caught here and not in the paper.
    """
    for value_col, flag_col in (("recall", "recall_suppressed"),
                                ("precision", "precision_suppressed"),
                                ("demand", "demand_suppressed")):
        leaked = df[df[flag_col] & df[value_col].notna()]
        if len(leaked):
            cells = ", ".join(f"{r.model}/{r.feature_set}" for r in leaked.itertuples())
            raise SystemExit(f"REFUSING to write: suppressed {value_col} present for {cells}")


def _metric(year_df: pd.DataFrame, model: str, feature_set: str,
            metric: str) -> tuple[float, bool]:
    """
    One cell's enrollment-level recall or precision, and whether it may be shown.

    The two are hidden together — an operating point pins down all four confusion
    cells — but each is asked for by name rather than inferred from the other, so
    the day operating_point_hidden stops treating them alike this still holds.
    A row that EXISTS but whose counts are missing is treated as hidden: the rule
    cannot be certified for it, and results_row_hidden refuses to guess a TP from
    a 3-decimal recall. A cell with no row at all is a different thing — the
    combination was never evaluated — and is reported as absent, not suppressed,
    the same distinction disclosure.is_small draws between NaN ("applies, nobody
    knows it") and None ("does not apply"). Marking it suppressed would promise a
    reader that one or two students are behind a cell that has none.
    """
    mask = year_df["model"] == model
    if model != "credits_policy":
        mask &= year_df["feature_set"] == feature_set
    rows = year_df[mask]
    if not len(rows):
        return float("nan"), False
    r = rows.iloc[0]
    try:
        hidden = disclosure.results_row_hidden(r)[metric]
    except (ValueError, KeyError):
        hidden = True
    return (np.nan if hidden else float(r[f"enr_{metric}"])), bool(hidden)


def prepare(results_df: pd.DataFrame, test_year: int, ruleset: str,
            evaluation_year: int | None = None) -> tuple[pd.DataFrame, dict]:
    """
    The gate on `test_year` — the cohort the rules are applied to. With an
    `evaluation_year`, one more row (role=evaluation) carries the selected
    configuration's recall and precision on that cohort, for the SI version
    that selects on one year and reports on the main one. The figure does not
    draw it; the table prints it.
    """
    rules   = get_rules(ruleset)
    year_df = results_df[results_df["test_year"] == test_year].copy()
    if year_df.empty:
        raise SystemExit(f"no rows for test_year={test_year} in the results CSV")

    best_model, best_fs, grid_df = select_grid_config(year_df, test_year, rules=rules)
    if best_model in _RULE_BASED_MODELS:
        raise SystemExit(
            f"REFUSING to write: no combination is permissible in {test_year} under the "
            f"{rules.name} rules, so the Credits policy is retained. This figure and its "
            "table circle a selected ML cell, and there is none."
        )
    hidden = _hidden_recalls(year_df)
    models = list(_ML_MODELS)
    cells, data_walks, expl_walks = _cell_demands(year_df, rules, hidden=hidden,
                                                  models=models)
    grid = grid_df.set_index(["model", "feature_set"])

    ## A mismatch means the drawn verdicts and the table's verdicts came apart,
    ## which is the one failure this figure exists to make impossible — so it
    ## stops the run rather than warn.
    problems = _cross_check(grid_df, data_walks, expl_walks, cells)
    if problems:
        raise SystemExit("REFUSING to write: the reconstructed walks disagree with "
                         "select_grid_config:\n  " + "\n  ".join(problems))

    def _half(model: str, fs: str, col: str):
        """One half of the verdict, or None for a cell the gates never ruled on."""
        key = (model, fs)
        return bool(grid.loc[key, col]) if key in grid.index else None

    rows = []
    for panel, model in enumerate(models):
        for x, fs in enumerate(_FS_ORDER):
            c = cells[(model, fs)]
            recall, recall_hidden = _metric(year_df, model, fs, "recall")
            precision, prec_hidden = _metric(year_df, model, fs, "precision")
            ## _cell_demands resolved the same question for the demand lines;
            ## disagreeing would mean the figure and the table were told
            ## different things about the same cell.
            assert recall_hidden == bool(c["recall_hidden"]), f"{model}/{fs}"
            rows.append({
                "role":              "panel",
                "panel":             panel,
                "panel_letter":      "abcde"[panel],
                "panel_title":       _MODEL_LABEL[model].replace("\n", " "),
                "model":             model,
                "x":                 x,
                "x_tick":            _FS_LABEL[fs].split("\n")[0],
                "x_label":           _FS_LABEL[fs].replace("\n", " "),
                "feature_set":       fs,
                ## Drawn values only — an empty cell is "not shown", and the
                ## flag beside it says so rather than leaving a reader to guess
                ## between suppressed and never computed.
                "recall":            recall,
                "precision":         precision,
                "demand":            c["demand_shown"],
                "recall_suppressed": recall_hidden,
                "precision_suppressed": prec_hidden,
                "demand_suppressed": bool(c["demand_hidden"]),
                "evaluated":         bool(c["reached"]),
                "data_permissible":  _half(model, fs, "data_permissible"),
                "expl_permissible":  _half(model, fs, "expl_permissible"),
                ## From the selection itself, like the two halves beside it, so
                ## the table cannot print Pass / Pass / No.
                "permissible":       bool(_half(model, fs, "permissible")),
                "fs_blocked":        fs in rules.blocked_feature_sets,
                "model_blocked":     model in rules.blocked_models,
                "selected":          model == best_model and fs == best_fs,
            })

    ## The credits-policy baseline: a horizontal line in the figure, the first
    ## row of the table. One row here rather than a scalar in the JSON, so its
    ## recall lives in exactly one place and is suppressed by the same guard.
    cp_recall, cp_recall_hidden = _metric(year_df, "credits_policy", "", "recall")
    cp_precision, cp_prec_hidden = _metric(year_df, "credits_policy", "", "precision")
    rows.append({
        "role": "baseline", "panel": None, "panel_letter": "", "panel_title": "",
        "model": "credits_policy", "x": None, "x_tick": "", "x_label": "",
        "feature_set": "rule_based",
        "recall": cp_recall, "precision": cp_precision, "demand": np.nan,
        "recall_suppressed": cp_recall_hidden, "precision_suppressed": cp_prec_hidden,
        "demand_suppressed": False, "evaluated": True,
        "data_permissible": None, "expl_permissible": None,
        "permissible": None,   # the reference, never a candidate
        "fs_blocked": False, "model_blocked": False, "selected": False,
    })

    if evaluation_year is not None and evaluation_year != test_year:
        eval_df = results_df[results_df["test_year"] == evaluation_year]
        if eval_df.empty:
            raise SystemExit(f"no rows for evaluation year {evaluation_year}")
        has_row = ((eval_df["model"] == best_model)
                   & (eval_df["feature_set"] == best_fs)).any()
        if not has_row:
            raise SystemExit(f"REFUSING to write: {best_model} + {best_fs} has no "
                             f"result for evaluation year {evaluation_year}")
        ev_recall, ev_recall_hidden = _metric(eval_df, best_model, best_fs, "recall")
        ev_prec, ev_prec_hidden = _metric(eval_df, best_model, best_fs, "precision")
        rows.append({
            "role": "evaluation", "panel": None, "panel_letter": "", "panel_title":
            _MODEL_LABEL[best_model].replace("\n", " "), "model": best_model, "x": None,
            "x_tick": "", "x_label": _FS_LABEL[best_fs].replace("\n", " "),
            "feature_set": best_fs, "recall": ev_recall, "precision": ev_prec,
            "demand": np.nan, "recall_suppressed": ev_recall_hidden,
            "precision_suppressed": ev_prec_hidden, "demand_suppressed": False,
            "evaluated": True, "data_permissible": None, "expl_permissible": None,
            "permissible": None, "fs_blocked": False, "model_blocked": False,
            "selected": False,
        })

    df = pd.DataFrame(rows)
    _assert_only_shown_values(df)

    ## Writing a pick with no panel would give the paper a selected model with no
    ## circled cell and no bold row.
    if not df.loc[df["role"] == "panel", "selected"].any():
        raise SystemExit(
            f"REFUSING to write: the selection picked {best_model} + {best_fs}, which "
            f"has no panel in the selection-gate figure (models drawn: "
            f"{', '.join(models)})."
        )

    panels_df = df[df["role"] == "panel"]
    blocked_x = [i for i, fs in enumerate(_FS_ORDER) if fs in rules.blocked_feature_sets]
    any_hidden = bool(df["recall_suppressed"].any() or df["precision_suppressed"].any()
                      or df["demand_suppressed"].any())

    meta = {
        "figsize": canvas(3.6 / 13),
        "test_year": test_year,
        "evaluation_year": evaluation_year if evaluation_year != test_year else None,
        "ruleset":   rules.name,
        "xlabel":    _xlabel(rules),
        "ylabel":    "Enrollment-level recall",
        ## Set here, from the drawable values alone, and never recomputed while
        ## drawing: one range across all four panels, and a range taken from a
        ## suppressed point would let it be read back off the padding.
        ## Unrounded, unlike the data — an axis limit is layout, and rounding it
        ## moves every mark on the panel by a fraction of a pixel.
        "ylim":      list(_ylim(cells, cp_recall)),
        "baseline":  {"label": "Credits policy (>=60 ECTS behind)",
                      "row": "the role=baseline row of the CSV"},
        "band":      {"from_x": min(blocked_x) if blocked_x else None,
                      "label":  "never permissible"},
        ## The explainability tier each model must buy its way into — rule text,
        ## so it belongs in the input, not in the draw script.
        "panels":    [{"model": m, "title": _MODEL_LABEL[m].replace("\n", " "),
                       "tier": _expl_threshold_label(m, rules)} for m in models],
        "selected":  {"model": best_model, "feature_set": best_fs},
        ## The suppression note as a finished sentence. Built from MIN_CELL_SIZE
        ## by src/disclosure.py, so the threshold itself never has to appear in
        ## figures/ at all.
        "note":      disclosure.note_text() if any_hidden else None,
    }
    assert len(panels_df) == len(models) * len(_FS_ORDER)
    return df, meta


def _assert_matches_stored_selection(meta: dict, results_dir: Path, test_year: int) -> None:
    """
    The circled cell must be the model every other figure calls Participatory.

    This figure and its table bold the grid pick recomputed here; every other
    artefact reads the pick stored by workshop_evaluation.py. They are the same
    algorithm on the same results, so a disagreement means the stored selection is
    stale (evaluation_results.csv changed after Step 4) — and the paper would
    circle one model while reporting another.
    """
    if meta["ruleset"] != "endline":
        return   # only endline selections are stored for comparison
    model, feature_set = (participatory_config(results_dir) if test_year == SELECTION_YEAR
                          else stored_selection(results_dir, "endline", test_year))
    if model is None:
        print("  ! no stored workshop selection — cannot check it against this figure")
        return
    picked = (meta["selected"]["model"], meta["selected"]["feature_set"])
    if picked != (model, feature_set):
        raise SystemExit(
            f"REFUSING to write: the selection gate picks {picked[0]} + {picked[1]}, "
            f"but the stored Participatory model is {model} + {feature_set}. "
            "Re-run src/workshop_evaluation.py (run_all.py Step 4)."
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-year", type=int, default=MAIN_TEST_YEAR)
    ap.add_argument("--ruleset", default="endline", choices=["endline", "baseline"])
    ap.add_argument("--evaluation-year", type=int, default=None, dest="evaluation_year",
                    help="also report the pick on this cohort (the SI version: select "
                         "on --test-year, evaluate on the main year)")
    ap.add_argument("--results-csv", type=Path,
                    default=RESULTS_PATH / "evaluation_results.csv")
    args = ap.parse_args()

    df, meta = prepare(pd.read_csv(args.results_csv), args.test_year, args.ruleset,
                       evaluation_year=args.evaluation_year)
    _assert_matches_stored_selection(meta, args.results_csv.parent, args.test_year)
    stem = f"selection_gate_{args.test_year}"
    if meta["ruleset"] != "endline":
        stem += f"_{meta['ruleset']}"
    write_input(stem, df, meta, source=source_label(args.results_csv, ROOT),
                repo=ROOT, script="prepare_selection_gate.py")
    print(f"Selected: {meta['selected']['model']} + {meta['selected']['feature_set']}")


if __name__ == "__main__":
    main()
