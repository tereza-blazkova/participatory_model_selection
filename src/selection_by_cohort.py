"""
Would the workshop rules pick the same Participatory model in every test cohort?

The selection is run on one year — SELECTION_YEAR — and the model it picks is
then reported for all of 2019-2023. That is only defensible if the pick does not
hinge on which cohort the rules happened to be applied to. This runs the grid
selection (workshop_rules.select_grid_config, the one selection — nothing is
re-implemented here) once per test cohort and per survey rule set.

Writes
  results/descriptive/selection_by_cohort.csv   every rule set x cohort, for the
                                                diagnostic report — stays on the
                                                machine (never exported)
  results/descriptive/selection_by_cohort.tex   SI table: the endline pick in each
                                                settled cohort and its margin over
                                                the runner-up

A pick that differs between cohorts is reported, not corrected.

The margin is the difference of two recalls, so it gives either one away when
the other is printed: it is suppressed whenever either operating point is
(disclosure.results_row_hidden).

    python src/selection_by_cohort.py
"""
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from paths.paths import RESULTS_PATH
from src import disclosure
from src.config import SELECTION_BY_COHORT_YEARS, SELECTION_YEAR, TEST_YEARS
from src.display import feature_set_label, model_label
from src.workshop_rules import (
    _SENSITIVITY_DATA_REFERENCE,
    _RULE_BASED_MODELS,
    DEFAULT_RULES,
    _precision,
    _recall,
    get_rules,
    select_grid_config,
)

OUT_CSV = "selection_by_cohort.csv"
OUT_TEX = "selection_by_cohort.tex"


def _select_quietly(results_df, year, rules, **kw):
    """select_grid_config prints its fallback warning; the table records it instead."""
    with contextlib.redirect_stdout(io.StringIO()):
        return select_grid_config(results_df, year, rules=rules, **kw)


def _row_hidden(results_df: pd.DataFrame, model, fs, year) -> bool:
    """Is this cell's recall suppressed? A missing or count-less row counts as hidden."""
    if model is None:
        return False
    rows = results_df[(results_df["model"] == model) & (results_df["test_year"] == year)
                      & (results_df["feature_set"] == fs)]
    if not len(rows):
        return True
    try:
        return bool(disclosure.results_row_hidden(rows.iloc[0])["recall"])
    except (ValueError, KeyError):
        return True


def selection_by_cohort(
    results_df: pd.DataFrame,
    years: list[int],
    rulesets: list[str],
) -> pd.DataFrame:
    """One row per (rule set, test year): the pick, its margin, and the sensitivity pick."""
    rows = []
    for name in rulesets:
        rules = get_rules(name)
        for year in years:
            if not (results_df["test_year"] == year).any():
                continue
            model, fs, grid = _select_quietly(results_df, year, rules)
            sens_model, sens_fs, _ = _select_quietly(
                results_df, year, rules, data_reference_model=_SENSITIVITY_DATA_REFERENCE)

            ## Read from the results, not the grid: a retained credits policy is
            ## selected without being one of the grid's candidates.
            recall = _recall(results_df, model, fs, year)
            others = (grid[grid["permissible"] & ~grid["selected"]]
                      .dropna(subset=["recall"])
                      .sort_values("recall", ascending=False))
            runner_up = others.iloc[0] if len(others) else None
            ru_model = runner_up["model"] if runner_up is not None else None
            ru_fs    = runner_up["feature_set"] if runner_up is not None else None

            rows.append({
                "ruleset":         rules.name,
                "test_year":       year,
                "is_2023":         year == 2023,
                "model":           model,
                "feature_set":     fs,
                "recall":          recall,
                "precision":       _precision(results_df, model, fs, year),
                "fallback":        not grid["permissible"].any(),
                "n_permissible":   int(grid["permissible"].sum()),
                "runner_up_model": ru_model,
                "runner_up_fs":    ru_fs,
                "margin_pp":       (100 * (recall - runner_up["recall"])
                                    if runner_up is not None else np.nan),
                "margin_hidden":   (_row_hidden(results_df, model, fs, year)
                                    or _row_hidden(results_df, ru_model, ru_fs, year)),
                "sensitivity_model":       sens_model,
                "sensitivity_feature_set": sens_fs,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    main = (out[out["test_year"] == SELECTION_YEAR]
            .set_index("ruleset")[["model", "feature_set"]])
    out["same_as_main_year"] = [
        r.ruleset in main.index
        and (r.model, r.feature_set) == tuple(main.loc[r.ruleset])
        for r in out.itertuples()
    ]
    return out


def build_table(table: pd.DataFrame, years: list[int] = SELECTION_BY_COHORT_YEARS) -> str:
    """The endline pick in each cohort, its runner-up and the margin, as LaTeX."""
    end = table[(table["ruleset"] == DEFAULT_RULES.name) & table["test_year"].isin(years)]
    missing = sorted(set(years) - set(end["test_year"]))
    if missing:
        raise ValueError(f"no endline selection for {missing}")

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{lllllr}",
        r"\toprule",
        r"Cohort & Selected model & Feature set & Runner-up model & Feature set & "
        r"Margin (pp) \\",
        r"\midrule",
    ]
    for r in end.sort_values("test_year").itertuples():
        cohort = f"{r.test_year}" + (" (main text)" if r.test_year == SELECTION_YEAR else "")
        if r.runner_up_model is None or pd.isna(r.runner_up_model):
            ru, ru_fs, margin = r"\multicolumn{2}{l}{none permissible}", None, "--"
        else:
            ru, ru_fs = model_label(r.runner_up_model), feature_set_label(r.runner_up_fs)
            margin = (disclosure.SUPPRESSED_TEX if r.margin_hidden
                      else f"{r.margin_pp:+.1f}")
        pick = model_label(r.model) + (r" (fallback)" if r.fallback else "")
        pick_fs = "--" if r.model in _RULE_BASED_MODELS else feature_set_label(r.feature_set)
        cells = [cohort, pick, pick_fs, ru]
        if ru_fs is not None:
            cells.append(ru_fs)
        cells.append(margin)
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{\textbf{Workshop-grounded selection by test cohort.} The endline "
        r"selection rules applied to each settled test cohort separately: the "
        r"configuration they select, the permissible configuration with the next "
        r"highest enrollment-level recall, and the recall margin between the two in "
        r"percentage points. The main text applies the rules to the "
        + str(SELECTION_YEAR) + r" cohort. 2023 is left out as an incomplete cohort.}",
        r"\label{tab:selection_by_cohort}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main() -> pd.DataFrame:
    results_df = pd.read_csv(RESULTS_PATH / "evaluation_results.csv")
    table = selection_by_cohort(results_df, TEST_YEARS,
                                ["baseline", DEFAULT_RULES.name])
    out_dir = RESULTS_PATH / "descriptive"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / OUT_CSV, index=False)
    (out_dir / OUT_TEX).write_text(build_table(table))
    print(f"Saved → {out_dir / OUT_CSV}")
    print(f"Saved → {out_dir / OUT_TEX}")
    return table


if __name__ == "__main__":
    main()
