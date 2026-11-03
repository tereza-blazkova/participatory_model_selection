"""Generate the LaTeX selection-gate table (recall, precision and workshop rule
verdicts for every model x feature-set combination, one test year).

This table and the main-text figure (fig:selection_gate) state the same verdicts
about the same cells, and both are exported, so they must suppress the same
cells. They do so by construction: both read
figures/input/selection_gate_{year}.csv, where
figures/scripts/prepare_selection_gate.py ran the selection logic itself
(src/workshop_rules.py via src/plot_selection_gate.py) and applied statistical
disclosure control once.

Both artefacts read the same prepared input, so they withhold the same cells:
if each decided for itself, a value one suppressed could be drawn exactly by
the other.

This module therefore does no analysis. It formats: row order, LaTeX, the words
"Pass" / "Fail" / "Blocked". A suppressed value arrives as an empty cell with a
flag beside it, and is printed as disclosure.SUPPRESSED_TEX.

Outputs results/descriptive/selection_table.tex for the main selection year
(SELECTION_YEAR), and selection_table_{year}.tex for any other — the SI version,
which applies the rules to another cohort and prints the pick's recall and
precision on the evaluation year as a last row.

Usage:
    python src/generate_selection_table.py [--test-year 2021]
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from paths.paths import PROJECT_ROOT, RESULTS_PATH
from src.config import MAIN_TEST_YEAR, SELECTION_YEAR
from src import disclosure

OUT_DIR = RESULTS_PATH / "descriptive"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIGURE_INPUT = PROJECT_ROOT / "figures" / "input"

## The row label for the baseline. Every other label — model names, feature-set
## names — is read from the prepared input, so the table and the figure cannot
## end up calling the same thing by two names.
_BASELINE_LABEL = ("Credits policy", r"Rule: $\geq$60 ECTS behind")


def _load(test_year: int) -> tuple[pd.DataFrame, dict]:
    """The prepared input both artefacts read, or an error naming how to build it."""
    stem = f"selection_gate_{test_year}"
    csv, meta = FIGURE_INPUT / f"{stem}.csv", FIGURE_INPUT / f"{stem}.json"
    missing = [p.name for p in (csv, meta) if not p.exists()]
    if missing:
        raise SystemExit(
            f"missing figures/input/{', '.join(missing)} — this table is built from the\n"
            f"same prepared input as its figure, so that its suppressed cells and the\n"
            f"figure's are the same cells. Build it first:\n"
            f"  python figures/scripts/prepare_selection_gate.py --test-year {test_year}"
        )
    return pd.read_csv(csv), json.loads(meta.read_text())


def _cell(row, metric: str) -> str:
    """One metric to 3 dp, the suppression marker, or nan for a cell with no value."""
    if bool(row[f"{metric}_suppressed"]):
        return disclosure.SUPPRESSED_TEX
    value = row[metric]
    return "nan" if pd.isna(value) else f"{float(value):.3f}"


def _verdict(row) -> tuple[str, str, str]:
    """The three verdict columns, in the table's own vocabulary.

    A cell the gates never ruled on prints "---" rather than a verdict: the
    hard-blocked ones are shown for reference, and marking them Fail would read
    as a rule they came close to passing.
    """
    if bool(row["fs_blocked"]) or bool(row["model_blocked"]):
        return ("Blocked (LMS)" if row["fs_blocked"] else "---",
                "Blocked" if row["model_blocked"] else "---",
                "---")
    if pd.isna(row["data_permissible"]) or pd.isna(row["expl_permissible"]):
        return "---", "---", "---"
    return ("Pass" if row["data_permissible"] else "Fail",
            "Pass" if row["expl_permissible"] else "Fail",
            "Yes" if row["permissible"] else "No")


def generate(test_year: int = MAIN_TEST_YEAR) -> None:
    df, meta = _load(test_year)
    if meta["test_year"] != test_year:
        raise SystemExit(f"prepared input is for {meta['test_year']}, not {test_year}")
    panels   = df[df["role"] == "panel"]
    baseline = df[df["role"] == "baseline"].iloc[0]
    evaluation = df[df["role"] == "evaluation"]
    main = test_year == SELECTION_YEAR
    label = "selection_gate" if main else f"selection_gate_{test_year}"

    if main:
        caption = (
            r"\caption{\textbf{Application of the workshop-grounded selection rules, "
            + str(test_year) +
            r" test cohort.} Recall and precision are enrollment-level."
            r" Data-input and explainability verdicts follow the selection algorithm"
            r" described in Section~\ref{sec:model_selection_algorithm}, which"
            r" evaluates every (model, feature set) combination independently and"
            r" selects the permissible combination with the highest recall."
            r" Figure~\ref{fig:selection_gate} in the main text shows the same"
            r" verdicts as recall comparisons against what each rule demanded."
            r" The neural network and feature sets (d) + LMS and (e) + Sensitive"
            r" are hard-blocked by workshop rule and are not evaluated by the algorithm"
            r" (marked ---); their recall/precision are shown for reference only."
            r" Bold row is the selected Participatory model.}"
        )
    else:
        eval_year = meta.get("evaluation_year")
        caption = (
            r"\caption{\textbf{Application of the workshop-grounded selection rules to "
            r"the " + str(test_year) + r" test cohort.} The endline rules of the main "
            r"text applied to a different cohort, as a robustness check of the "
            r"selection: recall and precision are enrollment-level on the "
            + str(test_year) + r" cohort, and the verdicts follow the same algorithm "
            r"(Section~\ref{sec:model_selection_algorithm})."
            r" Figure~\ref{fig:" + label + r"} shows the same verdicts as recall "
            r"comparisons against what each rule demanded. The neural network and feature sets "
            r"(d) + LMS and (e) + Sensitive are hard-blocked by workshop rule (marked ---)."
            + (r" The last row is the selected configuration evaluated on the "
               + str(eval_year) + r" cohort, the main text's test year."
               if eval_year else "")
            + r" Bold row is the selected configuration.}"
        )

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\footnotesize",
        # 7 columns don't fit \linewidth at \footnotesize alone; resizebox
        # here applies a modest ~0.83x shrink (verified), not an extreme one.
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{llccccc}",
        r"\hline",
        r"\textbf{Model} & \textbf{Feature set} & \textbf{Recall} & \textbf{Precision} & "
        r"\textbf{Data-input} & \textbf{Explainability} & \textbf{Permissible} \\",
        r"\hline",
    ]

    ## credits_policy row — the rule the whole exercise is measured against, so
    ## it sits above the grid rather than inside it.
    name, rule = _BASELINE_LABEL
    lines.append(
        f"{name} & {rule} & {_cell(baseline, 'recall')} & "
        f"{_cell(baseline, 'precision')} & Baseline & Baseline & Baseline \\\\"
    )
    lines.append(r"\hline")

    ## One block per feature set, in ladder order (the `x` the figure plots
    ## against), each block listing the models in panel order.
    for x, block in panels.groupby("x", sort=True):
        first = True
        for row in block.sort_values("panel").itertuples(index=False):
            row = pd.Series(row._asdict())
            fs_label = row["x_label"] if first else ""
            first = False
            data_v, expl_v, permissible_v = _verdict(row)
            cells = (
                f"{row['panel_title']} & {fs_label} & {_cell(row, 'recall')} & "
                f"{_cell(row, 'precision')} & "
                f"{data_v} & {expl_v} & {permissible_v}"
            )
            if bool(row["selected"]):
                cells = r"\textbf{" + cells.replace(" & ", "} & \\textbf{") + "}"
            lines.append(cells + r" \\")
        lines.append(r"\hline")

    lines += [
    ]
    if len(evaluation):
        ev = evaluation.iloc[0]
        lines += [
            f"\\multicolumn{{2}}{{l}}{{Selected, evaluated on {meta['evaluation_year']}: "
            f"{ev['panel_title']}, {ev['x_label']}}} & {_cell(ev, 'recall')} & "
            f"{_cell(ev, 'precision')} & & & \\\\",
            r"\hline",
        ]
    lines += [
        r"\end{tabular}%",
        r"}",
        caption,
        f"\\label{{tab:{label}}}",
        r"\end{table}",
    ]

    content = "\n".join(lines) + "\n"

    out = OUT_DIR / ("selection_table.tex" if main else f"selection_table_{test_year}.tex")
    out.write_text(content)
    print(f"Saved: {out}")
    print(f"Selected: {meta['selected']['model']} + {meta['selected']['feature_set']}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-year", type=int, default=MAIN_TEST_YEAR)
    args = parser.parse_args()
    generate(test_year=args.test_year)
