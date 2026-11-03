"""Generate LaTeX stability table (cross-year recall and AUC, 2019-2023).

Outputs results/descriptive/stability_table.tex — paste into SI via \input{}.

Usage:
    python src/generate_stability_table.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from paths.paths import RESULTS_PATH
from src import disclosure
from src.config import MAX_ML_MODEL, MAX_ML_FS

OUT_DIR = RESULTS_PATH / "descriptive"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _models() -> dict:
    """The three columns; the Participatory model is whatever the rules selected."""
    from src.plotting import _workshop_underlying_name
    model, fs = _workshop_underlying_name()
    if not (model and fs):
        raise SystemExit("No workshop selection found — run src/workshop_evaluation.py first.")
    return {
        "Credits policy": ("credits_policy", "ects_behind_ge_60"),
        "Participatory":  (model,            fs),
        "Max ML":         (MAX_ML_MODEL,     MAX_ML_FS),
    }


def _load(results_path: Path) -> pd.DataFrame:
    df = pd.read_csv(results_path / "evaluation_results.csv")
    return df[df["test_year"].between(2019, 2023)].copy()


def _row(df: pd.DataFrame, model: str, fs: str) -> pd.DataFrame:
    mask = (df["model"] == model) & (df["feature_set"] == fs)
    return df[mask].set_index("test_year")


def generate(results_path: Path = RESULTS_PATH) -> None:
    df = _load(results_path)
    years = sorted(df["test_year"].unique())

    MODELS = _models()
    rows = {label: _row(df, m, fs) for label, (m, fs) in MODELS.items()}

    caption = (
        r"\caption{\textbf{Cross-year stability of enrollment-level recall and ROC AUC, 2019--2023.}"
        r" Threshold $\tau$ is calibrated per year so the fraction of enrollments flagged matches"
        r" the training enrollment-level dropout rate."
        r" \textsuperscript{*}~2023 cohort is incomplete --- many enrollments unresolved;"
        r" recall is likely underestimated and results should be interpreted with caution.}"
    )

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{l" + "cc" * len(MODELS) + r"}",
        r"\toprule",
    ]

    # Header rows
    header1 = r" & " + " & ".join(
        r"\multicolumn{2}{c}{" + label + r"}" for label in MODELS
    ) + r" \\"
    header2 = r"Year & " + " & ".join(
        r"\textit{Recall} & \textit{Enr. AUC}" for _ in MODELS
    ) + r" \\"
    cmidrules = "".join(
        r"\cmidrule(lr){" + str(2 + i * 2) + r"-" + str(3 + i * 2) + r"}"
        for i in range(len(MODELS))
    )
    lines += [header1, cmidrules, header2, r"\midrule"]

    for year in years:
        suffix = r"\textsuperscript{*}" if year == 2023 else ""
        cells = [str(year) + suffix]
        for label, sub in rows.items():
            if year in sub.index:
                row = sub.loc[year]
                h = disclosure.results_row_hidden(row)
                cells.append(disclosure.SUPPRESSED_TEX if h["recall"] else f"{row['enr_recall']:.3f}")
                cells.append(disclosure.SUPPRESSED_TEX if h["auc"] else f"{row['roc_auc']:.3f}")
            else:
                cells += ["--", "--"]
        lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        caption,
        r"\label{tab:stability}",
        r"\end{table}",
    ]

    content = "\n".join(lines) + "\n"

    out = OUT_DIR / "stability_table.tex"
    out.write_text(content)
    print(f"Saved: {out}")



if __name__ == "__main__":
    generate()
