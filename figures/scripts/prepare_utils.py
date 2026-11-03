"""
Shared by the prepare_*.py scripts, and by nothing else.

A prepare script is the half of the pipeline that is allowed to reach anywhere —
results/, the analysis frame, src/ — and it is where the disclosure assertions
live. A draw script imports utils.py alone. Keeping the two halves in separate
modules is what makes that split checkable.
"""
from __future__ import annotations

import json
import subprocess
from datetime import date
from pathlib import Path

import pandas as pd

from utils import INPUT


def source_label(path: Path, repo: Path) -> str:
    """
    A short, machine-independent name for where a figure's numbers came from.

    Not `path.relative_to(repo)`: RESULTS_PATH need not be inside the checkout,
    and when it is its own tree relative_to() raises. Not the absolute path
    either — this string is written into figures/input/*.json, which travels
    with the figure, and a local directory layout does not belong in it.
    Not a ../.. chain, which would differ per machine and say nothing.

    Inside the checkout, the repo-relative path; otherwise the last two
    components, so `results/evaluation_results.csv` reads the same wherever the
    figure was built.
    """
    path = Path(path)
    try:
        return str(path.resolve().relative_to(Path(repo).resolve()))
    except ValueError:
        return str(Path(path.parent.name) / path.name)


def git_commit(repo: Path) -> str:
    """The code commit these inputs were built from, or '' outside a checkout."""
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:                                               # noqa: BLE001
        return ""


def write_input(stem: str, df: pd.DataFrame, meta: dict, *, source: str,
                repo: Path, script: str, decimals: int = 6) -> None:
    """
    Write one figure's prepared input: drawn values to .csv, scalars to .json.

    The provenance block goes in the .json rather than a side file, so a CSV and
    the run that produced it cannot be separated by a copy.

    Values are rounded on the way out, and that is a disclosure measure rather
    than tidiness: most of these numbers are ratios k/n, and a full-precision
    float64 ratio can be turned back into its k and n. Rounding removes the
    denominator while leaving the figure unchanged.

    `decimals` must stay comfortably above what the figure PRINTS, because a draw
    script formats its labels from these values: rounded to 4 decimals, a share
    just above a half-percent boundary can print one tenth lower than it should.
    Six decimals leaves three spare places under a one-decimal percentage.

    Six is also the point where the two purposes meet: it is coarse enough to
    destroy a ratio's denominator, and fine enough that the rounding is not
    visible in the drawn figure.
    """
    INPUT.mkdir(parents=True, exist_ok=True)
    df.round(decimals).to_csv(INPUT / f"{stem}.csv", index=False)
    meta = dict(meta)
    meta["provenance"] = {
        "written_by":     f"figures/scripts/{script}",
        "written_on":     date.today().isoformat(),
        "code_commit": git_commit(repo),
        "source":         source,
        "rows":           len(df),
    }
    (INPUT / f"{stem}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Saved -> {INPUT / f'{stem}.csv'}  ({len(df)} rows)")
    print(f"Saved -> {INPUT / f'{stem}.json'}")


def workshop_selection(results_path: Path) -> tuple["pd.DataFrame | None", str | None, str | None]:
    """
    The Participatory model: its per-year results, and the estimator behind it.

    A thin wrapper over src.workshop_rules.participatory_config() /
    participatory_results(), the one reader every part of the pipeline uses, so a
    figure cannot show a different Participatory model from the selection table.
    Getting it wrong is quiet: a missing results CSV does not raise, it just
    removes the Participatory bar from whatever figure was going to draw it, and
    the figure still looks complete — so a missing selection is announced here.
    """
    from src.workshop_rules import participatory_config, participatory_results

    model, feature_set = participatory_config(results_path)
    workshop_df = participatory_results(results_path)
    if model is None:
        print("  ! no workshop selection found — the Participatory model will be "
              "missing from every figure that shows it")
    return workshop_df, model, feature_set
