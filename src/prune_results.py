"""
Keep results/ to the models this pipeline runs.

A run that skips training reads result files written by earlier runs, and those
can hold rows and model files of a model the code no longer has. Most readers go
straight to these CSVs, and some take whatever models they find, so the rows are
removed from the files themselves rather than filtered by each reader. The test
is an allowed list (config.RESULT_MODELS), so no retired model is named here.

Kept rows stay byte for byte as they were: the columns are read and written as
text, because a float parsed and printed again need not come back the same.

Runs at the start of run_all.py and of evaluation.py --merge. Nothing is
retrained and no other model's numbers change.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from paths.paths import RESULTS_PATH
from src.config import RESULT_MODELS

## Files with one row per model × something, in results/ and results/partial/*/.
_ROW_FILES = (
    "evaluation_results.csv",
    "enrollment_predictions.csv",
    "row_predictions.csv",
    "training_times.csv",
    "evaluation_results_seeds.csv",
    "training_curves.csv",
    "bootstrap_cis*.csv",
    "workshop_candidates*.csv",
)


def _prune_rows(path: Path) -> dict[str, int]:
    """Drop rows of models not on the list; returns {model: rows dropped}."""
    if "model" not in pd.read_csv(path, nrows=0).columns:
        return {}
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    keep = df["model"].isin(RESULT_MODELS)
    if keep.all():
        return {}
    dropped = df.loc[~keep, "model"].value_counts().to_dict()
    df[keep].to_csv(path, index=False)
    return dropped


def _prune_model_files(models_dir: Path) -> list[Path]:
    """Files named {model}__{feature_set}, and any stored selection, of a model not on the list."""
    gone = []
    for path in sorted(models_dir.glob("*__*")):
        if (path.name.endswith((".joblib", "_best_params.json"))
                and path.name.split("__")[0] not in RESULT_MODELS):
            path.unlink()
            gone.append(path)
    ## The readers refuse a missing selection; one naming a model the code no
    ## longer has would pass until something tried to score it.
    for path in sorted(models_dir.glob("workshop_config*.json")):
        if json.loads(path.read_text()).get("model") not in RESULT_MODELS:
            path.unlink()
            gone.append(path)
    return gone


def prune_results(results_path: Path = RESULTS_PATH) -> None:
    """Remove everything of a model not on the list, and say what went."""
    folders = [results_path, *sorted(p for p in (results_path / "partial").glob("*") if p.is_dir())]
    for folder in folders:
        for pattern in _ROW_FILES:
            for path in sorted(folder.glob(pattern)):
                dropped = _prune_rows(path)
                if dropped:
                    print(f"  pruned {path.relative_to(results_path)}: {dropped}")
    for path in _prune_model_files(results_path / "models"):
        print(f"  pruned {path.relative_to(results_path)}")


if __name__ == "__main__":
    prune_results()
