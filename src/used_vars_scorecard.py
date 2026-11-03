"""
Check the FasterRisk model Step 1 saved for the used_vars feature set — the
variables that mirror the FAC_A/FAC_B counselor model. The scorecard figure and the
pilot figures read that model (figures/scripts/prepare_pilot.py).

The model is never refitted here: Step 1 (src/evaluation.py) is its only writer,
so the results table, its bootstrap CIs, the scorecard and the pilot figures all
describe the same model. This step loads it through load_model_checked(), which
refuses a file that no longer matches the code, and stops the run if it is
missing.

Run:
    python src/used_vars_scorecard.py
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from paths.paths import RESULTS_PATH
from src.config import USED_VARS, TARGET
from src.features import load_model_checked

_UV_COLOR   = "#56B4E9"  # sky blue (Wong) — reserved for FasterRisk+used_vars
_MODEL_PATH = RESULTS_PATH / "models" / "fasterrisk__used_vars.joblib"


# ── The model ─────────────────────────────────────────────────────────────────

def load_used_vars(feature_list: list):
    """The FasterRisk used_vars model saved by Step 1, staleness-checked."""
    if not _MODEL_PATH.exists():
        raise FileNotFoundError(f"{_MODEL_PATH} not found — run src/evaluation.py first.")
    return load_model_checked(_MODEL_PATH, feature_list)


# ── Per-faculty ROC AUC ───────────────────────────────────────────────────────

def _used_vars_roc_auc(pipeline, subset: pd.DataFrame) -> float:
    y    = subset[TARGET].values
    prob = pipeline.predict_proba(subset[USED_VARS])[:, 1]
    return float(roc_auc_score(y, prob)) if len(np.unique(y)) >= 2 else float("nan")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    meta = json.loads((RESULTS_PATH / "models" / "train_metadata.json").read_text())
    pipeline = load_used_vars(meta["available_features"]["used_vars"])
    n_terms = int(np.count_nonzero(pipeline["clf"].coef_))
    print(f"FasterRisk used_vars model OK ({n_terms} non-zero terms) — {_MODEL_PATH}")
