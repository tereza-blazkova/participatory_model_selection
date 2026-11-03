"""
Fairness-accuracy frontier, decomposed two ways: by data availability and by
model complexity. Companion to src/fa_frontier.py, same framework, different
question.

    2 × 3 grid
      row 1   — DATA ladder:  logistic regression, feature sets a → e
      row 2   — MODEL ladder: fixed feature set, FasterRisk → LR → XGBoost → MLP
      columns — fairness axis: sex, nationality, entry age (as in the FA frontier)

The two rows are not the same kind of exercise
-------------------------------------------------
**Data (row 1)** changes the covariate vector X. Because our ladder is nested
and *we* choose the algorithm, the feasible set can only grow as data is added,
so each rung's frontier can only weakly improve — no proposition is needed for
that. A crossing is therefore estimation noise or overfitting, not a finding,
which makes this row a validity check as much as a measurement.

The framework's comparative statics on adding covariates or group identity (e.g. "more data
can hurt a fairness-minded designer", or when adding group identity helps) are
results for the *input design* setting, where an accuracy-maximising agent picks
the algorithm and the designer only chooses its inputs. That wedge does not
exist here, so they are not predictions for this ladder. The d → e step also
adds three attributes at once (sex, entry age, nationality), not the panel's
group identity alone.

**Model (row 2)** is not an information-structure change at all: X is held
fixed and the permissible algorithm set shrinks. Formally that is the same move
as the workshop-permissible band in src/fa_frontier.py, decomposed by model, and
it expresses the cost of interpretability in group error rates. Model classes
are not nested, so nothing forces these frontiers to be ordered — a crossing
here is not automatically a bug. An integer score admits few cutoffs, so
FasterRisk's frontier is a polygon rather than a smooth curve; that is a
property of the score, not a plotting artifact.

Choices baked in
----------------
  - The data ladder is run on **logistic regression**, so each rung shows what
    LR can extract from that X. It is therefore a *lower bound* on F_X, which in
    the theory ranges over all algorithms: a flat step means "no additional
    value to LR", not "no value in principle".
  - The model ladder is held at **c_plus_enrolment**, a rung without the
    sensitive attribute — so all four models are compared group-blind, with
    no group-specific-threshold effect mixed into the model comparison.
  - `e_plus_sensitive` is drawn **twice** in the data row: solid without
    group-aware rules, dashed with them. That separates the effect of more
    covariates from the effect of group-specific thresholds becoming admissible.
  - The credits policy is not shown; these panels compare data and models, and
    the policy baseline lives on the main figure.

The figure is prepared by figures/scripts/prepare_fa_frontier_ladders.py and
drawn by figures/scripts/draw_fa_frontier_ladders.py.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fa_frontier import (
    _FS_LABELS,
    _MODEL_LABELS,
    _SUBGROUPS,
    _sweep,
    compute_frontier,
)

# Ordered data ladder — `used_vars` is deliberately excluded, it is not part of it.
_FS_LADDER = ["a_credits", "b_credits_performance", "c_plus_enrolment",
              "d_plus_lms", "e_plus_sensitive"]
_DATA_LADDER_MODEL = "logistic_regression"
_DEFAULT_MODEL_LADDER_FS = "c_plus_enrolment"

# An ordered ladder gets a sequential ramp; the model ladder
# reuses the repo-wide model colours so it matches every other model comparison.
_LADDER_CMAP = "viridis"


def _model_colors() -> tuple[list, list, list]:
    from src.plotting import _COLORS, _MARKERS, _ML_MODELS
    return list(_ML_MODELS), list(_COLORS), list(_MARKERS)


# ── one frontier per series ──────────────────────────────────────────────────

def _frontier_for(ep_year: pd.DataFrame, model: str, feature_set: str,
                  spec: dict, group_aware: bool) -> dict | None:
    """Threshold-sweep frontier for a single (model, feature set)."""
    col, g1, g0 = spec["col"], spec["g1"], spec["g0"]
    sub = ep_year[(ep_year["model"] == model) &
                  (ep_year["feature_set"] == feature_set)].dropna(
                      subset=["max_prob", col])
    if sub.empty:
        return None
    common, aware = _sweep(sub["max_prob"].to_numpy(float),
                           sub["dropped_out"].to_numpy(int),
                           sub[col].to_numpy(), g1, g0, group_aware)
    return compute_frontier(np.vstack([common, aware]))


def build_ladders(ep: pd.DataFrame, year: int, model_ladder_fs: str) -> dict:
    """Both ladders, for every fairness axis in _SUBGROUPS."""
    ep_year = ep[ep["test_year"] == year]
    models, _, _ = _model_colors()
    out = {}

    for name, spec in _SUBGROUPS.items():
        data_series, model_series = [], []

        for fs in _FS_LADDER:
            fr = _frontier_for(ep_year, _DATA_LADDER_MODEL, fs, spec,
                               group_aware=False)
            if fr is not None:
                data_series.append(dict(key=fs, label=_FS_LABELS.get(fs, fs),
                                        frontier=fr, group_aware=False))
        # the same X, now with group-specific thresholds admissible
        if "e_plus_sensitive" in _FS_LADDER:
            fr = _frontier_for(ep_year, _DATA_LADDER_MODEL, "e_plus_sensitive",
                               spec, group_aware=True)
            if fr is not None:
                data_series.append(dict(key="e_plus_sensitive__group_aware",
                                        label="+ Sensitive, group-aware rules",
                                        frontier=fr, group_aware=True))

        for model in models:
            fr = _frontier_for(ep_year, model, model_ladder_fs, spec,
                               group_aware=model_ladder_fs in spec["sensitive_in"])
            if fr is not None:
                model_series.append(dict(key=model,
                                         label=_MODEL_LABELS.get(model, model),
                                         frontier=fr, group_aware=False))

        out[name] = dict(spec=spec, data=data_series, model=model_series)
    return out


# ── plot ─────────────────────────────────────────────────────────────────────

def _axis_limits(ladders: dict, pad: float = 0.12) -> tuple[float, float]:
    """
    One range for all panels and both axes.

    Both ladders and all fairness axes are drawn on the same scale, so
    the rows can be read against each other — the whole point of the figure
    is how far data moves the frontier *relative to* how far model choice does,
    and per-panel autoscaling would destroy that comparison. Sharing x with y
    additionally keeps the 45° line a true diagonal.
    """
    P = np.vstack([s["frontier"]["path"]
                   for d in ladders.values()
                   for series in (d["data"], d["model"])
                   for s in series])
    lo, hi = float(np.nanmin(P)), float(np.nanmax(P))
    span = max(hi - lo, 1e-3)
    return max(0.0, lo - pad * span), hi + pad * span
