"""
Group-specific decision thresholds, used by the equalized-recall audit.

equal_opportunity_thresholds() sets a per-group τ_g so that recall is equal
across groups; budgeted_equal_opportunity_thresholds() does
the same under the capacity budget of the global threshold, and
apply_group_thresholds() flags each enrollment at its group's τ_g. All work on
the already-trained max_prob scores — no retraining.

Applying group-specific thresholds in practice means using a protected
attribute as a decision input, which may conflict with anti-discrimination law.
They are used here for evaluation only.
"""

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import paper_style

## The paper style, shared with every figure (src/paper_style.py).
paper_style.apply_style()


_STRATEGIES = ["equal_opportunity", "demographic_parity"]


# ── threshold calibration ──────────────────────────────────────────────────────

def equal_opportunity_thresholds(
    enr: pd.DataFrame,
    subgroup_col: str,
    reference_val,
) -> dict:
    """
    Find per-group τ_g so that recall_g equals the recall of the reference group
    at its globally calibrated threshold. Within each group, τ_g is set to the
    (1 − target_recall)-th quantile of max_prob among dropout enrollments.

    Returns {group_val: threshold}.
    """
    # Reference recall: recall of the reference group under current ever_flagged
    ref = enr[enr[subgroup_col] == reference_val]
    ref_dropouts = ref[ref["dropped_out"] == 1]
    if len(ref_dropouts) == 0:
        return {}
    target_recall = float((ref_dropouts["ever_flagged"] == 1).mean())

    thresholds = {}
    for g in enr[subgroup_col].dropna().unique():
        g_data   = enr[enr[subgroup_col] == g]
        dropouts = g_data[g_data["dropped_out"] == 1]["max_prob"].values
        if len(dropouts) == 0:
            thresholds[g] = enr["max_prob"].max()
            continue
        # τ_g = quantile of dropout max_probs such that fraction ≥ τ_g equals target_recall
        tau = float(np.quantile(dropouts, max(0.0, 1.0 - target_recall)))
        thresholds[g] = tau

    return thresholds


def budgeted_equal_opportunity_thresholds(
    enr: pd.DataFrame,
    subgroup_col: str,
    budget_rate: float,
) -> tuple[dict, float]:
    """
    Equal-opportunity thresholds subject to a capacity budget: per-group τ_g
    such that every group reaches the *same* recall, chosen as the highest
    common recall whose pooled flag rate does not exceed `budget_rate`.

    Rationale. `equal_opportunity_thresholds` pins the common recall to the
    reference group's recall at the global τ. When a comparison group is
    under-identified, matching it means lowering that group's threshold, which
    flags more enrollments than the global rule did — pooled recall rises, but
    the flagged share exceeds the intervention budget the global threshold was
    calibrated to (`conservative_threshold`, src/metrics.py). The comparison
    against the deliberative workshop's stated tolerance (accepting a small
    recall loss in exchange for equal recall across groups) is only meaningful
    at a fixed budget: unconstrained, "equalization" buys recall with extra
    counsellor contacts that were never part of the tradeoff put to
    participants.

    Here the common recall target is free and the budget binds instead. For a
    target recall r, group g's threshold is its k-th largest dropout score with
    k = floor(r * n_dropouts_g), so the realised group recall is r rounded
    *down* to an achievable value — the same round-down convention
    `conservative_threshold` uses, applied to recall rather than flag rate.
    Pooled flags are non-decreasing in r, so the largest feasible r is found by
    bisection over the achievable recall levels.

    Returns ({group_val: threshold}, realised_target_recall).
    """
    groups = sorted(enr[subgroup_col].dropna().unique())
    n_total = len(enr)
    if n_total == 0 or not groups:
        return {}, float("nan")

    # per group: dropout scores sorted descending (k-th largest = recall k/n_g),
    # and all scores sorted ascending (for counting flags at a threshold)
    drop_desc, all_asc = {}, {}
    for g in groups:
        g_data = enr[enr[subgroup_col] == g]
        drop_desc[g] = np.sort(
            g_data.loc[g_data["dropped_out"] == 1, "max_prob"].values
        )[::-1]
        all_asc[g] = np.sort(g_data["max_prob"].values)

    def _thresholds_at(r: float) -> dict:
        out = {}
        for g in groups:
            n_g = len(drop_desc[g])
            # epsilon: r is itself k/n for some group, and (k/n)*n can land just
            # below k in floating point (15/22*22 = 14.999…), dropping a student
            k = int(np.floor(r * n_g + 1e-9)) if n_g else 0
            # k == 0 → flag nobody in this group (recall 0 is the only
            # achievable level at or below r)
            out[g] = float(drop_desc[g][k - 1]) if k > 0 else np.inf
        return out

    def _flag_rate(thresholds: dict) -> float:
        flagged = 0
        for g in groups:
            tau = thresholds[g]
            if np.isinf(tau):
                continue
            s = all_asc[g]
            flagged += len(s) - int(np.searchsorted(s, tau, side="left"))
        return flagged / n_total

    # achievable common-recall levels, ascending; bisect for the largest
    # feasible one (flag rate is non-decreasing in r)
    levels = np.unique(
        np.concatenate([
            np.arange(len(drop_desc[g]) + 1) / max(len(drop_desc[g]), 1)
            for g in groups
        ] + [np.array([0.0])])
    )
    lo, hi = 0, len(levels) - 1
    if _flag_rate(_thresholds_at(levels[lo])) > budget_rate:
        # even recall 0 overshoots (only possible if budget < 0) — degenerate
        return _thresholds_at(levels[lo]), float(levels[lo])
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _flag_rate(_thresholds_at(levels[mid])) <= budget_rate:
            lo = mid
        else:
            hi = mid - 1

    return _thresholds_at(levels[lo]), float(levels[lo])


def apply_group_thresholds(
    enr: pd.DataFrame,
    thresholds: dict,
    subgroup_col: str,
) -> pd.DataFrame:
    """Apply per-group thresholds, returning a copy with updated ever_flagged."""
    out = enr.copy()
    out["ever_flagged"] = 0
    for g, tau in thresholds.items():
        mask = out[subgroup_col] == g
        out.loc[mask, "ever_flagged"] = (out.loc[mask, "max_prob"] >= tau).astype(int)
    return out
