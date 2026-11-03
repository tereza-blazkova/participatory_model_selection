"""The logic behind the selection-gate figure: the workshop-grounded selection
rules as two gates, one test year. figures/scripts/prepare_selection_gate.py
serialises it; the figure and tab:selection_gate are both built from that.

The visual counterpart of tab:selection_gate (src/generate_selection_table.py).
The table states a verdict per cell; this figure shows *why* each verdict falls
where it does, by drawing the rule as a line.

Both workshop gates are sequential recall comparisons, so both can be drawn the
same way. At every step the rules demand a recall gain over the currently held
configuration:

    demanded(step) = recall(current) + threshold(step)

Plotting `demanded` against the recall actually `achieved` turns each verdict
into a geometric fact: the step is accepted where the solid line ends above the
dashed one, rejected where it ends below. Both lines leave the same anchor
point, so the vertical gap at the step is the margin the rule was cleared (or
missed) by.

  Panel a — data-input gate, over the cumulative feature-set ladder.
            One walk per model (the gate is evaluated per model, see
            _data_permissible_feature_sets).
  Panel b — explainability gate, over models ordered most → least explainable.
            One walk per feature set; the selected set is drawn in full, the
            other reachable sets faintly for context.

Cells the rule set hard-blocks (LMS / sensitive feature sets, MLP) are drawn
faded inside a shaded band: they are shown for reference, never evaluated.

Verdicts are taken from select_grid_config() itself rather than recomputed, and
the walk reconstructed here for the demand lines is asserted against them — the
figure cannot disagree with the table.

"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import disclosure
from src.display import MODEL_LABELS
from src.plotting import _ML_MODELS
from src.workshop_rules import (
    _FS_ORDER,
    EXPLAINABILITY_LEVEL,
    SurveyRuleSet,
)

# ── Labels ────────────────────────────────────────────────────────────────────

_FS_LABEL = {
    "a_credits":             "(a)\nCredits",
    "b_credits_performance": "(b)\n+ Performance",
    "c_plus_enrolment":      "(c)\n+ Enrollment",
    "d_plus_lms":            "(d)\n+ LMS",
    "e_plus_sensitive":      "(e)\n+ Sensitive",
}

_MODEL_LABEL = {
    "credits_policy":      "Credits\npolicy",
    "fasterrisk":          "FasterRisk",
    "logistic_regression": "Logistic\nregression",
    "xgboost":             "XGBoost",
    "mlp":                 MODEL_LABELS["mlp"],   # one name, in src/display.py
}

# Explainability ladder, most → least explainable. credits_policy is the entry
# baseline the walk starts from, so it belongs on the axis even though it is
# never a workshop output.
_EXPL_AXIS = ["credits_policy", "fasterrisk", "logistic_regression", "xgboost", "mlp"]

_BAND_COLOR   = "#EFEFEF"
_FADE         = 0.30    # alpha for hard-blocked, reference-only cells
_CONTEXT_GREY = "#9A9A9A"

# ── Data access ───────────────────────────────────────────────────────────────

def _recall(results_df: pd.DataFrame, model: str, feature_set: str) -> float:
    """Enrollment-level recall for one cell; NaN when the cell was not trained."""
    mask = results_df["model"] == model
    if model != "credits_policy":
        mask &= results_df["feature_set"] == feature_set
    row = results_df[mask]
    return float(row["enr_recall"].iloc[0]) if len(row) else float("nan")


def _hidden_key(model: str, feature_set: str) -> tuple[str, str]:
    """The lookup key _recall uses — credits_policy ignores the feature set."""
    return (model, "") if model == "credits_policy" else (model, feature_set)


def _hidden_recalls(results_df: pd.DataFrame) -> set[tuple[str, str]]:
    """
    Cells whose enrollment-level recall must not be DRAWN.

    The table twin (generate_selection_table.py) already prints the marker for these,
    and this figure shows the same quantity from the same CSV: drawing the number
    would hand back every value the table suppressed. Recall and precision are
    hidden together by operating_point_hidden — one operating point pins down all
    four confusion cells — so this is exactly the table's `recall` verdict.

    A row whose counts are missing is treated as hidden rather than shown: the
    rule cannot be certified for it, and results_row_hidden refuses to guess a TP
    from a 3-decimal recall.
    """
    hidden = set()
    for _, r in results_df.iterrows():
        try:
            small = disclosure.results_row_hidden(r)["recall"]
        except (ValueError, KeyError):
            small = True
        if small:
            hidden.add(_hidden_key(str(r["model"]), str(r["feature_set"])))
    return hidden


# ── The two walks ─────────────────────────────────────────────────────────────

def _data_walk(results_df: pd.DataFrame, model: str, rules: SurveyRuleSet) -> list[dict]:
    """
    Reconstruct the data-input gate for one model, recording what each step
    demanded as well as whether it passed.

    Mirrors _data_permissible_feature_sets: sequential, and the walk stops at the
    first rejection — later steps are never reached, so they carry no demand.
    """
    steps: list[dict] = []
    entry = _FS_ORDER[0]

    # A model the rules hard-block is never put through the data gate at all
    # (the table marks its data-input verdict "---"), so it carries no demand.
    if model in rules.blocked_models:
        return [{"feature_set": fs, "prev": None, "recall": _recall(results_df, model, fs),
                 "threshold": None, "demanded": np.nan, "delta": np.nan,
                 "verdict": "reference", "blocked": True}
                for fs in _FS_ORDER]

    steps.append({
        "feature_set": entry, "prev": None, "recall": _recall(results_df, model, entry),
        "threshold": None, "demanded": np.nan, "delta": np.nan,
        "verdict": "entry", "blocked": False,
    })

    held    = entry           # last accepted set — the anchor the next step is measured from
    stopped = False
    for prev, fs in zip(_FS_ORDER, _FS_ORDER[1:]):
        r_cur     = _recall(results_df, model, fs)
        blocked   = fs in rules.blocked_feature_sets
        threshold = rules.fs_gain_required.get(fs)

        if blocked or stopped:
            steps.append({
                "feature_set": fs, "prev": prev, "recall": r_cur,
                "threshold": threshold, "demanded": np.nan, "delta": np.nan,
                "verdict": "blocked" if blocked else "not_reached", "blocked": blocked,
            })
            continue

        r_prev = _recall(results_df, model, held)
        delta  = r_cur - r_prev
        passed = delta >= threshold
        steps.append({
            "feature_set": fs, "prev": held, "recall": r_cur,
            "threshold": threshold, "demanded": r_prev + threshold, "delta": delta,
            "verdict": "accepted" if passed else "rejected", "blocked": False,
        })
        if passed:
            held = fs
        else:
            stopped = True   # the ladder is cumulative; nothing above is evaluated
    return steps


def _expl_walk(results_df: pd.DataFrame, feature_set: str, rules: SurveyRuleSet) -> list[dict]:
    """
    Reconstruct the explainability gate on one feature set.

    Mirrors select_model: start at the most explainable model with a recall
    (credits_policy), then step down a tier at a time. Unlike the data gate a
    rejection does not stop the walk — select_model keeps descending, measuring
    each tier against the model still held, at the summed price of every level
    between the two (SurveyRuleSet.expl_gain_between).
    """
    steps: list[dict] = []
    reachable = {level for level, _ in rules.expl_groups}

    entry = _EXPL_AXIS[0]
    steps.append({
        "model": entry, "recall": _recall(results_df, entry, feature_set),
        "threshold": None, "demanded": np.nan, "delta": np.nan, "measured_against": None,
        "verdict": "entry", "blocked": False,
    })

    held = entry
    ## Models sharing a tier would all be measured against
    ## the model held when the tier was reached, as select_model does; the tier
    ## then hands on its strongest accepted member.
    tier_level, tier_anchor, tier_best = None, entry, None
    for model in _EXPL_AXIS[1:]:  # noqa: PLR1702
        r_cur   = _recall(results_df, model, feature_set)
        blocked = model in rules.blocked_models
        level   = EXPLAINABILITY_LEVEL[model]
        if level != tier_level:
            if tier_best is not None:
                held = tier_best
            tier_level, tier_anchor, tier_best = level, held, None
        held = tier_anchor
        threshold = (rules.expl_gain_between(EXPLAINABILITY_LEVEL[held], level)
                     if level in reachable else None)

        if blocked or threshold is None:
            # A blocked tier is never measured against anything, so it carries no
            # threshold either — reporting one invites reading it as a near miss.
            steps.append({
                "model": model, "recall": r_cur, "threshold": None if blocked else threshold,
                "demanded": np.nan, "delta": np.nan, "measured_against": None,
                "verdict": "blocked" if blocked else "not_reached", "blocked": blocked,
            })
            continue

        r_held = _recall(results_df, held, feature_set)
        delta  = r_cur - r_held
        passed = delta >= threshold
        steps.append({
            "model": model, "recall": r_cur, "threshold": threshold,
            "demanded": r_held + threshold, "delta": delta, "measured_against": held,
            "verdict": "accepted" if passed else "rejected", "blocked": False,
        })
        if passed and (tier_best is None
                       or r_cur > _recall(results_df, tier_best, feature_set)):
            tier_best = model
    return steps


def _cell_demands(
    results_df: pd.DataFrame,
    rules: SurveyRuleSet,
    hidden: set[tuple[str, str]] | None = None,
    models: list[str] | None = None,
) -> tuple[dict[tuple[str, str], dict], dict, dict]:
    """
    Collapse both gates into one demanded recall per (model, feature set).

    Both rules put a floor under the same quantity — the recall of this model on
    this feature set — so a cell has to clear whichever floor is higher:

        demanded = max( recall(model, last accepted set) + data threshold,
                        recall(held model, this set)     + explainability threshold )

    The binding half is worth keeping: it separates "this step did not buy enough
    over the smaller feature set" from "this model is not explainable enough for
    what it bought over the simpler model". Each demand also carries the anchor it
    was measured from, so it can be drawn as a line leaving that point.

    A cell the gates never reach — a hard block, or anything above a rejected data
    step — has no demand at all, and is reported as unreachable rather than failed.

    Every cell carries the true value AND a `*_shown` twin that is NaN wherever
    statistical disclosure control forbids drawing it. Verdicts are computed from
    the true values — the gates must agree with select_grid_config whatever the
    figure may print — and only the drawing reads the `*_shown` ones. A demand is
    `recall(anchor) + a published threshold`, so it gives the anchor away exactly
    as the anchor's own point would: it is suppressed with its anchor, not with
    the cell it applies to.
    """
    hidden = _hidden_recalls(results_df) if hidden is None else hidden
    models = list(_ML_MODELS) if models is None else models
    data_walks = {m: _data_walk(results_df, m, rules) for m in models}
    expl_walks = {fs: _expl_walk(results_df, fs, rules)
                  for fs, _, _ in rules.fs_ladder}

    cells: dict[tuple[str, str], dict] = {}
    for model in models:
        for i, fs in enumerate(_FS_ORDER):
            step_d  = data_walks[model][i]
            recall  = step_d["recall"]
            reached = _evaluated(step_d["verdict"])

            recall_hidden = _hidden_key(model, fs) in hidden
            cell = {
                "model": model, "feature_set": fs, "recall": recall,
                "data_demand": np.nan, "expl_demand": np.nan, "demand": np.nan,
                "binding": None, "data_anchor": None, "expl_anchor": None,
                "reached": reached, "permissible": False,
                "recall_hidden": recall_hidden,
                "recall_shown": np.nan if recall_hidden else recall,
                "data_demand_shown": np.nan, "expl_demand_shown": np.nan,
                "demand_shown": np.nan, "demand_hidden": False,
            }
            if not reached:
                cells[(model, fs)] = cell
                continue

            cell["data_demand"] = step_d["demanded"]
            cell["data_anchor"] = step_d["prev"]

            step_e = next((s for s in expl_walks.get(fs, []) if s["model"] == model), None)
            if step_e is not None:
                cell["expl_demand"] = step_e["demanded"]
                cell["expl_anchor"] = step_e["measured_against"]

            # -inf, not NaN: the entry feature set imposes no data floor, and the
            # most explainable model imposes no explainability floor. Absent is
            # not the same as unmet.
            d_data = cell["data_demand"] if cell["data_demand"] == cell["data_demand"] else -np.inf
            d_expl = cell["expl_demand"] if cell["expl_demand"] == cell["expl_demand"] else -np.inf
            if max(d_data, d_expl) > -np.inf:
                cell["binding"] = "data" if d_data >= d_expl else "explainability"
                cell["demand"]  = max(d_data, d_expl)
            cell["permissible"] = recall >= max(d_data, d_expl)

            # A floor is drawn only when the cell it is measured FROM may be
            # drawn. The data floor is anchored on this model at a smaller
            # feature set, the explainability floor on a more explainable model
            # in this column.
            data_anchor_hidden = (cell["data_anchor"] is not None
                                  and _hidden_key(model, cell["data_anchor"]) in hidden)
            expl_anchor_hidden = (cell["expl_anchor"] is not None
                                  and _hidden_key(cell["expl_anchor"], fs) in hidden)
            if not data_anchor_hidden:
                cell["data_demand_shown"] = cell["data_demand"]
            if not expl_anchor_hidden:
                cell["expl_demand_shown"] = cell["expl_demand"]
            binding_hidden = (data_anchor_hidden if cell["binding"] == "data"
                              else expl_anchor_hidden if cell["binding"] else False)
            cell["demand_hidden"] = bool(binding_hidden)
            if not binding_hidden:
                cell["demand_shown"] = cell["demand"]
            cells[(model, fs)] = cell

    return cells, data_walks, expl_walks


def _ylim(cells, cp_recall) -> tuple[float, float]:
    """
    One recall range shared by all four panels of the paper figure — never
    per-panel autoscale, or the panels stop being comparable.

    Lives here with the rest of the gate's logic, and is written into the
    prepared input, so the drawing never autoscales from what it was handed.

    Both individual floors are included alongside the resolved demand so that
    the range does not shift if the figure is ever drawn with them visible.

    Only drawable values: taking the range from a suppressed point would let its
    position be read back off the padding around it.
    """
    y = [cp_recall]
    for cell in cells.values():
        y += [cell["recall_shown"], cell["demand_shown"],
              cell["data_demand_shown"], cell["expl_demand_shown"]]
    y = [v for v in y if v == v and v is not None]
    if not y:
        y = [0.0, 1.0]
    lo, hi = min(y), max(y)
    pad = 0.10 * (hi - lo)
    return max(0.0, lo - 1.7 * pad), min(1.0, hi + 1.9 * pad)


def _cross_check(
    grid_df: pd.DataFrame,
    data_walks: dict[str, list[dict]],
    expl_walks: dict[str, list[dict]],
    cells: dict[tuple[str, str], dict] | None = None,
) -> list[str]:
    """
    Assert the reconstructed walks agree with select_grid_config, cell by cell.
    The demand lines are only honest if the verdicts they imply are the ones the
    selection code produced.

    Here is where it would be caught if the figure's reconstruction of the walk
    and select_model came apart.
    """
    problems = []
    grid = grid_df.set_index(["model", "feature_set"])
    for model, steps in data_walks.items():
        for step in steps:
            key = (model, step["feature_set"])
            if key not in grid.index:
                continue
            expected = bool(grid.loc[key, "data_permissible"])
            drawn    = step["verdict"] in ("entry", "accepted")
            if expected != drawn:
                problems.append(
                    f"data gate — {model} / {step['feature_set']}: figure says "
                    f"{'pass' if drawn else 'fail'}, select_grid_config says "
                    f"{'pass' if expected else 'fail'}"
                )

    for fs, steps in expl_walks.items():
        for step in steps:
            key = (step["model"], fs)
            if key not in grid.index:
                continue
            expected = bool(grid.loc[key, "expl_permissible"])
            drawn    = step["verdict"] in ("entry", "accepted")   # own step, not a ceiling
            if expected != drawn:
                problems.append(
                    f"explainability gate — {step['model']} / {fs}: figure says "
                    f"{'pass' if drawn else 'fail'}, select_grid_config says "
                    f"{'pass' if expected else 'fail'}"
                )
    if cells:
        for (model, fs), cell in cells.items():
            key = (model, fs)
            if key not in grid.index or not cell["reached"]:
                continue
            expected = bool(grid.loc[key, "permissible"])
            if expected != cell["permissible"]:
                problems.append(
                    f"combined demand — {model} / {fs}: figure says "
                    f"{'permissible' if cell['permissible'] else 'not permissible'}, "
                    f"select_grid_config says "
                    f"{'permissible' if expected else 'not permissible'}"
                )
    return problems


# ── Drawing helpers ───────────────────────────────────────────────────────────

_DATA_DASH = (0, (4, 2))     # data-input rule — measured across feature sets
_EXPL_DOT  = (0, (1, 1.8))   # explainability rule — measured across models

# The selection ring annotates a cell rather than encoding a quantity, so it is
# deliberately the one black mark on the panel: nothing else is black, so it
# cannot be read as a fifth series. Bright fills were tried and are worse — a
# saturated colour has low contrast against white, so the ring reads as thin,
# and green or amber both sit close to colours already in use (the teals, and
# the credits-policy line).
_SELECTED_COLOR = "#333333"

# Margin shading: the band between what a cell achieved and a bar it had to
# clear. Height is the margin, colour is whether that bar was cleared. Green is
# a grass green rather than the Wong bluish green, which is too close to
# FasterRisk's teal to read as a different kind of object.
_MARGIN_PASS = "#66A61E"
_MARGIN_FAIL = "#D55E00"

_BAR_HALF_WIDTH = 0.048
_DODGE          = 0.11   # models get their own lane inside each feature-set column


def _evaluated(verdict: str) -> bool:
    """Did the gate actually rule on this step? Steps above an earlier rejection,
    and hard-blocked ones, are never reached — they carry no verdict to draw."""
    return verdict in ("entry", "accepted", "rejected")


# ── Figure ────────────────────────────────────────────────────────────────────

def _expl_threshold_label(model: str, rules: SurveyRuleSet) -> str:
    """The recall gain this model's tier costs over the next-more-explainable one."""
    if model in rules.blocked_models:
        return "never"
    thr = rules.expl_gain_required.get(EXPLAINABILITY_LEVEL[model])
    return f"+{thr * 100:g} pp" if thr is not None else "—"
