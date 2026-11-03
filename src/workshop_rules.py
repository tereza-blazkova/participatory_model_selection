"""
Workshop-grounded model configuration selection.

Applies two sets of workshop rules to existing evaluation results:
  1. Data-input rules  — which feature set is acceptable
  2. Explainability rules — which model is acceptable

Both sets of rules come from the workshop survey medians, and the survey was
run **twice**: once before any group discussion (baseline) and once after
deliberation (endline). Each administration therefore defines its own rule set:

  ENDLINE_RULES   — post-deliberation medians; the Participatory model reported
                    in the paper. Default everywhere.
  BASELINE_RULES  — pre-deliberation medians; the model the same participants
                    would have specified *before* discussing with each other.

Selecting under both and comparing the resulting models is what isolates the
effect of deliberation on the system (see src/workshop_evaluation.py
--ruleset both). Medians are taken from SI Table `tab:median-shift`.

Equal-opportunity recall across groups is then computed for the selected
configuration as an *audit* and reported alongside it. It is deliberately not
a selection rule: it never changes the selected model or feature set. The
acceptability assessment of those gaps happens downstream (see
src/equalized_recall_audit.py and src/fairness_audit.py).

Run via:  python src/workshop_evaluation.py
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

import numpy as np
import pandas as pd

from src.config import AUDIT_ONLY_CONFIGS, MAIN_TEST_YEAR, SELECTION_YEAR

# ── Thresholds ────────────────────────────────────────────────────────────────
# The survey response scale, expressed as the recall gain required to accept a
# less private / less explainable option. NEVER encodes "never acceptable".

NO_THRESHOLD         = 0.0    # "acceptable as long as accuracy does not worsen"
SMALL_IMPROVEMENT    = 0.01   # 1pp  (lower bound of 1–3pp)
MODERATE_IMPROVEMENT = 0.05   # 5pp
LARGE_IMPROVEMENT    = 0.10   # 10pp
NEVER: None          = None   # never acceptable — hard block
FAIRNESS_TOLERANCE   = 0.03   # recall gap ≤ 3pp is acceptable across a group

# ── Explainability ────────────────────────────────────────────────────────────

EXPLAINABILITY_LEVEL: dict[str, str] = {
    "credits_policy":      "very_high",  # rule-based baseline — not an ML candidate
    "fasterrisk":          "high",
    "logistic_regression": "moderate",
    "xgboost":             "low",
    "mlp":                 "very_low",
}

# Every explainability tier, most → least explainable. credits_policy (very_high)
# is the entry baseline — never a candidate, only retained when nothing passes
# (select_grid_config). Which of the tiers
# below it are reachable is decided by the rule set, not by this list.
_ALL_EXPL_GROUPS: list[tuple[str, list[str]]] = [
    ("very_high", ["credits_policy"]),
    ("high",      ["fasterrisk"]),
    ("moderate",  ["logistic_regression"]),
    ("low",       ["xgboost"]),
    ("very_low",  ["mlp"]),
]

# Explainability level ordering from most to least explainable (index = complexity rank)
_EXPL_LEVEL_ORDER: list[str] = [level for level, _ in _ALL_EXPL_GROUPS]

# Cumulative feature-set ladder, in order. a_credits is the entry point.
_FS_ORDER: list[str] = [
    "a_credits",
    "b_credits_performance",
    "c_plus_enrolment",
    "d_plus_lms",
    "e_plus_sensitive",
]

# Rule-based models that serve as ladder baselines. Never a candidate in the
# ranking; the credits policy is only retained when no combination is permissible.
_CREDITS_POLICY = "credits_policy"
_RULE_BASED_MODELS: frozenset[str] = frozenset({_CREDITS_POLICY})

# Sensitivity check only — never the Participatory model. The selection judges
# the data rule on each model's own recalls; the sensitivity variant judges it on
# this one model for all of them, so a reader can see whether that choice matters.
_SENSITIVITY_DATA_REFERENCE = "logistic_regression"

# ── Where the selection is stored ─────────────────────────────────────────────
# One name per artefact, one reader (participatory_config / participatory_results),
# so every figure and table resolves the same Participatory model as the
# selection table. Other rule sets write `{stem}_{ruleset}{suffix}`.
SELECTION_ALGORITHM   = "grid"
CONFIG_FILE           = "workshop_config.json"        # under results/models/
RESULTS_FILE          = "workshop_results.csv"        # under results/
CANDIDATES_FILE       = "workshop_candidates.csv"     # under results/
DELIBERATION_FILE     = "workshop_deliberation_comparison.csv"  # under results/


# ── Survey-derived rule sets ──────────────────────────────────────────────────

@dataclass(frozen=True)
class SurveyRuleSet:
    """
    One administration of the workshop survey, translated into selection rules.

    fs_gain_required
        feature set → recall gain required to add it to the previous set.
        `NEVER` (None) = never acceptable. Because the ladder is cumulative, a
        blocked set also blocks every set above it.
    expl_gain_required
        explainability level → recall gain required to accept a model at that
        level over the currently selected, more explainable one. `NEVER` blocks
        the level and everything below it. `very_high` is the entry baseline and
        takes no threshold.
    """
    name:  str
    label: str
    fs_gain_required:   Mapping[str, Optional[float]]
    expl_gain_required: Mapping[str, Optional[float]]
    notes: tuple[str, ...] = ()
    ## The cumulative ladder, entry first. None = the paper's (_FS_ORDER). Only
    ## the robustness check of the ladder order (src/selection_robustness.py)
    ## sets it; fs_gain_required is then keyed by these rung names.
    fs_order: Optional[tuple[str, ...]] = None

    # ── Feature sets ──────────────────────────────────────────────────────────
    @property
    def fs_ladder(self) -> list[tuple[str, Optional[str], Optional[float]]]:
        """Reachable feature sets as (feature_set, prev_fs, threshold), truncated
        at the first never-acceptable step."""
        order = list(self.fs_order or _FS_ORDER)
        ladder: list[tuple[str, Optional[str], Optional[float]]] = [(order[0], None, None)]
        for prev, fs in zip(order, order[1:]):
            thr = self.fs_gain_required.get(fs, NEVER)
            if thr is NEVER:
                break
            ladder.append((fs, prev, thr))
        return ladder

    @property
    def blocked_feature_sets(self) -> frozenset[str]:
        reachable = {fs for fs, _, _ in self.fs_ladder}
        return frozenset(fs for fs in (self.fs_order or _FS_ORDER) if fs not in reachable)

    # ── Models ────────────────────────────────────────────────────────────────
    @property
    def expl_groups(self) -> list[tuple[str, list[str]]]:
        """Reachable explainability tiers, truncated at the first blocked level."""
        groups = [_ALL_EXPL_GROUPS[0]]
        for level, models in _ALL_EXPL_GROUPS[1:]:
            if self.expl_gain_required.get(level, NEVER) is NEVER:
                break
            groups.append((level, models))
        return groups

    @property
    def blocked_models(self) -> frozenset[str]:
        reachable = {m for _, models in self.expl_groups for m in models}
        return frozenset(m for _, models in _ALL_EXPL_GROUPS for m in models
                         if m not in reachable)

    def expl_gain_between(self, held_level: str, target_level: str) -> float:
        """
        Recall gain a model at `target_level` needs over the held model at `held_level`.

        The survey priced each level against the NEXT MORE EXPLAINABLE one ("each
        explainability level is compared to the level with higher explainability"),
        so a threshold is the price of one step down. When the walk has rejected an
        intermediate level and moves past it, the target must pay for every step it
        takes: the sum of the thresholds of the levels after `held_level`, up to and
        including `target_level`. Without a skip this is the target's own threshold.

        Not the threshold of the level just above the target measured against that
        level's rejected model: then a weaker rejected model would make the next
        level EASIER to accept, although neither model in contention changed.
        """
        order = [level for level, _ in self.expl_groups]
        i, j = order.index(held_level), order.index(target_level)
        if j <= i:
            raise ValueError(f"{target_level!r} is not below {held_level!r} on the ladder")
        # Rounded so that 0.01 + 0.05 is 0.06, not 0.060000000000000005.
        return round(sum(self.expl_gain_required[lvl] for lvl in order[i + 1:j + 1]), 10)


# Post-deliberation (endline) medians — the rules reported in the paper.
#   data: performance = small · enrolment = large · LMS = never (→ sensitive blocked too)
#   explainability: high = moderate · moderate = small · low = moderate · very low = never
ENDLINE_RULES = SurveyRuleSet(
    name="endline",
    label="post-deliberation (endline survey)",
    fs_gain_required={
        "b_credits_performance": SMALL_IMPROVEMENT,
        "c_plus_enrolment":      LARGE_IMPROVEMENT,
        "d_plus_lms":            NEVER,
        "e_plus_sensitive":      NEVER,   # unreachable: cumulative set contains LMS
    },
    expl_gain_required={
        "high":     MODERATE_IMPROVEMENT,
        "moderate": SMALL_IMPROVEMENT,
        "low":      MODERATE_IMPROVEMENT,
        "very_low": NEVER,
    },
    notes=(
        "e_plus_sensitive is blocked because the cumulative set contains LMS data, "
        "which the endline median marks never acceptable.",
    ),
)

# Pre-deliberation (baseline) medians — the same participants, before discussion.
#   data: performance = large · enrolment = large · LMS = large (acceptable!)
#   sensitive (Q3.5, accuracy framing): age = no worsening · gender = moderate ·
#     nationality = large. e_plus_sensitive is a bundle, so one threshold is needed:
#     the default takes the most restrictive of the three attributes actually in the
#     data (nationality, large), matching the workshop's conservative tie-break rule.
#   explainability: high = no threshold · moderate = small · low = large · very low = never
BASELINE_RULES = SurveyRuleSet(
    name="baseline",
    label="pre-deliberation (baseline survey)",
    fs_gain_required={
        "b_credits_performance": LARGE_IMPROVEMENT,
        "c_plus_enrolment":      LARGE_IMPROVEMENT,
        "d_plus_lms":            LARGE_IMPROVEMENT,
        "e_plus_sensitive":      LARGE_IMPROVEMENT,
    },
    expl_gain_required={
        "high":     NO_THRESHOLD,
        "moderate": SMALL_IMPROVEMENT,
        "low":      LARGE_IMPROVEMENT,
        "very_low": NEVER,
    },
    notes=(
        "Sensitive-attribute threshold aggregated conservatively across the three "
        "attributes present in the data (age = no worsening, gender = moderate, "
        "nationality = large → large).",
        "Q3.6 (fairness framing, 'only if fairness improves') is identical at baseline "
        "and endline and is audited, never applied as a selection rule.",
    ),
)

# Sensitivity check: median rather than most-restrictive aggregation of the three
# baseline sensitive-attribute thresholds (0.0 / 0.05 / 0.10 → 0.05).
BASELINE_RULES_MEDIAN_SENSITIVE = SurveyRuleSet(
    name="baseline_median_sensitive",
    label="pre-deliberation (baseline survey, median sensitive aggregation)",
    fs_gain_required={**BASELINE_RULES.fs_gain_required,
                      "e_plus_sensitive": MODERATE_IMPROVEMENT},
    expl_gain_required=BASELINE_RULES.expl_gain_required,
    notes=BASELINE_RULES.notes[1:] + (
        "Sensitive-attribute threshold = median of the three baseline medians "
        "(age 0pp, gender 5pp, nationality 10pp) instead of the most restrictive.",
    ),
)

RULE_SETS: dict[str, SurveyRuleSet] = {
    rs.name: rs for rs in (ENDLINE_RULES, BASELINE_RULES, BASELINE_RULES_MEDIAN_SENSITIVE)
}

DEFAULT_RULES = ENDLINE_RULES


def get_rules(rules: "str | SurveyRuleSet | None") -> SurveyRuleSet:
    """Resolve a rule-set name (or pass through a SurveyRuleSet / None → default)."""
    if rules is None:
        return DEFAULT_RULES
    if isinstance(rules, SurveyRuleSet):
        return rules
    try:
        return RULE_SETS[rules]
    except KeyError:
        raise ValueError(
            f"unknown rule set {rules!r}; available: {sorted(RULE_SETS)}"
        ) from None


# ── Backwards-compatible module-level views of the default (endline) rules ────
# Existing importers (generate_selection_table.py) rely
# on these names. They describe ENDLINE_RULES only; anything rule-set-aware should
# read the SurveyRuleSet instead.

_EXPL_GROUPS: list[tuple[str, list[str]]]                     = DEFAULT_RULES.expl_groups
_NEVER_ACCEPTABLE_MODELS: frozenset[str]                      = DEFAULT_RULES.blocked_models
_NEVER_ACCEPTABLE_FEATURE_SETS: frozenset[str]                = DEFAULT_RULES.blocked_feature_sets
_FS_LADDER: list[tuple[str, str | None, float | None]]        = DEFAULT_RULES.fs_ladder


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class WorkshopConfig:
    feature_set: str
    model: str
    max_non_sensitive_fs: str          # best accepted set without sensitive features
    rules: str = DEFAULT_RULES.name    # which survey administration produced the rules
    test_year: int = MAIN_TEST_YEAR
    data_logs: dict[str, list[dict]] = field(default_factory=dict)   # model → data-ladder walk
    expl_logs: dict[str, list[dict]] = field(default_factory=dict)   # feature set → explainability walk
    sensitivity: Optional[tuple[str, str]] = None   # pick with the data rule judged on LR for all models
    fairness_df: Optional[pd.DataFrame] = None
    fairness_gaps: dict[str, float] = field(default_factory=dict)
    fairness_notes: list[str] = field(default_factory=list)
    candidates_df: Optional[pd.DataFrame] = None


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _recall(
    results_df: pd.DataFrame,
    model: str,
    feature_set: str,
    test_year: int,
) -> float:
    mask = (results_df["model"] == model) & (results_df["test_year"] == test_year)
    if model not in _RULE_BASED_MODELS:
        mask &= results_df["feature_set"] == feature_set
    row = results_df[mask]
    return float(row["enr_recall"].iloc[0]) if len(row) > 0 else np.nan


def _precision(
    results_df: pd.DataFrame,
    model: str,
    feature_set: str,
    test_year: int,
) -> float:
    mask = (results_df["model"] == model) & (results_df["test_year"] == test_year)
    if model not in _RULE_BASED_MODELS:
        mask &= results_df["feature_set"] == feature_set
    row = results_df[mask]
    return float(row["enr_precision"].iloc[0]) if len(row) > 0 else np.nan


def data_ladder_walk(
    results_df: pd.DataFrame,
    model: str,
    test_year: int,
    rules: SurveyRuleSet = DEFAULT_RULES,
) -> tuple[list[str], list[dict]]:
    """
    The data-input rule for one model, as the paper states it.

    Walks the cumulative ladder on this model's own recalls. Each step must gain
    at least its threshold over the previous step, and the walk STOPS at the first
    rejection: the sets are cumulative, so every set above a rejected step contains
    the data group that was just refused. Hard-blocked sets are never
    reached — `rules.fs_ladder` is truncated before them.

    Returns (permissible feature sets in ladder order, one log entry per step).
    """
    permissible: list[str] = []
    log: list[dict] = []
    for fs, prev_fs, threshold in rules.fs_ladder:
        if prev_fs is None:
            permissible.append(fs)
            log.append({"step": f"{fs} (baseline)", "verdict": "BASELINE",
                        "recall": _recall(results_df, model, fs, test_year)})
            continue
        delta  = (_recall(results_df, model, fs, test_year)
                  - _recall(results_df, model, prev_fs, test_year))
        accept = bool(delta >= threshold)   # NaN recall → rejected
        log.append({"step": f"{prev_fs} → {fs}",
                    "verdict": "ACCEPTED" if accept else "REJECTED",
                    "delta": delta, "threshold": threshold})
        if not accept:
            break
        permissible.append(fs)
    return permissible, log


def _data_permissible_feature_sets(
    results_df: pd.DataFrame,
    model: str,
    test_year: int,
    rules: SurveyRuleSet = DEFAULT_RULES,
) -> list[str]:
    """Feature sets `model` may use under the data-input rule (see data_ladder_walk)."""
    return data_ladder_walk(results_df, model, test_year, rules)[0]


def explainability_accepted(
    results_df: pd.DataFrame,
    feature_set: str,
    test_year: int,
    rules: SurveyRuleSet = DEFAULT_RULES,
) -> set[str]:
    """
    The models that pass the explainability rule on this feature set: those the
    walk ACCEPTED, each at its own step (select_model).

    Not "every model at least as explainable as the last one accepted": a model
    rejected at its own step fails even when a less explainable one is accepted
    later. Otherwise a model that did not clear its own threshold could be
    selected because a model that cannot be used on this feature set (it fails the
    data rule here) cleared a later one.

    A tier is a list, though each now holds one model: the walk tests the strongest,
    and every member that clears the same demand is accepted.
    """
    _, log = select_model(results_df, feature_set, test_year, rules=rules)
    accepted: set[str] = set()
    for e in log:
        if e["verdict"] == "BASELINE":
            accepted.add(e["model"])
        elif e["verdict"] == "ACCEPTED":
            for m in e["models"]:
                r = _recall(results_df, m, feature_set, test_year)
                if not np.isnan(r) and r - e["against_recall"] >= e["threshold"]:
                    accepted.add(m)
    return accepted


# ── Model selection ───────────────────────────────────────────────────────────

def select_model(
    results_df: pd.DataFrame,
    feature_set: str,
    test_year: int = MAIN_TEST_YEAR,
    rules: SurveyRuleSet = DEFAULT_RULES,
) -> tuple[str, list[dict]]:
    """
    Apply explainability workshop rules. Start at credits_policy (very_high).
    At each step down, accept only if the recall gain over the model still held
    meets the price of every level between the two (SurveyRuleSet.expl_gain_between).
    A rejection does not stop the walk: the rejected model is not used, so a less
    explainable level may still be worth its combined price.
    Levels the rule set marks never acceptable (MLP / very_low under both survey
    administrations) are excluded and absent from the ladder.
    credits_policy is the baseline — returned if no ML model clears its threshold.

    Returns (selected_model, log).
    """
    if feature_set in rules.blocked_feature_sets:
        raise ValueError(
            f"feature_set '{feature_set}' is never acceptable under the "
            f"{rules.name} rules (blocked: {sorted(rules.blocked_feature_sets)})."
        )

    log = []
    expl_groups = rules.expl_groups

    model_info: dict[str, dict] = {}
    for level, models in expl_groups:
        for m in models:
            m_recall = _recall(results_df, m, feature_set, test_year)
            model_info[m] = {
                "level":  level,
                "recall": m_recall,
            }

    # ── Baseline: most explainable model with a valid recall ──────────────────
    current_model  = None
    current_level  = None
    current_recall = np.nan
    baseline_idx   = 0

    for idx, (level, models) in enumerate(expl_groups):
        eligible = [m for m in models if not np.isnan(model_info[m]["recall"])]
        if eligible:
            best = max(eligible, key=lambda m: model_info[m]["recall"])
            current_model  = best
            current_level  = level
            current_recall = model_info[best]["recall"]
            baseline_idx   = idx
            break

    if current_model is None:
        current_model  = expl_groups[0][1][0]
        current_level  = expl_groups[0][0]
        baseline_idx   = 0

    recall_note = (
        f"enr_recall = {current_recall:.3f}"
        if not np.isnan(current_recall)
        else "enr_recall = N/A (model skipped)"
    )
    log.append({
        "step":    f"baseline: {current_model} ({current_level})",
        "model":   current_model,
        "verdict": "BASELINE",
        "delta":   None,
        "note":    recall_note,
    })

    # ── Walk ladder from baseline level down ──────────────────────────────────
    for next_level, next_models in expl_groups[baseline_idx + 1:]:
        threshold     = rules.expl_gain_between(current_level, next_level)
        eligible_next = [m for m in next_models if not np.isnan(model_info[m]["recall"])]

        if not eligible_next:
            log.append({
                "step":      "/".join(next_models) + f" ({next_level})",
                "verdict":   "REJECTED",
                "delta":     None,
                "threshold": threshold,
                "note":      "no recall available for any model in this group",
            })
            continue

        best_next   = max(eligible_next, key=lambda m: model_info[m]["recall"])
        best_recall = model_info[best_next]["recall"]
        eff_current = current_recall if not np.isnan(current_recall) else -np.inf
        delta       = best_recall - eff_current
        accept      = delta >= threshold
        held_model  = current_model

        if accept:
            current_model  = best_next
            current_level  = next_level
            current_recall = best_recall

        log.append({
            "step":          "/".join(next_models) + f" ({next_level})",
            "models":        list(next_models),
            "measured_against": held_model,
            "against_recall":   eff_current,
            "verdict":       "ACCEPTED" if accept else "REJECTED",
            "delta":         delta,
            "threshold":     threshold,
            "best_in_group": best_next,
            "best_recall":   best_recall,
        })

    return current_model, log


# ── Grid (independent data × explainability) selection ───────────────────────

def select_grid_config(
    results_df: pd.DataFrame,
    test_year: int = MAIN_TEST_YEAR,
    rules: SurveyRuleSet = DEFAULT_RULES,
    data_reference_model: Optional[str] = None,
) -> tuple[str, str, pd.DataFrame]:
    """
    Evaluate every (model, feature_set) combination independently against both rules.

    This is THE selection: the algorithm the paper's methods section
    describes, and the only one that produces the Participatory model.

    Data rule        — feature set must pass the sequential data-input gate for this
                       specific model (recall gain vs. same model on previous FS),
                       stopping at the first rejected step (data_ladder_walk).
                       With `data_reference_model`, every model inherits that one
                       model's verdicts instead — the sensitivity variant only.
    Explainability   — model must be ACCEPTED at its own step of the explainability
                       walk on this feature set (explainability_accepted).
    Hard blocks      — LMS / sensitive feature sets and very-low-explainability models
                       (mlp) are excluded unconditionally.

    A combination is permissible if both rules pass.
    The permissible combination with the highest recall is selected. The credits
    policy is the reference, never a candidate in that ranking; it is only the
    fallback: if no combination is permissible, no learned model has earned its
    required gain, and the credits policy is retained.

    Returns (selected_model, selected_fs, candidates_df). After the fallback,
    selected_fs is the label the policy is stored under in results_df
    ("ects_behind_ge_60"), and candidates_df["selected"] is False on every row:
    the frame holds candidates only, and none of them was selected.
    """
    ml_models    = [m for _, models in rules.expl_groups for m in models
                    if m not in _RULE_BASED_MODELS and m not in rules.blocked_models]
    candidate_fs = [fs for fs, _, _ in rules.fs_ladder]
    ## Audit-only configs (b_plus_sensitive) are trained and reported, but are
    ## not something the workshop could have selected. They are absent from every
    ## rule set's ladder; this makes that a guarantee rather than a coincidence.
    if set(candidate_fs) & AUDIT_ONLY_CONFIGS:
        raise ValueError(f"audit-only feature sets {sorted(set(candidate_fs) & AUDIT_ONLY_CONFIGS)} "
                         f"must never be selection candidates (rules {rules.name!r})")

    # Data-permissible feature sets per ML model
    data_ok: dict[str, set[str]] = {
        m: set(_data_permissible_feature_sets(
            results_df, data_reference_model or m, test_year, rules=rules))
        for m in ml_models
    }

    # Models accepted by the explainability walk, per feature set
    expl_ok_models: dict[str, set[str]] = {
        fs: explainability_accepted(results_df, fs, test_year, rules=rules)
        for fs in candidate_fs
    }

    # The credits policy is the reference the explainability walk starts from,
    # never a candidate: it does not compete in this ranking, and is only retained
    # below when no combination is permissible.
    rows = []
    for fs in candidate_fs:
        for model in ml_models:
            expl_ok     = model in expl_ok_models[fs]
            data_ok_    = fs in data_ok[model]
            permissible = data_ok_ and expl_ok
            r = _recall(results_df, model, fs, test_year)
            p = _precision(results_df, model, fs, test_year)
            rows.append({
                "feature_set":      fs,
                "model":            model,
                "recall":           r,
                "precision":        p,
                "data_permissible": data_ok_,
                "expl_permissible": expl_ok,
                "permissible":      permissible,
                "selected":         False,
            })

    candidates_df = pd.DataFrame(rows)

    eligible = candidates_df[candidates_df["permissible"]]
    if eligible.empty:
        # The paper's fallback: no learned model has earned its required gain, so
        # the credits policy is retained. With no ML results at all nothing was
        # tested, which is a broken run rather than a selection.
        evaluated = [m for m in ml_models
                     if not np.isnan(_recall(results_df, m, candidate_fs[0], test_year))]
        if not evaluated:
            raise ValueError(f"no ML model has a recall on {candidate_fs[0]} in {test_year}")
        best_model = _CREDITS_POLICY
        best_fs = str(results_df.loc[(results_df["model"] == best_model)
                                     & (results_df["test_year"] == test_year),
                                     "feature_set"].iloc[0])
        print(f"  ⚠ FALLBACK: no combination passes both rules — no learned model has "
              f"earned its required gain, so the credits policy is retained.")
    else:
        best_idx   = int(eligible["recall"].idxmax())
        best_model = str(candidates_df.loc[best_idx, "model"])
        best_fs    = str(candidates_df.loc[best_idx, "feature_set"])

    candidates_df["selected"] = (
        (candidates_df["model"] == best_model) &
        (candidates_df["feature_set"] == best_fs)
    )
    return best_model, best_fs, candidates_df


# ── Fairness assessment ───────────────────────────────────────────────────────

def compute_fairness_metrics(
    preds_df: pd.DataFrame,
    model: str,
    feature_set: str,
    df_original: pd.DataFrame,
    test_year: int = MAIN_TEST_YEAR,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Compute equal-opportunity recall for each binary group:
    female, danish_nationality, older_for_cohort.

    Returns
    -------
    detail_df : DataFrame  [subgroup_col, group_label, recall, n_dropout]
    gaps : dict  {subgroup_col: max_recall - min_recall}
    """
    pred = preds_df[
        (preds_df["test_year"] == test_year) &
        (preds_df["model"] == model) &
        (preds_df["feature_set"] == feature_set)
    ].copy()

    if len(pred) == 0:
        return pd.DataFrame(), {}

    ## Entry age relative to the level-and-cohort median, derived once in
    ## src/data.py, so this audit splits on exactly the axis every fairness
    ## artefact reports; a pooled median would largely reproduce BA vs MA.
    ## Not stored in preds_df — merged from the frame.
    if "older_for_cohort" in df_original.columns:
        age_lookup = (
            df_original.drop_duplicates("enrollment")
            .set_index("enrollment")["older_for_cohort"]
        )
        pred["older_for_cohort"] = pred["enrollment"].map(age_lookup)

    group_specs = [
        ("female",             {0: "male",        1: "female"}),
        ("danish_nationality", {0: "non-danish",   1: "danish"}),
        ("older_for_cohort",   {0: "younger for cohort", 1: "older for cohort"}),
    ]

    records = []
    for col, labels in group_specs:
        if col not in pred.columns:
            continue
        for val, label in labels.items():
            sub      = pred[(pred[col] == val) & (pred["dropped_out"] == 1)]
            n_drop   = len(sub)
            recall   = float((sub["ever_flagged"] == 1).mean()) if n_drop > 0 else np.nan
            records.append({
                "subgroup_col": col,
                "group_label":  label,
                "recall":       recall,
                "n_dropout":    n_drop,
            })

    detail_df = pd.DataFrame(records)
    gaps = {
        col: float(grp["recall"].max() - grp["recall"].min())
        for col, grp in detail_df.groupby("subgroup_col")
        if grp["recall"].notna().all()
    }

    return detail_df, gaps


# ── Main entry point ──────────────────────────────────────────────────────────

def run_workshop_selection(
    results_df: pd.DataFrame,
    preds_df: Optional[pd.DataFrame],
    df_original: pd.DataFrame,
    test_year: int = MAIN_TEST_YEAR,
    rules: str | SurveyRuleSet = DEFAULT_RULES,
) -> WorkshopConfig:
    """
    Run the grid selection (select_grid_config), print a formatted report, and
    return a WorkshopConfig.

    Parameters
    ----------
    preds_df : DataFrame or None
        Per-enrollment predictions from enrollment_predictions.csv.
        If None, the fairness section is skipped.
    rules : str or SurveyRuleSet
        Which survey administration's rules to apply — "endline" (default,
        post-deliberation, the paper's Participatory model) or "baseline"
        (pre-deliberation). See RULE_SETS.
    """
    rules = get_rules(rules)

    SEP = "═" * 72

    print(f"\n{SEP}")
    print("  WORKSHOP-GROUNDED MODEL SELECTION")
    print(f"  Evaluation year: {test_year}  |  Algorithm: {SELECTION_ALGORITHM}")
    print(f"  Rule set: {rules.name} — {rules.label}")
    print(SEP)
    for note in rules.notes:
        print(f"  · {note}")

    ml_models = [m for _, models in rules.expl_groups for m in models
                 if m not in _RULE_BASED_MODELS and m not in rules.blocked_models]

    # ── Data rule: one walk per model, on its own recalls ─────────────────────
    print("\n▸ DATA-INPUT RULE  (per model, own recalls, stops at the first rejection)\n")
    data_logs: dict[str, list[dict]] = {}
    for m in ml_models:
        _, data_logs[m] = data_ladder_walk(results_df, m, test_year, rules)
        for e in data_logs[m]:
            if e["verdict"] == "BASELINE":
                print(f"  — {m:<20} {e['step']:<46}  recall = {e['recall']:.3f}")
            else:
                icon = "✓" if e["verdict"] == "ACCEPTED" else "✗"
                print(f"  {icon} {m:<20} {e['step']:<46}  Δrecall = {e['delta']:+.1%}"
                      f"  (need ≥ {e['threshold']:.0%})  →  {e['verdict']}")

    # ── Explainability rule: one walk per feature set ─────────────────────────
    print("\n▸ EXPLAINABILITY RULE  (per feature set; a model passes if accepted at its own step)\n")
    expl_logs: dict[str, list[dict]] = {}
    for fs, _, _ in rules.fs_ladder:
        _, expl_logs[fs] = select_model(results_df, fs, test_year, rules=rules)
        accepted = explainability_accepted(results_df, fs, test_year, rules=rules)
        print(f"  {fs:<28}  accepted: "
              + ", ".join(m for m in EXPLAINABILITY_LEVEL if m in accepted))

    # ── Both rules together ───────────────────────────────────────────────────
    print("\n▸ GRID  (a combination is permissible if it passes both rules)\n")
    selected_model, selected_fs, candidates_df = select_grid_config(
        results_df, test_year, rules=rules
    )
    if selected_fs == "e_plus_sensitive":
        # Best permissible combination that uses no sensitive features.
        non_sens = candidates_df[
            candidates_df["permissible"] &
            (candidates_df["feature_set"] != "e_plus_sensitive")
        ]
        max_non_sensitive_fs = (
            str(non_sens.loc[non_sens["recall"].idxmax(), "feature_set"])
            if not non_sens.empty else selected_fs
        )
    else:
        max_non_sensitive_fs = selected_fs

    print(f"  {'feature_set':<28}  {'model':<22}  {'recall':>7}  {'data':>5}  {'expl':>5}  {'ok':>4}")
    for _, row in candidates_df.iterrows():
        marker   = "  ← selected" if row["selected"] else ""
        data_str = "✓" if row["data_permissible"] else "✗"
        expl_str = "✓" if row["expl_permissible"] else "✗"
        ok_str   = "✓" if row["permissible"] else "✗"
        print(f"  {row['feature_set']:<28}  {row['model']:<22}  {row['recall']:>7.3f}"
              f"  {data_str:>5}  {expl_str:>5}  {ok_str:>4}{marker}")

    print(f"\n  → Selected: {selected_model} + {selected_fs}"
          f"  ({EXPLAINABILITY_LEVEL.get(selected_model, '?')} explainability)")

    # ── Sensitivity: the data rule judged on one reference model ──────────────
    # Recorded, never used: says whether judging data acceptability per model
    # (the paper's rule) rather than on a single reference model decides the pick.
    sens_model, sens_fs, _ = select_grid_config(
        results_df, test_year, rules=rules,
        data_reference_model=_SENSITIVITY_DATA_REFERENCE,
    )
    same = (sens_model, sens_fs) == (selected_model, selected_fs)
    print(f"\n  Sensitivity — data rule judged on {_SENSITIVITY_DATA_REFERENCE} for every model: "
          f"{sens_model} + {sens_fs}  ({'same pick' if same else 'DIFFERENT PICK'})")

    # ── Fairness ──────────────────────────────────────────────────────────────
    fairness_df   = None
    fairness_gaps = {}
    fairness_notes: list[str] = []

    if preds_df is None:
        print("\n▸ FAIRNESS ASSESSMENT  — skipped (enrollment_predictions.csv not found)\n")
    else:
        print("\n▸ FAIRNESS AUDIT  (equal opportunity by group — reported, not applied)\n")

        fairness_df, fairness_gaps = compute_fairness_metrics(
            preds_df, selected_model, selected_fs, df_original, test_year
        )

        if fairness_df.empty:
            print("  (no predictions available for selected model — fairness assessment skipped)\n")
        for col, grp in (fairness_df.groupby("subgroup_col") if not fairness_df.empty else []):
            gap  = fairness_gaps.get(col, np.nan)
            flag = "⚠ " if gap > FAIRNESS_TOLERANCE else "✓ "
            print(f"  {flag}{col}")
            for _, row in grp.iterrows():
                print(f"      {row['group_label']:<18}  recall = {row['recall']:.3f}  "
                      f"(n_dropout = {row['n_dropout']:,})")
            if not np.isnan(gap):
                gap_note = "acceptable" if gap <= FAIRNESS_TOLERANCE else f"gap > {FAIRNESS_TOLERANCE:.0%} — review"
                print(f"      recall gap = {gap:.1%}  ({gap_note})\n")
            if gap > FAIRNESS_TOLERANCE:
                fairness_notes.append(f"{col}: recall gap = {gap:.1%}")

        # Audit only — gaps above tolerance are reported, never acted on. The
        # selected model and feature set are fixed by the data-input and
        # explainability rules above and are not revised here.
        problematic = [col for col, gap in fairness_gaps.items() if gap > FAIRNESS_TOLERANCE]
        if problematic:
            print(
                f"  ⚠ Recall gap above {FAIRNESS_TOLERANCE:.0%} for: {', '.join(problematic)}.\n"
                "    Reported as an audit finding; the selected configuration is unchanged."
            )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  WORKSHOP CONFIGURATION")
    print(SEP)
    print(f"  Rule set:        {rules.name}  ({rules.label})")
    print(f"  Model:           {selected_model}")
    print(f"  Feature set:     {selected_fs}")
    print(f"  Explainability:  {EXPLAINABILITY_LEVEL.get(selected_model, '?')}")
    if fairness_notes:
        for note in fairness_notes:
            print(f"  Fairness audit:  {note}")
    else:
        print("  Fairness audit:  no gaps above tolerance detected")
    print(SEP + "\n")

    return WorkshopConfig(
        feature_set=selected_fs,
        model=selected_model,
        max_non_sensitive_fs=max_non_sensitive_fs,
        rules=rules.name,
        test_year=test_year,
        data_logs=data_logs,
        expl_logs=expl_logs,
        sensitivity=(sens_model, sens_fs),
        fairness_df=fairness_df,
        fairness_gaps=fairness_gaps,
        fairness_notes=fairness_notes,
        candidates_df=candidates_df,
    )


# ── Reading the stored selection ──────────────────────────────────────────────

def ruleset_file(name: str, rules: str | SurveyRuleSet = DEFAULT_RULES,
                 selection_year: int = SELECTION_YEAR) -> str:
    """
    File name for one rule set's copy of an artefact, selected on one cohort.
    Endline on SELECTION_YEAR keeps `name` — that file is the Participatory
    model. Any other rule set appends `_{ruleset}`, any other selection year
    `_sel{year}`, so neither can be read as it.
    """
    rules = get_rules(rules)
    stem, dot, suffix = name.rpartition(".")
    if rules.name != DEFAULT_RULES.name:
        stem += f"_{rules.name}"
    if selection_year != SELECTION_YEAR:
        stem += f"_sel{selection_year}"
    return f"{stem}{dot}{suffix}"


def stored_selection(results_path, rules: str | SurveyRuleSet = DEFAULT_RULES,
                     selection_year: int = SELECTION_YEAR) -> tuple[Optional[str], Optional[str]]:
    """
    (model, feature_set) stored for one rule set and selection year, or
    (None, None). For the Participatory model itself use participatory_config.
    """
    import json
    from pathlib import Path

    path = Path(results_path) / "models" / ruleset_file(CONFIG_FILE, rules, selection_year)
    if not path.exists():
        return None, None
    cfg = json.loads(path.read_text())
    if cfg.get("selection") != SELECTION_ALGORITHM:
        raise ValueError(f"{path} was not written by the {SELECTION_ALGORITHM} selection")
    return cfg["model"], cfg["feature_set"]


def participatory_config(results_path) -> tuple[Optional[str], Optional[str]]:
    """
    (model, feature_set) of the Participatory model, or (None, None) if no
    selection has been run. The model is credits_policy, with no saved model file
    or feature list, when no combination was permissible (select_grid_config).

    THE ONLY READER of results/models/workshop_config.json. It refuses a file
    that the grid selection did not write, so an output left by an earlier
    version of this code is never taken for the Participatory model.
    """
    import json
    from pathlib import Path

    path = Path(results_path) / "models" / CONFIG_FILE
    if not path.exists():
        return None, None
    cfg = json.loads(path.read_text())
    if cfg.get("selection") != SELECTION_ALGORITHM:
        raise ValueError(
            f"{path} was not written by the {SELECTION_ALGORITHM} selection "
            f"(found selection={cfg.get('selection')!r}). It is from an earlier "
            "version of this code — re-run "
            "src/workshop_evaluation.py (run_all.py Step 4)."
        )
    return cfg["model"], cfg["feature_set"]


def participatory_results(results_path) -> Optional[pd.DataFrame]:
    """Per-year results of the Participatory model (workshop_results.csv), or None."""
    from pathlib import Path

    participatory_config(results_path)   # refuses a stale selection before its results are used
    path = Path(results_path) / RESULTS_FILE
    return pd.read_csv(path) if path.exists() else None
