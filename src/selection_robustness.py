"""
How much does the workshop selection depend on the exact rule thresholds? (SI)

The selection rules are survey medians turned into recall thresholds (SI
tab:median-shift). This asks how far each threshold can move before the
selection changes, and what changes with it. Each variant moves ONE rule and
re-runs the grid selection (workshop_rules.select_grid_config — nothing
re-implemented) on SELECTION_YEAR:

  Threshold shifts  each data and explainability threshold moved by -5, -2, +2
                    and +5 percentage points, and one level along the survey's
                    own scale (no worsening < small < moderate < large < never)
                    in each direction. Reported: the configuration selected and
                    the change in its enrollment-level recall against the main
                    selection, in pp.
  Ladder order      the order in which the data groups are added (see below).

No survey response is read: every variant is a hypothetical move of the
published thresholds, so the table says how much the result would move IF the
participants' answers had been different by that much.

The recall change is the difference of two recalls, so it is withheld whenever
either operating point is (disclosure.results_row_hidden).

── Ladder order: what a different order can and cannot answer ──────────────
A threshold belongs to a data GROUP (performance, enrollment, LMS, sensitive),
but a rung of the ladder is a cumulative FEATURE SET, and a model has a recall
only for the feature sets that were trained: (a)-(e) and the audit-only
(b) + sensitive. All 24 orders of the four groups after credits are walked. An
order is reported as evaluated only if no model's walk needs the recall of an
untrained set; otherwise it is reported as "needs <set> trained" — never filled
in. Under the endline rules the sensitive group carries no price of its own (its
"never" only records that the set above LMS is unreachable), so reaching it is
not evaluable either. Orders that stop at the same place share a row.

    python src/selection_robustness.py
Writes results/descriptive/selection_robustness.tex
"""
from __future__ import annotations

import contextlib
import io
import itertools
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from paths.paths import RESULTS_PATH
from src import disclosure
from src.config import (AUDIT_ONLY_CONFIGS, COURSE_CREDITS, COURSE_PERFORMANCE,
                        DATA_CONFIGS, ENROLMENT, LMS, SELECTION_YEAR, SENSITIVE)
from src.display import feature_set_label, model_label
from src.selection_by_cohort import _row_hidden
from src.workshop_rules import (
    _RULE_BASED_MODELS, DEFAULT_RULES, LARGE_IMPROVEMENT, MODERATE_IMPROVEMENT, NEVER, NO_THRESHOLD,
    SMALL_IMPROVEMENT, SurveyRuleSet, data_ladder_walk, select_grid_config,
)

OUT_TEX = RESULTS_PATH / "descriptive" / "selection_robustness.tex"

# ── Data groups ───────────────────────────────────────────────────────────────

GROUPS: dict[str, list[str]] = {
    "performance": COURSE_PERFORMANCE,
    "enrollment":  ENROLMENT,
    "lms":         LMS,
    "sensitive":   SENSITIVE,
}
GROUP_LABEL = {"performance": "Performance", "enrollment": "Enrollment",
               "lms": "LMS", "sensitive": "Sensitive"}
MAIN_ORDER = ("performance", "enrollment", "lms", "sensitive")
# The main ladder's rung that adds each group — where its threshold is stored.
GROUP_RUNG = dict(zip(MAIN_ORDER, ("b_credits_performance", "c_plus_enrolment",
                                   "d_plus_lms", "e_plus_sensitive")))
# Groups a rule set holds no price for. The endline "never" for sensitive only
# records that its rung sits above LMS (see ENDLINE_RULES.notes).
UNPRICED = {"endline": frozenset({"sensitive"})}

# ── Thresholds ────────────────────────────────────────────────────────────────

## The survey's answer scale, as recall gains, least to most demanding.
SCALE = [NO_THRESHOLD, SMALL_IMPROVEMENT, MODERATE_IMPROVEMENT, LARGE_IMPROVEMENT, NEVER]
SHIFTS_PP = (-5, -2, 2, 5)

## (kind, key, what the threshold prices). The sensitive rung is left out: the
## survey priced each attribute only on top of all non-sensitive data, so it
## holds no price for this rung on its own (UNPRICED).
PARAMETERS = [
    ("fs",   "b_credits_performance", "Adding performance data"),
    ("fs",   "c_plus_enrolment",      "Adding enrollment data"),
    ("fs",   "d_plus_lms",            "Adding LMS data"),
    ("expl", "high",     "FasterRisk over the credits policy"),
    ("expl", "moderate", "Logistic regression over FasterRisk"),
    ("expl", "low",      "XGBoost over logistic regression"),
    ("expl", "very_low", "Neural network over XGBoost"),
]


@dataclass
class Variant:
    family: str
    label: str
    model: Optional[str] = None
    feature_set: Optional[str] = None
    evaluable: bool = True
    note: str = ""
    fallback: bool = False


def _quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


# ── (b) Ladder order ──────────────────────────────────────────────────────────

def _config_for(features: list[str]) -> Optional[str]:
    """The trained, selectable config with exactly these features, if any."""
    for name, cols in DATA_CONFIGS.items():
        if name not in AUDIT_ONLY_CONFIGS and set(cols) == set(features):
            return name
    return None


def ladder_rules(order: tuple[str, ...], rules: SurveyRuleSet = DEFAULT_RULES
                 ) -> tuple[SurveyRuleSet, dict[str, str]]:
    """
    `rules` with the data groups added in `order`. A rung that adds a group the
    rules hold no price for, or whose feature set was not trained, gets a
    placeholder name, returned in the second value with the reason. A group the
    rules mark never acceptable truncates the ladder as usual, so its rung needs
    no recall.
    """
    unpriced = UNPRICED.get(rules.name, frozenset())
    rungs, gains, unavailable = ["a_credits"], {}, {}
    features = list(COURSE_CREDITS)
    for i, group in enumerate(order):
        features = features + GROUPS[group]
        added = "+".join(GROUP_LABEL[g] for g in order[: i + 1])
        name = _config_for(features)
        gain = rules.fs_gain_required.get(GROUP_RUNG[group], NEVER)
        if group in unpriced:
            name, gain = f"unavailable:{added}", 0.0
            unavailable[name] = f"the {rules.name} survey priced {GROUP_LABEL[group]} only on top of all non-sensitive data"
        elif gain is not NEVER and name is None:
            name = f"unavailable:{added}"
            unavailable[name] = f"Credits+{added} was not trained"
        rungs.append(name or f"never:{added}")
        gains[rungs[-1]] = gain
    variant = replace(rules, name=f"{rules.name}_order", fs_order=tuple(rungs),
                      fs_gain_required=gains)
    return variant, unavailable


def run_ladder_order(results_df: pd.DataFrame, year: int = SELECTION_YEAR,
                     rules: SurveyRuleSet = DEFAULT_RULES) -> list[Variant]:
    """
    Every order of the four groups, collapsed to one row per distinct stop: the
    groups some model's walk actually judged. Orders that agree on those give
    the same selection whatever comes after.
    """
    rows: dict[tuple, Variant] = {}
    counts: dict[tuple, int] = {}
    for order in itertools.permutations(MAIN_ORDER):
        variant, unavailable = ladder_rules(order, rules)
        ladder = [fs for fs, _, _ in variant.fs_ladder]
        ml_models = [m for _, ms in variant.expl_groups for m in ms
                     if m != "credits_policy" and m not in variant.blocked_models]
        ## A model whose walk accepted k rungs past the entry was judged on the
        ## (k+1)-th group — or stopped at a "never", which is the same group.
        judged, blocker = 0, None
        for m in ml_models:
            k = len(data_ladder_walk(results_df, m, year, variant)[0]) - 1
            judged = max(judged, min(k + 1, len(order)))
            if k + 1 < len(ladder) and ladder[k + 1] in unavailable:
                blocker = ladder[k + 1]
        key = order[:judged]
        never = len(ladder) - 1 < len(order) and judged == len(ladder)
        names = [GROUP_LABEL[g] for g in key]
        if never:
            names[-1] += " (never)"
        counts[key] = counts.get(key, 0) + 1
        if key in rows:
            continue
        label = " → ".join(["Credits"] + names)
        if blocker:
            rows[key] = Variant("(b) Ladder order", label, evaluable=False,
                                note=unavailable[blocker])
        else:
            model, fs, _ = _quiet(select_grid_config, results_df, year, rules=variant)
            rows[key] = Variant("(b) Ladder order", label, model, fs)
    for key, v in rows.items():
        v.label += f" ({counts[key]} of 24)"
        if key == MAIN_ORDER[:len(key)]:
            v.label += ", the paper's order"
    return list(rows.values())


# ── Threshold shifts ──────────────────────────────────────────────────────────

def _gain_label(v) -> str:
    """A threshold as printed: "never", or its recall gain in pp (may be negative)."""
    if v is NEVER or (isinstance(v, float) and v != v):
        return "never"
    return f"{round(v * 100, 6):g} pp"


def with_threshold(rules: SurveyRuleSet, kind: str, key: str, value) -> SurveyRuleSet:
    """`rules` with one threshold replaced."""
    if kind == "fs":
        return replace(rules, fs_gain_required={**rules.fs_gain_required, key: value})
    return replace(rules, expl_gain_required={**rules.expl_gain_required, key: value})


def _select(results_df, year, rules) -> tuple[str, str, bool]:
    model, fs, grid = _quiet(select_grid_config, results_df, year, rules=rules)
    return model, fs, not bool(grid["permissible"].any())


def run_threshold_shifts(results_df: pd.DataFrame, year: int = SELECTION_YEAR,
                         rules: SurveyRuleSet = DEFAULT_RULES) -> pd.DataFrame:
    """
    One row per (rule, shift): the selection when that one threshold moves.
    `value` is None where the shift does not apply (a pp shift of "never", a
    level beyond either end of the scale).
    """
    main_model, main_fs, _ = _select(results_df, year, rules)
    main_recall = _recall(results_df, main_model, main_fs, year)
    main_hidden = _row_hidden(results_df, main_model, main_fs, year)
    rows = []
    for kind, key, what in PARAMETERS:
        paper = (rules.fs_gain_required if kind == "fs" else rules.expl_gain_required).get(key, NEVER)
        level = SCALE.index(paper)
        ## (label, applies, value). NEVER is None, so whether a shift applies is
        ## carried separately from its value.
        shifts = [(f"{d:+d} pp", paper is not NEVER,
                   None if paper is NEVER else round(paper + d / 100, 10))
                  for d in SHIFTS_PP]
        shifts += [("one level looser", level > 0, SCALE[level - 1] if level > 0 else None),
                   ("one level stricter", level + 1 < len(SCALE),
                    SCALE[level + 1] if level + 1 < len(SCALE) else None)]
        for shift, applies, value in shifts:
            row = {"kind": kind, "key": key, "rule": what, "paper": paper,
                   "shift": shift, "value": value, "applies": applies}
            if applies:
                model, fs, fallback = _select(results_df, year,
                                              with_threshold(rules, kind, key, value))
                row.update(model=model, feature_set=fs, fallback=fallback,
                           same=(model, fs) == (main_model, main_fs),
                           delta_pp=100 * (_recall(results_df, model, fs, year) - main_recall),
                           delta_hidden=main_hidden or _row_hidden(results_df, model, fs, year))
            rows.append(row)
    return pd.DataFrame(rows)


def _recall(results_df, model, fs, year) -> float:
    m = (results_df["model"] == model) & (results_df["test_year"] == year)
    if model != "credits_policy":
        m &= results_df["feature_set"] == fs
    r = results_df[m]
    return float(r["enr_recall"].iloc[0]) if len(r) else float("nan")


# ── Table ─────────────────────────────────────────────────────────────────────

def _short(model: str, fs: str) -> str:
    if model in _RULE_BASED_MODELS:
        return model_label(model)   # retained when nothing passes; has no feature set
    return f"{model_label(model)} {feature_set_label(fs).split()[0]}"


def _shift_cell(r) -> str:
    if not r["applies"]:
        return "n/a"
    value = _gain_label(r["value"]) if r["shift"].startswith("one level") else None
    if r["same"]:
        out = "same"
    else:
        d = disclosure.SUPPRESSED_TEX if r["delta_hidden"] else f"{r['delta_pp']:+.1f}"
        out = f"{_short(r['model'], r['feature_set'])}{' (fallback)' if r['fallback'] else ''}, {d}"
    return f"{value}: {out}" if value else out


def build_table(shifts: pd.DataFrame, ladder: list[Variant], main: tuple[str, str]) -> str:
    cols = [f"{s:+d} pp" for s in SHIFTS_PP] + ["one level looser", "one level stricter"]
    head = " & ".join([r"Recall gain required for", "Paper"]
                      + [c.replace("one level ", "One level\\\\ ").join(["\\shortstack{", "}"])
                         if c.startswith("one") else c.replace("-", "$-$") for c in cols])
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\footnotesize",
        rf"\textit{{Main selection: {_short(*main)}.}}",
        r"\par\medskip",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{ll" + "c" * len(cols) + "}",
        r"\toprule",
        head + r" \\",
        r"\midrule",
    ]
    for (kind, key), grp in shifts.groupby(["kind", "key"], sort=False):
        by_shift = {r["shift"]: r for _, r in grp.iterrows()}
        cells = [grp.iloc[0]["rule"], _gain_label(grp.iloc[0]["paper"])]
        cells += [_shift_cell(by_shift[c]) for c in cols]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\par\bigskip",
              r"\begin{tabular}{ll}", r"\toprule",
              r"Order in which data groups are added & Selected \\", r"\midrule"]
    for v in ladder:
        sel = (f"{_short(v.model, v.feature_set)}"
               + ("" if (v.model, v.feature_set) != main else " (same)")
               if v.evaluable else f"not evaluable: {v.note}")
        lines.append(f"{v.label} & {sel} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{\textbf{How much the workshop-grounded selection depends on the rule "
        r"thresholds.} Each variant moves one endline threshold and re-runs the selection "
        r"of Section~\ref{sec:model_selection_algorithm} on the " + str(SELECTION_YEAR)
        + r" cohort. Top: each threshold shifted by a fixed number of percentage points, or "
        r"one level along the survey scale (no worsening, small, moderate, large, never); a "
        r"cell gives the configuration then selected and the change in its enrollment-level "
        r"recall against the main selection, in pp, or ``same''. A negative threshold accepts "
        r"a step even if recall falls by up to that much; n/a marks a shift that does not "
        r"apply to a ``never'' rule. Bottom: the order in which the data groups are added to "
        r"the cumulative ladder; orders that stop at the same point share a row, and an "
        r"order is not evaluable when it would need the recall of a feature set that was not "
        r"trained, or a price the endline survey did not set.}",
        r"\label{tab:selection_robustness}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main() -> tuple[pd.DataFrame, list[Variant]]:
    results_df = pd.read_csv(RESULTS_PATH / "evaluation_results.csv")
    main_pick = _quiet(select_grid_config, results_df, SELECTION_YEAR, rules=DEFAULT_RULES)[:2]
    shifts = run_threshold_shifts(results_df)
    ladder = run_ladder_order(results_df)
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text(build_table(shifts, ladder, main_pick))
    for (_, _), g in shifts.groupby(["kind", "key"], sort=False):
        print(f"  {g.iloc[0]['rule']:<48} " + " | ".join(
            _shift_cell(r) for _, r in g.iterrows()))
    for v in ladder:
        print(f"  {v.label:<60} " + (f"{v.model} + {v.feature_set}" if v.evaluable else f"— {v.note}"))
    print(f"Saved → {OUT_TEX}")
    return shifts, ladder


if __name__ == "__main__":
    main()
