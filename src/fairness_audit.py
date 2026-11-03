"""
Fairness snapshot for the SI fairness table.

Per-subgroup metrics at one test year, and their differences with 95%
bootstrap CIs (n=1000 resamples), for the three models reported in the main
text: the credits policy, the Participatory model and Max ML (see
_snapshot_models()). The Participatory model is resolved via
workshop_rules.participatory_config(), not hardcoded. Resampling is
clustered by person_id where available, since one person can hold multiple
enrollments — see `_resample()`.

Subgroups:
  binary: female, danish_nationality, older_for_cohort
  multi:  nationality (DK/Nordic/EU_EEA_CH/Other)
          unknown citizenship (unknown codes → NaN) is dropped from the nationality axis

Output → results/fairness/fairness_snapshot.csv

Usage:
    python src/fairness_audit.py [--year 2021] [--n-boot 1000]
"""

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from paths.paths import RESULTS_PATH
from src import paper_style
from src import disclosure
from src.config import (
    DANISH_CATEGORY,
    MAIN_TEST_YEAR,
    MAX_ML_FS,
    MAX_ML_MODEL,
    MIN_CELL_SIZE,
    RANDOM_SEED,
)
from src.data import (derive_danish_nationality)
from src.metrics import percentile_ci

_FLEXIBLE_CANDIDATES = ["xgboost", "mlp"]

# ── style ─────────────────────────────────────────────────────────────────────
#
# This runs at IMPORT, so whatever is drawing when the module is first imported
# gets the style — harmless now that every module applies the same one
# (src/paper_style.py). It used to carry its own font list and demoted the
# headline figure to Arial when plotting.py imported it.

## The paper style, shared with every figure (src/paper_style.py).
paper_style.apply_style()

_COLORS = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#D55E00", "#CC79A7", "#0072B2"]

# ── constants ─────────────────────────────────────────────────────────────────

_METRICS = ["auc", "recall", "precision", "tnr", "flag_rate"]
_METRIC_LABELS = {
    "auc":       "AUC",
    "recall":    "Recall",
    "precision": "Precision",
    "tnr":       "TNR",
    "flag_rate": "Flag rate",
}

# binary subgroups: col → (reference_val, reference_label, comparison_val, comparison_label)
_BINARY_SUBGROUPS = {
    "female":             (0, "Male",   1, "Female"),
    "danish_nationality": (1, "Danish", 0, "Non-Danish"),
    ## Age relative to the cohort, not degree level. Master's credit features
    ## may be incomplete, so a BA/MA gap could mix bias with measurement error.
    ## `undergraduate` remains a model feature and a descriptive row, not an
    ## axis. See src/data.py::add_age_group_within_level.
    "older_for_cohort":   (0, "Younger", 1, "Older"),
}

# multi-category subgroups: col → (reference_category, [other_categories_in_order])
# nationality levels follow the registry recode: DK < Nordic < EU_EEA_CH < Other.
# The old ("Danish", [... "EU" ...]) spelling matches nothing after the recode.
# Unknown citizenship (unknown codes) is NaN and is dropped, not treated as a level.
_MULTI_SUBGROUPS = {
    "nationality": (DANISH_CATEGORY, ["Nordic", "EU_EEA_CH", "Other"]),
}

_CREDITS_POLICY = ("credits_policy", "ects_behind_ge_60")
_MAX_ML         = (MAX_ML_MODEL, MAX_ML_FS)


def _resolve_participatory_model() -> tuple[str, str]:
    """
    (model, feature_set) actually selected as the Participatory model, read
    via workshop_rules.participatory_config() — the one reader of the saved selection.

    Unlike Max ML (always MAX_ML_MODEL/MAX_ML_FS by design), the Participatory
    model is the *output* of the workshop-grounded selection rules run against
    whichever evaluation results are currently loaded, so it can differ
    between runs. Never hardcode it; always resolve it from the saved
    selection config.
    """
    from src.workshop_rules import CONFIG_FILE, participatory_config
    model, feature_set = participatory_config(RESULTS_PATH)
    if model and feature_set:
        return model, feature_set
    raise FileNotFoundError(
        f"{CONFIG_FILE} not found in {RESULTS_PATH / 'models'} — run "
        "src/workshop_evaluation.py first so the Participatory model selection exists."
    )


def _snapshot_models() -> list[tuple[str, str]]:
    """
    The three models compared in the main text and in the SI fairness tables: credits
    policy baseline, the currently-selected Participatory model, and Max ML.
    Resolved fresh each call (not a module-level constant) since the
    Participatory model depends on the workshop selection config on disk.
    """
    ## dict.fromkeys: a retained credits policy is both of the first two.
    return list(dict.fromkeys([_CREDITS_POLICY, _resolve_participatory_model(), _MAX_ML]))

_FAIRNESS_DIR = RESULTS_PATH / "fairness"
_PLOT_DIR     = RESULTS_PATH / "plots" / "fairness"

def _hide(metrics: dict, hidden: dict) -> dict:
    """Blank every metric that rests on fewer than MIN_CELL_SIZE students; keep the flags."""
    for m in _METRICS:
        metrics[f"hidden_{m}"] = bool(hidden[m])
        if hidden[m]:
            metrics[m] = np.nan
    return metrics


def _raw_metrics(enr: pd.DataFrame, verbose: bool = False) -> dict:
    """
    AUC, recall, precision, TNR, flag_rate for a single group slice, WITHOUT
    disclosure control, plus the counts it rests on (_n, _tp, _fp, _fn).

    For bootstrap draws only, which are never printed. Suppressing inside a draw
    discards exactly the draws whose counts swing low — the tails — and the
    interval taken over the survivors comes out too narrow. Anything published
    goes through _compute_metrics.
    """
    n = len(enr)
    if n == 0:
        return {**{m: np.nan for m in _METRICS}, "_n": 0, "_tp": 0, "_fp": 0, "_fn": 0}
    do = enr["dropped_out"]
    fl = enr["ever_flagged"]
    tp = int(((do == 1) & (fl == 1)).sum())
    fp = int(((do == 0) & (fl == 1)).sum())
    fn = int(((do == 1) & (fl == 0)).sum())
    tn = int(((do == 0) & (fl == 0)).sum())
    recall    = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    tnr       = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    flag_rate = (tp + fp) / n

    if "max_prob" not in enr.columns:
        if verbose:
            print("_compute_metrics: no max_prob column — AUC is undefined for this frame.")
        auc = np.nan
    else:
        valid = enr["max_prob"].notna()
        n_dropped = int((~valid).sum())
        if n_dropped and verbose:
            print(f"_compute_metrics: dropping {n_dropped} row(s) with NaN max_prob before AUC.")
        do_valid = do[valid]
        if len(do_valid) < 2 or len(np.unique(do_valid)) < 2:
            auc = np.nan
        else:
            auc = float(roc_auc_score(do_valid, enr.loc[valid, "max_prob"]))
    return {"auc": auc, "recall": recall, "precision": precision, "tnr": tnr,
            "flag_rate": flag_rate, "_n": n, "_tp": tp, "_fp": fp, "_fn": fn}


def _compute_metrics(enr: pd.DataFrame) -> dict:
    """
    AUC, recall, precision, TNR, flag_rate for a single group slice.

    Disclosure control happens here, so every published fairness number — CSVs,
    tables, figures — inherits it: a metric resting on 1..MIN_CELL_SIZE-1
    students comes back NaN with hidden_<metric> = True. recall/precision/TNR
    rest on all four confusion cells (with the group's size and dropout rate they
    pin each other down), flag_rate on flagged vs not, AUC on dropouts vs
    non-dropouts. See src/disclosure.py. Bootstrap draws use _raw_metrics.
    """
    if len(enr) < MIN_CELL_SIZE:
        small = len(enr) > 0
        return _hide({m: np.nan for m in _METRICS}, {m: small for m in _METRICS})
    raw = _raw_metrics(enr, verbose=True)
    tp, fp, fn = raw["_tp"], raw["_fp"], raw["_fn"]
    h = disclosure.operating_point_hidden(raw["_n"], tp + fn, tp + fp, tp)
    hidden = {"recall": h["recall"], "precision": h["precision"], "tnr": h["recall"],
              "flag_rate": h["flag_rate"], "auc": h["auc"]}
    return _hide({m: raw[m] for m in _METRICS}, hidden)


def _fmt_group(val) -> str:
    """
    Render a subgroup value for a difference label.

    Binary subgroup columns carry NaN for missing (unknown citizenship), which
    forces them to float dtype — so a plain f-string yields "0.0 − 1" and misses
    _BINARY_LABEL_MAP, whose keys are written "0 − 1". Integral floats print as
    integers here so the lookup keeps working.
    """
    if isinstance(val, float) and float(val).is_integer():
        return str(int(val))
    return str(val)


_METRICS_FRAME_COLUMNS = _METRICS + ["group", "n", "is_difference"]
_CI_FRAME_COLUMNS      = [f"{m}_{bound}" for m in _METRICS for bound in ("lo", "hi")]
_DIFF_ROW_COLUMNS      = ["subgroup", "comparison", "metric", "value", "ci_lo", "ci_hi"]


def _empty_metrics_frame() -> pd.DataFrame:
    """
    `subgroup_metrics` output shape with no rows. Typed, not just named: an
    object-dtype `is_difference` makes `m_df[m_df["is_difference"]]` return a
    0×0 frame that has lost its columns.
    """
    dtypes = {c: "float64" for c in _METRICS + ["n"]}
    dtypes.update({"group": "object", "is_difference": "bool"})
    return pd.DataFrame({c: pd.Series(dtype=dtypes[c]) for c in _METRICS_FRAME_COLUMNS})


def _empty_ci_frame() -> pd.DataFrame:
    """`bootstrap_differences` output shape with no rows, index named `comparison`."""
    return pd.DataFrame(columns=_CI_FRAME_COLUMNS,
                        index=pd.Index([], name="comparison"))


def _diff_rows_frame(rows: list[dict], extra_cols: list[str]) -> pd.DataFrame:
    """
    Long-format difference rows, carrying the expected columns even when `rows`
    is empty. An analysis that produced nothing usable must still hand the
    plotters a frame they can filter on `comparison`, not a column-less one.
    """
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(columns=_DIFF_ROW_COLUMNS + extra_cols)


def subgroup_metrics(
    enr: pd.DataFrame,
    subgroup_col: str,
    reference_val,
) -> pd.DataFrame:
    """
    Per-group metrics and raw differences (comparison − reference).
    Returns one row per group plus one difference row per non-reference group.
    An empty slice (or one where the subgroup column is all-NaN) returns an
    empty frame with the full column set rather than a column-less one.
    """
    if subgroup_col not in enr.columns:
        return _empty_metrics_frame()

    groups = sorted(enr[subgroup_col].dropna().unique(), key=str)
    if not groups:
        return _empty_metrics_frame()
    rows = []
    for g in groups:
        m = _compute_metrics(enr[enr[subgroup_col] == g])
        m["group"] = g
        m["n"] = int((enr[subgroup_col] == g).sum())
        m["is_difference"] = False
        rows.append(m)

    result  = pd.DataFrame(rows)
    ref_m   = _compute_metrics(enr[enr[subgroup_col] == reference_val])
    diff_rows = []
    for _, row in result.iterrows():
        if row["group"] == reference_val:
            continue
        dr = {"group": f"{_fmt_group(row['group'])} − {_fmt_group(reference_val)}",
              "n": np.nan, "is_difference": True}
        for m in _METRICS:
            dr[m] = row[m] - ref_m[m]
            # a difference is hidden when either side is
            dr[f"hidden_{m}"] = bool(row[f"hidden_{m}"]) or bool(ref_m[f"hidden_{m}"])
        diff_rows.append(dr)

    if not diff_rows:
        return result
    return pd.concat([result, pd.DataFrame(diff_rows)], ignore_index=True)


def _resample(
    df: pd.DataFrame, rng: np.random.Generator, cluster_col: str = "person_id"
) -> pd.DataFrame:
    """
    One bootstrap resample of `df`, same size as `df`.

    If `cluster_col` is present, resamples whole clusters (e.g. all enrollments
    belonging to the same person) with replacement rather than individual rows.
    One person can have multiple enrollments; plain row resampling would treat
    them as independent, understating CI width.
    Falls back to plain row resampling when no cluster column is available.
    """
    if cluster_col in df.columns and df[cluster_col].notna().any():
        groups     = df.groupby(cluster_col, sort=False)
        cluster_ids = list(groups.groups.keys())
        sampled_ids = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        return pd.concat([groups.get_group(cid) for cid in sampled_ids],
                         ignore_index=True)
    idx = rng.integers(0, len(df), size=len(df))
    return df.iloc[idx].reset_index(drop=True)


def bootstrap_differences(
    enr: pd.DataFrame,
    subgroup_col: str,
    reference_val,
    n_boot: int = 1000,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Bootstrap 95% CIs for all metric differences (comparison − reference).
    Resamples by person_id cluster when available (see `_resample`), otherwise
    by individual enrollment row.
    Returns DataFrame indexed by comparison label with {metric}_lo / {metric}_hi columns.

    Two degenerate cases are distinguished:
      * no comparison group at all (the slice holds only the reference group, or
        no group) → empty frame with the expected columns. There is nothing to
        bootstrap; callers read this as "insufficient data" for the axis.
      * a comparison group exists but the metric is undefined in more than
        MAX_UNDEFINED_DRAWS of the draws (e.g. single-class draws) → NaN bounds,
        so the point estimate still gets a row with an undefined CI.

    Draws are computed WITHOUT disclosure control (_raw_metrics) and every draw
    counts: suppressing inside a draw dropped the draws whose counts swung low,
    which made intervals for small groups too narrow. The interval is published
    only next to a published point estimate, which rests on the same counts.
    """
    rng     = np.random.default_rng(seed)
    groups  = (sorted(enr[subgroup_col].dropna().unique(), key=str)
               if subgroup_col in enr.columns else [])
    non_ref = [g for g in groups if g != reference_val]
    if not non_ref:
        # a group can be empty in a period: `rows` below would be empty and set_index("comparison")
        # would raise KeyError on a column-less frame.
        return _empty_ci_frame()

    boot: dict[object, dict[str, list]] = {g: {m: [] for m in _METRICS} for g in non_ref}
    for _ in range(n_boot):
        s     = _resample(enr, rng)
        ref_m = _raw_metrics(s[s[subgroup_col] == reference_val])
        for g in non_ref:
            g_m = _raw_metrics(s[s[subgroup_col] == g])
            for m in _METRICS:
                boot[g][m].append(g_m[m] - ref_m[m])

    rows = []
    for g in non_ref:
        # Same label construction as subgroup_metrics — _fmt_group keeps integral
        # floats (danish_nationality carries NaN, so the column is float) printing
        # as "0 − 1", which is how the caller looks the row up.
        row = {"comparison": f"{_fmt_group(g)} − {_fmt_group(reference_val)}"}
        for m in _METRICS:
            row[f"{m}_lo"], row[f"{m}_hi"] = percentile_ci(boot[g][m])
        rows.append(row)
    if not rows:
        return _empty_ci_frame()
    return pd.DataFrame(rows).set_index("comparison")


_BINARY_LABEL_MAP = {
    col: {f"{comp_val} − {ref_val}": f"{comp_label} − {ref_label}"}
    for col, (ref_val, ref_label, comp_val, comp_label) in _BINARY_SUBGROUPS.items()
}


def _expected_comparisons(col: str) -> list[str]:
    """
    Comparison labels a subgroup axis yields when every group is populated, in
    the display form used by `_all_subgroup_diffs`. Used to fill in explicit
    "insufficient data" rows for comparisons a particular slice cannot form.
    """
    if col in _BINARY_SUBGROUPS:
        _, ref_label, _, comp_label = _BINARY_SUBGROUPS[col]
        return [f"{comp_label} − {ref_label}"]
    ref_cat, others = _MULTI_SUBGROUPS[col]
    return [f"{o} − {ref_cat}" for o in others]


def _all_subgroup_diffs(enr: pd.DataFrame, n_boot: int) -> list[dict]:
    """
    Run subgroup_metrics + bootstrap_differences for every subgroup; return rows.

    A comparison the slice cannot form — the group is absent, or below
    MIN_CELL_SIZE so every metric came back NaN — still gets one row per metric
    with NaN value and NaN CI. Insufficient data is recorded, not dropped, so
    the period/year axis stays aligned across slices and the CSV says so
    explicitly. An axis whose column is missing entirely is skipped instead:
    that variable was never measured here, which is not the same claim.
    """
    enr  = derive_danish_nationality(enr)
    rows = []

    for col, (ref_val, *_) in {**_BINARY_SUBGROUPS, **_MULTI_SUBGROUPS}.items():
        if col not in enr.columns or enr[col].isna().all():
            continue
        # Drop enrollments with no value on this axis before both the point
        # estimate and the bootstrap. For nationality that means unknown
        # citizenship (unknown codes); groupby would drop them from the point
        # estimate anyway, but they would still be resampled into the bootstrap
        # denominator, so the CI would be built on a different n than the
        # estimate it brackets.
        sub = enr[enr[col].notna()]

        m_df  = subgroup_metrics(sub, col, ref_val)
        ci_df = bootstrap_differences(sub, col, ref_val, n_boot=n_boot)
        diff  = m_df[m_df["is_difference"]]

        seen = set()
        for _, row in diff.iterrows():
            comp         = str(row["group"])
            display_comp = _BINARY_LABEL_MAP.get(col, {}).get(comp, comp)
            ci_row = ci_df.loc[comp] if comp in ci_df.index else None
            seen.add(display_comp)
            for met in _METRICS:
                # a hidden point takes its CI with it — the interval would locate it
                hidden = bool(row.get(f"hidden_{met}", False))
                rows.append({
                    "subgroup":   col,
                    "comparison": display_comp,
                    "metric":     met,
                    "value":      row[met],
                    "ci_lo":      ci_row[f"{met}_lo"] if ci_row is not None and not hidden else np.nan,
                    "ci_hi":      ci_row[f"{met}_hi"] if ci_row is not None and not hidden else np.nan,
                    "hidden":     hidden,
                })

        for missing in _expected_comparisons(col):
            if missing in seen:
                continue
            for met in _METRICS:
                rows.append({
                    "subgroup":   col,
                    "comparison": missing,
                    "metric":     met,
                    "value":      np.nan,
                    "ci_lo":      np.nan,
                    "ci_hi":      np.nan,
                    "hidden":     False,
                })
    return rows


# ── analysis 1: snapshot ───────────────────────────────────────────────────────

def run_snapshot(
    preds_df: pd.DataFrame,
    test_year: int         = MAIN_TEST_YEAR,
    models: list[tuple[str, str]] | None = None,
    n_boot: int            = 1000,
    save: bool             = True,
) -> pd.DataFrame:
    """
    Fairness snapshot: per-subgroup metric differences at one test year, for
    each (model, feature_set) pair in `models`. Defaults to _snapshot_models()
    (credits policy, Participatory model, Max ML) — the three models compared
    in the main text and in the SI fairness tables — rather than a single arbitrary model.
    Saves fairness_snapshot.csv → results/fairness/.
    """
    if models is None:
        models = _snapshot_models()

    all_rows = []
    for model, feature_set in models:
        subset = preds_df[
            (preds_df["test_year"]   == test_year) &
            (preds_df["model"]       == model) &
            (preds_df["feature_set"] == feature_set)
        ].copy()
        if len(subset) == 0:
            print(f"  Skipping {model} | {feature_set} — no rows for test_year={test_year}.")
            continue

        rows = _all_subgroup_diffs(subset, n_boot=n_boot)
        for r in rows:
            r["test_year"]   = test_year
            r["model"]       = model
            r["feature_set"] = feature_set
        all_rows.extend(rows)

    result = _diff_rows_frame(all_rows, ["test_year", "model", "feature_set"])
    if result.empty:
        print("  No usable subgroup comparisons — writing an empty snapshot.")
    if save:
        _FAIRNESS_DIR.mkdir(parents=True, exist_ok=True)
        result.to_csv(_FAIRNESS_DIR / "fairness_snapshot.csv", index=False)
        print(f"Saved → {_FAIRNESS_DIR / 'fairness_snapshot.csv'}")
    return result


## Pairwise intersections of the binary axes, for the SI table under the
## snapshot. Pairs, not the three-way split: eight cells cut the Non-Danish
## group in four, and most would rest on too few dropouts to print.
_INTERSECTIONAL_PAIRS = [
    ("female",             "danish_nationality"),
    ("female",             "older_for_cohort"),
    ("danish_nationality", "older_for_cohort"),
]


def intersectional_key(a: str, b: str) -> str:
    """The `subgroup` value of one pair's rows in fairness_intersectional.csv."""
    return f"{a} x {b}"


def intersectional_cells(a: str, b: str) -> tuple[str, list[str]]:
    """
    (reference cell, comparison cells) of one pair, labelled "Female / Danish".
    The reference is both axes' own reference group; the comparisons are in a
    fixed order — first axis moved, second axis moved, both moved.
    """
    _, ref_a, _, comp_a = _BINARY_SUBGROUPS[a]
    _, ref_b, _, comp_b = _BINARY_SUBGROUPS[b]
    return (f"{ref_a} / {ref_b}",
            [f"{comp_a} / {ref_b}", f"{ref_a} / {comp_b}", f"{comp_a} / {comp_b}"])


def _cell_column(enr: pd.DataFrame, a: str, b: str) -> pd.Series:
    """Each enrollment's cell label for the pair; NaN when either axis is unknown."""
    def label(col):
        ref_val, ref_label, comp_val, comp_label = _BINARY_SUBGROUPS[col]
        return enr[col].map({ref_val: ref_label, comp_val: comp_label})
    la, lb = label(a), label(b)
    return (la + " / " + lb).where(la.notna() & lb.notna())


def run_intersectional_snapshot(
    preds_df: pd.DataFrame,
    test_year: int         = MAIN_TEST_YEAR,
    models: list[tuple[str, str]] | None = None,
    n_boot: int            = 1000,
    save: bool             = True,
) -> pd.DataFrame:
    """
    Recall (and the other snapshot metrics) by pairwise intersectional cell,
    each cell against the pair's reference cell, for the snapshot's three
    models, with each cell's size and dropout count (raw — suppressed by the
    table generator). Same rows, disclosure control and bootstrap as run_snapshot — the
    cell is just a four-level subgroup column. An axis column that is missing
    or all-NaN raises: the table built from this file must not lose a pair
    quietly. Saves fairness_intersectional.csv → results/fairness/.
    """
    if models is None:
        models = _snapshot_models()

    all_rows = []
    for model, feature_set in models:
        subset = preds_df[
            (preds_df["test_year"]   == test_year) &
            (preds_df["model"]       == model) &
            (preds_df["feature_set"] == feature_set)
        ].copy()
        if len(subset) == 0:
            raise ValueError(f"no predictions for {model} | {feature_set}, "
                             f"test_year={test_year}")
        subset = derive_danish_nationality(subset)
        for a, b in _INTERSECTIONAL_PAIRS:
            empty = [c for c in (a, b) if c not in subset or subset[c].isna().all()]
            if empty:
                raise ValueError(f"{model} | {feature_set}: no values for {empty} — "
                                 "re-run src/evaluation.py --merge")
            key = intersectional_key(a, b)
            ref_cell, comp_cells = intersectional_cells(a, b)
            subset[key] = _cell_column(subset, a, b)
            sub = subset[subset[key].notna()]

            m_df  = subgroup_metrics(sub, key, ref_cell)
            ci_df = bootstrap_differences(sub, key, ref_cell, n_boot=n_boot)
            diff  = m_df[m_df["is_difference"]].set_index("group")
            size  = {c: (int((sub[key] == c).sum()),
                         int(sub.loc[sub[key] == c, "dropped_out"].sum()))
                     for c in [ref_cell] + comp_cells}
            for cell in comp_cells:
                comp = f"{cell} − {ref_cell}"
                row = diff.loc[comp] if comp in diff.index else None
                ci  = ci_df.loc[comp] if comp in ci_df.index else None
                for met in _METRICS:
                    hidden = row is not None and bool(row[f"hidden_{met}"])
                    all_rows.append({
                        "subgroup":    key,
                        "comparison":  comp,
                        "metric":      met,
                        "value":       row[met] if row is not None else np.nan,
                        "ci_lo":       ci[f"{met}_lo"] if ci is not None and not hidden else np.nan,
                        "ci_hi":       ci[f"{met}_hi"] if ci is not None and not hidden else np.nan,
                        "hidden":      hidden,
                        ## Raw counts, for the table's size columns; the
                        ## generator applies disclosure control to them.
                        "n":             size[cell][0],
                        "n_dropout":     size[cell][1],
                        "ref_n":         size[ref_cell][0],
                        "ref_n_dropout": size[ref_cell][1],
                        "test_year":   test_year,
                        "model":       model,
                        "feature_set": feature_set,
                    })

    result = _diff_rows_frame(all_rows, ["test_year", "model", "feature_set"])
    if save:
        _FAIRNESS_DIR.mkdir(parents=True, exist_ok=True)
        result.to_csv(_FAIRNESS_DIR / "fairness_intersectional.csv", index=False)
        print(f"Saved → {_FAIRNESS_DIR / 'fairness_intersectional.csv'}")
    return result


# ── main entry point ───────────────────────────────────────────────────────────

def run_fairness_audit(
    test_year: int   = MAIN_TEST_YEAR,
    analysis: str    = "snapshot",
    n_boot: int      = 1000,
) -> None:
    enr_preds_path  = RESULTS_PATH / "enrollment_predictions.csv"

    if not enr_preds_path.exists():
        print(f"ERROR: {enr_preds_path} not found — run evaluation.py first.")
        return

    preds_df = pd.read_csv(enr_preds_path)
    print(f"Loaded {len(preds_df):,} enrollment predictions.")

    train_meta_path = RESULTS_PATH / "models" / "train_metadata.json"
    train_pos_rate  = 0.3
    if train_meta_path.exists():
        import json
        with open(train_meta_path) as fh:
            meta = json.load(fh)
        train_pos_rate = meta.get("train_pos_rate", train_pos_rate)

    if analysis in ("snapshot", "all"):
        snapshot_models = _snapshot_models()
        print(f"\n── Snapshot: {len(snapshot_models)} models | year {test_year} ──")
        run_snapshot(preds_df, test_year=test_year, n_boot=n_boot)
        print("\n── Intersectional snapshot ──")
        run_intersectional_snapshot(preds_df, test_year=test_year, n_boot=n_boot)

    print("\nFairness audit complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EWS fairness audit")
    ## Only the snapshot feeds the paper (its SI fairness table).
    parser.add_argument("--analysis",    default="snapshot", choices=["snapshot"])
    parser.add_argument("--year",        type=int, default=MAIN_TEST_YEAR)
    parser.add_argument("--n-boot",      type=int, default=1000, dest="n_boot")
    args = parser.parse_args()

    run_fairness_audit(
        test_year=args.year,
        analysis=args.analysis,
        n_boot=args.n_boot,
    )
