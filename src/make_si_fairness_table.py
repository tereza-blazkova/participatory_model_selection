"""
Generate LaTeX supplementary information (SI) tables for fairness metrics.

Fairness-over-years table: metric differences (comparison − reference)
  with 95% bootstrap CIs for all subgroup comparisons across test years 2019–2023.
  Source: results/fairness/fairness_over_years.csv

Snapshot table (one test year, per model × feature set): metric
  differences for all binary subgroups.
  Source: results/fairness/fairness_snapshot.csv

Run from project root:
    python src/make_si_fairness_table.py
Writes: results/si_fairness_tables.tex
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from paths.paths import RESULTS_PATH
from src import disclosure
from src.config import MAIN_TEST_YEAR, MAX_ML_FS, MAX_ML_MODEL
from src.display import MODEL_LABELS

_OUT = RESULTS_PATH / "si_fairness_tables.tex"

_METRICS_ORDER  = ["auc", "recall", "precision"]
_METRIC_DISPLAY = {
    "auc":       "Enr. ROC AUC",
    "recall":    "Recall",
    "precision": "Precision",
}

_CREDITS_POLICY_PAIR = ("credits_policy", "ects_behind_ge_60")
_MAX_ML_PAIR         = (MAX_ML_MODEL, MAX_ML_FS)

_MODEL_NAME_DISPLAY = {
    "credits_policy":      "Credits policy",
    "fasterrisk":          "FasterRisk",
    "logistic_regression": "logistic regression",
    "xgboost":             "XGBoost",
    "mlp":                 MODEL_LABELS["mlp"].lower(),   # mid-sentence, like the rest
}


def _model_display_name(model: str) -> str:
    return _MODEL_NAME_DISPLAY.get(model, model.replace("_", " "))


def _model_fs_label(model: str, feature_set: str) -> str:
    """
    Display label for a (model, feature_set) pair in the snapshot table.

    Max ML is always MAX_ML_MODEL/MAX_ML_FS by design (never selected by
    AUC — see src/config.py). The Participatory model, by contrast, is
    whatever the workshop selection rules picked for the data currently
    loaded — so it must never be
    matched against a hardcoded (model, feature_set) tuple. Instead, any
    pair that isn't credits policy or Max ML is labelled as the
    Participatory model, whichever model/feature_set it actually is.
    """
    pair = (model, feature_set)
    if pair == _CREDITS_POLICY_PAIR:
        return "Credits policy"
    if pair == _MAX_ML_PAIR:
        return f"Max ML ({_model_display_name(model)})"
    return f"Participatory model ({_model_display_name(model)})"

## Must stay in step with fairness_audit._BINARY_SUBGROUPS; the table matches
## rows by this label.
_BINARY_COMPARISONS = [
    "Female − Male",
    "Non-Danish − Danish",
    "Older − Younger",
]

## The three-category nationality breakdown, in NATIONALITY_LEVELS order,
## against Danish. Printed as a separate block under the binary rows.
_CATEGORY_COMPARISONS = [
    "Nordic − DK",
    "EU_EEA_CH − DK",
    "Other − DK",
]


def _fmt(val: float, ci_lo: float, ci_hi: float, hidden: bool = False) -> str:
    if hidden:
        return disclosure.SUPPRESSED_TEX
    if pd.isna(val):
        return "—"
    s = f"{val:+.3f}"
    if not (pd.isna(ci_lo) or pd.isna(ci_hi)):
        s += f" [{ci_lo:+.3f}, {ci_hi:+.3f}]"
    return s


## Nationality levels arrive as their config codes ("EU_EEA_CH − DK"). A raw
## code reads badly in a paper, and its underscores would be LaTeX subscripts.
## DK reads as
## "Danish", as in the binary "Non-Danish − Danish" rows of the same table.
_CATEGORY_DISPLAY = {
    "EU_EEA_CH": "EU/EEA/CH",
    "DK":        "Danish",
}


def _comparison_label(comp: str) -> str:
    """Display form of a "comparison − reference" label, level by level."""
    return " − ".join(_CATEGORY_DISPLAY.get(part, part) for part in comp.split(" − "))


def _latex_escape(s: str) -> str:
    ## "_" last-resort escaped, so a level with no display name still compiles.
    return (s.replace("_", r"\_").replace("−", r"$-$")
             .replace("≤", r"$\leq$").replace("+", r"$+$"))


def _complementary_hidden(df: pd.DataFrame) -> pd.DataFrame:
    """
    Within one model x axis x metric, several comparisons share a reference
    group (Nordic/EU/Other − DK). If exactly one of them is
    hidden, it could be recovered from the others and the overall value, so a
    second one is hidden with it.
    """
    df = df.copy()
    for _, idx in df.groupby(["model", "feature_set", "subgroup", "metric"]).groups.items():
        block = df.loc[idx]
        if len(block) >= 2 and int(block["hidden"].sum()) == 1:
            df.loc[block.index[~block["hidden"].to_numpy()][0], "hidden"] = True
    return df


def _require_all_axes(df: pd.DataFrame) -> None:
    """
    Credits policy, the Participatory model and Max ML must each have every
    binary comparison, with a value or a suppressed cell. An axis with no data
    would otherwise drop out of a block silently, because fairness_audit skips an
    all-NaN subgroup column, so refuse.
    """
    pairs = set(map(tuple, df[["model", "feature_set"]].drop_duplicates().values.tolist()))
    others = pairs - {_CREDITS_POLICY_PAIR, _MAX_ML_PAIR}
    problems = [f"no rows for {p}" for p in (_CREDITS_POLICY_PAIR, _MAX_ML_PAIR)
                if p not in pairs]
    if len(others) != 1:
        problems.append(f"expected one Participatory model, found {sorted(others)}"
                        + (" (none exists if the Credits policy was retained: this table "
                           "has no block for that case)" if not others else ""))
    for model, fs in sorted(pairs):
        sub = df[(df["model"] == model) & (df["feature_set"] == fs)]
        for comp in _BINARY_COMPARISONS:
            csub = sub[sub["comparison"] == comp]
            if not (csub["value"].notna() | csub["hidden"].astype(bool)).any():
                problems.append(f"{model}/{fs}: no estimate for {comp}")
    if problems:
        raise ValueError(
            "fairness_snapshot.csv cannot fill the SI fairness table:\n  "
            + "\n  ".join(problems)
            + "\nCheck that enrollment_predictions.csv carries the subgroup columns, "
              "filled in, for every model (src/evaluation.py --merge refreshes them), "
              "then re-run src/fairness_audit.py --analysis snapshot."
        )


def _build_snapshot_table(df: pd.DataFrame) -> str:
    """
    Snapshot table: fairness for one year, all subgroups.

    One shared tabular for all models, with a bold section-header row per
    model × feature_set so every section has identical column widths and
    font size. Rows = comparison, Columns = metric (ROC AUC, Recall, Precision).
    """
    if "hidden" not in df.columns:
        raise ValueError(
            "fairness_snapshot.csv predates disclosure control (no 'hidden' column); "
            "re-run src/fairness_audit.py before building the SI fairness table."
        )
    _require_all_axes(df)
    df = _complementary_hidden(df)

    lines: list[str] = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\small")

    available_pairs = set(map(tuple, df[["model", "feature_set"]].drop_duplicates().values.tolist()))
    # Fixed display order (credits policy, Participatory, Max ML) rather than
    # an incidental alphabetical sort on feature_set, which doesn't match the
    # order these models are discussed in the main text. The Participatory
    # model's (model, feature_set) isn't known in advance — it's whichever
    # pair remains once credits policy and Max ML are accounted for.
    model_fs_pairs = (
        ([_CREDITS_POLICY_PAIR] if _CREDITS_POLICY_PAIR in available_pairs else [])
        + [p for p in available_pairs if p not in (_CREDITS_POLICY_PAIR, _MAX_ML_PAIR)]
        + ([_MAX_ML_PAIR] if _MAX_ML_PAIR in available_pairs else [])
    )

    metric_header = " & ".join(
        r"\multicolumn{1}{c}{" + _METRIC_DISPLAY[m] + "}" for m in _METRICS_ORDER
    )
    n_metrics = len(_METRICS_ORDER)
    n_cols = n_metrics + 1

    # One shared tabular for all models, so every section has identical
    # column widths/font size instead of each being sized independently.
    lines.append(r"\begin{tabular}{l" + "r" * n_metrics + "}")
    lines.append(r"\toprule")
    lines.append(f"Subgroup & {metric_header} \\\\")

    for model, fs in model_fs_pairs:
        sub = df[(df["model"] == model) & (df["feature_set"] == fs)]
        if sub.empty:
            continue

        present = set(sub["comparison"].unique())
        binary = [c for c in _BINARY_COMPARISONS if c in present]
        categories = [c for c in _CATEGORY_COMPARISONS if c in present]
        unknown = present - set(binary) - set(categories)
        if unknown:
            raise ValueError(f"fairness_snapshot.csv has comparisons this table "
                             f"does not place: {sorted(unknown)}")
        label = _model_fs_label(model, fs)
        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{{n_cols}}}{{l}}{{\textbf{{{label}}}}} \\")
        lines.append(r"\midrule")

        ## The binary axes first — the scheme every other figure and table uses —
        ## then the three-category nationality breakdown as its own block, so it
        ## is never read as a fourth axis alongside them.
        block = [(c, False) for c in binary] + [(c, True) for c in categories]
        for i, (comp, is_category) in enumerate(block):
            if is_category and (i == 0 or not block[i - 1][1]):
                lines.append(rf"\multicolumn{{{n_cols}}}{{l}}{{\textit{{Nationality, "
                             r"three categories (reference: Danish)}} \\")
            csub = sub[sub["comparison"] == comp].set_index("metric")
            comp_tex = _latex_escape(_comparison_label(comp))
            cells: list[str] = [comp_tex]
            for metric in _METRICS_ORDER:
                if metric in csub.index:
                    row = csub.loc[metric]
                    cells.append(_fmt(row["value"], row["ci_lo"], row["ci_hi"], bool(row["hidden"])))
                else:
                    cells.append("—")
            lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append("")

    lines.append(
        r"\caption{\textbf{Fairness metric differences across subgroups.} "
        r"Metric gaps (comparison group $-$ reference group) with 95\% bootstrap CIs "
        r"for test year " + str(MAIN_TEST_YEAR) + r". "
        r"Positive values indicate the comparison group has a higher metric value. "
        r"Nationality is compared as Non-Danish $-$ Danish, as everywhere in the paper; "
        r"the three-category breakdown below it in each block is supplementary. "
        r"ROC AUC gaps are signed differences in enrollment-level ROC AUC (maximum "
        r"predicted probability per enrollment), not ABROCA.}"
    )
    lines.append(r"\label{tab:si_fairness_snapshot}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


_PAIR_TITLE = {
    "female":             "Sex",
    "danish_nationality": "nationality",
    "older_for_cohort":   "entry age",
}


def _ordered_pairs(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Credits policy, the Participatory model, Max ML — refusing a missing one."""
    pairs = set(map(tuple, df[["model", "feature_set"]].drop_duplicates().values.tolist()))
    others = sorted(pairs - {_CREDITS_POLICY_PAIR, _MAX_ML_PAIR})
    if _CREDITS_POLICY_PAIR not in pairs or _MAX_ML_PAIR not in pairs or len(others) != 1:
        raise ValueError("fairness_intersectional.csv must hold credits policy, one "
                         f"Participatory model and Max ML; found {sorted(pairs)}")
    return [_CREDITS_POLICY_PAIR, others[0], _MAX_ML_PAIR]


def _cell_sizes(df: pd.DataFrame, key: str, ref: str, cells: list[str]) -> list[tuple[str, str]]:
    """
    (N, dropouts) as printed for [ref] + cells of one pair. Every method scores
    the same enrollments, so the sizes are printed once; a disagreement raises.
    A cell whose dropouts or non-dropouts number 1..MIN_CELL_SIZE-1 is
    suppressed, and since the four cells add up to totals the paper prints
    elsewhere, a second cell is suppressed with a lone one.
    """
    rows = df[df["subgroup"] == key]
    sizes = []
    for i, cell in enumerate([ref] + cells):
        if i == 0:
            got = rows[["ref_n", "ref_n_dropout"]].drop_duplicates()
        else:
            got = rows.loc[rows["comparison"] == f"{cell} − {ref}",
                           ["n", "n_dropout"]].drop_duplicates()
        if len(got) != 1:
            raise ValueError(f"{key}: the methods disagree on the size of {cell} — "
                             "they must score the same enrollments")
        sizes.append(tuple(int(v) for v in got.iloc[0]))
    small = {i for i, (n, k) in enumerate(sizes) if disclosure.rate_is_small(k, n)}
    hidden = disclosure.complementary([n for n, _ in sizes], small)
    X = disclosure.SUPPRESSED_TEX
    return [(X, X) if i in hidden else (f"{_round10(n):,}", f"{_round10(k):,}")
            for i, (n, k) in enumerate(sizes)]


def _round10(v: int) -> int:
    return int(v / 10 + 0.5) * 10


def _build_intersectional_table(df: pd.DataFrame) -> str:
    """
    Recall difference by pairwise intersectional cell, each cell against its
    pair's reference cell; one column per method. Printed directly under the
    snapshot table, whose caption it leans on. Every pair x cell x method must
    carry a value or a suppressed cell, or the table is refused.
    """
    from src.fairness_audit import (
        _INTERSECTIONAL_PAIRS, intersectional_cells, intersectional_key,
    )
    df = _complementary_hidden(df[df["metric"] == "recall"])
    methods = _ordered_pairs(df)

    missing = []
    for model, fs in methods:
        for a, b in _INTERSECTIONAL_PAIRS:
            ref, cells = intersectional_cells(a, b)
            for cell in cells:
                r = df[(df["model"] == model) & (df["feature_set"] == fs)
                       & (df["subgroup"] == intersectional_key(a, b))
                       & (df["comparison"] == f"{cell} − {ref}")]
                if not (r["value"].notna() | r["hidden"].astype(bool)).any():
                    missing.append(f"{model}/{fs}: {cell} − {ref}")
    if missing:
        raise ValueError("fairness_intersectional.csv has no estimate for:\n  "
                         + "\n  ".join(missing)
                         + "\nRe-run src/fairness_audit.py --analysis snapshot.")

    def cell_tex(cell: str) -> str:
        return _latex_escape(cell.replace(" / ", ", "))

    part = _model_fs_label(*methods[1]).removeprefix("Participatory model ")
    maxml = _model_fs_label(*methods[2]).removeprefix("Max ML ")
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Cell & \multicolumn{1}{c}{$N$} & \multicolumn{1}{c}{Dropouts} & "
        r"\multicolumn{1}{c}{Credits policy} & "
        r"\multicolumn{1}{c}{Participatory model} & \multicolumn{1}{c}{Max ML} \\",
        rf" & & & & \multicolumn{{1}}{{c}}{{{part}}} & \multicolumn{{1}}{{c}}{{{maxml}}} \\",
    ]
    for a, b in _INTERSECTIONAL_PAIRS:
        ref, cells = intersectional_cells(a, b)
        key = intersectional_key(a, b)
        sizes = _cell_sizes(df, key, ref, cells)
        lines.append(r"\midrule")
        title = f"{_PAIR_TITLE[a]} $\\times$ {_PAIR_TITLE[b]}"
        lines.append(rf"\multicolumn{{6}}{{l}}{{\textit{{{title[0].upper() + title[1:]}}}}} \\")
        lines.append(" & ".join([cell_tex(ref) + " (reference)", *sizes[0]]
                                + [r"\multicolumn{1}{c}{--}"] * 3) + r" \\")
        for cell, (n, k) in zip(cells, sizes[1:]):
            out = [cell_tex(cell), n, k]
            for model, fs in methods:
                row = df[(df["model"] == model) & (df["feature_set"] == fs)
                         & (df["subgroup"] == key)
                         & (df["comparison"] == f"{cell} − {ref}")].iloc[0]
                out.append(_fmt(row["value"], row["ci_lo"], row["ci_hi"], bool(row["hidden"])))
            lines.append(" & ".join(out) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{\textbf{Recall differences by intersectional cell.} "
        r"Enrollment-level recall of each cell minus that of its pair's reference "
        r"cell, for test year " + str(MAIN_TEST_YEAR) + r" and the three methods of "
        r"Table~\ref{tab:si_fairness_snapshot}, with 95\% bootstrap CIs computed as "
        r"there. $N$ is the number of enrollments in the cell and Dropouts the number "
        r"of them that dropped out, both rounded to the nearest 10 so that no group "
        r"can be recovered by subtracting across tables; the methods score the same enrollments. "
        r"Each pair of axes uses the enrollments with a value on both; entry "
        r"age is relative to the median of the student's level and cohort. The "
        r"comparisons are descriptive and not corrected for multiple testing.}",
        r"\label{tab:si_fairness_intersectional}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def _build_over_years_full_table(df: pd.DataFrame) -> str:
    """
    Full fairness-over-years table, including multi-category subgroups,
    all metrics. One table per subgroup, rows = comparison × metric, cols = year.
    """
    years      = sorted(df["test_year"].unique())
    year_cols  = " & ".join(str(y) for y in years)
    n_cols     = len(years)
    all_comps  = sorted(df["comparison"].unique())

    lines: list[str] = []
    lines.append(r"\subsection*{Complete fairness-over-years (all subgroups)}")
    lines.append("")
    lines.append(
        r"Metric gaps for all subgroup comparisons including multi-category "
        r"subgroups (nationality) across test years."
    )
    lines.append("")

    for comp in all_comps:
        sub = df[df["comparison"] == comp]
        if sub.empty:
            continue
        comp_tex = _latex_escape(_comparison_label(comp))
        lines.append(r"\bigskip")
        lines.append(rf"\textbf{{{comp_tex}}}")
        lines.append("")
        lines.append(r"\begin{tabular}{l" + "r" * n_cols + "}")
        lines.append(r"\toprule")
        lines.append(f"Metric & {year_cols} \\\\")
        lines.append(r"\midrule")

        for metric in _METRICS_ORDER:
            msub = sub[sub["metric"] == metric].set_index("test_year")
            cells: list[str] = [_METRIC_DISPLAY[metric]]
            for y in years:
                if y in msub.index:
                    row = msub.loc[y]
                    cells.append(_fmt(row["value"], row["ci_lo"], row["ci_hi"]))
                else:
                    cells.append("—")
            lines.append(" & ".join(cells) + r" \\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append("")

    return "\n".join(lines)


def build_si_tables() -> str:
    lines: list[str] = []

    lines.append(r"% ── SI Fairness Tables — auto-generated by make_si_fairness_table.py")
    lines.append(r"% Requires: \usepackage{booktabs} in preamble.")
    lines.append("")

    over_years_path = RESULTS_PATH / "fairness" / "fairness_over_years.csv"
    snapshot_path   = RESULTS_PATH / "fairness" / "fairness_snapshot.csv"

    if not snapshot_path.exists():
        raise FileNotFoundError(f"{snapshot_path} not found — run "
                                "src/fairness_audit.py --analysis snapshot first.")
    lines.append(_build_snapshot_table(pd.read_csv(snapshot_path)))

    ## Directly under the snapshot, so the reader has the marginal gaps at hand.
    intersectional_path = RESULTS_PATH / "fairness" / "fairness_intersectional.csv"
    if not intersectional_path.exists():
        raise FileNotFoundError(f"{intersectional_path} not found — run "
                                "src/fairness_audit.py --analysis snapshot first.")
    lines.append("")
    lines.append(_build_intersectional_table(pd.read_csv(intersectional_path)))

    return "\n".join(lines)


if __name__ == "__main__":
    print(f"Generating SI fairness tables → {_OUT}")
    content = build_si_tables()
    _OUT.write_text(content, encoding="utf-8")
    print(f"Saved → {_OUT}")
