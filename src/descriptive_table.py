"""
Descriptive statistics tables for the EWS dataset.

Outputs LaTeX tables and a plain-text summary to results/descriptive/.

Run from project root:
    python src/descriptive_table.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from paths.paths import RESULTS_PATH
from src.config import (
    DANISH_CATEGORY, MAIN_TEST_YEAR, TARGET, TEST_YEARS, TRAIN_YEARS,
)
from src import disclosure
from src.data import load_analysis_df, load_pilot_snapshot

OUT_DIR = RESULTS_PATH / "descriptive"


# ── formatting helpers ─────────────────────────────────────────────────────────

def _fmt_int(n: int) -> str:
    return f"{n:,}"


def _fmt_pct(x: float) -> str:
    return "---" if pd.isna(x) else f"{x * 100:.1f}\\%"


def _group_cells(enr: pd.DataFrame, hide_n: bool = False, hide_rate: bool = False) -> tuple[str, str]:
    """
    (N, dropout rate) for one group of enrollments, under disclosure control:
    N below MIN_CELL_SIZE prints the marker; the rate is hidden when the group's dropouts or
    non-dropouts are fewer than MIN_CELL_SIZE (N x rate would give them back). hide_n / hide_rate
    force a cell hidden, for complementary suppression.
    """
    n, k = len(enr), int(enr["dropout"].sum())
    n_txt = disclosure.SUPPRESSED_TEX if (hide_n or disclosure.is_small(n)) else _fmt_int(n)
    r_txt = (disclosure.SUPPRESSED_TEX if (hide_rate or hide_n or disclosure.rate_is_small(k, n))
             else _fmt_pct(enr["dropout"].mean() if n else np.nan))
    return n_txt, r_txt


def _complementary(groups: list[pd.DataFrame]) -> tuple[set[int], set[int]]:
    """
    Groups that add up to a total the reader knows (a parent row, the cohort):
    which Ns and which rates to hide so a hidden one cannot be recovered by
    subtraction. Returns (hide_n, hide_rate) as sets of group indices.
    """
    ns = [len(g) for g in groups]
    ks = [int(g["dropout"].sum()) for g in groups]
    hide_n = disclosure.complementary(ns, {i for i, n in enumerate(ns) if disclosure.is_small(n)})
    small_rate = {i for i, (k, n) in enumerate(zip(ks, ns)) if disclosure.rate_is_small(k, n)} | hide_n
    hide_rate = disclosure.complementary(ks, small_rate) if small_rate else set()
    return hide_n, hide_rate


def _enr_level(subset: pd.DataFrame, extra_cols=None) -> pd.DataFrame:
    """One row per enrollment with dropout status and optional subgroup cols."""
    agg = subset.groupby("enrollment")[TARGET].max().rename("dropout").to_frame()
    if extra_cols:
        meta = subset[["enrollment"] + extra_cols].drop_duplicates("enrollment").set_index("enrollment")
        agg = agg.join(meta)
    return agg.reset_index()


# ── Table 1: sample overview by year ──────────────────────────────────────────

def sample_overview_table(df: pd.DataFrame) -> str:
    def _year_stats(subset: pd.DataFrame) -> dict:
        enr = subset["enrollment"].nunique()
        n_rows = len(subset)
        dr = subset.groupby("enrollment")[TARGET].max().mean()
        return {"n_enr": enr, "n_rows": n_rows, "dropout_rate": dr,
                "rows_per_enr": n_rows / enr if enr > 0 else 0}

    train_years = list(range(TRAIN_YEARS[0], TRAIN_YEARS[1] + 1))

    body_rows: list[tuple[str, str, dict]] = []

    for year in train_years:
        body_rows.append((str(year), "Train", _year_stats(df[df["start_year"] == year])))

    train_stats = _year_stats(df[df["start_year"].between(*TRAIN_YEARS)])
    body_rows.append(("\\textit{Total train}", "", train_stats))

    for year in TEST_YEARS:
        label = "Test\\textsuperscript{*}" if year == 2023 else "Test"
        body_rows.append((str(year), label, _year_stats(df[df["start_year"] == year])))

    test_stats = _year_stats(df[df["start_year"].isin(TEST_YEARS)])
    body_rows.append(("\\textit{Total test}", "", test_stats))

    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{llrrrr}",
        "\\hline",
        ("\\textbf{Year} & \\textbf{Split} & \\textbf{Enrollments} "
         "& \\textbf{EWS rows} & \\textbf{Dropout rate} "
         "& \\textbf{Rows / enr.} \\\\"),
        "\\hline",
    ]

    total_indices = {len(train_years), len(train_years) + 1 + len(TEST_YEARS)}
    for i, (year_str, split, s) in enumerate(body_rows):
        if i in total_indices:
            lines.append("\\hline")
        n_do = int(round(s["dropout_rate"] * s["n_enr"])) if pd.notna(s["dropout_rate"]) else 0
        rate = (disclosure.SUPPRESSED_TEX if disclosure.rate_is_small(n_do, s["n_enr"])
                else _fmt_pct(s["dropout_rate"]))
        lines.append(
            f"{year_str} & {split} & {disclosure.fmt_count(s['n_enr'])} & {_fmt_int(s['n_rows'])} "
            f"& {rate} & {s['rows_per_enr']:.1f} \\\\"
        )
        if i in total_indices:
            lines.append("\\hline")

    lines += [
        "\\end{tabular}",
        ("\\caption{\\textbf{Sample overview by cohort year.} "
         "Enrollments are counted at the enrollment level; EWS rows are "
         "individual EWS-activation observations (up to two per year). "
         "Dropout rate is computed at the enrollment level. "
         "\\textsuperscript{*}~2023 cohort is incomplete — many enrollments "
         "unresolved; metrics for this year should be interpreted with caution.}"),
        "\\label{tab:sample_overview}",
        "\\end{table}",
    ]
    return "\n".join(lines)


# ── Table 1b: student-level sample overview ───────────────────────────────────

def person_level_table(df: pd.DataFrame) -> str | None:
    """
    Student-level counterpart to sample_overview_table().

    A student (person_id) can have multiple enrollments (e.g. re-enrolling
    after a dropout), so the enrollment-level dropout rate in Table
    sample_overview is not the same as the share of *students* who ever
    drop out. Reports, per split: distinct students, enrollments per
    student (mean/median/max), and the student-level dropout rate (share
    of students with at least one dropped-out enrollment).

    Returns None (and prints a warning) if `person_id` is not present in
    `df`, since the student-level grouping is not possible without it.
    """
    if "person_id" not in df.columns:
        print("Warning: 'person_id' not in dataframe — skipping person-level table.")
        return None

    def _split_stats(subset: pd.DataFrame) -> dict:
        enr_level = subset.groupby(["person_id", "enrollment"])[TARGET].max().reset_index()
        per_student = enr_level.groupby("person_id").agg(
            n_enr=("enrollment", "nunique"),
            ever_dropout=(TARGET, "max"),
        )
        return {
            "n_students":       per_student.index.nunique(),
            "n_enr":            int(per_student["n_enr"].sum()),
            "enr_per_student_mean":   per_student["n_enr"].mean(),
            "enr_per_student_median": per_student["n_enr"].median(),
            "enr_per_student_max":    int(per_student["n_enr"].max()),
            "student_dropout_rate":   per_student["ever_dropout"].mean(),
        }

    train = df[df["start_year"].between(*TRAIN_YEARS)]
    test  = df[df["start_year"].isin(TEST_YEARS)]

    rows = [("Train (2016--2018)", _split_stats(train)),
            ("Test (2019--2023)",  _split_stats(test))]

    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lrrrrrr}",
        "\\hline",
        ("\\textbf{Split} & \\textbf{Students} & \\textbf{Enrollments} "
         "& \\multicolumn{3}{c}{\\textbf{Enrollments / student}} "
         "& \\textbf{Student} \\\\"),
        "\\cmidrule(lr){4-6}",
        (" & & & \\textbf{Mean} & \\textbf{Median} & \\textbf{Max} "
         "& \\textbf{dropout rate} \\\\"),
        "\\hline",
    ]
    for label, s in rows:
        lines.append(
            f"{label} & {_fmt_int(s['n_students'])} & {_fmt_int(s['n_enr'])} "
            f"& {s['enr_per_student_mean']:.2f} & {s['enr_per_student_median']:.0f} "
            f"& {s['enr_per_student_max']} "
            f"& {_fmt_pct(s['student_dropout_rate'])} \\\\"
        )
    lines += [
        "\\hline",
        "\\end{tabular}",
        ("\\caption{\\textbf{Student-level sample overview.} "
         "A student (\\texttt{person\\_id}) can have multiple enrollments "
         "(e.g.\\ re-enrolling after a dropout). Student dropout rate is "
         "the share of students with at least one enrollment that ended "
         "in dropout, and is not directly comparable to the "
         "enrollment-level dropout rate in Table~\\ref{tab:sample_overview}, "
         "since one student's repeated enrollments each count separately "
         "there.}"),
        "\\label{tab:person_level_overview}",
        "\\end{table}",
    ]
    return "\n".join(lines)


# ── Table 2: subgroup dropout rates ───────────────────────────────────────────

def subgroup_dropout_table(df: pd.DataFrame) -> str:
    ## Level stays a descriptive row — it is only its use as a FAIRNESS axis
    ## that was dropped. `older_for_cohort` is added because the audit now
    ## reports on it, and a reader needs its base rates.
    SUB_COLS = ["female", "nationality", "older_for_cohort", "undergraduate"]

    train_enr = _enr_level(
        df[df["start_year"].between(*TRAIN_YEARS)], extra_cols=SUB_COLS
    )
    test21_enr = _enr_level(
        df[df["start_year"] == MAIN_TEST_YEAR], extra_cols=SUB_COLS
    )

    # Keys are the post-recode nationality levels (registry codes, ordered
    # DK < Nordic < EU_EEA_CH < Other), not the old ISO-2-style spellings.
    NAT_LABELS = {
        "DK":        "Danish",
        "Nordic":    "Nordic",
        "EU_EEA_CH": "EU/EEA/Switzerland (non-Nordic)",
        "Other":     "International (non-EU/EEA, non-Nordic)",
    }

    # Every row is built from a group of enrollments in each of the two
    # samples. Groups that add up to a known total are suppressed together:
    # Female + Male and Undergraduate + Graduate make the cohort (printed in the
    # sample overview), Nordic + EU + Other make Non-Danish.
    def _block(rows: list[tuple[str, callable, str]]) -> list[str]:
        """rows: (label, mask_fn, prefix); complementary suppression within the block."""
        out_cells = {}
        for name, enr in (("t", train_enr), ("e", test21_enr)):
            groups = [enr[m(enr)] for _, m, _ in rows]
            hide_n, hide_rate = _complementary(groups)
            out_cells[name] = [_group_cells(g, i in hide_n, i in hide_rate) for i, g in enumerate(groups)]
        return [
            f"{prefix}{label} & {out_cells['t'][i][0]} & {out_cells['t'][i][1]} "
            f"& {out_cells['e'][i][0]} & {out_cells['e'][i][1]} \\\\"
            for i, (label, _, prefix) in enumerate(rows)
        ]

    def _single(label: str, mask_fn, prefix: str) -> str:
        return _block([(label, mask_fn, prefix)])[0]

    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lrrrr}",
        "\\hline",
        (" & \\multicolumn{2}{c}{\\textbf{Training (2016--2018)}} "
         "& \\multicolumn{2}{c}{\\textbf{Test 2021}} \\\\"),
        ("\\textbf{Subgroup} & \\textbf{N} & \\textbf{Dropout rate} "
         "& \\textbf{N} & \\textbf{Dropout rate} \\\\"),
        "\\hline",
    ]

    ind = "\\hspace{1em}"
    # Sex
    lines.append("\\multicolumn{5}{l}{\\textbf{Sex}} \\\\")
    lines += _block([("Female", lambda d: d["female"] == 1, ind),
                     ("Male",   lambda d: d["female"] == 0, ind)])

    # Nationality — Danish, then Non-Danish aggregate, then sub-groups indented further.
    # `!= DANISH_CATEGORY` would count unknown citizenship (unknown codes → NaN) as
    # non-Danish; the explicit notna() keeps unknowns out of both rows, matching
    # how the fairness audit treats them. Their number goes in the caption.
    lines.append("\\multicolumn{5}{l}{\\textbf{Nationality}} \\\\")
    lines.append(_single("Danish", lambda d: d["nationality"] == DANISH_CATEGORY, ind))
    lines.append(_single(
        "Non-Danish",
        lambda d: d["nationality"].notna() & (d["nationality"] != DANISH_CATEGORY), ind,
    ))
    lines += _block([
        (NAT_LABELS[code], (lambda c: lambda d: d["nationality"] == c)(code), "\\hspace{2em}")
        for code in ("Nordic", "EU_EEA_CH", "Other")
    ])

    # Entry age, relative to the level-and-cohort median (src/data.py)
    lines.append("\\multicolumn{5}{l}{\\textbf{Entry age, relative to cohort}} \\\\")
    lines += _block([("Older",   lambda d: d["older_for_cohort"] == 1, ind),
                     ("Younger", lambda d: d["older_for_cohort"] == 0, ind)])

    # Level
    lines.append("\\multicolumn{5}{l}{\\textbf{Level}} \\\\")
    lines += _block([("Undergraduate", lambda d: d["undergraduate"] == 1, ind),
                     ("Graduate",      lambda d: d["undergraduate"] == 0, ind)])

    unknown = "~and~".join(
        disclosure.fmt_count(int(enr["nationality"].isna().sum())) for enr in (train_enr, test21_enr)
    )

    lines += [
        "\\hline",
        "\\end{tabular}",
        ("\\caption{\\textbf{Subgroup sizes and enrollment-level dropout rates "
         "in the training data and the 2021 test cohort.} "
         "Nationality categories: Danish; Nordic (non-Danish Nordic countries: "
         "Norway, Sweden, Finland, Iceland, Faroe Islands, Greenland); "
         "EU (non-Nordic EU member states); "
         "International (non-EU, non-Nordic). "
         "UK nationals are classified as EU for enrollments starting before 2021, "
         "reflecting their pre-Brexit free-tuition status in Denmark; "
         "from 2021 onwards they are classified as International (non-EU). "
         f"{unknown} students (training and 2021) have unknown citizenship and "
         "are not in the nationality rows.}"),
        "\\label{tab:subgroup_dropout}",
        "\\end{table}",
    ]
    return "\n".join(lines)


# ── plain-text summary ───────────────────────────────────────────────────────

def text_summary(df: pd.DataFrame) -> str:
    train = df[df["start_year"].between(*TRAIN_YEARS)]
    test  = df[df["start_year"].isin(TEST_YEARS)]
    test21 = df[df["start_year"] == MAIN_TEST_YEAR]
    ## Every number in the paragraph describes the same population, the
    ## analysis cohorts.
    analysis = pd.concat([train, test])

    train_n_enr = train["enrollment"].nunique()
    test_n_enr  = test["enrollment"].nunique()

    train_n_rows = len(train)
    test_n_rows  = len(test)

    train_rows_per_enr = train_n_rows / train_n_enr
    test_rows_per_enr  = test_n_rows  / test_n_enr

    # Student-level
    train_n_students = train["person_id"].nunique() if "person_id" in df.columns else None
    if train_n_students:
        train_person_enr = train.groupby(["person_id", "enrollment"])[TARGET].max().reset_index()
        enr_per_student = train_person_enr.groupby("person_id")["enrollment"].nunique().mean()
        student_dropout_rate = train_person_enr.groupby("person_id")[TARGET].max().mean()
    else:
        enr_per_student = None
        student_dropout_rate = None

    # EWS period distribution (number of distinct ews_dates per enrollment)
    periods = analysis.groupby("enrollment")["ews_date"].nunique()
    p_mean, p_median, p_max = periods.mean(), periods.median(), periods.max()

    # Oct / Mar split
    ews_counts = analysis["ews_month"].value_counts()
    n_oct = ews_counts.get("oct", 0)
    n_mar = ews_counts.get("mar", 0)

    # LMS availability: `lms_observed` is 1 when any LMS feature is non-null for
    # the enrollment-period. Not "any LMS column non-null": lms_observed is
    # itself in LMS and never null, so that test was true on every row (100%).
    if "lms_observed" in analysis.columns:
        lms_pct = analysis.groupby("enrollment")["lms_observed"].max().astype(bool).mean() * 100
    else:
        lms_pct = None

    train_dr  = train.groupby("enrollment")[TARGET].max().mean()
    test21_dr = test21.groupby("enrollment")[TARGET].max().mean()

    lines = [
        "% ===== Descriptive statistics summary =====",
        "%",
        "% Sample size",
        f"%   Training (2016-2018): {train_n_enr:,} enrollments, {train_n_rows:,} EWS rows",
        f"%   Test (2019-2023):     {test_n_enr:,} enrollments, {test_n_rows:,} EWS rows",
    ]
    if train_n_students:
        lines += [
            f"%   Unique students (train): {train_n_students:,}",
            f"%   Mean enrollments per student (train): {enr_per_student:.2f}",
            f"%   Student-level dropout rate (train): {student_dropout_rate:.1%}",
        ]
    lines += [
        "%",
        "% EWS observation structure",
        f"%   Mean EWS rows per enrollment: {train_rows_per_enr:.1f} (train), {test_rows_per_enr:.1f} (test)",
        f"%   EWS periods per enrollment:   mean {p_mean:.1f}, median {p_median:.0f}, max {int(p_max)}",
        f"%   Oct activations: {n_oct:,} | Mar activations: {n_mar:,}",
        "%",
        "% LMS data",
    ]
    if lms_pct is not None:
        lines.append(f"%   Enrollments with any LMS data: {lms_pct:.1f}%")
    else:
        lines.append("%   LMS columns not found in dataset")
    lines += [
        "%",
        "% Dropout rates (enrollment level)",
        f"%   Training:  {train_dr:.1%}",
        f"%   Test 2021: {test21_dr:.1%}",
        "",
        "% ===== Manuscript-ready paragraph =====",
        "%",
    ]

    para = (
        f"The training set comprises enrollments from cohorts 2016--2018 "
        f"({train_n_enr:,} enrollments, {train_n_rows:,} EWS-activation rows; "
        f"mean {train_rows_per_enr:.1f} activations per enrollment). "
    )
    if train_n_students:
        para += (
            f"These represent {train_n_students:,} unique students "
            f"(mean {enr_per_student:.2f} enrollments per student); "
            f"{_fmt_pct(student_dropout_rate)} of students had at least one "
            f"enrollment end in dropout. "
        )
    train_dr_str  = _fmt_pct(train_dr)
    test21_dr_str = _fmt_pct(test21_dr)
    para += (
        f"The test set (cohorts 2019--2023) contains {test_n_enr:,} enrollments "
        f"across {test_n_rows:,} rows. "
        f"EWS activations occur twice yearly (October and March); "
        f"enrollments are observed for a median of {p_median:.0f} activations "
        f"(mean {p_mean:.1f}, max {int(p_max)}). "
        f"The enrollment-level dropout rate is {train_dr_str} in the training set "
        f"and {test21_dr_str} in the 2021 test cohort."
    )
    if lms_pct is not None:
        para += (
            f" Canvas LMS data are available for {lms_pct:.1f}\\% of enrollments."
        )

    lines.append(para)
    return "\n".join(lines)


# ── pilot_snapshot sample description ───────────────────────────────────────────────

def pilot_snapshot_sample_description(df: pd.DataFrame, tab: pd.DataFrame) -> str:
    """
    LaTeX file describing the October 2023 pilot snapshot (pilot_snapshot).

    Outputs two blocks:
    - \\newcommand macros for inline use in text
    - A summary table with sample characteristics
    """
    SNAPSHOT = pd.Timestamp("2023-10-15")

    # --- cohort denominators ---
    all_enroll = df.drop_duplicates("enrollment")
    full_n = len(all_enroll)

    # The 2019–2023 cohort is the reference universe for pilot_snapshot
    cohort_2019_2023 = all_enroll[all_enroll["start_year"].between(2019, 2023)]
    n_cohort = len(cohort_2019_2023)
    n_tab = len(tab)

    # --- outcomes ---
    # Every enrollment in the snapshot was ACTIVE on the snapshot date: a row
    # exists only while ews_date < status_change_date. So nothing was resolved at the
    # snapshot, and the only outcome to report is the eventual one. A split into
    # "resolved by 15 Oct 2023" and "still enrolled" would leave the first group
    # empty, with a 0/0 dropout rate.
    n_ended_before = int((tab["status_change_date"] <= SNAPSHOT).sum())
    if n_ended_before:
        print(f"  WARNING: {n_ended_before:,} pilot_snapshot enrollments ended on or before "
              f"{SNAPSHOT.date()} yet appear in the snapshot — the data break "
              "ews_date < status_change_date.")
    n_dropout_eventual = int((tab["dropout"] == 1).sum())
    dropout_rate_tab = tab["dropout"].mean()          # eventual outcome
    dropout_rate_cohort = all_enroll["dropout"].mean()

    pct_tab_of_full   = n_tab / full_n * 100
    pct_tab_of_cohort = n_tab / n_cohort * 100

    # --- demographics ---
    pct_female    = tab["female"].mean() * 100
    pct_danish    = tab["danish_nationality"].mean() * 100
    pct_undergrad = tab["undergraduate"].mean() * 100
    mean_age      = tab["age_at_study_start"].mean()

    # --- faculty breakdown ---
    fac_counts = tab["faculty_code"].value_counts()
    fac_b_n  = int(fac_counts.get("FAC_B",  0))
    fac_a_n = int(fac_counts.get("FAC_A", 0))

    # --- contact pilot ---
    # Within the faculty: a flag set elsewhere is not one of its contacts. Faculty
    # B's flag is not a record of contacts at all (CONTACTS_RECORDED_FACULTIES).
    n_contact_fac_a = int(tab.loc[tab["faculty_code"] == "FAC_A", "contact_fac_a"].sum())
    pct_contact_fac_a = n_contact_fac_a / fac_a_n * 100 if fac_a_n > 0 else 0.0

    def _cmd(name: str, value: str) -> str:
        return f"\\newcommand{{\\{name}}}{{{value}}}"

    def _n(x: int) -> str:
        return f"{x:,}"

    def _pct(x: float) -> str:
        return "--" if pd.isna(x) else f"{x:.1f}\\%"

    # ── newcommand block ──────────────────────────────────────────────────────
    cmds = [
        "% ── pilot_snapshot sample — auto-generated by src/descriptive_table.py ──",
        "% Use \\input{results/descriptive/pilot_snapshot_sample.tex} to load these macros.",
        "",
        "% --- sample sizes ---",
        _cmd("tabN",          _n(n_tab)),
        _cmd("tabFullCohortN", _n(full_n)),
        _cmd("tabCohortN",    _n(n_cohort)),
        _cmd("tabShareOfFull",   _pct(pct_tab_of_full)),
        _cmd("tabShareOfCohort", _pct(pct_tab_of_cohort)),
        "",
        "% --- dropout (eventual: every enrollment was active at the snapshot) ---",
        _cmd("tabDropoutEventualN",    disclosure.fmt_count(n_dropout_eventual)),
        _cmd("tabDropoutRateTab",      _pct(dropout_rate_tab * 100)),
        _cmd("tabDropoutRateCohort",   _pct(dropout_rate_cohort * 100)),
        "",
        "% --- demographics ---",
        _cmd("tabPctFemale",    _pct(pct_female)),
        _cmd("tabPctDanish",    _pct(pct_danish)),
        _cmd("tabPctUndergrad", _pct(pct_undergrad)),
        _cmd("tabMeanAge",      f"{mean_age:.1f}"),
        "",
        "% --- contact pilot (FAC_B & FAC_A faculties) ---",
        _cmd("tabFacBN",          _n(fac_b_n)),
        _cmd("tabFacAN",          _n(fac_a_n)),
        _cmd("tabContactFacAN",   _n(n_contact_fac_a)),
        _cmd("tabContactFacAPct", _pct(pct_contact_fac_a)),
    ]

    # ── summary table ─────────────────────────────────────────────────────────
    table_rows = [
        ("Total enrollments",                   _n(n_tab)),
        ("Share of cohorts 2019--2023",          _pct(pct_tab_of_cohort)),
        ("",                                     ""),
        ("Eventually dropped out",               f"{disclosure.fmt_count(n_dropout_eventual)} ({_pct(dropout_rate_tab * 100)})"),
        ("",                                     ""),
        ("\\multicolumn{2}{l}{\\textit{Demographics}} \\\\[-2pt]", None),
        ("\\hspace{1em}Female",                  _pct(pct_female)),
        ("\\hspace{1em}Danish nationality",      _pct(pct_danish)),
        ("\\hspace{1em}Undergraduate",           _pct(pct_undergrad)),
        ("\\hspace{1em}Mean age at enrollment",  f"{mean_age:.1f}"),
        ("",                                     ""),
        ("\\multicolumn{2}{l}{\\textit{October 2023 operational rule-based model}} \\\\[-2pt]", None),
        ("\\hspace{1em}Faculty B enrollments",    _n(fac_b_n)),
        ("\\hspace{1em}Faculty A enrollments",    _n(fac_a_n)),
        ("\\hspace{2em}Contacted",               f"{_n(n_contact_fac_a)} ({_pct(pct_contact_fac_a)})"),
    ]

    table_lines = [
        "",
        "% ── summary table ───────────────────────────────────────────────────────",
        "\\begin{table}[ht]",
        "\\centering",
        "\\begin{tabular}{lr}",
        "\\hline",
        "\\textbf{Characteristic} & \\textbf{Value} \\\\",
        "\\hline",
    ]
    for label, val in table_rows:
        if label == "" and val == "":
            table_lines.append("\\\\[-4pt]")
        elif val is None:
            table_lines.append(label)
        else:
            table_lines.append(f"{label} & {val} \\\\")
    table_lines += [
        "\\hline",
        "\\end{tabular}",
        ("\\caption{\\textbf{October 2023 EWS snapshot: "
         "sample characteristics.} "
         "The snapshot covers all enrollments from cohorts 2019--2023 with at "
         "least one October EWS activation. "
         "Every enrollment in the snapshot was still active on 15 October 2023, "
         "so no outcome was known at the snapshot; the dropout count and rate "
         "are eventual, over outcomes observed after the snapshot. "
         "Percentages for demographics are computed over all \\tabN{} enrollments. "
         "The operational rule-based model was deployed at Faculties A and B only; "
         "the contact count is reported for Faculty A only, because it is not known "
         "which students Faculty B contacted.}"),
        "\\label{tab:pilot_snapshot_sample}",
        "\\end{table}",
    ]

    return "\n".join(cmds) + "\n" + "\n".join(table_lines) + "\n"


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df  = load_analysis_df()
    tab = load_pilot_snapshot()

    ## The three tables the paper prints; the student-level overview and the
    ## text summary are printed below but not written.
    overview_tex     = sample_overview_table(df)
    person_level_tex = person_level_table(df)
    subgroup_tex     = subgroup_dropout_table(df)
    summary_txt      = text_summary(df)
    pilot_snapshot_tex     = pilot_snapshot_sample_description(df, tab)

    (OUT_DIR / "sample_overview.tex").write_text(overview_tex)
    (OUT_DIR / "subgroup_dropout.tex").write_text(subgroup_tex)
    (OUT_DIR / "pilot_snapshot_sample.tex").write_text(pilot_snapshot_tex)

    print(f"Saved to {OUT_DIR}\n")
    print("--- Sample overview ---")
    print(overview_tex)
    if person_level_tex is not None:
        print("\n--- Student-level overview ---")
        print(person_level_tex)
    print("\n--- Subgroup dropout ---")
    print(subgroup_tex)
    print("\n--- Summary text ---")
    print(summary_txt)
    print("\n--- pilot_snapshot sample description ---")
    print(pilot_snapshot_tex)


if __name__ == "__main__":
    main()
