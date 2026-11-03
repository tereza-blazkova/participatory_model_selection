"""
Single entry point for loading the modelling frame.

Every script that reads the parquet must go through `load_analysis_df()` rather
than calling `pd.read_parquet` itself. The analysis population is defined by a
filter (see config.EXCLUDED_FACULTIES); with six independent
read sites the filter would apply to some outputs and not others, and the
mismatch would show up as tables that disagree with each other rather than as
an error.

    from src.data import load_analysis_df
    df = load_analysis_df()

A frame already in memory goes through the identical steps with
prepare_analysis_df(df).
"""
import numpy as np
import pandas as pd

from paths.paths import DATA_PATH_PARQUET, PILOT_SNAPSHOT_PARQUET
from src.config import (
    DANISH_CATEGORY,
    EXCLUDED_FACULTIES,
    GPA_REF_MIN_ENROLLMENTS,
    NATIONALITY_LEVELS,
)

_GPA_KEYS = ["programme_level", "start_year"]
_GPA_REF_COLS = ["enrollment", *_GPA_KEYS, "secondary_ed_gpa"]
_GPA_RAW_COLS = ["enrollment", "programme_code", "undergraduate", "start_year", "secondary_ed_gpa"]
_AGE_RAW_COLS = ["enrollment", "undergraduate", "start_year", "age_at_study_start"]


def _apply_population_filter(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Drop faculties excluded from the analysis population."""
    if "faculty_code" not in df.columns or not EXCLUDED_FACULTIES:
        return df

    mask = df["faculty_code"].isin(EXCLUDED_FACULTIES)
    if not mask.any():
        return df

    n_rows = int(mask.sum())
    n_enr = df.loc[mask, "enrollment"].nunique() if "enrollment" in df.columns else 0
    out = df.loc[~mask].copy()
    if verbose:
        print(f"  Population filter: dropped {', '.join(EXCLUDED_FACULTIES)} — "
              f"{n_rows:,} rows / {n_enr:,} enrollments")
    return out


def _drop_never_evaluated(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Drop the period-0 sentinel rows.

    Period 0 is not an EWS activation. Activations are 1-indexed; period 0
    is preprocessing's early-return branch for enrollments that ended before
    reaching any EWS date, carrying `ews_date` NaT, `ews_month` None, and
    `period_end` set to the enrollment's own end. It means "never evaluated".

    The unit of analysis here is (enrollment, ews_date), so a row with no
    ews_date is not one of them. Left in, it would take a model score at an
    activation that never happened, feed that score into the enrollment-level
    max-prob, and appear as a leading point on every period-indexed figure —
    the fairness-over-periods panels and the performance-by-period figure's months-of-enrollment axis
    both build their axis from sorted(period.unique()) and would have shown it.

    NOTE ON THE DENOMINATOR: these enrollments have only this one row, so
    dropping it drops the enrollment. Recall is therefore computed over students
    the EWS system could actually have flagged, which is the right denominator
    for "of the students we could have caught, how many did we", but NOT for
    "of all students who dropped out, how many did we catch". If the paper makes
    the second claim, these enrollments belong in the denominator as guaranteed
    misses and this filter has to be reconsidered.
    """
    if "period" not in df.columns:
        return df
    mask = df["period"] == 0
    if not mask.any():
        return df

    n_rows = int(mask.sum())
    n_enr = df.loc[mask, "enrollment"].nunique() if "enrollment" in df.columns else 0
    out = df.loc[~mask].copy()
    if verbose:
        print(f"  Dropped {n_rows:,} period-0 sentinel rows / {n_enr:,} enrollments "
              f"(ended before any EWS date — never evaluated)")
    return out


def _normalise_nationality(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Restore the ordered categorical dtype on `nationality`.

    Parquet usually preserves it, but a frame rebuilt from row dicts or read
    through a different engine comes back as plain object, which silently
    reorders every table and legend that relies on the level order. Unknown
    codes stay NaN — they are dropped from subgroup analyses, not folded into
    "Other".

    ANYTHING not in NATIONALITY_LEVELS becomes NaN, and NaN is read downstream
    as "citizenship unknown" and dropped from every nationality analysis. One
    recode, one trailing space or one casing difference in a level name would
    therefore delete that whole group from the fairness figures, the ABROCA
    heatmap and the subgroup table, with no error anywhere. So the coercion is
    counted and reported; a non-zero count is the first thing to check.
    """
    if "nationality" not in df.columns:
        return df
    df = df.copy()
    before = df["nationality"]
    after = pd.Categorical(before, categories=NATIONALITY_LEVELS, ordered=True)

    lost = pd.notna(before).to_numpy() & pd.isna(after)
    if lost.any():
        values = pd.Series(before.to_numpy()[lost]).value_counts()
        print(f"  WARNING: nationality — {int(lost.sum()):,} rows carry a value that is "
              f"not one of {NATIONALITY_LEVELS} and became NaN, so they will be "
              f"dropped from every nationality analysis:")
        for value, n in values.items():
            print(f"      {value!r}: {n:,} rows")
        print("    If one of these is a renamed level, fix NATIONALITY_LEVELS in "
              "src/config.py rather than letting the group disappear.")

    df["nationality"] = after
    return df


def add_programme_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add `programme_level`: the programme code split by degree level, "<code>|BA"
    or "<code>|MA".

    An `programme_code` code can be shared by the bachelor's and the master's in
    the same subject, whose dropout rates can differ. As one
    column, the two get one programme effect, and an additive model cannot
    separate them — `undergraduate` shifts every programme by the same amount.
    This is the programme as the student experiences it, and it is what enters
    the models; raw `programme_code` stays for descriptive use. NaN when either part
    is missing.
    """
    if not {"programme_code", "undergraduate"} <= set(df.columns):
        return df
    level = df["undergraduate"].map({1: "BA", 0: "MA"})
    level_code = df["programme_code"].astype("string") + "|" + level.astype("string")
    df = df.copy()
    df["programme_level"] = level_code.astype(object).where(level_code.notna(), np.nan)
    return df


def gpa_programme_reference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Secondary-GPA mean and sd per programme cohort (programme_level x start_year).

    Counted over ENROLLMENTS, one GPA each. Over rows, every enrollment would
    weigh by how many EWS dates it survived to, and the students who stay
    longest are not a random draw of the cohort's GPAs.
    """
    enr = df[_GPA_REF_COLS].drop_duplicates("enrollment").dropna(subset=["secondary_ed_gpa"])
    return (
        enr.groupby(_GPA_KEYS, observed=True)["secondary_ed_gpa"]
        .agg(gpa_ref_n="size", gpa_ref_mean="mean", gpa_ref_sd="std")
        .reset_index()
    )


def add_gpa_within_programme(
    df: pd.DataFrame, reference: pd.DataFrame | None = None
) -> pd.DataFrame:
    """
    Add `secondary_ed_gpa_z_programme`: GPA as a z-score in its programme cohort.

    The cohort is everyone admitted to the same programme — programme_level, so a
    BA student is compared with BA students only — in the same start_year. It uses no outcome and no information from any other cohort, and
    all of it is known on the first EWS date — so it is a feature definition, not
    a preprocessing step fitted on the test years. Deliberately per cohort, not
    per programme over the training years: admission cut-offs move from year to
    year, and a programme that is new in a test year would otherwise have no
    reference at all.

    NaN when the cohort has fewer than GPA_REF_MIN_ENROLLMENTS enrollments with a
    GPA, or no spread; the preprocessing turns that into a missing indicator.

    `reference` is the frame whose cohorts define the mean and sd — by default
    `df` itself. It must be the WHOLE admitted cohort: pass the full data
    before any row is dropped (load_analysis_df does), and pass the main data
    for pilot_snapshot, whose older cohorts contain only the students still enrolled
    in October 2023.
    """
    df = add_programme_level(df)
    ref_src = df if reference is None else add_programme_level(reference)
    if not set(_GPA_REF_COLS) <= set(df.columns) or not set(_GPA_REF_COLS) <= set(ref_src.columns):
        return df

    ref = gpa_programme_reference(ref_src)
    ref = ref[(ref["gpa_ref_n"] >= GPA_REF_MIN_ENROLLMENTS) & (ref["gpa_ref_sd"] > 0)]
    ref = ref.set_index(_GPA_KEYS)

    keys = pd.MultiIndex.from_frame(df[_GPA_KEYS])
    mean = ref["gpa_ref_mean"].reindex(keys).to_numpy(dtype=float)
    sd   = ref["gpa_ref_sd"].reindex(keys).to_numpy(dtype=float)

    df = df.copy()
    df["secondary_ed_gpa_z_programme"] = (df["secondary_ed_gpa"].to_numpy(dtype=float) - mean) / sd
    return df


_AGE_KEYS     = ["undergraduate", "start_year"]
_AGE_REF_COLS = ["enrollment", *_AGE_KEYS, "age_at_study_start"]


def age_level_reference(df: pd.DataFrame) -> pd.DataFrame:
    """Median age at start per degree level x cohort, over unique enrollments."""
    one = df[_AGE_REF_COLS].drop_duplicates("enrollment")
    out = (one.groupby(_AGE_KEYS)["age_at_study_start"]
              .agg(age_ref_median="median", age_ref_n="count")
              .reset_index())
    return out


def add_age_group_within_level(
    df: pd.DataFrame, reference: pd.DataFrame | None = None
) -> pd.DataFrame:
    """
    Add `older_for_cohort`: 1 if this enrollment started older than the median
    for its degree level and cohort, 0 if at or below it, NaN if unknown.

    This is the age axis the fairness audit reports, and it is deliberately
    RELATIVE, not absolute. A master's student starts after a bachelor's, so a
    split on absolute age — a pooled median, or a fixed cut near it — would
    largely reproduce the BA/MA distinction. The age split is made WITHIN level
    so that it does not reproduce the BA/MA split: the axis is orthogonal to
    level by construction.

    Per level x start_year, not level over the training years, for the same
    reason secondary_ed_gpa_z_programme is per cohort: what is held constant is
    the RELATIVE position, so each year's two groups are the same size and mean
    the same thing. A frozen cut would hold the age constant instead and let the
    group sizes drift as cohorts age, which puts composition change into every
    over-years trend. The numeric age behind the label does move between
    cohorts; that belongs in the caption, not in the definition.

    The median is taken over unique enrollments, so an enrollment observed at
    six EWS dates does not count six times. One person holding a BA and a MA
    enrollment is two enrollments and can land on opposite sides — the unit of
    analysis here is the enrollment, as everywhere else.

    Strictly above the median is "older"; exactly at it is "younger". NaN when
    the age is unknown (the fairness audit drops those rows rather than making
    them a third group) or when the cohort holds fewer than
    GPA_REF_MIN_ENROLLMENTS enrollments.

    `reference` is the frame whose cohorts define the medians, defaulting to
    `df`. Pass the main data for pilot_snapshot: its older cohorts hold only the
    students still enrolled in October 2023, so a median computed there is the
    median of the survivors, and the groups would be defined by an outcome in a
    variable used to audit outcomes.
    """
    ref_src = df if reference is None else reference
    if not set(_AGE_REF_COLS) <= set(df.columns) or not set(_AGE_REF_COLS) <= set(ref_src.columns):
        return df

    ref = age_level_reference(ref_src)
    ref = ref[ref["age_ref_n"] >= GPA_REF_MIN_ENROLLMENTS].set_index(_AGE_KEYS)

    keys   = pd.MultiIndex.from_frame(df[_AGE_KEYS])
    median = ref["age_ref_median"].reindex(keys).to_numpy(dtype=float)
    age    = df["age_at_study_start"].to_numpy(dtype=float)

    older = np.where(np.isnan(age) | np.isnan(median), np.nan, (age > median).astype(float))
    df = df.copy()
    df["older_for_cohort"] = older
    return df


def derive_danish_nationality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add `danish_nationality` from `nationality` if it is not already present.

    The test is against DANISH_CATEGORY ("DK"). The result is not cast to int:
    unknown citizenship (unknown codes) is NaN, and `.astype(int)` would turn it
    into 0 — counting unknowns as non-Danish. NaN is kept, so `== 0` / `== 1`
    subgroup masks drop those enrollments instead.
    """
    if "nationality" not in df.columns:
        return df
    df = df.copy()
    if "danish_nationality" in df.columns:
        # A column that arrived with the data may hold 0 for unknown citizenship.
        df["danish_nationality"] = df["danish_nationality"].where(df["nationality"].notna())
        return df
    is_dk = df["nationality"] == DANISH_CATEGORY
    df["danish_nationality"] = is_dk.astype(float).where(df["nationality"].notna())
    return df


def load_analysis_df(
    path=None, verbose: bool = True, keep_never_evaluated: bool = False
) -> pd.DataFrame:
    """
    Load the EWS modelling frame with the analysis population applied.

    Pass keep_never_evaluated=True for descriptive work that needs to count the
    enrollments which ended before any EWS date. Never for modelling.
    """
    path = DATA_PATH_PARQUET if path is None else path
    if verbose:
        print(f"Loading data from {path} ...")
    return prepare_analysis_df(
        pd.read_parquet(path), verbose=verbose, keep_never_evaluated=keep_never_evaluated
    )


def prepare_analysis_df(
    df: pd.DataFrame, verbose: bool = True, keep_never_evaluated: bool = False
) -> pd.DataFrame:
    """
    Everything load_analysis_df() does after reading the parquet.

    For a frame that is already in memory. Pass the FULL frame: the
    programme-cohort GPA reference is
    computed before any row is dropped, so the period-0 enrollments (admitted,
    then gone before the first EWS date) still count as members of their cohort.
    """
    df = add_programme_level(df)
    df = add_gpa_within_programme(df)
    df = add_age_group_within_level(df)
    df = _apply_population_filter(df, verbose=verbose)
    if not keep_never_evaluated:
        df = _drop_never_evaluated(df, verbose=verbose)
    df = _normalise_nationality(df)
    if verbose and {"enrollment", "start_year"} <= set(df.columns):
        print(f"  {len(df):,} rows | {df['enrollment'].nunique():,} enrollments | "
              f"start_year {df['start_year'].min()}–{df['start_year'].max()}")
    return df


def _gpa_reference_frame(verbose: bool = True) -> pd.DataFrame | None:
    """The main data's GPA columns — the whole-cohort reference for pilot_snapshot."""
    try:
        return pd.read_parquet(DATA_PATH_PARQUET, columns=_GPA_RAW_COLS)
    except (FileNotFoundError, OSError, KeyError, ValueError) as exc:
        if verbose:
            print(f"  WARNING: main data unavailable as GPA reference ({exc}); "
                  "secondary_ed_gpa_z_programme computed within pilot_snapshot, whose "
                  "older cohorts hold only students still enrolled.")
        return None


def _age_reference_frame(verbose: bool = True) -> pd.DataFrame | None:
    """The main data's age columns — the whole-cohort reference for pilot_snapshot."""
    try:
        return pd.read_parquet(DATA_PATH_PARQUET, columns=_AGE_RAW_COLS)
    except (FileNotFoundError, OSError, KeyError, ValueError) as exc:
        if verbose:
            print(f"  WARNING: main data unavailable as age reference ({exc}); "
                  "older_for_cohort computed within pilot_snapshot, whose older cohorts "
                  "hold only students still enrolled.")
        return None


def load_pilot_snapshot_unfiltered(path=None) -> pd.DataFrame:
    """pilot_snapshot WITHOUT the population filter, for tools/check_contact_flags.py."""
    path = PILOT_SNAPSHOT_PARQUET if path is None else path
    df = add_gpa_within_programme(
        pd.read_parquet(path), reference=_gpa_reference_frame(verbose=False)
    )
    df = add_age_group_within_level(df, reference=_age_reference_frame(verbose=False))
    return _normalise_nationality(df)


def load_pilot_snapshot(path=None, verbose: bool = True) -> pd.DataFrame:
    """
    Load the October 2023 pilot snapshot with the same population filter.
    """
    path = PILOT_SNAPSHOT_PARQUET if path is None else path
    if verbose:
        print(f"Loading pilot_snapshot from {path} ...")
    df = pd.read_parquet(path)
    df = add_gpa_within_programme(df, reference=_gpa_reference_frame(verbose))
    df = add_age_group_within_level(df, reference=_age_reference_frame(verbose))
    df = _apply_population_filter(df, verbose=verbose)
    return _normalise_nationality(df)


