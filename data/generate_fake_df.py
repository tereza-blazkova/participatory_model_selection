"""
Generate data/fake_df.parquet — synthetic EWS data.

Schema: the column names and dtypes the pipeline expects. All
values are randomly generated.

Run from project root:
    python data/generate_fake_df.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# ── Parameters ────────────────────────────────────────────────────────────────
N_ENROLLMENTS = 4000

# ── Settings — drawn at random, deliberately uninformative ───────────────────
# Every share, rate and effect below is drawn from the same broad, generic range
# rather than chosen to look like any particular student body, and every count
# is a round number. The data only has to exercise the code: it has the schema's
# columns and categories, some missing values, and some enrollments that end
# before their first activation. A separate generator keeps the draws
# reproducible without shifting the stream `rng` uses.
_srng = np.random.default_rng(7)


def _share() -> float:
    """A share or probability, drawn uniformly from 5–95%."""
    return float(_srng.uniform(0.05, 0.95))


SETTINGS = {
    "start_year_p":        _srng.dirichlet(np.ones(8) * 5),
    ## Kept near even: with a rare group, the small-cell rule leaves nothing to draw.
    "female":              float(_srng.uniform(0.35, 0.65)),
    # DK, Nordic, EU_EEA_CH, Other, unknown
    "nationality_p":       _srng.dirichlet(np.ones(5) * 2),
    "undergraduate":       _share(),
    "quota_student":       _share(),
    "dropped_out_before":  _share(),
    "finished_before":     _share(),
    "admission_p":         _srng.dirichlet(np.ones(4) * 2),
    "admission_quota_null":          _share(),
    "admission_gpa_null":       _share(),
    "spring_share":        _share(),
    # Dropouts who leave before the first activation (the period-0 rows).
    "early_dropout":       _share() / 5,
    "fac_c_ects_coverage":  _share(),
    "ma_ects_coverage":    _share(),
    # Share of enrollment-periods with nothing assessed yet, by period.
    "nothing_assessed":    {0: _share(), 1: _share(), 2: _share()},
    "nothing_assessed_later": _share(),
    "no_lms":              _share(),
    # Dropout: a baseline log-odds and one effect per risk factor, each of
    # either sign.
    "dropout_intercept":   float(_srng.uniform(-2.0, 0.0)),
    "dropout_effects":     _srng.normal(0.0, 0.5, size=4),
    # Outcomes by dropout status: (non-dropout, dropout).
    "pass_share":          (_share(), _share()),
    "fail_share":          (_share(), _share()),
    "ever_failed_share":   _share(),
    "grade_p":             (_srng.dirichlet(np.ones(7)), _srng.dirichlet(np.ones(7))),
    "last_grade_p":        (_srng.dirichlet(np.ones(6)), _srng.dirichlet(np.ones(6))),
    "lms_beta":            _srng.uniform(1.0, 5.0, size=(2, 4, 2)),
}

# ── Per-enrollment static attributes ─────────────────────────────────────────
start_years = rng.choice(
    range(2016, 2024), size=N_ENROLLMENTS,
    p=SETTINGS["start_year_p"],
)

female             = rng.binomial(1, SETTINGS["female"], N_ENROLLMENTS)

# ── Nationality ───────────────────────────────────────────────────────────────
# citizenship_country_code holds national registry nationality codes (international
# vehicle registration codes, some ISO-3 mixed in) — NOT ISO-2. `nationality` is
# an ordered categorical DK < Nordic < EU_EEA_CH < Other; unknown codes map to NaN,
# not to "Other". Downstream code must test nationality_raw for country codes.
_NATIONALITY_GROUPS = {
    "DK":        ["DK"],
    "Nordic":    ["N", "S"],
    "EU_EEA_CH": ["D", "F"],
    "Other":     ["US", "CN"],
}
_UNKNOWN_CODES = ["XXX", "UNK"]

_nat_group = rng.choice(
    ["DK", "Nordic", "EU_EEA_CH", "Other", None],
    size=N_ENROLLMENTS, p=SETTINGS["nationality_p"],
)
nationality_raw = np.array([
    rng.choice(_UNKNOWN_CODES if g is None else _NATIONALITY_GROUPS[g])
    for g in _nat_group
])
nationality = pd.Categorical(
    [g for g in _nat_group],
    categories=["DK", "Nordic", "EU_EEA_CH", "Other"],
    ordered=True,
)
# NaN where nationality is unknown, so subgroup analyses can drop those rows
# rather than silently folding them into "non-Danish".
danish = np.where(
    np.array([g is None for g in _nat_group]), np.nan,
    (np.array([g == "DK" for g in _nat_group])).astype(float),
)
nordic_nationality     = np.where(np.isnan(danish), np.nan,
                                  np.isin(_nat_group, ["DK", "Nordic"]).astype(float))
fee_exempt_nationality = np.where(np.isnan(danish), np.nan,
                                  np.isin(_nat_group, ["DK", "Nordic", "EU_EEA_CH"]).astype(float))

age                = rng.uniform(18, 45, N_ENROLLMENTS)
secondary_gpa      = rng.uniform(0, 12, N_ENROLLMENTS)
undergraduate      = rng.binomial(1, SETTINGS["undergraduate"],      N_ENROLLMENTS)
quota_student      = rng.binomial(1, SETTINGS["quota_student"],      N_ENROLLMENTS)
dropped_out_before = rng.binomial(1, SETTINGS["dropped_out_before"], N_ENROLLMENTS)
finished_before    = rng.binomial(1, SETTINGS["finished_before"],    N_ENROLLMENTS)

admission_type = rng.choice(
    ["direct", "quota_1", "quota_2", "credit_transfer"],
    size=N_ENROLLMENTS, p=SETTINGS["admission_p"],
)
faculty_code            = rng.choice(["FAC_A", "FAC_B", "FAC_C", "FAC_D"], size=N_ENROLLMENTS)
department_code           = rng.choice([f"dept_{i}" for i in range(1, 6)],   size=N_ENROLLMENTS)
department_short_code      = rng.choice([f"DS{i}"   for i in range(10)],      size=N_ENROLLMENTS)
programme_code         = rng.choice([f"prog_{i}" for i in range(1, 11)],  size=N_ENROLLMENTS)
degree_level      = np.where(undergraduate == 1, "BA", "MA")

person_id          = [f"per_{x}" for x in rng.integers(100000, 999999, N_ENROLLMENTS)]

current_status   = rng.choice(["A", "B"], size=N_ENROLLMENTS)
enrollment_code  = rng.choice([f"EC{i}" for i in range(10)], size=N_ENROLLMENTS)
admission_exam        = rng.choice([f"AE{i}" for i in range(10)], size=N_ENROLLMENTS)

admission_quota_null         = rng.random(N_ENROLLMENTS) < SETTINGS["admission_quota_null"]
admission_quota              = np.where(admission_quota_null, np.nan, rng.choice([1.0, 2.0], size=N_ENROLLMENTS).astype(float))
admission_gpa_null      = rng.random(N_ENROLLMENTS) < SETTINGS["admission_gpa_null"]
admission_gpa           = np.where(admission_gpa_null, np.nan, rng.uniform(0, 12, N_ENROLLMENTS))

required_ects_undergrad = np.full(N_ENROLLMENTS, 180.0)
required_ects_grad      = np.full(N_ENROLLMENTS, 120.0)
required_ects           = np.where(undergraduate == 1, required_ects_undergrad, required_ects_grad)

# ── Dropout assignment (random effects of random sign) ────────────────────────
_b = SETTINGS["dropout_effects"]
dropout_logit = (
    rng.normal(SETTINGS["dropout_intercept"], 0.3, N_ENROLLMENTS)
    + _b[0] * dropped_out_before
    + _b[1] * (1 - np.nan_to_num(danish, nan=1.0))
    + _b[2] * (secondary_gpa / 12)
    + _b[3] * (1 - undergraduate)
)
is_dropout = (rng.uniform(size=N_ENROLLMENTS) < 1 / (1 + np.exp(-dropout_logit))).astype(int)

# ── Enrollment timeline ───────────────────────────────────────────────────────
# Intake is February or September only. Spring intake is limited to a couple of
# programmes in this synthetic data; `spring_start` stays descriptive — never a
# feature.
_spring_eligible = np.isin(programme_code, ["prog_1", "prog_2"])
spring_start     = (_spring_eligible & (rng.uniform(size=N_ENROLLMENTS) < SETTINGS["spring_share"])).astype(int)
study_start = pd.to_datetime([
    f"{yr}-02-01" if sp else f"{yr}-09-01"
    for yr, sp in zip(start_years, spring_start)
])

# Graduation: 2–5 years; dropout: 2 months–4 years after start.
# Some dropouts leave within weeks, before the first EWS date can arrive —
# these are the enrollments that get a period-0 sentinel row instead of any
# activation. Without them the sentinel branch never fires and the code paths
# that have to cope with a NaT ews_date go untested.
grad_days    = rng.integers(730, 1825, N_ENROLLMENTS)
dropout_days = np.where(
    rng.uniform(size=N_ENROLLMENTS) < SETTINGS["early_dropout"],
    rng.integers(10, 40, N_ENROLLMENTS),      # too short to reach an EWS date
    rng.integers(60, 1460, N_ENROLLMENTS),
)
status_change_date  = pd.DatetimeIndex(
    np.where(
        is_dropout == 1,
        study_start + pd.to_timedelta(dropout_days, unit="D"),
        study_start + pd.to_timedelta(grad_days,    unit="D"),
    )
)

# ── Credit-coverage defects ─────────────────────────────────────────────────
# Credit coverage is reduced for one faculty and for master's enrollments, to
# exercise the FAC_C exclusion and the bachelor's-only audit.
ects_coverage = np.ones(N_ENROLLMENTS)
ects_coverage[faculty_code == "FAC_C"]       *= SETTINGS["fac_c_ects_coverage"]
ects_coverage[degree_level == "MA"]   *= SETTINGS["ma_ects_coverage"]

# Share of enrollment-periods with nothing assessed yet, i.e. the null rate of
# pass_rate_any (drawn in SETTINGS). Key 0 is the "never evaluated" sentinel,
# not an activation; activations are 1-indexed. Anything past period 2 falls
# back to the "later" share.
_NOTHING_ASSESSED_BY_PERIOD = SETTINGS["nothing_assessed"]

# ── All possible EWS dates 2016–2025 ─────────────────────────────────────────
ews_calendar = sorted(
    [pd.Timestamp(f"{yr}-{mo}-15") for yr in range(2016, 2026) for mo in (3, 10)]
)

# ── Build one row per enrollment × EWS date ───────────────────────────────────
rows = []
for i in range(N_ENROLLMENTS):
    start = study_start[i]
    end   = status_change_date[i]
    drop  = int(is_dropout[i])

    valid_ews = [d for d in ews_calendar if start < d < end]

    # PERIOD 0 IS A SENTINEL, NOT AN ACTIVATION.
    # Activations are 1-indexed (preprocessing's expand_ews_periods uses
    # enumerate(ews_dates, start=1)). Period 0 is the early-return branch for
    # enrollments that ended before reaching any EWS date: ews_date is NaT,
    # ews_month is None, and period_end is the enrollment's own end rather than
    # a 15 Mar / 15 Oct cutoff. It means "never evaluated" — these students were
    # never scored by the EWS system at all.
    is_sentinel = not valid_ews
    period_plan = [(0, pd.NaT)] if is_sentinel else list(enumerate(valid_ews, start=1))

    for period_idx, ews_dt in period_plan:
        if period_idx == 0:
            ews_mon  = None
            days_obs = (end - start).days      # observed to the enrollment's end
            per_end  = end                     # own end, not an EWS cutoff
        else:
            ews_mon  = "oct" if ews_dt.month == 10 else "mar"
            days_obs = (ews_dt - start).days
            per_end  = None                    # filled from the EWS calendar below

        # Registered courses (rows are kept even when the result is not yet
        # observable) vs assessed courses (a result exists by the EWS date).
        # Assessment results dated after period_end are blanked: a course
        # started this semester is not examined until the exam period that
        # FOLLOWS the EWS date, so the first activation carries almost nothing.
        # The share with nothing assessed is set per period (see
        # _NOTHING_ASSESSED_BY_PERIOD) rather than falling out of a ramp.
        n_courses = int(rng.integers(1, 11))
        _null_p = _NOTHING_ASSESSED_BY_PERIOD.get(period_idx, SETTINGS["nothing_assessed_later"])
        if rng.random() < _null_p:
            n_courses_assessed = 0
        else:
            n_courses_assessed = int(np.clip(
                rng.binomial(n_courses, float(rng.beta(2, 2))), 1, n_courses
            ))

        # Expected ECTS on the prescribed schedule: 30 per semester. This is the
        # benchmark ects_behind is measured against and is unaffected by which
        # results are observable yet.
        expected = period_idx * 30.0

        # Observable ECTS, lagging the schedule by one activation and pinned to
        # zero whenever nothing has been assessed, so passed_ects cannot
        # contradict n_courses_assessed.
        expected_assessed = (
            0.0 if n_courses_assessed == 0
            else max(0.0, (period_idx - 1) * 30.0) * ects_coverage[i]
        )

        # Noise scales with the observable total, so the early periods sit near
        # zero instead of picking up a half-normal floor from a fixed sd.
        passed = float(np.clip(
            rng.normal(expected_assessed * SETTINGS["pass_share"][drop],
                       max(0.5, 0.20 * expected_assessed)),
            0, expected_assessed + 30,
        ))
        failed = float(np.clip(
            rng.normal(expected_assessed * SETTINGS["fail_share"][drop],
                       max(0.3, 0.10 * expected_assessed)),
            0, 60,
        ))
        ects_behind = round(expected - passed, 1)
        ects_sum    = round(passed + failed, 1)

        n_courses_numeric = int(np.clip(rng.integers(0, n_courses_assessed + 1),
                                        0, n_courses_assessed))
        n_passed = int(np.clip(
            rng.binomial(n_courses_assessed, SETTINGS["pass_share"][drop]),
            0, n_courses_assessed,
        ))
        n_failed = int(np.clip(
            rng.binomial(n_courses_assessed - n_passed, SETTINGS["fail_share"][drop]),
            0, n_courses_assessed - n_passed,
        ))

        # pass_rate_any is over ASSESSED courses only, and is NaN when
        # nothing has been assessed yet.
        pass_rate = (round(n_passed / n_courses_assessed, 3)
                     if n_courses_assessed > 0 else np.nan)

        # Courses with a failed record pre-dedup; the pass-beats-fail dedup rule
        # means some of them are not currently failing.
        n_courses_ever_failed = int(np.clip(
            n_failed + rng.binomial(max(0, n_courses_assessed - n_failed),
                                    SETTINGS["ever_failed_share"]),
            0, n_courses_assessed,
        ))
        n_courses_with_failed_record = n_courses_ever_failed - n_failed

        grade_vals = [-3,  0,  2,  4,  7, 10, 12]
        grade_prob = SETTINGS["grade_p"][drop]
        # Grade averages are blanked along with the rest of the assessment block.
        if n_courses_assessed == 0:
            avg_grade_ects = avg_grade_numeric = np.nan
        else:
            avg_grade_ects    = float(rng.choice(grade_vals, p=grade_prob))
            avg_grade_numeric = float(rng.choice(grade_vals, p=grade_prob))

        # LMS. Guarded by course_end <= period_end, so Canvas activity before the
        # EWS date IS observable and is NOT blanked — but it is missing wholesale
        # for enrollments with no Canvas footprint, which is what lms_observed
        # lets a linear model separate from an imputed value.
        _has_lms = rng.random() > SETTINGS["no_lms"]
        if _has_lms:
            _lb = SETTINGS["lms_beta"][drop]
            pv       = round(float(rng.beta(*_lb[0])), 3)
            part     = round(float(rng.beta(*_lb[1])), 3)
            # late_pct and missing_pct are the modelling pair; late_or_missing_pct
            # is their sum by construction and is kept only as the alternative
            # single-variable version (never use all three).
            late_pct = round(float(rng.beta(*_lb[2])), 3)
            miss_pct = round(float(rng.beta(*_lb[3])), 3)
            late_or_missing = round(late_pct + miss_pct, 3)
        else:
            pv = part = late_pct = miss_pct = late_or_missing = np.nan
        lms_observed = int(_has_lms)

        n_courses_current = int(rng.integers(1, 8))
        # First year = the first two genuine EWS activations (Oct + Mar), i.e.
        # periods 1 and 2. In the schema first_year arrives precomputed from
        # preprocessing; this only mirrors it. Written between(1, 2) rather than
        # <= 2 so the period-0 sentinel is not counted as "observed in the first
        # year" — it makes no difference to any model, since those rows are
        # dropped by load_analysis_df(), but it keeps the column honest for
        # descriptive work loaded with keep_never_evaluated=True.
        first_year        = int(1 <= period_idx <= 2)

        # last_grade: mutually exclusive binary flags; all-zero = no grade yet
        _p_no_grade = 1.0 if n_courses_assessed == 0 else max(0.0, 0.25 - period_idx * 0.04)
        if rng.random() < _p_no_grade:
            lg_num_fail = lg_noshow = lg_letter = lg_low = lg_high = 0
        else:
            _lg_probs = SETTINGS["last_grade_p"][drop]
            _lg_cat = int(rng.choice(6, p=_lg_probs))
            lg_num_fail = int(_lg_cat == 1)
            lg_noshow   = int(_lg_cat == 2)
            lg_letter   = int(_lg_cat == 3)
            lg_low      = int(_lg_cat == 4)
            lg_high     = int(_lg_cat == 5)

        if period_idx == 0:
            # No activation to measure from: the window closes at the
            # enrollment's own end, and the next-EWS horizons are undefined.
            time_to_end = 0
            next_ews    = per_end
            _ay_ref     = per_end
        else:
            time_to_end = (end - ews_dt).days
            next_ews_candidates = [d for d in ews_calendar if d > ews_dt]
            next_ews = (next_ews_candidates[0] if next_ews_candidates
                        else ews_dt + pd.Timedelta(days=183))
            _ay_ref  = ews_dt

        rows.append({
            "enrollment":                        i + 1,
            "enrollment_period_id":              f"{i + 1}_{period_idx}",
            "person_id":                         person_id[i],
            "start_year":                        int(start_years[i]),
            # Descriptive only. start_year remains the train/test split variable.
            # Falls back to the window's closing date for the period-0 sentinel,
            # which has no ews_date to derive from.
            "academic_year":                     int(_ay_ref.year if _ay_ref.month >= 8
                                                     else _ay_ref.year - 1),
            "spring_start":                      int(spring_start[i]),
            "ews_date":                          ews_dt,
            "ews_month":                         ews_mon,
            "study_start":                       start,
            "original_study_start":              start,
            "period_start":                      start,
            "period_end":                        next_ews,
            "status_change_date":                       end,
            "current_status":                  current_status[i],
            "period":                            period_idx,
            "days_observed":                     int(days_obs),
            "passed_ects":                       round(passed,  1),
            "failed_ects":                       round(failed,  1),
            "ects_sum":                          ects_sum,
            "ects_behind":                       ects_behind,
            "n_courses_any":                     n_courses,
            "n_courses_assessed":                n_courses_assessed,
            "n_courses_numeric":                 n_courses_numeric,
            "n_courses_current_semester":        n_courses_current,
            "n_passed":                          n_passed,
            "n_failed":                          n_failed,
            "n_courses_ever_failed":             n_courses_ever_failed,
            "n_courses_with_failed_record":      n_courses_with_failed_record,
            "pass_rate_any":                     pass_rate,
            "avg_grade_numeric_ects":            avg_grade_ects,
            "avg_grade_numeric":                 avg_grade_numeric,
            "last_grade_numeric_fail":           lg_num_fail,
            "last_grade_noshow":                 lg_noshow,
            "last_grade_letter_pass":            lg_letter,
            "last_grade_low":                    lg_low,
            "last_grade_high":                   lg_high,
            "undergraduate":                     int(undergraduate[i]),
            "quota_student":                     int(quota_student[i]),
            "secondary_ed_gpa":                  round(float(secondary_gpa[i]), 2),
            "dropped_out_before":                int(dropped_out_before[i]),
            "finished_degree_before":            int(finished_before[i]),
            "first_year":                        first_year,
            "admission_type":                admission_type[i],
            "enrollment_code":                 enrollment_code[i],
            "admission_quota":                             float(admission_quota[i]) if not np.isnan(admission_quota[i]) else None,
            "admission_gpa":                          float(admission_gpa[i]) if not np.isnan(admission_gpa[i]) else None,
            "admission_exam":                       admission_exam[i],
            "faculty_code":                           faculty_code[i],
            "department_code":                          department_code[i],
            "department_short_code":                     department_short_code[i],
            "degree_level":                     degree_level[i],
            "programme_code":                        programme_code[i],
            "page_view_ratio_avg_period":        pv if _has_lms else np.nan,
            "participation_ratio_avg_period":    part if _has_lms else np.nan,
            "missing_pct_avg_period":            miss_pct,
            "late_pct_avg_period":               late_pct,
            "late_or_missing_pct_avg_period":    late_or_missing,
            "lms_observed":                      lms_observed,
            "female":                            int(female[i]),
            "sex_code":                              "F" if female[i] == 1 else "M",
            "danish_nationality":                float(danish[i]),
            "nordic_nationality":                float(nordic_nationality[i]),
            "fee_exempt_nationality":            float(fee_exempt_nationality[i]),
            "nationality_raw":                   nationality_raw[i],
            "nationality":                       nationality[i],
            "age_at_study_start":                round(float(age[i]), 1),
            "required_ects_undergrad":           float(required_ects_undergrad[i]),
            "required_ects_grad":                float(required_ects_grad[i]),
            "required_ects":                     float(required_ects[i]),
            "dropout":                           drop,
            "dropout_orig":                      drop,
            "dropout_next_year":                 int(drop and time_to_end <= 365),
            "dropout_next_semester":             int(drop and end < next_ews),
        })

df = pd.DataFrame(rows)

# Building from row dicts stores scalars, so the ordered dtype has to be
# re-applied. Downstream code must not test `nationality` against raw country
# codes — use nationality_raw for those.
df["nationality"] = pd.Categorical(
    df["nationality"],
    categories=["DK", "Nordic", "EU_EEA_CH", "Other"],
    ordered=True,
)

print(f"Rows:          {len(df):,}")
print(f"Enrollments:   {df['enrollment'].nunique():,}")
print(f"Dropout rate:  {df.groupby('enrollment')['dropout'].max().mean():.1%}")
print(f"Start years:   {df['start_year'].min()}–{df['start_year'].max()}")
print(f"Periods/enr:   {df.groupby('enrollment').size().describe()[['mean','min','max']].to_dict()}")
print(f"ects_behind>=60 rows: {(df['ects_behind'] >= 60).sum():,}")
print(f"Columns:       {len(df.columns)}")

print("\n— consistency checks —")
_sent = df[df["period"] == 0]
print(f"period range: {df['period'].min()}–{df['period'].max()} "
      f"(1-indexed activations; 0 = 'never evaluated' sentinel)")
print(f"period-0 sentinel rows: {len(_sent):,} ({_sent['enrollment'].nunique():,} enrollments) | "
      f"ews_date all NaT: {_sent['ews_date'].isna().all()} | "
      f"ews_month all None: {_sent['ews_month'].isna().all()}")
print(f"mean passed_ects @ period 1: {df.loc[df['period'] == 1, 'passed_ects'].mean():.2f}")
_pr_null = df.groupby("period")["pass_rate_any"].apply(lambda s: s.isna().mean())
print("pass_rate_any null share by period:")
print("  " + ", ".join(f"{p}: {v:.1%}" for p, v in _pr_null.head(4).items()))
print(f"first_year share: {df['first_year'].mean():.1%} "
      f"(periods {sorted(df.loc[df['first_year'] == 1, 'period'].unique())} — must exclude 0)")
print(f"nationality categories: {list(df['nationality'].cat.categories)} "
      f"(ordered={df['nationality'].cat.ordered}, null={df['nationality'].isna().mean():.1%})")
print(f"lms_observed share: {df['lms_observed'].mean():.1%}")
print(f"late+missing == late_or_missing: "
      f"{np.allclose((df['late_pct_avg_period'] + df['missing_pct_avg_period']).dropna(), df['late_or_missing_pct_avg_period'].dropna())}")
print(f"FAC_C share of rows: {(df['faculty_code'] == 'FAC_C').mean():.1%}")
_ba = df[(df['degree_level'] == 'BA')].groupby('enrollment')['passed_ects'].max().median()
_ma = df[(df['degree_level'] == 'MA')].groupby('enrollment')['passed_ects'].max().median()
print(f"median max passed_ects — BA: {_ba:.0f} (statutory 180) | MA: {_ma:.0f} (statutory 120)")

out = Path(__file__).parent / "fake_df.parquet"
df.to_parquet(out, index=False)
print(f"\nSaved → {out}")
