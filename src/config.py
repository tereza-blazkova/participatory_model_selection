RANDOM_SEED = 42   # single source of truth for all model random states and Optuna seeds

# Device the PyTorch MLP runs on, for BOTH the Optuna search and the final fit.
# "cpu" | "cuda" | "auto"  (env var EWS_MLP_DEVICE overrides this).
#
# Pinned rather than auto-detected: CPU and GPU give different results even with
# deterministic kernels, so "auto" makes a reported number depend on which
# hardware happened to be free. Whichever you pick, keep it fixed for a whole
# round of results — search and final fit use the same device, so the model that
# ships is the one the search actually scored.
#
# Under "cuda" the MLP search runs serially (n_jobs is forced to 1 for the MLP
# only): parallel Optuna trials cannot share one CUDA context. The other three
# models still tune in parallel. A parallel search is not reproducible anyway,
# so serialising the MLP costs less than it looks.
#
# On a machine with no GPU this falls back to CPU with a warning.
MLP_DEVICE = "cuda"

COURSE_CREDITS = ["passed_ects", "failed_ects", "ects_behind"]

COURSE_PERFORMANCE = [
    "n_courses_any", "n_courses_assessed", "n_passed", "n_failed",
    "n_courses_ever_failed",
    "pass_rate_any", "avg_grade_numeric_ects",
    "n_courses_current_semester",
    "last_grade_numeric_fail", "last_grade_noshow", "last_grade_letter_pass",
    "last_grade_low", "last_grade_high",
]
# n_courses_with_failed_record is deliberately NOT in COURSE_PERFORMANCE. It is
# n_courses_ever_failed - n_failed, i.e. a linear combination of two features
# already in the set.

ENROLMENT = [
    "undergraduate", "quota_student", "secondary_ed_gpa",
    "secondary_ed_gpa_z_programme",
    "dropped_out_before", "finished_degree_before",
    "period", "days_observed", "admission_type",
    "faculty_code", "department_code", "programme_level",
]
# The study programme enters at three levels of one hierarchy: faculty
# (faculty_code) > department (department_code) > programme
# (programme_level). The coarse level is there for the programmes the fine one serves
# badly — too rare to estimate, or new/renamed in a test year — following
# Nielsen et al.: ch. 2 gives the tabular models the ISCED broad group next to
# the programme ID, ch. 4 a broad subject area next to the qualification.
#
# programme_level is the programme code split by degree level ("<code>|BA",
# "<code>|MA"), derived in src/data.py. A code can be shared by a subject's
# bachelor's and master's; as one column they get one
# programme effect, and an additive model cannot tell them apart. Raw
# `programme_code` stays in the data for descriptive use, never as a feature.
#
# secondary_ed_gpa_z_programme is the GPA as a z-score among the enrollment's
# own programme cohort (same programme_level, same start_year), derived in
# src/data.py. The same GPA means different things in programmes with different
# admission cut-offs; without this, "weak for this programme" is a GPA x
# programme interaction to be learned from sparse dummies, which is the fitting
# the MLP does worst. Modelled on ch. 4's national tests "z-standardized within
# each year and course", and ch. 2's per-programme GPA cut-off. See
# GPA_REF_MIN_ENROLLMENTS.

# late_pct and missing_pct are used as a PAIR; late_or_missing_pct is their sum
# by construction and is excluded. Including all three lets the weight split
# arbitrarily between them, which is worst in a scorecard a person reads
# directly. They are also different behaviours — submitting late signals
# struggling but engaged, not submitting signals disengagement — and an adviser
# would act on them differently. late_or_missing_pct_avg_period stays in the
# data as the alternative single-variable version; do not add it back here.
LMS = [
    "page_view_ratio_avg_period",
    "participation_ratio_avg_period",
    "missing_pct_avg_period",
    "late_pct_avg_period",
    "lms_observed",
]

SENSITIVE = [
    "female",
    "age_at_study_start", "nationality",
]

USED_VARS = [
    "first_year",                       # first-year student flag
    "ects_behind",                      # ECTS deficit vs prescribed schedule
    "n_courses_current_semester",       # proxy for registered exams this semester
    "page_view_ratio_avg_period",       # Canvas activity — page views
    "participation_ratio_avg_period",   # Canvas activity — participation
    "missing_pct_avg_period",           # Canvas: share of missing submissions
    "late_pct_avg_period",              # Canvas: share of late submissions
    "last_grade_numeric_fail",          # latest grade: fail (00 or −3)
    "last_grade_noshow",                # latest grade: no-show
]

DATA_CONFIGS = {
    "a_credits":             COURSE_CREDITS,
    "b_credits_performance": COURSE_CREDITS + COURSE_PERFORMANCE,
    "c_plus_enrolment":      COURSE_CREDITS + COURSE_PERFORMANCE + ENROLMENT,
    "d_plus_lms":            COURSE_CREDITS + COURSE_PERFORMANCE + ENROLMENT + LMS,
    "e_plus_sensitive":      COURSE_CREDITS + COURSE_PERFORMANCE + ENROLMENT + LMS + SENSITIVE,
    # Placeholder for variables under evaluation; not part of the main ladder.
    "used_vars":             USED_VARS,
    # AUDIT-ONLY. The workshop accepted sensitive attributes only if they gave a
    # large identification gain over the largest ACCEPTED non-sensitive set, or
    # demonstrably improved fairness. The accepted set is credits + performance
    # (enrolment was rejected), so this is the comparison participants actually
    # approved — and the five-rung ladder, which adds sensitive data only on top
    # of LMS, cannot test it. Never a selection candidate, never on a main-text
    # ladder figure: see AUDIT_ONLY_CONFIGS. Last, so the --config-index of the
    # configs above keeps its meaning.
    "b_plus_sensitive":      COURSE_CREDITS + COURSE_PERFORMANCE + SENSITIVE,
}

# Trained and evaluated like every config, but outside the selection and the
# ladder: workshop_rules only ever considers its fs_ladder, and fa_frontier
# drops these from the achievable set. Reported in the SI sensitive-attribute
# audit (src/make_sensitive_audit_table.py) and nowhere else in the paper.
AUDIT_ONLY_CONFIGS = frozenset({"b_plus_sensitive"})

# The models the pipeline trains, one per explainability level from "high" down.
# evaluation.MODELS is the same list.
TRAINED_MODELS = ("fasterrisk", "logistic_regression", "xgboost", "mlp")
# Every `model` value a result file may hold. Anything else is removed from
# results/ at the start of a run (src/prune_results.py).
RESULT_MODELS = frozenset(TRAINED_MODELS) | {"credits_policy"}

CATEGORICAL_FEATURES = ["admission_type", "faculty_code", "department_code", "programme_level", "nationality"]

# A programme cohort (programme_code x start_year) needs at least this many
# enrollments with a secondary_ed_gpa before its mean and sd are used as the
# reference for secondary_ed_gpa_z_programme. Below it the z-score is NaN, which
# the preprocessing turns into "mean + missing indicator" — the model is told the
# cohort was too small to compare against, rather than handed a z-score computed
# from three classmates. Counted in ENROLLMENTS, not rows.
GPA_REF_MIN_ENROLLMENTS = 10

# ── One-hot encoding of the categorical block ─────────────────────────────────
# min_frequency=N collapses every level with fewer than N training rows into a
# single "infrequent" column. Set it in *rows*, not enrollments — the encoder
# counts rows, and there are several EWS rows per enrollment. The encoder is fit
# on the training years only.
#
# Why a threshold: a rare level's own column would be estimated from very few
# students. Why pooling costs little: a pooled programme still carries its
# department and faculty columns.
#
# Precedent for the size: Nielsen et al. pool rare levels too — ch. 2 maps
# strings seen fewer than 250 times in training to [UNK], ch. 4 gives a course
# its own column only above 25,000 occurrences and pools the rest.
#
# Turning it on changes every fitted model. No TRAINING_CODE_VERSION bump is
# needed: load_model_checked() compares each saved encoder's min_frequency and
# handle_unknown with the values here and refuses a mismatch.
OHE_MIN_FREQUENCY = 300

# "infrequent_if_exist": a level unseen in training maps to the column's
# "infrequent" column when one exists, i.e. once OHE_MIN_FREQUENCY has pooled at
# least one rare level of that column. Otherwise it is an all-zero row, exactly
# as under "ignore" — which is the case for a column whose levels are all above
# the threshold, e.g. faculty_code.
#
# Why it matters most for the MLP. In training every row has exactly one 1 in
# each categorical block, so an all-zero block — a programme new or renamed in
# 2019-2023 — is an input the network never saw, and its output there is an
# unconstrained extrapolation. For a tree it is just "not programme X" at every
# split, and for L2-penalised LR it is roughly the average programme. The
# infrequent column was fitted on (rare) training programmes, so it is a trained
# answer to "a programme we cannot say much about". Nielsen et al. ch. 2 maps
# unseen strings to [UNK] for the same reason.
OHE_HANDLE_UNKNOWN = "infrequent_if_exist"

# ── Columns that must never enter a feature list ──────────────────────────────
# Not "not currently used" — structurally unusable. Enforced by
# _assert_no_blocked_features() below, called from evaluation.py at startup so a
# later edit to DATA_CONFIGS fails loudly instead of silently training on them.
#
#   spring_start   February vs September intake. Descriptive only, never a
#                  feature.
#   study_start    Perfectly separable between train (2016–2018) and test
#                  (2019–2023): tree models cannot extrapolate, so every test
#                  row lands in the late-2018 leaf.
#   academic_year  Same separability problem; start_year is the split variable.
#   nationality_raw High-cardinality country code; the modelling variable is the
#                  4-level `nationality` collapse.
BLOCKED_FEATURES = frozenset({
    "spring_start", "academic_year", "study_start", "original_study_start",
    "nationality_raw",
})


def _assert_no_blocked_features(configs: dict = None) -> None:
    """Raise if any feature set contains a structurally unusable column."""
    configs = DATA_CONFIGS if configs is None else configs
    offenders = {
        name: sorted(set(feats) & BLOCKED_FEATURES)
        for name, feats in configs.items()
        if set(feats) & BLOCKED_FEATURES
    }
    if offenders:
        raise ValueError(
            "Blocked columns present in feature sets — see BLOCKED_FEATURES in "
            f"src/config.py for why each is excluded: {offenders}"
        )


# ── Nationality ───────────────────────────────────────────────────────────────
# `nationality` is an ordered categorical DK < Nordic < EU_EEA_CH < Other,
# derived from national registry nationality codes (vehicle-registration style,
# some ISO-3), NOT ISO-2. The reference level is "DK". Unknown codes are
# NaN, not "Other".
DANISH_CATEGORY    = "DK"
NATIONALITY_LEVELS = ["DK", "Nordic", "EU_EEA_CH", "Other"]

# ── Analysis population ───────────────────────────────────────────────────────
# Faculties excluded from the analysis population.
EXCLUDED_FACULTIES = ["FAC_C"]

# Faculties whose contact flag records who was actually contacted in the October
# 2023 pilot. Only Faculty A's does. It is not known who Faculty B contacted, and
# contact_fac_b is also set at other faculties (tools/check_contact_flags.py). So its operational model is judged by ranking
# only (AUC, ABROCA on points_total): no recall, precision or contact count.
CONTACTS_RECORDED_FACULTIES = frozenset({"FAC_A"})

TARGET       = "dropout"
# Objective for the hyperparameter search. "year" holds out a whole cohort per
# fold, matching the paper's train/test split; "group" is the alternative
# GroupKFold over enrollments, which only measures within-cohort generalisation
# (see src/cv.py).
CV_SCHEME    = "year"

TRAIN_YEARS  = (2016, 2018)
TEST_YEARS   = list(range(2019, 2024))
# Cohorts the dropout-timing figure pools: the test cohorts without 2023, which
# has had at most a year to drop out and would pull a timing curve early.
DROPOUT_TIMING_YEARS = [y for y in TEST_YEARS if y != 2023]
MAIN_TEST_YEAR = 2021

# The cohort the workshop rules are applied to. The main text selects AND
# evaluates on MAIN_TEST_YEAR; only that selection is "the Participatory model"
# (results/models/workshop_config.json). SI_SELECTION_YEAR is the SI robustness
# version: the endline rules applied to 2019, the pick then evaluated on
# MAIN_TEST_YEAR. Its files carry `_sel{year}` and are never read as the
# Participatory model.
SELECTION_YEAR    = MAIN_TEST_YEAR
SI_SELECTION_YEAR = 2019
# Cohorts the by-cohort selection table reports: the settled test cohorts.
SELECTION_BY_COHORT_YEARS = [y for y in TEST_YEARS if y != 2023]

# Max ML: fixed to the most complex model with the most data — never selected by AUC.
MAX_ML_MODEL = "mlp"
MAX_ML_FS    = "e_plus_sensitive"
ECTS_THRESHOLD = 60

# The points total above which data/generate_pilot_snapshot.py marks a synthetic student
# as contacted. An arbitrary value for the synthetic data; set your own with real data.
PILOT_CONTACT_THRESHOLD = 7

# Statistical disclosure control: any number or plotted element resting on
# fewer than this many students (enrollments) is not shown — tables print the
# marker in its place, figures leave the element out and mark it. So
# 1..MIN_CELL_SIZE-1 is suppressed and MIN_CELL_SIZE is the smallest count that
# may appear. A true zero is shown: "0 of 40" is a statement about 40 students.
# Every guard compares `n < MIN_CELL_SIZE`; the shared helpers are in
# src/disclosure.py.
#
# The value follows the research environment's rules for publishing analysis
# results: "minimum 3 observations per table cell", and a graph must satisfy
# the same rule if converted to a table.
#
# Nothing downstream spells the threshold out: cell markers come from
# disclosure.SUPPRESSED / SUPPRESSED_TEX and figure notes from
# disclosure.note_text(), so a change here reaches all of them. Table captions
# carry no explanatory sentence. Do not write the numeral into a caption, a
# docstring or a test expectation.
MIN_CELL_SIZE = 3