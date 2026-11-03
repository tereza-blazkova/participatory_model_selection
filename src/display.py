"""
What the paper calls each model and feature set.

Internal keys (`mlp`, `c_plus_enrolment`, ...) name files, configs and columns
and never change; these are the words a reader sees. A caption or label that
prints a key instead — `\\texttt{mlp}`, `d\\_plus\\_lms` — is a leak;
figures/scripts/utils.save() refuses to write a figure that draws one.

Feature names likewise (feature_label): the columns keep their snake_case
names, the figures print words.

The paper writes American English ("enrollment"); the keys keep the British
spelling they were created with, because renaming them would orphan every saved
result, model file and workshop_config.json.
"""

MODEL_LABELS = {
    "credits_policy":      "Credits policy",
    "fasterrisk":          "FasterRisk",
    "logistic_regression": "Logistic regression",
    "xgboost":             "XGBoost",
    "mlp":                 "Neural network",
    "contact_fac_a":        "Operational rule-based model",
    "contact_fac_b":         "Operational rule-based model",
}

FEATURE_SET_LABELS = {
    "ects_behind_ge_60":      "Rule: ≥60 ECTS behind",
    "a_credits":              "(a) Credits",
    "b_credits_performance":  "(b) + Performance",
    "c_plus_enrolment":       "(c) + Enrollment",
    "d_plus_lms":             "(d) + LMS",
    "e_plus_sensitive":       "(e) + Sensitive",
    ## Audit-only: sensitive attributes on the accepted credits + performance
    ## set. Not a rung of the ladder, hence no letter of its own.
    "b_plus_sensitive":       "(b) + Sensitive",
    "used_vars":              "Experimental (used\\_vars)",
    "contact":                "Faculty pilot (contact column)",
}

## Internal keys that must never reach printed text (see the module docstring).
## used_vars is not here: the paper names that set "used vars" on purpose.
INTERNAL_KEYS = (
    "mlp", "a_credits", "b_credits_performance", "c_plus_enrolment",
    "d_plus_lms", "e_plus_sensitive", "b_plus_sensitive",
    "logistic_regression", "credits_policy", "ects_behind_ge_60",
)


def model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model.replace("_", " "))


def feature_set_label(feature_set: str) -> str:
    return FEATURE_SET_LABELS.get(feature_set, feature_set.replace("_", " "))


# ── Features ──────────────────────────────────────────────────────────────────

import re as _re

FEATURE_LABELS = {
    "passed_ects":                    "Passed ECTS",
    "failed_ects":                    "Failed ECTS",
    "ects_behind":                    "ECTS behind schedule",
    "n_courses_any":                  "N courses (any)",
    "n_courses_assessed":             "N courses assessed",
    "n_courses_numeric":              "N courses with a numeric grade",
    "n_passed":                       "N courses passed",
    "n_failed":                       "N courses failed",
    "n_courses_ever_failed":          "N courses ever failed",
    "pass_rate_any":                  "Pass rate",
    "avg_grade_numeric_ects":         "Avg grade",
    "n_courses_current_semester":     "N courses this semester",
    "last_grade_numeric_fail":        "Last grade: fail",
    "last_grade_noshow":              "Last grade: no-show",
    "last_grade_letter_pass":         "Last grade: letter pass",
    "last_grade_low":                 "Last grade: low",
    "last_grade_high":                "Last grade: high",
    "undergraduate":                  "Undergraduate",
    "quota_student":                  "Quota admission",
    "secondary_ed_gpa":               "Secondary GPA",
    "secondary_ed_gpa_z_programme":   "Relative secondary GPA",
    "dropped_out_before":             "Dropped out before",
    "finished_degree_before":         "Finished degree before",
    "period":                         "Semester number",
    "days_observed":                  "Days observed",
    "admission_type":             "Admission type",
    "faculty_code":                        "Faculty",
    "department_code":                       "Department",
    "programme_code":                     "Programme",
    "programme_level":                "Programme",
    "page_view_ratio_avg_period":     "Canvas page views (z)",
    "participation_ratio_avg_period": "Canvas participation (z)",
    "missing_pct_avg_period":         "Missing submissions (z)",
    "late_pct_avg_period":            "Late submissions (z)",
    "late_or_missing_pct_avg_period": "Late or missing submissions (z)",
    "lms_observed":                   "Any Canvas activity",
    "female":                         "Female",
    "age_at_study_start":             "Age at enrollment",
    "nationality":                    "Nationality",
    "first_year":                     "First-year student",
}

## Categorical columns whose LEVELS must not be printed: a department or
## programme names a unit, and the unit names its faculty — which no output
## may (the pilot faculties are only ever Faculty A / B). Their dummies are
## numbered instead, in a stable order within one figure.
_ANONYMISED_LEVELS = ("faculty_code", "department_code", "programme_level", "programme_code")
_SHOWN_LEVELS = ("admission_type", "nationality")
_LEVEL_LABELS = {"DK": "Danish", "EU_EEA_CH": "EU/EEA/CH"}
## Optional printed names for admission types. Keyed case- and space-insensitively.
_ADMISSION_LEVEL_LABELS = {
}
_PREFIXES = ("num__", "cat__", "bin__", "remainder__", "passthrough__")


def _strip_prefix(name: str) -> str:
    for prefix in _PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def feature_labels(names) -> list[str]:
    """
    What a figure prints for each model input, in the same order.

    Handles the transformer prefixes (num__, cat__), missing-value indicators
    (missingindicator_X), a scorecard's cut ("X > 0.7"),
    and one-hot dummies — levels of _ANONYMISED_LEVELS become "Department 1",
    "Department 2", ... numbered by their order in `names`.
    """
    names = list(names)
    numbering: dict[tuple[str, str], int] = {}

    def one(raw: str) -> str:
        name = _strip_prefix(raw)
        cut = _re.match(r"^(.+?) ([>≥<≤]) (.+)$", name)
        if cut:
            return f"{one(cut.group(1))} {cut.group(2)} {cut.group(3)}"
        if name.startswith("missingindicator_"):
            return f"{one(name[len('missingindicator_'):])} missing"
        if name in FEATURE_LABELS:
            return FEATURE_LABELS[name]
        for col in sorted(_ANONYMISED_LEVELS + _SHOWN_LEVELS, key=len, reverse=True):
            if name.startswith(col + "_"):
                level = name[len(col) + 1:]
                base = FEATURE_LABELS[col]
                if "infrequent" in level:
                    return f"{base}: other (pooled)"
                if col in _SHOWN_LEVELS:
                    english = _ADMISSION_LEVEL_LABELS.get(" ".join(level.lower().split()))
                    if col == "admission_type" and english:
                        return english
                    return f"{base}: {_LEVEL_LABELS.get(level, level)}"
                key = (base, level)
                if key not in numbering:
                    numbering[key] = sum(1 for b, _ in numbering if b == base) + 1
                return f"{base} {numbering[key]}"
        return name.replace("_", " ")

    return [one(n) for n in names]


def feature_label(name: str) -> str:
    """One input's printed name (see feature_labels; a lone dummy is numbered 1)."""
    return feature_labels([name])[0]
