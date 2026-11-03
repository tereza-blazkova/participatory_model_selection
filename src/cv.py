"""
Cross-validation folds for the hyperparameter search.

The tuning objective decides which configuration ships, so it has to measure the
thing the paper evaluates. The paper's split is on `start_year` — train
2016-2018, test 2019-2023 — precisely because a student's outcome is a property
of their cohort and an EWS-date split would leak. `GroupKFold(groups=enrollment)`
holds out *students* drawn from the *same cohorts* it trains on, so it can only
measure within-cohort generalisation.

Such an objective rewards a model that fits `department_code`/`programme_code` structure
specific to the training cohorts, and steers the search towards less
regularisation and more training. Flexible models, which can express
cohort-by-feature interactions at high resolution, are the most exposed.

`scheme="year"` holds out a whole cohort per fold, so fitting structure specific
to one training cohort costs CV score. Since `start_year` is a property of the enrollment, splitting by
year is automatically enrollment-clean: no student can appear on both sides.
"""
import numpy as np
from sklearn.model_selection import GroupKFold


def make_splits(
    n_rows: int,
    groups: np.ndarray,
    years: np.ndarray | None,
    n_splits: int,
    scheme: str = "year",
) -> list:
    """
    Return a list of (train_idx, val_idx) pairs.

    scheme="year"  leave-one-cohort-out over the distinct values in `years`.
                   The fold count is set by the data, not by n_splits — three
                   training years means three folds — so n_splits is ignored and
                   the caller is told.
    scheme="group" the previous behaviour: GroupKFold over `groups`. Kept so the
                   two objectives can be compared, and as the fallback when
                   `years` is unavailable or degenerate.

    Falls back to "group" with a printed warning rather than raising, so a caller
    that cannot supply years still runs — but never silently: a run whose
    objective quietly changed is exactly what this module exists to prevent.
    """
    if scheme not in ("year", "group"):
        ## This module exists to stop the objective changing without anyone
        ## noticing, so a scheme it does not recognise cannot be the one case
        ## that slips through to the fallback quietly.
        print(f"  WARNING: unknown cv scheme {scheme!r} (expected 'year' or "
              f"'group') — falling back to GroupKFold over enrollments.", flush=True)

    if scheme == "year":
        if years is None:
            print("  WARNING: cv_scheme='year' but no start_year supplied — "
                  "falling back to GroupKFold over enrollments.", flush=True)
        else:
            distinct = np.unique(years)
            if len(distinct) < 2:
                print(f"  WARNING: cv_scheme='year' needs at least 2 cohorts, found "
                      f"{len(distinct)} — falling back to GroupKFold.", flush=True)
            else:
                print(f"  CV: leave-one-cohort-out over {len(distinct)} training "
                      f"years {list(distinct)} (n_splits ignored)", flush=True)
                return [(np.where(years != y)[0], np.where(years == y)[0])
                        for y in distinct]

    gkf = GroupKFold(n_splits=n_splits)
    print(f"  CV: GroupKFold over enrollments, {n_splits} folds "
          f"(within-cohort — cannot see cohort-specific fitting)", flush=True)
    return list(gkf.split(np.zeros(n_rows), np.zeros(n_rows), groups=groups))


def trials_frame(study):
    """
    The search's whole trial history, not just the winner.

    `cv_best_value` records what the winning configuration scored. It cannot say
    what the search *explored*, and those are different questions whenever a
    tuned parameter comes back at a surprising value. A parameter that settles
    low could mean either of two things —

      the search TRIED high values and CV scored them worse
          -> the objective itself prefers the low value
      the search NEVER sampled high values
          -> the sampler, the pruner or the trial budget is the constraint

    Only the trial history separates them. Pruned trials are kept, with a null
    value: "tried and killed after two folds" is a different fact from "never
    sampled", and MedianPruner(n_startup_trials=5, n_warmup_steps=1) can end a
    trial on 2 of only 3 year-folds.

    Written by evaluation.py to {model}__{feature_set}_trials.csv. Returns None
    rather than raising if the Optuna version does not support the call — this
    is a diagnostic, and it must never be the reason a tuning job dies.
    """
    try:
        return study.trials_dataframe(
            attrs=("number", "value", "state", "params", "duration"))
    except Exception as exc:   # noqa: BLE001 — diagnostic only, never fatal
        print(f"  NOTE: could not export trial history ({exc}); "
              f"cv_best_value is unaffected.", flush=True)
        return None
