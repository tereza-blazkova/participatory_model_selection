"""
Evaluation orchestrator: 4 models × N feature sets × test years 2019–2023.
credits_policy baseline evaluated separately.

Run from project root:
    python src/evaluation.py                           # default hyperparameters, all configs
    python src/evaluation.py --tune                    # Optuna (TPE) + leave-one-cohort-out CV (slow)
    python src/evaluation.py --tune --n-iter 20        # fewer trials for a quick test
    python src/evaluation.py --config-index 2          # tune/train only DATA_CONFIGS[2]; saves
                                                       #   partial results to results/partial/<name>/
    python src/evaluation.py --n-jobs N                # parallel Optuna trials — NOT
                                                       #   reproducible (trials reach the TPE
                                                       #   sampler out of order). Default is 1;
                                                       #   XGBoost threads internally there.
    python src/evaluation.py --merge                   # merge partial results into final CSVs
    python src/evaluation.py --use-saved-params        # refit from saved _best_params.json,
                                                       #   no search — use after a --tune run
                                                       #   instead of retraining at defaults
"""
import argparse
import inspect
import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import fasterrisk_model, logistic_regression, mlp_model, xgboost_model
from paths.paths import RESULTS_PATH
from src.config import (
    DATA_CONFIGS,
    MAIN_TEST_YEAR,
    RANDOM_SEED,
    TARGET,
    TEST_YEARS,
    TRAIN_YEARS,
    _assert_no_blocked_features,
)
from src.data import load_analysis_df
from src.prune_results import prune_results
from src.metrics import (
    enrollment_level_metrics,
    row_level_metrics,
    credits_policy_metrics,
    credits_policy_row_level_metrics,
    enrollment_predictions_df,
    credits_policy_predictions_df,
    row_predictions_df,
    _SUBGROUP_COLS,
    _attach_subgroup_cols,
)

MODELS = {
    "fasterrisk":          fasterrisk_model,
    "logistic_regression": logistic_regression,
    "xgboost":             xgboost_model,
    "mlp":                 mlp_model,
}

# ── Degeneracy guards ─────────────────────────────────────────────────────────
# A feature set can silently collapse: columns that are entirely null or constant
# across the training years are dropped by SimpleImputer at fit time (all-null) or
# carry no signal (constant), and nothing in the pipeline says so — the columns
# exist in the dataframe, so the missing-column warning above never fires. When
# enough of them collapse, every model reduces to a monotone function of whatever
# is left. If that remainder is coarse, max-prob-per-enrollment becomes constant
# and enrollment metrics degenerate to recall 1.0 / AUC 0.500 / precision = base
# rate, identically for every model. These checks name the cause at the point it
# happens instead of leaving a suspiciously round 0.500 in the results table.

_MIN_DISTINCT_ENROLLMENT_SCORES = 3

# Bookkeeping keys written alongside the tuned hyperparameters — not model args.
# "device" is deliberately NOT here: mlp_model.train_with_params reads it to pin
# the refit to the device the search ran on.
_PARAM_METADATA_KEYS = frozenset({"tuning_metric", "random_seed", "torch_version", "cv_auc_by_epoch",
                                  "search_n_jobs", "cv_best_value", "cv_n_trials"})


def _supports_groups(fn) -> bool:
    """
    True if `fn` accepts a `groups` argument.

    Only the MLP needs enrollment IDs at fit time (for its best-epoch validation
    split); the other three models' train()/train_with_params() take no such
    argument. Checking the signature keeps this to one call site instead of
    adding an ignored parameter to every model module.
    """
    return "groups" in inspect.signature(fn).parameters


def _purge_stale_model(model_name: str, config_name: str, models_dir: Path) -> None:
    """
    Delete a model's .joblib when this run skipped training it.

    A skip leaves the previous file on disk, and every downstream step loads by
    filename without knowing the run did not produce it. A stale file whose
    feature list differs from the current one fails deep inside a
    ColumnTransformer with a missing-columns error; one whose stale feature list
    is a *superset* of the current one is worse — it transforms without
    complaint and scores on the wrong features.

    Removing the file turns both cases into an obvious FileNotFoundError naming
    the model that did not train.
    """
    path = models_dir / f"{model_name}__{config_name}.joblib"
    if path.exists():
        path.unlink()
        print(f"       removed stale {path.name} (not retrained this run)", flush=True)


def _load_saved_params(model_name: str, config_name: str, models_dir: Path) -> dict | None:
    """
    Load tuned hyperparameters saved by a previous --tune run, if present.

    Retraining without --tune silently rebuilds every model at library defaults
    and overwrites the .joblib files; the _best_params.json files survive because
    they are only written when tuning. This reads them back so a re-run can
    reproduce the tuned models without repeating the search.
    """
    path = models_dir / f"{model_name}__{config_name}_best_params.json"
    if not path.exists():
        return None
    with open(path) as fh:
        params = json.load(fh)
    return {k: v for k, v in params.items() if k not in _PARAM_METADATA_KEYS}


def _curve_rows(model, model_name: str, config_name: str, seed: int) -> list[dict]:
    """
    Per-epoch loss trace, for models that record one.

    Only the MLP does: it is the only model here trained by an explicit epoch
    loop we control. The trace is what says whether the fixed epoch budget is
    enough — a held-out curve still descending at the last epoch means the
    budget binds. Returns [] for models with no such notion, so callers need no
    special-casing.
    """
    clf = getattr(model, "named_steps", {}).get("clf") if hasattr(model, "named_steps") else model
    train_losses = getattr(clf, "train_losses_", None)
    if not train_losses:
        return []
    # val_losses is empty whenever _VAL_FRAC is 0, which it now is: the holdout
    # selected nothing once best-epoch checkpointing was removed, so keeping it
    # only cost 10% of the training rows. Requiring BOTH lists here meant the
    # curves silently stopped being written the moment that changed. Emit the
    # training trace regardless and leave val_loss empty; the reader has to say
    # what it can and cannot show, rather than the file quietly not existing.
    val_losses = getattr(clf, "val_losses_", None) or []
    return [
        {
            "model":       model_name,
            "feature_set": config_name,
            "seed":        seed,
            "epoch":       i + 1,
            "train_loss":  round(float(tr), 6),
            "val_loss":    round(float(val_losses[i]), 6) if i < len(val_losses) else None,
        }
        for i, tr in enumerate(train_losses)
    ]


def _report_health(health_rows: list[dict], out_dir: Path, strict: bool = False) -> None:
    """
    Summarise every degeneracy found during the run, at the end of the log where
    it is still on screen. With strict=True the run exits non-zero so a pipeline
    stops rather than writing a results table nobody should quote.
    """
    if not health_rows:
        print("\nFeature health: no degenerate columns or prediction cells detected.")
        return

    health_df = pd.DataFrame(health_rows)

    dead_cols   = health_df[health_df["issue"].isin(["all_null", "constant"])]
    absent_cols = health_df[health_df["issue"] == "missing_column"]
    dead_scores = health_df[health_df["issue"] == "degenerate_enrollment_scores"]

    print(f"\n{'!' * 64}")
    print("  FEATURE HEALTH WARNINGS")
    print(f"{'!' * 64}")
    if not absent_cols.empty:
        for fs, grp in absent_cols.groupby("feature_set"):
            cols = ", ".join(grp["column"])
            print(f"  {fs}: {len(grp)} column(s) NOT IN THE DATA, trained without "
                  f"them — {cols}")
        print("     Results are recorded under the full feature-set name even so, and "
              "every\n     downstream load will report these models as stale. Fix the "
              "data or the\n     feature set before reporting.")
    if not dead_cols.empty:
        for fs, grp in dead_cols.groupby("feature_set"):
            cols = ", ".join(grp["column"])
            print(f"  {fs}: {len(grp)} signal-free column(s) — {cols}")
    if not dead_scores.empty:
        affected = sorted(dead_scores["feature_set"].unique())
        print(
            f"  Degenerate enrollment scores in {len(dead_scores)} model x year cell(s), "
            f"feature set(s): {', '.join(affected)}"
        )
        print("  Enrollment-level recall/precision/AUC for those cells are not meaningful.")
    print(f"{'!' * 64}")

    if strict:
        sys.exit(
            "ERROR: --strict set and feature-health warnings were raised "
            "(listed above)."
        )


def _audit_feature_matrix(X_train: pd.DataFrame, config_name: str) -> list[dict]:
    """Report training columns that carry no signal. Returns one row per problem."""
    problems = []
    for col in X_train.columns:
        s = X_train[col]
        null_rate = float(s.isna().mean())
        n_unique  = int(s.nunique(dropna=True))
        if null_rate == 1.0:
            kind = "all_null"
        elif n_unique <= 1:
            kind = "constant"
        else:
            continue
        problems.append({
            "feature_set": config_name, "column": col, "issue": kind,
            "null_rate": round(null_rate, 4), "n_unique": n_unique,
        })

    if problems:
        n_usable = len(X_train.columns) - len(problems)
        print(
            f"  WARNING: {config_name} — {len(problems)} of {len(X_train.columns)} "
            f"columns carry no signal in the training years and are effectively "
            f"dropped ({n_usable} usable):"
        )
        for p in problems:
            print(f"      {p['column']}  ({p['issue']}, null={p['null_rate']:.3f}, "
                  f"n_unique={p['n_unique']})")
        if n_usable <= 1:
            print(
                f"  WARNING: {config_name} — only {n_usable} usable column(s) remain. "
                "Every model will collapse to the same score; enrollment-level "
                "metrics for this feature set will be meaningless."
            )
    return problems


def _audit_predictions(
    preds: pd.DataFrame, model_name: str, config_name: str, year: int
) -> dict | None:
    """Report an enrollment score distribution too coarse to threshold or rank."""
    n_distinct = int(preds["max_prob"].nunique(dropna=True))
    if n_distinct >= _MIN_DISTINCT_ENROLLMENT_SCORES:
        return None
    print(
        f"  WARNING: {model_name}/{config_name}/{year} — only {n_distinct} distinct "
        f"max-prob value(s) across {len(preds)} enrollments. Enrollment-level recall, "
        "precision and AUC are degenerate for this cell; do not report them."
    )
    return {
        "feature_set": config_name, "model": model_name, "test_year": year,
        "issue": "degenerate_enrollment_scores", "n_distinct_max_prob": n_distinct,
        "n_enrollments": int(len(preds)),
    }


# ── Tuning defaults (overridden by CLI args) ───────────────────────────────────
TUNE        = False
## 30 trials, as in Nielsen et al. (2025, ch. 4), whose MLP this project
## follows; they gave their linear model 30 too. Applies to every model.
N_TUNE_ITER = 30
TUNE_CV     = 3

PARTIAL_DIR = RESULTS_PATH / "partial"


def run_all(
    df: pd.DataFrame,
    tune: bool = TUNE,
    n_tune_iter: int = N_TUNE_ITER,
    tune_cv: int = TUNE_CV,
    n_jobs: int = 1,
    config_filter: str | None = None,  # if set, train only this one feature set
    strict: bool = False,              # exit non-zero on feature-health warnings
    use_saved_params: bool = False,    # refit from saved _best_params.json, no search
    n_seeds: int = 1,                  # extra refits at RANDOM_SEED+i, for uncertainty
) -> pd.DataFrame:
    train_df = df[df["start_year"].between(*TRAIN_YEARS)].copy()
    train_pos_rate = float(
        train_df.drop_duplicates("enrollment")[TARGET].mean()
    )

    models_dir = RESULTS_PATH / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Determine output directory for CSVs
    if config_filter:
        out_dir = PARTIAL_DIR / config_filter
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = RESULTS_PATH

    if tune:
        print(f"Hyperparameter tuning ON  — n_iter={n_tune_iter}, cv={tune_cv}, n_jobs={n_jobs}")
    elif use_saved_params:
        print("Hyperparameter tuning OFF — refitting from saved _best_params.json (no search)")
    else:
        print("Hyperparameter tuning OFF — using default parameters")
        saved = sorted(models_dir.glob("*_best_params.json"))
        if saved:
            print(
                f"\n  WARNING: {len(saved)} tuned parameter file(s) exist in {models_dir}\n"
                "  but are being IGNORED — every model will be trained at library defaults,\n"
                "  and the existing .joblib files will be overwritten with those defaults.\n"
                "  Models whose performance depends on tuning may perform differently here.\n"
                "  Pass --use-saved-params to refit the tuned configurations without re-searching,\n"
                "  or --tune to run the search again.\n"
            )

    # Select configs to run
    configs_to_run = {
        name: feats for name, feats in DATA_CONFIGS.items()
        if config_filter is None or name == config_filter
    }

    total_jobs = len(MODELS) * len(configs_to_run)
    job = 0
    wall_start = time.time()
    timing_rows = []
    results = []
    preds_rows = []
    row_preds_rows = []
    health_rows = []   # degenerate columns / prediction cells (see _audit_* above)
    seed_rows   = []   # one row per (model, feature_set, year, SEED) when --seeds > 1
    curve_rows  = []   # per-epoch loss traces for models that record them

    # ── Rule-based baseline (only when running all configs) ───────────────────
    if config_filter is None:
        for yr in TEST_YEARS:
            test_df = df[df["start_year"] == yr].copy()
            if len(test_df) == 0:
                continue
            metrics     = credits_policy_metrics(test_df, target=TARGET)
            row_metrics = credits_policy_row_level_metrics(test_df, target=TARGET)
            results.append({
                "model":       "credits_policy",
                "feature_set": f"ects_behind_ge_{60}",
                "test_year":   yr,
                "is_2023":     yr == 2023,
                **metrics,
                **row_metrics,
            })
            preds = credits_policy_predictions_df(test_df, target=TARGET)
            preds["model"]       = "credits_policy"
            preds["feature_set"] = "ects_behind_ge_60"
            preds["test_year"]   = yr
            preds_rows.append(preds)

    # ── ML models × feature sets ───────────────────────────────────────────────
    for config_name, feature_list in configs_to_run.items():
        available = [f for f in feature_list if f in df.columns]
        missing = sorted(set(feature_list) - set(available))
        if missing:
            # A health row, not just a warning. The run still records this
            # config under its own name, so evaluation_results.csv says
            # "c_plus_enrolment" for a model that was never given the whole of
            # it — and a warning printed an hour into a run is not what anyone
            # reads before quoting a table. It also makes --strict stop the run.
            # Downstream this is the cause of an otherwise baffling error: every
            # plotting call checks the saved model against the FULL feature set
            # from config.py, so a model fitted on a short list is reported as
            # stale, and re-running evaluation.py — which the message suggests —
            # reproduces it exactly.
            print(f"  WARNING: {config_name} — columns not in data, skipping: {missing}")
            health_rows.extend({
                "feature_set": config_name, "column": col, "issue": "missing_column",
                "null_rate": 1.0, "n_unique": 0,
            } for col in missing)
        X_train = train_df[available]
        y_train = train_df[TARGET]
        health_rows.extend(_audit_feature_matrix(X_train, config_name))

        for model_name, model_module in MODELS.items():
            job += 1
            elapsed = time.time() - wall_start
            eta_str = ""
            if job > 1:
                eta_sec = elapsed / (job - 1) * (total_jobs - job + 1)
                eta_str = f"  ETA {eta_sec/60:.1f} min"
            saved_params = (
                _load_saved_params(model_name, config_name, models_dir)
                if use_saved_params and not tune
                else None
            )

            if tune:
                mode_tag = f"  [tuning n_iter={n_tune_iter}]"
            elif saved_params is not None:
                mode_tag = "  [saved params]"
            elif use_saved_params:
                mode_tag = "  [defaults — no saved params found]"
            else:
                mode_tag = ""
            print(
                f"  [{job:2d}/{total_jobs}] {model_name:22s} | {config_name}{mode_tag}{eta_str} ...",
                flush=True,
            )

            t0 = time.time()
            try:
                if tune:
                    if model_name == "fasterrisk":
                        model = model_module.train_tuned(X_train, y_train, available)
                    else:
                        model = model_module.train_tuned(
                            X_train, y_train, available,
                            n_iter=n_tune_iter, cv=tune_cv,
                            groups=train_df["enrollment"],
                            n_jobs=n_jobs,
                            # The cohort each enrollment belongs to. With
                            # CV_SCHEME="year" the search holds out a whole
                            # cohort per fold, so a configuration that only
                            # works within 2016-2018 is scored as such.
                            years=train_df["start_year"],
                        )
                elif saved_params is not None:
                    fit_fn = model_module.train_with_params
                    extra  = ({"groups": train_df["enrollment"]}
                              if _supports_groups(fit_fn) else {})
                    model = fit_fn(X_train, y_train, available, saved_params, **extra)
                else:
                    fit_fn = model_module.train
                    extra  = ({"groups": train_df["enrollment"]}
                              if _supports_groups(fit_fn) else {})
                    model = fit_fn(X_train, y_train, available, **extra)
            except ImportError as exc:
                print(f"  SKIP {model_name}: {exc}", flush=True)
                _purge_stale_model(model_name, config_name, models_dir)
                continue
            train_sec = time.time() - t0

            for yr in TEST_YEARS:
                test_df = df[df["start_year"] == yr].copy()
                if len(test_df) == 0:
                    continue
                X_test = test_df[available]
                y_prob = model.predict_proba(X_test)[:, 1]
                metrics     = enrollment_level_metrics(
                    test_df, y_prob, train_pos_rate, target=TARGET
                )
                row_metrics = row_level_metrics(
                    test_df, y_prob, train_pos_rate, target=TARGET
                )
                results.append({
                    "model":       model_name,
                    "feature_set": config_name,
                    "test_year":   yr,
                    "is_2023":     yr == 2023,
                    **metrics,
                    **row_metrics,
                })
                preds = enrollment_predictions_df(test_df, y_prob, train_pos_rate, target=TARGET)
                degenerate = _audit_predictions(preds, model_name, config_name, yr)
                if degenerate:
                    health_rows.append(degenerate)
                preds["model"]       = model_name
                preds["feature_set"] = config_name
                preds["test_year"]   = yr
                preds_rows.append(preds)

                row_preds = row_predictions_df(
                    test_df, y_prob, model_name, config_name, yr, target=TARGET
                )
                row_preds_rows.append(row_preds)

            # Detach the trial history before pickling. It is a diagnostic
            # artefact, not part of the fitted model, and every .joblib is
            # loaded by load_model_checked on every plotting run.
            trials_df = getattr(model, "cv_trials_", None)
            if trials_df is not None:
                del model.cv_trials_

            model_path = models_dir / f"{model_name}__{config_name}.joblib"
            joblib.dump(model, model_path)

            if tune:
                params_to_save = {
                    **model.best_params_,
                    "tuning_metric": "enrollment_level_roc_auc",
                    "random_seed":   RANDOM_SEED,
                    # The winning trial's CV score, and how many trials it beat.
                    # It separates two explanations for a feature set scoring
                    # below its own subset: the SEARCH also preferred the subset
                    # (structural — more trials will not help), or CV was fine
                    # and only the test year moved (noise). None for
                    # FasterRisk, which runs no search.
                    "cv_best_value": getattr(model, "cv_best_value_", None),
                    "cv_n_trials":   getattr(model, "cv_n_trials_", None),
                }
                params_path = models_dir / f"{model_name}__{config_name}_best_params.json"
                with open(params_path, "w") as fh:
                    json.dump(params_to_save, fh, indent=2, default=str)

            curve_rows.extend(_curve_rows(model, model_name, config_name, RANDOM_SEED))

            # ── seed repeats ──────────────────────────────────────────────────
            # The shipped .joblib stays the RANDOM_SEED fit, so every downstream
            # artefact (workshop config, FA frontier, ABROCA) is unchanged. These
            # extra fits exist only to measure how much of a reported number is
            # the configuration and how much is the draw. They go to their own
            # CSV; evaluation_results.csv keeps its single-fit meaning.
            if n_seeds > 1:
                repeat_params = getattr(model, "best_params_", None) or saved_params or {}
                for offset in range(n_seeds):
                    seed = RANDOM_SEED + offset
                    if offset == 0:
                        seed_model = model      # already fitted at RANDOM_SEED
                    else:
                        fit_fn = model_module.train_with_params
                        extra  = ({"groups": train_df["enrollment"]}
                                  if _supports_groups(fit_fn) else {})
                        seed_model = fit_fn(
                            X_train, y_train, available,
                            {**repeat_params, "random_state": seed}, **extra,
                        )
                        curve_rows.extend(
                            _curve_rows(seed_model, model_name, config_name, seed)
                        )
                    for yr in TEST_YEARS:
                        test_df = df[df["start_year"] == yr]
                        if len(test_df) == 0:
                            continue
                        y_prob = seed_model.predict_proba(test_df[available])[:, 1]
                        seed_rows.append({
                            "model":       model_name,
                            "feature_set": config_name,
                            "test_year":   yr,
                            "seed":        seed,
                            **enrollment_level_metrics(
                                test_df, y_prob, train_pos_rate, target=TARGET),
                            **row_level_metrics(
                                test_df, y_prob, train_pos_rate, target=TARGET),
                        })
                print(f"         + {n_seeds - 1} seed repeat(s)", flush=True)

            print(f"         done in {train_sec:.1f}s", flush=True)
            timing_rows.append({
                "model":        model_name,
                "feature_set":  config_name,
                "train_sec":    round(train_sec, 2),
            })

    total_sec = time.time() - wall_start
    print(f"\nTotal wall time: {total_sec/60:.1f} min")

    out_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / "evaluation_results.csv", index=False)

    if preds_rows:
        preds_df = pd.concat(preds_rows, ignore_index=True)
        preds_df.to_csv(out_dir / "enrollment_predictions.csv", index=False)

    if row_preds_rows:
        row_preds_df = pd.concat(row_preds_rows, ignore_index=True)
        row_preds_df.to_csv(out_dir / "row_predictions.csv", index=False)

    _report_health(health_rows, out_dir, strict=strict)

    if config_filter:
        # Partial metadata so the merge step can reconstruct train_metadata.json
        partial_meta = {
            "train_pos_rate":     train_pos_rate,
            "available_features": {
                config_filter: [f for f in configs_to_run[config_filter] if f in df.columns]
            },
        }
        with open(out_dir / "train_metadata.json", "w") as fh:
            json.dump(partial_meta, fh, indent=2)
        print(f"Partial results saved → {out_dir}/")
    else:
        train_metadata = {
            "train_pos_rate":     train_pos_rate,
            "random_seed":        RANDOM_SEED,
            "available_features": {
                cfg: [f for f in feats if f in df.columns]
                for cfg, feats in DATA_CONFIGS.items()
            },
        }
        meta_path = models_dir / "train_metadata.json"
        with open(meta_path, "w") as fh:
            json.dump(train_metadata, fh, indent=2)
        print(f"Models saved → {models_dir}/")
        print(f"Train metadata saved → {meta_path}")
        print(f"Results saved → {RESULTS_PATH / 'evaluation_results.csv'}")
        print(f"Timings saved → {RESULTS_PATH / 'training_times.csv'}")

    return results_df


def _refresh_subgroup_cols(frames: list[pd.DataFrame], df: pd.DataFrame,
                           what: str) -> pd.DataFrame:
    """
    Concatenate per-config prediction frames and take every subgroup column
    (_SUBGROUP_COLS) from the analysis frame, by enrollment.

    Subgroup columns describe the enrollment, not the model, so they are the
    same whichever run wrote a partial; taking them from the data means a
    partial missing one cannot leave that axis empty for its configs.
    """
    merged = pd.concat(frames, ignore_index=True)
    stale = sorted({c for f in frames for c in _SUBGROUP_COLS
                    if c in df.columns and c not in f.columns})
    if stale:
        print(f"  {what}: {stale} missing from some partials — taken from the data")
    merged = merged.drop(columns=[c for c in _SUBGROUP_COLS if c in merged.columns])
    return _attach_subgroup_cols(merged, df)


def merge_results(df: pd.DataFrame) -> None:
    """
    Collect partial results written by --config-index runs, prepend the
    credits_policy baseline, and write the final evaluation CSVs +
    train_metadata.json expected by the rest of the pipeline.
    """
    if not PARTIAL_DIR.exists() or not any(PARTIAL_DIR.iterdir()):
        print("No partial results found in", PARTIAL_DIR)
        return

    prune_results()

    train_df = df[df["start_year"].between(*TRAIN_YEARS)].copy()
    train_pos_rate = float(train_df.drop_duplicates("enrollment")[TARGET].mean())

    # Credits-policy baseline (fast; run once here rather than in every --config-index run)
    cp_result_rows = []
    preds_rows     = []
    for yr in TEST_YEARS:
        test_df = df[df["start_year"] == yr].copy()
        if len(test_df) == 0:
            continue
        metrics     = credits_policy_metrics(test_df, target=TARGET)
        row_metrics = credits_policy_row_level_metrics(test_df, target=TARGET)
        cp_result_rows.append({
            "model":       "credits_policy",
            "feature_set": f"ects_behind_ge_{60}",
            "test_year":   yr,
            "is_2023":     yr == 2023,
            **metrics,
            **row_metrics,
        })
        preds = credits_policy_predictions_df(test_df, target=TARGET)
        preds["model"]       = "credits_policy"
        preds["feature_set"] = "ects_behind_ge_60"
        preds["test_year"]   = yr
        preds_rows.append(preds)

    # Convert to DataFrames before mixing with pd.read_csv() results below
    results = [pd.DataFrame(cp_result_rows)] if cp_result_rows else []

    timing_rows     = []
    row_preds_rows  = []
    seed_rows       = []
    curve_rows      = []
    all_avail_feats = {}

    config_dirs = sorted(p for p in PARTIAL_DIR.iterdir() if p.is_dir())
    ## Every config must have a partial. Otherwise a merge would quietly write
    ## results holding only the configs that were run, and every table downstream
    ## would lose part of its ladder.
    missing = [name for name in DATA_CONFIGS if not (PARTIAL_DIR / name).is_dir()]
    if missing:
        raise SystemExit(
            f"REFUSING to merge: no partial results for {missing} in {PARTIAL_DIR}.\n"
            f"Run them with --config-index, then merge."
        )
    print(f"Merging {len(config_dirs)} partial result set(s) from {PARTIAL_DIR}")

    for config_dir in config_dirs:
        name = config_dir.name
        for fname, collector in [
            ("evaluation_results.csv",   results),
            ("enrollment_predictions.csv", preds_rows),
            ("row_predictions.csv",      row_preds_rows),
            ("training_times.csv",       timing_rows),
        ]:
            p = config_dir / fname
            if p.exists():
                collector.append(pd.read_csv(p))
            else:
                print(f"  WARNING: missing {p}")

        # Written only when --seeds > 1, so absence is normal and silent. They
        # still have to be merged: without this the --config-index path drops them and
        # the uncertainty numbers exist per-config but never reach results/.
        for fname, collector in [
            ("evaluation_results_seeds.csv", seed_rows),
            ("training_curves.csv",          curve_rows),
        ]:
            p = config_dir / fname
            if p.exists():
                collector.append(pd.read_csv(p))

        meta_p = config_dir / "train_metadata.json"
        if meta_p.exists():
            with open(meta_p) as fh:
                meta = json.load(fh)
            all_avail_feats.update(meta.get("available_features", {}))
        else:
            print(f"  WARNING: missing {meta_p}")

    RESULTS_PATH.mkdir(exist_ok=True)

    pd.concat(results,     ignore_index=True).to_csv(
        RESULTS_PATH / "evaluation_results.csv",    index=False)
    _refresh_subgroup_cols(preds_rows, df, "enrollment predictions").to_csv(
        RESULTS_PATH / "enrollment_predictions.csv", index=False)
    if row_preds_rows:
        _refresh_subgroup_cols(row_preds_rows, df, "row predictions").to_csv(
            RESULTS_PATH / "row_predictions.csv",   index=False)

    models_dir = RESULTS_PATH / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    train_metadata = {
        "train_pos_rate":     train_pos_rate,
        "random_seed":        RANDOM_SEED,
        "available_features": all_avail_feats,
    }
    with open(models_dir / "train_metadata.json", "w") as fh:
        json.dump(train_metadata, fh, indent=2)

    print("Merge complete.")
    print(f"  → {RESULTS_PATH / 'evaluation_results.csv'}")
    print(f"  → {RESULTS_PATH / 'enrollment_predictions.csv'}")
    print(f"  → {models_dir / 'train_metadata.json'}")


def print_main_comparison(results_df: pd.DataFrame, test_year: int = MAIN_TEST_YEAR) -> None:
    data = results_df[results_df["test_year"] == test_year].copy()
    cols = ["model", "feature_set", "n_enrollments", "n_dropout_enrollments",
            "n_flagged_enrollments", "enr_recall", "enr_precision", "early_flag_rate"]
    print(f"\n{'='*80}")
    print(f"Main comparison — test year {test_year}")
    print(f"{'='*80}")
    print(data[cols].sort_values(["model", "feature_set"]).to_string(index=False))


def print_stability(results_df: pd.DataFrame, feature_set: str = "e_plus_sensitive") -> None:
    ml = results_df[
        (results_df["model"] != "credits_policy") &
        (results_df["feature_set"] == feature_set)
    ][["model", "test_year", "enr_recall", "enr_precision"]].copy()

    rule = results_df[results_df["model"] == "credits_policy"][
        ["test_year", "enr_recall", "enr_precision"]
    ].copy()
    rule["model"] = "credits_policy"

    combined = pd.concat([ml, rule], ignore_index=True)
    pivot = combined.pivot_table(
        index="model", columns="test_year",
        values="enr_recall", aggfunc="first"
    )
    print(f"\n{'='*80}")
    print(f"Recall by test year — feature set: {feature_set}  (2023 = incomplete cohort)")
    print(f"{'='*80}")
    print(pivot.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune",         action="store_true",
                        help="Enable hyperparameter search (Optuna TPE + leave-one-cohort-out CV)")
    parser.add_argument("--n-iter",       type=int, default=N_TUNE_ITER,
                        help=f"Search trials per model (default {N_TUNE_ITER})")
    parser.add_argument("--cv",           type=int, default=TUNE_CV,
                        help=f"CV folds during search (default {TUNE_CV})")
    parser.add_argument("--skip",         nargs="+", default=[], metavar="MODEL",
                        help="Model names to skip, e.g. --skip mlp fasterrisk")
    parser.add_argument("--n-jobs",       type=int, default=1,
                        help="Parallel Optuna trials (default 1: reproducible; above 1 the search is not)")
    parser.add_argument("--config-index", type=int, default=None, metavar="I",
                        help="Run only DATA_CONFIGS[I] (0-based); saves to results/partial/<name>/")
    parser.add_argument("--merge",        action="store_true",
                        help="Merge partial results from --config-index runs into final CSVs")
    parser.add_argument("--seeds", type=int, default=1, metavar="N", dest="n_seeds",
                        help="Refit each model at N seeds (RANDOM_SEED..RANDOM_SEED+N-1) and "
                             "write evaluation_results_seeds.csv. The shipped .joblib is always "
                             "the RANDOM_SEED fit, so downstream artefacts are unchanged. "
                             "Default 1 (no repeats).")
    parser.add_argument("--use-saved-params", action="store_true", dest="use_saved_params",
                        help="Refit each model from its saved _best_params.json instead of "
                             "library defaults (no Optuna search). Ignored with --tune.")
    parser.add_argument("--strict",       action="store_true",
                        help="Exit non-zero if any feature set collapses (signal-free columns "
                             "or degenerate enrollment scores)")
    args = parser.parse_args()

    if args.skip:
        unknown = set(args.skip) - set(MODELS)
        if unknown:
            parser.error(f"Unknown model(s): {sorted(unknown)}. Valid: {sorted(MODELS)}")
        for name in args.skip:
            del MODELS[name]
        print(f"Skipping: {args.skip}")

    config_names = list(DATA_CONFIGS.keys())

    # Fail before any training if a blocked column has crept into a feature set.
    _assert_no_blocked_features()

    if args.merge:
        df = load_analysis_df()
        merge_results(df)
        sys.exit(0)

    config_filter = None
    if args.config_index is not None:
        if args.config_index >= len(config_names):
            parser.error(
                f"--config-index {args.config_index} out of range "
                f"(DATA_CONFIGS has {len(config_names)} entries: {config_names})"
            )
        config_filter = config_names[args.config_index]
        print(f"Config index {args.config_index} → '{config_filter}'")

    df = load_analysis_df()

    results = run_all(
        df,
        tune=args.tune,
        n_tune_iter=args.n_iter,
        tune_cv=args.cv,
        n_jobs=args.n_jobs,
        config_filter=config_filter,
        strict=args.strict,
        use_saved_params=args.use_saved_params,
        n_seeds=args.n_seeds,
    )

    if config_filter is None:
        print_main_comparison(results)
        print_stability(results)
