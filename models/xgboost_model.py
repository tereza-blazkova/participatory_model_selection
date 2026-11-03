import numpy as np
import optuna
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import CV_SCHEME, RANDOM_SEED
from src.cv import make_splits, trials_frame
from src.features import build_preprocessor
from src.metrics import _enrollment_auc_from_arrays

optuna.logging.set_verbosity(optuna.logging.WARNING)

_DEFAULTS = {
    "n_estimators":     300,
    "max_depth":        5,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
}


class _VersionedXGBClassifier(XGBClassifier):
    """
    xgboost's XGBClassifier, plus a training-procedure stamp.

    See _VersionedLogisticRegression in models/logistic_regression.py for why the
    subclass is needed: the guard reads TRAINING_CODE_VERSION off the class of
    the "clf" step, which has to be a class we own. Behaviour is unchanged.

      1  n_estimators tuned by the grouped-CV Optuna search and used by the final
         fit. Replaces early stopping on the fold's own validation rows (which
         made every trial score optimistic) followed by a re-pick on an ungrouped
         15% split (which let EWS rows of one enrollment sit on both sides).
    """

    TRAINING_CODE_VERSION = 1

    def fit(self, X, y, **kwargs):
        out = super().fit(X, y, **kwargs)
        self._training_code_version_ = self.TRAINING_CODE_VERSION
        return out


def train(X_train: pd.DataFrame, y_train: pd.Series, feature_list: list) -> Pipeline:
    prep = build_preprocessor(feature_list, scale=False)
    scale_pos_weight = float((y_train == 0).sum()) / float((y_train == 1).sum())
    clf = _VersionedXGBClassifier(
        # n_estimators comes from _DEFAULTS below. Passing it here as well made
        # this a duplicate keyword and raised TypeError before XGBClassifier was
        # ever constructed, so every untuned run died on the first XGBoost job.
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=0,
        **_DEFAULTS,
    )
    model = Pipeline([("prep", prep), ("clf", clf)])
    model.fit(X_train, y_train)
    return model


def train_with_params(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_list: list,
    params: dict,
) -> Pipeline:
    # Model seed, normally RANDOM_SEED. Taken from params so the pipeline can
    # refit the same configuration at several seeds and report how much of a
    # number is the configuration and how much is the draw. Every model in this
    # project reads it the same way, including the ones assumed deterministic:
    # XGBoost subsamples, so it has a draw too, and a
    # like-for-like comparison has to measure it rather than assume it away.
    seed = int(params.get("random_state", RANDOM_SEED))
    prep = build_preprocessor(feature_list, scale=False)
    scale_pos_weight = float((y_train == 0).sum()) / float((y_train == 1).sum())
    clf = _VersionedXGBClassifier(
        n_estimators=int(params.get("n_estimators", 300)),
        max_depth=int(params.get("max_depth", _DEFAULTS["max_depth"])),
        learning_rate=float(params.get("learning_rate", _DEFAULTS["learning_rate"])),
        subsample=float(params.get("subsample", _DEFAULTS["subsample"])),
        colsample_bytree=float(params.get("colsample_bytree", _DEFAULTS["colsample_bytree"])),
        min_child_weight=int(params.get("min_child_weight", _DEFAULTS["min_child_weight"])),
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )
    model = Pipeline([("prep", prep), ("clf", clf)])
    model.fit(X_train, y_train)
    return model


def train_tuned(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_list: list,
    n_iter: int = 50,
    cv: int = 3,
    groups: pd.Series | None = None,
    n_jobs: int = 1,
    years: pd.Series | None = None,
) -> Pipeline:
    scale_pos_weight = float((y_train == 0).sum()) / float((y_train == 1).sum())
    y_arr = np.asarray(y_train)
    groups_arr = np.asarray(groups) if groups is not None else np.arange(len(y_arr))
    splits = make_splits(len(y_arr), groups_arr,
                         None if years is None else np.asarray(years),
                         cv, CV_SCHEME)

    # Put the parallelism in exactly one place:
    #   n_jobs>1  → Optuna runs trials in parallel, each trial single-threaded
    #   n_jobs==1 → trials run serially, each using every core
    # The second is the reproducible option: a parallel Optuna search is
    # nondeterministic regardless of seeding, because trials reach the TPE
    # sampler in completion order. Threading the work inside a serial search
    # keeps the cores busy without that cost.
    trial_n_jobs = 1 if n_jobs > 1 else -1

    def objective(trial: optuna.Trial) -> float:
        params = {
            # Capacity is a tuned axis, like every other hyperparameter. It is
            # not set by early stopping on the fold's own validation set — the
            # same rows the trial is scored on — which would make every trial
            # score optimistic. Nothing may be selected on the data that scores
            # the selection.
            "n_estimators":     trial.suggest_int("n_estimators", 100, 1000, step=100),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
        }
        fold_aucs: list[float] = []
        for fold_idx, (tr_idx, val_idx) in enumerate(splits):
            X_tr_raw = X_train.iloc[tr_idx]
            X_val_raw = X_train.iloc[val_idx]
            y_tr = y_arr[tr_idx]
            y_val = y_arr[val_idx]

            fold_prep = build_preprocessor(feature_list, scale=False)
            X_tr = fold_prep.fit_transform(X_tr_raw)
            X_val = fold_prep.transform(X_val_raw)

            clf = XGBClassifier(
                **params,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=RANDOM_SEED,
                n_jobs=trial_n_jobs,
                verbosity=0,
            )
            clf.fit(X_tr, y_tr, verbose=False)
            y_prob_val = clf.predict_proba(X_val)[:, 1]
            auc = _enrollment_auc_from_arrays(groups_arr[val_idx], y_prob_val, y_val)
            fold_aucs.append(auc)

            trial.report(float(np.mean(fold_aucs)), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_aucs))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        # n_warmup_steps=1, not 0: trials report the running mean after each
        # fold, so 0 would let a configuration be pruned on ONE fold's evidence.
        # Requiring two folds makes the pruning decision rest on more than a
        # single held-out cohort, for roughly double the cost of a pruned trial.
    )
    study.enqueue_trial(_DEFAULTS)
    study.optimize(objective, n_trials=n_iter, n_jobs=n_jobs, show_progress_bar=False)
    best = study.best_params

    # No probe: n_estimators came out of the search above, on grouped folds.
    # The final fit therefore uses the configuration that was actually scored,
    # and trains on 100% of the training rows.
    model = train_with_params(X_train, y_train, feature_list, best)
    model.best_params_   = best
    model.cv_best_value_ = float(study.best_value)
    model.cv_n_trials_   = len(study.trials)
    # What the search explored, not just what it picked. See src/cv.trials_frame.
    model.cv_trials_     = trials_frame(study)
    return model
