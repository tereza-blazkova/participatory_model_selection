import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import CV_SCHEME, RANDOM_SEED
from src.cv import make_splits, trials_frame
from src.features import build_preprocessor
from src.metrics import _enrollment_auc_from_arrays

optuna.logging.set_verbosity(optuna.logging.WARNING)

_DEFAULTS = {"C": 1.0}


class _VersionedLogisticRegression(LogisticRegression):
    """
    sklearn's LogisticRegression, plus a training-procedure stamp.

    The staleness guard in src/features.py reads TRAINING_CODE_VERSION off the
    CLASS of the estimator in the pipeline's "clf" slot and compares it against
    the stamp fit() left on the instance. That only works for a class we own —
    sklearn's is not one, hence this three-line subclass. Behaviour is
    unchanged; the only addition is the stamp.

    Version of the TRAINING PROCEDURE, not the interface. Bump whenever a change
    makes a previously-fitted model non-comparable with one fitted now.

      1  RobustScaler + one-hot prep, class_weight="balanced", lbfgs,
         max_iter=1000; C selected by the Optuna search on enrollment-level AUC
         (grouped CV when v1 was written; leave-one-cohort-out since, src/cv.py —
         a change to the search, not to the fit, so no bump).
    """

    TRAINING_CODE_VERSION = 1

    def fit(self, X, y, sample_weight=None):
        out = super().fit(X, y, sample_weight=sample_weight)
        self._training_code_version_ = self.TRAINING_CODE_VERSION
        return out


def train(X_train: pd.DataFrame, y_train: pd.Series, feature_list: list) -> Pipeline:
    prep = build_preprocessor(feature_list, scale=True)
    clf = _VersionedLogisticRegression(
        C=_DEFAULTS["C"], max_iter=1000, solver="lbfgs",
        class_weight="balanced", random_state=RANDOM_SEED,
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
    prep = build_preprocessor(feature_list, scale=True)
    clf = _VersionedLogisticRegression(
        C=float(params.get("C", _DEFAULTS["C"])),
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        random_state=seed,
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
    y_arr = np.asarray(y_train)
    groups_arr = np.asarray(groups) if groups is not None else np.arange(len(y_arr))
    splits = make_splits(len(y_arr), groups_arr,
                         None if years is None else np.asarray(years),
                         cv, CV_SCHEME)

    def objective(trial: optuna.Trial) -> float:
        C = trial.suggest_float("C", 1e-3, 1e2, log=True)
        fold_aucs: list[float] = []
        for fold_idx, (tr_idx, val_idx) in enumerate(splits):
            X_tr_raw = X_train.iloc[tr_idx]
            X_val_raw = X_train.iloc[val_idx]
            y_tr = y_arr[tr_idx]
            y_val = y_arr[val_idx]

            fold_prep = build_preprocessor(feature_list, scale=True)
            X_tr = fold_prep.fit_transform(X_tr_raw)
            X_val = fold_prep.transform(X_val_raw)

            clf = LogisticRegression(
                C=C, max_iter=1000, solver="lbfgs",
                class_weight="balanced", random_state=RANDOM_SEED,
            )
            clf.fit(X_tr, y_tr)
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
    best_C = float(study.best_params["C"])

    model = train_with_params(X_train, y_train, feature_list, {"C": best_C})
    model.best_params_    = {"C": best_C}
    # What the search's own objective scored, recorded with the chosen
    # parameters in _best_params.json.
    model.cv_best_value_  = float(study.best_value)
    model.cv_n_trials_    = len(study.trials)
    # What the search explored, not just what it picked. See src/cv.trials_frame.
    model.cv_trials_      = trials_frame(study)
    return model
