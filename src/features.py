import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from src.config import CATEGORICAL_FEATURES, OHE_HANDLE_UNKNOWN, OHE_MIN_FREQUENCY


_ALLOW_STALE_ENV = "EWS_ALLOW_STALE_MODELS"


def _classifier_of(model):
    """The estimator inside a saved pipeline, or the object itself."""
    steps = getattr(model, "named_steps", None)
    if steps is not None and "clf" in steps:
        return steps["clf"]
    return model   # e.g. FasterRiskClassifier, which is not an sklearn Pipeline


def _check_training_code_version(model, path) -> None:
    """
    Refuse a model fitted by a superseded version of its own training procedure.

    The feature-list check below catches changed *inputs*. It cannot catch a
    changed *procedure*: when the batch size, LR schedule or validation split
    change, the feature lists can stay untouched, so a model fitted under the
    old procedure would load cleanly and be scored next to current models in
    the same table.

    An estimator opts in by defining TRAINING_CODE_VERSION on the class and
    stamping it onto the instance in fit(). Unpickling restores the instance
    dict but takes the class from current code, so comparing the two compares
    "what fitted this file" against "what would fit it now". A class without the
    attribute is not participating and is left alone.

    Set EWS_ALLOW_STALE_MODELS=1 to downgrade this to a warning — for inspecting
    or plotting existing artefacts deliberately. Never for a run whose numbers
    will be reported.
    """
    import os
    import warnings

    clf      = _classifier_of(model)
    expected = getattr(type(clf), "TRAINING_CODE_VERSION", None)
    if expected is None:
        return

    found = getattr(clf, "_training_code_version_", None)
    if found == expected:
        return

    found_str = "no stamp (predates version stamping)" if found is None else f"version {found}"
    msg = (
        f"Saved model was fitted by superseded training code: {path}\n"
        f"  file was fitted by:  {found_str}\n"
        f"  current code is:     version {expected}\n"
        f"  The feature list matches, so this model loads and scores — but it was\n"
        f"  produced by a different training procedure and is not comparable with\n"
        f"  models fitted now.\n"
        f"  --use-saved-params will NOT fix this: the saved hyperparameters were\n"
        f"  themselves chosen under the superseded procedure. Re-tune:\n"
        f"    python src/evaluation.py --tune\n"
        f"  Set {_ALLOW_STALE_ENV}=1 to load it anyway (inspection only)."
    )
    if os.environ.get(_ALLOW_STALE_ENV) == "1":
        warnings.warn(msg, stacklevel=2)
        return
    raise ValueError(msg)


def _check_encoder_settings(model, path) -> None:
    """
    Refuse a model whose one-hot encoder was built under other OHE_* settings.

    OHE_MIN_FREQUENCY and OHE_HANDLE_UNKNOWN change every model that goes
    through build_preprocessor while leaving both the feature list and the
    training code untouched, so neither of the other two checks would notice.
    Reading the settings off the saved encoder means setting the threshold in
    config.py is the whole decision — no version stamps to remember.
    """
    import os
    import warnings

    prep = getattr(model, "named_steps", {}).get("prep")
    cat  = getattr(prep, "named_transformers_", {}).get("cat")
    enc  = getattr(cat, "named_steps", {}).get("encoder")
    if enc is None:
        return   # no categorical block

    saved = {"min_frequency": enc.min_frequency, "handle_unknown": enc.handle_unknown}
    now   = {"min_frequency": OHE_MIN_FREQUENCY, "handle_unknown": OHE_HANDLE_UNKNOWN}
    if saved == now:
        return

    msg = (
        f"Saved model was encoded under other one-hot settings: {path}\n"
        f"  file:    {saved}\n"
        f"  config:  {now}\n"
        f"  Re-tune before reporting: the saved hyperparameters were chosen on the\n"
        f"  old encoding, so --use-saved-params only refits them onto the new one.\n"
        f"  Set {_ALLOW_STALE_ENV}=1 to load it anyway (inspection only)."
    )
    if os.environ.get(_ALLOW_STALE_ENV) == "1":
        warnings.warn(msg, stacklevel=2)
        return
    raise ValueError(msg)


def load_model_checked(path, feature_list: list):
    """
    joblib.load a saved pipeline, refusing one that no longer matches the code.

    Two independent staleness checks:

    Inputs — a .joblib records the columns it was fitted on. When a feature list
    changes — as LMS did when late_or_missing_pct was split into late_pct +
    missing_pct — every saved model predates the change, and the first thing the
    caller sees is sklearn's "columns are missing: {...}" from six frames down a
    ColumnTransformer. Worse, a model whose old list is a *superset* of the new
    one can still transform and quietly score on the wrong features.

    Procedure — the training code itself may have been rewritten with the
    feature list untouched, which the input check cannot see. See
    _check_training_code_version.

    Encoding — OHE_MIN_FREQUENCY / OHE_HANDLE_UNKNOWN changed since the fit. See
    _check_encoder_settings.

    Both cases mean the same thing: refit before trusting the numbers.
    """
    import joblib

    model = joblib.load(path)

    _check_training_code_version(model, path)
    _check_encoder_settings(model, path)

    fitted = getattr(model, "feature_names_in_", None)
    if fitted is None:
        return model   # model does not record its inputs — nothing more to check

    fitted, wanted = set(map(str, fitted)), set(map(str, feature_list))
    if fitted != wanted:
        # Two different causes, and they need different fixes. `wanted - fitted`
        # alone — the model is missing columns the config asks for, and has none
        # the config does not — is also what a column ABSENT FROM THE DATA looks
        # like: evaluation.py trains on `[f for f in feature_list if f in
        # df.columns]` and records the config under its full name anyway. Telling
        # someone to re-run evaluation.py in that case sends them round a loop
        # that reproduces the same file, so the message says where to look.
        only_in_config = sorted(wanted - fitted)
        only_in_model  = sorted(fitted - wanted)
        if only_in_config and not only_in_model:
            cause = (
                "  The model is missing columns the feature set asks for and has no\n"
                "  extras, which happens in two ways: the feature set grew since the\n"
                "  fit, or those columns were not in the data when it was fitted and\n"
                "  evaluation.py trained without them. Check evaluation.py's feature-\n"
                "  health warnings for missing columns before re-running — if there\n"
                "  are any, retraining will produce exactly this file again.\n"
            )
        else:
            cause = ("Re-run  python src/evaluation.py  to retrain against the current "
                     "feature sets.\n")
        raise ValueError(
            f"Saved model is stale: {path}\n"
            f"  trained on {len(fitted)} features, current feature set has {len(wanted)}\n"
            f"  only in model:  {only_in_model or '—'}\n"
            f"  only in config: {only_in_config or '—'}\n"
            + cause
        )
    return model


class _ToObject(BaseEstimator, TransformerMixin):
    """
    Cast categorical columns to plain object before imputation.

    `nationality` is an ordered pandas Categorical (DK < Nordic < EU_EEA_CH <
    Other), and its ordering is meaningless to a one-hot encoder. Casting to
    object also lets NaN through as a value the encoder can treat as its own
    level, which a Categorical dtype would not. The dtype is kept on the
    dataframe — tables and legends rely on the level order — and dropped only
    here, on the way into the model.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(X).astype(object)

    def get_feature_names_out(self, input_features=None):
        """
        Pass names through unchanged — this step only recasts dtype.

        Without it the whole ColumnTransformer loses get_feature_names_out, since
        one non-conforming step is enough. That silently broke feature naming for
        every feature set containing a categorical, which is every set from
        c_plus_enrolment on: the FasterRisk scorecard (plotting.py) and the SHAP
        plots both call it and would raise AttributeError from inside sklearn.
        """
        return np.asarray(input_features, dtype=object)


def build_preprocessor(
    feature_list: list,
    scale: bool = True,
    min_frequency=None,
    standardize: bool = False,
) -> ColumnTransformer:
    """Sklearn ColumnTransformer for a given feature list.

    scale=True for LR: median imputation, RobustScaler on numerics.
    scale=False for tree-based models: no scaling (NaN handled natively).
    standardize=True for the MLP: MEAN imputation and StandardScaler — the
      numeric recipe of the tabular network in Nielsen et al. (2025, ch. 4),
      which the MLP follows. Overrides `scale`.
    Missing-value indicators are added in every case — see below.
    Fit only on training data.

    min_frequency overrides config.OHE_MIN_FREQUENCY for one call — for tests
    on toy frames, which the configured threshold would otherwise pool entirely.
    Leave it None in the pipeline so every model in a run shares one setting; a
    preprocessor that differs between models is not a comparison.
    """
    min_freq = OHE_MIN_FREQUENCY if min_frequency is None else min_frequency
    cats = [f for f in feature_list if f in CATEGORICAL_FEATURES]
    nums = [f for f in feature_list if f not in CATEGORICAL_FEATURES]

    transformers = []

    if nums:
        # add_indicator is unconditional, not tied to `scale`. Missing-value
        # indicators are added because missingness can carry information: for
        # the performance features, "no result yet" (assessment results dated
        # after period_end are blanked) is a substantive state, and an imputed
        # value that hides it is a different statement about the student than
        # an explicit flag. Trees handle NaN natively but still cannot
        # represent "this was imputed" once the imputer has run.
        ## The imputer fills with the median by default, or the mean for the
        ## chapter-4 recipe; either way it appends the missing-value indicators.
        strategy  = "mean" if standardize else "median"
        num_steps = [("imputer", SimpleImputer(strategy=strategy, add_indicator=True))]
        if standardize:
            num_steps.append(("scaler", StandardScaler()))
        elif scale:
            num_steps.append(("scaler", RobustScaler()))
        transformers.append(("num", Pipeline(num_steps), nums))

    if cats:
        # No imputer. OneHotEncoder treats NaN as its own level, which is what we
        # want: the numeric block gets add_indicator so "missing" survives as an
        # explicit state, and this is the categorical equivalent. Imputing with
        # the mode would, for `nationality`, model every student of unknown
        # citizenship as the majority category. An explicit "unknown" column
        # lets the model use unknown-ness as a signal instead.
        cat_pipe = Pipeline([
            ("to_object", _ToObject()),
            ("encoder", OneHotEncoder(
                handle_unknown=OHE_HANDLE_UNKNOWN,
                # None = every level gets a column, however few rows carry it.
                # See the block on OHE_MIN_FREQUENCY in src/config.py.
                min_frequency=min_freq,
                sparse_output=False,
            )),
        ])
        transformers.append(("cat", cat_pipe, cats))

    return ColumnTransformer(transformers, remainder="drop")