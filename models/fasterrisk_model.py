"""
FasterRisk — integer risk score model.
Liu et al. (2022):https://proceedings.neurips.cc/paper_files/paper/2022/hash/7103444259031cc58051f8c9a4868533-Abstract-Conference.html

Preprocessing pipeline: standard prep (impute + missing indicators + one-hot, no
scaling) then _MedianBinarizer: 0/1 columns pass through unchanged, every other
column becomes "above its training median → 1". FasterRisk then selects up to k
binary features and assigns each an integer score in [lb, ub].

Install:  pip install fasterrisk
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from src.config import RANDOM_SEED
from src.features import build_preprocessor

_K_DEFAULT = 8    # max non-zero terms in the risk score
_LB        = -5   # integer coefficient lower bound
_UB        =  5   # integer coefficient upper bound


_smoke_result: str | None = None   # None = not yet run, "" = passed, else the failure


def fasterrisk_smoke_test() -> None:
    """
    Fit a tiny risk score to check FasterRisk actually works in this environment.

    This replaces a `numpy.__version__ >= 2` check. That check was written when
    FasterRisk broke on numpy 2.0, but 0.1.10 fits and predicts fine on numpy
    2.4 — so the version string had become a guess that was wrong in the
    direction that costs the most: it dropped the one interpretable model from
    every table and figure in an environment where it would have run.

    A real fit is the only honest test, and it is cheap (30 rows, 4 features).
    The result is cached, so the cost is paid once per process no matter how
    many feature sets are trained.

    Raises ImportError on failure, because that is what `evaluation.py` catches
    to skip a model — but see `_purge_stale_model` there: a skip must also
    remove the previous .joblib, or downstream steps silently load a model
    trained against an older feature set.
    """
    global _smoke_result
    if _smoke_result is None:
        try:
            from fasterrisk.fasterrisk import RiskScoreOptimizer

            rng = np.random.RandomState(RANDOM_SEED)
            X = (rng.randn(30, 4) > 0).astype(float)
            y = np.where(X[:, 0] > 0, 1, -1)          # FasterRisk wants ±1 labels
            opt = RiskScoreOptimizer(X=X, y=y, k=2)
            opt.optimize()
            multipliers, intercepts, coefficients = opt.get_models()
            if len(multipliers) == 0 or np.asarray(coefficients).size == 0:
                raise RuntimeError("optimizer returned no models")
            _smoke_result = ""
        except Exception as exc:
            _smoke_result = f"{type(exc).__name__}: {exc}"

    if _smoke_result:
        raise ImportError(
            f"FasterRisk smoke test failed in this environment — {_smoke_result}\n"
            f"  (numpy {np.__version__}; run `python -m models.fasterrisk_model` "
            "for the standalone check)\n"
            "Fix the install, or run with --skip fasterrisk to accept its absence "
            "from every table and figure."
        )


class _MedianBinarizer(TransformerMixin, BaseEstimator):
    """
    Turn every preprocessed column into a 0/1 feature for the risk score.

    A column that is already 0/1 in training passes through unchanged. Any other
    column becomes `x > median`, the median taken on the training rows. When no
    training row lies strictly above the median (the median is the maximum) the
    cut becomes `x >= median` instead, so the column still splits the data. A constant column
    becomes constant 0.

    This replaces RobustScaler -> Binarizer(0), which is the same rule for
    continuous columns (RobustScaler centres on the median) but destroys any 0/1
    column where 1 is the majority: its median is 1, it is centred to -1/0, and
    nothing is > 0, so such a column would reach the risk score as constant 0.
    """

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n_cols = X.shape[1]
        self.thresholds_  = np.zeros(n_cols)
        self.strict_      = np.ones(n_cols, dtype=bool)   # True: x > t, False: x >= t
        self.passthrough_ = np.zeros(n_cols, dtype=bool)
        for j in range(n_cols):
            col  = X[:, j]
            vals = np.unique(col)
            if len(vals) <= 1:
                self.thresholds_[j] = np.inf                # constant -> constant 0
            elif np.isin(vals, (0.0, 1.0)).all():
                self.thresholds_[j]  = 0.5
                self.passthrough_[j] = True
            else:
                m = float(np.median(col))
                self.thresholds_[j] = m
                self.strict_[j]     = bool((col > m).any())
        self.n_features_in_ = n_cols
        return self

    def transform(self, X):
        check_is_fitted(self, "thresholds_")
        X = np.asarray(X, dtype=float)
        return np.where(self.strict_, X > self.thresholds_, X >= self.thresholds_).astype(float)

    def get_feature_names_out(self, input_features=None):
        """One output per input, names unchanged."""
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        return np.asarray(input_features, dtype=object)

    def cut_labels(self, names) -> list[str]:
        """Human-readable meaning of each binary column, for the scorecard."""
        out = []
        for name, t, strict, keep in zip(names, self.thresholds_, self.strict_, self.passthrough_):
            if keep or not np.isfinite(t):
                out.append(name)
            else:
                out.append(f"{name} {'>' if strict else '≥'} {t:.3g}")
        return out


class FasterRiskClassifier(ClassifierMixin, BaseEstimator):
    """Sklearn-compatible wrapper around FasterRisk's RiskScoreOptimizer."""

    # Version of the TRAINING PROCEDURE — see PyTorchMLP.TRAINING_CODE_VERSION in
    # models/mlp_model.py for the full mechanism.
    #
    #   1  RobustScaler -> Binarizer(0) -> RiskScoreOptimizer, first model of the
    #      returned pool. k fixed at 8 by train(), min(10, n_features) by
    #      train_tuned(), which runs no search.
    #   2  _MedianBinarizer replaces RobustScaler -> Binarizer(0). Continuous
    #      columns get the same cut (x > training median); 0/1 columns now pass
    #      through, where majority-1 ones had been zeroed; a column whose
    #      median is its maximum is cut at >= instead of becoming constant 0.
    TRAINING_CODE_VERSION = 2

    def __init__(
        self,
        k: int = _K_DEFAULT,
        lb: int = _LB,
        ub: int = _UB,
        random_state: int = RANDOM_SEED,
    ):
        self.k = k
        self.lb = lb
        self.ub = ub
        self.random_state = random_state

    def fit(self, X, y):
        try:
            from fasterrisk.fasterrisk import RiskScoreClassifier, RiskScoreOptimizer
        except ImportError:
            raise ImportError("FasterRisk not installed. Run: pip install fasterrisk")

        self._training_code_version_ = self.TRAINING_CODE_VERSION
        X, y = check_X_y(X, y)
        X = np.asarray(X, dtype=float)
        y_pm1 = np.where(np.asarray(y) == 1, 1.0, -1.0)

        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]

        optimizer = RiskScoreOptimizer(
            X=X, y=y_pm1,
            k=self.k, lb=self.lb, ub=self.ub,
        )
        optimizer.optimize()
        multipliers, intercepts, betas = optimizer.get_models()

        if len(multipliers) == 0:
            raise RuntimeError(
                "FasterRisk found no feasible models. Try increasing k or ub."
            )

        self.classifier_ = RiskScoreClassifier(
            multiplier=multipliers[0],
            intercept=intercepts[0],
            coefficients=betas[0],
        )
        self.coef_ = np.asarray(betas[0])
        self.intercept_ = np.array([intercepts[0]])
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "classifier_")
        X = np.asarray(check_array(X), dtype=float)
        p = np.asarray(self.classifier_.predict_prob(X)).ravel()
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def get_scorecard(self, feature_names: list) -> pd.DataFrame:
        check_is_fitted(self, "classifier_")
        rows = [
            {"feature": name, "score": int(w)}
            for name, w in zip(feature_names, self.coef_)
            if w != 0
        ]
        df = pd.DataFrame(rows).sort_values("score", key=abs, ascending=False)
        df["intercept"] = round(float(self.intercept_[0]), 3)
        return df.reset_index(drop=True)


def train(X_train: pd.DataFrame, y_train: pd.Series, feature_list: list) -> Pipeline:
    fasterrisk_smoke_test()
    prep = build_preprocessor(feature_list, scale=False)
    model = Pipeline([
        ("prep",     prep),
        ("binarize", _MedianBinarizer()),
        ("clf",      FasterRiskClassifier(k=_K_DEFAULT, lb=_LB, ub=_UB)),
    ])
    model.fit(X_train, y_train)
    return model


def train_with_params(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_list: list,
    params: dict,
) -> Pipeline:
    fasterrisk_smoke_test()
    prep = build_preprocessor(feature_list, scale=False)
    ub = int(params.get("ub", _UB))
    model = Pipeline([
        ("prep",     prep),
        ("binarize", _MedianBinarizer()),
        ("clf",      FasterRiskClassifier(
            k=int(params.get("k", _K_DEFAULT)),
            lb=-ub,
            ub=ub,
            random_state=int(params.get("random_state", RANDOM_SEED)),
        )),
    ])
    model.fit(X_train, y_train)
    return model


def train_tuned(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_list: list,
) -> Pipeline:
    """
    Named for symmetry with the other models' train_tuned; there is no search.
    FasterRisk has no hyperparameters worth searching here, so the only
    difference from train() is the size of the score: k = min(10, number of
    binarised inputs) terms instead of the default 8. best_params_ records k and
    the coefficient bound, which is what the SI hyperparameter table reports.
    """
    fasterrisk_smoke_test()

    prep = build_preprocessor(feature_list, scale=False)
    binarizer = _MedianBinarizer()
    n_features = binarizer.fit_transform(prep.fit_transform(X_train, y_train)).shape[1]
    k = min(10, n_features)

    final = Pipeline([
        ("prep",     prep),
        ("binarize", binarizer),
        ("clf",      FasterRiskClassifier(k=k, lb=_LB, ub=_UB)),
    ])
    final.fit(X_train, y_train)
    final.best_params_ = {"k": k, "ub": _UB}
    return final


if __name__ == "__main__":
    # Standalone environment check — run this before a tuning run:
    #   python -m models.fasterrisk_model
    # A silent FasterRisk skip costs a whole tuning run, so it is worth 5 seconds.
    import sys

    print(f"numpy      {np.__version__}")
    try:
        import fasterrisk as _fr
        print(f"fasterrisk {getattr(_fr, '__version__', 'unknown')}")
    except ImportError:
        print("fasterrisk NOT INSTALLED")
        sys.exit(1)

    try:
        fasterrisk_smoke_test()
    except ImportError as exc:
        print(f"\nFAIL\n{exc}")
        sys.exit(1)
    print("\nOK — FasterRisk fits and predicts in this environment.")
