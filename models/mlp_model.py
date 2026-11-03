"""
PyTorch feedforward network for tabular input.

Search space, training and tuning follow the tabular feedforward network of
Nielsen et al. (2025), "From Birth to Adulthood", chapter 4 of M. L. Nielsen's
PhD dissertation, as closely as our task allows.

  train()             — default params, returns an sklearn Pipeline
  train_with_params() — given params dict, returns an sklearn Pipeline
  train_tuned()       — Optuna search on leave-one-cohort-out CV, returns a
                        Pipeline carrying best_params_

What follows chapter 4
  - Architecture: Linear/ReLU stack. Depth 1-5 and the width of every layer
    (16-256) are searched independently.
  - Regularisation: an elastic-net penalty on the weights added to the loss,
    alpha * (l1_ratio * |w|_1 + (1 - l1_ratio) * |w|_2^2), with alpha and
    l1_ratio searched.
  - Optimiser: AdamW with its default parameters and a CONSTANT learning rate,
    searched on a log scale. No schedule.
  - Fixed batch size and epoch count; the best epoch is chosen on validation.
  - Optuna, no pruning. The trial count is --n-iter; chapter 4 used 30.
  - Numerics mean-imputed and standardised (build_preprocessor(standardize=True)).
  - Hyperparameter names match theirs: n_layers, n_units_l{i}, alpha,
    l1_ratio, lr — so the two search-space tables read the same way.

Where we differ, and why
  - Batch 512, smaller than the reference's 4096, to keep a comparable number
    of optimiser steps on a smaller training set. See _BATCH_SIZE.
  - Dropout is searched as ONE EXTRA axis (0-0.9, default 0). Dropout 0 is
    chapter 4's network exactly, so theirs is a point inside our search space.
    It gives the search a second form of regularisation for the wide
    one-hot block of high-cardinality categoricals; the L1 term may make it
    redundant, and the selected values will say so.
  - Classification, not regression: BCEWithLogitsLoss with pos_weight =
    n_neg / n_pos, the same imbalance correction the other models make, and
    the search maximises enrollment-level AUC rather than minimising MSE.
  - Validation is leave-one-cohort-out over the three training years, not a
    random 80/10/10 split — chapter 4 itself names the random split as a
    limitation, and a split within cohorts cannot penalise cohort-specific
    fitting. The best epoch is the one with the highest MEAN AUC across the
    three held-out cohorts.
  - The final model is refit on all training rows for that best number of
    epochs. With cross-validation there is no single trained model to keep.
  - Missing-value indicators are kept, as for every model in this project, so
    whether a feature is observed can itself inform the prediction.
  - Trial 0 is the enqueued _DEFAULTS, as for every model in this project, so
    tuning can never end below the default configuration.

Unchanged from before the alignment: device pinning and determinism (see
_resolve_device and _set_determinism) and the absence of the linear skip path
(see _FeedforwardNet). Weights use PyTorch's default initialisation, as chapter
4 describes no other.

The superseded design — fixed architecture grid, one-cycle schedule, searched
batch size and epoch budget, weight decay as the only penalty — is summarised
in TRAINING_CODE_VERSION's changelog below.
"""
import os
import random

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline

from src.config import CV_SCHEME, MLP_DEVICE, RANDOM_SEED
from src.cv import make_splits, trials_frame
from src.features import build_preprocessor
from src.metrics import _enrollment_auc_from_arrays

optuna.logging.set_verbosity(optuna.logging.WARNING)

## Fixed training budget. Chapter 4 trains every candidate for 5 epochs at
## batch 4096 and keeps its best epoch. The batch is scaled rather than copied:
## 512, smaller than the reference's 4096, keeps a comparable number of
## optimiser steps on a smaller training set.
##
## The search fold is the one that matters: it is where settings are chosen.
_BATCH_SIZE = 512
_EPOCHS     = 5

## Fraction of rows held out INSIDE a single fit, for a diagnostic loss curve
## only. Zero: every fit trains on all the rows it is given, and the held-out
## signal that matters — per-epoch AUC on each held-out cohort — already comes
## from the cross-validation (recorded as cv_auc_by_epoch in best_params_). Set
## above zero only to draw a diagnostic held-out loss curve, and never report a
## model fitted that way: it trains on less data.
_VAL_FRAC   = 0.0

## Device policy: src.config.MLP_DEVICE is the setting; EWS_MLP_DEVICE overrides
## it per-run without editing code. Pinned rather than auto-detected.
_DEVICE_ENV     = "EWS_MLP_DEVICE"
_DEFAULT_DEVICE = MLP_DEVICE

## Deterministic cuBLAS matmuls require this before the CUDA context is created,
## so it is set at import time rather than inside _set_determinism().
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

## Trial 0 of every search, and what an untuned train() fits. Chapter 4 gives
## no defaults, so these are chosen to sit inside its space and start mild:
##   (128, 64) — two layers of moderate, tapering width;
##   alpha at the bottom of its range, l1_ratio at an even L1/L2 split
##               (scikit-learn's ElasticNet default);
##   lr 1e-3   — AdamW's own default;
##   dropout 0 — so trial 0 is chapter 4's network, with no extra axis in use.
_DEFAULTS = {
    "n_layers":     2,
    "n_units_l0":   128,
    "n_units_l1":   64,
    "alpha":        1e-4,
    "l1_ratio":     0.5,
    "lr":           1e-3,
    "dropout_rate": 0.0,
}


def _hidden_sizes(params: dict) -> tuple:
    """The layer widths named by n_layers / n_units_l{i}, as a tuple."""
    return tuple(int(params[f"n_units_l{i}"]) for i in range(int(params["n_layers"])))


# ── Network ───────────────────────────────────────────────────────────────────

class _FeedforwardNet(nn.Module):
    """
    A plain feedforward network: Linear/ReLU/(Dropout) stack to a single logit.

    There is no parallel linear input-to-output ("skip") path. Nothing forces
    the linear part of the signal into such a path: the deep branch has far
    more capacity and fits the target first. A skip path would also blur the
    model class — "MLP" in the comparison ladder should mean an MLP, not a
    wide-and-deep hybrid that partly contains the logistic regression it is
    being compared against.

    Every layer keeps PyTorch's default initialisation. Chapter 4 describes no
    custom initialisation, so none is used.
    """

    def __init__(self, input_dim: int, hidden_sizes: tuple, dropout_rate: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _set_determinism(seed: int) -> None:
    """
    Pin every RNG this model touches and disable nondeterministic kernels.

    torch.manual_seed alone is not enough: cuDNN picks nondeterministic
    algorithms by default and autotunes kernel choice per input shape, so two
    runs of identical code on the same GPU can differ. Combined with a device
    that was auto-detected rather than pinned, that can produce a run-to-run
    spread large enough to move a reported number. Deterministic kernels are
    somewhat slower; for a net trained over a few epochs that is a trivial
    cost next to irreproducible results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    # warn_only: a few ops have no deterministic implementation; warn rather
    # than abort a long run, so the warning is visible but the run completes.
    torch.use_deterministic_algorithms(True, warn_only=True)


def _resolve_device(preferred: str | None = None) -> torch.device:
    """
    Device to run on, in priority order: explicit argument, EWS_MLP_DEVICE, default.

    The default is src.config.MLP_DEVICE, currently "cuda" — a pinned setting,
    not "GPU if available". Results differ between CPU and GPU even with
    deterministic kernels, so auto-detection would make reproducibility depend
    on what hardware happened to be free.

    Pinning "cuda" is what gives each machine the right device without editing
    anything: a machine with a GPU uses it, and a machine without one falls
    through the warning below to CPU. The warning is the
    point — the fallback is visible, where "auto" would be silent.

    train_tuned() resolves through this same function, so the search scores
    candidates on the device the winner is refit on.
    """
    choice = (preferred or os.environ.get(_DEVICE_ENV, _DEFAULT_DEVICE)).lower()
    if choice == "auto":
        choice = "cuda" if torch.cuda.is_available() else "cpu"
    if choice.startswith("cuda") and not torch.cuda.is_available():
        print(f"  WARNING: device '{choice}' requested but CUDA is unavailable — using CPU.",
              flush=True)
        choice = "cpu"
    return torch.device(choice)


def _elastic_net(net: nn.Module, alpha: float, l1_ratio: float) -> torch.Tensor:
    """
    Chapter 4's penalty: alpha * (l1_ratio * |w|_1 + (1 - l1_ratio) * |w|_2^2),
    summed over the weight matrices of every layer. Biases are not penalised.

    The L1 part can drive a weight to exactly zero, which on the first layer
    switches an input column off — the property that matters for the sparse
    one-hot block of programme and department.
    """
    weights = [m.weight for m in net.modules() if isinstance(m, nn.Linear)]
    l1 = sum(w.abs().sum() for w in weights)
    l2 = sum((w ** 2).sum() for w in weights)
    return alpha * (l1_ratio * l1 + (1.0 - l1_ratio) * l2)


def _fit_net(
    net: _FeedforwardNet,
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    l1_ratio: float,
    lr: float,
    device: torch.device,
    random_state: int,
    groups: np.ndarray | None = None,
    batch_size: int = _BATCH_SIZE,
    epochs: int = _EPOCHS,
    val_frac: float = _VAL_FRAC,
    on_epoch_end=None,
) -> tuple:
    """
    Train for a fixed number of epochs; return the net as it is after the last.

    on_epoch_end(net, epoch) is called after every epoch. The CV objective uses
    it to score each epoch on the held-out cohort, which is how the best epoch
    is found; nothing inside this function selects weights.

    val_frac > 0 holds out a slice of rows for a diagnostic loss curve only
    (see _VAL_FRAC). With groups (enrollment IDs) the slice is drawn by
    enrollment, so no enrollment sits on both sides.
    """
    rng = np.random.RandomState(random_state)

    ## Optional diagnostic holdout.
    if val_frac <= 0.0:
        tr_idx, val_idx = np.arange(len(X)), np.empty(0, dtype=int)
    elif groups is not None:
        unique_enr = np.unique(groups)
        rng.shuffle(unique_enr)
        val_enr  = set(unique_enr[:max(1, int(len(unique_enr) * val_frac))])
        val_mask = np.isin(groups, list(val_enr))
        tr_idx, val_idx = np.where(~val_mask)[0], np.where(val_mask)[0]
    else:
        perm = rng.permutation(len(X))
        val_n = max(1, int(len(X) * val_frac))
        val_idx, tr_idx = perm[:val_n], perm[val_n:]

    X_tr = torch.tensor(X[tr_idx], dtype=torch.float32, device=device)
    y_tr = torch.tensor(y[tr_idx], dtype=torch.float32, device=device)
    has_val = len(val_idx) > 0
    if has_val:
        X_val = torch.tensor(X[val_idx], dtype=torch.float32, device=device)
        y_val = torch.tensor(y[val_idx], dtype=torch.float32, device=device)

    net = net.to(device)

    ## Class imbalance: weight the positive class by n_neg / n_pos — the same
    ## correction as class_weight="balanced" in logistic regression and
    ## scale_pos_weight in XGBoost.
    n_pos = float((y[tr_idx] == 1).sum())
    n_neg = float((y[tr_idx] == 0).sum())
    pos_w = torch.tensor(n_neg / n_pos if n_pos > 0 else 1.0, dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    ## AdamW with its default parameters and a constant learning rate, as in
    ## chapter 4. Note the defaults include weight_decay=0.01, applied on top
    ## of the elastic-net term — "default parameters", taken literally.
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)

    loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(random_state),
    )

    train_losses: list[float] = []
    val_losses:   list[float] = []

    for epoch in range(1, epochs + 1):
        net.train()
        epoch_loss, n_batches = 0.0, 0
        for X_b, y_b in loader:
            optimizer.zero_grad()
            ## Data loss plus the elastic-net penalty, per batch.
            loss = criterion(net(X_b), y_b) + _elastic_net(net, alpha, l1_ratio)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1
        ## Mean batch loss over the epoch, penalty included and with dropout
        ## active — an optimisation trace, not a clean training-loss estimate.
        train_losses.append(epoch_loss / max(1, n_batches))

        if has_val:
            net.eval()
            with torch.no_grad():
                val_losses.append(criterion(net(X_val), y_val).item())

        if on_epoch_end is not None:
            on_epoch_end(net, epoch)

    info = {
        "val_losses":   val_losses,
        "train_losses": train_losses,
        "epochs_run":   epochs,
        "val_n_rows":   int(len(val_idx)),
        "train_n_rows": int(len(tr_idx)),
    }
    return net, info


# ── Sklearn-compatible wrapper ────────────────────────────────────────────────

class PyTorchMLP(ClassifierMixin, BaseEstimator):
    """
    Sklearn-compatible wrapper. Sits in the Pipeline's clf step and receives the
    preprocessed numpy arrays from the prep step.
    """

    # Version of the TRAINING PROCEDURE, not of the class interface. Bump it
    # whenever a change would make a previously-fitted model non-comparable with
    # one fitted now. fit() stamps the value onto the instance, so it is pickled
    # with the model, and load_model_checked() compares the stamp against this
    # attribute (which always comes from the CURRENT code).
    #
    # Do NOT bump for changes that leave a fitted model's predictions and their
    # provenance intact — renames, added logging, docstrings.
    #
    #   1  original sklearn MLPClassifier, and the first PyTorch port
    #   2  batch size tuned, OneCycleLR schedule, best-epoch validation split by
    #      enrollment rather than by row
    #   3  best-epoch checkpointing and early stopping removed; the end of the
    #      OneCycleLR schedule is what ships. The holdout only monitors now.
    #   4  epoch budget tuned over _EPOCH_BUDGETS instead of fixed at 10
    #   5  linear skip path; narrower epoch-budget and dropout ranges
    #   6  no inner holdout — every fit trains on all rows it is given
    #   7  pos_weight in the loss instead of random oversampling to 1:1
    #   8  skip path zero-initialised too, so an untrained skip adds nothing
    #   9  batch and epoch grids narrowed
    #  10  linear skip path removed; default architecture (128, 64)
    #  11  aligned with Nielsen et al. (2025, ch. 4): free depth/width search,
    #      elastic-net penalty in the loss, AdamW defaults with a constant LR
    #      (no one-cycle), fixed batch 1024 x 5 epochs with the best epoch chosen
    #      on cross-validated AUC, mean imputation + StandardScaler. Dropout kept
    #      as one extra searched axis.
    #  12  output layer no longer zero-initialised: PyTorch's default
    #      initialisation throughout, as in chapter 4. The zeroing was needed only
    #      while the skip path existed.
    TRAINING_CODE_VERSION = 12

    def __init__(
        self,
        hidden_layer_sizes: tuple = (_DEFAULTS["n_units_l0"], _DEFAULTS["n_units_l1"]),
        alpha: float              = _DEFAULTS["alpha"],
        l1_ratio: float           = _DEFAULTS["l1_ratio"],
        lr: float                 = _DEFAULTS["lr"],
        dropout_rate: float       = _DEFAULTS["dropout_rate"],
        random_state: int         = RANDOM_SEED,
        device: str | None        = None,
        batch_size: int           = _BATCH_SIZE,
        max_iter: int             = _EPOCHS,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha              = alpha
        self.l1_ratio           = l1_ratio
        self.lr                 = lr
        self.dropout_rate       = dropout_rate
        self.random_state       = random_state
        self.device             = device
        self.batch_size         = batch_size
        self.max_iter           = max_iter

    def fit(
        self, X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None
    ) -> "PyTorchMLP":
        """
        `groups` = enrollment ID per row. Only used when _VAL_FRAC > 0, to draw
        the diagnostic holdout by enrollment rather than by row.
        """
        _set_determinism(self.random_state)
        device = _resolve_device(self.device)
        ## Stamp the training procedure, so a model that outlives a rewrite of
        ## it is refused as stale instead of being scored alongside new ones.
        self._training_code_version_ = self.TRAINING_CODE_VERSION
        ## Remember the device: predict and unpickle must match fit.
        self.device_used_   = str(device)
        print(f"PyTorchMLP using device: {device}", flush=True)
        self.classes_       = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        net = _FeedforwardNet(X.shape[1], tuple(self.hidden_layer_sizes), self.dropout_rate)
        self._net, info = _fit_net(
            net, X, y, self.alpha, self.l1_ratio, self.lr, device, self.random_state,
            groups=groups, batch_size=self.batch_size, epochs=self.max_iter,
        )
        ## Diagnostics only — nothing downstream branches on these.
        self.val_losses_   = info["val_losses"]
        self.train_losses_ = info["train_losses"]
        self.epochs_run_   = info["epochs_run"]
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        ## Predict on the device that fit ran on, not on whatever is available now.
        device = _resolve_device(getattr(self, "device_used_", None) or self.device)
        self._net.to(device).eval()
        with torch.no_grad():
            X_t   = torch.tensor(np.asarray(X, dtype=np.float32), device=device)
            probs = torch.sigmoid(self._net(X_t)).cpu().numpy()
        return np.column_stack([1.0 - probs, probs])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        if "_net" in state:
            state["_net"] = state["_net"].cpu()
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        if hasattr(self, "_net"):
            self._net = self._net.to(
                _resolve_device(state.get("device_used_") or state.get("device"))
            )


# ── Public API ────────────────────────────────────────────────────────────────

def _pipeline(feature_list: list, X_train, y_train, clf: PyTorchMLP, groups) -> Pipeline:
    """Fit the chapter-4 preprocessing and the network; return them as one Pipeline."""
    prep  = build_preprocessor(feature_list, standardize=True)
    X_pre = prep.fit_transform(X_train, y_train)
    clf.fit(X_pre, np.asarray(y_train),
            groups=np.asarray(groups) if groups is not None else None)
    return Pipeline([("prep", prep), ("clf", clf)])


def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_list: list,
    groups: pd.Series | None = None,
) -> Pipeline:
    """Fit the default configuration (_DEFAULTS) for the full _EPOCHS."""
    clf = PyTorchMLP(
        hidden_layer_sizes=_hidden_sizes(_DEFAULTS),
        alpha=_DEFAULTS["alpha"],
        l1_ratio=_DEFAULTS["l1_ratio"],
        lr=_DEFAULTS["lr"],
        dropout_rate=_DEFAULTS["dropout_rate"],
        random_state=RANDOM_SEED,
        batch_size=_BATCH_SIZE,
        max_iter=_EPOCHS,
    )
    return _pipeline(feature_list, X_train, y_train, clf, groups)


def train_with_params(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_list: list,
    params: dict,
    groups: pd.Series | None = None,
) -> Pipeline:
    """
    Fit a given configuration. `params` is best_params_ from train_tuned() or
    its saved _best_params.json. Missing keys fall back to _DEFAULTS.

    random_state is overridable so a caller can refit one configuration at
    several seeds and measure how much of a number is the draw.
    """
    seed   = int(params.get("random_state", RANDOM_SEED))
    hidden = params.get("hidden_layer_sizes", _hidden_sizes(_DEFAULTS))
    clf = PyTorchMLP(
        hidden_layer_sizes=tuple(hidden),
        alpha=float(params.get("alpha", _DEFAULTS["alpha"])),
        l1_ratio=float(params.get("l1_ratio", _DEFAULTS["l1_ratio"])),
        lr=float(params.get("lr", _DEFAULTS["lr"])),
        dropout_rate=float(params.get("dropout_rate", _DEFAULTS["dropout_rate"])),
        random_state=seed,
        ## Refit on the device the search scored the configuration on, so
        ## --use-saved-params reproduces the tuned model rather than approximating it.
        device=params.get("device"),
        batch_size=int(params.get("batch_size", _BATCH_SIZE)),
        ## The best epoch found by the search, or the full budget if untuned.
        max_iter=int(params.get("max_iter", _EPOCHS)),
    )
    return _pipeline(feature_list, X_train, y_train, clf, groups)


def train_tuned(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_list: list,
    n_iter: int = 30,
    cv: int = 3,
    groups: pd.Series | None = None,
    n_jobs: int = 1,
    years: pd.Series | None = None,
) -> Pipeline:
    """
    Optuna search over the chapter-4 space (plus dropout), scored on
    leave-one-cohort-out CV.

    Every trial trains each fold for _EPOCHS epochs and scores the held-out
    cohort after EVERY epoch. The trial's best epoch is the one with the highest
    AUC averaged over the folds, and the trial's value is that AUC — chapter 4's
    "best epoch, then best trial", with the mean over held-out cohorts in place
    of their single validation set. The winner is refit on all training rows for
    exactly that many epochs.
    """
    y_arr      = np.asarray(y_train)
    groups_arr = np.asarray(groups) if groups is not None else np.arange(len(y_arr))
    splits     = make_splits(len(y_arr), groups_arr,
                             None if years is None else np.asarray(years),
                             cv, CV_SCHEME)

    ## The search scores candidates on the SAME device the winner is refit on.
    tune_device = _resolve_device()
    if tune_device.type == "cuda" and n_jobs > 1:
        print(f"  NOTE: MLP device is CUDA and n_jobs={n_jobs} — parallel Optuna trials cannot\n"
              "    share a CUDA context, so the MLP search runs SERIALLY on the GPU (n_jobs=1).",
              flush=True)
        n_jobs = 1
    if n_jobs > 1:
        print(f"  WARNING: n_jobs={n_jobs} — parallel trials make the search NOT reproducible.",
              flush=True)
    print(f"  MLP search device: {tune_device}", flush=True)

    ## Preprocess each fold once; it does not depend on the hyperparameters.
    folds = []
    for fold_idx, (tr_idx, val_idx) in enumerate(splits):
        prep = build_preprocessor(feature_list, standardize=True)
        folds.append({
            "seed":       RANDOM_SEED + fold_idx,
            "X_tr":       prep.fit_transform(X_train.iloc[tr_idx]),
            "y_tr":       y_arr[tr_idx],
            "groups_tr":  groups_arr[tr_idx],
            "X_val":      prep.transform(X_train.iloc[val_idx]).astype(np.float32),
            "y_val":      y_arr[val_idx],
            "groups_val": groups_arr[val_idx],
        })

    def objective(trial: optuna.Trial) -> float:
        ## Chapter 4's search space, with their names and ranges.
        n_layers = trial.suggest_int("n_layers", 1, 5)
        hidden   = tuple(trial.suggest_int(f"n_units_l{i}", 16, 256) for i in range(n_layers))
        alpha    = trial.suggest_float("alpha", 1e-4, 0.1, log=True)
        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
        lr       = trial.suggest_float("lr", 1e-4, 0.1, log=True)
        ## Our one addition. 0 reproduces chapter 4's network exactly.
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.9)

        ## auc[fold][epoch-1]: held-out-cohort AUC after each epoch.
        auc = np.zeros((len(folds), _EPOCHS))
        for f, fold in enumerate(folds):
            _set_determinism(fold["seed"])
            X_val_t = torch.tensor(fold["X_val"], device=tune_device)

            def score_epoch(net, epoch, f=f, fold=fold, X_val_t=X_val_t):
                net.eval()
                with torch.no_grad():
                    probs = torch.sigmoid(net(X_val_t)).cpu().numpy()
                auc[f, epoch - 1] = _enrollment_auc_from_arrays(
                    fold["groups_val"], probs, fold["y_val"])

            net = _FeedforwardNet(fold["X_tr"].shape[1], hidden, dropout_rate)
            _fit_net(net, fold["X_tr"], fold["y_tr"], alpha, l1_ratio, lr, tune_device,
                     fold["seed"], groups=fold["groups_tr"], batch_size=_BATCH_SIZE,
                     epochs=_EPOCHS, on_epoch_end=score_epoch)

        ## Best epoch = highest AUC averaged over the held-out cohorts.
        mean_by_epoch = auc.mean(axis=0)
        best_epoch    = int(np.argmax(mean_by_epoch)) + 1
        trial.set_user_attr("best_epoch", best_epoch)
        trial.set_user_attr("cv_auc_by_epoch", [round(float(v), 5) for v in mean_by_epoch])
        return float(mean_by_epoch.max())

    ## No pruner: chapter 4 describes none, and every trial runs in full.
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner=optuna.pruners.NopPruner(),
    )
    study.enqueue_trial(_DEFAULTS)
    study.optimize(objective, n_trials=n_iter, n_jobs=n_jobs, show_progress_bar=False)

    best = study.best_trial
    best_params = {
        "hidden_layer_sizes": _hidden_sizes(best.params),
        "alpha":              best.params["alpha"],
        "l1_ratio":           best.params["l1_ratio"],
        "lr":                 best.params["lr"],
        "dropout_rate":       best.params["dropout_rate"],
        "batch_size":         _BATCH_SIZE,
        ## The refit trains for exactly the best epoch count.
        "max_iter":           best.user_attrs["best_epoch"],
        ## Diagnostic: if the best epoch is always the last, the budget binds.
        "cv_auc_by_epoch":    best.user_attrs["cv_auc_by_epoch"],
        ## Provenance: pins the refit to this device.
        "device":             str(tune_device),
        "torch_version":      torch.__version__,
        "search_n_jobs":      n_jobs,
    }

    model = train_with_params(X_train, y_train, feature_list, best_params, groups=groups)
    model.best_params_   = best_params
    model.cv_best_value_ = float(study.best_value)
    model.cv_n_trials_   = len(study.trials)
    ## What the search explored, not just what it picked. See src/cv.trials_frame.
    model.cv_trials_     = trials_frame(study)
    return model
