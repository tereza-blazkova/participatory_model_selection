"""
Bootstrap confidence intervals for enrollment-level and row-level metrics.

Draws N_BOOT resamples of the test-set enrollments (with replacement) and
computes metrics for every model × feature-set configuration on each resample.
All configurations share the same resamples per test year, so any pairwise
comparison (e.g. workshop vs max ML) is automatically paired.

Outputs
-------
results/bootstrap_cis.csv
    Enrollment-level (enr_recall, enr_precision, roc_auc) + optionally
    row-level (row_roc_auc) metrics.
    Columns: model, feature_set, test_year, metric, observed, ci_low, ci_high, n_boot

results/bootstrap_cis_pilot_snapshot.csv  (--pilot-snapshot only)
    Enrollment-level metrics (enr_recall, enr_precision, roc_auc) on the
    October 2023 pilot snapshot, split by faculty (FAC_A, FAC_B).
    Columns: model, feature_set, faculty, metric, observed, ci_low, ci_high, n_boot
    All ML models × all feature sets (a–e + used_vars).
    ML models only; credits_policy and contact pilot excluded (no continuous score).

CLI
---
    python src/bootstrap.py                        # enrollment-level, all years
    python src/bootstrap.py --year 2021            # single year
    python src/bootstrap.py --row-level            # also compute row_roc_auc
    python src/bootstrap.py --pilot-snapshot             # pilot_snapshot bootstrap
    python src/bootstrap.py --n-boot 1000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from paths.paths import RESULTS_PATH
from src.config import RANDOM_SEED
from src.metrics import conservative_threshold
from src.data import load_pilot_snapshot

_N_BOOT_DEFAULT = 1000


def _auc_safe(y: np.ndarray, scores: np.ndarray) -> float:
    valid = ~np.isnan(scores)
    if not valid.any() or len(np.unique(y[valid])) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y[valid], scores[valid]))
    except Exception:
        return float("nan")


def _ci_row(model, fs, test_year, metric, obs, boot_vals, n_boot) -> dict:
    return {
        "model":       model,
        "feature_set": fs,
        "test_year":   test_year,
        "metric":      metric,
        "observed":    float(obs),
        "ci_low":      float(np.nanpercentile(boot_vals, 2.5)),
        "ci_high":     float(np.nanpercentile(boot_vals, 97.5)),
        "n_boot":      n_boot,
    }


# ── Enrollment-level bootstrap ────────────────────────────────────────────────

def bootstrap_year(
    preds_df: pd.DataFrame,
    test_year: int,
    n_boot: int = _N_BOOT_DEFAULT,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Paired enrollment-level bootstrap for one test year.
    Metrics: enr_recall, enr_precision, roc_auc.
    """
    yr = preds_df[preds_df["test_year"] == test_year]
    yr = yr.assign(_cfg=yr["model"] + "__" + yr["feature_set"])

    enr_ids = np.array(sorted(yr["enrollment"].unique()))
    n       = len(enr_ids)

    cfg_meta = (
        yr[["model", "feature_set", "_cfg"]]
        .drop_duplicates()
        .sort_values(["model", "feature_set"])
        .reset_index(drop=True)
    )
    cfg_keys = cfg_meta["_cfg"].tolist()
    configs  = cfg_meta[["model", "feature_set"]].values.tolist()
    n_cfg    = len(configs)

    ef_mat = (
        yr.pivot(index="enrollment", columns="_cfg", values="ever_flagged")
        .reindex(index=enr_ids, columns=cfg_keys)
        .fillna(0).values.astype(bool)
    )
    mp_mat = (
        yr.pivot(index="enrollment", columns="_cfg", values="max_prob")
        .reindex(index=enr_ids, columns=cfg_keys)
        .values.astype(float)
    )
    do_vec = (
        yr[["enrollment", "dropped_out"]]
        .drop_duplicates("enrollment")
        .set_index("enrollment")
        .reindex(enr_ids)["dropped_out"]
        .fillna(0).values.astype(bool)
    )

    n_do_obs = do_vec.sum()
    tp_obs   = (ef_mat & do_vec[:, None]).sum(axis=0)
    nf_obs   = ef_mat.sum(axis=0)
    obs_recall = np.where(n_do_obs > 0, tp_obs / n_do_obs, np.nan).astype(float)
    obs_prec   = np.where(nf_obs   > 0, tp_obs / nf_obs,   np.nan).astype(float)
    obs_auc    = np.array([_auc_safe(do_vec, mp_mat[:, j]) for j in range(n_cfg)])

    boot_recall = np.full((n_boot, n_cfg), np.nan)
    boot_prec   = np.full((n_boot, n_cfg), np.nan)
    boot_auc    = np.full((n_boot, n_cfg), np.nan)

    rng = np.random.default_rng(seed)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y, ef, mp = do_vec[idx], ef_mat[idx], mp_mat[idx]
        n_do = y.sum()
        tp   = (ef & y[:, None]).sum(axis=0)
        nf   = ef.sum(axis=0)
        boot_recall[b] = np.where(n_do > 0, tp / n_do, np.nan)
        boot_prec[b]   = np.where(nf   > 0, tp / nf,   np.nan)
        if len(np.unique(y)) >= 2:
            for j in range(n_cfg):
                boot_auc[b, j] = _auc_safe(y, mp[:, j])

    rows = []
    for j, (model, fs) in enumerate(configs):
        for metric, b_arr, obs_arr in [
            ("enr_recall",    boot_recall, obs_recall),
            ("enr_precision", boot_prec,   obs_prec),
            ("roc_auc",       boot_auc,    obs_auc),
        ]:
            rows.append(_ci_row(model, fs, test_year, metric,
                                obs_arr[j], b_arr[:, j], n_boot))
    return pd.DataFrame(rows)


# ── Row-level bootstrap ───────────────────────────────────────────────────────

def bootstrap_row_year(
    row_preds_df: pd.DataFrame,
    test_year: int,
    n_boot: int = _N_BOOT_DEFAULT,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Enrollment-resampled bootstrap for row-level AUC (one test year).

    Resamples enrollment IDs; all EWS rows for a sampled enrollment appear
    (with repetition), preserving within-enrollment structure.
    Metric: row_roc_auc.
    """
    yr = (
        row_preds_df[row_preds_df["test_year"] == test_year]
        .assign(_cfg=lambda d: d["model"] + "__" + d["feature_set"])
        .copy()
    )

    enr_ids    = np.array(sorted(yr["enrollment"].unique()))
    n          = len(enr_ids)
    enr_to_idx = {eid: i for i, eid in enumerate(enr_ids)}
    yr["_eidx"] = yr["enrollment"].map(enr_to_idx)

    cfg_meta = (
        yr[["model", "feature_set", "_cfg"]]
        .drop_duplicates()
        .sort_values(["model", "feature_set"])
        .reset_index(drop=True)
    )
    cfg_keys = cfg_meta["_cfg"].tolist()
    configs  = cfg_meta[["model", "feature_set"]].values.tolist()
    n_cfg    = len(configs)

    # Pre-build flat arrays + enrollment-index map per config (CSR-like)
    flat_probs  = []
    flat_douts  = []
    row_eidx    = []
    obs_auc     = np.full(n_cfg, np.nan)

    for j, cfg in enumerate(cfg_keys):
        sub = yr[yr["_cfg"] == cfg].sort_values("_eidx")
        fp  = sub["y_prob"].values.astype(float)
        fd  = sub["dropout"].values.astype(bool)
        ei  = sub["_eidx"].values.astype(int)
        flat_probs.append(fp)
        flat_douts.append(fd)
        row_eidx.append(ei)
        obs_auc[j] = _auc_safe(fd, fp)

    boot_auc = np.full((n_boot, n_cfg), np.nan)
    rng = np.random.default_rng(seed)

    for b in range(n_boot):
        idx    = rng.integers(0, n, size=n)
        counts = np.bincount(idx, minlength=n)
        for j in range(n_cfg):
            weights = counts[row_eidx[j]]
            mask    = np.repeat(np.arange(len(flat_probs[j])), weights)
            probs   = flat_probs[j][mask]
            douts   = flat_douts[j][mask]
            boot_auc[b, j] = _auc_safe(douts, probs)

    rows = []
    for j, (model, fs) in enumerate(configs):
        rows.append(_ci_row(model, fs, test_year, "row_roc_auc",
                            obs_auc[j], boot_auc[:, j], n_boot))
    return pd.DataFrame(rows)


# ── Pilot snapshot bootstrap ────────────────────────────────────────────────────────

def bootstrap_pilot_snapshot(
    n_boot: int = _N_BOOT_DEFAULT,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Bootstrap enrollment-level metrics on the October 2023 pilot snapshot.

    Computes enr_recall, enr_precision, roc_auc for all ML models × all
    feature sets (a–e plus used_vars), split by faculty (FAC_A, FAC_B).
    Threshold τ is calibrated per-resample to the training positive rate.

    Since pilot_snapshot has one EWS row per enrollment, enrollment-level and
    row-level metrics are identical — only enrollment-level metrics are stored.

    ML models only — credits_policy and contact pilot have no continuous score.
    """

    from src.config import DATA_CONFIGS
    from src.features import load_model_checked
    from src.pilot_snapshot_comparison import FACULTIES, _CONFIG_ORDER, _ML_MODELS, _load_metadata

    _ALL_CONFIGS = list(_CONFIG_ORDER) + ["used_vars"]

    pilot_snapshot   = load_pilot_snapshot()
    meta       = _load_metadata()
    train_pos_rate = meta["train_pos_rate"]
    models_dir = RESULTS_PATH / "models"

    # Load all models once using DATA_CONFIGS for feature lists (bypasses stale metadata).
    # load_model_checked refuses a model fitted on a different feature list — without
    # it a stale .joblib surfaces as sklearn's "columns are missing: {...}" from deep
    # inside a ColumnTransformer, or worse, transforms fine and scores on the wrong
    # features. The message names the model and says to re-run evaluation.py.
    loaded_models: dict = {}
    for config_name in _ALL_CONFIGS:
        feature_list = [c for c in DATA_CONFIGS[config_name] if c in pilot_snapshot.columns]
        for model_name in _ML_MODELS:
            path = models_dir / f"{model_name}__{config_name}.joblib"
            if path.exists():
                loaded_models[f"{model_name}__{config_name}"] = load_model_checked(
                    path, feature_list
                )

    all_rows = []

    for faculty in FACULTIES:
        subset  = pilot_snapshot[pilot_snapshot["faculty_code"] == faculty].copy().reset_index(drop=True)
        enr_ids = np.array(sorted(subset["enrollment"].unique()))
        n       = len(enr_ids)
        y_true  = subset["dropout"].values.astype(bool)

        # Collect (model, fs, y_prob) for every available combination
        configs: list[tuple[str, str, np.ndarray]] = []
        for config_name in _ALL_CONFIGS:
            features = DATA_CONFIGS.get(config_name, [])
            if not features:
                continue
            X = subset[features]
            for model_name in _ML_MODELS:
                key = f"{model_name}__{config_name}"
                if key not in loaded_models:
                    continue
                y_prob = loaded_models[key].predict_proba(X)[:, 1]
                configs.append((model_name, config_name, y_prob))

        n_cfg = len(configs)
        # Stack probabilities: shape (n, n_cfg)
        prob_mat = np.column_stack([c[2] for c in configs])

        # Observed metrics
        def _flagged(probs: np.ndarray) -> np.ndarray:
            tau = conservative_threshold(probs, train_pos_rate)
            return probs >= tau

        obs_recall = np.full(n_cfg, np.nan)
        obs_prec   = np.full(n_cfg, np.nan)
        obs_auc    = np.full(n_cfg, np.nan)
        n_do = y_true.sum()
        for j in range(n_cfg):
            ef = _flagged(prob_mat[:, j])
            nf = ef.sum()
            tp = (ef & y_true).sum()
            obs_recall[j] = tp / n_do if n_do > 0 else np.nan
            obs_prec[j]   = tp / nf   if nf   > 0 else np.nan
            obs_auc[j]    = _auc_safe(y_true, prob_mat[:, j])

        # Bootstrap
        boot_recall = np.full((n_boot, n_cfg), np.nan)
        boot_prec   = np.full((n_boot, n_cfg), np.nan)
        boot_auc    = np.full((n_boot, n_cfg), np.nan)

        rng = np.random.default_rng(seed)
        for b in range(n_boot):
            idx   = rng.integers(0, n, size=n)
            y_b   = y_true[idx]
            pm_b  = prob_mat[idx]
            n_do_b = y_b.sum()
            for j in range(n_cfg):
                ef_b = _flagged(pm_b[:, j])
                nf_b = ef_b.sum()
                tp_b = (ef_b & y_b).sum()
                boot_recall[b, j] = tp_b / n_do_b if n_do_b > 0 else np.nan
                boot_prec[b, j]   = tp_b / nf_b   if nf_b   > 0 else np.nan
                if len(np.unique(y_b)) >= 2:
                    boot_auc[b, j] = _auc_safe(y_b, pm_b[:, j])

        for j, (model_name, config_name, _) in enumerate(configs):
            for metric, obs_arr, boot_arr in [
                ("enr_recall",    obs_recall, boot_recall),
                ("enr_precision", obs_prec,   boot_prec),
                ("roc_auc",       obs_auc,    boot_auc),
            ]:
                all_rows.append({
                    "model":       model_name,
                    "feature_set": config_name,
                    "faculty":     faculty,
                    "metric":      metric,
                    "observed":    float(obs_arr[j]),
                    "ci_low":      float(np.nanpercentile(boot_arr[:, j], 2.5)),
                    "ci_high":     float(np.nanpercentile(boot_arr[:, j], 97.5)),
                    "n_boot":      n_boot,
                })

        print(f"  {faculty}: {n} enrollments, {n_cfg} configs done")

    return pd.DataFrame(all_rows)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired bootstrap CIs for enrollment-level and row-level metrics."
    )
    parser.add_argument("--year",      type=int,  default=None)
    parser.add_argument("--n-boot",    type=int,  default=_N_BOOT_DEFAULT)
    parser.add_argument("--row-level", action="store_true",
                        help="Also bootstrap row_roc_auc from row_predictions.csv")
    parser.add_argument("--pilot-snapshot",  action="store_true",
                        help="Bootstrap row_roc_auc on the pilot snapshot")
    args = parser.parse_args()

    # ── Enrollment-level (main test set) ─────────────────────────────────────
    preds_df = pd.read_csv(RESULTS_PATH / "enrollment_predictions.csv")
    years = (
        [args.year]
        if args.year
        else sorted(preds_df["test_year"].dropna().unique().astype(int).tolist())
    )

    parts = []
    for year in years:
        n_enr = preds_df.loc[preds_df["test_year"] == year, "enrollment"].nunique()
        print(f"Enrollment-level — year {year} — {n_enr} enrollments ...", flush=True)
        parts.append(bootstrap_year(preds_df, test_year=year, n_boot=args.n_boot))
        print(f"  done")

    # ── Row-level (main test set) ─────────────────────────────────────────────
    if args.row_level:
        row_preds_df = pd.read_csv(RESULTS_PATH / "row_predictions.csv")
        for year in years:
            n_enr = row_preds_df.loc[row_preds_df["test_year"] == year, "enrollment"].nunique()
            print(f"Row-level     — year {year} — {n_enr} enrollments ...", flush=True)
            parts.append(bootstrap_row_year(row_preds_df, test_year=year, n_boot=args.n_boot))
            print(f"  done")

    out      = pd.concat(parts, ignore_index=True)
    out_path = RESULTS_PATH / "bootstrap_cis.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}  ({len(out):,} rows)")

    # ── Pilot snapshot ──────────────────────────────────────────────────────────────
    if args.pilot_snapshot:
        print("\nPilot snapshot bootstrap ...", flush=True)
        tab_out      = bootstrap_pilot_snapshot(n_boot=args.n_boot, seed=RANDOM_SEED)
        tab_out_path = RESULTS_PATH / "bootstrap_cis_pilot_snapshot.csv"
        tab_out.to_csv(tab_out_path, index=False)
        print(f"Saved → {tab_out_path}  ({len(tab_out):,} rows)")


if __name__ == "__main__":
    main()
