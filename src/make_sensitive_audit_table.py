"""
SI audit of the workshop's condition on sensitive attributes.

The workshop accepted sensitive attributes (sex, age, nationality) only if they
gave a LARGE identification gain over the largest accepted non-sensitive feature
set, or demonstrably improved fairness. The accepted set is credits +
performance — enrolment was rejected — so the comparison participants actually
approved is

    b_plus_sensitive  vs  b_credits_performance

per model, on the main test cohort. The five-rung ladder adds sensitive data
only on top of LMS (e_plus_sensitive), so it cannot answer this.
b_plus_sensitive is AUDIT-ONLY (config.AUDIT_ONLY_CONFIGS): never a selection
candidate, so this table cannot change the Participatory model.

The table answers two questions directly:

  (a) accuracy — enrollment-level recall (calibrated tau) and ROC AUC on both
      feature sets, the recall gain in pp with a paired bootstrap CI, and
      whether the gain reaches the workshop's bar;
  (b) fairness — ABROCA for sex, nationality and entry age (relative to cohort)
      on both feature sets, the change with a paired bootstrap CI, and whether
      the gap narrows (CI entirely below 0), widens (above 0) or neither.

Paired bootstrap: the same resample of enrollments scores both feature sets, so
the interval is on the DIFFERENCE, not two overlapping marginal intervals. A
change in ABROCA of a few thousandths is well inside resampling noise; without
the interval, any decrease would read as "sensitive data narrows the gap".

Disclosure: a recall is hidden when its confusion cells rest on 1-2 students
(operating_point_hidden) — and the recall gain with it, since gain plus either
recall gives the other. AUC and ABROCA are hidden when a group's dropouts or
non-dropouts are that few (roc_hidden). An interval is printed only beside a
printed estimate (percentile_ci).

    python src/make_sensitive_audit_table.py [--year 2021] [--n-boot 1000]

Writes
    results/fairness/sensitive_audit_{year}.csv
    results/descriptive/sensitive_audit_table.tex
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from paths.paths import RESULTS_PATH
from src import disclosure
from src.config import MAIN_TEST_YEAR, RANDOM_SEED
from src.display import feature_set_label, model_label
from src.fairness_audit import _BINARY_SUBGROUPS
from src.metrics import ABROCA_N_GRID, abroca, percentile_ci, roc_on_grid
from src.workshop_rules import BASELINE_RULES

BASE_FS  = "b_credits_performance"
AUDIT_FS = "b_plus_sensitive"

MODELS = ["fasterrisk", "logistic_regression", "xgboost", "mlp"]

## The workshop's bar for sensitive attributes. The endline rules never reach
## the sensitive step (LMS is blocked first), so the number lives in the
## baseline survey: its sensitive threshold, aggregated conservatively over the
## three attributes. Read from the rules, never restated here.
GAIN_REQUIRED = BASELINE_RULES.fs_gain_required["e_plus_sensitive"]

_AXIS_DISPLAY = {"female": "Sex", "danish_nationality": "Nationality",
                 "older_for_cohort": "Entry age"}

_OUT_CSV = RESULTS_PATH / "fairness"
_OUT_TEX = RESULTS_PATH / "descriptive" / "sensitive_audit_table.tex"

_FPR_GRID = np.linspace(0, 1, ABROCA_N_GRID)


# ── paired frame ─────────────────────────────────────────────────────────────

def paired_predictions(ep: pd.DataFrame, model: str, year: int) -> pd.DataFrame | None:
    """
    One row per enrollment, both feature sets side by side:
    y, flag_b / score_b (credits + performance), flag_s / score_s (+ sensitive),
    and the three fairness axes.
    """
    sub = ep[(ep["test_year"] == year) & (ep["model"] == model)]
    b = sub[sub["feature_set"] == BASE_FS].set_index("enrollment")
    s = sub[sub["feature_set"] == AUDIT_FS].set_index("enrollment")
    if b.empty or s.empty:
        return None
    df = pd.DataFrame({
        "y":       b["dropped_out"].astype(int),
        "flag_b":  b["ever_flagged"].astype(int),
        "score_b": b["max_prob"].astype(float),
    }).join(pd.DataFrame({"flag_s": s["ever_flagged"].astype(int),
                          "score_s": s["max_prob"].astype(float),
                          "y_s": s["dropped_out"].astype(int)}), how="inner")
    if len(df) != len(b) or len(df) != len(s) or not (df["y"] == df["y_s"]).all():
        raise ValueError(f"{model}: the two feature sets were not scored on the same "
                         f"{year} enrollments — re-run src/evaluation.py")
    return df.drop(columns="y_s").join(b[list(_AXIS_DISPLAY)])


# ── metrics, no disclosure inside (applied at output) ────────────────────────

def _recall(y: np.ndarray, flag: np.ndarray) -> float:
    pos = y == 1
    return float(flag[pos].mean()) if pos.any() else np.nan


def _abroca(y: np.ndarray, score: np.ndarray, g: np.ndarray, ref, comp) -> float:
    curves = []
    for val in (ref, comp):
        m = g == val
        if m.sum() == 0 or len(np.unique(y[m])) < 2:
            return np.nan
        curves.append(roc_on_grid(y[m], score[m], _FPR_GRID)[0])
    return abroca(curves[0], curves[1], _FPR_GRID)


def _abrocas(d: pd.DataFrame) -> dict:
    """ABROCA per axis on both feature sets, over enrollments with a known group."""
    out = {}
    y = d["y"].to_numpy()
    for col in _AXIS_DISPLAY:
        ref, _, comp, _ = _BINARY_SUBGROUPS[col]
        g = d[col].to_numpy(float)
        known = ~np.isnan(g)
        out[col] = tuple(_abroca(y[known], d[score].to_numpy()[known], g[known], ref, comp)
                         for score in ("score_b", "score_s"))
    return out


def audit_model(df: pd.DataFrame, n_boot: int, rng: np.random.Generator) -> list[dict]:
    y = df["y"].to_numpy()
    n, n_pos = len(y), int(y.sum())

    # ── point estimates + disclosure ──────────────────────────────────────────
    rec, hid_rec = {}, {}
    for fs, flag_col in ((BASE_FS, "flag_b"), (AUDIT_FS, "flag_s")):
        flag = df[flag_col].to_numpy()
        rec[fs] = _recall(y, flag)
        hid_rec[fs] = disclosure.operating_point_hidden(
            n, n_pos, int(flag.sum()), int(flag[y == 1].sum()))["recall"]
    auc_hidden = disclosure.is_small(n_pos, n - n_pos)
    auc = {fs: (roc_auc_score(y, df[col]) if 0 < n_pos < n else np.nan)
           for fs, col in ((BASE_FS, "score_b"), (AUDIT_FS, "score_s"))}
    abr = _abrocas(df)

    # ── paired bootstrap of the two differences ───────────────────────────────
    d_rec, d_abr = [], {col: [] for col in _AXIS_DISPLAY}
    for _ in range(n_boot):
        d = df.iloc[rng.integers(0, n, n)]
        yb = d["y"].to_numpy()
        d_rec.append(_recall(yb, d["flag_s"].to_numpy()) - _recall(yb, d["flag_b"].to_numpy()))
        for col, (a_b, a_s) in _abrocas(d).items():
            d_abr[col].append(a_s - a_b)

    rows = []
    lo, hi = percentile_ci(d_rec)
    gain = rec[AUDIT_FS] - rec[BASE_FS]
    rows.append({
        "quantity": "recall", "axis": "",
        "base": rec[BASE_FS], "audit": rec[AUDIT_FS],
        "delta": gain, "ci_lo": lo, "ci_hi": hi,
        "base_hidden": hid_rec[BASE_FS], "audit_hidden": hid_rec[AUDIT_FS],
        ## Gain + one recall gives the other, so the gain goes with either.
        "delta_hidden": hid_rec[BASE_FS] or hid_rec[AUDIT_FS],
        "verdict": ("" if np.isnan(gain) else
                    "reaches bar" if gain >= GAIN_REQUIRED - 1e-12 else "below bar"),
    })
    rows.append({
        "quantity": "auc", "axis": "",
        "base": auc[BASE_FS], "audit": auc[AUDIT_FS],
        "delta": auc[AUDIT_FS] - auc[BASE_FS], "ci_lo": np.nan, "ci_hi": np.nan,
        "base_hidden": auc_hidden, "audit_hidden": auc_hidden, "delta_hidden": auc_hidden,
        "verdict": "",
    })
    for col, (a_b, a_s) in abr.items():
        ref, _, comp, _ = _BINARY_SUBGROUPS[col]
        known = df[df[col].notna()]
        hidden = disclosure.roc_hidden(known, col, [ref, comp], outcome="y")
        lo, hi = percentile_ci(d_abr[col])
        verdict = ("" if np.isnan(lo) else
                   "narrower" if hi < 0 else "wider" if lo > 0 else "no clear change")
        rows.append({
            "quantity": "abroca", "axis": col,
            "base": a_b, "audit": a_s, "delta": a_s - a_b, "ci_lo": lo, "ci_hi": hi,
            "base_hidden": hidden, "audit_hidden": hidden, "delta_hidden": hidden,
            "verdict": verdict,
        })
    return rows


def build(ep: pd.DataFrame, year: int, n_boot: int) -> pd.DataFrame:
    if AUDIT_FS not in set(ep["feature_set"]):
        raise SystemExit(
            f"no {AUDIT_FS} predictions in enrollment_predictions.csv. It is a new config: "
            f"train it (python src/evaluation.py --config-index <its index>, then "
            f"--merge) before building this table."
        )
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for model in MODELS:
        df = paired_predictions(ep, model, year)
        if df is None:
            print(f"  {model}: no predictions on both feature sets — skipped")
            continue
        for r in audit_model(df, n_boot, rng):
            rows.append({"model": model, "test_year": year, "n_boot": n_boot,
                         "gain_required": GAIN_REQUIRED, **r})
    return pd.DataFrame(rows)


# ── LaTeX ────────────────────────────────────────────────────────────────────

def _num(v, hidden: bool, spec: str = "{:.3f}") -> str:
    if hidden:
        return disclosure.SUPPRESSED_TEX
    return "--" if pd.isna(v) else spec.format(v)


def _delta(r, scale: float, spec: str) -> str:
    if r["delta_hidden"]:
        return disclosure.SUPPRESSED_TEX
    if pd.isna(r["delta"]):
        return "--"
    def fmt(v):
        s = spec.format(v * scale)
        ## A value that rounds to zero prints as +0, never as "-0.000".
        if float(s) == 0:
            s = spec.format(0.0)
        return s.replace("-", "$-$")
    s = fmt(r["delta"])
    if pd.notna(r["ci_lo"]) and pd.notna(r["ci_hi"]):
        s += f" [{fmt(r['ci_lo'])}, {fmt(r['ci_hi'])}]"
    return s


def to_latex(table: pd.DataFrame, year: int, n_boot: int) -> str:
    base_lbl, audit_lbl = feature_set_label(BASE_FS), feature_set_label(AUDIT_FS)
    bar_pp = GAIN_REQUIRED * 100
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\footnotesize",
        r"\textbf{(a) Identification}\\[2pt]",
        r"\begin{tabular}{lrrrrrl}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Recall} & & \multicolumn{2}{c}{ROC AUC} & \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){5-6}",
        rf"Model & {base_lbl} & {audit_lbl} & $\Delta$ recall (pp) [95\% CI] "
        rf"& {base_lbl} & {audit_lbl} & $\geq${bar_pp:.0f}\,pp? \\",
        r"\midrule",
    ]
    for model, g in table.groupby("model", sort=False):
        r = g[g["quantity"] == "recall"].iloc[0]
        a = g[g["quantity"] == "auc"].iloc[0]
        verdict = ("" if r["delta_hidden"] or not r["verdict"]
                   else "Yes" if r["verdict"] == "reaches bar" else "No")
        lines.append(
            f"{model_label(model)} & {_num(r['base'], r['base_hidden'])} "
            f"& {_num(r['audit'], r['audit_hidden'])} & {_delta(r, 100, '{:+.1f}')} "
            f"& {_num(a['base'], a['base_hidden'])} & {_num(a['audit'], a['audit_hidden'])} "
            f"& {verdict} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\\[8pt]",
        r"\textbf{(b) Subgroup gaps (ABROCA)}\\[2pt]",
        r"\begin{tabular}{llrrrl}",
        r"\toprule",
        rf"Model & Groups & {base_lbl} & {audit_lbl} & $\Delta$ ABROCA [95\% CI] & Change \\",
        r"\midrule",
    ]
    for i, (model, g) in enumerate(table.groupby("model", sort=False)):
        if i:
            lines.append(r"\addlinespace")
        for j, r in enumerate(g[g["quantity"] == "abroca"].itertuples()):
            _, ref_lbl, _, comp_lbl = _BINARY_SUBGROUPS[r.axis]
            r = r._asdict()
            verdict = "" if r["delta_hidden"] else r["verdict"]
            lines.append(
                f"{model_label(model) if j == 0 else ''} & {comp_lbl} vs {ref_lbl} "
                f"& {_num(r['base'], r['base_hidden'])} & {_num(r['audit'], r['audit_hidden'])} "
                f"& {_delta(r, 1, '{:+.3f}')} & {verdict} \\\\"
            )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{\textbf{Audit of the workshop's condition on sensitive attributes, "
        + str(year) + r" test cohort.} "
        r"Each model is fitted on credits + performance, the largest feature set the "
        r"workshop accepted, without (" + base_lbl + r") and with (" + audit_lbl + r") "
        r"the sensitive attributes (sex, age at enrollment, nationality). The workshop "
        r"accepted sensitive attributes only for a large identification gain "
        rf"(at least {bar_pp:.0f} percentage points of recall) or a demonstrable "
        r"improvement in fairness. "
        r"(a) Enrollment-level recall at the calibrated threshold and ROC AUC. "
        r"(b) ABROCA between subgroups; entry age is relative to the student's cohort. "
        r"A negative change means the sensitive attributes narrow the gap; ``narrower'' "
        r"and ``wider'' are stated only where the interval excludes zero. "
        rf"Intervals are 95\% paired bootstrap intervals over enrollments ({n_boot} "
        r"resamples). This configuration is an audit only and was never a candidate "
        r"in the model selection.}",
        r"\label{tab:si_sensitive_audit}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, default=MAIN_TEST_YEAR)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    ep = pd.read_csv(RESULTS_PATH / "enrollment_predictions.csv")
    table = build(ep, args.year, args.n_boot)

    _OUT_CSV.mkdir(parents=True, exist_ok=True)
    csv = _OUT_CSV / f"sensitive_audit_{args.year}.csv"
    table.to_csv(csv, index=False)
    print(f"Saved -> {csv}")
    _OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    _OUT_TEX.write_text(to_latex(table, args.year, args.n_boot), encoding="utf-8")
    print(f"Saved -> {_OUT_TEX}")

    for model, g in table.groupby("model", sort=False):
        r = g[g["quantity"] == "recall"].iloc[0]
        print(f"  {model:<20} recall gain {r['delta'] * 100:+.1f} pp "
              f"[{r['ci_lo'] * 100:+.1f}, {r['ci_hi'] * 100:+.1f}]  ({r['verdict']})  "
              + "  ".join(f"{_AXIS_DISPLAY[a.axis]}: {a.verdict}"
                          for a in g[g["quantity"] == "abroca"].itertuples()))


if __name__ == "__main__":
    main()
