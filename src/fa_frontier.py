"""
Fairness-accuracy (FA) frontier, following the framework cited in the paper.

Three panels, one per fairness axis, in one row:

    (a) sex          — x = female error rate,     y = male error rate
    (b) nationality  — x = non-Danish error rate, y = Danish error rate
    (c) entry age    — x = older error rate,      y = younger error rate

Group 1 (the x axis) is always the comparison group and group 0 (the y axis)
the reference, the same direction as every other output: Female − Male,
Non-Danish − Danish, Older − Younger (fairness_audit._BINARY_SUBGROUPS).

Methodology
-----------
1. Group loss is plain 0-1 misclassification error within each group,
   e_g = 1 - accuracy_g, computed at the **enrollment** level (an enrollment is
   flagged if its max-prob over EWS dates clears τ). No reweighting, no balanced
   error.

2. The *achievable set* is the set of (e_1, e_0) pairs reachable by some
   admissible algorithm, closed under randomisation. We enumerate it as

     - every (model, feature set, threshold) combination, with the threshold
       swept over all achievable score values, including the two constant rules
       (flag nobody / flag everybody);
     - the credits policy swept over its ECTS-behind cutoff, not just the legal
       ≥60 one;
     - for feature sets that *contain* the sensitive attribute, the additional
       points reachable with group-specific thresholds (once the algorithm can condition on group identity the achievable set is
       a rectangle, and the frontier is the line from r_X = b_X to f_X);

   then take the convex hull. Convexification is what randomisation buys, and it
   is what makes the boundary a frontier rather than a scatter.

3. The FA frontier is the lower-left (Pareto) boundary of that hull between the
   two group-optimal points, extended along the boundary to the fairest
   achievable point f_X when the information structure is group-skewed.
   f_X is the lower-left crossing of the hull boundary with the 45° line — the
   smallest common error rate attainable with equal errors across groups.
   Group-skew is exactly the case where f_X is *not* on the Pareto arc: both
   group-optimal algorithms err more on the same group, so closing the gap costs
   accuracy for everyone.

4. Two nested algorithm sets are drawn per panel:

     - unconstrained         — all four ML models × all feature sets, plus the
                               credits policy;
     - workshop-permissible  — only combinations passing the two workshop gates
                               (data-input and explainability), resolved live
                               from src.workshop_rules.select_grid_config().

   A smaller algorithm set gives a subset of the achievable set, hence a weakly
   dominated frontier. The shaded band between the two is the cost of the
   participation constraints, expressed in group error rates.

Point labels use the paper's notation: the two group-optimal points and f_X.
No descriptive prose is drawn inside the axes — every number a caption or the
body text might want is written to
results/fairness/fa_frontier_summary_{year}.csv instead.

Usage
-----
    python src/fa_frontier.py                    # MAIN_TEST_YEAR (2021)
    python src/fa_frontier.py --year 2022
    python src/fa_frontier.py --n-boot 0         # skip the group-skew bootstrap
    python src/fa_frontier.py --zoom             # frame on the frontiers only

The view is framed on the frontiers plus the deployed operating points, but an
operating point that would stretch the range more than 3x is dropped from the
framing and reported on stdout instead — a point far from the frontiers would
otherwise crush every frontier into a corner.

The figure is prepared by figures/scripts/prepare_fa_frontier.py and drawn by
figures/scripts/draw_fa_frontier.py.
"""

import re
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import paper_style
from src import disclosure
from src.display import MODEL_LABELS
from src.config import AUDIT_ONLY_CONFIGS, MAX_ML_FS, MAX_ML_MODEL, RANDOM_SEED

# ── style ─────────────────────────────────────
## The paper style, shared with every figure (src/paper_style.py).
paper_style.apply_style()

# Wong colorblind-safe palette
_C_UNCONSTRAINED = "#0072B2"   # blue      — unconstrained frontier
_C_PERMISSIBLE   = "#D55E00"   # vermilion — workshop-permissible frontier
_C_BAND          = "#E69F00"   # orange    — cost-of-constraints band
_C_CLOUD         = "#999999"   # grey      — candidate algorithms
_C_CREDITS       = "#000000"   # black     — credits policy
_C_PARTICIPATORY = "#CC79A7"   # pink      — participatory model
_C_MAXML         = "#009E73"   # green     — max ML

_MODEL_LABELS = {
    "logistic_regression": "Logistic regression",
    "xgboost":             "XGBoost",
    "mlp":                 MODEL_LABELS["mlp"],   # one name, in src/display.py
    "fasterrisk":          "FasterRisk",
    "credits_policy":      "Credits policy",
}
_FS_LABELS = {
    "a_credits":             "Credits only",
    "b_credits_performance": "+ Performance",
    "c_plus_enrolment":      "+ Enrollment",
    "d_plus_lms":            "+ LMS",
    "e_plus_sensitive":      "+ Sensitive",
    "used_vars":             "Used vars",
}

# Panel definitions. `sensitive_in` lists the feature sets that contain this
# attribute, i.e. the ones for which group-specific thresholds are admissible.
_SUBGROUPS = {
    "sex": dict(
        col="female", g1=1, g0=0,
        lbl1="Female", lbl0="Male",
        sym1=r"$w_X$", sym0=r"$m_X$",
        sensitive_in={"e_plus_sensitive"},
    ),
    "nationality": dict(
        col="danish_nationality", g1=0, g0=1,
        lbl1="Non-Danish", lbl0="Danish",
        sym1=r"$n_X$", sym0=r"$d_X$",
        sensitive_in={"e_plus_sensitive"},
    ),
    ## Entry age relative to cohort (src/data.py). e_plus_sensitive admits
    ## group-specific thresholds here too: within one test cohort the group is a
    ## function of age_at_study_start and undergraduate, both of which it holds.
    "age": dict(
        col="older_for_cohort", g1=1, g0=0,
        lbl1="Older", lbl0="Younger",
        sym1=r"$o_X$", sym0=r"$y_X$",
        sensitive_in={"e_plus_sensitive"},
    ),
}

_OPS_SPAN_TOLERANCE = 3.0  # how far an operating point may stretch the view
                           # before it is dropped from the framing (see
                           # _axis_limits); it is still reported, not hidden


# ── group error rates ────────────────────────────────────────────────────────

def _error_curve(scores: np.ndarray, y: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """
    0-1 misclassification error of the rule `flag iff score >= τ`, for every τ.

    Exact over the whole grid: an enrollment is misclassified when it dropped
    out and scored below τ, or did not drop out and scored at or above τ.
    """
    pos = np.sort(scores[y == 1])
    neg = np.sort(scores[y == 0])
    n = len(y)
    if n == 0:
        return np.full(len(taus), np.nan)
    fn = np.searchsorted(pos, taus, side="left")            # positives below τ
    fp = len(neg) - np.searchsorted(neg, taus, side="left")  # negatives at/above τ
    return (fn + fp) / n


def _small_cell_taus(scores: np.ndarray, y: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """
    Per τ: does the rule `flag iff score >= τ` leave a confusion cell holding
    1..MIN_CELL_SIZE-1 students in this group? Same counting as _error_curve.
    """
    pos = np.sort(scores[y == 1])
    neg = np.sort(scores[y == 0])
    fn = np.searchsorted(pos, taus, side="left")
    tn = np.searchsorted(neg, taus, side="left")
    return disclosure.confusion_is_small(len(pos) - fn, len(neg) - tn, fn, tn)


def _disclosable_taus(scores: np.ndarray, y: np.ndarray, g: np.ndarray, g1, g0) -> np.ndarray:
    """
    The threshold grid minus every τ that leaves a confusion cell of fewer than
    MIN_CELL_SIZE (but more than zero) students in either group. Disclosure control: a frontier point is two groups'
    confusion matrices at one τ, so those thresholds are not part of the
    achievable set drawn or used (the constant rules keep only zero cells and
    always survive). Every caller that indexes _sweep's output by τ must use
    this grid, not _threshold_grid. See src/disclosure.py.
    """
    taus = _threshold_grid(scores)
    m1, m0 = g == g1, g == g0
    keep = ~(_small_cell_taus(scores[m1], y[m1], taus) | _small_cell_taus(scores[m0], y[m0], taus))
    return taus[keep]


def _threshold_grid(scores: np.ndarray) -> np.ndarray:
    """
    Every achievable cutoff, plus the two constant rules.

    The distinct score values are the only cutoffs that change any decision, so
    they enumerate the admissible deterministic rules exactly — all of them, with
    no cap. _error_curve is vectorised, so the full set is cheap.
    """
    uniq = np.unique(scores[np.isfinite(scores)])
    if len(uniq) == 0:
        return np.array([-np.inf, np.inf])
    # -inf flags everybody, +inf flags nobody: the two no-information rules,
    # which are achievable regardless of the model and anchor the hull.
    return np.concatenate([[-np.inf], uniq, [np.inf]])


def _sweep(scores: np.ndarray, y: np.ndarray, g: np.ndarray,
           g1, g0, group_aware: bool, with_rules: bool = False):
    """
    (e_1, e_0) pairs achievable by thresholding `scores`.

    Returns
    -------
    common : (K, 2) points from a single threshold applied to both groups
    aware  : (4, 2) or (0, 2) — group-specific thresholds, admissible only when
             the group variable is itself an input. A group-specific rule is a
             PAIR (τ_1, τ_0), and it reaches (e_1(τ_1), e_0(τ_0)) for any pair
             independently, so the achievable set is the product of the two
             groups' error sets; its convex hull is the rectangle
             [min e_1, max e_1] × [min e_0, max e_0].
             All four vertices are returned. The lower-left one alone is not
             enough: a common threshold ties e_1 to e_0, so the other vertices
             are not convex combinations of it and the common-threshold points,
             and without the edges leaving it f_X is overstated.

    With `with_rules`, also the rule behind every point as (τ_1, τ_0) arrays:
    common_rules (K, 2) with τ_1 = τ_0, aware_rules (4, 2) or (0, 2).
    """
    taus = _disclosable_taus(scores, y, g, g1, g0)
    m1, m0 = g == g1, g == g0
    e1 = _error_curve(scores[m1], y[m1], taus)
    e0 = _error_curve(scores[m0], y[m0], taus)
    common = np.column_stack([e1, e0])
    common_rules = np.column_stack([taus, taus])
    if group_aware:
        i1 = (int(np.nanargmin(e1)), int(np.nanargmax(e1)))
        i0 = (int(np.nanargmin(e0)), int(np.nanargmax(e0)))
        pairs = [(a, b) for a in i1 for b in i0]      # (lo,lo) (lo,hi) (hi,lo) (hi,hi)
        aware = np.array([[e1[a], e0[b]] for a, b in pairs])
        aware_rules = np.array([[taus[a], taus[b]] for a, b in pairs])
    else:
        aware = aware_rules = np.empty((0, 2))
    if with_rules:
        return common, aware, common_rules, aware_rules
    return common, aware


def _operating_point(df: pd.DataFrame, col: str, g1, g0) -> tuple[float, float]:
    """
    Group error rates at the deployed rule (calibrated τ, `ever_flagged`).
    (nan, nan) — not drawn — if either group has a confusion cell of fewer than
    MIN_CELL_SIZE (but more than zero) students.
    """
    m1, m0 = df[col] == g1, df[col] == g0
    for m in (m1, m0):
        fl, do = df.loc[m, "ever_flagged"] == 1, df.loc[m, "dropped_out"] == 1
        if disclosure.confusion_is_small((fl & do).sum(), (fl & ~do).sum(),
                                         (~fl & do).sum(), (~fl & ~do).sum()).any():
            return np.nan, np.nan
    e1 = float(np.mean(df.loc[m1, "ever_flagged"] != df.loc[m1, "dropped_out"]))
    e0 = float(np.mean(df.loc[m0, "ever_flagged"] != df.loc[m0, "dropped_out"]))
    return e1, e0


def _credits_enrollments(raw_year: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    One row per enrollment: worst ECTS deficit, outcome, group.

    An enrollment with no ects_behind at any activation scores -inf: the policy
    never flags a student with no recorded deficit, and that is how the main
    evaluation counts it. Left NaN, it sorted above every cutoff in _error_curve
    and counted as flagged at all of them, "flag nobody" included.
    """
    out = raw_year.groupby("enrollment").agg(
        score=("ects_behind", "max"),
        dropped_out=("dropout", "max"),
        **{col: (col, "first")},
    ).reset_index()
    out["score"] = out["score"].fillna(-np.inf)
    return out


# ── achievable set and frontier geometry ─────────────────────────────────────

def _hull_vertices(points: np.ndarray) -> np.ndarray:
    """Convex hull vertices in counter-clockwise order (scipy's 2-D convention)."""
    return points[ConvexHull(points).vertices]


def _walk(V: np.ndarray, i: int, j: int) -> np.ndarray:
    """Vertices from index i to index j walking counter-clockwise around V."""
    n = len(V)
    return V[[(i + k) % n for k in range((j - i) % n + 1)]]


def _pareto_arc(V: np.ndarray) -> tuple[np.ndarray, int, int]:
    """
    Lower-left boundary of the hull: the achievable points nothing dominates.

    The two endpoints are the group-optimal algorithms — lexicographic minima,
    so a weakly dominated axis-parallel edge (same e_1, worse e_0) is excluded
    rather than drawn as frontier. On a counter-clockwise hull, walking forward
    from the e_1-optimal vertex reaches the e_0-optimal one along exactly this
    boundary. When one point minimises both errors the arc collapses to it.
    """
    i1 = int(np.lexsort((V[:, 1], V[:, 0]))[0])   # min e_1, ties → min e_0
    i0 = int(np.lexsort((V[:, 0], V[:, 1]))[0])   # min e_0, ties → min e_1
    return _walk(V, i1, i0), i1, i0


def _fairest_point(V: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    f_X — the fairest achievable point.

    Normally that is where the boundary meets the 45° line: the 45° line can cut
    a convex hull in at most two places, and the fairer of them is the one
    nearer the origin. If the algorithm set never equalises the two error rates
    at all, f_X falls back to the boundary vertex with the smallest gap, and the
    second return value is False.
    """
    C = np.vstack([V, V[:1]])
    diff = C[:, 0] - C[:, 1]
    hits = []
    for i in range(len(C) - 1):
        a, b = diff[i], diff[i + 1]
        if a == 0:
            hits.append(C[i])
        elif a * b < 0:
            hits.append(C[i] + (a / (a - b)) * (C[i + 1] - C[i]))
    if hits:
        return min(hits, key=lambda p: p[0]), True
    gap = np.abs(V[:, 0] - V[:, 1])
    return V[int(np.lexsort((V[:, 0], gap))[0])], False


def _extend_to_fair(V: np.ndarray, start: int, f: np.ndarray,
                    forward: bool) -> np.ndarray:
    """
    Boundary arc from a group-optimal vertex to f_X, in the direction that
    closes the error gap.

    Under group skew both group-optimal algorithms favour the same group, so
    reaching equal errors means leaving the Pareto boundary: every step raises
    both error rates and is bought purely for fairness. That arc is part of the
    FA frontier, which is why it is drawn.
    """
    n = len(V)
    step = 1 if forward else -1
    path = [V[start]]
    for k in range(1, n + 1):
        cur = V[(start + step * k) % n]
        prev = path[-1]
        a, b = prev[0] - prev[1], cur[0] - cur[1]
        if a == 0 or a * b < 0:                    # this edge reaches the diagonal
            break
        path.append(cur)
        if np.allclose(cur, f):
            break
    if not np.allclose(path[-1], f):
        path.append(f)
    return np.array(path)


def compute_frontier(points: np.ndarray) -> dict:
    """
    FA frontier of an algorithm set, given the (e_1, e_0) pairs it can achieve.

    Returns
    -------
    dict with
        hull      — hull vertices (the achievable set, after randomisation)
        pareto    — lower-left boundary between the two group-optimal points
        path      — the FA frontier: `pareto`, extended to f_X when group-skewed
        p1, p0    — the e_1-optimal and e_0-optimal points
        f         — f_X
        f_exact   — False when equal error rates are unachievable and f_X is only
                    the smallest attainable gap
        skewed    — True when both group-optimal algorithms err more on the same
                    group, so no point on the Pareto boundary equalises errors
        skew_side — "group1" / "group0" / None: the group both optima favour
    """
    V = _hull_vertices(points)
    pareto, i1, i0 = _pareto_arc(V)
    p1, p0 = V[i1], V[i0]
    f, f_exact = _fairest_point(V)

    # convexity: the arc between the two lexicographic optima cannot cross the
    # diagonal if its endpoints share a side, so testing the endpoints suffices
    d1, d0 = p1[0] - p1[1], p0[0] - p0[1]
    skewed = bool(d1 != 0 and d0 != 0 and np.sign(d1) == np.sign(d0))
    skew_side = ("group1" if d1 < 0 else "group0") if skewed else None

    path = pareto
    if skewed:
        if d1 < 0:      # both errors lower for group 1 → raise e_1 past the e_0 optimum
            path = np.vstack([pareto, _extend_to_fair(V, i0, f, forward=True)[1:]])
        else:           # mirror image: raise e_0 past the e_1 optimum
            ext = _extend_to_fair(V, i1, f, forward=False)
            path = np.vstack([ext[::-1][:-1], pareto])

    return dict(hull=V, pareto=pareto, path=path, p1=p1, p0=p0, f=f,
                f_exact=f_exact, skewed=skewed, skew_side=skew_side)


# ── group-skew bootstrap ─────────────────────────

def _skew_pvalue(scores: np.ndarray, y: np.ndarray, g: np.ndarray,
                 g_favoured, g_other, tau: float, n_boot: int,
                 seed: int = RANDOM_SEED, tau_other: float | None = None) -> float:
    """
    Bootstrap p-value for group-skew at a fixed algorithm: hold the algorithm
    fixed, resample the test set within each group, recompute both group errors,
    and centre the bootstrap distribution on the point estimate.

    The algorithm used is the one optimal for the group the
    point estimate says is *worse* off, and the statistic is how much lower the
    favoured group's error is under it. H0 is no skew, so a small p means the
    gap survives sampling noise — i.e. the FA frontier really does extend beyond
    the group-optimal points and a fairness–accuracy tradeoff exists.

    `tau` applies to the favoured group and `tau_other` (default: the same) to
    the other, so a group-specific rule is tested as the rule it is.
    """
    if n_boot <= 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    mf, mo = g == g_favoured, g == g_other
    yf, sf = y[mf], scores[mf]
    yo, so = y[mo], scores[mo]
    nf, no = len(yf), len(yo)
    if nf == 0 or no == 0:
        return float("nan")

    tau_o = tau if tau_other is None else tau_other

    def err(yy, ss, idx, t):
        return float(np.mean((ss[idx] >= t).astype(int) != yy[idx]))

    obs = err(yo, so, np.arange(no), tau_o) - err(yf, sf, np.arange(nf), tau)
    draws = np.array([err(yo, so, rng.integers(0, no, no), tau_o) -
                      err(yf, sf, rng.integers(0, nf, nf), tau)
                      for _ in range(n_boot)])
    return float(np.mean(draws - obs > obs))


# ── assembling the two algorithm sets ────────────────────────────────────────

def _permissible_combos(results_df: pd.DataFrame, year: int) -> set[tuple[str, str]]:
    """
    (model, feature_set) pairs passing both workshop gates, resolved live from
    the selection rules rather than hardcoded — the gates depend on the recall
    ladder in evaluation_results.csv and so on the data at hand.
    """
    from src.workshop_rules import select_grid_config
    _, _, candidates = select_grid_config(results_df, year)
    ok = candidates[candidates["permissible"]]
    return {(r.model, r.feature_set) for r in ok.itertuples()}


def build_subgroup_data(ep: pd.DataFrame, raw_df: pd.DataFrame,
                        results_df: pd.DataFrame, year: int, subgroup: str,
                        n_boot: int = 1000) -> dict:
    """Everything one panel needs: both frontiers, operating points, skew test."""
    from src.fairness_audit import _resolve_participatory_model
    from src.plotting import _ML_MODELS

    spec = _SUBGROUPS[subgroup]
    col, g1, g0 = spec["col"], spec["g1"], spec["g0"]

    ep_year = ep[ep["test_year"] == year]
    permissible = _permissible_combos(results_df, year)

    all_pts, perm_pts = [], []
    ## The rule behind every row of the achievable set, row for row with
    ## all_pts, so the skew test can run on whichever rule the hull puts at the
    ## group optimum — a group-specific pair (τ_1, τ_0), or a credits cutoff.
    sources: list[dict] = []

    for (model, fs), grp in ep_year.groupby(["model", "feature_set"], sort=True):
        if model not in _ML_MODELS:
            continue                                  # the credits policy: its sweep is below
        if fs in AUDIT_ONLY_CONFIGS:
            continue                                  # b_plus_sensitive: SI audit only
        sub = grp.dropna(subset=["max_prob", col])
        if sub.empty:
            continue
        scores = sub["max_prob"].to_numpy(float)
        y = sub["dropped_out"].to_numpy(int)
        g = sub[col].to_numpy()
        aware_ok = fs in spec["sensitive_in"]
        common, aware, c_rules, a_rules = _sweep(scores, y, g, g1, g0, aware_ok,
                                                 with_rules=True)
        all_pts.append(np.vstack([common, aware]))
        for t1, t0 in np.vstack([c_rules, a_rules]):
            sources.append(dict(model=model, feature_set=fs, tau1=float(t1), tau0=float(t0),
                                scores=scores, y=y, g=g))
        if (model, fs) in permissible:
            perm_pts.append(common)                   # no sensitive attribute → no group-aware rule

    # credits policy — always permissible, swept over its ECTS cutoff
    raw_year = raw_df[raw_df["start_year"] == year]
    cp = _credits_enrollments(raw_year, col)
    cp_s, cp_y, cp_g = cp["score"].to_numpy(float), cp["dropped_out"].to_numpy(int), cp[col].to_numpy()
    cp_common, _, cp_rules, _ = _sweep(cp_s, cp_y, cp_g, g1, g0, False, with_rules=True)
    all_pts.append(cp_common)
    perm_pts.append(cp_common)
    for t1, t0 in cp_rules:
        sources.append(dict(model="credits_policy", feature_set=None, tau1=float(t1),
                            tau0=float(t0), scores=cp_s, y=cp_y, g=cp_g))

    ALL = np.vstack(all_pts)
    PERM = np.vstack(perm_pts)

    fr_all = compute_frontier(ALL)
    fr_perm = compute_frontier(PERM)

    def rule_at(point: np.ndarray) -> dict:
        """The first rule (in model/feature-set order) that achieves this hull vertex."""
        return sources[int(np.flatnonzero((ALL == point).all(axis=1))[0])]

    best = {"e1": rule_at(fr_all["p1"]), "e0": rule_at(fr_all["p0"])}

    # deployed operating points
    p_model, p_fs = _resolve_participatory_model()
    ops = {}
    for label, (model, fs) in (("Credits policy", ("credits_policy", None)),
                               ("Participatory", (p_model, p_fs)),
                               ("Max ML", (MAX_ML_MODEL, MAX_ML_FS))):
        sel = ep_year[ep_year["model"] == model]
        if fs is not None:
            sel = sel[sel["feature_set"] == fs]
        sel = sel.dropna(subset=[col])
        ops[label] = (_operating_point(sel, col, g1, g0) if not sel.empty
                      else (np.nan, np.nan))

    # Group-skew test — only meaningful when the point estimate is skewed; run at
    # the algorithm optimal for the group the frontier says is worse off.
    p_skew = float("nan")
    if fr_all["skewed"]:
        favoured, other = ((g1, g0) if fr_all["skew_side"] == "group1" else (g0, g1))
        ref = best["e0" if fr_all["skew_side"] == "group1" else "e1"]
        tau_f, tau_o = ((ref["tau1"], ref["tau0"]) if favoured == g1
                        else (ref["tau0"], ref["tau1"]))
        p_skew = _skew_pvalue(ref["scores"], ref["y"], ref["g"],
                              favoured, other, tau_f, n_boot, tau_other=tau_o)

    return dict(spec=spec, cloud=ALL, frontier_all=fr_all, frontier_perm=fr_perm,
                ops=ops, participatory=(p_model, p_fs),
                optimal_algorithms={k: {kk: vv for kk, vv in v.items()
                                        if kk in ("model", "feature_set", "tau1", "tau0")}
                                    for k, v in best.items()},
                p_skew=p_skew)


# ── plot ─────────────────────────────────────────────────────────────────────

def _axis_limits(panels: list, pad: float = 0.12,
                 include_ops: bool = True) -> tuple[float, float]:
    """
    One set of limits for every panel in the figure, on both axes.

    Two things are being forced here. Within a panel, x and y share a range so
    the 45° equal-error line is a true diagonal and a vertical gap means the
    same as a horizontal one. Across panels, the range is shared so the sex,
    nationality and age panels can be read against each other — a disparity that looks
    twice as wide really is twice as wide.

    The range is set from the frontiers, f_X and the deployed operating points,
    not from the candidate cloud: sweeping τ to its extremes includes the two
    constant rules, whose error rates run up to ~1 − base rate, and framing on
    those would squeeze every object of interest into a corner.

    The operating points are included **only while they stay near the frontiers**
    (within `_OPS_SPAN_TOLERANCE`× the frontier span). A point far from the
    frontiers would compress every frontier into an unreadable corner; when that
    happens the view falls back to the frontiers alone. `include_ops=False`
    (CLI `--zoom`) forces that fallback. Either way the caller reports any point
    left outside the view — nothing is ever silently clipped.
    """
    def bounds(with_ops: bool) -> tuple[float, float]:
        pts = []
        for d in panels:
            pts += [d["frontier_all"]["path"], d["frontier_perm"]["path"],
                    np.atleast_2d(d["frontier_all"]["f"]),
                    np.atleast_2d(d["frontier_perm"]["f"])]
            if with_ops:
                pts += [np.atleast_2d(e) for e in d["ops"].values()
                        if np.isfinite(e).all()]
        P = np.vstack(pts)
        return float(np.nanmin(P)), float(np.nanmax(P))

    lo_f, hi_f = bounds(False)
    lo, hi = bounds(include_ops)
    if include_ops and (hi - lo) > _OPS_SPAN_TOLERANCE * max(hi_f - lo_f, 1e-9):
        lo, hi = lo_f, hi_f
    span = max(hi - lo, 1e-3)
    return max(0.0, lo - pad * span), hi + pad * span


def _label_width_pt(sym: str) -> float:
    """Rough rendered width of a mathtext label at fontsize 10, in points."""
    return 6.5 * len(re.sub(r"\\[a-z]+|[${}_^]", "", sym))


def label_offsets(named: list, span: float, away_from: np.ndarray,
                  lim: tuple[float, float] | None = None,
                  avoid: list | None = None,
                  pts_per_span: float = 250.0) -> list[tuple[float, float]]:
    """
    Where each named frontier point's label goes, as an offset in points.

    Each label is placed on the outward side of the frontier, and any that would
    still collide are moved. f_X can sit very close to the group-optimal
    point whose arc it terminates, which is exactly when two labels land on top
    of each other; and a label dropped on the inward side lands on
    the frontier line.

    Collisions are tested between label boxes, sized from the label's text, and
    against every named marker, so labels of points a few points apart in a
    corner do not run into one another. With `lim`, a box must also stay inside
    the axes, so a point on the axis does not push its label into the ticks.
    `avoid` lists other drawn markers (the deployed operating points), so a
    label is never read as naming one of them.
    If no candidate is clear, the one overlapping least is used.

    Returns offsets rather than drawing them, so the figure's prepared input can
    carry the placement and the draw script does not have to reimplement this.
    It is geometry over points the figure already shows, not a quantity.
    """
    scale = span / pts_per_span                       # data units per point
    to_pt = lambda xy: np.asarray(xy, float) / scale  # noqa: E731
    h = 11.0                                          # text height incl. descent
    ## (centre, half-width) of each marker a label must not touch. Other
    ## markers get a wider berth: a label close to one reads as naming it.
    markers = [(to_pt(pt), 6.0) for pt, _ in named]
    markers += [(to_pt(pt), 12.0) for pt in (avoid or []) if np.isfinite(pt).all()]

    def overlap(a, b):
        return (max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
                * max(0.0, min(a[3], b[3]) - max(a[1], b[1])))

    placed, chosen_all = [], []
    for pt, sym in named:
        w = _label_width_pt(sym)
        p = to_pt(pt)
        out = pt - away_from
        norm = np.linalg.norm(out)
        out = out / norm if norm > 0 else np.array([1.0, 0.0])
        cands = [np.array(o, float) for o in
                 ((8, 4), (8, -14), (-w - 6, 4), (-w - 6, -14),
                  (8, 15), (-w - 6, 15), (8, -25), (-w - 6, -25),
                  (-w / 2, 10), (-w / 2, -20))]
        cands.sort(key=lambda o: -float(o / np.linalg.norm(o) @ out))

        def cost(o):
            x0, y0 = p + o
            box = (x0, y0 - 3, x0 + w, y0 - 3 + h)
            c = sum(overlap(box, b) for b in placed)
            c += sum(overlap(box, (m[0] - r, m[1] - r, m[0] + r, m[1] + r))
                     for m, r in markers)
            if lim is not None:
                lo, hi = to_pt(lim[0]), to_pt(lim[1])
                inside = (max(lo, box[0]), max(lo, box[1]), min(hi, box[2]), min(hi, box[3]))
                c += w * h - max(0.0, inside[2] - inside[0]) * max(0.0, inside[3] - inside[1])
            return c, box

        scored = [(cost(o), i, o) for i, o in enumerate(cands)]
        clear = [t for t in scored if t[0][0] == 0]
        (_, box), _, chosen = clear[0] if clear else min(scored, key=lambda t: (t[0][0], t[1]))
        placed.append(box)
        chosen_all.append((float(chosen[0]), float(chosen[1])))
    return chosen_all
