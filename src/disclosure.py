"""
Statistical disclosure control for every table and figure the paper exports.

One rule, applied before a file is written: a number or plotted element that
rests on fewer than MIN_CELL_SIZE students is not shown. Tables print the
marker in its place; figures leave the element out and put the marker where
it would have been.

"Rests on" means every count the number is computed from: a rate's numerator
and denominator, a recall's dropouts in the group (and the flagged ones among
them), an AUC's positives and negatives. A true zero is shown — "0 of 40" is a
statement about 40 students — so 1..MIN_CELL_SIZE-1 is suppressed, and so is a
count that is present but unknown (NaN): a number nobody can certify as
resting on MIN_CELL_SIZE students is not one that may be printed.

Suppressing one cell is not enough when the others add up to a printed total
(stacked shares summing to 100%, a row total): the hidden cell comes back by
subtraction. `complementary()` picks a second cell to hide in that case, and
falls back to hiding every remaining cell when they are all zero and no single
partner would conceal anything.

A figure and its table twin must apply this together: a value suppressed in a
table but drawn exactly in its figure twin is not suppressed at all.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from src.config import MIN_CELL_SIZE


SUPPRESSED = f"<{MIN_CELL_SIZE}"                  # plain text, for figures
SUPPRESSED_TEX = f"$<${MIN_CELL_SIZE}"            # LaTeX, for tables


def is_small(*counts) -> bool:
    """
    True if any count is 1..MIN_CELL_SIZE-1, or is present but unknown.

    A true zero is not small — "0 of 40" is a statement about 40 students.

    NaN is treated as small, None is skipped, and the difference is the point:
    None means the count does not apply to this number (an optional AUC
    denominator on a rule that has none), while NaN means it applies and we do
    not know it. Publishing a number whose supporting count is unknown is
    exactly the risk this module exists to remove, so the unknown case fails
    closed.
    """
    for c in counts:
        if c is None:
            continue
        c = float(c)
        if not np.isfinite(c):
            return True
        if 0 < c < MIN_CELL_SIZE:
            return True
    return False


def small_mask(*counts) -> np.ndarray:
    """
    Element-wise is_small over aligned arrays/Series of counts.

    Matches is_small, including that a NaN count is small: an element whose
    supporting count is unknown is not drawn.
    """
    arrs = [np.asarray(c, dtype=float) for c in counts]
    out = np.zeros(np.broadcast(*arrs).shape, dtype=bool)
    for a in arrs:
        out |= ~np.isfinite(a)
        out |= np.isfinite(a) & (a > 0) & (a < MIN_CELL_SIZE)
    return out


def rate_is_small(k, n) -> bool:
    """
    A rate k/n printed next to a known n gives away both k and n-k, so it is
    hidden when either is small (or n itself is).
    """
    if k is None or n is None:
        return False
    return is_small(k, n - k, n)


def confusion_is_small(tp, fp, fn, tn):
    """
    Element-wise: does any cell of a confusion matrix hold 1..MIN_CELL_SIZE-1
    students? Recall, precision, flag rate and the dropout rate of one group
    together pin down all four cells, so a shown operating point rests on each
    of them.
    """
    return small_mask(tp, fp, fn, tn)


def operating_point_hidden(n, n_pos, n_flag, n_tp, n_auc_pos=None, n_auc_neg=None) -> dict:
    """
    Which metrics of one operating point (one model, one population) to hide.

    recall / precision — together with the group's size and dropout rate, which
        are printed or derivable, they pin down all four confusion cells, so they
        are hidden when ANY of TP, FP, FN, TN holds 1..MIN_CELL_SIZE-1 students.
    flag_rate          — reveals only flagged vs not flagged.
    auc                — ranks dropouts against non-dropouts; rests on both.
    """
    tp = n_tp
    fp = n_flag - n_tp
    fn = n_pos - n_tp
    tn = n - n_pos - fp
    pos = n_pos if n_auc_pos is None else n_auc_pos
    neg = (n - n_pos) if n_auc_neg is None else n_auc_neg
    cells = bool(confusion_is_small(tp, fp, fn, tn).any())
    return {
        "recall":    cells,
        "precision": cells,
        "flag_rate": is_small(n_flag, n - n_flag),
        "auc":       is_small(pos, neg),
        "accuracy":  cells,
    }


def results_row_hidden(r) -> dict:
    """
    operating_point_hidden() for one row of evaluation_results.csv (or any Series
    carrying the same count columns).

    Needs the exact TP (n_tp_enrollments). A CSV without it is refused rather
    than guessed at: TP rebuilt from the 3-decimal recall can be off by one,
    which is exactly the margin that decides suppression. Row-level metrics
    follow their enrollment-level twin — several EWS rows can be one student.
    """
    r = pd.Series(r)
    for col in ("n_enrollments", "n_dropout_enrollments", "n_flagged_enrollments", "n_tp_enrollments"):
        if col not in r.index or pd.isna(r[col]):
            raise ValueError(
                f"no {col!r} for {r.get('model', '?')}/{r.get('feature_set', '?')}: the "
                "results predate disclosure control. Re-run src/evaluation.py first."
            )
    opt = lambda c: int(r[c]) if c in r.index and pd.notna(r[c]) else None   # noqa: E731
    return operating_point_hidden(
        int(r["n_enrollments"]), int(r["n_dropout_enrollments"]),
        int(r["n_flagged_enrollments"]), int(r["n_tp_enrollments"]),
        opt("n_auc_pos"), opt("n_auc_neg"),
    )


def roc_hidden(preds: pd.DataFrame, group_col: str, values, outcome: str = "dropped_out") -> bool:
    """
    True if a ROC curve (or AUC, ABROCA) for any of these groups would rest on
    1..MIN_CELL_SIZE-1 students, dropouts or non-dropouts.
    """
    if group_col not in preds.columns:
        return False
    for val in values:
        g = preds[preds[group_col] == val]
        n_pos = int((g[outcome] == 1).sum())
        if is_small(len(g), n_pos, len(g) - n_pos):
            return True
    return False


def fmt(value, *counts, spec: str = "{:.3f}", tex: bool = True, na: str = "--") -> str:
    """`value` formatted with `spec`, or the suppression marker if it rests on a small count."""
    if is_small(*counts):
        return SUPPRESSED_TEX if tex else SUPPRESSED
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return na
    return spec.format(value)


def fmt_count(n, tex: bool = True, na: str = "--") -> str:
    """A count of students, or the suppression marker for 1..MIN_CELL_SIZE-1."""
    if n is None or (isinstance(n, float) and not np.isfinite(n)):
        return na
    return fmt(int(n), n, spec="{:,}", tex=tex)


def complementary(counts: Sequence[float], hidden: Iterable[int]) -> set[int]:
    """
    Indices to hide so no hidden cell can be recovered from a known total.

    `counts` are the cells of one group that add up to something the reader
    knows (a total, or 100%). If exactly one cell is hidden, the smallest other
    non-zero cell is hidden with it.

    When every other cell is zero there is no non-zero partner to pair with, and
    the hidden cell is then simply the total minus a row of zeros. Hiding the
    zeros too is what closes that: several hidden cells summing to a known total
    do not say which of them holds it.
    """
    hidden = set(hidden)
    if len(hidden) == 1:
        others = [(c, i) for i, c in enumerate(counts) if i not in hidden and c > 0]
        if others:
            hidden.add(min(others)[1])
        else:
            hidden |= {i for i in range(len(counts)) if i not in hidden}
    return hidden


def annotate(ax, x, y, **kwargs):
    """Put the suppression marker where a hidden element would have been."""
    style = dict(ha="center", va="center", fontsize=8, color="0.35")
    style.update(kwargs)
    return ax.text(x, y, SUPPRESSED, **style)


def note_text() -> str:
    """
    The sentence a FIGURE carries when some element was suppressed.

    Built from MIN_CELL_SIZE and exported with the figure: a prepared figure input
    carries this string rather than the threshold, so a draw script never has to
    know the number to print the note. Table captions carry no such sentence.
    """
    return f"{SUPPRESSED}: fewer than {MIN_CELL_SIZE} students, not shown"


def note(ax, loc: str = "above", text: str | None = None, **kwargs):
    """
    A small note that some elements were suppressed. Default: just above the
    plot area, right-aligned, where it cannot sit on data. Kept out of the
    layout engine so a long note never shrinks the axes.
    """
    xy = {"above":       (1.0, 1.01, "right", "bottom"),
          "lower right": (0.99, 0.01, "right", "bottom"), "upper right": (0.99, 0.99, "right", "top"),
          "lower left":  (0.01, 0.01, "left", "bottom"),  "upper left":  (0.01, 0.99, "left", "top")}[loc]
    style = dict(transform=ax.transAxes, ha=xy[2], va=xy[3], fontsize=8, color="0.35")
    style.update(kwargs)
    t = ax.text(xy[0], xy[1], text or note_text(), **style)
    t.set_in_layout(False)
    return t
