"""
SHAP values behind the two SHAP figures:
  1. Max ML        — mlp + e_plus_sensitive
  2. Participatory — the model and feature set the workshop rules selected

One explainer for every model — a PermutationExplainer over predict_proba on a
200-row background sample — so both figures are on the probability scale. SHAP
values are computed on a random sample of at most 500 enrollments of the test
year, one row per enrollment.

Used by figures/scripts/prepare_shap.py, which writes the figures' input.
"""
import json
import sys
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).parent.parent))

from paths.paths import HELVETICA, RESULTS_PATH
from src import paper_style
from src.display import MODEL_LABELS, feature_label, feature_labels, feature_set_label, model_label
from src.config import (
    DATA_CONFIGS,
    MAIN_TEST_YEAR,
    MAX_ML_FS,
    MAX_ML_MODEL,
    RANDOM_SEED,
)
from src import disclosure
from src.data import load_analysis_df
from src.features import load_model_checked

# ── Matplotlib style ──────────────────────────────────────────────────────────

## The paper style, shared with every figure (src/paper_style.py).
paper_style.apply_style()

_MODELS_DIR = RESULTS_PATH / "models"

_MODEL_DISPLAY = {
    "fasterrisk":          "FasterRisk",
    "logistic_regression": "Logistic Regression",
    "xgboost":             "XGBoost",
    "mlp":                 MODEL_LABELS["mlp"],   # one name, in src/display.py
}

## Feature names come from src/display.py (feature_labels), shared with the scorecard.


def _load_workshop_config() -> tuple[str | None, str | None]:
    """Return (model, feature_set) for the workshop-selected (participatory) model."""
    from src.workshop_rules import participatory_config
    return participatory_config(RESULTS_PATH)


def _get_test_sample(
    raw_df: pd.DataFrame,
    features: list[str],
    test_year: int,
    n_sample: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    # No dropna. Missing values are part of what the models see — the pipeline
    # imputes them and adds missing indicators — so rows with missing values
    # are kept. Missingness is not random, and dropping incomplete rows would
    # remove whole groups of students from the SHAP sample.
    test = raw_df[raw_df["start_year"] == test_year]
    if "enrollment" in test.columns:
        # One row per student, so every count in the figure — including the
        # disclosure check in shap_violin_shapes — is a count of students, not
        # of EWS rows (one student can have many).
        test = test.iloc[rng.permutation(len(test))].drop_duplicates("enrollment")
    test = test[features]
    if len(test) > n_sample:
        idx = rng.choice(len(test), n_sample, replace=False)
        test = test.iloc[idx]
    return test.reset_index(drop=True)


def _pretty_feature_names(features: list[str]) -> list[str]:
    return feature_labels(features)


def _preprocess(model, X_sample: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Transform X_sample through the pipeline's preprocessor (all steps except the
    final classifier) and return a numeric DataFrame plus human-readable feature names.
    """
    has_steps = hasattr(model, "named_steps") and len(model) > 1
    if not has_steps:
        return X_sample, _pretty_feature_names(list(X_sample.columns))

    preprocessor = model[:-1]
    X_pre = preprocessor.transform(X_sample)

    if hasattr(X_pre, "toarray"):
        X_pre = X_pre.toarray()

    try:
        raw_names = list(preprocessor.get_feature_names_out())
    except Exception:
        raw_names = [f"f{i}" for i in range(X_pre.shape[1])]

    feature_names = feature_labels(raw_names)

    X_pre_df = pd.DataFrame(X_pre, columns=raw_names)
    return X_pre_df, feature_names


def _compute_shap_values(
    model,
    X_sample: pd.DataFrame,
    model_name: str,
    rng: np.random.Generator,
    n_background: int = 200,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """Return (shap_values_2d, feature_names, X_pre) where features are post-preprocessing."""
    X_pre, feature_names = _preprocess(model, X_sample)

    clf = list(model.named_steps.values())[-1] if hasattr(model, "named_steps") else model

    ## ONE explainer for every model, over predict_proba.
    ##
    ## Model-specific explainers (TreeExplainer, LinearExplainer) explain the
    ## model's raw output — log-odds — while a PermutationExplainer over
    ## predict_proba explains a probability. Mixing them would put figures
    ## printed side by side on incomparable scales, so every model is explained
    ## the same way, on the probability scale.
    ##
    ## Probability is the scale the rest of the paper speaks in, and it is what
    ## the axis label states ("impact on dropout prediction"). The exact
    ## explainers are faster; they are not worth two incomparable figures.
    try:
        bg_idx = rng.choice(len(X_pre), min(n_background, len(X_pre)), replace=False)
        background = X_pre.iloc[bg_idx]
        cols = list(X_pre.columns)
        ## Column names only for a classifier fitted with them. The pipeline's
        ## preprocessor hands its last step an unnamed array, so a DataFrame here
        ## made sklearn warn on every one of the explainer's calls. Order is the
        ## same either way; the values do not change.
        named = hasattr(clf, "feature_names_in_")
        explainer = shap.PermutationExplainer(
            lambda x: clf.predict_proba(pd.DataFrame(x, columns=cols) if named
                                        else np.asarray(x))[:, 1],
            background,
            seed=int(RANDOM_SEED),
        )
        sv = explainer(X_pre).values
    except Exception as exc:
        warnings.warn(f"SHAP failed for {model_name}: {exc}")
        return None, feature_names, X_pre

    if sv is None:
        return None, feature_names, X_pre
    if isinstance(sv, list):
        sv = sv[1] if len(sv) == 2 else sv[0]
    return np.array(sv), feature_names, X_pre


def _bbox_center(bbox):
    """The middle of a rendered text's box, in display coordinates."""
    return ((bbox.x0 + bbox.x1) / 2.0, (bbox.y0 + bbox.y1) / 2.0)


def _skipped_note(skipped: int) -> str:
    return (f"{skipped} feature(s) not shown: a value is held by fewer than "
            f"{disclosure.MIN_CELL_SIZE} sampled students")


def _merged_note() -> str:
    return (f"Values held by fewer than {disclosure.MIN_CELL_SIZE} sampled students "
            f"are merged with the neighbouring value")


def _merge_small_layers(col: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Recode a feature so that none of its layers rests on too few students.

    shap's layered violin gives each distinct value of a feature with <=20 of
    them its own layer, coloured by the layer's RANK, not its value. Values are
    walked in ascending order and grouped until a group holds enough students; a
    small group left at the top joins the one below it. Each group takes the
    value of its largest member, so order is kept and the common values stay
    exact — e.g. values 0/1/2/3 with too few students at 3 become 0/1/2+.

    Returns the recoded column and whether anything was merged. A feature with
    more than 20 distinct values is returned untouched: shap bins it into equal
    count layers of ~n/20 students.
    """
    ok = ~pd.isna(col)
    values, counts = np.unique(col[ok], return_counts=True)
    if len(values) > 20 or not disclosure.small_mask(counts).any():
        return col, False
    groups, cur = [], []
    for i in range(len(values)):
        cur.append(i)
        if not disclosure.is_small(counts[cur].sum()):
            groups.append(cur)
            cur = []
    if cur:
        if groups:
            groups[-1].extend(cur)
        else:
            groups.append(cur)
    target = {}
    for g in groups:
        rep = values[g[int(np.argmax(counts[g]))]]
        for i in g:
            target[values[i]] = rep
    out = col.astype(float, copy=True)
    out[ok] = [target[v] for v in col[ok]]
    return out, True


def shap_violin_shapes(
    model,
    model_name: str,
    X_sample: pd.DataFrame,
    shap_values: np.ndarray,
    feature_names: list[str],
    X_pre: pd.DataFrame,
    feature_set: str,
    title: str,
    max_display: int = 15,
) -> dict | None:
    """
    The layered violin's SHAPE — the density bands themselves, with no per-student
    value left in them.

    A layered violin is drawn as one filled band per value-quantile of the
    feature, so what reaches the paper is an aggregate outline. The SHAP matrix
    behind it is not: it holds one row per sampled enrollment. This function
    therefore renders the violin into a throwaway figure, harvests the polygons
    shap actually drew, and returns those. The band outlines are exactly what the
    figure shows and carry nothing else — which is what lets the paper's copy of
    this figure be re-rendered without the pipeline.

    Harvested rather than recomputed on purpose: reimplementing shap's layering
    would be a second copy of its binning, free to drift from the library that
    produced every earlier version of this figure.

    Disclosure control: the layered violin draws one layer per value of a
    feature with <=20 distinct values, and a layer of a few students shows their
    SHAP values (recolouring would not help: the values still form a visible
    bump). A value held by too few students is merged into its neighbour
    (_merge_small_layers) rather than dropping the feature, so one rare value in
    its tail does not remove an otherwise highly ranked feature. Only a feature left with a single layer after
    merging (a rare dummy) is skipped, and the next one in the ranking takes its
    place. Continuous features are binned by shap into layers of ~25 students.
    See src/disclosure.py.
    """
    if shap_values is None:
        return None

    mean_abs = np.abs(shap_values).mean(axis=0)
    ranking = np.argsort(mean_abs)[::-1]

    order, cols, skipped, merged = [], [], 0, False
    for i in ranking:
        col, was_merged = _merge_small_layers(np.asarray(X_pre.values[:, i], dtype=float))
        layers = np.unique(col[~pd.isna(col)])
        if was_merged and len(layers) < 2:      # a rare dummy: nothing left to layer
            skipped += 1
            continue
        merged |= was_merged
        order.append(i)
        cols.append(col)
        if len(order) == max_display:
            break
    order = np.array(order, dtype=int)

    sv_top = shap_values[:, order]
    fn_top = [feature_names[i] for i in order]
    xv_top = np.column_stack(cols) if cols else X_pre.values[:, order]
    note_text = "; ".join(t for t in (_merged_note() if merged else None,
                                      _skipped_note(skipped) if skipped else None) if t)

    n = len(order)
    ## Height follows the number of rows the figure was designed to hold, not the
    ## number it ended up with, so two of these figures side by side line up.
    figsize = [7, max(4, 0.4 * max_display + 1.5)]

    ## Same creation call the figure has always used: shap draws into the
    ## current axes, and an axes created by subplots() starts where subplots()
    ## puts it, not where a bare figure would.
    fig, _ax0 = plt.subplots(figsize=figsize, constrained_layout=True)
    shap.plots.violin(
        shap.Explanation(
            values=sv_top,
            base_values=np.zeros(len(sv_top)),
            data=xv_top,
            feature_names=fn_top,
        ),
        max_display=max_display,
        plot_type="layered_violin",
        show=False,
    )
    ## Finish the figure exactly as it has always been finished, THEN harvest:
    ## the axis labels and the title are part of what the layout engine solves
    ## for, so measuring before they exist would record positions the published
    ## figure never had.
    ax = fig.axes[0]
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("SHAP value (impact on dropout prediction)", fontsize=10)
    ax.set_title(title, fontsize=10, pad=8)
    ## Let constrained_layout converge, then freeze it, so the harvested
    ## positions are the ones the figure keeps.
    fig.canvas.draw()
    fig.set_layout_engine(None)
    fig.canvas.draw()

    polygons = [
        {"vertices": np.asarray(coll.get_paths()[0].vertices, dtype=float),
         "facecolor": [float(v) for v in np.asarray(coll.get_facecolor())[0]]}
        for coll in ax.collections if coll.get_paths()
    ]
    ## A reference line drawn with axvline lives in a BLENDED transform — data
    ## on x, axes fraction on y — so its y data is (0, 1) and replotting it in
    ## data coordinates would put a stub at the bottom of the panel. Record which
    ## kind each line is.
    lines = []
    for ln in ax.lines:
        blended = ln.get_transform() is ax.get_xaxis_transform()
        lines.append({
            "kind": "axvline" if blended else "data",
            "x": [float(v) for v in ln.get_xdata()],
            "y": [float(v) for v in ln.get_ydata()],
            "color": mpl.colors.to_hex(ln.get_color()),
            "linewidth": float(ln.get_linewidth()),
            "linestyle": str(ln.get_linestyle()),
        })

    ## shap adds its own colour key as a second axes. It is part of the figure,
    ## so it is harvested like everything else: the 256 quads of the gradient,
    ## their values and colormap, and the two labels at its ends.
    colorbar = None
    if len(fig.axes) > 1:
        cb = fig.axes[1]
        mesh = next((c for c in cb.collections if type(c).__name__ == "QuadMesh"), None)
        if mesh is not None:
            coords = np.asarray(mesh.get_coordinates())
            colorbar = {
                "position": [float(v) for v in cb.get_position().bounds],
                "cmap": mesh.get_cmap().name,
                ## shap rasterises the key: 256 vector quads leave a seam
                ## between every pair, and the gradient reads as stripes.
                "rasterized": bool(mesh.get_rasterized()),
                "linewidth": float(np.ravel(mesh.get_linewidth())[0]),
                "clim": [float(v) for v in mesh.get_clim()],
                "x_edges": [float(v) for v in coords[0, :, 0]],
                "y_edges": [float(v) for v in coords[:, 0, 1]],
                "values": [float(v) for v in np.asarray(mesh.get_array()).ravel()],
                "xlim": [float(v) for v in cb.get_xlim()],
                "ylim": [float(v) for v in cb.get_ylim()],
                "yticks": [float(v) for v in cb.get_yticks()],
                "yticklabels": [t.get_text() for t in cb.get_yticklabels()],
                "ylabel": cb.get_ylabel(),
                "ylabel_size": float(cb.yaxis.label.get_fontsize()),
                "ytick_size": float(cb.get_yticklabels()[0].get_fontsize())
                              if cb.get_yticklabels() else 9.0,
                "tick_position": cb.yaxis.get_ticks_position(),
                ## Which side the key's caption sits on. shap puts it right of
                ## the bar; matplotlib's default is left, and getting this wrong
                ## silently narrows the whole figure.
                "label_position": cb.yaxis.get_label_position(),
                "labelpad": float(cb.yaxis.labelpad),
                ## Where the caption ACTUALLY landed, in figure coordinates. A
                ## Colorbar places its label with its own padding rule, which a
                ## plain axes does not reproduce; recording the position skips
                ## having to re-derive it.
                "label_xy": [float(v) for v in
                             fig.transFigure.inverted().transform(
                                 _bbox_center(cb.yaxis.label.get_window_extent(
                                     fig.canvas.get_renderer()))
                             )],
                "label_rotation": float(cb.yaxis.label.get_rotation()),
                "aspect": cb.get_aspect() if isinstance(cb.get_aspect(), str) else float(cb.get_aspect()),
            }

    shapes = {
        "polygons": polygons,
        "lines": lines,
        "axes_position": [float(v) for v in ax.get_position().bounds],
        "colorbar": colorbar,
        "xlim": [float(v) for v in ax.get_xlim()],
        "ylim": [float(v) for v in ax.get_ylim()],
        "yticks": [float(v) for v in ax.get_yticks()],
        "yticklabels": [t.get_text() for t in ax.get_yticklabels()],
        ## shap sets its own tick sizes, larger than the house default. They are
        ## part of the figure's proportions — the feature names are what makes it
        ## as wide as it is — so they travel with the shape.
        "ytick_size": float(ax.get_yticklabels()[0].get_fontsize())
                      if ax.get_yticklabels() else 9.0,
        "xtick_size": float(ax.get_xticklabels()[0].get_fontsize())
                      if ax.get_xticklabels() else 9.0,
        "n_features": int(n),
        ## The size AFTER drawing: shap widens the canvas to make room for its
        ## colour key, so the requested size is not the size the figure has.
        "figsize": [float(v) for v in fig.get_size_inches()],
        "title": title,
        "xlabel": "SHAP value (impact on dropout prediction)",
        "skipped": int(skipped),
        "merged": bool(merged),
        "note": note_text or None,
        "extra_axes": len(fig.axes) - 1,
    }
    plt.close(fig)
    return shapes


def _run_one(
    label: str,
    model_name: str,
    feature_set: str,
    raw_df: pd.DataFrame,
    test_year: int,
    n_sample: int,
    max_display: int,
    rng: np.random.Generator,
) -> dict | None:
    features = DATA_CONFIGS.get(feature_set)
    if features is None:
        print(f"  SKIP {label} — unknown feature set '{feature_set}'")
        return None

    model_path = _MODELS_DIR / f"{model_name}__{feature_set}.joblib"
    if not model_path.exists():
        print(f"  SKIP {label} — model file not found: {model_path}")
        return None

    X_sample = _get_test_sample(raw_df, features, test_year, n_sample, rng)
    if len(X_sample) == 0:
        print(f"  SKIP {label} — no rows for test_year={test_year}")
        return None

    print(f"\n  Computing SHAP for {label} ({model_name} + {feature_set}, "
          f"{len(features)} features, {len(X_sample)} sample rows) ...")
    # Checked load, like everywhere else: a figure must not explain a model
    # fitted on another feature list, procedure or encoding.
    model = load_model_checked(model_path, features)

    sv, fn, X_pre = _compute_shap_values(model, X_sample, model_name, rng)
    if sv is None:
        print(f"  SKIP {label} — SHAP computation failed.")
        return None

    title = f"{label} — {model_label(model_name)}, {feature_set_label(feature_set)}"
    return {"model": model_name, "feature_set": feature_set, "label": label,
            "shapes": shap_violin_shapes(model, model_name, X_sample, sv, fn, X_pre,
                                         feature_set, title=title,
                                         max_display=max_display)}


def generate_shap_shapes(
    test_year: int = MAIN_TEST_YEAR,
    n_sample: int = 500,
    max_display: int = 15,
) -> list[dict]:
    """
    The layered-violin shapes for Max ML and the Participatory model.

    Returns one entry per figure; the drawing is
    figures/scripts/draw_shap.py, which reads only the band outlines.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    raw_df = load_analysis_df()

    targets = [("Max ML", MAX_ML_MODEL, MAX_ML_FS)]

    from src.workshop_rules import _RULE_BASED_MODELS
    ws_model, ws_fs = _load_workshop_config()
    if ws_model and ws_fs:
        ## Refused rather than skipped: skipping would leave the previous run's
        ## combined input in figures/input/, and the draw step would redraw it.
        if ws_model in _RULE_BASED_MODELS:
            raise ValueError("no combination was permissible, so the Credits policy is "
                             "retained: it is a rule with nothing to explain, and the "
                             "SHAP figure has no Participatory panel to draw.")
        if (ws_model, ws_fs) == (MAX_ML_MODEL, MAX_ML_FS):
            print("NOTE: participatory model is identical to Max ML — skipping duplicate.")
        else:
            targets.append(("Participatory", ws_model, ws_fs))
    else:
        print("WARNING: no workshop config found — participatory model plot skipped.")

    out = []
    for label, model_name, feature_set in targets:
        got = _run_one(label, model_name, feature_set, raw_df, test_year,
                       n_sample, max_display, rng)
        if got is not None and got.get("shapes") is not None:
            out.append(got)
    return out

## No __main__: the SHAP figures are prepared and drawn by
## figures/scripts/{prepare,draw}_shap.py, which calls generate_shap_shapes()
## from here. Everything above reads student records; nothing in it draws a
## figure that is kept.
