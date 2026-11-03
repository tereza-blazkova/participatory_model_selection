"""
Distil the SHAP figures to their SHAPE — the density bands, and nothing else.

    figures/input/shap_{model}_{feature_set}.csv    the band outlines
    figures/input/shap_{model}_{feature_set}.json   colours, axes, labels

A layered violin is an aggregate: one filled band per value-quantile of the
feature. The matrix behind it is not — it holds one row per sampled enrollment,
and unlike a line or a bar, a published PDF of this figure does not already give
those numbers away. So the SHAP values themselves never enter figures/input/;
what does is the outline of each band, which is what the paper prints.

src/shap_plots.py computes the values, applies the rule that skips a feature
whose layers would rest on too few sampled students, renders the violin into a
throwaway figure and harvests the polygons shap drew. Harvested rather than
recomputed: reimplementing shap's layering would be a second copy of its
binning, free to drift from the library's own.

    python figures/scripts/prepare_shap.py [--year 2021] [--n-sample 500]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from prepare_utils import write_input                                  # noqa: E402
from src.config import MAIN_TEST_YEAR, MAX_ML_FS, MAX_ML_MODEL         # noqa: E402
from src.shap_plots import generate_shap_shapes                        # noqa: E402
from utils import canvas                                               # noqa: E402


SCRIPT = "prepare_shap.py"


def to_input(entry: dict) -> tuple[pd.DataFrame, dict]:
    shapes = entry["shapes"]
    rows = []
    for i, poly in enumerate(shapes["polygons"]):
        for order, (x, y) in enumerate(poly["vertices"]):
            rows.append({"band": i, "order": order, "x": x, "y": y})
    for j, line in enumerate(shapes["lines"]):
        if line["kind"] == "axvline":
            continue          # the x is enough; the y spans the panel
        for order, (x, y) in enumerate(zip(line["x"], line["y"])):
            rows.append({"band": -1 - j, "order": order, "x": x, "y": y})

    meta = {
        ## Printed at 0.8\\textwidth. shap widened the canvas to fit its colour
        ## key, so the aspect ratio comes from the size it ended up with.
        "figsize": canvas(shapes["figsize"][1] / shapes["figsize"][0],
                          width_fraction=0.8),
        ## Both axes are placed explicitly: the layout was resolved when the
        ## input was built, so the drawing never has to re-run it.
        "axes_position": shapes["axes_position"],
        "colorbar": shapes["colorbar"],
        "title": shapes["title"],
        "xlabel": shapes["xlabel"],
        "xlim": shapes["xlim"],
        "ylim": shapes["ylim"],
        "yticks": shapes["yticks"],
        "yticklabels": shapes["yticklabels"],
        "ytick_size": shapes["ytick_size"],
        "xtick_size": shapes["xtick_size"],
        ## One RGBA per band. The colour is where that band's feature values sit
        ## in the feature's own range under shap's colormap — data, not a style
        ## choice, and the only part of it the draw script could not derive.
        "band_colors": [p["facecolor"] for p in shapes["polygons"]],
        "lines": [{**{k: v for k, v in ln.items() if k not in ("x", "y")},
                    **({"at": ln["x"][0]} if ln["kind"] == "axvline" else {})}
                  for ln in shapes["lines"]],
        ## Named features whose layers would have rested on too few sampled
        ## students; the next feature in the ranking took each one's place.
        "skipped": shapes["skipped"],
        ## Built by src/shap_plots.py from MIN_CELL_SIZE, like every other
        ## suppression sentence, so figures/ never spells the threshold. It is
        ## the CAPTION's sentence: draw_shap.py does not print it in the figure.
        "note": shapes["note"],
        "text_columns": [],
    }
    return pd.DataFrame(rows), meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year", type=int, default=MAIN_TEST_YEAR)
    ap.add_argument("--n-sample", type=int, default=500)
    ap.add_argument("--max-display", type=int, default=15)
    args = ap.parse_args()

    entries = list(generate_shap_shapes(test_year=args.year, n_sample=args.n_sample,
                                        max_display=args.max_display))
    built = []
    for entry in entries:
        df, meta = to_input(entry)
        stem = f"shap_{entry['model']}_{entry['feature_set']}"
        write_input(stem, df, meta, source="data + results/models/",
                    repo=ROOT, script=SCRIPT)
        built.append((entry, df, meta))

    ## The paper's figure: both models side by side, the Participatory model
    ## first (a), Max ML second (b). Each half keeps its own single-figure
    ## layout, so the canvas is the full text width at half each one's aspect.
    built.sort(key=lambda b: (b[0]["model"], b[0]["feature_set"]) == (MAX_ML_MODEL, MAX_ML_FS))
    if len(built) == 2:
        frames, panels, aspects = [], [], []
        for i, (_, df, meta) in enumerate(built):
            frames.append(df.assign(panel=i))
            w, h = meta["figsize"]
            aspects.append(h / w)
            panels.append({k: v for k, v in meta.items() if k != "figsize"})
        ## A little taller than the halves need, so the feature names breathe.
        combined = {"figsize": canvas(max(aspects) / 2 * 1.08), "panels": panels,
                    "text_columns": built[0][2].get("text_columns", [])}
        write_input("shap_participatory_maxml", pd.concat(frames, ignore_index=True),
                    combined, source="data + results/models/", repo=ROOT, script=SCRIPT)


if __name__ == "__main__":
    main()
