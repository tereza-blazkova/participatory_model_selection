# `figures/` — the panels the paper prints

Every figure in the paper or the SI is three tracked things: a **prepare**
script that distils the pipeline's output into a small table, that **input**
table, and a **draw** script that turns it into the panel.

```
figures/
  scripts/   utils.py + one prepare_*.py and one draw_*.py per figure
  input/     the distilled CSV each figure reads, plus its scalars as JSON
  output/    the rendered panels, PDF (and PNG for review)
  fonts/     empty as shipped: Helvetica is licensed, so add it here if you have it
```

`results/` holds the pipeline's own output. **`figures/` is the small set of
finished panels the paper includes.** Different purpose, different rules.

## Four rules

**1. A draw script reads `figures/input/` and nothing else.** Never `results/`,
never `src/`, never the parquet. That is what lets the tree render on a machine
that has the paper but not the pipeline, and it is the disclosure guarantee: a
draw script cannot print a number the prepare step withheld, because the code to
reach one is not imported. Getting data *into* `input/` is the prepare script's
job, and the prepare script is allowed to reach anywhere.

**2. Nothing but `utils.py` decides how a figure looks.** Font, sizes, palette,
markers, dpi, output paths, spine policy. A draw script chooses *what* to draw.

**3. `input/` carries the values that are DRAWN — post-suppression, never the
true ones.** A suppressed value is an empty cell, and a `*_suppressed` flag says
so. The prepare script re-checks this as a post-condition on the frame it is
about to write, against `disclosure.results_row_hidden`, rather than trusting the
flags that built it.

Why the prepared inputs are safe to share at all: **a vector PDF already
carries the exact geometry of everything it draws.** Anyone with the figure can
read its coordinates. An input CSV of post-suppression drawn values is the same
information in a readable form — never more. It is *not* safe if the CSV carries
the true value behind a hidden point, which is what rule 3 exists to prevent.

**4. A figure and its table twin are built from the same distilled input.** If a
table and a figure showing the same quantity each applied suppression on their
own, one could draw a value the other withholds, and the withholding would buy
nothing. Two artefacts showing the same quantity get one prepare step between
them and cannot disagree.

`src/generate_selection_table.py` therefore reads
`input/selection_gate_{year}.csv`, not `evaluation_results.csv`, and does no
analysis at all: it formats. That is why the prepare step runs *before* the table
in `run_all.py` — the ordering is load-bearing. A table generator is welcome to
stay in `src/` and import `disclosure`; the wall (rule 1) binds the draw scripts,
which are the half that has to render without the pipeline present.

## Where the files live

`.gitignore` ignores `*.csv` anywhere in the checkout, which is deliberate and
applies here: an `input/` CSV is a run product. So only `scripts/` and `fonts/`
are tracked; `input/` and `output/` are run products, exactly like `results/`.
A copy of the tree that carries `input/` re-renders its figures with no pipeline
present.

## The figures

Every figure the paper prints, main text and SI. Each is one prepare step,
one input, one draw step, one panel. Several figures share a prepare script where
they share an expensive computation; none share a draw script except where the
same panel is drawn from two inputs.

| figure | prepared by | drawn by | input |
|---|---|---|---|
| selection gate (2021 main text; 2019 SI) | `prepare_selection_gate.py [--test-year 2019 --evaluation-year 2021]` | `draw_selection_gate.py [--year 2019]` | `selection_gate_{year}` |
| model comparison | `prepare_model_comparison.py` | `draw_model_comparison.py` | `model_comparison` |
| model comparison with the pilot panel | `prepare_model_comparison.py` | `draw_model_comparison.py --stem model_comparison_2021_2023` | `model_comparison_2021_2023` |
| performance by period | `prepare_performance_by_period.py` | `draw_performance_by_period.py` | `performance_by_period` |
| dropout timing | `prepare_dropout_timing.py` | `draw_dropout_timing.py` | `dropout_timing` |
| dropout timing, with Max ML | `prepare_dropout_timing.py --max-ml` | `draw_dropout_timing.py --stem dropout_timing_maxml` | `dropout_timing_maxml` |
| FA frontier | `prepare_fa_frontier.py` | `draw_fa_frontier.py` | `fa_frontier_{year}` |
| FA frontier ladders | `prepare_fa_frontier_ladders.py` | `draw_fa_frontier_ladders.py` | `fa_frontier_ladders_{year}` |
| pilot: outcome breakdown, Faculty A (a real thresholds, b recalibrated) | `prepare_pilot.py` | `draw_pilot_breakdown.py --stem pilot_breakdown_faculty_a` | `pilot_breakdown_faculty_a` |
| pilot: ROC AUC, Faculty A (main text) | `prepare_pilot.py` | `draw_pilot_auc.py --stem pilot_auc_faculty_a` | `pilot_auc_faculty_a` |
| pilot: ROC AUC, Faculty B (SI) | `prepare_pilot.py` | `draw_pilot_auc.py --stem pilot_auc_faculty_b` | `pilot_auc_faculty_b` |
| pilot: points profile, one faculty each (rate and count side by side; Faculty A only draws the outreach-threshold line) | `prepare_pilot.py` | `draw_pilot_points_profile.py --stem pilot_points_profile_faculty_a` (`_b`) | `pilot_points_profile_faculty_{a,b}` |
| pilot: ABROCA heatmap | `prepare_pilot.py` | `draw_pilot_abroca.py` | `pilot_abroca` |
| FasterRisk scorecard | `prepare_pilot.py` | `draw_scorecard.py` | `scorecard_used_vars` |
| SHAP: Participatory and Max ML side by side | `prepare_shap.py` | `draw_shap.py --stem shap_participatory_maxml` | `shap_participatory_maxml` |
| SHAP: Max ML alone | `prepare_shap.py` | `draw_shap.py --stem shap_mlp_e_plus_sensitive` | `shap_mlp_e_plus_sensitive` |
| SHAP: Participatory | `prepare_shap.py` | `draw_shap.py --stem shap_logistic_...` | `shap_logistic_regression_b_credits_performance` |

The five pilot figures share `prepare_pilot.py` because they rest on the same
expensive step — scoring every saved model on the pilot snapshot, faculty by
faculty. The two FA frontier prepare steps also write their summary CSVs into
`results/fairness/`, from the same build, rather than recomputing a 1000-draw
bootstrap in a second process.

### selection gate, in more detail

| | |
|---|---|
| input | one row per cell, plus the credits-policy baseline as a `role=baseline` row |
| table twin | `tab:selection_gate`, `src/generate_selection_table.py`, from the same CSV |

Both sides of the verdict — what the gates decided, and what may be shown of the
recall behind it — are resolved once, in `src/plot_selection_gate.py`, which
cross-checks its reconstruction against `select_grid_config()`. The CSV carries
`precision` and the two verdict halves because the table prints them and the
figure does not; every one of them is something the paper already publishes.

### SHAP: the one figure whose input is not simply what it draws

A layered violin is an aggregate — one filled band per value-quantile of the
feature — but the matrix behind it is one row per sampled enrollment, and unlike
a line or a bar, a published PDF does not already give those numbers away. So the
SHAP values never enter `input/`. What does is the outline of each band, which
`src/shap_plots.py` obtains by rendering the violin into a throwaway figure and
harvesting the polygons shap drew. Harvested rather than recomputed: a
reimplementation of shap's layering would be a second copy of its binning, free
to drift from the library's own.

**One explainer, one scale.** Every model is explained through `predict_proba`,
with the same explainer. The exact explainers for trees and linear models
explain the raw output, which is log-odds, while a model-agnostic explainer over
`predict_proba` explains a probability; mixing them would put the two panels the
paper prints side by side on incomparable scales. The probability scale is the
one the rest of the paper speaks in and the one the axis label names.

## Type size: one canvas rule

matplotlib bakes font sizes into a PDF as absolute points, so a figure's printed
type size is its designed size times how much `\includegraphics` scales it:

    printed_pt = designed_pt x (printed width / canvas width)

Several figures print at the same type size only if all of them share that
ratio. `utils.canvas()` sets every canvas from the width the paper prints the
figure at, and one constant, `CANVAS_SCALE`, fixes the ratio for all of them, so
every figure prints its text at the same size. The constant is set by the
densest figure, `dropout_timing` (panels side by side, each with its own
legend above it).

## Running

`run_all.py` runs them in order — every prepare step, then every draw step
(step 12), with the selection gate at step 14 and the two FA frontiers at step
22. By hand:

```bash
python figures/scripts/prepare_selection_gate.py [--test-year 2021] [--ruleset endline]
python figures/scripts/draw_selection_gate.py    [--year 2021] [--png]
```

Prepare needs a finished `results/`; draw needs only `figures/input/`. Run
with the project environment (`environment.yml`).

## House style

Applied by `utils.apply_style()`: Wong-derived
palette, no legend borders, no top or right spines, 9 pt floor, PDF only.
Helvetica is not shipped, and a missing font falls back **with a printed
warning** — matplotlib's font lookup never fails, so without the warning the
substitution is invisible until the PDF is in the paper. 
The palette is duplicated from `src/plotting.py` rather than imported, because
importing it would breach rule 1. The two must be kept identical.

Panels are named by **what they show, never by their number in the paper**.
Numbering shifts whenever a figure is added or moved to the SI, and a number in a
file name goes stale silently.

## Output

`save()` writes a vector PDF with `pdf.fonttype=42`, so the text in the panel is
real embedded glyphs rather than Type-3 outlines. `--png` adds a 300 dpi copy for
review and slides; only the PDF goes in the paper.

Design at the final printed size; never scale a figure in LaTeX.
