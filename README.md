# Participatory Model Selection: code and workshop materials

Code and materials accompanying the paper *Participatory Model Selection: Turning
Stakeholder Deliberation into Constraints for Student Dropout Prediction*. The
framework turns what stakeholders decide in a deliberative workshop into rules
that a predictive model must pass before it can be selected.

**This repository contains no student data, and it cannot reproduce the paper's
numbers.** The student records cannot be shared. Two generators write *synthetic*
data in the same schema, filled with random values. The whole pipeline runs on
that data, so every step can be inspected and re-run, but the synthetic records
carry no real relationship between students' trajectories and dropout, so the
results differ from those in the paper.

## What you can do here

| I want to | Go to |
|---|---|
| See how the selection test works, without any data | `python -m examples.select_with_your_rules` |
| Use the framework at my own institution | [`ADAPTING.md`](ADAPTING.md) |
| Reuse the surveys and workshop plan | [`instruments/`](instruments/) |
| Check how the paper's analysis was done | The full pipeline, below |

## Setup

```bash
conda env create -f environment.yml
conda activate ews-modeling
```

`pyproject.toml` lists the same dependencies for a pip-based environment. `cvxpy`
is optional.

## Running the full pipeline

```bash
python data/generate_fake_df.py      # synthetic enrollment-period panel
python data/generate_pilot_snapshot.py     # synthetic October 2023 snapshot
python run_all.py                    # the full pipeline
```

`run_all.py` trains every model at library defaults, then runs the bootstrap
confidence intervals, the participatory selection, the fairness analyses, and
every table and figure. `python run_all.py --help` lists the options; the most
useful are

```bash
python run_all.py --tune             # Optuna hyperparameter search, as in the paper
python run_all.py --skip mlp         # leave out a model
python run_all.py --start-at 12      # re-run from a step, reusing the trained models
```

The neural network uses a GPU when one is available and otherwise falls back to
the CPU with a warning. The figures use Helvetica, which is licensed and not
included: add `Helvetica.ttf` and `Helvetica-Bold.ttf` to `figures/fonts/` if you
have them. Otherwise another font is used, with a printed warning.

## What the pipeline does

Students are scored at two activation dates per academic year (15 October and
15 March). Each row is one enrollment at one activation date, with features
computed cumulatively up to that date.

- **Models.** An integer risk score (FasterRisk), logistic regression, XGBoost and
  a feedforward neural network, each on five nested feature sets: credits;
  + course performance; + enrollment; + learning-management-system activity;
  + sensitive attributes. Also one set of variables under evaluation, and one
  audit-only set that is never a selection candidate.
- **Split.** Trained on the 2016–2018 cohorts and tested cohort by cohort on
  2019–2023, with 2021 as the main comparison year. The split is on start year,
  so no enrollment appears on both sides.
- **Threshold.** Every model flags the same share of enrollments, the training
  cohorts' dropout rate and never above it, so models compare at one contact budget.
- **Baseline.** The Credits policy: flag any enrollment at least 60 ECTS behind.
- **Participatory selection.** The workshop's rules on data and explainability are
  applied as constraints (`src/workshop_rules.py`), for both surveys.
- **Fairness.** Differences by sex, nationality and entry age relative to cohort,
  ABROCA, and the fairness–accuracy frontier.
- **Pilot.** A comparison against the operational rule-based model in an October
  2023 outreach pilot.
- **Small cells.** Nothing in a figure or table rests on fewer than 3 students;
  the rule is implemented once, in `src/disclosure.py`.

`tools/` holds two checks the paper refers to, run by hand after the pipeline:
`python -m tools.check_selection_paired_ci` (the paired interval for the recall
gain that decides the model class) and `python -m tools.check_contact_flags`
(why Faculty B's operational model is evaluated by ranking only).

## Where the outputs go

| | |
|---|---|
| `figures/output/` | the paper's figures (PDF) |
| `figures/input/` | the values each figure draws, after suppression (CSV + JSON) |
| `results/descriptive/`, `results/*.tex` | the paper's tables (LaTeX) |
| `results/` | metrics, predictions, trained models and diagnostics |

Each figure has a `figures/scripts/prepare_*.py` step that writes the values it
shows and a `draw_*.py` step that draws the figure from those values alone. See
[`figures/README.md`](figures/README.md).

## Layout

```
ADAPTING.md       using the framework at another institution
examples/         the selection test on its own, with made-up recalls
instruments/      surveys, acceptability assessment, workshop plan
run_all.py        the entry point
src/              data loading, features, evaluation, metrics, selection rules,
                  fairness analyses, tables
models/           one file per model class
figures/          the figure scripts and README
tools/            checks run by hand
data/             the two synthetic-data generators
paths/paths.py    where data is read and results are written
```

`FAC_A` and `FAC_B` are the paper's Faculty A and Faculty B. Compared with the code run
on the student records, column and faculty names are translated and anonymized, some
diagnostics are left out, and two steps skip cleanly when other data selects a different
model, or none.

## License

Code: MIT. Materials in `instruments/`: CC BY 4.0. See [`LICENSE`](LICENSE).
