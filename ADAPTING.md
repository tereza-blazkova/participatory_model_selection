# Using the framework at another institution

You don't need this whole pipeline to use the framework. The workshop produces
rules, the rules are a small data structure, and the selection test needs only a
table of recalls. How much you take depends on what you already have.

## 1. Run the workshop

Start from [`instruments/`](instruments/). Replace the data sources and
explainability levels with the options your institution is actually considering.
A participant should only be asked about a choice that could really be made.

Run the survey before and after deliberation, and use the medians of the second
round as your rules.

## 2. Turn the answers into rules

Each survey answer is the minimum gain in recall that makes a step acceptable.
`src/workshop_rules.py` holds the scale:

| Survey answer | Constant | Required recall gain |
|---|---|---|
| Acceptable as long as accuracy does not worsen | `NO_THRESHOLD` | 0 pp |
| Small improvement | `SMALL_IMPROVEMENT` | 1 pp |
| Moderate improvement | `MODERATE_IMPROVEMENT` | 5 pp |
| Large improvement | `LARGE_IMPROVEMENT` | 10 pp |
| Never acceptable | `NEVER` | blocked |

Write the medians as a `SurveyRuleSet`, with one entry for each data step and each
explainability level. Examples are in `examples/select_with_your_rules.py`, and the
paper's are `ENDLINE_RULES` and `BASELINE_RULES`.

## 3. Run the selection test

`select_grid_config(results, test_year, rules)` needs one row per model and feature
set with these columns:

| Column | Meaning |
|---|---|
| `model`, `feature_set`, `test_year` | which configuration, on which cohort |
| `enr_recall` | recall at your contact budget |
| `enr_precision` | precision there; reported, and not used by the rules |

It also needs one `credits_policy` row, for the rule your institution uses now.
That rule is where the explainability comparison starts, and it is kept if no model
earns its required gain. Any modeling code can produce this table.

```bash
python -m examples.select_with_your_rules
```

## 4. Change what is specific to our setting

| What | Where | Why you would change it |
|---|---|---|
| The data ladder | `_FS_ORDER` in `src/workshop_rules.py`, or `fs_order` on a `SurveyRuleSet`; features in `DATA_CONFIGS` in `src/config.py` | Your data sources differ |
| The candidate models | `_ALL_EXPL_GROUPS` and `EXPLAINABILITY_LEVEL` in `src/workshop_rules.py`; `models/` | Your workshop considered other models or levels |
| The reference rule | `credits_policy_metrics()` in `src/metrics.py` | The 60-ECTS rule is Danish law |
| The train and test cohorts | `src/config.py` | Your years differ |
| The analysis population | `src/data.py`, `EXCLUDED_FACULTIES` in `src/config.py` | The exclusion reflects a data problem in our extract |
| The small-cell threshold | `MIN_CELL_SIZE` in `src/config.py` | It follows the rule of our research environment; use yours |
| The pilot | `src/pilot_snapshot_comparison.py`, `src/make_pilot_budget_table.py`, `figures/scripts/prepare_pilot.py` | It evaluates one specific pilot |

## 5. Running the full pipeline on your data

Point `DATA_ROOT` in `paths/paths.py` at a folder holding `fake_df.parquet` and
`pilot_snapshot.parquet`, in the schema the generators in `data/` write. You'll need to rename
your columns to match, or adapt `src/data.py`. All data is read through `src/data.py`.

Set `PILOT_CONTACT_THRESHOLD` in `src/config.py` to your own pilot's threshold, or
skip the pilot steps.

## Before relying on a selection

- **Check the decisive margins.** The test compares point estimates, so look at the
  uncertainty of any gain close to its requirement. `tools/check_selection_paired_ci.py`
  shows one way to do this.
- **Select and report on different cohorts.** A deployed selection should use a
  held-out cohort, and should require the lower confidence bound of the paired recall
  difference to clear the required gain. The test here applies rules to the cohort it
  also reports.
- **Audit fairness separately.** The rules here concern data and explainability.
  Fairness is audited after selection and does not change the selected model.
- **Keep what the rules cannot express.** Objections and conditions from the workshop
  go to the people who decide on implementation, alongside the selected model.
