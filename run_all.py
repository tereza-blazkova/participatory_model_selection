"""
Full pipeline orchestrator.

Steps:
  1.  Model training & evaluation       (src/evaluation.py)
  3.  Bootstrap CIs                     (src/bootstrap.py  --row-level --pilot-snapshot)
  4.  Workshop selection & evaluation   (src/workshop_evaluation.py; again with
                                         --selection-year 2019 for the SI version)
  8.  Used-vars FasterRisk model check (src/used_vars_scorecard.py)
  9.  Descriptive tables                (src/descriptive_table.py)
  [step 12 skipped with --skip-plots]
  12. Paper figures                     (figures/scripts/prepare_*.py, then draw_*.py)
  13. Results tables                    (src/make_results_tables.py,
                                         src/bachelor_only_audit.py,
                                         src/make_pilot_budget_table.py)
  14. Selection gate                    (figures/scripts/prepare_selection_gate.py,
                                         src/generate_selection_table.py,
                                         figures/scripts/draw_selection_gate.py —
                                         main year, then the SI 2019 version;
                                         src/selection_by_cohort.py,
                                         src/selection_robustness.py)
  15. Stability table (LaTeX)           (src/generate_stability_table.py)
  [steps 16–19 skipped with --skip-fairness]
  16. Fairness audit                    (src/fairness_audit.py)
  17. SI fairness tables (LaTeX)        (src/make_si_fairness_table.py,
                                         src/make_sensitive_audit_table.py)
  18. Equalized-recall threshold audit  (src/equalized_recall_audit.py)
  19. Equalized-recall table (LaTeX)    (src/make_equalized_recall_table.py)
  21. Rule-based threshold recalibration (src/check_rule_based_recalibration.py)
  [step 22 skipped with --skip-fa-frontier]
  22. Fairness-accuracy frontier — both figures, runs last
      (figures/scripts/{prepare,draw}_fa_frontier*.py)

Only the paper's figures (figures/output/) and tables (results/**/*.tex) are
produced, plus the intermediate files later steps read. Step numbers are not
contiguous.

Usage examples:
  python run_all.py                          # skip training if results exist, run all other steps
  python run_all.py --retrain               # force re-train with defaults, then all steps
  python run_all.py --tune                  # tune all models (30 trials), then all steps
  python run_all.py --use-saved-params      # re-train from saved tuned params (no search)
  python run_all.py --strict                # abort if a feature set collapses
  python run_all.py --skip mlp fasterrisk   # skip specific models during training
  python run_all.py --skip-plots            # everything except the paper figures of Step 12
  python run_all.py --skip-fairness         # skip fairness audit (Steps 16–19)
  python run_all.py --skip-fa-frontier      # skip the FA frontier figures (Step 22)
  python run_all.py --fa-frontier-only      # ONLY the FA frontier figures, nothing else
  python run_all.py --start-at 16           # resume at Step 16 after a crash there
  python run_all.py --skip-bootstrap --skip-workshop  # reuse bootstrap CIs and workshop selection

Flag reference:
  --start-at N         resume at step N (1-22), skipping the earlier ones. The pipeline
                       is linear and every step reads finished artefacts off disk, so a
                       run that died at step N can pick up there. Training (Step 1) is
                       skipped for any N > 1 and the run aborts up front if
                       evaluation_results.csv is missing, since nothing downstream works
                       without it. Combines with the --skip-* flags; either one excluding
                       a step is enough to skip it.
  --retrain            force re-training with defaults (warns if tuned params exist)
  --use-saved-params   re-train from saved _best_params.json, no search (implies re-training)
  --strict             abort if a feature set collapses (all-null/constant cols, flat scores)
  --tune               run Optuna hyperparameter search (implies re-training)
  --n-iter N           tuning trials per model (default 30; only used with --tune)
  --cv K               CV folds during tuning (default 5; only used with --tune)
  --skip MODEL...      skip one or more models, e.g. --skip mlp fasterrisk
  --ruleset STR        which workshop survey administration supplies the selection rules:
                       endline (post-deliberation, default — the paper's Participatory
                       model) or baseline (pre-deliberation)
  --skip-bootstrap     skip bootstrap CIs (Step 3); use if bootstrap_cis.csv already exists
  --skip-workshop      skip workshop selection (Step 4)
  --skip-plots         skip the paper figures of Step 12; every table still runs
  --skip-fairness      skip fairness/equalized-recall steps (Steps 16–19)
  --skip-fa-frontier   skip the fairness-accuracy frontier figures (Step 22)
  --fa-frontier-only   run ONLY Step 22; assumes evaluation and workshop selection
                       have already produced enrollment_predictions.csv,
                       evaluation_results.csv and workshop_config.json
  --fa-year YEAR       test year for the FA frontier figures (default: MAIN_TEST_YEAR)
  --fa-n-boot N        group-skew bootstrap draws (default 1000; 0 disables)
  --fa-zoom            frame the FA frontier on the frontiers alone (drop operating points)
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from paths.paths import RESULTS_PATH
from src.config import MAIN_TEST_YEAR, SI_SELECTION_YEAR
from src.workshop_rules import RULE_SETS


_N_STEPS = 22

# Set once from --start-at. Steps below it are not run: the pipeline is linear
# and every step reads finished artefacts off disk, so a run that died at step N
# can be resumed at N without redoing the ones that already wrote their outputs.
_START_AT = 1


def _banner(step: int, label: str) -> None:
    # flush: when stdout is redirected to a log it is block-buffered, so without
    # this the parent's banners land after all the child output and the log
    # reads as if the steps ran in a different order than they did
    print(f"\n{'=' * 64}", flush=True)
    print(f"  Step {step} / {_N_STEPS} — {label}", flush=True)
    print(f"{'=' * 64}", flush=True)


def _skipped_by_start_at(step: int) -> bool:
    """True if `step` falls before --start-at; prints a one-line note if so."""
    if step >= _START_AT:
        return False
    print(f"  · step {step} skipped (--start-at {_START_AT})", flush=True)
    return True


def _note_skipped(first: int, last: int, message: str) -> None:
    """
    Print a --skip-* notice, unless --start-at already excluded the whole range.
    Without this, `--start-at 16 --skip-plots` would announce it was skipping
    steps that were never going to run.
    """
    if last >= _START_AT:
        print(f"\n{message}", flush=True)


def _run(cmd: list[str], step: int, label: str) -> None:
    if _skipped_by_start_at(step):
        return
    _banner(step, label)
    subprocess.run(cmd, check=True)


## One entry per call, in order: the prepare steps first, because the draw
## steps read what they write. A draw step takes seconds; the cost is all in
## the prepare half, which is where the pipeline is read.
_SHAP_PARTICIPATORY_STEM = "shap_logistic_regression_b_credits_performance"
_PAPER_FIGURE_STEPS = [
    ("figures/scripts/prepare_pilot.py",                 "Pilot figure inputs (Oct 2023 snapshot)"),
    ("figures/scripts/prepare_dropout_timing.py",        "Dropout-timing input"),
    ("figures/scripts/prepare_dropout_timing.py --max-ml",
     "Dropout-timing input, with Max ML"),
    ("figures/scripts/prepare_performance_by_period.py", "Performance-by-period input"),
    ("figures/scripts/prepare_model_comparison.py",      "Model-comparison input"),
    ("figures/scripts/prepare_shap.py",                  "SHAP violin inputs"),
    ("figures/scripts/draw_pilot_breakdown.py --stem pilot_breakdown_faculty_a",
     "Figure: pilot outcome breakdown, Faculty A (real and recalibrated)"),
    ("figures/scripts/draw_pilot_auc.py --stem pilot_auc_faculty_a",
     "Figure: pilot ROC AUC, Faculty A (main text)"),
    ("figures/scripts/draw_pilot_auc.py --stem pilot_auc_faculty_b",
     "Figure: pilot ROC AUC, Faculty B (SI)"),
    ("figures/scripts/draw_pilot_points_profile.py --stem pilot_points_profile_faculty_a",
     "Figure: pilot points profile, Faculty A"),
    ("figures/scripts/draw_pilot_points_profile.py --stem pilot_points_profile_faculty_b",
     "Figure: pilot points profile, Faculty B"),
    ("figures/scripts/draw_pilot_abroca.py",             "Figure: pilot ABROCA heatmap"),
    ("figures/scripts/draw_scorecard.py",                "Figure: FasterRisk scorecard"),
    ("figures/scripts/draw_dropout_timing.py",           "Figure: dropout timing"),
    ("figures/scripts/draw_dropout_timing.py --stem dropout_timing_maxml",
     "Figure: dropout timing, with Max ML"),
    ("figures/scripts/draw_performance_by_period.py",    "Figure: performance by period"),
    ("figures/scripts/draw_model_comparison.py",         "Figure: model comparison"),
    ("figures/scripts/draw_model_comparison.py --stem model_comparison_2021_2023",
     "Figure: model comparison, with the Oct 2023 pilot panel"),
    ("figures/scripts/draw_shap.py --stem shap_mlp_e_plus_sensitive",
     "Figure: SHAP, Max ML"),
    ("figures/scripts/draw_shap.py --stem shap_logistic_regression_b_credits_performance",
     "Figure: SHAP, Participatory"),
    ("figures/scripts/draw_shap.py --stem shap_participatory_maxml",
     "Figure: SHAP, Participatory and Max ML side by side"),
]


def _run_fa_frontier_or_skip(py: str, args) -> None:
    """A group too small to pass the small-cell rule leaves no frontier to draw."""
    try:
        _run_fa_frontier(py, args)
    except subprocess.CalledProcessError:
        print("\n  The FA frontier could not be built on this data, usually because a "
              "subgroup is too small: skipping its figures.")


def _run_fa_frontier(py: str, args) -> None:
    """
    Step 22 — every fairness-accuracy frontier figure, in one place, last.

    Runs last because it consumes finished artefacts and produces nothing any
    other step needs: enrollment_predictions.csv and evaluation_results.csv from
    Step 1, and workshop_config.json from Step 4 (the main figure resolves
    the Participatory model from it, and the workshop-permissible frontier
    re-derives the selection gates live).

    Two figures:
      - fa_frontier_{year}.pdf          policy view: both nested frontiers, the
                                        cost-of-constraints band, the deployed
                                        operating points
      - fa_frontier_ladders_{year}.pdf  2×3 decomposition by data availability
                                        and by model complexity
    """
    if args.skip_fa_frontier:
        _note_skipped(22, 22,
                      "Skipping fairness-accuracy frontier figures (--skip-fa-frontier).")
        return

    ## Both frontiers are built the figures/ way: prepare reads the predictions
    ## and also writes the summary CSVs, draw reads only the prepared input.
    main_cmd = [py, "figures/scripts/prepare_fa_frontier.py"]
    ladder_cmd = [py, "figures/scripts/prepare_fa_frontier_ladders.py"]
    if args.fa_year is not None:
        for cmd in (main_cmd, ladder_cmd):
            cmd += ["--year", str(args.fa_year)]
    if args.fa_n_boot is not None:
        main_cmd += ["--n-boot", str(args.fa_n_boot)]
    if args.fa_zoom:
        main_cmd += ["--zoom"]

    _run(main_cmd, 22,
         "FA frontier (1/2): by sex, nationality and entry age  "
         "(figures/input/fa_frontier_{year}.csv)")
    _run([py, "figures/scripts/draw_fa_frontier.py"]
         + (["--year", str(args.fa_year)] if args.fa_year is not None else []),
         22, "FA frontier (1/2): figure  (figures/output/fa_frontier_{year}.pdf)")
    _run(ladder_cmd, 22,
         "FA frontier (2/2): data and model ladders  "
         "(figures/input/fa_frontier_ladders_{year}.csv)")
    _run([py, "figures/scripts/draw_fa_frontier_ladders.py"]
         + (["--year", str(args.fa_year)] if args.fa_year is not None else []),
         22, "FA frontier (2/2): figure  "
             "(figures/output/fa_frontier_ladders_{year}.pdf)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full EWS modeling pipeline.")
    parser.add_argument("--retrain",   action="store_true",
                        help="Re-run training even if results already exist.")
    parser.add_argument("--tune",      action="store_true",
                        help="Run Optuna hyperparameter tuning (implies --retrain).")
    parser.add_argument("--n-iter",    type=int, default=30,
                        help="Tuning trials per model (default 30, only used with --tune).")
    parser.add_argument("--cv",        type=int, default=5,
                        help="CV folds during tuning (default 5, only used with --tune).")
    parser.add_argument("--skip",      nargs="+", default=[], metavar="MODEL",
                        help="Models to skip during training, e.g. --skip mlp fasterrisk.")
    parser.add_argument("--use-saved-params", action="store_true", dest="use_saved_params",
                        help="Re-train from saved _best_params.json instead of the fixed defaults "
                             "(no Optuna search). Implies re-training. Use after a --tune run.")
    parser.add_argument("--strict", action="store_true",
                        help="Abort if any feature set collapses (all-null or constant columns, flat scores).")
    parser.add_argument("--skip-bootstrap", action="store_true",
                        help="Skip bootstrap CIs (Step 3); use if bootstrap_cis.csv already exists.")
    parser.add_argument("--skip-workshop", action="store_true",
                        help="Skip workshop selection (Step 4).")
    parser.add_argument("--ruleset", default="endline", choices=sorted(RULE_SETS),
                        help="Workshop survey administration supplying the selection rules "
                             "in Step 4: endline (post-deliberation, default) or baseline "
                             "(pre-deliberation).")
    parser.add_argument("--skip-plots", action="store_true",
                        help="Skip the paper figures of Step 12; every table still runs. Does not affect the selection gate (Step 14) or the FA frontier (Step 22).")
    parser.add_argument("--skip-fairness", action="store_true",
                        help="Skip fairness audit, SI table, and equalized-recall steps (Steps 16–19). The FA frontier is Step 22 — use --skip-fa-frontier for that.")
    parser.add_argument("--skip-fa-frontier", action="store_true",
                        help="Skip the fairness-accuracy frontier figures (Step 22).")
    parser.add_argument("--fa-frontier-only", action="store_true",
                        help="Run ONLY Step 22, assuming earlier artefacts exist.")
    parser.add_argument("--fa-year", type=int, default=None,
                        help="Test year for the FA frontier figures (default: MAIN_TEST_YEAR).")
    parser.add_argument("--fa-n-boot", type=int, default=None,
                        help="Group-skew bootstrap draws (default 1000; 0 disables).")
    parser.add_argument("--fa-zoom", action="store_true",
                        help="Frame the FA frontier on the frontiers alone, "
                             "excluding the deployed operating points.")
    parser.add_argument("--start-at", type=int, default=1, dest="start_at",
                        metavar="N",
                        help=f"Resume at step N (1-{_N_STEPS}); earlier steps are "
                             "skipped. Use after a late step crashed — everything "
                             "before it already wrote its outputs.")
    args = parser.parse_args()

    if not 1 <= args.start_at <= _N_STEPS:
        parser.error(f"--start-at must be between 1 and {_N_STEPS}, got {args.start_at}")

    global _START_AT
    _START_AT = args.start_at
    if _START_AT > 1:
        print(f"\nResuming at step {_START_AT} / {_N_STEPS} — steps 1–{_START_AT - 1} "
              "will be skipped (their outputs are read from results/).")

    py = sys.executable

    ## Before anything reads results/: a run that skips training works from files
    ## an earlier run wrote, and those may hold a model the code no longer has.
    from src.prune_results import prune_results
    prune_results()

    # Step 22 consumes finished artefacts only, so it can be run on its own
    # after a full pipeline run — useful while iterating on the figures.
    if args.fa_frontier_only:
        _run_fa_frontier_or_skip(py, args)
        print(f"\n{'=' * 64}")
        print("  Done (FA frontier only).")
        print(f"{'=' * 64}\n")
        return

    # ── Step 1: training & evaluation ─────────────────────────────────────────
    results_csv = RESULTS_PATH / "evaluation_results.csv"
    metadata    = RESULTS_PATH / "models" / "train_metadata.json"
    has_results = results_csv.exists() and metadata.exists()

    if _START_AT > 1:
        # Resuming: everything downstream reads the artefacts on disk, so a
        # missing evaluation_results.csv is a real problem rather than a cue to
        # train. Say so here instead of failing four steps later.
        _skipped_by_start_at(1)
        if not has_results:
            print(f"\nERROR: --start-at {_START_AT} skips training, but "
                  f"{results_csv} (or models/train_metadata.json) is missing.\n"
                  "  Every later step reads them. Run without --start-at first.")
            sys.exit(1)
    else:
        _run_training(py, args, has_results)

    # ── Step 3: bootstrap CIs (enrollment + row + pilot snapshot) ──────────────────
    if args.skip_bootstrap:
        _note_skipped(3, 3, "Skipping bootstrap CIs (--skip-bootstrap).")
    else:
        _run(
            [py, "src/bootstrap.py", "--row-level", "--pilot-snapshot"], 3,
            "Bootstrap CIs  (bootstrap_cis.csv + bootstrap_cis_pilot_snapshot.csv)",
        )

    # ── Step 4: workshop selection ────────────────────────────────────────────
    if args.skip_workshop:
        _note_skipped(4, 4, "Skipping workshop selection (--skip-workshop).")
    else:
        _run(
            [py, "src/workshop_evaluation.py", "--ruleset", args.ruleset], 4,
            f"Workshop selection & evaluation  (ruleset: {args.ruleset})",
        )
        ## The SI version: endline rules applied to another cohort. Writes
        ## `_sel{year}` files only — never the stored Participatory model.
        _run(
            [py, "src/workshop_evaluation.py", "--ruleset", "endline",
             "--selection-year", str(SI_SELECTION_YEAR)], 4,
            f"Workshop selection on {SI_SELECTION_YEAR} (SI)",
        )

    _run([py, "src/used_vars_scorecard.py"], 8, "Used-vars FasterRisk model check")
    _run([py, "src/descriptive_table.py"],   9, "Descriptive tables")
    if args.skip_plots:
        _note_skipped(12, 12, "Skipping the paper figures (--skip-plots).")
    else:
        ## Every paper figure, as input + script + output. Each prepare_*.py
        ## reads the pipeline and writes figures/input/; each draw_*.py reads
        ## only that. They are listed prepare-then-draw so a figure is never
        ## drawn from an input older than the run. See figures/README.md.
        for script, label in _PAPER_FIGURE_STEPS:
            ## The Participatory model's SHAP figure is named after whatever the
            ## rules selected, which differs from the paper's on other data.
            if script.endswith(_SHAP_PARTICIPATORY_STEM):
                from src.workshop_rules import participatory_config
                model, feature_set = participatory_config(RESULTS_PATH)
                if model == "credits_policy":
                    print("\n  The credits policy was retained; no SHAP figure for it.")
                    continue
                script = script.replace(_SHAP_PARTICIPATORY_STEM, f"shap_{model}_{feature_set}")
            _run([py, *script.split()], 12, label)

    # ── Step 13: results tables (needs bootstrap CSVs; always runs) ──────────
    _run([py, "src/make_results_tables.py"], 13, "Results tables")
    ## Keyed to the same step: it reads enrollment_predictions.csv and writes a
    ## results table, so --start-at treats the pair as one.
    _run(
        [py, "src/bachelor_only_audit.py"], 13,
        "Bachelor's-only audit  (results/descriptive/bachelor_only.tex)",
    )
    _run(
        [py, "src/make_pilot_budget_table.py"], 13,
        "Pilot at the real contact budget, Faculty A  (results/descriptive/pilot_budget_table.tex)",
    )

    # ── Step 14: selection gate — needs evaluation_results.csv ───────────────
    # Same verdicts, two renderings: the table states them, the figure shows the
    # rule each one came from. Both are published, so both must withhold the same
    # cells — which is why the prepare step runs FIRST and they read its output
    # rather than evaluation_results.csv. Ordering is load-bearing here, not
    # convenience. All three keyed to step 14, so --start-at treats them as one.
    _run(
        [py, "figures/scripts/prepare_selection_gate.py"], 14,
        "Selection-gate input  (figures/input/selection_gate_{year}.csv)",
    )
    _run(
        [py, "src/generate_selection_table.py"], 14,
        "Selection-gate table  (results/descriptive/selection_table.tex)",
    )
    _run(
        [py, "figures/scripts/draw_selection_gate.py"], 14,
        "Selection-gate figure (figures/output/selection_gate_{year}.pdf)",
    )
    ## The SI version: the same three, selecting on SI_SELECTION_YEAR and
    ## reporting the pick on the main year. Same order, same reason.
    y = str(SI_SELECTION_YEAR)
    ## On other data the rules may retain the Credits policy on this cohort. The
    ## figure then has no selected model to circle, and only this SI version is lost.
    try:
        _run(
            [py, "figures/scripts/prepare_selection_gate.py", "--test-year", y,
             "--evaluation-year", str(MAIN_TEST_YEAR)], 14,
            f"Selection-gate input, selected on {y} (SI)",
        )
        _run(
            [py, "src/generate_selection_table.py", "--test-year", y], 14,
            f"Selection-gate table, selected on {y}  (results/descriptive/selection_table_{y}.tex)",
        )
        _run(
            [py, "figures/scripts/draw_selection_gate.py", "--year", y], 14,
            f"Selection-gate figure, selected on {y} (figures/output/selection_gate_{y}.pdf)",
        )
    except subprocess.CalledProcessError:
        print(f"\n  No selected model on {y}: skipping its selection-gate table and figure.")
    _run(
        [py, "src/selection_by_cohort.py"], 14,
        "Selection by cohort  (results/descriptive/selection_by_cohort.tex)",
    )
    ## Each rule threshold moved on its own, and the ladder order varied.
    _run(
        [py, "src/selection_robustness.py"], 14,
        "Selection robustness  (results/descriptive/selection_robustness.tex)",
    )

    # ── Step 15: stability table (LaTeX) — needs evaluation_results.csv ───────
    _run(
        [py, "src/generate_stability_table.py"], 15,
        "Stability table  (results/descriptive/stability_table.tex)",
    )
    # The hyperparameter table reads the model modules, not results/, so it has
    # no data dependency and could run anywhere. Keyed to step 15 so it travels
    # with the other SI LaTeX tables and --start-at treats them as one step.
    _run(
        [py, "src/generate_hyperparam_table.py"], 15,
        "Hyperparameter table  (results/descriptive/si_hyperparams.tex)",
    )

    if args.skip_fairness:
        _note_skipped(16, 19,
                      "Skipping fairness audit and equalized-recall steps (--skip-fairness). "
                      "The FA frontier is Step 22 and still runs.")
    else:
        _run([py, "src/fairness_audit.py", "--analysis", "snapshot"], 16,
             "Fairness audit  (results/fairness/fairness_snapshot.csv)")

        # ── Step 17: SI fairness tables (LaTeX) — depends only on Step 16 ────
        _run(
            [py, "src/make_si_fairness_table.py"], 17,
            "SI fairness tables  (results/si_fairness_tables.tex)",
        )
        # The sensitive-attribute audit (b_plus_sensitive vs b_credits_performance)
        # is an SI fairness table too; kept in this step so step numbers hold.
        _run(
            [py, "src/make_sensitive_audit_table.py"], 17,
            "SI sensitive-attribute audit  (results/descriptive/sensitive_audit_table.tex)",
        )

        # ── Steps 18–19: equalized-recall audit — audits whichever model the
        #    workshop rules selected in Step 4, so it must run after it ───────
        _run(
            [py, "src/equalized_recall_audit.py"], 18,
            "Equalized-recall threshold audit  "
            "(results/fairness/equalized_recall_audit.csv)",
        )
        _run(
            [py, "src/make_equalized_recall_table.py"], 19,
            "Equalized-recall table  (results/equalized_recall_table.tex)",
        )

    # ── Step 21: rule-based threshold recalibration table ────────────────────
    _run(
        [py, "src/check_rule_based_recalibration.py"], 21,
        "Rule-based threshold recalibration  "
        "(results/descriptive/pilot_snapshot_recalibration.tex)",
    )

    _run_fa_frontier_or_skip(py, args)

    print(f"\n{'=' * 64}")
    print("  Done.")
    print(f"{'=' * 64}\n")


def _run_training(py: str, args, has_results: bool) -> None:
    """Step 1 — train and evaluate, unless finished results are already on disk."""
    results_csv = RESULTS_PATH / "evaluation_results.csv"
    need_train  = args.tune or args.retrain or args.use_saved_params or not has_results

    # Warn if --retrain would silently drop tuning (--use-saved-params refits from them)
    if args.retrain and not args.tune and not args.use_saved_params:
        saved_params = list((RESULTS_PATH / "models").glob("*_best_params.json"))
        if saved_params:
            print(
                f"\nWARNING: --retrain without --tune will overwrite your tuned .joblib models\n"
                f"  with default-parameter models.  {len(saved_params)} best_params.json file(s)\n"
                f"  will be preserved on disk but ignored.\n"
                f"  Use --use-saved-params to refit those tuned configurations without re-running\n"
                f"  the search, --tune to search again, or omit --retrain to keep existing models."
            )
            answer = input("  Continue anyway? [y/N] ").strip().lower()
            if answer != "y":
                print("Aborted.")
                sys.exit(0)

    cpus = os.cpu_count() or 1
    print(f"\nSystem: {cpus} CPU(s) available | Optuna n_jobs=1 (sequential, local mode)")

    if not need_train:
        print(f"\nFound existing results — skipping training.")
        print(f"  {results_csv}")
        print(f"  Pass --retrain to force re-training, --tune to run with tuning.")
    else:
        ## Name the flag that actually forced it: --retrain and
        ## --use-saved-params fit different configurations.
        if not has_results:
            reason = "no existing results"
        elif args.tune:
            reason = "--tune"
        elif args.use_saved_params:
            reason = "--use-saved-params"
        else:
            reason = "--retrain"
        print(f"\nRunning training ({reason}).")
        eval_cmd = [py, "src/evaluation.py"]
        if args.tune:
            eval_cmd += ["--tune", "--n-iter", str(args.n_iter), "--cv", str(args.cv)]
        if args.use_saved_params:
            eval_cmd += ["--use-saved-params"]
        if args.strict:
            eval_cmd += ["--strict"]
        if args.skip:
            eval_cmd += ["--skip"] + args.skip
        _run(eval_cmd, 1, "Model training & evaluation")


if __name__ == "__main__":
    main()
