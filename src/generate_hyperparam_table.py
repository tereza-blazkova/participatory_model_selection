"""Generate the SI hyperparameter table (tab:SI_hyperparams) from the code.

The table is built from the model code so it cannot drift from the search: a
number that is typed in two places will eventually disagree with itself.

Two sources, both authoritative:

  defaults       imported from each model module's _DEFAULTS dict — the exact
                 object passed to study.enqueue_trial(), so trial 0 is what the
                 table says it is.
  search ranges  read out of the trial.suggest_*() calls in the module source.
                 They are literals inside objective() closures and cannot be
                 imported, so they are parsed instead. The table therefore
                 renders the same numbers Optuna will execute, and a bound
                 changed in the code changes the table on the next run.

A non-literal bound (a name or an expression rather than a number) is refused
rather than guessed at — see _literal().

Outputs results/descriptive/si_hyperparams.tex, for \\input{}.

Usage:
    python src/generate_hyperparam_table.py
"""
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from paths.paths import RESULTS_PATH

import models.fasterrisk_model as fasterrisk_model
import models.logistic_regression as logistic_regression
import models.mlp_model as mlp_model
import models.xgboost_model as xgboost_model
from src.display import MODEL_LABELS

OUT_DIR = RESULTS_PATH / "descriptive"

## Table row order. Within a model, parameters appear in the order their
## suggest_*() calls appear in the source, so the table cannot fall out of step
## with the code by gaining or losing a row.
MODELS = [
    ("Logistic regression", logistic_regression),
    ("XGBoost",             xgboost_model),
    (MODEL_LABELS["mlp"],   mlp_model),
]

## Display names. A parameter missing here is printed as \texttt{name}, so
## adding an Optuna axis never silently drops a row.
LABELS = {
    "C":            r"$C$ (inverse regularization strength)",
    "n_layers":     "hidden layers",
    "n_units_l*":   "units per hidden layer (each drawn separately)",
    "alpha":        r"elastic-net strength $\alpha$",
    "l1_ratio":     r"L1 share of the penalty $\lambda$",
    "lr":           "learning rate (constant)",
    "dropout_rate": "dropout rate",
}


def _literal(node: ast.AST, context: str):
    """A suggest_*() argument, or a hard failure naming what could not be read."""
    try:
        return ast.literal_eval(node)
    except ValueError:
        raise ValueError(
            f"{context}: search bound is not a literal, so the table cannot "
            f"report it ({ast.dump(node)[:60]}). Either inline the number or "
            f"name the parameter with a literal."
        ) from None


def _num(value) -> str:
    """
    LaTeX for one number.

    Powers of ten are written as $10^{n}$ only when the range reaches small
    magnitudes — |n| >= 3 — where decimals read badly. So 1e-4 becomes
    $10^{-4}$, while 1.0, 10.0 and 0.05 stay as they are.
    """
    if isinstance(value, float) and value > 0:
        exponent = round(__import__("math").log10(value))
        if abs(10.0 ** exponent - value) < 1e-12 and abs(exponent) >= 3:
            return f"10^{{{exponent}}}"
    if isinstance(value, float) and value.is_integer() and abs(value) < 1000:
        return f"{value:.1f}" if value != int(value) or "." in repr(value) else str(int(value))
    return str(value)


def _render_range(kind: str, lo, hi, keywords: dict) -> str:
    """One suggest_*() call as the table's 'Search range' cell."""
    ## Powers of ten are chosen per RANGE, not per number, so both ends match.
    use_power = any(
        isinstance(v, float) and v > 0
        and abs(10.0 ** round(__import__("math").log10(v)) - v) < 1e-12
        and abs(round(__import__("math").log10(v))) >= 3
        for v in (lo, hi)
    )
    fmt = (lambda v: f"10^{{{round(__import__('math').log10(v))}}}") if use_power else _num
    span = rf"$[{fmt(lo)},\, {fmt(hi)}]$"

    if kind == "suggest_int":
        text = f"integer {span}"
        if "step" in keywords:
            text += f", step ${keywords['step']}$"
        return text
    return f"log-uniform {span}" if keywords.get("log") else f"uniform {span}"


def _search_space(module) -> dict[str, str]:
    """Every trial.suggest_*() call in a module, as rendered range cells."""
    tree = ast.parse(Path(module.__file__).read_text())
    space: dict[str, str] = {}
    ## Visit the suggest_*() calls in SOURCE order, so rows follow the code
    ## (ast.walk alone is breadth-first and would put nested calls last).
    calls = sorted((n for n in ast.walk(tree)
                    if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and n.func.attr.startswith("suggest_")),
                   key=lambda n: (n.lineno, n.col_offset))
    for node in calls:
        ## A per-layer axis is named with an f-string (n_units_l{i}) — one
        ## row for all layers, keyed by its constant prefix plus "*".
        if isinstance(node.args[0], ast.JoinedStr):
            name = "".join(v.value for v in node.args[0].values
                           if isinstance(v, ast.Constant)) + "*"
        else:
            name = _literal(node.args[0], f"{module.__name__}")
        lo = _literal(node.args[1], f"{module.__name__}.{name}")
        hi = _literal(node.args[2], f"{module.__name__}.{name}")
        kw = {k.arg: _literal(k.value, f"{module.__name__}.{name}") for k in node.keywords}
        space[name] = _render_range(node.func.attr, lo, hi, kw)
    return space


def _default(module, name: str) -> str:
    """The enqueued trial-0 value. A per-layer axis shows the whole default shape."""
    if name.endswith("*"):
        prefix = name[:-1]
        widths = [v for k, v in module._DEFAULTS.items() if k.startswith(prefix)]
        return f"$({', '.join(map(str, widths))})$"
    return f"${_num(module._DEFAULTS[name])}$"


def generate() -> str:
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        ## \resizebox keeps the widest row inside the text block. Without it the
        ## architecture row overflows by ~86pt.
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{llll}",
        r"\toprule",
        r"Model & Parameter & Search range & Default \\",
        r"\midrule",
    ]

    for display, module in MODELS:
        space = _search_space(module)
        names = list(space)
        for i, name in enumerate(names):
            ## \multirow spans the model name down its own parameters.
            cell = (rf"\multirow{{{len(names)}}}{{*}}{{{display}}}"
                    if len(names) > 1 and i == 0 else "")
            if len(names) == 1:
                cell = display
            label = LABELS.get(name, rf"\texttt{{{name.replace('_', chr(92) + '_')}}}")
            lines.append(rf"  {cell} & {label} & {space[name]} & {_default(module, name)} \\")
        lines.append(r"\midrule")

    ## FasterRisk runs no search, so its two rows are structural values read
    ## straight off the module rather than parsed out of a suggest_*() call.
    ## k differs by entry point, and both are reported: the plain pipeline path
    ## (train) and the tuned path (train_tuned) do not agree.
    k_default = fasterrisk_model._K_DEFAULT
    lines += [
        r"\multirow{2}{*}{FasterRisk}",
        r"  & $k$ (max active terms) & $\min(10,\,d)$; $d$ = binary feature "
        r"count after preprocessing & --- \\",
        rf"  & coefficient bounds & fixed $[{fasterrisk_model._LB},\,"
        rf"{fasterrisk_model._UB}]$ & --- \\",
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\vspace{4pt}",
        r"\raggedright\footnotesize",
        r"\textit{Note:} The neural network follows the tabular feedforward network of",
        r"\citet{nielsenBirthAdulthood2025}: ReLU layers, an elastic-net penalty",
        r"$\alpha\,(\lambda\|w\|_1 + (1-\lambda)\|w\|_2^2)$ on the weights added to",
        r"the loss, and AdamW with default parameters at a constant learning rate.",
        rf"Every candidate trains for {mlp_model._EPOCHS} epochs at batch size "
        rf"{mlp_model._BATCH_SIZE}, is scored on",
        r"each held-out cohort after every epoch, and keeps the epoch with the highest",
        r"mean AUC; the final model is refitted on the full training partition for",
        r"that many epochs. Dropout is searched in addition to their space, and a",
        r"rate of 0 is their network. Logistic regression and XGBoost are",
        r"refitted on the full training partition after the search, so no model",
        r"reserves a private validation split. FasterRisk's $k$ is $\min(10,\,d)$ on",
        r"the tuned path",
        rf"(\texttt{{train\_tuned()}}); an untuned fit via \texttt{{train()}} uses",
        rf"$k={k_default}$.",
        r"\caption{\textbf{Hyperparameter search spaces for Optuna tuning.} TPE sampler,",
        r"leave-one-cohort-out CV over the three training years, optimizing",
        r"enrollment-level ROC AUC. Trial~0 is the default listed, so tuning cannot",
        r"score below it in cross-validation. For logistic regression and",
        r"XGBoost the defaults are the library defaults and poor trials are pruned",
        r"(median pruner); the neural network runs every trial in full, as in",
        r"\citet{nielsenBirthAdulthood2025}, whose search space reports no defaults,",
        r"so the neural network's trial~0 is a mild configuration inside it. FasterRisk is not",
        r"tuned by search; its structural parameters are set from the feature set",
        r"directly. This table is generated by",
        r"\texttt{src/generate\_hyperparam\_table.py} directly from the model",
        r"modules.}",
        r"\label{tab:SI_hyperparams}",
        r"\end{table}",
    ]

    content = "\n".join(lines) + "\n"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "si_hyperparams.tex"
    out.write_text(content)
    print(f"Saved: {out}")


    return content


if __name__ == "__main__":
    generate()
