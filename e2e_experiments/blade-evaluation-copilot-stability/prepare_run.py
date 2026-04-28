"""Prepare run directories for each Blade dataset.

For each dataset, creates a subdirectory under the output directory containing:
  - info.json   (task metadata with research question)
  - {dataset}.csv  (the data)
  - AGENTS.md   (instructions for Codex)
  - interp_models.py (custom interpretability tools, only in --mode custom)

Usage:
    python prepare_run.py                          # prepare all datasets (standard mode)
    python prepare_run.py --mode custom            # prepare with custom interp tools
    python prepare_run.py --dataset soccer         # prepare one dataset
"""

import argparse
import json
import os
import shutil
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Try example-blade-repo first, fall back to existing outputs/ directory
_BLADE_DIR = os.path.join(
    SCRIPT_DIR, "..", "example-blade-repo", "blade_bench", "datasets"
)
_CANDIDATES = [
    os.path.join(SCRIPT_DIR, "outputs_standard_run1"),
    os.path.join(SCRIPT_DIR, "..", "blade-evaluation-copilot", "outputs_standard_run1"),
    os.path.join(SCRIPT_DIR, "..", "blade-evaluation-codex", "outputs_standard_run1"),
]
DATASETS_DIR = _BLADE_DIR if os.path.isdir(_BLADE_DIR) else next(
    (p for p in _CANDIDATES if os.path.isdir(p)),
    os.path.join(SCRIPT_DIR, "outputs"),
)

# Location of the agentic-imodels skill library (used in custom_v2 mode).
AGENTIC_IMODELS_DIR = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "..", "result_libs_processed", "agentic-imodels")
)

DATASETS = [
    "affairs",
    "amtl",
    "boxes",
    "caschools",
    "crofoot",
    "fertility",
    "fish",
    "hurricane",
    "mortgage",
    "panda_nuts",
    "reading",
    "soccer",
    "teachingratings",
]

AGENTS_MD_STANDARD = """You are an expert data scientist. You MUST write and execute a Python script to analyze a dataset and answer a research question.

## Instructions

1. Read `info.json` to get the research question and dataset metadata.
2. Load the dataset from `{dataset_name}.csv`.
3. The `imodels` library at <https://github.com/csinva/imodels> provides
   interpretable scikit-learn-compatible regressors and classifiers that
   you can use alongside scikit-learn.
4. Write a Python script called `analysis.py` that:
   - Loads and explores the data (summary statistics, distributions, correlations).
   - Builds interpretable models using scikit-learn and imodels to understand feature relationships.
   - Performs appropriate statistical tests (t-tests, ANOVA, regression, etc.).
   - Interprets the results in context of the research question.
5. **Execute the script** by running: `python3 analysis.py`
6. The script MUST write a file called `conclusion.txt` containing ONLY a JSON object:

```json
{{"response": <integer 0-100>, "explanation": "<your reasoning>"}}
```

Where `response` is a Likert scale score: 0 = strong "No", 100 = strong "Yes".

## Interpretability Tools

You should use interpretable models to understand the data. Available tools:

- **scikit-learn**: Use `LinearRegression`, `Ridge`, `Lasso`, `DecisionTreeRegressor`, `DecisionTreeClassifier` for interpretable models. Use `feature_importances_` and `coef_` to understand feature effects.
- **imodels** (<https://github.com/csinva/imodels>): Use `from imodels import RuleFitRegressor, FIGSRegressor, HSTreeRegressor` for rule-based and tree-based interpretable models. These provide human-readable rules and feature importance.
- **statsmodels**: Use `statsmodels.api.OLS` for regression with p-values and confidence intervals.
- **scipy.stats**: Use for statistical tests (t-test, chi-square, correlation, ANOVA).

Focus on building interpretable models that help you understand the relationships in the data, not just black-box predictions. Use the model coefficients, rules, and feature importances to inform your conclusion.

## Important

- You MUST actually run the script, not just write it. The `conclusion.txt` file must exist when you are done.
- When asked if a relationship between two variables exists, use statistical significance tests.
- Relationships lacking significance should receive a "No" (low score), significant ones a "Yes" (high score).
- Available packages: numpy, pandas, scipy, statsmodels, sklearn, imodels, matplotlib, seaborn.
"""

# Stability prompt variant: identical to AGENTS_MD_STANDARD but pointing to
# the InterpretML library (https://github.com/interpretml/interpret) instead
# of imodels. Used to check whether the prompt-stability gain depends on the
# specific package being mentioned.
AGENTS_MD_STANDARD_INTERPRETML = """You are an expert data scientist. You MUST write and execute a Python script to analyze a dataset and answer a research question.

## Instructions

1. Read `info.json` to get the research question and dataset metadata.
2. Load the dataset from `{dataset_name}.csv`.
3. The `interpret` library at <https://github.com/interpretml/interpret> provides
   interpretable scikit-learn-compatible regressors and classifiers that
   you can use alongside scikit-learn.
4. Write a Python script called `analysis.py` that:
   - Loads and explores the data (summary statistics, distributions, correlations).
   - Builds interpretable models using scikit-learn and interpret to understand feature relationships.
   - Performs appropriate statistical tests (t-tests, ANOVA, regression, etc.).
   - Interprets the results in context of the research question.
5. **Execute the script** by running: `python3 analysis.py`
6. The script MUST write a file called `conclusion.txt` containing ONLY a JSON object:

```json
{{"response": <integer 0-100>, "explanation": "<your reasoning>"}}
```

Where `response` is a Likert scale score: 0 = strong "No", 100 = strong "Yes".

## Interpretability Tools

You should use interpretable models to understand the data. Available tools:

- **scikit-learn**: Use `LinearRegression`, `Ridge`, `Lasso`, `DecisionTreeRegressor`, `DecisionTreeClassifier` for interpretable models. Use `feature_importances_` and `coef_` to understand feature effects.
- **interpret** (<https://github.com/interpretml/interpret>): Use `from interpret.glassbox import ExplainableBoostingRegressor, ExplainableBoostingClassifier, DecisionListClassifier` for additive and rule-based interpretable models. These provide human-readable explanations and feature importance.
- **statsmodels**: Use `statsmodels.api.OLS` for regression with p-values and confidence intervals.
- **scipy.stats**: Use for statistical tests (t-test, chi-square, correlation, ANOVA).

Focus on building interpretable models that help you understand the relationships in the data, not just black-box predictions. Use the model coefficients, rules, and feature importances to inform your conclusion.

## Important

- You MUST actually run the script, not just write it. The `conclusion.txt` file must exist when you are done.
- When asked if a relationship between two variables exists, use statistical significance tests.
- Relationships lacking significance should receive a "No" (low score), significant ones a "Yes" (high score).
- Available packages: numpy, pandas, scipy, statsmodels, sklearn, imodels, interpret, matplotlib, seaborn.
"""

AGENTS_MD_CUSTOM_V2 = """You are an expert data scientist. You MUST write and execute a Python script to analyze a dataset and answer a research question.

## Instructions

1. Read `info.json` to get the research question and dataset metadata.
2. Load the dataset from `{dataset_name}.csv`.
3. **Carefully read `SKILL.md`** in this directory — it documents the `agentic_imodels`
   library (a set of evolved interpretable regressors) and the recommended
   analysis workflow. Follow those instructions closely; they are what the
   library was designed to support.
4. Write a Python script called `analysis.py` that:
   - Loads and explores the data (summary statistics, distributions, correlations).
   - Runs the classical statistical test appropriate to the research question
     (OLS / logistic / GLM) with relevant control variables, via `statsmodels`.
   - Heavily uses the `agentic_imodels` interpretable regressors as described in
     `SKILL.md` to characterize the **direction, magnitude, shape, and
     robustness** of each feature's effect. Fit at least two models from
     `agentic_imodels` and `print(model)` each so the interpretable form is
     captured in your reasoning.
   - Interprets the results in the context of the research question, weighing
     bivariate and controlled results, and null evidence from Lasso/hinge
     zeroing as well as *p*-values.
5. **Execute the script** by running: `python3 analysis.py`
6. The script MUST write a file called `conclusion.txt` containing ONLY a JSON object:

```json
{{"response": <integer 0-100>, "explanation": "<your reasoning>"}}
```

Where `response` is a Likert scale score: 0 = strong "No", 100 = strong "Yes".

## Interpretability Tools

You should heavily use interpretable models to understand the data. The
primary tool here is the **`agentic_imodels`** library — read `SKILL.md`
in this run directory for the full API, the model selection table, and the
recommended analysis workflow (framing the question, classical tests with
controls, shape/direction/importance from the interpretable models,
calibrated conclusion).

Supporting tools you can also use:

- **scikit-learn** / **imodels**: additional interpretable baselines
  (`LinearRegression`, `DecisionTreeRegressor`, `RuleFitRegressor`, etc.).
- **statsmodels**: `statsmodels.api.OLS` / `Logit` / `GLM` for classical
  regression with *p*-values and confidence intervals — use this for the
  formal statistical test with controls.
- **scipy.stats**: for bivariate statistical tests (t-test, chi-square,
  correlation, ANOVA).

Focus on building interpretable models that reveal **how** features
relate to the outcome, not just whether a relationship is significant.
Use the printed form of each `agentic_imodels` regressor (coefficients,
shape tables, rule sets, importance rankings) to inform your conclusion.

## Important

- You MUST actually run the script, not just write it. The `conclusion.txt` file must exist when you are done.
- When asked if a relationship between two variables exists, use statistical significance tests AND corroborate with the interpretable models.
- Relationships lacking significance AND ranked low / zeroed out by the interpretable models should receive a "No" (low score); significant ones that persist across models should receive a "Yes" (high score).
- Calibrate the Likert score to the strength of evidence, following the scoring guidelines in `SKILL.md`.
- Available packages: numpy, pandas, scipy, statsmodels, sklearn, imodels, agentic_imodels, interpret, matplotlib, seaborn.
"""


def get_installed_packages():
    """Get list of installed Python packages for reference."""
    try:
        result = subprocess.run(
            ["pip", "list", "--format=columns"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout
    except Exception:
        return "numpy\npandas\nscipy\nstatsmodels\nsklearn\nmatplotlib\nseaborn\nimodels\ninterpret\n"


def prepare_dataset(dataset_name: str, mode: str, output_dir: str):
    """Create a run directory for a single dataset."""
    src_dir = os.path.join(DATASETS_DIR, dataset_name)
    dst_dir = os.path.join(output_dir, dataset_name)

    if not os.path.isdir(src_dir):
        print(f"  SKIP: {dataset_name} (source not found at {src_dir})")
        return False

    os.makedirs(dst_dir, exist_ok=True)

    # Copy info.json
    shutil.copy2(os.path.join(src_dir, "info.json"), os.path.join(dst_dir, "info.json"))

    # Copy data CSV (may be data.csv or {dataset_name}.csv)
    src_csv = os.path.join(src_dir, "data.csv")
    if not os.path.exists(src_csv):
        src_csv = os.path.join(src_dir, f"{dataset_name}.csv")
    if os.path.exists(src_csv):
        shutil.copy2(src_csv, os.path.join(dst_dir, f"{dataset_name}.csv"))
    else:
        print(f"  WARN: {dataset_name} - no CSV found")
        return False

    # Write AGENTS.md based on mode
    templates = {
        "standard": AGENTS_MD_STANDARD,
        "interpretml": AGENTS_MD_STANDARD_INTERPRETML,
        "custom_v2": AGENTS_MD_CUSTOM_V2,
    }
    template = templates[mode]
    with open(os.path.join(dst_dir, "AGENTS.md"), "w") as f:
        f.write(template.format(dataset_name=dataset_name))

    # Copy the agentic-imodels package + SKILL.md for custom modes so the
    # agent can `from agentic_imodels import ...` locally.
    if mode == "custom_v2":
        pkg_src = os.path.join(AGENTIC_IMODELS_DIR, "agentic_imodels")
        pkg_dst = os.path.join(dst_dir, "agentic_imodels")
        if os.path.isdir(pkg_dst):
            shutil.rmtree(pkg_dst)
        shutil.copytree(
            pkg_src,
            pkg_dst,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        shutil.copy2(
            os.path.join(AGENTIC_IMODELS_DIR, "SKILL.md"),
            os.path.join(dst_dir, "SKILL.md"),
        )

    # Write packages.txt
    with open(os.path.join(dst_dir, "packages.txt"), "w") as f:
        f.write(get_installed_packages())

    print(f"  OK: {dataset_name} -> {dst_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Prepare Blade dataset run directories")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Single dataset to prepare (default: all 13)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["standard", "interpretml", "custom_v2"],
        default="standard",
        help="Tool mode: 'standard' (sklearn/imodels), 'interpretml' (sklearn/interpret), or 'custom_v2' (+ SKILL.md)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Explicit output directory name (overrides default outputs_{mode})",
    )
    args = parser.parse_args()

    if args.output_dir:
        output_dir = os.path.join(SCRIPT_DIR, args.output_dir)
    else:
        output_dir = os.path.join(SCRIPT_DIR, f"outputs_{args.mode}")
    datasets = [args.dataset] if args.dataset else DATASETS
    print(f"Preparing {len(datasets)} dataset(s) in {args.mode} mode...")

    success = 0
    for ds in datasets:
        if prepare_dataset(ds, args.mode, output_dir):
            success += 1

    print(f"\nDone: {success}/{len(datasets)} datasets prepared in {output_dir}/")


if __name__ == "__main__":
    main()
