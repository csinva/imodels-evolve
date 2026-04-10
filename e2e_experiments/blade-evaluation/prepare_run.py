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
DATASETS_DIR = _BLADE_DIR if os.path.isdir(_BLADE_DIR) else os.path.join(SCRIPT_DIR, "outputs")

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
3. Write a Python script called `analysis.py` that:
   - Loads and explores the data (summary statistics, distributions, correlations)
   - Builds interpretable models using scikit-learn and imodels to understand feature relationships
   - Performs appropriate statistical tests (t-tests, ANOVA, regression, etc.)
   - Interprets the results in context of the research question
4. **Execute the script** by running: `python3 analysis.py`
5. The script MUST write a file called `conclusion.txt` containing ONLY a JSON object:

```json
{{"response": <integer 0-100>, "explanation": "<your reasoning>"}}
```

Where `response` is a Likert scale score: 0 = strong "No", 100 = strong "Yes".

## Interpretability Tools

You should heavily use interpretable models to understand the data. Available tools:

- **scikit-learn**: Use `LinearRegression`, `Ridge`, `Lasso`, `DecisionTreeRegressor`, `DecisionTreeClassifier` for interpretable models. Use `feature_importances_` and `coef_` to understand feature effects.
- **imodels**: Use `from imodels import RuleFitRegressor, FIGSRegressor, HSTreeRegressor` for rule-based and tree-based interpretable models. These provide human-readable rules and feature importance.
- **statsmodels**: Use `statsmodels.api.OLS` for regression with p-values and confidence intervals.
- **scipy.stats**: Use for statistical tests (t-test, chi-square, correlation, ANOVA).

Focus on building interpretable models that help you understand the relationships in the data, not just black-box predictions. Use the model coefficients, rules, and feature importances to inform your conclusion.

## Important

- You MUST actually run the script, not just write it. The `conclusion.txt` file must exist when you are done.
- When asked if a relationship between two variables exists, use statistical significance tests.
- Relationships lacking significance should receive a "No" (low score), significant ones a "Yes" (high score).
- Available packages: numpy, pandas, scipy, statsmodels, sklearn, imodels, matplotlib, seaborn.
"""

AGENTS_MD_CUSTOM = """You are an expert data scientist. You MUST write and execute a Python script to analyze a dataset and answer a research question.

## Instructions

1. Read `info.json` to get the research question and dataset metadata.
2. Load the dataset from `{dataset_name}.csv`.
3. Write a Python script called `analysis.py` that:
   - Loads and explores the data (summary statistics, distributions, correlations)
   - Builds interpretable models using the provided custom interpretability tools AND standard tools
   - Performs appropriate statistical tests (t-tests, ANOVA, regression, etc.)
   - Interprets the results in context of the research question
4. **Execute the script** by running: `python3 analysis.py`
5. The script MUST write a file called `conclusion.txt` containing ONLY a JSON object:

```json
{{"response": <integer 0-100>, "explanation": "<your reasoning>"}}
```

Where `response` is a Likert scale score: 0 = strong "No", 100 = strong "Yes".

## Custom Interpretability Tools (IMPORTANT: use these!)

A file called `interp_models.py` is provided in this directory. It contains custom
scikit-learn compatible interpretable regressors that produce human-readable model
descriptions. **You should heavily use these tools** in your analysis.

### Available models:

1. **SmartAdditiveRegressor**: Greedy additive model that learns per-feature shape
   functions. After fitting, `str(model)` shows:
   - Linear coefficients for features with linear effects
   - Piecewise-constant lookup tables for features with nonlinear effects
   - Which features are excluded (zero effect)

2. **HingeEBMRegressor**: Two-stage model using piecewise-linear hinge functions
   with Lasso sparsity. After fitting, `str(model)` shows:
   - Clean linear equation with only the active terms
   - Feature coefficients sorted by importance
   - Which features have zero coefficients

### Example usage:

```python
from interp_models import SmartAdditiveRegressor, HingeEBMRegressor
import pandas as pd
import numpy as np

df = pd.read_csv("{dataset_name}.csv")
# Prepare X (features) and y (target variable relevant to the research question)
# ... select appropriate columns based on the research question ...

model = SmartAdditiveRegressor(n_rounds=200)
model.fit(X, y)
print("Model interpretation:")
print(model)  # Shows human-readable equation and feature effects

# Also try HingeEBMRegressor for comparison
model2 = HingeEBMRegressor(n_knots=3)
model2.fit(X, y)
print("HingeEBM interpretation:")
print(model2)

# Use the interpretable output to understand which features matter
# and how they relate to the research question
```

### Why use these tools:

- `str(model)` gives you a complete, human-readable explanation of the learned model
- You can see exactly which features matter and their directional effects
- Nonlinear effects are shown as piecewise lookup tables (thresholds and values)
- Combine with standard statistical tests for a comprehensive analysis

## Standard Tools (also available)

- **scikit-learn**: `LinearRegression`, `Ridge`, `Lasso`, `DecisionTreeRegressor`
- **imodels**: `RuleFitRegressor`, `FIGSRegressor`, `HSTreeRegressor`
- **statsmodels**: `statsmodels.api.OLS` for regression with p-values
- **scipy.stats**: Statistical tests (t-test, chi-square, correlation, ANOVA)

## Important

- You MUST actually run the script, not just write it. The `conclusion.txt` file must exist when you are done.
- When asked if a relationship between two variables exists, use statistical significance tests.
- Relationships lacking significance should receive a "No" (low score), significant ones a "Yes" (high score).
- Available packages: numpy, pandas, scipy, statsmodels, sklearn, imodels, interpret, matplotlib, seaborn.
- **Use the custom interpretability tools from `interp_models.py`** — they provide deeper insight into feature relationships than standard models.
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
    template = AGENTS_MD_CUSTOM if mode == "custom" else AGENTS_MD_STANDARD
    with open(os.path.join(dst_dir, "AGENTS.md"), "w") as f:
        f.write(template.format(dataset_name=dataset_name))

    # Copy interp_models.py for custom mode
    if mode == "custom":
        shutil.copy2(
            os.path.join(SCRIPT_DIR, "interp_models.py"),
            os.path.join(dst_dir, "interp_models.py"),
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
        choices=["standard", "custom"],
        default="standard",
        help="Tool mode: 'standard' (sklearn/imodels) or 'custom' (+ interp_models.py)",
    )
    args = parser.parse_args()

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
