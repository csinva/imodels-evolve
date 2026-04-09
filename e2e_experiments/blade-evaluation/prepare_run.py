"""Prepare run directories for each Blade dataset.

For each dataset, creates a subdirectory under outputs/ containing:
  - info.json   (task metadata with research question)
  - {dataset}.csv  (the data)
  - AGENTS.md   (instructions for Codex)

Usage:
    python prepare_run.py                    # prepare all 12 datasets
    python prepare_run.py --dataset soccer   # prepare one dataset
"""

import argparse
import json
import os
import shutil
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(
    SCRIPT_DIR, "..", "example-blade-repo", "blade", "blade_bench", "datasets"
)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")

# The 12 Blade benchmark datasets
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

AGENTS_MD_TEMPLATE = """You are an expert data scientist. You MUST write and execute a Python script to analyze a dataset and answer a research question.

## Instructions

1. Read `info.json` to get the research question and dataset metadata.
2. Load the dataset from `{dataset_name}.csv`.
3. Write a Python script called `analysis.py` that:
   - Loads and explores the data
   - Performs appropriate statistical tests and/or builds models
   - Interprets the results in context of the research question
4. **Execute the script** by running: `python3 analysis.py`
5. The script MUST write a file called `conclusion.txt` containing ONLY a JSON object:

```json
{{"response": <integer 0-100>, "explanation": "<your reasoning>"}}
```

Where `response` is a Likert scale score: 0 = strong "No", 100 = strong "Yes".

## Important

- You MUST actually run the script, not just write it. The `conclusion.txt` file must exist when you are done.
- When asked if a relationship between two variables exists, use statistical significance tests.
- Relationships lacking significance should receive a "No" (low score), significant ones a "Yes" (high score).
- Available packages: numpy, pandas, scipy, statsmodels, sklearn, matplotlib, seaborn.
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
        return "numpy\npandas\nscipy\nstatsmodels\nsklearn\nmatplotlib\nseaborn\n"


def prepare_dataset(dataset_name: str):
    """Create a run directory for a single dataset."""
    src_dir = os.path.join(DATASETS_DIR, dataset_name)
    dst_dir = os.path.join(OUTPUT_DIR, dataset_name)

    if not os.path.isdir(src_dir):
        print(f"  SKIP: {dataset_name} (source not found at {src_dir})")
        return False

    os.makedirs(dst_dir, exist_ok=True)

    # Copy info.json
    shutil.copy2(os.path.join(src_dir, "info.json"), os.path.join(dst_dir, "info.json"))

    # Copy data.csv as {dataset_name}.csv
    shutil.copy2(
        os.path.join(src_dir, "data.csv"),
        os.path.join(dst_dir, f"{dataset_name}.csv"),
    )

    # Write AGENTS.md
    with open(os.path.join(dst_dir, "AGENTS.md"), "w") as f:
        f.write(AGENTS_MD_TEMPLATE.format(dataset_name=dataset_name))

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
        help="Single dataset to prepare (default: all 12)",
    )
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else DATASETS
    print(f"Preparing {len(datasets)} dataset(s)...")

    success = 0
    for ds in datasets:
        if prepare_dataset(ds):
            success += 1

    print(f"\nDone: {success}/{len(datasets)} datasets prepared in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
