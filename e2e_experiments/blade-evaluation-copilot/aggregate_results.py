"""Aggregate results from multiple Codex runs and judge evaluations.

Loads results_*_run*_judge*.csv files and computes mean ± SE for each mode.

Usage:
    python aggregate_results.py
"""

import glob
import os
import re

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_all_results():
    """Load all results CSVs and tag with mode, run, judge."""
    pattern = os.path.join(SCRIPT_DIR, "judge_results", "results_*_run*_judge*.csv")
    files = sorted(glob.glob(pattern))

    all_rows = []
    for f in files:
        basename = os.path.basename(f).replace(".csv", "")
        # Parse: results_{mode}_run{R}_judge{J}
        m = re.match(r"results_(.+)_run(\d+)_judge(\d+)", basename)
        if not m:
            continue
        mode, run_id, judge_id = m.group(1), int(m.group(2)), int(m.group(3))

        df = pd.read_csv(f)
        df = df[df["status"] == "ok"].copy()
        df["mode"] = mode
        df["run_id"] = run_id
        df["judge_id"] = judge_id
        all_rows.append(df)

    if not all_rows:
        print("No results files found matching pattern results_*_run*_judge*.csv")
        return None

    return pd.concat(all_rows, ignore_index=True)


def main():
    df = load_all_results()
    if df is None:
        return

    dims = ["correctness", "completeness", "clarity"]

    print("=" * 80)
    print("AGGREGATED RESULTS (mean ± SE)")
    print("=" * 80)

    for mode in sorted(df["mode"].unique()):
        mdf = df[df["mode"] == mode]
        n_runs = mdf["run_id"].nunique()
        n_judges = mdf["judge_id"].nunique()
        n_obs = len(mdf) // 13  # number of (run, judge) pairs

        print(f"\n{'='*40}")
        print(f"Mode: {mode}")
        print(f"  Codex runs: {n_runs}, Judge repeats per run: {n_judges}")
        print(f"  Total evaluations per dataset: {n_obs}")
        print(f"{'='*40}")

        # Per-dataset breakdown: average across all (run, judge) pairs
        print(f"\n{'Dataset':20s} {'Correct':>12s} {'Complete':>12s} {'Clarity':>12s}")
        print("-" * 58)
        for dataset in sorted(mdf["dataset"].unique()):
            ddf = mdf[mdf["dataset"] == dataset]
            parts = []
            for dim in dims:
                vals = ddf[dim].dropna().values
                mean = np.mean(vals)
                se = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
                parts.append(f"{mean:.1f}±{se:.1f}")
            print(f"{dataset:20s} {parts[0]:>12s} {parts[1]:>12s} {parts[2]:>12s}")

        # Overall summary
        print("-" * 58)
        parts = []
        for dim in dims:
            vals = mdf.groupby(["run_id", "judge_id"])[dim].mean().values
            mean = np.mean(vals)
            se = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
            parts.append((mean, se))

        print(f"{'AVERAGE':20s} {parts[0][0]:>7.2f}±{parts[0][1]:.2f} {parts[1][0]:>7.2f}±{parts[1][1]:.2f} {parts[2][0]:>7.2f}±{parts[2][1]:.2f}")

        overall_means = [(p[0] + parts[1][0] + parts[2][0]) / 3 for p in [parts[0]]]
        overall_vals = mdf.groupby(["run_id", "judge_id"])[dims].mean().mean(axis=1).values
        overall_mean = np.mean(overall_vals)
        overall_se = np.std(overall_vals, ddof=1) / np.sqrt(len(overall_vals)) if len(overall_vals) > 1 else 0
        print(f"\nOverall: {overall_mean:.2f} ± {overall_se:.2f} / 10.00")

    # Side-by-side comparison
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    modes = sorted(df["mode"].unique())
    header = f"{'Dimension':15s}"
    for mode in modes:
        header += f" {mode:>20s}"
    print(header)
    print("-" * (15 + 21 * len(modes)))

    for dim in dims + ["overall"]:
        row = f"{dim:15s}"
        for mode in modes:
            mdf = df[df["mode"] == mode]
            if dim == "overall":
                vals = mdf.groupby(["run_id", "judge_id"])[dims].mean().mean(axis=1).values
            else:
                vals = mdf.groupby(["run_id", "judge_id"])[dim].mean().values
            mean = np.mean(vals)
            se = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
            row += f" {mean:>13.2f}±{se:.2f}"
        print(row)

    # Save summary
    summary_rows = []
    for mode in modes:
        mdf = df[df["mode"] == mode]
        row = {"mode": mode}
        for dim in dims:
            vals = mdf.groupby(["run_id", "judge_id"])[dim].mean().values
            row[f"{dim}_mean"] = round(np.mean(vals), 2)
            row[f"{dim}_se"] = round(np.std(vals, ddof=1) / np.sqrt(len(vals)), 2) if len(vals) > 1 else 0
        overall_vals = mdf.groupby(["run_id", "judge_id"])[dims].mean().mean(axis=1).values
        row["overall_mean"] = round(np.mean(overall_vals), 2)
        row["overall_se"] = round(np.std(overall_vals, ddof=1) / np.sqrt(len(overall_vals)), 2) if len(overall_vals) > 1 else 0
        row["n_codex_runs"] = mdf["run_id"].nunique()
        row["n_judge_repeats"] = mdf["judge_id"].nunique()
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(SCRIPT_DIR, "judge_results", "results_aggregated.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nAggregated summary saved to {summary_path}")


if __name__ == "__main__":
    main()
