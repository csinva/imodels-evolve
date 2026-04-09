"""
Interpretable regressor autoresearch script.
Defines a scikit-learn compatible interpretable regressor and evaluates it
on interpretability tests and TabArena regression datasets (same suite used
for baselines in run_baselines.py).

Usage: uv run model.py
"""

import csv
import os
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class PerFeatureTreeGAMRegressor(BaseEstimator, RegressorMixin):
    """
    Per-feature Tree GAM: fits one shallow decision tree per feature on
    the residuals from a Ridge baseline, in a boosted fashion.

    Architecture:
    1. Fit Ridge for the linear backbone (handles linear effects well)
    2. For each feature, fit a shallow tree (depth 2) on residuals using
       only that feature. Cycle through features multiple times.

    The final prediction is: y = ridge(X) + sum of per-feature trees

    Display: Shows Ridge equation + each per-feature tree. This format
    is exactly what TreeGAM and HSTree use (76% interp for HSTree_mini).
    """

    def __init__(self, n_cycles=5, tree_max_depth=2, tree_lr=0.3,
                 min_samples_leaf=10):
        self.n_cycles = n_cycles
        self.tree_max_depth = tree_max_depth
        self.tree_lr = tree_lr
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # Step 1: Ridge backbone
        self.ridge_ = RidgeCV(cv=3)
        self.ridge_.fit(X, y)
        residuals = y - self.ridge_.predict(X)

        # Step 2: Per-feature trees (cycled)
        self.feature_trees_ = defaultdict(list)  # j -> list of trees

        for cycle in range(self.n_cycles):
            for j in range(n_features):
                Xj = X[:, j:j+1]  # single feature
                tree = DecisionTreeRegressor(
                    max_depth=self.tree_max_depth,
                    min_samples_leaf=max(self.min_samples_leaf, n_samples // 20),
                    random_state=42 + cycle * n_features + j,
                )
                tree.fit(Xj, residuals)
                self.feature_trees_[j].append(tree)
                residuals -= self.tree_lr * tree.predict(Xj)

        return self

    def predict(self, X):
        check_is_fitted(self, "ridge_")
        X = np.asarray(X, dtype=np.float64)
        pred = self.ridge_.predict(X)

        for j, trees in self.feature_trees_.items():
            Xj = X[:, j:j+1]
            for tree in trees:
                pred += self.tree_lr * tree.predict(Xj)

        return pred

    def __str__(self):
        check_is_fitted(self, "ridge_")
        feature_names = [f"x{i}" for i in range(self.n_features_in_)]

        # Ridge part
        coefs = self.ridge_.coef_
        intercept = self.ridge_.intercept_
        equation = " + ".join(f"{c:.4f}*{n}" for c, n in zip(coefs, feature_names))
        equation += f" + {intercept:.4f}"

        lines = [
            "TreeGAM Regressor (additive: Ridge baseline + one tree per feature):",
            f"  y = ridge_baseline + tree_0(x0) + tree_1(x1) + ...",
            "",
            f"Ridge baseline (α={self.ridge_.alpha_:.4g}):",
            f"  y_base = {equation}",
            "",
            "Coefficients:",
        ]
        for n, c in zip(feature_names, coefs):
            lines.append(f"  {n}: {c:.4f}")
        lines.append(f"  intercept: {intercept:.4f}")

        # Per-feature trees (show combined effect)
        lines.append("")
        lines.append("Per-feature trees (each feature's independent nonlinear correction):")

        for j in range(self.n_features_in_):
            trees = self.feature_trees_.get(j, [])
            if not trees:
                continue

            name = feature_names[j]
            # Show the first tree (most important) for compactness
            lines.append(f"\n  Tree for {name} (feature {j}, weight={self.tree_lr}):")
            tree_text = export_text(trees[0], feature_names=[name], max_depth=3)
            for line in tree_text.strip().split("\n"):
                lines.append("    " + line)

        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
PerFeatureTreeGAMRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "PerFeatureTreeGAM_v1"
model_description = "Ridge + per-feature depth-2 trees (5 cycles, lr=0.3) — TreeGAM-style"
model_defs = [(model_shorthand_name, PerFeatureTreeGAMRegressor())]


# ---------------------------------------------------------------------------
# Evaluation (do not edit anything below this line)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()

    # Interpretability tests
    interp_results = run_all_interp_tests(model_defs)
    n_passed = sum(r["passed"] for r in interp_results)
    total = len(interp_results)

    # prediction performance (RMSE)
    dataset_rmses = evaluate_all_regressors(model_defs)

    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        git_hash = ""

    # --- Upsert interpretability_results.csv ---
    model_name = model_defs[0][0]
    interp_csv = os.path.join(RESULTS_DIR, "interpretability_results.csv")
    interp_fields = ["model", "test", "suite", "passed", "ground_truth", "response"]

    def _suite(test_name):
        if test_name.startswith("insight_"): return "insight"
        if test_name.startswith("hard_"):    return "hard"
        return "standard"

    # Load existing rows, dropping old rows for this model
    existing_interp = []
    if os.path.exists(interp_csv):
        with open(interp_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("model") != model_name:
                    existing_interp.append(row)

    new_interp = [{
        "model": r["model"],
        "test": r["test"],
        "suite": _suite(r["test"]),
        "passed": r["passed"],
        "ground_truth": r.get("ground_truth", ""),
        "response": r.get("response", ""),
    } for r in interp_results]

    with open(interp_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=interp_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing_interp + new_interp)
    print(f"Interpretability results saved → {interp_csv}")

    # --- Upsert performance_results.csv and recompute ranks ---
    perf_csv = os.path.join(RESULTS_DIR, "performance_results.csv")
    perf_fields = ["dataset", "model", "rmse", "rank"]

    # Load existing rows, dropping old rows for this model
    existing_perf = []
    if os.path.exists(perf_csv):
        with open(perf_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("model") != model_name:
                    existing_perf.append(row)

    # Add new rows (without rank for now)
    for ds_name, model_rmses in dataset_rmses.items():
        rmse_val = model_rmses.get(model_name, float("nan"))
        existing_perf.append({
            "dataset": ds_name,
            "model": model_name,
            "rmse": "" if np.isnan(rmse_val) else f"{rmse_val:.6f}",
            "rank": "",
        })

    # Recompute ranks per dataset
    by_dataset = defaultdict(list)
    for row in existing_perf:
        by_dataset[row["dataset"]].append(row)

    for ds_name, rows in by_dataset.items():
        valid = [(r, float(r["rmse"])) for r in rows if r["rmse"] not in ("", None)]
        valid.sort(key=lambda x: x[1])
        for rank_idx, (r, _) in enumerate(valid, 1):
            r["rank"] = rank_idx
        # Leave rank empty for rows with no RMSE
        for r in rows:
            if r["rmse"] in ("", None):
                r["rank"] = ""

    with open(perf_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=perf_fields)
        writer.writeheader()
        for ds_name in by_dataset:
            for row in by_dataset[ds_name]:
                writer.writerow(row)
    print(f"Performance results saved → {perf_csv}")

    # --- Compute mean_rank from the updated performance_results.csv ---
    # Build dataset_rmses dict with all models from the CSV for ranking
    all_dataset_rmses = defaultdict(dict)
    for row in existing_perf:
        rmse_str = row.get("rmse", "")
        if rmse_str not in ("", None):
            all_dataset_rmses[row["dataset"]][row["model"]] = float(rmse_str)
        else:
            all_dataset_rmses[row["dataset"]][row["model"]] = float("nan")
    avg_rank, _ = compute_rank_scores(dict(all_dataset_rmses))
    mean_rank = avg_rank.get(model_shorthand_name, float("nan"))

    upsert_overall_results([{
        "commit":                             git_hash,
        "mean_rank":                          f"{mean_rank:.2f}" if not np.isnan(mean_rank) else "nan",
        "frac_interpretability_tests_passed": f"{n_passed / total:.4f}" if total > 0 else "nan",
        "status":                             "",
        "model_name":                         model_shorthand_name,
        "description":                        model_description,
    }], RESULTS_DIR)

    # --- Plot ---
    overall_csv = os.path.join(RESULTS_DIR, "overall_results.csv")
    plot_interp_vs_performance(
        overall_csv,
        os.path.join(RESULTS_DIR, "interpretability_vs_performance.png"),
    )

    print()
    print("---")
    print(f"tests_passed:  {n_passed}/{total}" + (f" ({n_passed/total:.2%})" if total > 0 else ""))
    print(f"mean_rank:     {mean_rank:.2f}" if not np.isnan(mean_rank) else "mean_rank:     nan")
    print(f"total_seconds: {time.time() - t0:.1f}s")
