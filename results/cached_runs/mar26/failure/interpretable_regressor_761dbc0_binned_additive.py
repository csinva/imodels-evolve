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

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LassoCV
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


class BinnedAdditiveRegressor(BaseEstimator, RegressorMixin):
    """
    Additive binned regression model (step-function GAM).

    y = intercept + f_0(x0) + f_1(x1) + ...

    Each f_j is a step function: the feature range is split into bins
    using quantile boundaries, and each bin has a learned effect value.
    Effects are learned via cyclic coordinate descent (backfitting).

    The __str__() shows a simple lookup table per feature — no arithmetic
    needed, just find which bin the input falls in.
    """

    def __init__(self, n_bins=5, n_rounds=15, importance_threshold=0.02):
        self.n_bins = n_bins
        self.n_rounds = n_rounds
        self.importance_threshold = importance_threshold

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape
        self.n_features_ = n_features

        # Compute bin boundaries per feature using quantiles
        self.boundaries_ = []
        for j in range(n_features):
            quantiles = np.linspace(0, 100, self.n_bins + 1)[1:-1]
            bounds = np.unique(np.percentile(X[:, j], quantiles))
            self.boundaries_.append(bounds)

        # Assign each sample to bins for each feature
        bin_indices = np.zeros((n_samples, n_features), dtype=int)
        for j in range(n_features):
            bin_indices[:, j] = np.digitize(X[:, j], self.boundaries_[j])

        # Initialize
        self.intercept_ = float(np.mean(y))
        residual = y - self.intercept_

        # Per-feature bin effects
        n_actual_bins = [len(self.boundaries_[j]) + 1 for j in range(n_features)]
        self.bin_effects_ = [np.zeros(n_actual_bins[j]) for j in range(n_features)]

        # Backfitting: cyclic coordinate descent
        for round_idx in range(self.n_rounds):
            for j in range(n_features):
                # Add back this feature's current contribution
                for b in range(n_actual_bins[j]):
                    mask = bin_indices[:, j] == b
                    if mask.any():
                        residual[mask] += self.bin_effects_[j][b]

                # Re-estimate bin effects from residual
                new_effects = np.zeros(n_actual_bins[j])
                for b in range(n_actual_bins[j]):
                    mask = bin_indices[:, j] == b
                    if mask.sum() > 0:
                        new_effects[b] = np.mean(residual[mask])

                self.bin_effects_[j] = new_effects

                # Subtract updated contribution
                for b in range(n_actual_bins[j]):
                    mask = bin_indices[:, j] == b
                    if mask.any():
                        residual[mask] -= new_effects[b]

        # Compute feature importance (range of bin effects)
        self.feature_importances_ = np.array([
            np.max(self.bin_effects_[j]) - np.min(self.bin_effects_[j])
            for j in range(n_features)
        ])
        total_imp = self.feature_importances_.sum()
        if total_imp > 0:
            self.feature_importances_ /= total_imp

        # Identify active features
        self.active_features_ = np.where(
            self.feature_importances_ > self.importance_threshold
        )[0]
        if len(self.active_features_) == 0:
            self.active_features_ = np.array([np.argmax(self.feature_importances_)])

        return self

    def _get_bin(self, x_val, j):
        """Return bin index for a single value of feature j."""
        return int(np.digitize(x_val, self.boundaries_[j]))

    def predict(self, X):
        check_is_fitted(self, "bin_effects_")
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        pred = np.full(n_samples, self.intercept_)
        for j in range(self.n_features_):
            bins = np.digitize(X[:, j], self.boundaries_[j])
            for b in range(len(self.bin_effects_[j])):
                mask = bins == b
                pred[mask] += self.bin_effects_[j][b]
        return pred

    def __str__(self):
        check_is_fitted(self, "bin_effects_")
        names = [f"x{i}" for i in range(self.n_features_)]

        lines = [
            "Additive Regression Model (step-function GAM):",
            "  y = intercept + f(x0) + f(x1) + ...  (each feature's effect is INDEPENDENT)",
            f"  intercept = {self.intercept_:.4f}",
            "",
        ]

        active = self.active_features_
        if len(active) < self.n_features_:
            inactive = [names[j] for j in range(self.n_features_) if j not in active]
            lines.append(f"Features with negligible effect (excluded): {', '.join(inactive)}")
            lines.append("")

        lines.append("LOOKUP TABLE (find which bin the input falls in, read the effect):")

        for j in active:
            name = names[j]
            bounds = self.boundaries_[j]
            effects = self.bin_effects_[j]
            imp = self.feature_importances_[j]

            lines.append(f"\n  {name} (importance: {imp:.3f}):")

            # First bin: x < first boundary
            if len(bounds) > 0:
                lines.append(f"    if {name} < {bounds[0]:.2f}: effect = {effects[0]:+.4f}")

                # Middle bins
                for b in range(len(bounds) - 1):
                    lines.append(f"    if {bounds[b]:.2f} <= {name} < {bounds[b+1]:.2f}: effect = {effects[b+1]:+.4f}")

                # Last bin
                lines.append(f"    if {name} >= {bounds[-1]:.2f}: effect = {effects[len(bounds)]:+.4f}")
            else:
                lines.append(f"    effect = {effects[0]:+.4f} (constant)")

        lines.append("")
        lines.append("HOW TO PREDICT: For each feature, find which bin the value falls in,")
        lines.append("read the effect, then sum: y = intercept + effect_x0 + effect_x1 + ...")

        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
BinnedAdditiveRegressor.__module__ = "interpretable_regressor"


# ---------------------------------------------------------------------------
# Evaluation (do not edit any of the evaluation functions, only the names and model descriptions below this line)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()

    model_defs = [("BinnedAdditive", BinnedAdditiveRegressor())]

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
    model_name = "BinnedAdditive"
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
    from collections import defaultdict
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
    mean_rank = avg_rank.get(model_name, float("nan"))

    upsert_overall_results([{
        "commit":                             git_hash,
        "mean_rank":                          f"{mean_rank:.2f}" if not np.isnan(mean_rank) else "",
        "frac_interpretability_tests_passed": f"{n_passed / total:.4f}" if total > 0 else "",
        "status":                             "",
        "model_name":                         "BinnedAdditive",
        "description":                        "Binned additive model (step-function GAM with backfitting)",
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

