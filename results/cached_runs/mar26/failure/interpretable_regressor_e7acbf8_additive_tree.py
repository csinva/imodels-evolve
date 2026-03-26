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


class AdditiveTreeRegressor(BaseEstimator, RegressorMixin):
    """
    Additive model: y = intercept + f_0(x0) + f_1(x1) + ...
    Each f_i is a small decision tree stump (depth 2) fit on residuals
    via cyclic boosting (backfitting). Produces a GAM-like partial-effect
    table for each feature, making it highly interpretable.
    """

    def __init__(self, max_depth=2, n_rounds=10, learning_rate=0.5,
                 min_samples_leaf=5, importance_threshold=0.01):
        self.max_depth = max_depth
        self.n_rounds = n_rounds
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.importance_threshold = importance_threshold

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape
        self.n_features_ = n_features

        self.intercept_ = float(np.mean(y))
        residual = y - self.intercept_

        # Per-feature tree list (list of lists)
        self.trees_ = [[] for _ in range(n_features)]

        for round_idx in range(self.n_rounds):
            for j in range(n_features):
                tree = DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=42 + round_idx * n_features + j,
                )
                tree.fit(X[:, j:j+1], residual)
                pred = tree.predict(X[:, j:j+1]) * self.learning_rate
                residual = residual - pred
                self.trees_[j].append(tree)

        # Compute feature importances to identify active features
        self.feature_importances_ = np.zeros(n_features)
        for j in range(n_features):
            vals = self._feature_partial(X[:, j:j+1], j)
            self.feature_importances_[j] = np.std(vals)

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

    def _feature_partial(self, x_col, j):
        """Sum predictions from all trees for feature j."""
        total = np.zeros(len(x_col))
        for tree in self.trees_[j]:
            total += tree.predict(x_col) * self.learning_rate
        return total

    def predict(self, X):
        check_is_fitted(self, "trees_")
        X = np.asarray(X, dtype=np.float64)
        pred = np.full(X.shape[0], self.intercept_)
        for j in range(self.n_features_):
            pred += self._feature_partial(X[:, j:j+1], j)
        return pred

    def __str__(self):
        check_is_fitted(self, "trees_")
        names = [f"x{i}" for i in range(self.n_features_)]
        lines = [
            "Additive Regression Model (GAM-style):",
            "  y = intercept + f0(x0) + f1(x1) + ...  (each feature's effect is INDEPENDENT)",
            f"  intercept: {self.intercept_:.4f}",
            "",
        ]

        # Show active features with partial effect tables
        active = self.active_features_
        if len(active) < self.n_features_:
            inactive = [names[j] for j in range(self.n_features_) if j not in active]
            lines.append(f"Features with negligible effect (excluded): {', '.join(inactive)}")
            lines.append("")

        lines.append("Feature partial effects (each feature's independent contribution):")

        for j in active:
            name = names[j]
            imp = self.feature_importances_[j]
            lines.append(f"\n  {name} (importance: {imp:.3f}):")

            # Generate a grid of representative values for this feature
            # Use percentiles from training data approximation via tree thresholds
            thresholds = set()
            for tree in self.trees_[j]:
                t = tree.tree_
                for node_id in range(t.node_count):
                    if t.children_left[node_id] != -1:  # not a leaf
                        thresholds.add(float(t.threshold[node_id]))

            if thresholds:
                sorted_thresh = sorted(thresholds)
                # Create evaluation points: below min, at thresholds, above max
                grid = [sorted_thresh[0] - 1.0]
                grid.extend(sorted_thresh)
                grid.append(sorted_thresh[-1] + 1.0)
                # Limit to ~7 points
                if len(grid) > 9:
                    indices = np.linspace(0, len(grid) - 1, 7, dtype=int)
                    grid = [grid[i] for i in indices]
            else:
                grid = np.linspace(-2, 2, 5).tolist()

            grid_arr = np.array(grid).reshape(-1, 1)
            effects = self._feature_partial(grid_arr, j)
            for xv, ev in zip(grid, effects):
                lines.append(f"    {name}={xv:+.2f}  →  effect={ev:+.4f}")

            # Describe shape
            if effects[-1] > effects[0] + 0.3:
                shape = "increasing"
            elif effects[-1] < effects[0] - 0.3:
                shape = "decreasing"
            elif max(effects) - min(effects) < 0.2:
                shape = "flat/negligible"
            else:
                shape = "non-monotone"
            lines.append(f"    (shape: {shape})")

        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
AdditiveTreeRegressor.__module__ = "interpretable_regressor"


# ---------------------------------------------------------------------------
# Evaluation (do not edit any of the evaluation functions, only the names and model descriptions below this line)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()

    model_defs = [("AdditiveTree", AdditiveTreeRegressor())]

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
    model_name = "AdditiveTree"
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
        "model_name":                         "AdditiveTree",
        "description":                        "Additive piecewise-linear model (GAM-style, backfitting with depth-2 trees)",
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

