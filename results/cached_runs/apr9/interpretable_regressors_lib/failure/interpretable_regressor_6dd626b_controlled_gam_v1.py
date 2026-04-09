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


class ControlledGAMRegressor(BaseEstimator, RegressorMixin):
    """
    Additive model with controlled complexity per feature.

    Uses cyclic gradient boosting with depth-1 stumps, but limits each
    feature to at most max_splits_per_feature unique split points.
    This keeps each feature's shape function naturally compact (≤ max_splits+1
    intervals) without any post-hoc quantization that would degrade predictions.

    The __str__ shows each feature as a compact lookup table that the LLM
    can directly compute predictions from.

    Model: y = intercept + f0(x0) + f1(x1) + ... + fp(xp)
    """

    def __init__(self, n_outer_rounds=80, learning_rate=0.15,
                 min_samples_leaf=5, max_splits_per_feature=4):
        self.n_outer_rounds = n_outer_rounds
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.max_splits_per_feature = max_splits_per_feature

    def _find_best_split(self, xj, residuals, n_samples, existing_thresholds=None):
        """Find best split, optionally restricted to existing thresholds."""
        order = np.argsort(xj)
        xj_sorted = xj[order]
        r_sorted = residuals[order]

        cum_sum = np.cumsum(r_sorted)
        total_sum = cum_sum[-1]

        min_leaf = self.min_samples_leaf
        if n_samples < 2 * min_leaf:
            return None
        lo = min_leaf - 1
        hi = n_samples - min_leaf - 1
        if hi < lo:
            return None

        valid = np.where(xj_sorted[lo:hi + 1] != xj_sorted[lo + 1:hi + 2])[0] + lo
        if len(valid) == 0:
            return None

        left_sum = cum_sum[valid]
        left_count = valid + 1
        right_sum = total_sum - left_sum
        right_count = n_samples - left_count

        reduction = left_sum ** 2 / left_count + right_sum ** 2 / right_count
        best_idx = np.argmax(reduction)

        split_pos = valid[best_idx]
        threshold = (xj_sorted[split_pos] + xj_sorted[split_pos + 1]) / 2
        left_mean = left_sum[best_idx] / left_count[best_idx]
        right_mean = right_sum[best_idx] / right_count[best_idx]

        return (threshold, left_mean * self.learning_rate,
                right_mean * self.learning_rate, float(reduction[best_idx]))

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        self.intercept_ = float(np.mean(y))

        # Track stumps and unique splits per feature
        feature_stumps = defaultdict(list)  # j -> [(threshold, left_val, right_val)]
        feature_splits = defaultdict(set)   # j -> set of unique thresholds
        residuals = y - self.intercept_

        for _ in range(self.n_outer_rounds):
            for j in range(n_features):
                # Check if this feature has room for more splits
                at_limit = len(feature_splits[j]) >= self.max_splits_per_feature

                result = self._find_best_split(X[:, j], residuals, n_samples)
                if result is None:
                    continue

                threshold, left_val, right_val, score = result

                # If at limit, only allow splits at existing thresholds
                if at_limit and threshold not in feature_splits[j]:
                    # Try to find a split at an existing threshold
                    # (reuse an existing split point to refine values)
                    # Just skip — the feature has enough splits
                    continue

                feature_stumps[j].append((threshold, left_val, right_val))
                feature_splits[j].add(threshold)

                mask = X[:, j] <= threshold
                residuals[mask] -= left_val
                residuals[~mask] -= right_val

        # Collapse into shape functions
        self.shape_functions_ = {}

        for j in range(n_features):
            stumps = feature_stumps.get(j, [])
            if not stumps:
                continue

            thresholds = sorted(set(t for t, _, _ in stumps))
            intervals = []
            for i in range(len(thresholds) + 1):
                if i == 0:
                    test_x = thresholds[0] - 1
                elif i == len(thresholds):
                    test_x = thresholds[-1] + 1
                else:
                    test_x = (thresholds[i - 1] + thresholds[i]) / 2
                val = sum(lv if test_x <= t else rv for t, lv, rv in stumps)
                intervals.append(val)

            self.shape_functions_[j] = (thresholds, intervals)

        # Feature importance
        self.feature_importances_ = np.zeros(n_features)
        for j, (thresholds, intervals) in self.shape_functions_.items():
            self.feature_importances_[j] = max(intervals) - min(intervals)

        return self

    def predict(self, X):
        check_is_fitted(self, "shape_functions_")
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        pred = np.full(n, self.intercept_)

        for j, (thresholds, intervals) in self.shape_functions_.items():
            xj = X[:, j]
            bins = np.digitize(xj, thresholds)
            pred += np.array([intervals[b] for b in bins])

        return pred

    def __str__(self):
        check_is_fitted(self, "shape_functions_")
        feature_names = [f"x{i}" for i in range(self.n_features_in_)]

        lines = [
            "Additive Model (each feature contributes independently):",
            f"  y = {self.intercept_:.4f}",
        ]

        # Sort by importance
        importance_order = np.argsort(self.feature_importances_)[::-1]

        for j in importance_order:
            if j not in self.shape_functions_:
                continue
            if self.feature_importances_[j] < 1e-6:
                continue

            thresholds, intervals = self.shape_functions_[j]
            name = feature_names[j]

            # Show as compact lookup
            parts = []
            for i, val in enumerate(intervals):
                if i == 0:
                    parts.append(f"    {name} <= {thresholds[0]:.4f}: {val:+.4f}")
                elif i == len(thresholds):
                    parts.append(f"    {name} >  {thresholds[-1]:.4f}: {val:+.4f}")
                else:
                    parts.append(f"    {thresholds[i-1]:.4f} < {name} <= {thresholds[i]:.4f}: {val:+.4f}")

            lines.append(f"  + f({name}):")
            lines.extend(parts)

        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
ControlledGAMRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "ControlledGAM_v1"
model_description = "Cyclic additive stumps with max 4 splits per feature (5 intervals), 80 outer rounds"
model_defs = [(model_shorthand_name, ControlledGAMRegressor())]


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
