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


class CompactAdditiveRegressor(BaseEstimator, RegressorMixin):
    """
    Additive model with compact per-feature shape functions.

    Uses gradient boosting with depth-1 stumps, then collapses all stumps
    into per-feature piecewise-constant functions. Finally, each shape function
    is quantized to at most max_bins bins to keep the string representation
    concise and LLM-readable.

    Model: y = intercept + f0(x0) + f1(x1) + ... + fp(xp)
    """

    def __init__(self, n_rounds=150, learning_rate=0.1, min_samples_leaf=5, max_bins=5, max_features_shown=10):
        self.n_rounds = n_rounds
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.max_bins = max_bins
        self.max_features_shown = max_features_shown

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        self.intercept_ = np.mean(y)

        # Collect stumps: (feature_idx, threshold, left_val, right_val)
        stumps = []
        residuals = y - self.intercept_

        for _ in range(self.n_rounds):
            best_stump = None
            best_reduction = -np.inf

            for j in range(n_features):
                xj = X[:, j]
                order = np.argsort(xj)
                xj_sorted = xj[order]
                r_sorted = residuals[order]

                cum_sum = np.cumsum(r_sorted)
                total_sum = cum_sum[-1]
                total_sse = np.sum(r_sorted ** 2)

                min_leaf = self.min_samples_leaf
                if n_samples < 2 * min_leaf:
                    continue

                lo = min_leaf - 1
                hi = n_samples - min_leaf - 1
                if hi < lo:
                    continue

                # Find valid split points where values change
                valid = np.where(xj_sorted[lo:hi + 1] != xj_sorted[lo + 1:hi + 2])[0] + lo

                if len(valid) == 0:
                    continue

                left_sum = cum_sum[valid]
                left_count = valid + 1
                right_sum = total_sum - left_sum
                right_count = n_samples - left_count

                # SSE reduction = left_sum^2/left_count + right_sum^2/right_count
                reduction = left_sum ** 2 / left_count + right_sum ** 2 / right_count

                best_idx = np.argmax(reduction)
                if reduction[best_idx] > best_reduction:
                    best_reduction = reduction[best_idx]
                    split_pos = valid[best_idx]
                    threshold = (xj_sorted[split_pos] + xj_sorted[split_pos + 1]) / 2
                    left_mean = left_sum[best_idx] / left_count[best_idx]
                    right_mean = right_sum[best_idx] / right_count[best_idx]
                    best_stump = (j, threshold,
                                  left_mean * self.learning_rate,
                                  right_mean * self.learning_rate)

            if best_stump is None:
                break

            j, threshold, left_val, right_val = best_stump
            stumps.append(best_stump)
            mask = X[:, j] <= threshold
            residuals[mask] -= left_val
            residuals[~mask] -= right_val

        # Collapse stumps into per-feature shape functions
        self.shape_functions_ = {}
        for j in range(n_features):
            feature_stumps = [(t, lv, rv) for fj, t, lv, rv in stumps if fj == j]
            if not feature_stumps:
                continue

            thresholds = sorted(set(t for t, _, _ in feature_stumps))

            # Compute value for each interval
            intervals = []
            for i in range(len(thresholds) + 1):
                if i == 0:
                    test_x = thresholds[0] - 1
                elif i == len(thresholds):
                    test_x = thresholds[-1] + 1
                else:
                    test_x = (thresholds[i - 1] + thresholds[i]) / 2

                val = sum(lv if test_x <= t else rv for t, lv, rv in feature_stumps)
                intervals.append(val)

            # Quantize to max_bins bins if needed
            if len(thresholds) > self.max_bins:
                thresholds, intervals = self._quantize(thresholds, intervals, self.max_bins)

            self.shape_functions_[j] = (thresholds, intervals)

        # Compute feature importance for display ordering
        self.feature_importances_ = np.zeros(n_features)
        for j, (thresholds, intervals) in self.shape_functions_.items():
            self.feature_importances_[j] = max(abs(v) for v in intervals) - min(abs(v) for v in intervals)

        return self

    def _quantize(self, thresholds, intervals, max_bins):
        """Merge adjacent intervals with smallest value difference."""
        thresholds = list(thresholds)
        intervals = list(intervals)

        while len(thresholds) > max_bins:
            # Find pair of adjacent intervals with smallest difference
            min_diff = np.inf
            min_idx = 0
            for i in range(len(intervals) - 1):
                diff = abs(intervals[i] - intervals[i + 1])
                if diff < min_diff:
                    min_diff = diff
                    min_idx = i

            # Merge: remove the threshold between intervals[min_idx] and intervals[min_idx+1]
            # Use weighted average (approximate: equal weight)
            merged_val = (intervals[min_idx] + intervals[min_idx + 1]) / 2
            intervals[min_idx] = merged_val
            intervals.pop(min_idx + 1)
            thresholds.pop(min_idx)

        return thresholds, intervals

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

        lines = [f"Additive Model: y = {self.intercept_:.4f}", ""]

        # Sort by importance, show top features
        importance_order = np.argsort(self.feature_importances_)[::-1]
        shown = 0
        for j in importance_order:
            if j not in self.shape_functions_:
                continue
            if shown >= self.max_features_shown:
                remaining = sum(1 for jj in importance_order[shown:] if jj in self.shape_functions_)
                if remaining > 0:
                    lines.append(f"(+ {remaining} more features with small effects)")
                break

            thresholds, intervals = self.shape_functions_[j]
            name = feature_names[j]

            # Show as piecewise function with contribution to y
            parts = []
            for i, val in enumerate(intervals):
                if i == 0:
                    parts.append(f"  if {name} <= {thresholds[0]:.4f}: {val:+.4f}")
                elif i == len(thresholds):
                    parts.append(f"  if {name} >  {thresholds[-1]:.4f}: {val:+.4f}")
                else:
                    parts.append(f"  if {thresholds[i-1]:.4f} < {name} <= {thresholds[i]:.4f}: {val:+.4f}")

            lines.append(f"+ f({name}):")
            lines.extend(parts)
            shown += 1

        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
CompactAdditiveRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "CompactAdditive_v1"
model_description = "Boosted stumps with per-feature shape functions quantized to 5 bins max, top 10 features shown"
model_defs = [(model_shorthand_name, CompactAdditiveRegressor())]


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
