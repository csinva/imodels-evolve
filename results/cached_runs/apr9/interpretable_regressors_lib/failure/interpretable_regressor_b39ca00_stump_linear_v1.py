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


class StumpLinearRegressor(BaseEstimator, RegressorMixin):
    """
    Two-stage interpretable regressor:
    1. Fit greedy additive stumps to discover important thresholds
    2. Create indicator features I(x_j > threshold) for top thresholds
    3. Fit Ridge on original features + indicators

    The result is a linear equation with threshold indicator terms:
      y = a*x0 + b*x1 + c*I(x0>0.5) + d*I(x1>-0.3) + ... + intercept

    This is trivially computable by the LLM (just like Ridge) but captures
    nonlinear threshold effects (unlike Ridge). Both display and prediction
    use the same equation, ensuring consistency.
    """

    def __init__(self, n_stump_rounds=100, stump_lr=0.1,
                 min_samples_leaf=5, max_indicators=10, max_input_features=20):
        self.n_stump_rounds = n_stump_rounds
        self.stump_lr = stump_lr
        self.min_samples_leaf = min_samples_leaf
        self.max_indicators = max_indicators
        self.max_input_features = max_input_features

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # Feature selection if too many features
        if n_features > self.max_input_features:
            corrs = np.array([abs(np.corrcoef(X[:, j], y)[0, 1])
                              if np.std(X[:, j]) > 1e-10 else 0
                              for j in range(n_features)])
            self.selected_ = np.sort(np.argsort(corrs)[-self.max_input_features:])
        else:
            self.selected_ = np.arange(n_features)

        X_sel = X[:, self.selected_]

        # Stage 1: Discover thresholds via greedy stumps
        intercept = float(np.mean(y))
        residuals = y - intercept
        stump_info = []  # (feature_idx, threshold, importance)

        for _ in range(self.n_stump_rounds):
            best = None
            best_reduction = -np.inf

            for j in range(X_sel.shape[1]):
                xj = X_sel[:, j]
                order = np.argsort(xj)
                xj_sorted = xj[order]
                r_sorted = residuals[order]
                cum_sum = np.cumsum(r_sorted)
                total_sum = cum_sum[-1]

                min_leaf = self.min_samples_leaf
                if n_samples < 2 * min_leaf:
                    continue
                lo = min_leaf - 1
                hi = n_samples - min_leaf - 1
                if hi < lo:
                    continue

                valid = np.where(xj_sorted[lo:hi + 1] != xj_sorted[lo + 1:hi + 2])[0] + lo
                if len(valid) == 0:
                    continue

                left_sum = cum_sum[valid]
                left_count = valid + 1
                right_sum = total_sum - left_sum
                right_count = n_samples - left_count

                reduction = left_sum ** 2 / left_count + right_sum ** 2 / right_count
                best_idx = np.argmax(reduction)

                if reduction[best_idx] > best_reduction:
                    best_reduction = reduction[best_idx]
                    split_pos = valid[best_idx]
                    threshold = (xj_sorted[split_pos] + xj_sorted[split_pos + 1]) / 2
                    left_mean = left_sum[best_idx] / left_count[best_idx]
                    right_mean = right_sum[best_idx] / right_count[best_idx]
                    best = (j, threshold, left_mean, right_mean, float(best_reduction))

            if best is None:
                break

            j, threshold, left_mean, right_mean, importance = best
            stump_info.append((j, threshold, importance))

            mask = X_sel[:, j] <= threshold
            residuals[mask] -= self.stump_lr * left_mean
            residuals[~mask] -= self.stump_lr * right_mean

        # Stage 2: Select top unique thresholds
        # Group by (feature, threshold) and sum importances
        threshold_importance = defaultdict(float)
        for j, t, imp in stump_info:
            threshold_importance[(j, t)] += imp

        # Sort by importance and take top max_indicators
        sorted_thresholds = sorted(threshold_importance.items(),
                                   key=lambda x: x[1], reverse=True)
        self.indicators_ = [(j, t) for (j, t), _ in sorted_thresholds[:self.max_indicators]]

        # Stage 3: Build augmented features and fit Ridge
        X_aug = self._augment(X_sel)
        self.aug_names_ = [f"x{self.selected_[j]}" for j in range(X_sel.shape[1])]
        for j, t in self.indicators_:
            orig_j = self.selected_[j]
            self.aug_names_.append(f"I(x{orig_j}>{t:.4f})")

        self.ridge_ = RidgeCV(cv=3)
        self.ridge_.fit(X_aug, y)

        return self

    def _augment(self, X_sel):
        """Add indicator features to selected X."""
        cols = [X_sel]
        for j, t in self.indicators_:
            cols.append((X_sel[:, j] > t).astype(np.float64).reshape(-1, 1))
        return np.hstack(cols)

    def predict(self, X):
        check_is_fitted(self, "ridge_")
        X = np.asarray(X, dtype=np.float64)
        X_sel = X[:, self.selected_]
        X_aug = self._augment(X_sel)
        return self.ridge_.predict(X_aug)

    def __str__(self):
        check_is_fitted(self, "ridge_")

        coefs = self.ridge_.coef_
        intercept = self.ridge_.intercept_
        names = self.aug_names_

        # Build equation
        equation = " + ".join(f"{c:.4f}*{n}" for c, n in zip(coefs, names))
        equation += f" + {intercept:.4f}"

        lines = [
            f"Ridge Regression (L2 regularization, α={self.ridge_.alpha_:.4g} chosen by CV):",
            f"  y = {equation}",
            "",
            "Coefficients:",
        ]

        for n, c in zip(names, coefs):
            lines.append(f"  {n}: {c:.4f}")
        lines.append(f"  intercept: {intercept:.4f}")

        # Note about indicator features
        if self.indicators_:
            lines.append("")
            lines.append("Note: I(condition) = 1 if condition is true, 0 otherwise.")

        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
StumpLinearRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "StumpLinear_v1"
model_description = "Ridge on original features + stump-derived indicator features I(x>threshold)"
model_defs = [(model_shorthand_name, StumpLinearRegressor())]


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
