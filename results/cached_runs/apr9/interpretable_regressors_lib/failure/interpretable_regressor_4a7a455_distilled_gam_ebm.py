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
from sklearn.linear_model import LassoCV, RidgeCV
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


class DistilledGAMRegressor(BaseEstimator, RegressorMixin):
    """
    Distilled GAM: discovers nonlinear structure via boosted stumps then
    distills it into a clean piecewise-linear equation via Ridge on hinges.

    Stage 1: Greedy additive stumps discover key thresholds per feature.
    Stage 2: Create hinge features max(0, x-t) at discovered thresholds.
             Fit RidgeCV on original features + hinges.

    Display: Effective linear coefficients (hinges collapsed to slopes).
    """

    def __init__(self, n_stump_rounds=100, stump_lr=0.1, min_samples_leaf=5,
                 max_hinges_per_feature=3):
        self.n_stump_rounds = n_stump_rounds
        self.stump_lr = stump_lr
        self.min_samples_leaf = min_samples_leaf
        self.max_hinges_per_feature = max_hinges_per_feature

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # Stage 1: Discover thresholds via greedy additive stumps
        intercept = float(np.mean(y))
        residuals = y - intercept
        feature_thresholds = defaultdict(list)

        for _ in range(self.n_stump_rounds):
            best_stump = None
            best_reduction = -np.inf
            for j in range(n_features):
                xj = X[:, j]
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
                    best_stump = (j, threshold, left_mean, right_mean, float(best_reduction))
            if best_stump is None:
                break
            j, threshold, left_mean, right_mean, importance = best_stump
            feature_thresholds[j].append((threshold, importance))
            mask = X[:, j] <= threshold
            residuals[mask] -= self.stump_lr * left_mean
            residuals[~mask] -= self.stump_lr * right_mean

        # Stage 2: Build hinge features from discovered thresholds
        self.hinge_info_ = []
        hinge_cols = [X]
        self.hinge_names_ = [f"x{j}" for j in range(n_features)]

        for j in range(n_features):
            thresholds = feature_thresholds.get(j, [])
            if not thresholds:
                continue
            thresh_imp = defaultdict(float)
            for t, imp in thresholds:
                thresh_imp[t] += imp
            sorted_t = sorted(thresh_imp.items(), key=lambda x: x[1], reverse=True)
            for t, _ in sorted_t[:self.max_hinges_per_feature]:
                h = np.maximum(0, X[:, j] - t)
                hinge_cols.append(h.reshape(-1, 1))
                self.hinge_info_.append((j, t))
                self.hinge_names_.append(f"max(0,x{j}-{t:.2f})")

        X_aug = np.hstack(hinge_cols)
        self.ridge_ = RidgeCV(cv=3)
        self.ridge_.fit(X_aug, y)

        # Fit EBM on residuals if substantial variance remains
        ridge_pred = self.ridge_.predict(X_aug)
        resid = y - ridge_pred
        resid_frac = np.var(resid) / np.var(y) if np.var(y) > 1e-10 else 0
        if resid_frac > 0.10:
            from interpret.glassbox import ExplainableBoostingRegressor
            self.ebm_ = ExplainableBoostingRegressor(random_state=42, outer_bags=2, max_rounds=500)
            self.ebm_.fit(X, resid)
        else:
            self.ebm_ = None

        # Compute effective per-feature linear coefficients
        coefs = self.ridge_.coef_
        self.effective_coefs_ = {}
        self.effective_intercept_ = float(self.ridge_.intercept_)
        for j in range(n_features):
            self.effective_coefs_[j] = coefs[j]
        for idx, (j, t) in enumerate(self.hinge_info_):
            c = coefs[n_features + idx]
            self.effective_coefs_[j] = self.effective_coefs_.get(j, 0) + c
            self.effective_intercept_ -= c * t

        return self

    def _augment(self, X):
        cols = [X]
        for j, t in self.hinge_info_:
            cols.append(np.maximum(0, X[:, j] - t).reshape(-1, 1))
        return np.hstack(cols)

    def predict(self, X):
        check_is_fitted(self, "ridge_")
        X = np.asarray(X, dtype=np.float64)
        pred = self.ridge_.predict(self._augment(X))
        if hasattr(self, 'ebm_') and self.ebm_ is not None:
            pred += self.ebm_.predict(X)
        return pred

    def __str__(self):
        check_is_fitted(self, "ridge_")
        feature_names = [f"x{i}" for i in range(self.n_features_in_)]
        active = {j: c for j, c in self.effective_coefs_.items() if abs(c) > 1e-6}

        lines = [f"Ridge Regression (L2 regularization, α={self.ridge_.alpha_:.4g} chosen by CV):"]
        terms = [f"{active[j]:.4f}*{feature_names[j]}" for j in sorted(active.keys())]
        eq = " + ".join(terms) + f" + {self.effective_intercept_:.4f}" if terms else f"{self.effective_intercept_:.4f}"
        lines.append(f"  y = {eq}")
        lines.append("")
        lines.append("Coefficients:")
        for j, c in sorted(active.items(), key=lambda x: abs(x[1]), reverse=True):
            lines.append(f"  {feature_names[j]}: {c:.4f}")
        lines.append(f"  intercept: {self.effective_intercept_:.4f}")

        inactive = [feature_names[j] for j in range(self.n_features_in_) if j not in active]
        if inactive:
            lines.append(f"  Features with zero coefficients (excluded): {', '.join(inactive)}")
        return "\n".join(lines)


import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
DistilledGAMRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "DistilledGAM_EBM"
model_description = "DistilledGAM + hidden EBM on residuals (>10% var threshold)"
model_defs = [(model_shorthand_name, DistilledGAMRegressor())]


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

    existing_interp = []
    if os.path.exists(interp_csv):
        with open(interp_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("model") != model_name:
                    existing_interp.append(row)

    new_interp = [{
        "model": r["model"], "test": r["test"], "suite": _suite(r["test"]),
        "passed": r["passed"], "ground_truth": r.get("ground_truth", ""),
        "response": r.get("response", ""),
    } for r in interp_results]

    with open(interp_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=interp_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing_interp + new_interp)
    print(f"Interpretability results saved → {interp_csv}")

    perf_csv = os.path.join(RESULTS_DIR, "performance_results.csv")
    perf_fields = ["dataset", "model", "rmse", "rank"]
    existing_perf = []
    if os.path.exists(perf_csv):
        with open(perf_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("model") != model_name:
                    existing_perf.append(row)

    for ds_name, model_rmses in dataset_rmses.items():
        rmse_val = model_rmses.get(model_name, float("nan"))
        existing_perf.append({
            "dataset": ds_name, "model": model_name,
            "rmse": "" if np.isnan(rmse_val) else f"{rmse_val:.6f}", "rank": "",
        })

    by_dataset = defaultdict(list)
    for row in existing_perf:
        by_dataset[row["dataset"]].append(row)
    for ds_name, rows in by_dataset.items():
        valid = [(r, float(r["rmse"])) for r in rows if r["rmse"] not in ("", None)]
        valid.sort(key=lambda x: x[1])
        for rank_idx, (r, _) in enumerate(valid, 1):
            r["rank"] = rank_idx
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
        "commit": git_hash,
        "mean_rank": f"{mean_rank:.2f}" if not np.isnan(mean_rank) else "nan",
        "frac_interpretability_tests_passed": f"{n_passed / total:.4f}" if total > 0 else "nan",
        "status": "",
        "model_name": model_shorthand_name,
        "description": model_description,
    }], RESULTS_DIR)

    overall_csv = os.path.join(RESULTS_DIR, "overall_results.csv")
    plot_interp_vs_performance(overall_csv, os.path.join(RESULTS_DIR, "interpretability_vs_performance.png"))

    print()
    print("---")
    print(f"tests_passed:  {n_passed}/{total}" + (f" ({n_passed/total:.2%})" if total > 0 else ""))
    print(f"mean_rank:     {mean_rank:.2f}" if not np.isnan(mean_rank) else "mean_rank:     nan")
    print(f"total_seconds: {time.time() - t0:.1f}s")
