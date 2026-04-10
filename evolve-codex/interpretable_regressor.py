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
from sklearn.linear_model import LassoCV
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class SparseBinnedAdditiveRegressor(BaseEstimator, RegressorMixin):
    """
    Interpretable scikit-learn compatible regressor.

    Sparse additive step-function regressor.
    Selects informative features, bins each selected feature by quantiles,
    and fits an L1-regularized linear model on the resulting bin indicators.
    Must implement: fit(X, y), predict(X), and __str__().
    """

    def __init__(self, max_features=8, max_bins=6, min_bin_frac=0.08):
        self.max_features = max_features
        self.max_bins = max_bins
        self.min_bin_frac = min_bin_frac

    @staticmethod
    def _as_2d_float(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def _impute(self, X):
        return np.where(np.isnan(X), self.feature_medians_, X)

    def _feature_screen(self, X, y):
        y_centered = y - y.mean()
        y_scale = np.linalg.norm(y_centered) + 1e-12
        scores = np.zeros(X.shape[1], dtype=float)
        for j in range(X.shape[1]):
            xj = X[:, j]
            xj_centered = xj - xj.mean()
            denom = (np.linalg.norm(xj_centered) * y_scale) + 1e-12
            scores[j] = abs(np.dot(xj_centered, y_centered) / denom)
        order = np.argsort(-scores)
        k = min(self.max_features, X.shape[1])
        return order[:k], scores

    def _build_bins(self, x):
        n = x.shape[0]
        target_bins = min(self.max_bins, max(2, int(1.0 / max(self.min_bin_frac, 1e-3))))
        q = np.linspace(0.0, 1.0, target_bins + 1)
        edges = np.quantile(x, q)
        edges = np.unique(edges)
        if edges.size < 2:
            return None
        return edges

    def _transform(self, X):
        n = X.shape[0]
        if len(self.feature_bins_) == 0:
            return np.zeros((n, 1), dtype=float)
        cols = []
        for feat_idx, edges in self.feature_bins_:
            xj = X[:, feat_idx]
            bin_ids = np.searchsorted(edges[1:-1], xj, side="right")
            n_bins = edges.size - 1
            for b in range(1, n_bins):
                cols.append((bin_ids == b).astype(float))
        if len(cols) == 0:
            return np.zeros((n, 1), dtype=float)
        return np.column_stack(cols)

    def fit(self, X, y):
        X = self._as_2d_float(X)
        y = np.asarray(y, dtype=float).ravel()
        self.feature_medians_ = np.nanmedian(X, axis=0)
        self.feature_medians_ = np.where(np.isnan(self.feature_medians_), 0.0, self.feature_medians_)
        X_imp = self._impute(X)

        selected, self.screen_scores_ = self._feature_screen(X_imp, y)
        self.feature_bins_ = []
        for j in selected:
            edges = self._build_bins(X_imp[:, j])
            if edges is not None:
                self.feature_bins_.append((int(j), edges))

        Z = self._transform(X_imp)
        self.linear_ = LassoCV(
            cv=3,
            random_state=42,
            max_iter=6000,
            n_alphas=60,
            fit_intercept=True,
        )
        self.linear_.fit(Z, y)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self, "linear_")
        X = self._as_2d_float(X)
        X_imp = self._impute(X)
        Z = self._transform(X_imp)
        return self.linear_.predict(Z)

    def __str__(self):
        check_is_fitted(self, "linear_")
        lines = [
            "SparseBinnedAdditiveRegressor",
            f"intercept = {self.linear_.intercept_:.6f}",
            "prediction = intercept + sum(feature_bin_effects)",
        ]
        coef = self.linear_.coef_.ravel()
        c = 0
        for feat_idx, edges in self.feature_bins_:
            n_bins = edges.size - 1
            if n_bins <= 1:
                continue
            effects = np.zeros(n_bins, dtype=float)
            for b in range(1, n_bins):
                effects[b] = coef[c]
                c += 1
            if np.max(np.abs(effects)) < 1e-6:
                continue
            lines.append(f"x{feat_idx}:")
            for b in range(n_bins):
                lo = edges[b]
                hi = edges[b + 1]
                lines.append(f"  [{lo:.4g}, {hi:.4g}] -> {effects[b]:+.5f}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseBinnedAdditiveRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseBinnedAdditive_v1"
model_description = "Custom sparse additive step-function model: correlation screening + quantile bins + L1 shrinkage on bin indicators"
model_defs = [(model_shorthand_name, SparseBinnedAdditiveRegressor())]


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
