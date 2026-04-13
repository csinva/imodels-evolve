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
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class StabilitySparseLinearRegressor(BaseEstimator, RegressorMixin):
    """
    Bootstrap stability-selected sparse linear regression.

    1) Repeatedly subsample rows and record top-|corr| features.
    2) Keep features that appear frequently (or top-frequency fallback).
    3) Fit ridge on selected standardized features and map back to original units.
    """

    def __init__(
        self,
        max_active_features=8,
        n_bootstraps=10,
        subsample_frac=0.7,
        per_bootstrap_topk=10,
        min_selection_freq=0.35,
        ridge_alpha=0.12,
        coef_tol=1e-3,
        random_state=42,
    ):
        self.max_active_features = max_active_features
        self.n_bootstraps = n_bootstraps
        self.subsample_frac = subsample_frac
        self.per_bootstrap_topk = per_bootstrap_topk
        self.min_selection_freq = min_selection_freq
        self.ridge_alpha = ridge_alpha
        self.coef_tol = coef_tol
        self.random_state = random_state

    def _impute(self, X):
        X = np.asarray(X, dtype=float).copy()
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(self.feature_medians_, np.where(bad)[1])
        return X

    @staticmethod
    def _corr_scores(X, y):
        Xc = X - np.mean(X, axis=0, keepdims=True)
        yc = y - np.mean(y)
        denom = (np.linalg.norm(Xc, axis=0) * np.linalg.norm(yc)) + 1e-12
        return np.abs((Xc.T @ yc) / denom)

    def _fit_ridge_with_intercept(self, Xs, y):
        n, p = Xs.shape
        A = np.column_stack([np.ones(n), Xs])
        gram = A.T @ A
        reg = float(self.ridge_alpha) * np.eye(p + 1)
        reg[0, 0] = 0.0
        return np.linalg.solve(gram + reg, A.T @ y)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.feature_medians_ = np.nanmedian(X, axis=0)
        X = self._impute(X)

        n, p = X.shape
        rng = np.random.RandomState(self.random_state)

        counts = np.zeros(p, dtype=float)
        b_rows = max(20, int(self.subsample_frac * n))
        b_topk = min(max(1, int(self.per_bootstrap_topk)), p)
        for _ in range(max(1, int(self.n_bootstraps))):
            idx = rng.choice(n, size=b_rows, replace=True)
            scores = self._corr_scores(X[idx], y[idx])
            top = np.argsort(scores)[::-1][:b_topk]
            counts[top] += 1.0

        freq = counts / float(max(1, int(self.n_bootstraps)))
        selected = np.where(freq >= float(self.min_selection_freq))[0].tolist()
        if not selected:
            selected = np.argsort(freq)[::-1][: min(int(self.max_active_features), p)].tolist()
        selected = sorted(selected, key=lambda j: -freq[j])[: min(int(self.max_active_features), p)]

        self.selection_frequency_ = freq
        self.active_features_ = np.array(selected, dtype=int)

        if self.active_features_.size == 0:
            self.intercept_ = float(np.mean(y))
            self.coef_ = np.zeros(p, dtype=float)
            self.feature_importance_ = np.zeros(p, dtype=float)
            self.selected_feature_order_ = np.arange(p)
            return self

        Xs = X[:, self.active_features_]
        self.active_means_ = np.mean(Xs, axis=0)
        self.active_scales_ = np.std(Xs, axis=0)
        self.active_scales_[self.active_scales_ < 1e-8] = 1.0
        Xz = (Xs - self.active_means_) / self.active_scales_

        beta = self._fit_ridge_with_intercept(Xz, y)
        b0 = float(beta[0])
        bz = beta[1:]

        self.coef_ = np.zeros(p, dtype=float)
        for k, j in enumerate(self.active_features_):
            self.coef_[j] = float(bz[k] / self.active_scales_[k])

        self.intercept_ = float(b0 - np.sum((bz * self.active_means_) / self.active_scales_))

        tiny = np.abs(self.coef_) <= float(self.coef_tol)
        self.coef_[tiny] = 0.0
        self.feature_importance_ = np.abs(self.coef_)
        self.selected_feature_order_ = np.argsort(self.feature_importance_)[::-1]
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "coef_", "feature_importance_"])
        X = self._impute(X)
        return self.intercept_ + X @ self.coef_

    def __str__(self):
        check_is_fitted(self, ["intercept_", "coef_", "feature_importance_"])
        active = [int(j) for j in np.where(np.abs(self.coef_) > 0)[0]]
        lines = [
            "StabilitySparseLinearRegressor",
            "Prediction equation:",
            f"  y = {self.intercept_:+.6f}",
        ]
        for j in active:
            lines.append(f"      {self.coef_[j]:+.6f} * x{j}")
        if not active:
            lines.append("      (no active feature terms)")
        lines.append("")
        lines.append("Feature summary (sorted by |coefficient|):")
        shown = [int(j) for j in self.selected_feature_order_[: min(12, self.n_features_in_)]]
        for j in shown:
            lines.append(
                f"  x{j}: coef={self.coef_[j]:+.6f}, "
                f"selection_freq={self.selection_frequency_[j]:.2f}"
            )
        inactive = [f"x{j}" for j in range(self.n_features_in_) if abs(self.coef_[j]) <= float(self.coef_tol)]
        if inactive:
            lines.append("Inactive (near-zero) features: " + ", ".join(inactive))
        lines.append("To predict, plug feature values into the equation and sum all terms.")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
StabilitySparseLinearRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "StabilitySparseLinearV1"
model_description = "Bootstrap stability-selected sparse linear equation with ridge refit and explicit coefficient/selection-frequency summary"
model_defs = [(model_shorthand_name, StabilitySparseLinearRegressor())]

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------

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
