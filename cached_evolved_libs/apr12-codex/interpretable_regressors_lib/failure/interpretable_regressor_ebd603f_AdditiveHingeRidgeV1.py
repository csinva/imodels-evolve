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


class AdditiveHingeRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Compact additive spline-like regressor with one learned hinge per feature.

    Model:
      y = intercept + sum_j a_j * x_j + sum_j h_j * max(0, x_j - t_j)
    where each threshold t_j is selected from feature quantiles to maximize
    univariate hinge usefulness, then all terms are jointly ridge-refit.
    """

    def __init__(
        self,
        ridge_alpha=0.25,
        n_knot_candidates=7,
        coef_tol=5e-3,
        sparsity_rel=0.025,
    ):
        self.ridge_alpha = ridge_alpha
        self.n_knot_candidates = n_knot_candidates
        self.coef_tol = coef_tol
        self.sparsity_rel = sparsity_rel

    def _impute(self, X):
        X = np.asarray(X, dtype=float).copy()
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(self.feature_medians_, np.where(bad)[1])
        return X

    def _choose_knot(self, xj, y):
        q = np.linspace(0.1, 0.9, max(3, int(self.n_knot_candidates)))
        candidates = np.unique(np.quantile(xj, q))
        if candidates.size == 0:
            return 0.0
        y0 = y - np.mean(y)
        best_t = float(candidates[0])
        best_score = -np.inf
        for t in candidates:
            h = np.maximum(0.0, xj - t)
            h = h - np.mean(h)
            denom = np.linalg.norm(h) * np.linalg.norm(y0) + 1e-12
            score = abs(float(h @ y0) / denom)
            if score > best_score:
                best_score = score
                best_t = float(t)
        return best_t

    def _fit_ridge(self, Z, y):
        n, p = Z.shape
        A = np.column_stack([np.ones(n), Z])
        reg = float(self.ridge_alpha) * np.eye(p + 1)
        reg[0, 0] = 0.0
        return np.linalg.solve(A.T @ A + reg, A.T @ y)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.feature_medians_ = np.nanmedian(X, axis=0)
        X = self._impute(X)
        n, p = X.shape

        self.feature_means_ = np.mean(X, axis=0)
        self.feature_scales_ = np.std(X, axis=0)
        self.feature_scales_[self.feature_scales_ < 1e-8] = 1.0
        Xs = (X - self.feature_means_) / self.feature_scales_

        self.knots_ = np.array([self._choose_knot(Xs[:, j], y) for j in range(p)], dtype=float)
        H = np.maximum(0.0, Xs - self.knots_[None, :])
        self.hinge_means_ = np.mean(H, axis=0)
        self.hinge_scales_ = np.std(H, axis=0)
        self.hinge_scales_[self.hinge_scales_ < 1e-8] = 1.0
        Hz = (H - self.hinge_means_) / self.hinge_scales_

        Z = np.hstack([Xs, Hz])
        beta = self._fit_ridge(Z, y)
        self.intercept_ = float(beta[0])
        self.linear_coef_std_ = beta[1 : 1 + p].astype(float)
        self.hinge_coef_std_ = beta[1 + p : 1 + 2 * p].astype(float)

        self.linear_coef_ = self.linear_coef_std_ / self.feature_scales_
        self.hinge_coef_ = self.hinge_coef_std_ / (self.hinge_scales_ * self.feature_scales_)
        self.intercept_ -= float(np.sum(self.linear_coef_std_ * self.feature_means_ / self.feature_scales_))
        self.intercept_ -= float(np.sum(self.hinge_coef_std_ * self.hinge_means_ / self.hinge_scales_))

        abs_terms = np.abs(np.concatenate([self.linear_coef_, self.hinge_coef_]))
        rel_cut = float(self.sparsity_rel) * (float(np.max(abs_terms)) if abs_terms.size else 0.0)
        cut = max(float(self.coef_tol), rel_cut)
        self.linear_coef_[np.abs(self.linear_coef_) < cut] = 0.0
        self.hinge_coef_[np.abs(self.hinge_coef_) < cut] = 0.0

        self.feature_importance_ = np.abs(self.linear_coef_) + np.abs(self.hinge_coef_)
        self.selected_feature_order_ = np.argsort(self.feature_importance_)[::-1]
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "linear_coef_", "hinge_coef_", "knots_"])
        X = self._impute(X)
        Xs = (X - self.feature_means_) / self.feature_scales_
        H = np.maximum(0.0, Xs - self.knots_[None, :])
        return self.intercept_ + X @ self.linear_coef_ + H @ self.hinge_coef_

    def __str__(self):
        check_is_fitted(self, ["intercept_", "linear_coef_", "hinge_coef_", "knots_"])
        lines = [
            "AdditiveHingeRidgeRegressor",
            "Prediction equation:",
            f"  y = {self.intercept_:+.6f}",
        ]
        active_linear = [j for j in range(self.n_features_in_) if abs(self.linear_coef_[j]) > 0]
        active_hinge = [j for j in range(self.n_features_in_) if abs(self.hinge_coef_[j]) > 0]
        for j in active_linear:
            lines.append(f"      {self.linear_coef_[j]:+.6f} * x{j}")
        for j in active_hinge:
            lines.append(
                f"      {self.hinge_coef_[j]:+.6f} * max(0, ((x{j} - {self.feature_means_[j]:+.6f})/{self.feature_scales_[j]:.6f}) - {self.knots_[j]:+.6f})"
            )
        if not active_linear and not active_hinge:
            lines.append("      (all terms pruned; constant model)")
        lines.append("")
        lines.append("Feature summary (sorted by |linear|+|hinge|):")
        shown = [int(j) for j in self.selected_feature_order_[: min(12, self.n_features_in_)]]
        for j in shown:
            lines.append(
                f"  x{j}: linear={self.linear_coef_[j]:+.6f}, "
                f"hinge={self.hinge_coef_[j]:+.6f}, "
                f"hinge_threshold_raw≈{self.feature_means_[j] + self.feature_scales_[j] * self.knots_[j]:+.4f}"
            )
        inactive = [f"x{j}" for j in range(self.n_features_in_) if self.feature_importance_[j] <= 0]
        if inactive:
            lines.append("Inactive features: " + ", ".join(inactive))
        lines.append("To predict, compute each listed term and sum with the intercept.")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
AdditiveHingeRidgeRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "AdditiveHingeRidgeV1"
model_description = "Custom additive linear-plus-one-hinge-per-feature regressor with quantile-selected knots, joint ridge fit, and pruned explicit equation"
model_defs = [(model_shorthand_name, AdditiveHingeRidgeRegressor())]

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
