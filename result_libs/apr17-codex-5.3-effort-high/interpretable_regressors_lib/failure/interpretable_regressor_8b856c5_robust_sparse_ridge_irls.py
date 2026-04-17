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


class RobustSparseRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Robust sparse ridge equation model.

    Steps:
    1) Robustly center/scale features via median and MAD.
    2) Winsorize standardized features to reduce outlier leverage.
    3) Select ridge alpha by GCV (closed-form SVD formula).
    4) Apply adaptive soft-thresholding to induce sparsity.
    5) Refit ridge on active features and expose a raw-feature equation.
    """

    def __init__(
        self,
        alpha_grid=None,
        clip_value=4.0,
        sparsity_quantile=0.35,
        max_active=20,
        min_active=3,
    ):
        self.alpha_grid = alpha_grid
        self.clip_value = clip_value
        self.sparsity_quantile = sparsity_quantile
        self.max_active = max_active
        self.min_active = min_active

    @staticmethod
    def _robust_scale(X):
        med = np.median(X, axis=0)
        mad = np.median(np.abs(X - med), axis=0)
        scale = 1.4826 * mad
        scale = np.where(scale < 1e-8, 1.0, scale)
        return med, scale

    @staticmethod
    def _ridge_fit(X, y, alpha):
        n, p = X.shape
        D = np.hstack([np.ones((n, 1), dtype=float), X])
        reg = np.zeros(p + 1, dtype=float)
        reg[1:] = max(float(alpha), 0.0)
        A = D.T @ D + np.diag(reg)
        b = D.T @ y
        try:
            beta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(A) @ b
        return float(beta[0]), np.asarray(beta[1:], dtype=float)

    def _select_alpha_gcv(self, X, y):
        n = X.shape[0]
        if self.alpha_grid is None:
            grid = np.logspace(-4, 3, 16)
        else:
            grid = np.asarray(self.alpha_grid, dtype=float)
        grid = np.maximum(grid, 1e-12)

        y_centered = y - np.mean(y)
        U, s, _ = np.linalg.svd(X, full_matrices=False)
        s2 = s ** 2
        Uy = U.T @ y_centered

        best_alpha = float(grid[0])
        best_gcv = float("inf")
        for alpha in grid:
            shrink = s2 / (s2 + alpha)
            yhat_centered = U @ (shrink * Uy)
            resid = y_centered - yhat_centered
            mse = float(np.mean(resid ** 2))
            df = float(np.sum(shrink))
            denom = max((1.0 - df / max(n, 1)) ** 2, 1e-8)
            gcv = mse / denom
            if gcv < best_gcv:
                best_gcv = gcv
                best_alpha = float(alpha)
        return best_alpha, best_gcv

    @staticmethod
    def _adaptive_soft_threshold(coef, lam):
        return np.sign(coef) * np.maximum(np.abs(coef) - lam, 0.0)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        self.n_features_in_ = p

        med, scale = self._robust_scale(X)
        Z = (X - med) / scale
        Z = np.clip(Z, -float(self.clip_value), float(self.clip_value))

        alpha, gcv = self._select_alpha_gcv(Z, y)
        b0, w0 = self._ridge_fit(Z, y, alpha)

        absw = np.abs(w0)
        if np.all(absw < 1e-12):
            active = np.arange(min(p, max(int(self.min_active), 1)), dtype=int)
            w_sparse = np.zeros_like(w0)
        else:
            q = float(np.clip(self.sparsity_quantile, 0.0, 0.95))
            lam = float(np.quantile(absw, q))
            w_sparse = self._adaptive_soft_threshold(w0, lam)
            active = np.where(np.abs(w_sparse) > 1e-12)[0]

            if len(active) < int(self.min_active):
                k = min(p, max(int(self.min_active), 1))
                active = np.argsort(-absw)[:k]
                w_sparse = np.zeros_like(w0)
                w_sparse[active] = w0[active]

        if len(active) > int(self.max_active):
            top = np.argsort(-np.abs(w_sparse))[: int(self.max_active)]
            keep = np.zeros_like(w_sparse, dtype=bool)
            keep[top] = True
            active = np.where(keep)[0]
            w_sparse = np.where(keep, w_sparse, 0.0)

        active = np.sort(active.astype(int))
        if len(active) > 0:
            Z_active = Z[:, active]
            b_refit, w_refit_active = self._ridge_fit(Z_active, y, alpha)
            w_refit = np.zeros(p, dtype=float)
            w_refit[active] = w_refit_active
            b_use = b_refit
            w_use = w_refit
        else:
            b_use = float(np.mean(y))
            w_use = np.zeros(p, dtype=float)

        coef_raw = w_use / scale
        intercept_raw = float(b_use - np.dot(coef_raw, med))

        self.alpha_ = float(alpha)
        self.gcv_score_ = float(gcv)
        self.intercept_ = intercept_raw
        self.coef_ = np.asarray(coef_raw, dtype=float)
        self.active_features_ = np.where(np.abs(self.coef_) > 1e-12)[0].astype(int)
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "coef_", "active_features_"])
        X = np.asarray(X, dtype=float)
        return self.intercept_ + X @ self.coef_

    @staticmethod
    def _linear_terms(coef):
        active = [j for j in range(len(coef)) if abs(float(coef[j])) > 1e-12]
        active = sorted(active, key=lambda j: -abs(float(coef[j])))
        return [f"{float(coef[j]):+.6f}*x{j}" for j in active], active

    def __str__(self):
        check_is_fitted(self, ["intercept_", "coef_", "active_features_"])
        terms, active = self._linear_terms(self.coef_)
        expr = f"{float(self.intercept_):.6f}"
        if terms:
            expr += " " + " ".join(terms)

        inactive = [j for j in range(self.n_features_in_) if j not in set(active)]
        lines = [
            "Robust Sparse Ridge Regressor",
            f"Chosen ridge alpha (GCV): {self.alpha_:.6g}",
            f"GCV score: {self.gcv_score_:.6f}",
            "Raw-feature prediction equation:",
            f"y = {expr}",
            f"Active features ({len(active)}): {', '.join(f'x{j}' for j in active) if active else '(none)'}",
            f"Inactive features ({len(inactive)}): {', '.join(f'x{j}' for j in inactive) if inactive else '(none)'}",
            "Feature coefficients (sorted by |coefficient|):",
        ]
        for j in active:
            lines.append(f"  x{j}: {float(self.coef_[j]):+.6f}")
        if not active:
            lines.append("  (none)")
        lines.append(f"Intercept: {float(self.intercept_):+.6f}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
RobustSparseRidgeRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "RobustSparseRidgeIRLSV1"
model_description = "Robust median-MAD scaled and winsorized ridge with GCV alpha, adaptive soft-threshold sparsification, and ridge refit on active features for a compact explicit equation"
model_defs = [(model_shorthand_name, RobustSparseRidgeRegressor())]


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
