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


class AdaptiveMassSparseRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Closed-form ridge with adaptive coefficient-mass sparsification and refit.

    Steps:
      1) standardize features, choose ridge alpha by holdout from a fixed grid
      2) keep the smallest set of coefficients covering a target L1 mass
      3) refit ordinary least squares on just those selected features
      4) quantize coefficients used in both prediction and printed equation
    """

    def __init__(
        self,
        alpha_grid=(1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0),
        val_frac=0.2,
        min_val_samples=120,
        min_terms=1,
        max_terms=14,
        mass_keep=0.985,
        min_coef_abs=2e-4,
        coef_decimals=4,
        random_state=42,
    ):
        self.alpha_grid = alpha_grid
        self.val_frac = val_frac
        self.min_val_samples = min_val_samples
        self.min_terms = min_terms
        self.max_terms = max_terms
        self.mass_keep = mass_keep
        self.min_coef_abs = min_coef_abs
        self.coef_decimals = coef_decimals
        self.random_state = random_state

    @staticmethod
    def _ridge_closed_form(Z, y, alpha):
        p = Z.shape[1]
        reg = float(alpha) * np.eye(p)
        reg[0, 0] = 0.0
        return np.linalg.solve(Z.T @ Z + reg, Z.T @ y)

    def _impute(self, X):
        X = np.asarray(X, dtype=float).copy()
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(self.feature_medians_, np.where(bad)[1])
        return X

    def _split_idx(self, n):
        if n < int(self.min_val_samples) + 20:
            idx = np.arange(n)
            return idx, idx
        rng = np.random.RandomState(self.random_state)
        perm = rng.permutation(n)
        n_val = max(int(round(float(self.val_frac) * n)), int(self.min_val_samples))
        n_val = min(n_val, n // 2)
        return perm[n_val:], perm[:n_val]

    def _fit_dense_ridge(self, Xtr, ytr, Xval, yval):
        x_mean = Xtr.mean(axis=0)
        x_std = Xtr.std(axis=0)
        x_std[x_std < 1e-12] = 1.0
        Xtr_s = (Xtr - x_mean) / x_std
        Xval_s = (Xval - x_mean) / x_std

        Ztr = np.column_stack([np.ones(Xtr_s.shape[0]), Xtr_s])
        Zval = np.column_stack([np.ones(Xval_s.shape[0]), Xval_s])

        best_alpha = float(self.alpha_grid[0])
        best_beta = self._ridge_closed_form(Ztr, ytr, best_alpha)
        best_mse = float(np.mean((yval - Zval @ best_beta) ** 2))
        for alpha in self.alpha_grid[1:]:
            beta = self._ridge_closed_form(Ztr, ytr, float(alpha))
            mse = float(np.mean((yval - Zval @ beta) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_alpha = float(alpha)
                best_beta = beta

        beta0_s = float(best_beta[0])
        beta_s = np.asarray(best_beta[1:], dtype=float)
        coef = beta_s / x_std
        intercept = beta0_s - float(np.dot(beta_s, x_mean / x_std))
        return intercept, coef, best_alpha, best_mse

    def _select_features(self, coef):
        abs_coef = np.abs(coef)
        total = float(abs_coef.sum())
        if total <= 1e-14:
            return np.array([int(np.argmax(abs_coef))], dtype=int)

        order = np.argsort(abs_coef)[::-1]
        running = 0.0
        selected = []
        max_terms = min(int(self.max_terms), coef.shape[0])
        min_terms = min(max(1, int(self.min_terms)), max_terms)
        for idx in order:
            if abs_coef[idx] < float(self.min_coef_abs) and len(selected) >= min_terms:
                break
            selected.append(int(idx))
            running += float(abs_coef[idx])
            if len(selected) >= max_terms:
                break
            if len(selected) >= min_terms and (running / total) >= float(self.mass_keep):
                break

        if not selected:
            selected = [int(order[0])]
        selected = np.array(sorted(selected), dtype=int)
        return selected

    @staticmethod
    def _ols_with_intercept(X, y):
        Z = np.column_stack([np.ones(X.shape[0]), X])
        beta, _, _, _ = np.linalg.lstsq(Z, y, rcond=None)
        return float(beta[0]), np.asarray(beta[1:], dtype=float)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.feature_medians_ = np.nanmedian(X, axis=0)
        X = self._impute(X)

        tr_idx, val_idx = self._split_idx(X.shape[0])
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xval, yval = X[val_idx], y[val_idx]

        dense_intercept, dense_coef, alpha, _ = self._fit_dense_ridge(Xtr, ytr, Xval, yval)
        selected = self._select_features(dense_coef)

        sparse_intercept, sparse_coef = self._ols_with_intercept(X[:, selected], y)
        full_coef = np.zeros(self.n_features_in_, dtype=float)
        full_coef[selected] = sparse_coef

        q = int(self.coef_decimals)
        self.intercept_ = float(np.round(sparse_intercept, q))
        self.coef_ = np.round(full_coef, q).astype(float)
        self.selected_features_ = [int(j) for j in np.where(np.abs(self.coef_) > 0.0)[0]]
        if len(self.selected_features_) == 0:
            j = int(np.argmax(np.abs(full_coef)))
            self.selected_features_ = [j]
            self.coef_[j] = float(np.round(full_coef[j], q))
        self.alpha_ = float(alpha)

        self.feature_importance_ = np.abs(self.coef_)
        self.feature_rank_ = np.argsort(self.feature_importance_)[::-1]
        self.fitted_mse_ = float(np.mean((y - self.predict(X)) ** 2))
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "coef_", "selected_features_"])
        X = self._impute(X)
        return self.intercept_ + X @ self.coef_

    def __str__(self):
        check_is_fitted(self, ["intercept_", "coef_", "selected_features_", "feature_importance_"])
        lines = [
            "AdaptiveMassSparseRidgeRegressor",
            f"Selected alpha (dense ridge stage): {self.alpha_:.5g}",
            "",
            "Prediction equation (exact coefficients used by predict):",
            f"  y = {self.intercept_:+.4f}",
        ]
        for j in self.selected_features_:
            lines.append(f"    + ({float(self.coef_[j]):+.4f})*x{int(j)}")

        lines.append("")
        lines.append("Top feature importance (abs coefficient):")
        for j in self.feature_rank_[: min(10, self.n_features_in_)]:
            lines.append(f"  x{int(j)}: {self.feature_importance_[int(j)]:.5f}")

        inactive = [f"x{j}" for j in range(self.n_features_in_) if abs(float(self.coef_[j])) <= 1e-12]
        if inactive:
            lines.append("Features with negligible effect: " + ", ".join(inactive))
        lines.append(f"Training MSE: {self.fitted_mse_:.6f}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
AdaptiveMassSparseRidgeRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "AdaptiveMassSparseRidgeV1"
model_description = "Custom closed-form ridge with holdout alpha selection, adaptive coefficient-mass sparsification, sparse OLS refit, and quantized exact equation"
model_defs = [(model_shorthand_name, AdaptiveMassSparseRidgeRegressor())]

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
