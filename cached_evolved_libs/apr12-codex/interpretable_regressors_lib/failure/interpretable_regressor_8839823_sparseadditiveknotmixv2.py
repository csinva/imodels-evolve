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


class SparseAdditiveKnotRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse additive one-knot model.

    The model chooses a small subset of features, then for each selected feature
    fits two basis terms: x_j and max(0, x_j - t_j), where t_j is a learned knot.
    """

    def __init__(
        self,
        max_active_features=4,
        alpha_grid=(1e-3, 1e-2, 1e-1, 1.0, 10.0),
        knot_quantiles=(0.2, 0.4, 0.6, 0.8),
        val_frac=0.2,
        min_val_samples=60,
        min_coef_abs=1e-7,
        random_state=42,
    ):
        self.max_active_features = max_active_features
        self.alpha_grid = alpha_grid
        self.knot_quantiles = knot_quantiles
        self.val_frac = val_frac
        self.min_val_samples = min_val_samples
        self.min_coef_abs = min_coef_abs
        self.random_state = random_state

    @staticmethod
    def _ridge_closed_form(Z, y, alpha):
        n_cols = Z.shape[1]
        reg = float(alpha) * np.eye(n_cols)
        reg[0, 0] = 0.0  # do not penalize intercept
        beta = np.linalg.solve(Z.T @ Z + reg, Z.T @ y)
        return beta

    @staticmethod
    def _corr_strength(X, y):
        yc = y - float(np.mean(y))
        xs = np.std(X, axis=0)
        xs[xs < 1e-12] = 1.0
        x_centered = X - np.mean(X, axis=0)
        corr_like = np.abs((x_centered.T @ yc) / (len(y) * xs))
        return corr_like

    def _impute(self, X):
        X = np.asarray(X, dtype=float).copy()
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(self.feature_medians_, np.where(bad)[1])
        return X

    def _make_split(self, n):
        if n < int(self.min_val_samples) + 20:
            idx = np.arange(n)
            return idx, idx
        rng = np.random.RandomState(self.random_state)
        perm = rng.permutation(n)
        n_val = max(int(round(float(self.val_frac) * n)), int(self.min_val_samples))
        n_val = min(n_val, n // 2)
        return perm[n_val:], perm[:n_val]

    def _build_design(self, X):
        cols = [np.ones(X.shape[0])]
        for j, t in zip(self.active_features_, self.knots_):
            xj = X[:, int(j)]
            cols.append(xj)
            cols.append(np.maximum(0.0, xj - float(t)))
        return np.column_stack(cols)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        self.n_features_in_ = p

        self.feature_medians_ = np.nanmedian(X, axis=0)
        X = self._impute(X)

        tr_idx, val_idx = self._make_split(n)
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xval, yval = X[val_idx], y[val_idx]

        strength = self._corr_strength(Xtr, ytr)
        k = min(max(int(self.max_active_features), 1), p)
        ordered = np.argsort(strength)[::-1]
        self.active_features_ = [int(j) for j in np.sort(ordered[:k])]

        self.knots_ = []
        for j in self.active_features_:
            xj_tr = Xtr[:, j]
            cand = np.unique(np.quantile(xj_tr, self.knot_quantiles))
            if cand.size == 0:
                self.knots_.append(float(np.median(xj_tr)))
                continue
            best_t, best_mse = float(cand[0]), np.inf
            for t in cand:
                Ztr = np.column_stack([np.ones(len(xj_tr)), xj_tr, np.maximum(0.0, xj_tr - float(t))])
                beta = self._ridge_closed_form(Ztr, ytr, alpha=1e-3)
                xj_val = Xval[:, j]
                Zval = np.column_stack([np.ones(len(xj_val)), xj_val, np.maximum(0.0, xj_val - float(t))])
                mse = float(np.mean((yval - Zval @ beta) ** 2))
                if mse < best_mse:
                    best_mse = mse
                    best_t = float(t)
            self.knots_.append(best_t)

        Ztr_full = self._build_design(Xtr)
        Zval_full = self._build_design(Xval)

        best_alpha = float(self.alpha_grid[0])
        best_beta = None
        best_mse = np.inf
        for alpha in self.alpha_grid:
            beta = self._ridge_closed_form(Ztr_full, ytr, alpha=float(alpha))
            mse = float(np.mean((yval - Zval_full @ beta) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_alpha = float(alpha)
                best_beta = beta

        self.alpha_ = best_alpha

        # Refit on all samples using selected features/knots and chosen alpha.
        Zall = self._build_design(X)
        beta_all = self._ridge_closed_form(Zall, y, self.alpha_)
        self.intercept_ = float(beta_all[0])
        self.linear_coefs_ = np.asarray(beta_all[1::2], dtype=float)
        self.hinge_coefs_ = np.asarray(beta_all[2::2], dtype=float)

        # Prune weak features and refit once for a cleaner equation.
        keep = []
        for i in range(len(self.active_features_)):
            if abs(float(self.linear_coefs_[i])) + abs(float(self.hinge_coefs_[i])) > float(self.min_coef_abs):
                keep.append(i)
        if keep:
            self.active_features_ = [self.active_features_[i] for i in keep]
            self.knots_ = [self.knots_[i] for i in keep]
            Zall = self._build_design(X)
            beta_all = self._ridge_closed_form(Zall, y, self.alpha_)
            self.intercept_ = float(beta_all[0])
            self.linear_coefs_ = np.asarray(beta_all[1::2], dtype=float)
            self.hinge_coefs_ = np.asarray(beta_all[2::2], dtype=float)
        else:
            self.active_features_ = []
            self.knots_ = []
            self.linear_coefs_ = np.zeros(0, dtype=float)
            self.hinge_coefs_ = np.zeros(0, dtype=float)

        imp = np.zeros(p, dtype=float)
        for j, bl, bh in zip(self.active_features_, self.linear_coefs_, self.hinge_coefs_):
            imp[int(j)] = abs(float(bl)) + abs(float(bh))
        self.feature_importance_ = imp
        self.feature_rank_ = np.argsort(imp)[::-1]
        self.fitted_mse_ = float(np.mean((y - self.predict(X)) ** 2))
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "active_features_", "knots_", "linear_coefs_", "hinge_coefs_"])
        X = self._impute(X)
        yhat = np.full(X.shape[0], self.intercept_, dtype=float)
        for j, t, bl, bh in zip(self.active_features_, self.knots_, self.linear_coefs_, self.hinge_coefs_):
            xj = X[:, int(j)]
            yhat += float(bl) * xj + float(bh) * np.maximum(0.0, xj - float(t))
        return yhat

    def __str__(self):
        check_is_fitted(self, ["intercept_", "active_features_", "knots_", "linear_coefs_", "hinge_coefs_"])
        lines = [
            "SparseAdditiveKnotRegressor",
            f"Ridge alpha: {self.alpha_:.5g}",
            "",
            "Prediction equation:",
            f"  y = {self.intercept_:+.4f}",
        ]
        if len(self.active_features_) == 0:
            lines.append("  (no active feature terms)")
        else:
            for j, t, bl, bh in zip(self.active_features_, self.knots_, self.linear_coefs_, self.hinge_coefs_):
                lines.append(
                    f"    + ({float(bl):+.4f})*x{int(j)}"
                    f" + ({float(bh):+.4f})*max(0, x{int(j)}-{float(t):+.4f})"
                )

        lines.append("")
        lines.append("Top feature importance:")
        for j in self.feature_rank_[: min(10, self.n_features_in_)]:
            lines.append(f"  x{int(j)}: {self.feature_importance_[int(j)]:.5f}")

        inactive = [f"x{j}" for j in range(self.n_features_in_) if self.feature_importance_[j] <= 1e-10]
        if inactive:
            lines.append("Features with negligible effect: " + ", ".join(inactive))
        lines.append(f"Training MSE: {self.fitted_mse_:.6f}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseAdditiveKnotRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseAdditiveKnotMixV2"
model_description = "Sparse additive one-knot-per-feature regressor with correlation-screened features and closed-form ridge fitting"
model_defs = [(model_shorthand_name, SparseAdditiveKnotRegressor())]

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
