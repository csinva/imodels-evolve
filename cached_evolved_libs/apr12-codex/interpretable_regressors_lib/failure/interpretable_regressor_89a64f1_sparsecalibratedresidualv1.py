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


class SparseCalibratedResidualRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse ridge-like linear model with one optional residual offset split.

    Final prediction:
      y = intercept + sum_j coef_j * x_j + offset(x_split)
    where offset(x_split) is a two-branch constant correction.
    """

    def __init__(
        self,
        alpha_grid=(0.01, 0.03, 0.1, 0.3, 1.0, 3.0),
        max_terms=8,
        coef_tol=1e-4,
        top_split_features=8,
        split_quantiles=(0.15, 0.3, 0.5, 0.7, 0.85),
        min_leaf_frac=0.1,
        split_shrinkage=40.0,
        min_split_gain_frac=0.006,
        random_state=42,
    ):
        self.alpha_grid = alpha_grid
        self.max_terms = max_terms
        self.coef_tol = coef_tol
        self.top_split_features = top_split_features
        self.split_quantiles = split_quantiles
        self.min_leaf_frac = min_leaf_frac
        self.split_shrinkage = split_shrinkage
        self.min_split_gain_frac = min_split_gain_frac
        self.random_state = random_state

    def _impute(self, X):
        X = np.asarray(X, dtype=float).copy()
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(self.feature_medians_, np.where(bad)[1])
        return X

    @staticmethod
    def _fit_ridge_standardized(X, y, alpha):
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        sigma[sigma < 1e-10] = 1.0
        Xs = (X - mu) / sigma

        y_mean = float(np.mean(y))
        yc = y - y_mean

        p = X.shape[1]
        gram = Xs.T @ Xs
        beta_s = np.linalg.solve(gram + float(alpha) * np.eye(p), Xs.T @ yc)
        coef = beta_s / sigma
        intercept = y_mean - float(np.dot(mu, coef))
        return float(intercept), coef.astype(float)

    def _choose_alpha(self, X, y):
        n = X.shape[0]
        if n < 40:
            return float(self.alpha_grid[0])

        rng = np.random.RandomState(self.random_state)
        idx = np.arange(n)
        rng.shuffle(idx)
        cut = max(int(0.8 * n), n - 40)
        tr = idx[:cut]
        va = idx[cut:]
        if va.size < 12:
            return float(self.alpha_grid[0])

        best_alpha = float(self.alpha_grid[0])
        best_mse = np.inf
        for a in self.alpha_grid:
            intercept, coef = self._fit_ridge_standardized(X[tr], y[tr], a)
            pred = intercept + X[va] @ coef
            mse = float(np.mean((y[va] - pred) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_alpha = float(a)
        return best_alpha

    def _fit_sparse_backbone(self, X, y):
        self.alpha_ = self._choose_alpha(X, y)
        intercept_dense, coef_dense = self._fit_ridge_standardized(X, y, self.alpha_)

        active = np.where(np.abs(coef_dense) > 1e-12)[0]
        if active.size == 0:
            active = np.array([int(np.argmax(np.abs(coef_dense)))], dtype=int)
        if active.size > int(self.max_terms):
            order = np.argsort(np.abs(coef_dense[active]))[::-1]
            active = active[order[: int(self.max_terms)]]

        # Refit selected terms with least squares for an exact compact equation.
        A = np.column_stack([np.ones(X.shape[0]), X[:, active]])
        beta, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        intercept_sparse = float(beta[0])
        coef_sparse = np.zeros(X.shape[1], dtype=float)
        coef_sparse[active] = beta[1:]

        pred_sparse = intercept_sparse + X @ coef_sparse
        mse_sparse = float(np.mean((y - pred_sparse) ** 2))

        pred_dense = intercept_dense + X @ coef_dense
        mse_dense = float(np.mean((y - pred_dense) ** 2))

        # Keep sparse unless it degrades too much.
        if mse_sparse <= mse_dense * 1.06:
            return intercept_sparse, coef_sparse
        return intercept_dense, coef_dense

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.feature_medians_ = np.nanmedian(X, axis=0)
        X = self._impute(X)

        self.intercept_, self.coef_ = self._fit_sparse_backbone(X, y)
        pred_linear = self.intercept_ + X @ self.coef_
        residual = y - pred_linear
        base_mse = float(np.mean(residual ** 2))

        corr = np.zeros(self.n_features_in_, dtype=float)
        for j in range(self.n_features_in_):
            xj = X[:, j]
            if np.std(xj) > 1e-12:
                corr[j] = abs(float(np.corrcoef(xj, residual)[0, 1]))
        candidate = np.argsort(corr)[::-1][: min(int(self.top_split_features), self.n_features_in_)]

        n = X.shape[0]
        min_leaf = max(10, int(self.min_leaf_frac * n))
        best = None

        for j in candidate:
            xj = X[:, j]
            thresholds = np.unique(np.quantile(xj, self.split_quantiles))
            for thr in thresholds:
                left = xj <= thr
                right = ~left
                nl = int(left.sum())
                nr = int(right.sum())
                if nl < min_leaf or nr < min_leaf:
                    continue

                # Shrink offsets toward zero for stability.
                c_left = float(np.mean(residual[left])) * (nl / (nl + float(self.split_shrinkage)))
                c_right = float(np.mean(residual[right])) * (nr / (nr + float(self.split_shrinkage)))

                pred = pred_linear + np.where(left, c_left, c_right)
                mse = float(np.mean((y - pred) ** 2))
                if best is None or mse < best["mse"]:
                    best = {
                        "feature": int(j),
                        "threshold": float(thr),
                        "left_offset": float(c_left),
                        "right_offset": float(c_right),
                        "mse": mse,
                    }

        self.has_split_ = False
        self.split_feature_ = -1
        self.split_threshold_ = 0.0
        self.left_offset_ = 0.0
        self.right_offset_ = 0.0

        if best is not None:
            gain = base_mse - float(best["mse"])
            if gain > float(self.min_split_gain_frac) * (np.var(y) + 1e-12):
                self.has_split_ = True
                self.split_feature_ = int(best["feature"])
                self.split_threshold_ = float(best["threshold"])
                self.left_offset_ = float(best["left_offset"])
                self.right_offset_ = float(best["right_offset"])

        self.coef_[np.abs(self.coef_) < self.coef_tol] = 0.0
        if abs(self.left_offset_) < self.coef_tol:
            self.left_offset_ = 0.0
        if abs(self.right_offset_) < self.coef_tol:
            self.right_offset_ = 0.0

        self.feature_importance_ = np.abs(self.coef_)
        if self.has_split_:
            self.feature_importance_[self.split_feature_] += abs(self.left_offset_) + abs(self.right_offset_)
        self.selected_feature_order_ = np.argsort(self.feature_importance_)[::-1]
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "coef_", "has_split_"])
        X = self._impute(X)
        yhat = self.intercept_ + X @ self.coef_
        if self.has_split_:
            yhat = yhat + np.where(
                X[:, self.split_feature_] <= self.split_threshold_,
                self.left_offset_,
                self.right_offset_,
            )
        return yhat

    def __str__(self):
        check_is_fitted(self, ["intercept_", "coef_", "has_split_"])
        lines = [
            "SparseCalibratedResidualRegressor",
            "Exact prediction equation:",
            f"  base = {self.intercept_:+.6f}",
        ]

        active = np.where(np.abs(self.coef_) >= self.coef_tol)[0]
        if active.size:
            for j in active[np.argsort(np.abs(self.coef_[active]))[::-1]]:
                lines.append(f"       {self.coef_[j]:+.6f} * x{int(j)}")
        else:
            lines.append("       (no active linear terms)")

        if self.has_split_:
            lines.append(
                f"  residual_offset = {self.left_offset_:+.6f} if x{self.split_feature_} <= {self.split_threshold_:+.6f} "
                f"else {self.right_offset_:+.6f}"
            )
        else:
            lines.append("  residual_offset = 0")

        lines.append("  prediction = base + residual_offset")
        lines.append("")
        lines.append("Feature summary (sorted by total attributed effect):")
        for j in self.selected_feature_order_[: min(12, self.n_features_in_)]:
            lines.append(f"  x{int(j)}: linear={self.coef_[int(j)]:+.6f}, importance={self.feature_importance_[int(j)]:.6f}")
        inactive = [f"x{j}" for j in range(self.n_features_in_) if self.feature_importance_[j] < self.coef_tol]
        if inactive:
            lines.append("Features with near-zero effect: " + ", ".join(inactive))
        lines.append("To simulate: compute base from linear terms, then add residual_offset.")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseCalibratedResidualRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseCalibratedResidualV1"
model_description = "Sparse ridge-selected linear backbone with one shrunk residual threshold offset for compact nonlinear calibration"
model_defs = [(model_shorthand_name, SparseCalibratedResidualRegressor())]

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
