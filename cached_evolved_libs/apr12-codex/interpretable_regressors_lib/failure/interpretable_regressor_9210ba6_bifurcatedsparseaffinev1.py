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


class BifurcatedSparseAffineRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse linear backbone plus one piecewise-affine residual correction.

    Final form:
      y = b + sum_j w_j x_j + g(x_s)
    where g uses one threshold on a single feature x_s and has one affine branch
    on each side of the threshold.
    """

    def __init__(
        self,
        max_terms=6,
        top_split_features=8,
        split_quantiles=(0.2, 0.35, 0.5, 0.65, 0.8),
        min_leaf_frac=0.12,
        min_split_gain=0.015,
        coef_tol=8e-4,
    ):
        self.max_terms = max_terms
        self.top_split_features = top_split_features
        self.split_quantiles = split_quantiles
        self.min_leaf_frac = min_leaf_frac
        self.min_split_gain = min_split_gain
        self.coef_tol = coef_tol

    def _impute(self, X):
        X = np.asarray(X, dtype=float).copy()
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(self.feature_medians_, np.where(bad)[1])
        return X

    @staticmethod
    def _fit_affine_1d(x, y):
        xc = x - float(np.mean(x))
        A = np.column_stack([np.ones_like(xc), xc])
        beta, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        return float(beta[0]), float(beta[1]), float(np.mean(x))

    def _fit_sparse_linear(self, X, y):
        x_mean = np.mean(X, axis=0)
        x_std = np.std(X, axis=0)
        x_std[x_std < 1e-12] = 1.0
        Xs = (X - x_mean) / x_std

        sel = LassoCV(cv=3, n_alphas=32, max_iter=5000, random_state=42)
        sel.fit(Xs, y)

        coef_dense = sel.coef_ / x_std
        intercept_dense = float(sel.intercept_ - np.dot(coef_dense, x_mean))

        active = np.where(np.abs(coef_dense) > 1e-8)[0]
        if active.size == 0:
            active = np.array([int(np.argmax(np.abs(coef_dense)))], dtype=int)
        if active.size > int(self.max_terms):
            order = np.argsort(np.abs(coef_dense[active]))[::-1]
            active = active[order[: int(self.max_terms)]]

        coef = np.zeros(X.shape[1], dtype=float)
        if active.size:
            A = np.column_stack([np.ones(X.shape[0]), X[:, active]])
            beta, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            intercept = float(beta[0])
            coef[active] = beta[1:]
        else:
            intercept = intercept_dense
            coef[:] = coef_dense

        pred = intercept + X @ coef
        mse = float(np.mean((y - pred) ** 2))
        mse_dense = float(np.mean((y - (intercept_dense + X @ coef_dense)) ** 2))
        if mse > mse_dense * 1.12:
            intercept = intercept_dense
            coef = coef_dense
            pred = intercept + X @ coef
            mse = float(np.mean((y - pred) ** 2))
            active = np.where(np.abs(coef) > 1e-8)[0]

        return {
            "intercept": float(intercept),
            "coef": coef.astype(float),
            "active": active,
            "pred": pred,
            "mse": mse,
        }

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.feature_medians_ = np.nanmedian(X, axis=0)
        X = self._impute(X)

        base = self._fit_sparse_linear(X, y)
        self.base_intercept_ = base["intercept"]
        self.linear_coef_ = base["coef"]

        residual = y - base["pred"]
        base_res_mse = float(np.mean(residual ** 2))

        corr_scores = np.array(
            [abs(float(np.corrcoef(X[:, j], residual)[0, 1])) if np.std(X[:, j]) > 1e-12 else 0.0 for j in range(self.n_features_in_)],
            dtype=float,
        )
        split_order = np.argsort(corr_scores)[::-1][: min(int(self.top_split_features), self.n_features_in_)]

        n = X.shape[0]
        min_leaf = max(12, int(self.min_leaf_frac * n))
        best = None

        for j in split_order:
            xj = X[:, j]
            for thr in np.unique(np.quantile(xj, self.split_quantiles)):
                left = xj <= thr
                right = ~left
                if int(left.sum()) < min_leaf or int(right.sum()) < min_leaf:
                    continue

                l0, l1, lmu = self._fit_affine_1d(xj[left], residual[left])
                r0, r1, rmu = self._fit_affine_1d(xj[right], residual[right])

                piece = np.empty(n, dtype=float)
                piece[left] = l0 + l1 * (xj[left] - lmu)
                piece[right] = r0 + r1 * (xj[right] - rmu)
                mse = float(np.mean((residual - piece) ** 2))
                gain = base_res_mse - mse

                if (best is None) or (gain > best["gain"]):
                    best = {
                        "feature": int(j),
                        "threshold": float(thr),
                        "left_offset": float(l0),
                        "left_slope": float(l1),
                        "left_mean": float(lmu),
                        "right_offset": float(r0),
                        "right_slope": float(r1),
                        "right_mean": float(rmu),
                        "gain": float(gain),
                    }

        self.has_piecewise_ = bool(best is not None and best["gain"] > float(self.min_split_gain) * (np.var(y) + 1e-12))
        if self.has_piecewise_:
            self.split_feature_ = best["feature"]
            self.split_threshold_ = best["threshold"]
            self.left_offset_ = best["left_offset"]
            self.left_slope_ = best["left_slope"]
            self.left_mean_ = best["left_mean"]
            self.right_offset_ = best["right_offset"]
            self.right_slope_ = best["right_slope"]
            self.right_mean_ = best["right_mean"]
        else:
            self.split_feature_ = -1
            self.split_threshold_ = 0.0
            self.left_offset_ = self.left_slope_ = self.left_mean_ = 0.0
            self.right_offset_ = self.right_slope_ = self.right_mean_ = 0.0

        self.linear_coef_[np.abs(self.linear_coef_) < self.coef_tol] = 0.0
        if abs(self.left_offset_) < self.coef_tol:
            self.left_offset_ = 0.0
        if abs(self.right_offset_) < self.coef_tol:
            self.right_offset_ = 0.0
        if abs(self.left_slope_) < self.coef_tol:
            self.left_slope_ = 0.0
        if abs(self.right_slope_) < self.coef_tol:
            self.right_slope_ = 0.0

        self.feature_importance_ = np.abs(self.linear_coef_)
        if self.has_piecewise_:
            j = self.split_feature_
            self.feature_importance_[j] += (
                abs(self.left_offset_) + abs(self.right_offset_) + abs(self.left_slope_) + abs(self.right_slope_)
            )
        self.selected_feature_order_ = np.argsort(self.feature_importance_)[::-1]
        return self

    def _piecewise_residual(self, X):
        if not self.has_piecewise_:
            return np.zeros(X.shape[0], dtype=float)
        xj = X[:, self.split_feature_]
        left = xj <= self.split_threshold_
        out = np.empty(X.shape[0], dtype=float)
        out[left] = self.left_offset_ + self.left_slope_ * (xj[left] - self.left_mean_)
        out[~left] = self.right_offset_ + self.right_slope_ * (xj[~left] - self.right_mean_)
        return out

    def predict(self, X):
        check_is_fitted(self, ["base_intercept_", "linear_coef_", "has_piecewise_"])
        X = self._impute(X)
        return self.base_intercept_ + X @ self.linear_coef_ + self._piecewise_residual(X)

    def __str__(self):
        check_is_fitted(self, ["base_intercept_", "linear_coef_", "has_piecewise_"])
        lines = [
            "BifurcatedSparseAffineRegressor",
            "Exact prediction equation:",
            f"  base = {self.base_intercept_:+.6f}",
        ]

        active = np.where(np.abs(self.linear_coef_) >= self.coef_tol)[0]
        for j in active[np.argsort(np.abs(self.linear_coef_[active]))[::-1]]:
            lines.append(f"       {self.linear_coef_[j]:+.6f} * x{j}")

        if self.has_piecewise_:
            j = self.split_feature_
            lines.append(
                f"  residual_rule: if x{j} <= {self.split_threshold_:+.6f}, add "
                f"({self.left_offset_:+.6f} + {self.left_slope_:+.6f}*(x{j} - {self.left_mean_:+.6f})); "
                f"else add ({self.right_offset_:+.6f} + {self.right_slope_:+.6f}*(x{j} - {self.right_mean_:+.6f}))."
            )
            lines.append("  prediction = base + residual_rule")
        else:
            lines.append("  residual_rule: none")
            lines.append("  prediction = base")

        if active.size == 0:
            lines.append("  (all linear terms are near zero)")

        lines.append("")
        lines.append("Feature summary (sorted by total attributed effect):")
        for j in self.selected_feature_order_[: min(12, self.n_features_in_)]:
            lines.append(
                f"  x{int(j)}: linear={self.linear_coef_[int(j)]:+.6f}, importance={self.feature_importance_[int(j)]:.6f}"
            )
        inactive = [f"x{j}" for j in range(self.n_features_in_) if self.feature_importance_[j] < self.coef_tol]
        if inactive:
            lines.append("Features with near-zero effect: " + ", ".join(inactive))
        lines.append("To predict, compute base first, then apply the single residual rule if present.")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
BifurcatedSparseAffineRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "BifurcatedSparseAffineV1"
model_description = "Sparse linear backbone plus one thresholded piecewise-affine residual correction on a single feature"
model_defs = [(model_shorthand_name, BifurcatedSparseAffineRegressor())]

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
