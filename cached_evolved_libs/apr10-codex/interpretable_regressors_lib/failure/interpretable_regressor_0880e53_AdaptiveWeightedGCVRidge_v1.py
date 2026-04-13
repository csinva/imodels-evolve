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


class AdaptiveWeightedGCVRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Custom interpretable linear equation model.

    Steps:
      1) Robust-standardize features.
      2) Build adaptive feature penalties from |corr(x_j, y)| (less shrinkage for
         more predictive features).
      3) Solve weighted ridge in closed form with SVD and choose alpha by GCV.

    Final prediction is a transparent linear equation on original features.
    """

    def __init__(
        self,
        alpha_grid_size=31,
        alpha_min_log10=-7.0,
        alpha_max_log10=4.0,
        corr_floor=0.03,
        corr_power=0.7,
        max_display_terms=12,
    ):
        self.alpha_grid_size = alpha_grid_size
        self.alpha_min_log10 = alpha_min_log10
        self.alpha_max_log10 = alpha_max_log10
        self.corr_floor = corr_floor
        self.corr_power = corr_power
        self.max_display_terms = max_display_terms

    @staticmethod
    def _robust_standardize(X):
        med = np.median(X, axis=0)
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        iqr = q3 - q1
        scale = np.where(iqr < 1e-8, 1.0, iqr / 1.349)
        Z = (X - med) / scale
        return Z, med, scale

    @staticmethod
    def _safe_corr(x, y):
        xc = x - np.mean(x)
        yc = y - np.mean(y)
        den = np.sqrt(np.sum(xc * xc) * np.sum(yc * yc))
        if den <= 1e-12:
            return 0.0
        return float((xc @ yc) / den)

    def _solve_weighted_ridge_gcv(self, Z, y, penalty_weights):
        n, p = Z.shape
        if n == 0:
            return {
                "coef_z": np.zeros(p, dtype=float),
                "intercept": 0.0,
                "alpha": 1.0,
                "pred": np.zeros(0, dtype=float),
                "gcv": 0.0,
            }

        # Weighted ridge can be solved as ordinary ridge on transformed columns.
        w = np.asarray(penalty_weights, dtype=float)
        w = np.where(w < 1e-8, 1e-8, w)
        Zw = Z / w[np.newaxis, :]

        x_mean = np.mean(Zw, axis=0)
        y_mean = float(np.mean(y))
        Xc = Zw - x_mean
        yc = y - y_mean

        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        s2 = s * s
        uty = U.T @ yc

        alphas = np.logspace(
            float(self.alpha_min_log10),
            float(self.alpha_max_log10),
            num=int(max(7, self.alpha_grid_size)),
        )

        best = None
        for alpha in alphas:
            shrink = s / (s2 + alpha)
            coef_w = Vt.T @ (shrink * uty)
            intercept = y_mean - float(x_mean @ coef_w)
            pred = Zw @ coef_w + intercept

            mse = float(np.mean((y - pred) ** 2))
            df = float(np.sum(s2 / (s2 + alpha)))
            denom = max(1e-8, 1.0 - df / max(1, n))
            gcv = mse / (denom * denom)

            cur = {
                "coef_z": coef_w / w,
                "intercept": intercept,
                "alpha": float(alpha),
                "pred": pred,
                "gcv": gcv,
            }
            if best is None or cur["gcv"] < best["gcv"]:
                best = cur
        return best

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape

        self.n_features_in_ = int(p)
        self.feature_names_ = [f"x{j}" for j in range(p)]

        if n == 0:
            self.center_ = np.zeros(p, dtype=float)
            self.scale_ = np.ones(p, dtype=float)
            self.penalty_weights_ = np.ones(p, dtype=float)
            self.coef_z_ = np.zeros(p, dtype=float)
            self.coef_ = np.zeros(p, dtype=float)
            self.intercept_ = float(np.mean(y)) if y.size else 0.0
            self.alpha_ = 1.0
            self.training_mse_ = 0.0
            self.is_fitted_ = True
            return self

        Z, center, scale = self._robust_standardize(X)
        self.center_ = center
        self.scale_ = scale

        corr_abs = np.zeros(p, dtype=float)
        for j in range(p):
            corr_abs[j] = abs(self._safe_corr(Z[:, j], y))

        # Higher |corr| -> smaller penalty weight.
        weight_base = np.maximum(float(self.corr_floor), corr_abs)
        penalty_weights = 1.0 / np.power(weight_base, float(self.corr_power))
        penalty_weights = penalty_weights / np.median(penalty_weights)
        self.penalty_weights_ = penalty_weights

        sol = self._solve_weighted_ridge_gcv(Z, y, penalty_weights)
        self.coef_z_ = sol["coef_z"]
        self.alpha_ = sol["alpha"]
        self.training_mse_ = float(np.mean((y - sol["pred"]) ** 2))

        # Convert from standardized equation y = b + sum coef_z_j * z_j
        # to original coordinates y = intercept + X @ coef.
        self.coef_ = self.coef_z_ / self.scale_
        self.intercept_ = float(sol["intercept"] - np.sum(self.coef_z_ * (self.center_ / self.scale_)))

        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_

    def __str__(self):
        check_is_fitted(self, "is_fitted_")
        lines = ["Adaptive Weighted GCV Ridge Regressor:"]
        lines.append("  prediction = intercept + sum_j coef_j * x_j")
        lines.append("  coefficients estimated by weighted ridge with GCV alpha selection")
        lines.append("  adaptive penalty: penalty_j proportional to 1 / max(|corr(x_j,y)|, floor)^power")
        lines.append(f"  chosen alpha: {self.alpha_:.4g}")
        lines.append(f"  training MSE: {self.training_mse_:.6f}")
        lines.append(f"  intercept: {self.intercept_:+.6f}")

        max_terms = int(max(1, self.max_display_terms))
        order = np.argsort(np.abs(self.coef_))[::-1][:max_terms]
        lines.append("  strongest terms in original-feature equation:")
        for j in order:
            lines.append(
                f"    x{int(j)}: coef={self.coef_[j]:+.6f}  (adaptive_penalty_weight={self.penalty_weights_[j]:.3f})"
            )
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
AdaptiveWeightedGCVRidgeRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "AdaptiveWeightedGCVRidge_v1"
model_description = "Custom linear equation model with robust scaling, correlation-adaptive ridge penalties, and SVD/GCV alpha selection"
model_defs = [(model_shorthand_name, AdaptiveWeightedGCVRidgeRegressor())]


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

    std_tests = {t.__name__ for t in ALL_TESTS}
    hard_tests = {t.__name__ for t in HARD_TESTS}
    insight_tests = {t.__name__ for t in INSIGHT_TESTS}
    std_passed = sum(r["passed"] for r in interp_results if r["test"] in std_tests)
    hard_passed = sum(r["passed"] for r in interp_results if r["test"] in hard_tests)
    insight_passed = sum(r["passed"] for r in interp_results if r["test"] in insight_tests)
    print(f"[std {std_passed}/{len(std_tests)}  hard {hard_passed}/{len(hard_tests)}  insight {insight_passed}/{len(insight_tests)}]")
    print(f"total_seconds: {time.time() - t0:.1f}s")
