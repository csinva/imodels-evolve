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


class SparseSingleIndexSplineRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse single-index regressor with a compact piecewise-linear calibration.

    Steps:
    1) Standardize features and fit ridge with GCV alpha selection.
    2) Keep only influential features, then refit ridge on that active subset.
    3) Build a one-dimensional score s = b + w^T x.
    4) Learn a tiny two-knot hinge calibration y ~= a0 + a1*s + a2*(s-t1)+ + a3*(s-t2)+.
    """

    def __init__(
        self,
        alpha_grid=(0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0),
        max_active_features=14,
        min_effect_ratio=0.02,
        calibration_alpha=0.03,
        calibration_quantiles=(0.33, 0.66),
        min_rel_gain=0.002,
    ):
        self.alpha_grid = alpha_grid
        self.max_active_features = max_active_features
        self.min_effect_ratio = min_effect_ratio
        self.calibration_alpha = calibration_alpha
        self.calibration_quantiles = calibration_quantiles
        self.min_rel_gain = min_rel_gain

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

    def _select_alpha_gcv(self, Z, yc):
        n, p = Z.shape
        XtX = Z.T @ Z
        eigvals = np.linalg.eigvalsh(XtX)
        best = (float("inf"), float(self.alpha_grid[0]))
        for alpha in self.alpha_grid:
            _, w = self._ridge_fit(Z, yc, alpha)
            resid = yc - Z @ w
            rss = float(np.dot(resid, resid))
            df = float(np.sum(eigvals / (eigvals + float(alpha))))
            denom = (n - df) ** 2
            if denom < 1e-9:
                continue
            gcv = rss / denom
            if gcv < best[0]:
                best = (gcv, float(alpha))
        return best[1]

    @staticmethod
    def _hinge(v, t):
        return np.maximum(0.0, v - t)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        self.n_features_in_ = p

        x_mean = X.mean(axis=0)
        x_std = X.std(axis=0)
        x_std = np.where(x_std < 1e-8, 1.0, x_std)
        Z = (X - x_mean) / x_std

        y_mean = float(y.mean())
        yc = y - y_mean

        alpha = self._select_alpha_gcv(Z, yc)
        _, w0 = self._ridge_fit(Z, yc, alpha)

        effect = np.abs(w0)
        eff_max = max(float(effect.max()), 1e-12)
        active = np.where(effect >= float(self.min_effect_ratio) * eff_max)[0]
        if len(active) == 0:
            active = np.array([int(np.argmax(effect))], dtype=int)
        if len(active) > int(self.max_active_features):
            keep = np.argsort(-effect[active])[: int(self.max_active_features)]
            active = np.sort(active[keep])

        b_active, w_active = self._ridge_fit(Z[:, active], yc, alpha)
        coef_raw = np.zeros(p, dtype=float)
        coef_raw[active] = w_active / x_std[active]
        intercept_raw = float(y_mean + b_active - np.dot(coef_raw, x_mean))

        s = intercept_raw + X @ coef_raw
        t1 = float(np.quantile(s, float(self.calibration_quantiles[0])))
        t2 = float(np.quantile(s, float(self.calibration_quantiles[1])))
        C = np.column_stack([np.ones(n), s, self._hinge(s, t1), self._hinge(s, t2)])
        reg = np.diag([0.0, float(self.calibration_alpha), float(self.calibration_alpha), float(self.calibration_alpha)])
        try:
            gamma = np.linalg.solve(C.T @ C + reg, C.T @ y)
        except np.linalg.LinAlgError:
            gamma = np.linalg.pinv(C.T @ C + reg) @ (C.T @ y)

        pred_linear = s
        pred_cal = C @ gamma
        mse_linear = float(np.mean((y - pred_linear) ** 2))
        mse_cal = float(np.mean((y - pred_cal) ** 2))
        rel_gain = (mse_linear - mse_cal) / max(mse_linear, 1e-12)

        if rel_gain < float(self.min_rel_gain):
            gamma = np.array([0.0, 1.0, 0.0, 0.0], dtype=float)
            mse_cal = mse_linear

        raw_effect = np.abs(coef_raw) * X.std(axis=0)
        raw_max = max(float(raw_effect.max()), 1e-12)
        meaningful = np.where(raw_effect > 0.03 * raw_max)[0]
        negligible = np.where(raw_effect <= 0.03 * raw_max)[0]

        self.intercept_ = intercept_raw
        self.coef_ = coef_raw
        self.active_features_ = np.asarray(active, dtype=int)
        self.alpha_ = float(alpha)
        self.score_thresholds_ = np.array([t1, t2], dtype=float)
        self.calibration_coefs_ = np.asarray(gamma, dtype=float)
        self.meaningful_features_ = np.asarray(meaningful, dtype=int)
        self.negligible_features_ = np.asarray(negligible, dtype=int)
        self.fit_mse_linear_ = mse_linear
        self.fit_mse_final_ = mse_cal
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "coef_", "score_thresholds_", "calibration_coefs_"])
        X = np.asarray(X, dtype=float)
        s = self.intercept_ + X @ self.coef_
        t1, t2 = self.score_thresholds_
        g0, g1, g2, g3 = self.calibration_coefs_
        return g0 + g1 * s + g2 * self._hinge(s, t1) + g3 * self._hinge(s, t2)

    def __str__(self):
        check_is_fitted(self, ["intercept_", "coef_", "score_thresholds_", "calibration_coefs_"])
        order = np.argsort(-np.abs(self.coef_))
        shown = [j for j in order if abs(float(self.coef_[j])) > 1e-12]

        score_eq = f"s = {float(self.intercept_):.6f}"
        for j in shown:
            score_eq += f" {float(self.coef_[j]):+.6f}*x{int(j)}"

        t1, t2 = self.score_thresholds_
        g0, g1, g2, g3 = self.calibration_coefs_
        pred_eq = (
            f"y = {float(g0):+.6f} {float(g1):+.6f}*s "
            f"{float(g2):+.6f}*max(0, s-{float(t1):.6f}) "
            f"{float(g3):+.6f}*max(0, s-{float(t2):.6f})"
        )

        lines = [
            "Sparse Single-Index Spline Regressor",
            f"Selected ridge alpha (GCV): {self.alpha_:.6g}",
            "Step 1 (linear score):",
            score_eq,
            "Step 2 (score calibration):",
            pred_eq,
            "Active linear features (sorted by |coefficient|):",
        ]
        for j in shown:
            lines.append(f"  x{int(j)}: coef={float(self.coef_[j]):+.6f}")

        meaningful = ", ".join(f"x{int(j)}" for j in self.meaningful_features_) or "(none)"
        negligible = ", ".join(f"x{int(j)}" for j in self.negligible_features_) or "(none)"
        lines.append(f"Meaningful features: {meaningful}")
        lines.append(f"Negligible features: {negligible}")
        lines.append(f"Training MSE linear -> final: {self.fit_mse_linear_:.6f} -> {self.fit_mse_final_:.6f}")
        ops = 2 * len(shown) + 8
        lines.append(f"Approx operations to evaluate: {ops}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseSingleIndexSplineRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseSingleIndexSplineV1"
model_description = "GCV-selected sparse ridge single-index score with a compact two-knot piecewise-linear calibration map for lightweight nonlinearity"
model_defs = [(model_shorthand_name, SparseSingleIndexSplineRegressor())]

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
