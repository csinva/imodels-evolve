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


class RobustAdaptiveLinearMapRegressor(BaseEstimator, RegressorMixin):
    """
    Robust adaptive linear regressor with explicit equation output.

    Pipeline:
    1) Pilot ridge solved from scratch via SVD with GCV alpha selection.
    2) Few IRLS robustification steps using Huber-style sample weights.
    3) Adaptive ridge penalty weighted by pilot coefficient magnitudes.
    4) Conservative sparsification + weighted least-squares refit on active set.

    This keeps a single simulatable linear equation while improving robustness.
    """

    def __init__(
        self,
        alpha_grid=(1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0),
        huber_k=1.5,
        irls_steps=3,
        adaptive_gamma=0.5,
        coef_display_threshold=0.03,
        coef_prune_threshold=0.015,
        min_active_features=3,
    ):
        self.alpha_grid = alpha_grid
        self.huber_k = huber_k
        self.irls_steps = irls_steps
        self.adaptive_gamma = adaptive_gamma
        self.coef_display_threshold = coef_display_threshold
        self.coef_prune_threshold = coef_prune_threshold
        self.min_active_features = min_active_features

    @staticmethod
    def _safe_std(X):
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std = np.where(std > 1e-12, std, 1.0)
        return mean.astype(float), std.astype(float)

    @staticmethod
    def _weighted_center(X, y, w):
        ws = float(np.sum(w))
        if ws <= 0:
            ws = float(len(w))
            w = np.ones_like(w)
        x_mu = (w[:, None] * X).sum(axis=0) / ws
        y_mu = float((w * y).sum() / ws)
        return x_mu, y_mu, ws

    def _pilot_gcv_ridge(self, X, y):
        x_mean, x_std = self._safe_std(X)
        Xz = (X - x_mean) / x_std
        y_mean = float(np.mean(y))
        yc = y - y_mean

        U, s, Vt = np.linalg.svd(Xz, full_matrices=False)
        Uy = U.T @ yc
        s2 = s * s
        n = Xz.shape[0]

        best = None
        for alpha in self.alpha_grid:
            a = float(alpha)
            shrink = s / (s2 + a)
            beta_z = Vt.T @ (shrink * Uy)
            fit_c = U @ ((s2 / (s2 + a)) * Uy)
            resid = yc - fit_c
            tr_h = float(np.sum(s2 / (s2 + a)))
            denom = max(n - tr_h, 1e-8)
            gcv = float(np.sum(resid * resid) / (denom * denom))
            if best is None or gcv < best["gcv"]:
                best = {"alpha": a, "coef_z": beta_z, "gcv": gcv}

        beta_raw = best["coef_z"] / x_std
        intercept = y_mean - float(np.dot(beta_raw, x_mean))
        return {
            "alpha": float(best["alpha"]),
            "coef": np.asarray(beta_raw, dtype=float),
            "intercept": float(intercept),
            "x_mean": x_mean,
            "x_std": x_std,
            "gcv": float(best["gcv"]),
        }

    def _weighted_adaptive_ridge(self, X, y, w, penalty_diag, alpha):
        x_mu, y_mu, ws = self._weighted_center(X, y, w)
        Xc = X - x_mu
        yc = y - y_mu

        Xw = Xc * np.sqrt(w)[:, None]
        yw = yc * np.sqrt(w)

        p = X.shape[1]
        A = Xw.T @ Xw + float(alpha) * np.diag(penalty_diag)
        b = Xw.T @ yw

        beta = np.linalg.solve(A + 1e-10 * np.eye(p), b)
        intercept = float(y_mu - np.dot(x_mu, beta))
        return intercept, beta

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        self.n_features_in_ = p

        pilot = self._pilot_gcv_ridge(X, y)
        alpha = float(pilot["alpha"])
        beta = np.asarray(pilot["coef"], dtype=float)
        intercept = float(pilot["intercept"])

        resid0 = y - (intercept + X @ beta)
        mad = float(np.median(np.abs(resid0 - np.median(resid0))))
        robust_scale = max(1.4826 * mad, 1e-6)

        for _ in range(max(int(self.irls_steps), 1)):
            resid = y - (intercept + X @ beta)
            r_scaled = np.abs(resid) / robust_scale
            w = np.where(r_scaled <= self.huber_k, 1.0, self.huber_k / np.maximum(r_scaled, 1e-12))

            adapt = 1.0 / np.power(np.abs(beta) + 1e-4, self.adaptive_gamma)
            adapt = adapt / max(float(np.median(adapt)), 1e-8)

            intercept, beta = self._weighted_adaptive_ridge(X, y, w, adapt, alpha)

        abs_b = np.abs(beta)
        max_b = float(np.max(abs_b)) if p > 0 else 0.0
        prune_cut = float(self.coef_prune_threshold) * max(max_b, 1e-12)
        active = np.where(abs_b >= prune_cut)[0]

        min_keep = min(max(int(self.min_active_features), 1), p)
        if len(active) < min_keep:
            top_idx = np.argsort(-abs_b)[:min_keep]
            active = np.array(sorted(set(active.tolist() + top_idx.tolist())), dtype=int)

        if len(active) > 0:
            Xa = X[:, active]
            x_mu, y_mu, _ = self._weighted_center(Xa, y, w)
            Xac = Xa - x_mu
            yc = y - y_mu
            Xaw = Xac * np.sqrt(w)[:, None]
            yw = yc * np.sqrt(w)
            A = Xaw.T @ Xaw + 1e-8 * np.eye(len(active))
            b = Xaw.T @ yw
            beta_a = np.linalg.solve(A, b)
            intercept = float(y_mu - np.dot(x_mu, beta_a))
            beta_full = np.zeros(p, dtype=float)
            beta_full[active] = beta_a
            beta = beta_full

        self.intercept_ = float(intercept)
        self.coef_ = np.asarray(beta, dtype=float)
        self.alpha_ = float(alpha)
        self.gcv_score_ = float(pilot["gcv"])
        self.robust_scale_ = float(robust_scale)

        abs_coef = np.abs(self.coef_)
        mean_abs = float(np.mean(abs_coef)) if p > 0 else 0.0
        disp_cut = float(self.coef_display_threshold) * max(mean_abs, 1e-12)
        self.meaningful_features_ = np.where(abs_coef >= disp_cut)[0].astype(int)
        self.negligible_features_ = np.where(abs_coef < disp_cut)[0].astype(int)
        self.active_features_ = np.where(abs_coef > 0)[0].astype(int)
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "coef_"])
        X = np.asarray(X, dtype=float)
        return np.asarray(self.intercept_ + X @ self.coef_, dtype=float)

    def __str__(self):
        check_is_fitted(self, ["intercept_", "coef_", "alpha_"])
        lines = [
            "Robust Adaptive Linear Map Regressor",
            f"pilot_ridge_alpha={self.alpha_:.6g} (selected by generalized cross-validation)",
            f"robust_scale={self.robust_scale_:.6f} (Huber-style residual weighting)",
            "Prediction uses this exact equation:",
        ]

        equation = f"y = {self.intercept_:+.6f}"
        for j, c in enumerate(self.coef_):
            if abs(float(c)) > 0:
                equation += f" {float(c):+.6f}*x{int(j)}"
        lines.append(equation)

        lines.append("Feature coefficients (largest absolute effect first):")
        ranked = sorted(enumerate(self.coef_), key=lambda t: -abs(float(t[1])))
        for j, c in ranked:
            lines.append(f"  x{int(j)}: {float(c):+.6f}")

        if len(self.active_features_) > 0:
            lines.append("Active features in the equation: " + ", ".join(f"x{int(j)}" for j in self.active_features_))
        if len(self.negligible_features_) > 0:
            lines.append(
                "Features with negligible or zero effect in this fitted equation: "
                + ", ".join(f"x{int(j)}" for j in self.negligible_features_)
            )

        ops = 1 + len(self.active_features_)
        lines.append(f"Approx arithmetic operations: {ops}")
        lines.append(f"Pilot GCV objective: {self.gcv_score_:.6f}")
        return "\n".join(lines)


import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
RobustAdaptiveLinearMapRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "RobustAdaptiveLinearMapV1"
model_description = "From-scratch robust adaptive linear map: SVD-GCV pilot ridge, Huber-style IRLS weighting, adaptive ridge penalties, and conservative sparse weighted refit into an explicit raw-feature equation"
model_defs = [(model_shorthand_name, RobustAdaptiveLinearMapRegressor())]


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
