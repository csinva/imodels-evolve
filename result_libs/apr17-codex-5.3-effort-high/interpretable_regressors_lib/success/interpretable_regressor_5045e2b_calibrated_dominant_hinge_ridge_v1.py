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


class CalibratedDominantHingeRidgeV1(BaseEstimator, RegressorMixin):
    """
    From-scratch interpretable regressor:
    1) closed-form ridge regression on standardized features (alpha by GCV)
    2) optional 2-knot hinge correction on the single dominant feature
       retained only if train MSE improves materially.
    """

    def __init__(
        self,
        alpha_grid=None,
        hinge_ridge=0.5,
        min_relative_gain=0.03,
        eps=1e-12,
    ):
        self.alpha_grid = alpha_grid
        self.hinge_ridge = hinge_ridge
        self.min_relative_gain = min_relative_gain
        self.eps = eps

    @staticmethod
    def _safe_scale(x):
        s = np.std(x, axis=0)
        s = np.where(s > 1e-12, s, 1.0)
        return s

    def _solve_ridge_gcv(self, Xz, yc):
        n, _ = Xz.shape
        U, sing, Vt = np.linalg.svd(Xz, full_matrices=False)
        UTy = U.T @ yc
        s2 = sing ** 2

        if self.alpha_grid is None:
            alphas = np.logspace(-4, 3, 18)
        else:
            alphas = np.asarray(self.alpha_grid, dtype=float)

        best = None
        for alpha in alphas:
            filt = sing / (s2 + alpha)
            beta = Vt.T @ (filt * UTy)
            yhat = Xz @ beta
            resid = yc - yhat
            num = float(np.mean(resid ** 2))

            hat_trace = float(np.sum(s2 / (s2 + alpha)))
            denom = (1.0 - hat_trace / max(n, 1)) ** 2
            gcv = num / max(denom, self.eps)
            if (best is None) or (gcv < best[0]):
                best = (gcv, float(alpha), beta)

        _, alpha_best, beta_best = best
        return alpha_best, beta_best

    @staticmethod
    def _hinge_basis(x, t1, t2):
        return np.column_stack([np.maximum(0.0, x - t1), np.maximum(0.0, x - t2)])

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape

        self.n_features_in_ = p
        self.x_mean_ = np.mean(X, axis=0)
        self.x_scale_ = self._safe_scale(X)
        Xz = (X - self.x_mean_) / self.x_scale_

        self.y_mean_ = float(np.mean(y))
        yc = y - self.y_mean_

        alpha, beta_z = self._solve_ridge_gcv(Xz, yc)
        self.alpha_ = float(alpha)
        self.coef_ = beta_z / self.x_scale_
        self.intercept_ = float(self.y_mean_ - np.dot(self.coef_, self.x_mean_))

        linear_pred = self.intercept_ + X @ self.coef_
        base_mse = float(np.mean((y - linear_pred) ** 2))

        dom = int(np.argmax(np.abs(self.coef_)))
        x_dom = X[:, dom]
        q1, q2 = np.quantile(x_dom, [0.33, 0.66])

        H = self._hinge_basis(x_dom, float(q1), float(q2))
        HtH = H.T @ H + float(self.hinge_ridge) * np.eye(2)
        rhs = H.T @ (y - linear_pred)
        gamma = np.linalg.solve(HtH, rhs)

        corr = H @ gamma
        full_pred = linear_pred + corr
        corr_mse = float(np.mean((y - full_pred) ** 2))

        rel_gain = (base_mse - corr_mse) / max(base_mse, self.eps)
        use_hinge = bool(rel_gain >= float(self.min_relative_gain))

        self.dominant_feature_ = dom
        self.hinge_knots_ = (float(q1), float(q2))
        self.hinge_coef_ = gamma if use_hinge else np.zeros(2, dtype=float)
        self.use_hinge_ = use_hinge
        self.base_train_mse_ = base_mse
        self.final_train_mse_ = corr_mse if use_hinge else base_mse
        self.relative_gain_ = float(rel_gain) if use_hinge else 0.0

        magnitudes = np.abs(self.coef_)
        cutoff = max(1e-6, np.quantile(magnitudes, 0.6))
        self.meaningful_features_ = np.where(magnitudes >= cutoff)[0].astype(int)
        self.negligible_features_ = np.where(magnitudes < cutoff)[0].astype(int)
        return self

    def predict(self, X):
        check_is_fitted(self, ["coef_", "intercept_", "dominant_feature_", "hinge_knots_", "hinge_coef_"])
        X = np.asarray(X, dtype=float)
        out = self.intercept_ + X @ self.coef_
        if self.use_hinge_:
            j = self.dominant_feature_
            t1, t2 = self.hinge_knots_
            H = self._hinge_basis(X[:, j], t1, t2)
            out = out + H @ self.hinge_coef_
        return out

    def __str__(self):
        check_is_fitted(self, ["coef_", "intercept_", "dominant_feature_", "hinge_knots_", "hinge_coef_"])
        lines = [
            "Calibrated Dominant-Hinge Ridge Regressor",
            "Exact prediction recipe:",
            "  1) Start with linear map: y_lin = intercept + sum_j (coef_j * xj)",
            f"  2) intercept = {self.intercept_:+.6f}",
            f"  3) ridge alpha selected by GCV = {self.alpha_:.6g}",
            "",
            "Linear coefficients (raw-feature equation):",
        ]

        order = np.argsort(-np.abs(self.coef_))
        for j in order:
            lines.append(f"  x{int(j)}: {float(self.coef_[j]):+.6f}")

        if self.use_hinge_:
            j = int(self.dominant_feature_)
            t1, t2 = self.hinge_knots_
            g1, g2 = float(self.hinge_coef_[0]), float(self.hinge_coef_[1])
            lines.extend([
                "",
                "Dominant-feature hinge correction (additive):",
                f"  dominant feature: x{j}",
                f"  correction = ({g1:+.6f}) * max(0, x{j} - ({t1:+.6f})) + ({g2:+.6f}) * max(0, x{j} - ({t2:+.6f}))",
                "  final prediction: y = y_lin + correction",
            ])
        else:
            lines.extend([
                "",
                "No hinge correction retained (linear map already sufficient on training data).",
                "Final prediction: y = y_lin",
            ])

        if len(self.meaningful_features_) > 0:
            lines.append("Meaningful features by coefficient magnitude: " + ", ".join(f"x{int(i)}" for i in self.meaningful_features_))
        if len(self.negligible_features_) > 0:
            lines.append("Negligible features by coefficient magnitude: " + ", ".join(f"x{int(i)}" for i in self.negligible_features_))

        lines.append(f"Train MSE: {self.final_train_mse_:.6f}")
        if self.use_hinge_:
            lines.append(f"Relative train MSE gain from hinge correction: {self.relative_gain_:.2%}")
        return "\n".join(lines)


import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
CalibratedDominantHingeRidgeV1.__module__ = "interpretable_regressor"

model_shorthand_name = "CalibratedDominantHingeRidgeV1"
model_description = "From-scratch GCV ridge raw-feature equation with optional two-knot hinge calibration on the dominant feature, retained only when it yields meaningful fit gain"
model_defs = [(model_shorthand_name, CalibratedDominantHingeRidgeV1())]

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
