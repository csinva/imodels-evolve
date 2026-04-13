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


class ResidualTanhRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Two-view transparent equation:
      1) dense ridge on standardized raw features
      2) residual ridge on smooth tanh-transformed features
      3) clipped mixing weight for the nonlinear residual component
    """

    def __init__(
        self,
        alpha_grid_size=19,
        linear_alpha_min_log10=-6.0,
        linear_alpha_max_log10=3.0,
        nonlin_alpha_min_log10=-3.0,
        nonlin_alpha_max_log10=4.0,
        tanh_temperature=1.8,
        max_nonlin_weight=0.45,
        display_top_k=8,
    ):
        self.alpha_grid_size = alpha_grid_size
        self.linear_alpha_min_log10 = linear_alpha_min_log10
        self.linear_alpha_max_log10 = linear_alpha_max_log10
        self.nonlin_alpha_min_log10 = nonlin_alpha_min_log10
        self.nonlin_alpha_max_log10 = nonlin_alpha_max_log10
        self.tanh_temperature = tanh_temperature
        self.max_nonlin_weight = max_nonlin_weight
        self.display_top_k = display_top_k

    @staticmethod
    def _standardize(X):
        center = np.mean(X, axis=0)
        scale = np.std(X, axis=0)
        scale = np.where(scale < 1e-8, 1.0, scale)
        Z = (X - center) / scale
        return Z, center, scale

    @staticmethod
    def _ridge_gcv(Z, y, alpha_min_log10, alpha_max_log10, alpha_grid_size):
        n, p = Z.shape
        if n == 0:
            return {
                "coef": np.zeros(p, dtype=float),
                "intercept": 0.0,
                "pred": np.zeros(0, dtype=float),
                "alpha": 1.0,
                "gcv": 0.0,
            }
        if p == 0:
            y_mean = float(np.mean(y))
            pred = np.full(n, y_mean, dtype=float)
            mse = float(np.mean((y - pred) ** 2))
            return {
                "coef": np.zeros(0, dtype=float),
                "intercept": y_mean,
                "pred": pred,
                "alpha": 1.0,
                "gcv": mse,
            }

        z_mean = np.mean(Z, axis=0)
        y_mean = float(np.mean(y))
        Zc = Z - z_mean
        yc = y - y_mean

        U, s, Vt = np.linalg.svd(Zc, full_matrices=False)
        s2 = s * s
        uty = U.T @ yc
        alphas = np.logspace(
            float(alpha_min_log10),
            float(alpha_max_log10),
            num=int(max(5, alpha_grid_size)),
        )

        best = None
        for alpha in alphas:
            shrink = s / (s2 + alpha)
            coef = Vt.T @ (shrink * uty)
            intercept = y_mean - float(z_mean @ coef)
            pred = Z @ coef + intercept
            resid = y - pred
            mse = float(np.mean(resid * resid))
            df = float(np.sum(s2 / (s2 + alpha)))
            denom = max(1e-8, 1.0 - (df / max(n, 1)))
            gcv = mse / (denom * denom)
            cur = {
                "coef": coef,
                "intercept": intercept,
                "pred": pred,
                "alpha": float(alpha),
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
            self.coef_linear_ = np.zeros(p, dtype=float)
            self.coef_nonlinear_ = np.zeros(p, dtype=float)
            self.intercept_linear_ = float(np.mean(y)) if y.size else 0.0
            self.intercept_nonlinear_ = 0.0
            self.nonlinear_weight_ = 0.0
            self.alpha_linear_ = 1.0
            self.alpha_nonlinear_ = 1.0
            self.training_mse_ = 0.0
            self.is_fitted_ = True
            return self

        Z, center, scale = self._standardize(X)
        self.center_ = center
        self.scale_ = scale

        lin = self._ridge_gcv(
            Z,
            y,
            self.linear_alpha_min_log10,
            self.linear_alpha_max_log10,
            self.alpha_grid_size,
        )
        residual = y - lin["pred"]
        temp = float(max(0.25, self.tanh_temperature))
        Z_nl = np.tanh(Z / temp)
        nl = self._ridge_gcv(
            Z_nl,
            residual,
            self.nonlin_alpha_min_log10,
            self.nonlin_alpha_max_log10,
            self.alpha_grid_size,
        )

        residual_pred = nl["pred"]
        denom = float(np.dot(residual_pred, residual_pred) + 1e-12)
        raw_weight = float(np.dot(residual_pred, residual) / denom)
        max_w = float(max(0.0, self.max_nonlin_weight))
        nonlin_weight = float(np.clip(raw_weight, 0.0, max_w))

        pred = lin["pred"] + nonlin_weight * residual_pred
        self.coef_linear_ = lin["coef"]
        self.intercept_linear_ = float(lin["intercept"])
        self.alpha_linear_ = float(lin["alpha"])
        self.coef_nonlinear_ = nl["coef"]
        self.intercept_nonlinear_ = float(nl["intercept"])
        self.alpha_nonlinear_ = float(nl["alpha"])
        self.nonlinear_weight_ = nonlin_weight
        self.training_mse_ = float(np.mean((y - pred) ** 2))
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        X = np.asarray(X, dtype=float)
        Z = (X - self.center_) / self.scale_
        linear_part = Z @ self.coef_linear_ + self.intercept_linear_
        temp = float(max(0.25, self.tanh_temperature))
        Z_nl = np.tanh(Z / temp)
        nonlinear_part = Z_nl @ self.coef_nonlinear_ + self.intercept_nonlinear_
        return linear_part + self.nonlinear_weight_ * nonlinear_part

    def __str__(self):
        check_is_fitted(self, "is_fitted_")
        lines = ["Residual-Tanh Ridge Equation Regressor:"]
        lines.append(
            "  prediction = linear_ridge(z) + w * residual_ridge(tanh(z / temperature))"
        )
        lines.append(f"  linear alpha (GCV): {self.alpha_linear_:.4g}")
        lines.append(f"  residual alpha (GCV): {self.alpha_nonlinear_:.4g}")
        lines.append(f"  nonlinear weight w: {self.nonlinear_weight_:.4f}")
        lines.append(f"  training MSE: {self.training_mse_:.6f}")

        top_k = int(max(1, self.display_top_k))
        if self.coef_linear_.size:
            lines.append(f"  top linear terms (|coef|): {top_k}")
            order_lin = np.argsort(np.abs(self.coef_linear_))[::-1][:top_k]
            for idx in order_lin:
                lines.append(f"    {self.feature_names_[idx]}: {self.coef_linear_[idx]:+.4f}")
        if self.coef_nonlinear_.size:
            lines.append(f"  top nonlinear tanh terms (|coef|): {top_k}")
            order_nl = np.argsort(np.abs(self.coef_nonlinear_))[::-1][:top_k]
            for idx in order_nl:
                lines.append(
                    f"    tanh({self.feature_names_[idx]}/temp): {self.coef_nonlinear_[idx]:+.4f}"
                )
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
ResidualTanhRidgeRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "ResidualTanhRidge_v1"
model_description = "Two-view equation model: dense GCV ridge on standardized raw features plus clipped-weight residual ridge on tanh feature maps"
model_defs = [(model_shorthand_name, ResidualTanhRidgeRegressor())]


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
