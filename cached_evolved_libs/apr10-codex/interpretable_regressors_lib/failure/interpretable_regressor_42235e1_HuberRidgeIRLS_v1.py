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


class HuberRidgeIRLSRegressor(BaseEstimator, RegressorMixin):
    """
    Transparent robust linear equation:
      1) robust feature scaling via median and MAD
      2) GCV-selected ridge fit on standardized features
      3) a few Huber-style IRLS refinement steps for outlier-robust residuals
    """

    def __init__(
        self,
        alpha_grid_size=17,
        alpha_min_log10=-5.0,
        alpha_max_log10=3.0,
        huber_delta=1.35,
        irls_steps=4,
        display_top_k=10,
        clip_value=5.0,
    ):
        self.alpha_grid_size = alpha_grid_size
        self.alpha_min_log10 = alpha_min_log10
        self.alpha_max_log10 = alpha_max_log10
        self.huber_delta = huber_delta
        self.irls_steps = irls_steps
        self.display_top_k = display_top_k
        self.clip_value = clip_value

    @staticmethod
    def _robust_standardize(X):
        med = np.median(X, axis=0)
        mad = np.median(np.abs(X - med), axis=0)
        scale = 1.4826 * mad
        scale = np.where(scale < 1e-8, np.std(X, axis=0), scale)
        scale = np.where(scale < 1e-8, 1.0, scale)
        Z = (X - med) / scale
        return Z, med, scale

    def _ridge_gcv(self, Z, y, sample_weight=None):
        n = Z.shape[0]
        if Z.shape[1] == 0:
            pred = np.full(n, np.mean(y) if sample_weight is None else np.average(y, weights=sample_weight), dtype=float)
            return {
                "coef": np.zeros(0, dtype=float),
                "intercept": float(np.mean(pred)),
                "pred": pred,
                "alpha": 0.0,
                "gcv": float(np.mean((y - pred) ** 2)),
                "mse": float(np.mean((y - pred) ** 2)),
                "df": 0.0,
            }

        if sample_weight is None:
            w = np.ones(n, dtype=float)
        else:
            w = np.asarray(sample_weight, dtype=float).ravel()
            w = np.clip(w, 1e-8, np.inf)
        w = w / np.mean(w)
        sqrt_w = np.sqrt(w)

        z_mean = np.average(Z, axis=0, weights=w)
        y_mean = float(np.average(y, weights=w))
        Zc = Z - z_mean
        yc = y - y_mean

        Zw = Zc * sqrt_w[:, None]
        yw = yc * sqrt_w

        U, s, Vt = np.linalg.svd(Zw, full_matrices=False)
        uty = U.T @ yw
        s2 = s * s

        a_lo = float(self.alpha_min_log10)
        a_hi = float(self.alpha_max_log10)
        grid_n = int(max(5, self.alpha_grid_size))
        alphas = np.logspace(a_lo, a_hi, num=grid_n)

        best = None
        for a in alphas:
            shrink = s / (s2 + a)
            coef = Vt.T @ (shrink * uty)
            intercept = y_mean - float(z_mean @ coef)
            pred = Z @ coef + intercept
            resid = y - pred
            mse = float(np.mean(w * resid * resid))
            df = float(np.sum(s2 / (s2 + a)))
            denom = max(1e-8, (1.0 - (df / max(n, 1))))
            gcv = mse / (denom * denom)
            cur = {
                "coef": coef,
                "intercept": intercept,
                "pred": pred,
                "alpha": float(a),
                "gcv": gcv,
                "mse": mse,
                "df": df,
            }
            if best is None or cur["gcv"] < best["gcv"]:
                best = cur
        return best

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        self.n_features_in_ = int(p)

        if n == 0:
            self.mean_ = float(np.mean(y)) if y.size else 0.0
            self.center_ = np.zeros(p, dtype=float)
            self.scale_ = np.ones(p, dtype=float)
            self.coef_ = np.zeros(0, dtype=float)
            self.intercept_ = self.mean_
            self.training_mse_ = 0.0
            self.alpha_ = 0.0
            self.irls_steps_used_ = 0
            self.is_fitted_ = True
            return self

        Z, med, scale = self._robust_standardize(X)
        c = float(max(1.0, self.clip_value))
        Z = np.clip(Z, -c, c)
        self.center_ = med
        self.scale_ = scale
        self.feature_names_ = [f"x{j}" for j in range(p)]

        result = self._ridge_gcv(Z, y, sample_weight=None)
        alpha = float(result["alpha"])
        coef = result["coef"].copy()
        intercept = float(result["intercept"])

        steps = int(max(0, self.irls_steps))
        delta = float(max(0.1, self.huber_delta))
        w = np.ones(n, dtype=float)
        for _ in range(steps):
            pred = Z @ coef + intercept
            resid = y - pred
            scale_r = 1.4826 * np.median(np.abs(resid - np.median(resid)))
            scale_r = float(max(1e-6, scale_r))
            u = np.abs(resid) / scale_r
            w = np.where(u <= delta, 1.0, (delta / np.maximum(u, 1e-8)))
            refined = self._ridge_gcv(Z, y, sample_weight=w)
            coef = refined["coef"]
            intercept = float(refined["intercept"])

        final_pred = Z @ coef + intercept
        final_mse = float(np.mean((y - final_pred) ** 2))

        self.coef_ = coef
        self.intercept_ = intercept
        self.alpha_ = alpha
        self.training_mse_ = final_mse
        self.training_gcv_ = float(result["gcv"])
        self.training_df_ = float(result["df"])
        self.n_basis_terms_ = int(Z.shape[1])
        self.irls_steps_used_ = steps
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        X = np.asarray(X, dtype=float)
        Z = (X - self.center_) / self.scale_
        c = float(max(1.0, self.clip_value))
        Z = np.clip(Z, -c, c)
        return Z @ self.coef_ + self.intercept_

    def __str__(self):
        check_is_fitted(self, "is_fitted_")
        lines = ["Huber-Refined Ridge Equation Regressor:"]
        lines.append("  prediction = intercept + weighted sum of scaled raw features")
        lines.append(f"  intercept: {self.intercept_:+.4f}")
        lines.append(f"  ridge alpha (GCV): {self.alpha_:.4g}")
        lines.append(f"  IRLS steps: {self.irls_steps_used_}")
        lines.append(f"  terms: {self.n_basis_terms_}")
        lines.append(f"  training MSE: {self.training_mse_:.6f}")
        if self.coef_.size == 0:
            lines.append("  no learned terms")
            return "\n".join(lines)

        top_k = int(max(1, self.display_top_k))
        order = np.argsort(np.abs(self.coef_))[::-1][:top_k]
        lines.append(f"  top |coefficient| terms shown: {len(order)}")
        for idx in order:
            lines.append(f"    {self.feature_names_[idx]}: {self.coef_[idx]:+.4f}")
        if self.coef_.size > top_k:
            lines.append(f"  ... plus {self.coef_.size - top_k} smaller terms")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
HuberRidgeIRLSRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "HuberRidgeIRLS_v1"
model_description = "Robust-scaled linear ridge equation with GCV alpha selection and Huber-style IRLS reweighting"
model_defs = [(model_shorthand_name, HuberRidgeIRLSRegressor())]


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
