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


class SignalWeightedTrimmedRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Transparent weighted-ridge equation:
      1) Winsorize each feature (robust clipping at quantiles).
      2) Standardize features.
      3) Build per-feature penalties from signal strength + tail sensitivity.
      4) Fit weighted ridge with scalar alpha selected by GCV.
      5) Prune tiny coefficients and refit once for readability.
    """

    def __init__(
        self,
        winsor_quantile=0.01,
        alpha_grid_size=41,
        alpha_min_log10=-7.0,
        alpha_max_log10=4.0,
        min_coef_ratio=0.04,
        min_kept_features=3,
        max_display_terms=14,
    ):
        self.winsor_quantile = winsor_quantile
        self.alpha_grid_size = alpha_grid_size
        self.alpha_min_log10 = alpha_min_log10
        self.alpha_max_log10 = alpha_max_log10
        self.min_coef_ratio = min_coef_ratio
        self.min_kept_features = min_kept_features
        self.max_display_terms = max_display_terms

    def _fit_preprocess(self, X):
        q = float(self.winsor_quantile)
        q = min(max(q, 0.0), 0.2)
        lo = np.quantile(X, q, axis=0)
        hi = np.quantile(X, 1.0 - q, axis=0)
        Xw = np.clip(X, lo, hi)
        center = np.mean(Xw, axis=0)
        scale = np.std(Xw, axis=0)
        scale = np.where(scale < 1e-8, 1.0, scale)
        Z = (Xw - center) / scale
        return Z, lo, hi, center, scale

    def _transform(self, X):
        Xw = np.clip(X, self.clip_lo_, self.clip_hi_)
        return (Xw - self.center_) / self.scale_

    def _signal_penalties(self, Z, y):
        n = max(1, Z.shape[0])
        yc = y - float(np.mean(y))
        y_scale = float(np.std(yc)) + 1e-12
        corr_score = np.abs((Z.T @ yc) / n) / y_scale

        tail_low = (Z <= self.low_tail_z_).mean(axis=0)
        tail_high = (Z >= self.high_tail_z_).mean(axis=0)
        tail_score = tail_low + tail_high

        base = 1.0 / (0.15 + corr_score)
        tail_mult = 1.0 + 2.0 * tail_score
        penalties = base * tail_mult
        penalties = np.clip(penalties, 0.25, 6.0)
        penalties = penalties / max(1e-12, float(np.mean(penalties)))
        return penalties

    def _weighted_ridge_gcv(self, X, y, penalties):
        n, p = X.shape
        if n == 0 or p == 0:
            mu = float(np.mean(y)) if y.size else 0.0
            return {
                "coef": np.zeros(p, dtype=float),
                "intercept": mu,
                "pred": np.full(n, mu),
                "alpha": 1.0,
                "gcv": 0.0,
            }

        x_mean = np.mean(X, axis=0)
        y_mean = float(np.mean(y))
        Xc = X - x_mean
        yc = y - y_mean

        sqrt_pen = np.sqrt(np.maximum(1e-12, penalties))
        Xt = Xc / sqrt_pen
        U, s, Vt = np.linalg.svd(Xt, full_matrices=False)
        s2 = s * s
        uty = U.T @ yc

        alphas = np.logspace(
            float(self.alpha_min_log10),
            float(self.alpha_max_log10),
            num=int(max(9, self.alpha_grid_size)),
        )
        best = None
        for alpha in alphas:
            theta = Vt.T @ ((s / (s2 + alpha)) * uty)
            coef = theta / sqrt_pen
            intercept = y_mean - float(x_mean @ coef)
            pred = X @ coef + intercept
            mse = float(np.mean((y - pred) ** 2))
            df = float(np.sum(s2 / (s2 + alpha)))
            denom = max(1e-8, 1.0 - df / max(1, n))
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
            self.clip_lo_ = np.zeros(p, dtype=float)
            self.clip_hi_ = np.zeros(p, dtype=float)
            self.center_ = np.zeros(p, dtype=float)
            self.scale_ = np.ones(p, dtype=float)
            self.penalties_ = np.ones(p, dtype=float)
            self.selected_idx_ = np.arange(p, dtype=int)
            self.coef_ = np.zeros(p, dtype=float)
            self.intercept_ = float(np.mean(y)) if y.size else 0.0
            self.alpha_ = 1.0
            self.training_mse_ = 0.0
            self.is_fitted_ = True
            return self

        Z, lo, hi, center, scale = self._fit_preprocess(X)
        self.clip_lo_ = lo
        self.clip_hi_ = hi
        self.center_ = center
        self.scale_ = scale

        self.low_tail_z_ = float(np.quantile(Z, 0.05))
        self.high_tail_z_ = float(np.quantile(Z, 0.95))
        penalties = self._signal_penalties(Z, y)
        self.penalties_ = penalties

        full = self._weighted_ridge_gcv(Z, y, penalties)
        coef_z = full["coef"]
        abs_coef = np.abs(coef_z)
        max_abs = float(np.max(abs_coef)) if abs_coef.size else 0.0
        threshold = float(self.min_coef_ratio) * max_abs
        keep = abs_coef >= threshold
        if int(np.sum(keep)) < int(self.min_kept_features):
            keep[np.argsort(abs_coef)[::-1][: int(min(p, self.min_kept_features))]] = True
        keep_idx = np.where(keep)[0]

        if keep_idx.size < p:
            refit = self._weighted_ridge_gcv(Z[:, keep_idx], y, penalties[keep_idx])
            final_coef_z = np.zeros(p, dtype=float)
            final_coef_z[keep_idx] = refit["coef"]
            final_intercept = float(refit["intercept"])
            final_alpha = float(refit["alpha"])
            final_pred = Z[:, keep_idx] @ refit["coef"] + refit["intercept"]
        else:
            final_coef_z = coef_z
            final_intercept = float(full["intercept"])
            final_alpha = float(full["alpha"])
            final_pred = full["pred"]

        self.coef_z_ = final_coef_z
        self.selected_idx_ = keep_idx
        self.intercept_ = final_intercept
        self.alpha_ = final_alpha
        self.training_mse_ = float(np.mean((y - final_pred) ** 2))

        self.coef_ = self.coef_z_ / self.scale_
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        X = np.asarray(X, dtype=float)
        Z = self._transform(X)
        return Z @ self.coef_z_ + self.intercept_

    def __str__(self):
        check_is_fitted(self, "is_fitted_")
        lines = ["Signal-Weighted Trimmed Ridge Equation Regressor:"]
        lines.append(
            "  prediction = intercept + sum_j coef_j * standardized_winsorized_feature_j"
        )
        lines.append(f"  winsor quantile: {self.winsor_quantile:.3f}")
        lines.append(f"  chosen alpha: {self.alpha_:.4g}")
        lines.append(f"  training MSE: {self.training_mse_:.6f}")
        lines.append(f"  intercept: {self.intercept_:+.6f}")
        lines.append(
            f"  kept features: {int(self.selected_idx_.size)}/{int(self.n_features_in_)}"
        )

        ranked = np.argsort(np.abs(self.coef_z_))[::-1]
        ranked = [j for j in ranked if abs(self.coef_z_[j]) > 0]
        max_terms = int(max(1, self.max_display_terms))
        lines.append("  strongest terms:")
        for j in ranked[:max_terms]:
            lines.append(
                f"    x{int(j)}: coef_z={self.coef_z_[j]:+.6f}, penalty={self.penalties_[j]:.3f}"
            )
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SignalWeightedTrimmedRidgeRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "SignalWeightedTrimmedRidge_v1"
model_description = "Custom winsorized linear equation with signal-weighted per-feature ridge penalties, GCV alpha selection, and one-pass tiny-coefficient pruning"
model_defs = [(model_shorthand_name, SignalWeightedTrimmedRidgeRegressor())]


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
