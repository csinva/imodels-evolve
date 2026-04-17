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


class SVDGCVRidgeMapRegressor(BaseEstimator, RegressorMixin):
    """
    Dense ridge regressor built from scratch with closed-form SVD and GCV alpha
    selection, then mapped into an explicit raw-feature equation.

    Design goals:
    - Preserve predictive strength (dense ridge, no hard feature dropping in fit).
    - Keep representation highly simulatable (single equation with all terms).
    - Surface negligible-effect features for interpretability tests.
    """

    def __init__(
        self,
        alpha_grid=(1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0),
        coef_display_threshold=0.03,
    ):
        self.alpha_grid = alpha_grid
        self.coef_display_threshold = coef_display_threshold

    @staticmethod
    def _safe_std(X):
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std = np.where(std > 1e-12, std, 1.0)
        return mean.astype(float), std.astype(float)

    def _fit_gcv_ridge(self, X, y):
        mean, std = self._safe_std(X)
        Xz = (X - mean) / std
        yc = y - float(np.mean(y))

        U, s, Vt = np.linalg.svd(Xz, full_matrices=False)
        Uy = U.T @ yc
        s2 = s * s
        n = Xz.shape[0]

        best = None
        for alpha in self.alpha_grid:
            a = float(alpha)
            shrink = s / (s2 + a)
            coef_z = Vt.T @ (shrink * Uy)
            fitted_centered = U @ ((s2 / (s2 + a)) * Uy)
            resid = yc - fitted_centered

            trace_h = float(np.sum(s2 / (s2 + a)))
            denom = max((n - trace_h), 1e-8)
            gcv = float(np.sum(resid * resid) / (denom * denom))
            train_mse = float(np.mean(resid * resid))

            if best is None or gcv < best["gcv"]:
                coef_raw = coef_z / std
                intercept_raw = float(np.mean(y)) - float(np.dot(coef_raw, mean))
                best = {
                    "alpha": a,
                    "gcv": gcv,
                    "train_mse": train_mse,
                    "coef": coef_raw,
                    "intercept": float(intercept_raw),
                }
        return best

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        _, p = X.shape
        self.n_features_in_ = p

        fit = self._fit_gcv_ridge(X, y)
        self.intercept_ = float(fit["intercept"])
        self.coef_ = np.asarray(fit["coef"], dtype=float)
        self.alpha_ = float(fit["alpha"])
        self.gcv_score_ = float(fit["gcv"])
        self.train_mse_ = float(fit["train_mse"])

        thr = float(self.coef_display_threshold)
        abs_coef = np.abs(self.coef_)
        self.mean_abs_coef_ = float(np.mean(abs_coef)) if len(abs_coef) > 0 else 0.0
        cutoff = thr * max(self.mean_abs_coef_, 1e-12)
        self.meaningful_features_ = np.where(abs_coef >= cutoff)[0].astype(int)
        self.negligible_features_ = np.where(abs_coef < cutoff)[0].astype(int)
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "coef_"])
        X = np.asarray(X, dtype=float)
        out = self.intercept_ + X @ self.coef_
        return np.asarray(out, dtype=float)

    def __str__(self):
        check_is_fitted(self, ["intercept_", "coef_", "alpha_"])
        lines = [
            "SVD-GCV Ridge Map Regressor",
            f"ridge_alpha={self.alpha_:.6g} (chosen by generalized cross-validation)",
            "Prediction uses this exact equation:",
        ]

        equation = f"y = {self.intercept_:+.6f}"
        for j, c in enumerate(self.coef_):
            equation += f" {float(c):+.6f}*x{int(j)}"
        lines.append(equation)

        lines.append("Feature coefficients (largest absolute effect first):")
        ranked = sorted(enumerate(self.coef_), key=lambda t: -abs(float(t[1])))
        for j, c in ranked:
            lines.append(f"  x{int(j)}: {float(c):+.6f}")

        if len(self.meaningful_features_) > 0:
            lines.append(
                "Meaningful features (relative effect above threshold): "
                + ", ".join(f"x{int(j)}" for j in self.meaningful_features_)
            )
        if len(self.negligible_features_) > 0:
            lines.append(
                "Features with negligible or zero effect in this fitted equation: "
                + ", ".join(f"x{int(j)}" for j in self.negligible_features_)
            )

        ops = 1 + len(self.coef_)
        lines.append(f"Approx arithmetic operations: {ops}")
        lines.append(f"GCV objective: {self.gcv_score_:.6f}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
SVDGCVRidgeMapRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "SVDGCVRidgeMapV1"
model_description = "From-scratch dense ridge solved by SVD with generalized cross-validation alpha selection and explicit raw-feature equation plus negligible-feature reporting"
model_defs = [(model_shorthand_name, SVDGCVRidgeMapRegressor())]

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
