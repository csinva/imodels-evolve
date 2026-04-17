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


class MassConservingGCVRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    From-scratch ridge solved by SVD + generalized cross-validation (GCV),
    then conservative coefficient-mass pruning for compactness.
    """

    def __init__(
        self,
        alpha_grid=(1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0),
        retain_coef_mass=0.995,
        min_active_features=6,
    ):
        self.alpha_grid = alpha_grid
        self.retain_coef_mass = retain_coef_mass
        self.min_active_features = min_active_features

    @staticmethod
    def _safe_std(v):
        s = float(np.std(v))
        return s if s > 1e-12 else 1.0

    def _fit_ridge_standardized(self, Z, yc):
        n, p = Z.shape
        if p == 0:
            return np.zeros(0, dtype=float), float(self.alpha_grid[0]), float("inf")

        U, s, Vt = np.linalg.svd(Z, full_matrices=False)
        Uy = U.T @ yc

        best_alpha = float(self.alpha_grid[0])
        best_gcv = np.inf
        best_coef = np.zeros(p, dtype=float)
        n_eff = max(n, 1)

        for a in self.alpha_grid:
            alpha = float(max(a, 1e-12))
            shrink = s / (s * s + alpha)
            coef = Vt.T @ (shrink * Uy)
            resid = yc - Z @ coef
            trace_h = float(np.sum((s * s) / (s * s + alpha)))
            denom = max(1.0 - trace_h / n_eff, 1e-6)
            gcv = float(np.mean(resid * resid) / (denom * denom))
            if gcv < best_gcv:
                best_gcv = gcv
                best_alpha = alpha
                best_coef = coef
        return best_coef, best_alpha, best_gcv

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        self.n_features_in_ = p

        self.x_mean_ = np.mean(X, axis=0)
        self.x_scale_ = np.array([self._safe_std(X[:, j]) for j in range(p)], dtype=float)
        Z = (X - self.x_mean_) / self.x_scale_

        y_mean = float(np.mean(y))
        yc = y - y_mean

        coef_std, alpha, best_gcv = self._fit_ridge_standardized(Z, yc)
        coef_raw = coef_std / self.x_scale_

        abs_coef = np.abs(coef_raw)
        total_mass = float(np.sum(abs_coef))
        if total_mass <= 1e-15:
            active = np.array([], dtype=int)
            pruned_coef = np.zeros_like(coef_raw)
        else:
            order = np.argsort(-abs_coef)
            csum = np.cumsum(abs_coef[order])
            target = float(max(0.50, min(0.9999, self.retain_coef_mass))) * total_mass
            k_mass = int(np.searchsorted(csum, target) + 1)
            k = int(max(min(p, k_mass), min(max(1, self.min_active_features), p)))
            active = np.sort(order[:k].astype(int))
            pruned_coef = np.zeros_like(coef_raw)
            pruned_coef[active] = coef_raw[active]

        intercept = y_mean - float(np.dot(self.x_mean_, pruned_coef))

        self.alpha_ = float(alpha)
        self.gcv_score_ = float(best_gcv)
        self.intercept_ = float(intercept)
        self.coef_ = pruned_coef
        self.active_features_ = active
        self.negligible_features_ = np.array([j for j in range(p) if j not in set(active.tolist())], dtype=int)
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "coef_"])
        X = np.asarray(X, dtype=float)
        return self.intercept_ + X @ self.coef_

    def __str__(self):
        check_is_fitted(self, ["intercept_", "coef_"])
        lines = [
            "Mass-Conserving GCV Ridge Regressor",
            "Exact prediction equation in raw features:",
        ]
        eq = f"y(x) = {self.intercept_:+.6f}"
        for j in self.active_features_:
            eq += f" {float(self.coef_[j]):+.6f}*x{int(j)}"
        lines.append(eq)
        lines.append(f"Ridge alpha (GCV): {self.alpha_:.6g}")
        lines.append(f"GCV objective: {self.gcv_score_:.6g}")
        lines.append("Meaningfully used features: " + ", ".join(f"x{int(j)}" for j in self.active_features_))
        if len(self.negligible_features_) > 0:
            lines.append(
                "Features with negligible/zero effect: "
                + ", ".join(f"x{int(j)}" for j in self.negligible_features_)
            )
        lines.append("Contribution table (active features only):")
        for j in self.active_features_:
            lines.append(f"  x{int(j)}: coef={float(self.coef_[j]):+.6f}")
        return "\n".join(lines)


import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
MassConservingGCVRidgeRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "MassConservingGCVRidgeV1"
model_description = "From-scratch SVD+GCV ridge in raw-feature equation form with conservative coefficient-mass pruning for compact explicit simulation"
model_defs = [(model_shorthand_name, MassConservingGCVRidgeRegressor())]

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
