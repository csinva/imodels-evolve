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


class RobustSparseRidgeEquationRegressor(BaseEstimator, RegressorMixin):
    """
    Robust sparse linear regressor:
    1) median-impute and quantile-clip features,
    2) choose ridge alpha with closed-form GCV,
    3) keep only top-contribution features and refit.
    """

    def __init__(
        self,
        max_active_features=8,
        min_active_features=2,
        cumulative_importance=0.96,
        clip_quantile=0.01,
        coef_tol=1e-5,
        alpha_min_exp=-4.0,
        alpha_max_exp=2.0,
        alpha_grid_size=19,
    ):
        self.max_active_features = max_active_features
        self.min_active_features = min_active_features
        self.cumulative_importance = cumulative_importance
        self.clip_quantile = clip_quantile
        self.coef_tol = coef_tol
        self.alpha_min_exp = alpha_min_exp
        self.alpha_max_exp = alpha_max_exp
        self.alpha_grid_size = alpha_grid_size

    def _impute(self, X):
        X = np.asarray(X, dtype=float).copy()
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(self.feature_medians_, np.where(bad)[1])
        return X

    def _clip(self, X):
        return np.clip(X, self.feature_lower_, self.feature_upper_)

    @staticmethod
    def _ridge_gcv_fit(Xs, y_centered, alphas):
        n = Xs.shape[0]
        U, s, Vt = np.linalg.svd(Xs, full_matrices=False)
        Uy = U.T @ y_centered

        best = None
        for alpha in alphas:
            denom = (s * s) + alpha
            beta_std = Vt.T @ ((s / denom) * Uy)
            resid = y_centered - (Xs @ beta_std)
            rss = float(np.dot(resid, resid))
            df = float(np.sum((s * s) / denom))
            gcv = rss / (max(n - df, 1e-9) ** 2)
            if (best is None) or (gcv < best[0]):
                best = (gcv, float(alpha), beta_std)
        return best[1], best[2]

    def _prepare_design(self, X):
        Xc = self._clip(self._impute(X))
        Xs = (Xc - self.feature_mean_) / self.feature_scale_
        return Xc, Xs

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        self.n_features_in_ = p

        self.feature_medians_ = np.nanmedian(X, axis=0)
        X = self._impute(X)

        q = float(self.clip_quantile)
        self.feature_lower_ = np.quantile(X, q, axis=0)
        self.feature_upper_ = np.quantile(X, 1.0 - q, axis=0)
        X = self._clip(X)

        self.feature_mean_ = np.mean(X, axis=0)
        self.feature_scale_ = np.std(X, axis=0)
        self.feature_scale_[self.feature_scale_ < 1e-12] = 1.0
        Xs = (X - self.feature_mean_) / self.feature_scale_

        y_mean = float(np.mean(y))
        yc = y - y_mean

        alphas = np.logspace(float(self.alpha_min_exp), float(self.alpha_max_exp), int(self.alpha_grid_size))
        alpha_all, beta_std_all = self._ridge_gcv_fit(Xs, yc, alphas)
        coef_all = beta_std_all / self.feature_scale_

        abs_coef = np.abs(coef_all)
        order = np.argsort(abs_coef)[::-1]
        order = [int(i) for i in order if abs_coef[i] > float(self.coef_tol)]

        if len(order) == 0:
            active = []
        else:
            total = float(np.sum(abs_coef[order])) + 1e-12
            running = 0.0
            active = []
            for j in order:
                active.append(j)
                running += float(abs_coef[j])
                if (
                    len(active) >= int(self.min_active_features)
                    and running / total >= float(self.cumulative_importance)
                ):
                    break
                if len(active) >= int(self.max_active_features):
                    break

        if len(active) == 0:
            self.selected_features_ = []
            self.coef_ = np.zeros(0, dtype=float)
            self.intercept_ = y_mean
            self.alpha_ = alpha_all
            self.full_coef_ = coef_all
            self.fitted_mse_ = float(np.mean((y - y_mean) ** 2))
            return self

        active = sorted(active)
        Xs_sel = Xs[:, active]
        alpha_sel, beta_std_sel = self._ridge_gcv_fit(Xs_sel, yc, alphas)

        coef_sel = beta_std_sel / self.feature_scale_[active]
        intercept = y_mean - float(np.dot(self.feature_mean_[active], coef_sel))

        keep = np.where(np.abs(coef_sel) > float(self.coef_tol))[0]
        if keep.size == 0:
            self.selected_features_ = []
            self.coef_ = np.zeros(0, dtype=float)
            self.intercept_ = y_mean
        else:
            active = [active[int(i)] for i in keep]
            coef_sel = coef_sel[keep]
            self.selected_features_ = active
            self.coef_ = np.asarray(coef_sel, dtype=float)
            self.intercept_ = float(intercept)

        self.alpha_ = alpha_sel
        self.full_coef_ = coef_all
        preds = self.predict(X)
        self.fitted_mse_ = float(np.mean((y - preds) ** 2))
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "coef_", "selected_features_"])
        Xc, _ = self._prepare_design(X)
        yhat = np.full(Xc.shape[0], self.intercept_, dtype=float)
        if len(self.selected_features_) > 0:
            yhat += Xc[:, self.selected_features_] @ self.coef_
        return yhat

    def __str__(self):
        check_is_fitted(self, ["intercept_", "coef_", "selected_features_"])
        lines = [
            "RobustSparseRidgeEquationRegressor",
            f"Selected ridge alpha: {self.alpha_:.6g}",
            "Prediction equation:",
        ]
        if len(self.selected_features_) == 0:
            lines.append(f"  y = {self.intercept_:+.6f}")
            lines.append("")
            lines.append("No active features selected.")
            lines.append(f"Training MSE: {self.fitted_mse_:.6f}")
            return "\n".join(lines)

        eq = [f"({float(c):+.6f})*x{int(j)}" for j, c in zip(self.selected_features_, self.coef_)]
        lines.append(f"  y = {self.intercept_:+.6f} " + " ".join("+ " + t for t in eq))
        lines.append("")
        lines.append("Active features (ordered by absolute coefficient):")
        order = np.argsort(np.abs(self.coef_))[::-1]
        for idx in order:
            j = int(self.selected_features_[idx])
            c = float(self.coef_[idx])
            lines.append(f"  x{j}: coefficient={c:+.6f}")

        omitted = [f"x{j}" for j in range(self.n_features_in_) if j not in set(self.selected_features_)]
        if omitted:
            lines.append("")
            lines.append("Negligible/unused features: " + ", ".join(omitted))
        lines.append(f"Training MSE: {self.fitted_mse_:.6f}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
RobustSparseRidgeEquationRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "RobustSparseRidgeEqV1"
model_description = "Custom robust sparse ridge equation: median-impute, quantile clip, closed-form GCV alpha selection, and top-contribution feature pruning with explicit linear rule"
model_defs = [(model_shorthand_name, RobustSparseRidgeEquationRegressor())]

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
