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


class RidgeSingleHingeRegressor(BaseEstimator, RegressorMixin):
    """
    Closed-form ridge backbone plus one optional learned hinge correction:

      y = intercept + sum_j coef_j * x_j + hinge_coef * max(0, sgn * (x_hinge - threshold))
    """

    def __init__(
        self,
        alpha_grid=(0.003, 0.01, 0.03, 0.1, 0.3, 1.0),
        hinge_l2=0.05,
        hinge_quantiles=(0.15, 0.3, 0.5, 0.7, 0.85),
        max_hinge_features=8,
        min_hinge_gain_frac=0.0015,
        coef_tol=1e-4,
        random_state=42,
    ):
        self.alpha_grid = alpha_grid
        self.hinge_l2 = hinge_l2
        self.hinge_quantiles = hinge_quantiles
        self.max_hinge_features = max_hinge_features
        self.min_hinge_gain_frac = min_hinge_gain_frac
        self.coef_tol = coef_tol
        self.random_state = random_state

    def _impute(self, X):
        X = np.asarray(X, dtype=float).copy()
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(self.feature_medians_, np.where(bad)[1])
        return X

    @staticmethod
    def _fit_ridge_standardized(X, y, alpha):
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        sigma[sigma < 1e-10] = 1.0
        Xs = (X - mu) / sigma

        y_mean = float(np.mean(y))
        yc = y - y_mean

        p = X.shape[1]
        gram = Xs.T @ Xs
        beta_s = np.linalg.solve(gram + float(alpha) * np.eye(p), Xs.T @ yc)
        coef = beta_s / sigma
        intercept = y_mean - float(np.dot(mu, coef))
        return float(intercept), coef.astype(float)

    def _choose_alpha(self, X, y):
        n = X.shape[0]
        if n < 40:
            return float(self.alpha_grid[0])

        rng = np.random.RandomState(self.random_state)
        idx = np.arange(n)
        rng.shuffle(idx)
        cut = max(int(0.8 * n), n - 40)
        tr = idx[:cut]
        va = idx[cut:]
        if va.size < 12:
            return float(self.alpha_grid[0])

        best_alpha = float(self.alpha_grid[0])
        best_mse = np.inf
        for a in self.alpha_grid:
            intercept, coef = self._fit_ridge_standardized(X[tr], y[tr], a)
            pred = intercept + X[va] @ coef
            mse = float(np.mean((y[va] - pred) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_alpha = float(a)
        return best_alpha

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.feature_medians_ = np.nanmedian(X, axis=0)
        X = self._impute(X)

        self.alpha_ = self._choose_alpha(X, y)
        self.intercept_, self.coef_ = self._fit_ridge_standardized(X, y, self.alpha_)

        pred_linear = self.intercept_ + X @ self.coef_
        residual = y - pred_linear
        base_mse = float(np.mean(residual ** 2))
        y_var = float(np.var(y)) + 1e-12

        self.has_hinge_ = False
        self.hinge_feature_ = -1
        self.hinge_threshold_ = 0.0
        self.hinge_sign_ = 1.0
        self.hinge_coef_ = 0.0

        order = np.argsort(np.abs(self.coef_))[::-1]
        candidate_features = order[: min(int(self.max_hinge_features), self.n_features_in_)]
        best = None
        n = X.shape[0]

        for j in candidate_features:
            xj = X[:, j]
            thresholds = np.unique(np.quantile(xj, self.hinge_quantiles))
            for thr in thresholds:
                for sgn in (1.0, -1.0):
                    phi = np.maximum(0.0, sgn * (xj - float(thr)))
                    denom = float(np.dot(phi, phi) + self.hinge_l2 * n)
                    if denom <= 1e-12:
                        continue
                    h = float(np.dot(phi, residual) / denom)
                    pred = pred_linear + h * phi
                    mse = float(np.mean((y - pred) ** 2))
                    if best is None or mse < best["mse"]:
                        best = {
                            "feature": int(j),
                            "threshold": float(thr),
                            "sign": float(sgn),
                            "coef": float(h),
                            "mse": mse,
                        }

        if best is not None and (base_mse - best["mse"]) > float(self.min_hinge_gain_frac) * y_var:
            self.has_hinge_ = True
            self.hinge_feature_ = int(best["feature"])
            self.hinge_threshold_ = float(best["threshold"])
            self.hinge_sign_ = float(best["sign"])
            self.hinge_coef_ = float(best["coef"])

        self.coef_[np.abs(self.coef_) < self.coef_tol] = 0.0
        if abs(self.hinge_coef_) < self.coef_tol:
            self.has_hinge_ = False
            self.hinge_coef_ = 0.0

        self.feature_importance_ = np.abs(self.coef_)
        if self.has_hinge_:
            self.feature_importance_[self.hinge_feature_] += abs(self.hinge_coef_)
        self.selected_feature_order_ = np.argsort(self.feature_importance_)[::-1]
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "coef_", "has_hinge_"])
        X = self._impute(X)
        yhat = self.intercept_ + X @ self.coef_
        if self.has_hinge_:
            xh = X[:, self.hinge_feature_]
            yhat = yhat + self.hinge_coef_ * np.maximum(0.0, self.hinge_sign_ * (xh - self.hinge_threshold_))
        return yhat

    def __str__(self):
        check_is_fitted(self, ["intercept_", "coef_", "has_hinge_"])
        lines = [
            "RidgeSingleHingeRegressor",
            "Exact prediction equation:",
            f"  y = {self.intercept_:+.6f}",
        ]
        for j in np.argsort(np.abs(self.coef_))[::-1]:
            if abs(self.coef_[j]) >= self.coef_tol:
                lines.append(f"    {self.coef_[j]:+.6f} * x{int(j)}")

        if self.has_hinge_:
            if self.hinge_sign_ > 0:
                hinge_text = f"max(0, x{self.hinge_feature_} - {self.hinge_threshold_:+.6f})"
            else:
                hinge_text = f"max(0, {self.hinge_threshold_:+.6f} - x{self.hinge_feature_})"
            lines.append(f"    {self.hinge_coef_:+.6f} * {hinge_text}")
        else:
            lines.append("    (no hinge correction)")

        lines.append("")
        lines.append("Feature summary (sorted by total effect magnitude):")
        for j in self.selected_feature_order_[: min(12, self.n_features_in_)]:
            lines.append(f"  x{int(j)}: linear={self.coef_[int(j)]:+.6f}, importance={self.feature_importance_[int(j)]:.6f}")

        rel_cut = 0.06 * (np.max(np.abs(self.coef_)) + 1e-12)
        negligible = [f"x{j}" for j in range(self.n_features_in_) if abs(self.coef_[j]) <= rel_cut]
        if negligible:
            lines.append("Likely negligible features (tiny linear coefficients): " + ", ".join(negligible))
        lines.append("To simulate: compute linear part, then add hinge term (if present).")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
RidgeSingleHingeRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "RidgeSingleHingeV1"
model_description = "Closed-form ridge linear backbone with one optional learned directional hinge correction on a single feature"
model_defs = [(model_shorthand_name, RidgeSingleHingeRegressor())]

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
