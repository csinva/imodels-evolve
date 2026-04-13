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


class RidgePlusOneHingeRegressor(BaseEstimator, RegressorMixin):
    """
    Dense ridge backbone + optional single hinge correction:
      y = b + sum_j w_j * x_j + v * max(0, s * (x_h - t))

    This keeps the representation arithmetic and compact while still allowing
    one clear nonlinear threshold effect.
    """

    def __init__(
        self,
        ridge_alpha=1.0,
        coef_tol=3e-3,
        top_hinge_features=6,
        min_hinge_gain=5e-4,
    ):
        self.ridge_alpha = ridge_alpha
        self.coef_tol = coef_tol
        self.top_hinge_features = top_hinge_features
        self.min_hinge_gain = min_hinge_gain

    def _impute(self, X):
        X = np.asarray(X, dtype=float).copy()
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(self.feature_medians_, np.where(bad)[1])
        return X

    @staticmethod
    def _centered_ridge(A, y, alpha):
        if A.shape[1] == 0:
            return float(np.mean(y)), np.zeros(0, dtype=float)
        y_mean = float(np.mean(y))
        yc = y - y_mean
        a_mean = np.mean(A, axis=0)
        Ac = A - a_mean
        gram = Ac.T @ Ac
        rhs = Ac.T @ yc
        coef = np.linalg.solve(gram + float(alpha) * np.eye(A.shape[1]), rhs)
        intercept = y_mean - float(np.dot(a_mean, coef))
        return float(intercept), coef

    @staticmethod
    def _hinge_col(x, threshold, sign):
        if sign > 0:
            return np.maximum(0.0, x - threshold)
        return np.maximum(0.0, threshold - x)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.feature_medians_ = np.nanmedian(X, axis=0)
        X = self._impute(X)

        # Dense linear backbone for strong predictive ranking.
        b_lin, w_lin = self._centered_ridge(X, y, self.ridge_alpha)
        pred_lin = b_lin + X @ w_lin
        mse_lin = float(np.mean((y - pred_lin) ** 2))

        # Search a single hinge on influential features for simple nonlinearity.
        order = np.argsort(np.abs(w_lin))[::-1]
        cand_feats = order[: min(self.top_hinge_features, self.n_features_in_)]
        best = None
        for j in cand_feats:
            xj = X[:, int(j)]
            for q in (0.2, 0.35, 0.5, 0.65, 0.8):
                t = float(np.quantile(xj, q))
                for s in (1, -1):
                    h = self._hinge_col(xj, t, s)
                    if float(np.std(h)) < 1e-10:
                        continue
                    A = np.column_stack([X, h])
                    b_try, coef_try = self._centered_ridge(A, y, self.ridge_alpha)
                    pred_try = b_try + A @ coef_try
                    mse_try = float(np.mean((y - pred_try) ** 2))
                    if (best is None) or (mse_try < best["mse"]):
                        best = {
                            "mse": mse_try,
                            "j": int(j),
                            "t": t,
                            "sign": int(s),
                            "intercept": b_try,
                            "coef": coef_try,
                        }

        use_hinge = best is not None and (mse_lin - best["mse"]) > self.min_hinge_gain

        if use_hinge:
            self.intercept_ = float(best["intercept"])
            self.linear_coef_ = best["coef"][:-1].copy()
            self.hinge_feature_ = int(best["j"])
            self.hinge_threshold_ = float(best["t"])
            self.hinge_sign_ = int(best["sign"])
            self.hinge_coef_ = float(best["coef"][-1])
        else:
            self.intercept_ = float(b_lin)
            self.linear_coef_ = w_lin.copy()
            self.hinge_feature_ = -1
            self.hinge_threshold_ = 0.0
            self.hinge_sign_ = 1
            self.hinge_coef_ = 0.0

        # Soft sparsification only for presentation and feature-set clarity.
        tiny = np.abs(self.linear_coef_) < self.coef_tol
        self.linear_coef_[tiny] = 0.0

        fi = np.abs(self.linear_coef_)
        if self.hinge_feature_ >= 0 and abs(self.hinge_coef_) >= self.coef_tol:
            fi[self.hinge_feature_] += abs(self.hinge_coef_)
        self.feature_importance_ = fi
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "linear_coef_", "hinge_feature_", "hinge_coef_"])
        X = self._impute(X)
        yhat = self.intercept_ + X @ self.linear_coef_
        if self.hinge_feature_ >= 0 and abs(self.hinge_coef_) > 0:
            h = self._hinge_col(X[:, self.hinge_feature_], self.hinge_threshold_, self.hinge_sign_)
            yhat = yhat + self.hinge_coef_ * h
        return yhat

    def __str__(self):
        check_is_fitted(self, ["intercept_", "linear_coef_", "feature_importance_"])
        lines = [
            "RidgePlusOneHingeRegressor",
            "Prediction equation:",
            f"y = {self.intercept_:+.5f}",
        ]

        term_count = 0
        lin_terms = []
        for j, c in enumerate(self.linear_coef_):
            if abs(float(c)) >= self.coef_tol:
                lin_terms.append((abs(float(c)), f"{c:+.5f} * x{j}"))
        lin_terms.sort(key=lambda x: -x[0])
        for _, t in lin_terms:
            lines.append(f"  {t}")
            term_count += 1

        if self.hinge_feature_ >= 0 and abs(self.hinge_coef_) >= self.coef_tol:
            if self.hinge_sign_ > 0:
                htxt = f"max(0, x{self.hinge_feature_} - {self.hinge_threshold_:.5f})"
            else:
                htxt = f"max(0, {self.hinge_threshold_:.5f} - x{self.hinge_feature_})"
            lines.append(f"  {self.hinge_coef_:+.5f} * {htxt}")
            term_count += 1
            lines.append("  (single hinge term captures a threshold-like nonlinear effect)")

        lines.append("")
        lines.append(f"Model size: {term_count} active terms")
        active = [j for j in np.argsort(self.feature_importance_)[::-1] if self.feature_importance_[j] >= self.coef_tol]
        lines.append("Feature importance (sum of absolute term weights):")
        for j in active:
            lines.append(f"  x{j}: {self.feature_importance_[j]:.5f}")
        near_zero = [f"x{j}" for j in range(self.n_features_in_) if self.feature_importance_[j] < self.coef_tol]
        if near_zero:
            lines.append("Features with near-zero effect: " + ", ".join(near_zero))
        return "\n".join(lines)



# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
RidgePlusOneHingeRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "RidgePlusOneHingeV1"
model_description = "Dense ridge linear backbone with one optional learned hinge correction to capture a single threshold-like nonlinearity"
model_defs = [(model_shorthand_name, RidgePlusOneHingeRegressor())]


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
