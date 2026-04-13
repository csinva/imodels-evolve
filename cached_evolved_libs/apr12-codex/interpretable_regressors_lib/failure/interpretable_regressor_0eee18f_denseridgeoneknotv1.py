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


class DenseRidgeOneKnotRegressor(BaseEstimator, RegressorMixin):
    """
    Dense linear backbone + one explicit nonlinear correction.
    The correction is a one-feature two-hinge term:
      w_pos * max(0, x_j - t) + w_neg * max(0, t - x_j)
    """

    def __init__(
        self,
        alpha_grid=(0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0),
        valid_frac=0.2,
        top_features_for_knot=10,
        knot_quantiles=(0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9),
        min_leaf_frac=0.1,
        nonlinear_gain=0.01,
        coef_tol=1e-5,
        random_state=42,
    ):
        self.alpha_grid = alpha_grid
        self.valid_frac = valid_frac
        self.top_features_for_knot = top_features_for_knot
        self.knot_quantiles = knot_quantiles
        self.min_leaf_frac = min_leaf_frac
        self.nonlinear_gain = nonlinear_gain
        self.coef_tol = coef_tol
        self.random_state = random_state

    def _impute(self, X):
        X = np.asarray(X, dtype=float).copy()
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(self.feature_medians_, np.where(bad)[1])
        return X

    @staticmethod
    def _corr_abs(a, b):
        ac = a - float(np.mean(a))
        bc = b - float(np.mean(b))
        denom = (float(np.std(ac)) + 1e-12) * (float(np.std(bc)) + 1e-12)
        return abs(float(np.mean(ac * bc)) / denom)

    def _fit_ridge_with_split_valid(self, X, y):
        n, p = X.shape
        n_valid = max(20, int(n * float(self.valid_frac)))
        n_valid = min(n_valid, n - 20) if n > 40 else max(1, n // 4)

        rng = np.random.RandomState(self.random_state)
        perm = rng.permutation(n)
        valid_idx = perm[:n_valid]
        train_idx = perm[n_valid:]
        if train_idx.size == 0:
            train_idx = perm
            valid_idx = perm[: max(1, n // 5)]

        Xtr = X[train_idx]
        ytr = y[train_idx]
        Xva = X[valid_idx]
        yva = y[valid_idx]

        mu = np.mean(Xtr, axis=0)
        sigma = np.std(Xtr, axis=0)
        sigma[sigma < 1e-10] = 1.0
        Xtr_s = (Xtr - mu) / sigma
        Xva_s = (Xva - mu) / sigma
        y_mean = float(np.mean(ytr))
        ytr_c = ytr - y_mean

        best = None
        I = np.eye(p)
        for alpha in self.alpha_grid:
            beta_s = np.linalg.solve(Xtr_s.T @ Xtr_s + float(alpha) * len(Xtr) * I, Xtr_s.T @ ytr_c)
            pred_va = y_mean + Xva_s @ beta_s
            mse_va = float(np.mean((yva - pred_va) ** 2))
            if best is None or mse_va < best["mse"]:
                best = {"alpha": float(alpha), "beta_s": beta_s.copy(), "mse": mse_va, "mu": mu, "sigma": sigma}

        Xs = (X - best["mu"]) / best["sigma"]
        y_mean_all = float(np.mean(y))
        beta_s = np.linalg.solve(Xs.T @ Xs + best["alpha"] * n * I, Xs.T @ (y - y_mean_all))
        coef = beta_s / best["sigma"]
        intercept = y_mean_all - float(np.dot(best["mu"], coef))
        return float(intercept), coef.astype(float), float(best["alpha"])

    def _fit_one_knot(self, xj, residual):
        n = xj.shape[0]
        min_leaf = max(10, int(float(self.min_leaf_frac) * n))
        best = None
        base_mse = float(np.mean(residual ** 2))

        for t in np.unique(np.quantile(xj, self.knot_quantiles)):
            left_n = int(np.sum(xj <= t))
            right_n = n - left_n
            if left_n < min_leaf or right_n < min_leaf:
                continue

            b_pos = np.maximum(0.0, xj - float(t))
            b_neg = np.maximum(0.0, float(t) - xj)
            B = np.column_stack([b_pos, b_neg])
            gram = B.T @ B + 1e-6 * n * np.eye(2)
            w = np.linalg.solve(gram, B.T @ residual)
            pred = B @ w
            mse = float(np.mean((residual - pred) ** 2))
            gain = base_mse - mse
            if best is None or gain > best["gain"]:
                best = {
                    "threshold": float(t),
                    "w_pos": float(w[0]),
                    "w_neg": float(w[1]),
                    "gain": gain,
                    "mse": mse,
                    "left_n": left_n,
                    "right_n": right_n,
                }
        return best, base_mse

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.feature_medians_ = np.nanmedian(X, axis=0)
        X = self._impute(X)

        self.intercept_, self.coef_, self.best_alpha_ = self._fit_ridge_with_split_valid(X, y)
        base_pred = self.intercept_ + X @ self.coef_
        residual = y - base_pred

        p = X.shape[1]
        nonlin_score = np.array([
            self._corr_abs(X[:, j], residual)
            + 0.9 * self._corr_abs(np.abs(X[:, j]), residual)
            + 0.7 * self._corr_abs(X[:, j] * X[:, j], residual)
            for j in range(p)
        ])
        top = np.argsort(nonlin_score)[::-1][: min(int(self.top_features_for_knot), p)]

        self.use_knot_ = False
        self.knot_feature_ = -1
        self.knot_threshold_ = 0.0
        self.knot_w_pos_ = 0.0
        self.knot_w_neg_ = 0.0
        self.knot_gain_ = 0.0

        base_mse = float(np.mean(residual ** 2))
        best = None
        for j in top:
            cand, _ = self._fit_one_knot(X[:, j], residual)
            if cand is None:
                continue
            if best is None or cand["gain"] > best["gain"]:
                best = {"feature": int(j), **cand}

        if best is not None and best["gain"] > float(self.nonlinear_gain) * max(base_mse, 1e-12):
            self.use_knot_ = True
            self.knot_feature_ = int(best["feature"])
            self.knot_threshold_ = float(best["threshold"])
            self.knot_w_pos_ = float(best["w_pos"])
            self.knot_w_neg_ = float(best["w_neg"])
            self.knot_gain_ = float(best["gain"])

        self.feature_importance_ = np.abs(self.coef_).copy()
        if self.use_knot_:
            self.feature_importance_[self.knot_feature_] += abs(self.knot_w_pos_) + abs(self.knot_w_neg_)
        self.selected_feature_order_ = np.argsort(self.feature_importance_)[::-1]
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "coef_", "use_knot_"])
        X = self._impute(X)
        yhat = self.intercept_ + X @ self.coef_
        if self.use_knot_:
            xj = X[:, self.knot_feature_]
            yhat += self.knot_w_pos_ * np.maximum(0.0, xj - self.knot_threshold_)
            yhat += self.knot_w_neg_ * np.maximum(0.0, self.knot_threshold_ - xj)
        return yhat

    def __str__(self):
        check_is_fitted(self, ["intercept_", "coef_", "use_knot_"])
        lines = [
            "DenseRidgeOneKnotRegressor",
            "Prediction equation:",
        ]
        terms = [f"{self.intercept_:+.6f}"]
        for j in np.argsort(np.abs(self.coef_))[::-1]:
            c = self.coef_[int(j)]
            if abs(c) >= self.coef_tol:
                terms.append(f"{c:+.6f}*x{int(j)}")
        lines.append("  y = " + " ".join(terms))

        if self.use_knot_:
            j = self.knot_feature_
            t = self.knot_threshold_
            lines.append(
                f"  + ({self.knot_w_pos_:+.6f})*max(0, x{j}-{t:+.6f})"
                f" + ({self.knot_w_neg_:+.6f})*max(0, {t:+.6f}-x{j})"
            )
            lines.append(
                f"One-knot correction uses feature x{j} with threshold {t:+.6f}."
            )
        else:
            lines.append("No nonlinear correction selected.")

        lines.append("")
        lines.append("Most important features (absolute effect size):")
        for j in self.selected_feature_order_[: min(12, self.n_features_in_)]:
            lines.append(
                f"  x{int(j)}: coef={self.coef_[int(j)]:+.6f}, importance={self.feature_importance_[int(j)]:.6f}"
            )

        rel_thr = max(0.02, 0.05 * float(np.max(np.abs(self.coef_) + 1e-12)))
        negligible = [f"x{j}" for j in range(self.n_features_in_) if abs(self.coef_[j]) < rel_thr]
        if negligible:
            lines.append("Likely negligible linear-effect features: " + ", ".join(negligible))
        lines.append(f"Selected ridge alpha: {self.best_alpha_:.4g}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
DenseRidgeOneKnotRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "DenseRidgeOneKnotV1"
model_description = "Dense ridge linear equation with holdout-selected alpha plus one learned one-feature two-hinge knot correction for nonlinear residuals"
model_defs = [(model_shorthand_name, DenseRidgeOneKnotRegressor())]

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
