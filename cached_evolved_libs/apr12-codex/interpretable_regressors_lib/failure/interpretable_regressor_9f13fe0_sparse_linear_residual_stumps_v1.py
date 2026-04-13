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


class SparseLinearResidualStumpsRegressor(BaseEstimator, RegressorMixin):
    """
    Compact additive model:
      1) Sparse ridge-like linear backbone over top correlated features.
      2) Small list of greedy residual decision stumps (depth-1 rules).

    Prediction is an explicit arithmetic equation plus a short rule list.
    """

    def __init__(
        self,
        alpha=0.06,
        max_linear_terms=8,
        max_stumps=4,
        split_quantiles=(0.12, 0.25, 0.38, 0.5, 0.62, 0.75, 0.88),
        top_split_features=10,
        min_leaf_frac=0.10,
        min_stump_gain=1e-3,
        learning_rate=0.85,
        coef_tol=1e-5,
    ):
        self.alpha = alpha
        self.max_linear_terms = max_linear_terms
        self.max_stumps = max_stumps
        self.split_quantiles = split_quantiles
        self.top_split_features = top_split_features
        self.min_leaf_frac = min_leaf_frac
        self.min_stump_gain = min_stump_gain
        self.learning_rate = learning_rate
        self.coef_tol = coef_tol

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

    def _fit_sparse_linear(self, X, y):
        n, p = X.shape
        k = min(int(self.max_linear_terms), p)
        corr_scores = np.array([self._corr_abs(X[:, j], y) for j in range(p)])
        active = np.argsort(corr_scores)[::-1][:k]
        active = np.array([j for j in active if corr_scores[j] > 1e-7], dtype=int)

        intercept = float(np.mean(y))
        coef = np.zeros(p, dtype=float)
        if active.size == 0:
            return intercept, coef, active

        Xa = X[:, active]
        mu = np.mean(Xa, axis=0)
        sigma = np.std(Xa, axis=0)
        sigma[sigma < 1e-10] = 1.0
        Xs = (Xa - mu) / sigma
        yc = y - intercept

        gram = Xs.T @ Xs
        beta_s = np.linalg.solve(gram + float(self.alpha) * n * np.eye(active.size), Xs.T @ yc)
        beta = beta_s / sigma

        coef[active] = beta
        intercept = intercept - float(np.dot(mu, beta))
        coef[np.abs(coef) < self.coef_tol] = 0.0
        active = np.where(np.abs(coef) >= self.coef_tol)[0]
        return float(intercept), coef, active

    def _candidate_feature_order(self, X, residual):
        p = X.shape[1]
        scores = np.array([
            self._corr_abs(X[:, j], residual)
            + 0.45 * self._corr_abs(X[:, j] * X[:, j], residual)
            + 0.25 * self._corr_abs(np.abs(X[:, j]), residual)
            for j in range(p)
        ])
        return np.argsort(scores)[::-1][: min(int(self.top_split_features), p)]

    def _best_stump(self, X, residual):
        n, p = X.shape
        base_mse = float(np.mean(residual * residual))
        min_leaf = max(10, int(float(self.min_leaf_frac) * n))

        best = None
        best_gain = 0.0
        feature_order = self._candidate_feature_order(X, residual)

        for j in feature_order:
            xj = X[:, j]
            thresholds = np.unique(np.quantile(xj, self.split_quantiles))
            for thr in thresholds:
                left = xj <= float(thr)
                n_left = int(np.sum(left))
                n_right = n - n_left
                if n_left < min_leaf or n_right < min_leaf:
                    continue

                r_left = residual[left]
                r_right = residual[~left]
                w_left = float(np.mean(r_left))
                w_right = float(np.mean(r_right))

                pred = np.where(left, w_left, w_right)
                mse = float(np.mean((residual - pred) ** 2))
                gain = base_mse - mse
                if gain > best_gain:
                    best_gain = gain
                    best = {
                        "feature": int(j),
                        "threshold": float(thr),
                        "left_weight": w_left,
                        "right_weight": w_right,
                        "gain": gain,
                        "coverage": (n_left / n, n_right / n),
                    }

        if best is None or best_gain < float(self.min_stump_gain):
            return None
        return best

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.feature_medians_ = np.nanmedian(X, axis=0)
        X = self._impute(X)

        self.intercept_, self.coef_, self.active_linear_ = self._fit_sparse_linear(X, y)
        pred = self.intercept_ + X @ self.coef_

        self.stumps_ = []
        for _ in range(int(self.max_stumps)):
            residual = y - pred
            stump = self._best_stump(X, residual)
            if stump is None:
                break

            stump["scale"] = float(self.learning_rate)
            self.stumps_.append(stump)
            left = X[:, stump["feature"]] <= stump["threshold"]
            pred += float(self.learning_rate) * np.where(left, stump["left_weight"], stump["right_weight"])

        self.feature_importance_ = np.abs(self.coef_).copy()
        for stump in self.stumps_:
            f = stump["feature"]
            self.feature_importance_[f] += abs(stump["scale"] * (stump["right_weight"] - stump["left_weight"]))
        self.selected_feature_order_ = np.argsort(self.feature_importance_)[::-1]

        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "coef_", "stumps_"])
        X = self._impute(X)
        yhat = self.intercept_ + X @ self.coef_
        for stump in self.stumps_:
            left = X[:, stump["feature"]] <= stump["threshold"]
            yhat += stump["scale"] * np.where(left, stump["left_weight"], stump["right_weight"])
        return yhat

    def _linear_equation(self):
        terms = [f"{self.intercept_:+.6f}"]
        for j in np.argsort(np.abs(self.coef_))[::-1]:
            c = self.coef_[int(j)]
            if abs(c) >= self.coef_tol:
                terms.append(f"{c:+.6f}*x{int(j)}")
        return "y_base = " + " ".join(terms)

    def __str__(self):
        check_is_fitted(self, ["intercept_", "coef_", "stumps_"])

        lines = [
            "SparseLinearResidualStumpsRegressor",
            "Prediction recipe:",
            f"  {self._linear_equation()}",
        ]

        if not self.stumps_:
            lines.append("  Final prediction: y = y_base")
        else:
            lines.append("  Then apply additive stump adjustments in order:")
            for i, stump in enumerate(self.stumps_, 1):
                s = stump["scale"]
                wl = s * stump["left_weight"]
                wr = s * stump["right_weight"]
                lines.append(
                    f"  Rule {i}: if x{stump['feature']} <= {stump['threshold']:+.6f}, add {wl:+.6f}; else add {wr:+.6f}"
                )
            lines.append("  Final prediction: y = y_base + sum(rule adjustments)")

        lines.append("")
        lines.append("Feature summary (overall effect magnitude):")
        for j in self.selected_feature_order_[: min(12, self.n_features_in_)]:
            lines.append(
                f"  x{int(j)}: linear={self.coef_[int(j)]:+.6f}, importance={self.feature_importance_[int(j)]:.6f}"
            )

        negligible = [f"x{j}" for j in range(self.n_features_in_) if abs(self.coef_[j]) < max(0.02, 0.05 * np.max(np.abs(self.coef_) + 1e-12))]
        if negligible:
            lines.append("Likely negligible linear-effect features: " + ", ".join(negligible))

        ops_est = int(1 + np.sum(np.abs(self.coef_) >= self.coef_tol) + 2 * len(self.stumps_))
        lines.append(f"Compactness estimate: about {ops_est} arithmetic/rule steps to simulate one prediction.")

        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseLinearResidualStumpsRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseLinearResidualStumpsV1"
model_description = "Sparse ridge-like linear backbone plus a short additive list of greedy residual decision stumps with explicit rule equations"
model_defs = [(model_shorthand_name, SparseLinearResidualStumpsRegressor())]

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
