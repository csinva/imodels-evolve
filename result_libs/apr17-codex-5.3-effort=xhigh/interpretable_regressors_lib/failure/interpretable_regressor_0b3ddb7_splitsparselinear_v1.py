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


class SplitSparseLinearRegressor(BaseEstimator, RegressorMixin):
    """Single-split sparse model tree with compact branch-wise linear equations."""

    def __init__(
        self,
        max_features=6,
        ridge_lambda=1e-3,
        max_split_features=10,
        min_leaf_samples=80,
        min_split_rel_gain=0.03,
        negligible_feature_eps=5e-3,
    ):
        self.max_features = max_features
        self.ridge_lambda = ridge_lambda
        self.max_split_features = max_split_features
        self.min_leaf_samples = min_leaf_samples
        self.min_split_rel_gain = min_split_rel_gain
        self.negligible_feature_eps = negligible_feature_eps

    def _safe_abs_corr(self, a, b):
        a_std = float(np.std(a))
        b_std = float(np.std(b))
        if a_std < 1e-12 or b_std < 1e-12:
            return 0.0
        c = np.corrcoef(a, b)[0, 1]
        if not np.isfinite(c):
            return 0.0
        return float(abs(c))

    def _fit_linear(self, X, y, feat_idx):
        if len(feat_idx) == 0:
            return float(np.mean(y)), np.zeros(0, dtype=float)

        Xs = X[:, feat_idx]
        X_mean = np.mean(Xs, axis=0)
        X_std = np.std(Xs, axis=0)
        X_std[X_std < 1e-12] = 1.0
        Z = (Xs - X_mean) / X_std

        y_mean = float(np.mean(y))
        y_centered = y - y_mean

        p = Z.shape[1]
        gram = Z.T @ Z + self.ridge_lambda * np.eye(p)
        rhs = Z.T @ y_centered
        coef_std = np.linalg.solve(gram, rhs)

        coef_raw = coef_std / X_std
        intercept = float(y_mean - np.dot(coef_raw, X_mean))
        return intercept, coef_raw

    def _predict_linear(self, X, feat_idx, intercept, coef):
        if len(feat_idx) == 0:
            return np.full(X.shape[0], intercept, dtype=float)
        return intercept + X[:, feat_idx] @ coef

    def _sse(self, y, yhat):
        r = y - yhat
        return float(np.dot(r, r))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        corr = np.array([self._safe_abs_corr(X[:, j], y) for j in range(n_features)])
        ranked = np.argsort(corr)[::-1]
        k = min(self.max_features, n_features)
        self.feature_idx_ = np.sort(ranked[:k].astype(int))

        g_intercept, g_coef = self._fit_linear(X, y, self.feature_idx_)
        g_pred = self._predict_linear(X, self.feature_idx_, g_intercept, g_coef)
        g_sse = self._sse(y, g_pred)

        self.use_split_ = False
        self.split_feature_ = None
        self.split_threshold_ = None
        self.left_intercept_ = g_intercept
        self.left_coef_ = g_coef
        self.right_intercept_ = g_intercept
        self.right_coef_ = g_coef
        self.global_intercept_ = g_intercept
        self.global_coef_ = g_coef

        split_candidates = ranked[: min(self.max_split_features, n_features)]
        min_gain = self.min_split_rel_gain * (g_sse + 1e-12)

        best = None
        for j in split_candidates:
            xj = X[:, int(j)]
            thresholds = np.unique(np.round(np.quantile(xj, [0.2, 0.35, 0.5, 0.65, 0.8]), 8))
            for t in thresholds:
                left = xj <= t
                right = ~left
                n_left = int(np.sum(left))
                n_right = int(np.sum(right))
                if n_left < self.min_leaf_samples or n_right < self.min_leaf_samples:
                    continue

                li, lc = self._fit_linear(X[left], y[left], self.feature_idx_)
                ri, rc = self._fit_linear(X[right], y[right], self.feature_idx_)

                pred = np.empty(n_samples, dtype=float)
                pred[left] = self._predict_linear(X[left], self.feature_idx_, li, lc)
                pred[right] = self._predict_linear(X[right], self.feature_idx_, ri, rc)
                sse = self._sse(y, pred)

                if best is None or sse < best["sse"]:
                    best = {
                        "feature": int(j),
                        "threshold": float(t),
                        "left_intercept": float(li),
                        "left_coef": lc.copy(),
                        "right_intercept": float(ri),
                        "right_coef": rc.copy(),
                        "sse": sse,
                    }

        if best is not None and (g_sse - best["sse"]) >= min_gain:
            self.use_split_ = True
            self.split_feature_ = best["feature"]
            self.split_threshold_ = best["threshold"]
            self.left_intercept_ = best["left_intercept"]
            self.left_coef_ = best["left_coef"]
            self.right_intercept_ = best["right_intercept"]
            self.right_coef_ = best["right_coef"]

        contrib = np.zeros(n_features, dtype=float)
        feat = self.feature_idx_
        if len(feat):
            if self.use_split_:
                x_left = X[:, feat]
                x_right = X[:, feat]
                c_left = np.mean(np.abs(x_left * self.left_coef_.reshape(1, -1)), axis=0)
                c_right = np.mean(np.abs(x_right * self.right_coef_.reshape(1, -1)), axis=0)
                c = 0.5 * (c_left + c_right)
            else:
                c = np.mean(np.abs(X[:, feat] * self.global_coef_.reshape(1, -1)), axis=0)
            contrib[feat] = c

        if self.use_split_ and self.split_feature_ is not None:
            contrib[self.split_feature_] += np.std(y) * 0.1

        self.feature_importances_ = contrib
        return self

    def predict(self, X):
        check_is_fitted(self, ["feature_idx_", "use_split_", "global_intercept_", "global_coef_"])
        X = np.asarray(X, dtype=float)

        if not self.use_split_:
            return self._predict_linear(X, self.feature_idx_, self.global_intercept_, self.global_coef_)

        left = X[:, self.split_feature_] <= self.split_threshold_
        right = ~left
        pred = np.empty(X.shape[0], dtype=float)
        if np.any(left):
            pred[left] = self._predict_linear(X[left], self.feature_idx_, self.left_intercept_, self.left_coef_)
        if np.any(right):
            pred[right] = self._predict_linear(X[right], self.feature_idx_, self.right_intercept_, self.right_coef_)
        return pred

    def _linear_expr(self, intercept, coef):
        pieces = [f"{intercept:+.5f}"]
        for j, c in zip(self.feature_idx_, coef):
            pieces.append(f"{c:+.5f}*x{j}")
        return " ".join(pieces)

    def __str__(self):
        check_is_fitted(self, ["feature_idx_", "use_split_", "feature_importances_"])

        active = [f"x{j}" for j in self.feature_idx_]
        inactive = [
            f"x{j}"
            for j in range(self.n_features_in_)
            if self.feature_importances_[j] <= self.negligible_feature_eps and j not in self.feature_idx_
        ]

        order = np.argsort(self.feature_importances_)[::-1]
        top = [f"x{j}:{self.feature_importances_[j]:.4f}" for j in order[: min(10, self.n_features_in_)]]

        lines = [
            "Split Sparse Linear Regressor (single-rule model tree)",
            f"Active linear features ({len(active)}): " + (", ".join(active) if active else "none"),
        ]

        if self.use_split_:
            lines.extend([
                f"Rule: if x{self.split_feature_} <= {self.split_threshold_:.5f}",
                "  y = " + self._linear_expr(self.left_intercept_, self.left_coef_),
                "Else:",
                "  y = " + self._linear_expr(self.right_intercept_, self.right_coef_),
            ])
        else:
            lines.extend([
                "No split selected (global linear model):",
                "  y = " + self._linear_expr(self.global_intercept_, self.global_coef_),
            ])

        lines.extend([
            "",
            "Most influential features (mean absolute contribution):",
            "  " + ", ".join(top),
            "Features with negligible effect:",
            "  " + (", ".join(inactive) if inactive else "none"),
            "",
            "Simulation tip: choose branch by the rule, then plug feature values into that branch equation.",
        ])
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SplitSparseLinearRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SplitSparseLinear_v1"
model_description = "Single-split sparse linear model tree with branch-specific equations and explicit simulation-friendly rule output"
model_defs = [(model_shorthand_name, SplitSparseLinearRegressor())]


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
