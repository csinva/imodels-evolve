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


class AxisGatedSparseLinearRegressor(BaseEstimator, RegressorMixin):
    """
    One-split sparse linear model:
      - fit a compact global linear equation
      - optionally split on one feature threshold
      - fit compact linear equations on each side
    """

    def __init__(
        self,
        top_features=12,
        leaf_features=5,
        split_feature_candidates=6,
        split_quantiles=(0.2, 0.35, 0.5, 0.65, 0.8),
        ridge_alpha=2e-2,
        min_leaf=40,
        min_gain=1e-4,
        coef_tol=1e-4,
        random_state=42,
    ):
        self.top_features = top_features
        self.leaf_features = leaf_features
        self.split_feature_candidates = split_feature_candidates
        self.split_quantiles = split_quantiles
        self.ridge_alpha = ridge_alpha
        self.min_leaf = min_leaf
        self.min_gain = min_gain
        self.coef_tol = coef_tol
        self.random_state = random_state

    def _impute(self, X):
        X = np.asarray(X, dtype=float).copy()
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(self.feature_medians_, np.where(bad)[1])
        return X

    @staticmethod
    def _abs_corr(x, y):
        xc = x - np.mean(x)
        yc = y - np.mean(y)
        den = (np.linalg.norm(xc) * np.linalg.norm(yc)) + 1e-12
        return abs(float(np.dot(xc, yc) / den))

    def _fit_sparse_linear(self, X, y, feature_pool):
        n = X.shape[0]
        if n == 0:
            return {"intercept": 0.0, "features": np.array([], dtype=int), "coefs": np.array([], dtype=float)}

        pool = np.array(feature_pool, dtype=int)
        if len(pool) == 0:
            return {"intercept": float(np.mean(y)), "features": np.array([], dtype=int), "coefs": np.array([], dtype=float)}

        Z = X[:, pool]
        A = np.column_stack([np.ones(n), Z])
        gram = A.T @ A
        reg = float(self.ridge_alpha) * np.eye(gram.shape[0])
        reg[0, 0] = 0.0
        beta = np.linalg.solve(gram + reg, A.T @ y)
        coefs_pool = beta[1:]

        keep_k = min(int(self.leaf_features), len(pool))
        keep_local = np.argsort(np.abs(coefs_pool))[::-1][:keep_k]
        keep_local = keep_local[np.abs(coefs_pool[keep_local]) > self.coef_tol]
        keep_feats = pool[keep_local]

        if len(keep_feats) == 0:
            return {"intercept": float(np.mean(y)), "features": np.array([], dtype=int), "coefs": np.array([], dtype=float)}

        Z2 = X[:, keep_feats]
        A2 = np.column_stack([np.ones(n), Z2])
        gram2 = A2.T @ A2
        reg2 = float(self.ridge_alpha) * np.eye(gram2.shape[0])
        reg2[0, 0] = 0.0
        beta2 = np.linalg.solve(gram2 + reg2, A2.T @ y)
        return {
            "intercept": float(beta2[0]),
            "features": keep_feats.astype(int),
            "coefs": beta2[1:].astype(float),
        }

    @staticmethod
    def _predict_linear(model_dict, X):
        out = np.full(X.shape[0], model_dict["intercept"], dtype=float)
        if len(model_dict["features"]) > 0:
            out += X[:, model_dict["features"]] @ model_dict["coefs"]
        return out

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.feature_medians_ = np.nanmedian(X, axis=0)
        X = self._impute(X)
        n, p = X.shape

        scores = np.array([self._abs_corr(X[:, j], y) for j in range(p)])
        top = np.argsort(scores)[::-1][: min(int(self.top_features), p)]
        top = np.array(top, dtype=int)

        global_model = self._fit_sparse_linear(X, y, top)
        global_pred = self._predict_linear(global_model, X)
        best_mse = float(np.mean((y - global_pred) ** 2))

        self.use_split_ = False
        self.split_feature_ = None
        self.split_threshold_ = None
        self.left_model_ = global_model
        self.right_model_ = global_model

        split_feats = top[: min(int(self.split_feature_candidates), len(top))]
        for j in split_feats:
            xj = X[:, j]
            for q in self.split_quantiles:
                thr = float(np.quantile(xj, q))
                left = xj <= thr
                right = ~left
                if left.sum() < self.min_leaf or right.sum() < self.min_leaf:
                    continue

                left_model = self._fit_sparse_linear(X[left], y[left], top)
                right_model = self._fit_sparse_linear(X[right], y[right], top)
                pred = np.empty(n, dtype=float)
                pred[left] = self._predict_linear(left_model, X[left])
                pred[right] = self._predict_linear(right_model, X[right])
                mse = float(np.mean((y - pred) ** 2))

                if (best_mse - mse) > self.min_gain:
                    best_mse = mse
                    self.use_split_ = True
                    self.split_feature_ = int(j)
                    self.split_threshold_ = float(thr)
                    self.left_model_ = left_model
                    self.right_model_ = right_model

        self.intercept_ = float(global_model["intercept"])
        self.coef_ = np.zeros(self.n_features_in_, dtype=float)
        if len(global_model["features"]) > 0:
            self.coef_[global_model["features"]] = global_model["coefs"]

        fi = np.zeros(self.n_features_in_, dtype=float)
        for model_dict in [self.left_model_, self.right_model_]:
            for j, c in zip(model_dict["features"], model_dict["coefs"]):
                fi[int(j)] += abs(float(c))
        self.feature_importance_ = fi
        self.selected_feature_order_ = np.argsort(fi)[::-1]
        return self

    def predict(self, X):
        check_is_fitted(self, ["use_split_", "left_model_", "right_model_", "feature_importance_"])
        X = self._impute(X)
        if not self.use_split_:
            return self._predict_linear(self.left_model_, X)
        left = X[:, self.split_feature_] <= self.split_threshold_
        out = np.empty(X.shape[0], dtype=float)
        out[left] = self._predict_linear(self.left_model_, X[left])
        out[~left] = self._predict_linear(self.right_model_, X[~left])
        return out

    @staticmethod
    def _equation_str(model_dict):
        parts = [f"{model_dict['intercept']:+.6f}"]
        for j, c in zip(model_dict["features"], model_dict["coefs"]):
            parts.append(f"({float(c):+.6f}*x{int(j)})")
        return " + ".join(parts)

    def __str__(self):
        check_is_fitted(self, ["use_split_", "left_model_", "right_model_", "feature_importance_"])
        lines = [
            "AxisGatedSparseLinearRegressor",
            "Prediction rule:",
        ]
        if not self.use_split_:
            lines.append("  y = " + self._equation_str(self.left_model_))
            return "\n".join(lines)

        lines.extend([
            f"  if x{self.split_feature_} <= {self.split_threshold_:.6f}:",
            "    y = " + self._equation_str(self.left_model_),
            "  else:",
            "    y = " + self._equation_str(self.right_model_),
        ])
        return "\n".join(lines)



# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
AxisGatedSparseLinearRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "AxisGatedSparseLinearV1"
model_description = "Compact sparse linear model with one optional axis threshold split into two sparse linear leaf equations"
model_defs = [(model_shorthand_name, AxisGatedSparseLinearRegressor())]


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
