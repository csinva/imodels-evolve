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


class SparseModelTreeRegressor(BaseEstimator, RegressorMixin):
    """
    Compact one-split model tree:
      1) fit a sparse linear backbone with GCV ridge + hard term pruning,
      2) optionally add one threshold split and fit sparse linear equations in
         each branch if the split materially improves MSE.

    The final model is always explicit arithmetic that can be simulated from text.
    """

    def __init__(
        self,
        alphas=(0.0, 1e-4, 1e-3, 1e-2, 5e-2, 0.2, 1.0, 5.0),
        max_linear_terms=10,
        min_coef_rel=0.08,
        split_screen_features=6,
        split_quantiles=(0.2, 0.4, 0.5, 0.6, 0.8),
        min_leaf_samples=40,
        min_split_gain=0.03,
        split_penalty=0.01,
        refit_ridge=1e-3,
        coef_tol=1e-10,
        meaningful_rel=0.12,
    ):
        self.alphas = alphas
        self.max_linear_terms = max_linear_terms
        self.min_coef_rel = min_coef_rel
        self.split_screen_features = split_screen_features
        self.split_quantiles = split_quantiles
        self.min_leaf_samples = min_leaf_samples
        self.min_split_gain = min_split_gain
        self.split_penalty = split_penalty
        self.refit_ridge = refit_ridge
        self.coef_tol = coef_tol
        self.meaningful_rel = meaningful_rel

    @staticmethod
    def _solve_linear_system_with_intercept(D, y, ridge):
        n_samples = D.shape[0]
        if D.shape[1] == 0:
            return float(np.mean(y)), np.zeros(0, dtype=float)
        A = np.column_stack([np.ones(n_samples, dtype=float), D])
        reg = np.eye(A.shape[1], dtype=float)
        reg[0, 0] = 0.0
        lhs = A.T @ A + max(float(ridge), 0.0) * reg
        rhs = A.T @ y
        try:
            sol = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        return float(sol[0]), np.asarray(sol[1:], dtype=float)

    def _fit_ridge_gcv(self, X, y):
        n_samples, n_features = X.shape
        if n_features == 0:
            return 0.0, np.zeros(0, dtype=float), np.zeros(0, dtype=float), np.ones(0, dtype=float)

        x_mean = X.mean(axis=0)
        x_scale = X.std(axis=0)
        x_scale[x_scale < 1e-12] = 1.0
        Xs = (X - x_mean) / x_scale

        y_mean = float(y.mean())
        y_centered = y - y_mean

        U, svals, Vt = np.linalg.svd(Xs, full_matrices=False)
        uy = U.T @ y_centered
        s2 = svals * svals

        best_alpha = 0.0
        best_score = np.inf
        best_coef_std = np.zeros(n_features, dtype=float)

        for alpha in self.alphas:
            a = max(float(alpha), 0.0)
            denom = s2 + a
            shrink = s2 / denom
            y_hat = U @ (shrink * uy)
            resid = y_centered - y_hat
            rss = float(resid @ resid)
            df = float(np.sum(shrink))
            gcv = rss / max((n_samples - df) ** 2, 1e-8)
            if gcv < best_score:
                best_score = gcv
                best_alpha = a
                best_coef_std = Vt.T @ ((svals / denom) * uy)

        coef = best_coef_std / x_scale
        return float(best_alpha), np.asarray(coef, dtype=float), x_mean, x_scale

    def _fit_sparse_linear(self, X, y):
        n_features = X.shape[1]
        _, dense_coef, _, _ = self._fit_ridge_gcv(X, y)

        abs_coef = np.abs(dense_coef)
        keep = np.array([], dtype=int)
        if n_features > 0:
            max_keep = min(int(self.max_linear_terms), n_features)
            if max_keep > 0:
                order = np.argsort(abs_coef)[::-1]
                top = order[:max_keep]
                rel_thr = float(self.min_coef_rel) * (float(abs_coef[top[0]]) if len(top) > 0 else 0.0)
                keep = np.array([int(j) for j in top if abs_coef[int(j)] >= rel_thr], dtype=int)
                if keep.size == 0 and top.size > 0:
                    keep = np.array([int(top[0])], dtype=int)

        if keep.size == 0:
            intercept = float(np.mean(y))
            coef = np.zeros(n_features, dtype=float)
            pred = np.full(X.shape[0], intercept, dtype=float)
            mse = float(np.mean((y - pred) ** 2))
            return intercept, coef, pred, mse, keep

        keep.sort()
        D = X[:, keep]
        intercept, small_coef = self._solve_linear_system_with_intercept(D, y, self.refit_ridge)
        coef = np.zeros(n_features, dtype=float)
        coef[keep] = small_coef
        pred = intercept + D @ small_coef
        mse = float(np.mean((y - pred) ** 2))
        return float(intercept), coef, pred, mse, keep

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        root_intercept, root_coef, root_pred, root_mse, root_keep = self._fit_sparse_linear(X, y)
        self.root_intercept_ = float(root_intercept)
        self.root_coef_ = np.asarray(root_coef, dtype=float)
        self.root_mse_ = float(root_mse)
        self.root_keep_ = np.asarray(root_keep, dtype=int)

        centered_X = X - X.mean(axis=0, keepdims=True)
        centered_y = y - float(y.mean())
        x_norm = np.sqrt(np.sum(centered_X * centered_X, axis=0))
        y_norm = float(np.sqrt(np.sum(centered_y * centered_y)) + 1e-12)
        corr = np.abs(centered_X.T @ centered_y) / (x_norm * y_norm + 1e-12)
        corr[np.isnan(corr)] = 0.0
        split_features = np.argsort(corr)[::-1][: min(int(self.split_screen_features), n_features)]

        best = None
        best_adj_mse = float("inf")

        min_leaf = max(int(self.min_leaf_samples), 5)
        for feat in split_features:
            xj = X[:, int(feat)]
            thresholds = np.unique(np.quantile(xj, self.split_quantiles))
            for thr in thresholds:
                mask = xj <= float(thr)
                n_left = int(np.sum(mask))
                n_right = n_samples - n_left
                if n_left < min_leaf or n_right < min_leaf:
                    continue

                left_inter, left_coef, left_pred, _, left_keep = self._fit_sparse_linear(X[mask], y[mask])
                right_inter, right_coef, right_pred, _, right_keep = self._fit_sparse_linear(X[~mask], y[~mask])

                pred = np.empty(n_samples, dtype=float)
                pred[mask] = left_pred
                pred[~mask] = right_pred
                mse = float(np.mean((y - pred) ** 2))

                root_terms = max(int(np.sum(np.abs(root_coef) > self.coef_tol)), 1)
                branch_terms = int(np.sum(np.abs(left_coef) > self.coef_tol) + np.sum(np.abs(right_coef) > self.coef_tol))
                complexity = max(branch_terms - root_terms, 0) / max(1, n_features)
                adj_mse = mse * (1.0 + float(self.split_penalty) * complexity)

                if adj_mse < best_adj_mse:
                    best_adj_mse = adj_mse
                    best = {
                        "feature": int(feat),
                        "threshold": float(thr),
                        "left_intercept": float(left_inter),
                        "left_coef": np.asarray(left_coef, dtype=float),
                        "right_intercept": float(right_inter),
                        "right_coef": np.asarray(right_coef, dtype=float),
                        "raw_mse": float(mse),
                        "left_count": n_left,
                        "right_count": n_right,
                        "left_keep": np.asarray(left_keep, dtype=int),
                        "right_keep": np.asarray(right_keep, dtype=int),
                    }

        self.has_split_ = False
        if best is not None:
            rel_gain = (root_mse - best["raw_mse"]) / (abs(root_mse) + 1e-12)
            if rel_gain >= float(self.min_split_gain):
                self.has_split_ = True
                self.split_feature_ = int(best["feature"])
                self.split_threshold_ = float(best["threshold"])
                self.left_intercept_ = float(best["left_intercept"])
                self.left_coef_ = np.asarray(best["left_coef"], dtype=float)
                self.right_intercept_ = float(best["right_intercept"])
                self.right_coef_ = np.asarray(best["right_coef"], dtype=float)
                self.left_count_ = int(best["left_count"])
                self.right_count_ = int(best["right_count"])
                self.left_keep_ = np.asarray(best["left_keep"], dtype=int)
                self.right_keep_ = np.asarray(best["right_keep"], dtype=int)

        if self.has_split_:
            w_left = float(self.left_count_) / max(n_samples, 1)
            w_right = float(self.right_count_) / max(n_samples, 1)
            self.feature_importance_ = w_left * np.abs(self.left_coef_) + w_right * np.abs(self.right_coef_)
        else:
            self.feature_importance_ = np.abs(self.root_coef_)

        self.selected_features_ = sorted(int(i) for i in np.where(self.feature_importance_ > self.coef_tol)[0])
        return self

    def predict(self, X):
        check_is_fitted(self, ["root_intercept_", "root_coef_", "has_split_"])
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if not self.has_split_:
            return self.root_intercept_ + X @ self.root_coef_

        mask = X[:, self.split_feature_] <= self.split_threshold_
        pred = np.empty(X.shape[0], dtype=float)
        if np.any(mask):
            pred[mask] = self.left_intercept_ + X[mask] @ self.left_coef_
        if np.any(~mask):
            pred[~mask] = self.right_intercept_ + X[~mask] @ self.right_coef_
        return pred

    def _equation_line(self, intercept, coef):
        terms = [f"{float(intercept):+.6f}"]
        nz = np.where(np.abs(coef) > self.coef_tol)[0]
        for j in nz:
            terms.append(f"{float(coef[j]):+.6f}*x{int(j)}")
        return " ".join(terms)

    def __str__(self):
        check_is_fitted(self, ["feature_importance_", "selected_features_", "has_split_"])
        lines = [
            "Sparse Model Tree Regressor",
            "Compute prediction with exact arithmetic below.",
        ]

        if self.has_split_:
            lines.append(f"Step 1: Check split condition: x{self.split_feature_} <= {self.split_threshold_:.6f}")
            lines.append("Step 2: If TRUE (left branch), compute:")
            lines.append("  y = " + self._equation_line(self.left_intercept_, self.left_coef_))
            lines.append("Step 3: If FALSE (right branch), compute:")
            lines.append("  y = " + self._equation_line(self.right_intercept_, self.right_coef_))
        else:
            lines.append("Single-equation model (no split):")
            lines.append("  y = " + self._equation_line(self.root_intercept_, self.root_coef_))

        active = [int(i) for i in self.selected_features_]
        lines.append("Features used: " + (", ".join(f"x{i}" for i in active) if active else "none"))

        if self.feature_importance_.size > 0:
            max_imp = float(np.max(self.feature_importance_))
            if max_imp > 0:
                threshold = float(self.meaningful_rel) * max_imp
                meaningful = [f"x{i}" for i in range(self.n_features_in_) if self.feature_importance_[i] >= threshold]
                lines.append(
                    "Meaningful features (>= "
                    f"{float(self.meaningful_rel):.2f} * max importance): "
                    + (", ".join(meaningful) if meaningful else "none")
                )

        if self.n_features_in_ <= 30 and len(active) < self.n_features_in_:
            active_set = set(active)
            zero_feats = [f"x{i}" for i in range(self.n_features_in_) if i not in active_set]
            lines.append("Zero-contribution features: " + ", ".join(zero_feats))
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseModelTreeRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseModelTree_v1"
model_description = "Sparse GCV-ridge linear backbone with optional one-threshold split into two compact sparse linear branch equations"
model_defs = [(model_shorthand_name, SparseModelTreeRegressor())]


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
