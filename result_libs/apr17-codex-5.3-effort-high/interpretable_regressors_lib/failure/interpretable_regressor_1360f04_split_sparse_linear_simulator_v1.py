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


class SplitSparseLinearSimulatorV1(BaseEstimator, RegressorMixin):
    """
    One-threshold piecewise sparse linear regressor with explicit equations.

    Workflow:
    1) Fit a global ridge and keep only a tiny active feature set.
    2) Search one threshold on top global features using a validation split.
    3) Fit one sparse linear equation per side of the threshold.
    """

    def __init__(
        self,
        alpha_grid=None,
        max_active_features=3,
        candidate_split_features=2,
        n_thresholds=5,
        min_leaf_frac=0.12,
        min_split_gain=1e-4,
        seed=42,
        display_precision=5,
    ):
        self.alpha_grid = alpha_grid
        self.max_active_features = max_active_features
        self.candidate_split_features = candidate_split_features
        self.n_thresholds = n_thresholds
        self.min_leaf_frac = min_leaf_frac
        self.min_split_gain = min_split_gain
        self.seed = seed
        self.display_precision = display_precision

    @staticmethod
    def _safe_standardize(X):
        mean = np.mean(X, axis=0)
        scale = np.std(X, axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        return mean.astype(float), scale.astype(float)

    @staticmethod
    def _ridge_closed_form(Xs, y, alpha):
        n = Xs.shape[0]
        D = np.column_stack([np.ones(n, dtype=float), Xs])
        reg = np.eye(D.shape[1], dtype=float)
        reg[0, 0] = 0.0
        A = D.T @ D + float(alpha) * reg
        b = D.T @ y
        try:
            theta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            theta = np.linalg.pinv(A) @ b
        return np.asarray(theta, dtype=float)

    def _fit_raw_ridge(self, X, y, alpha):
        mean, scale = self._safe_standardize(X)
        Xs = (X - mean) / scale
        theta = self._ridge_closed_form(Xs, y, alpha)
        coef_raw = theta[1:] / scale
        intercept_raw = float(theta[0] - np.dot(coef_raw, mean))
        return intercept_raw, np.asarray(coef_raw, dtype=float)

    def _predict_raw_linear(self, X, intercept, coef):
        return float(intercept) + np.asarray(X, dtype=float) @ np.asarray(coef, dtype=float)

    def _select_alpha_holdout(self, X, y):
        alphas = (
            np.asarray(self.alpha_grid, dtype=float)
            if self.alpha_grid is not None
            else np.logspace(-5, 2, 12)
        )
        n = X.shape[0]
        if n < 12:
            return float(alphas[0]), float("nan")

        rng = np.random.RandomState(self.seed)
        idx = np.arange(n)
        rng.shuffle(idx)
        n_val = max(8, int(0.25 * n))
        val_idx = idx[:n_val]
        tr_idx = idx[n_val:]

        best_alpha = float(alphas[0])
        best_mse = float("inf")
        for alpha in alphas:
            intercept, coef = self._fit_raw_ridge(X[tr_idx], y[tr_idx], float(alpha))
            preds = self._predict_raw_linear(X[val_idx], intercept, coef)
            mse = float(np.mean((y[val_idx] - preds) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_alpha = float(alpha)
        return best_alpha, best_mse

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        self.n_features_in_ = p

        alpha, alpha_val_mse = self._select_alpha_holdout(X, y)
        global_intercept, global_coef = self._fit_raw_ridge(X, y, alpha)
        abs_global = np.abs(global_coef)
        order = np.argsort(-abs_global)

        k = int(max(1, min(self.max_active_features, p)))
        active = order[:k].astype(int)
        active_set = set(active.tolist())
        negligible = np.array([j for j in range(p) if j not in active_set], dtype=int)

        rng = np.random.RandomState(self.seed)
        idx = np.arange(n)
        rng.shuffle(idx)
        n_val = max(8, int(0.25 * n))
        val_idx = idx[:n_val]
        tr_idx = idx[n_val:]

        X_tr = X[tr_idx]
        y_tr = y[tr_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]

        base_intercept, base_coef_act = self._fit_raw_ridge(X_tr[:, active], y_tr, alpha)
        base_val_pred = self._predict_raw_linear(X_val[:, active], base_intercept, base_coef_act)
        base_val_mse = float(np.mean((y_val - base_val_pred) ** 2))

        split_candidates = order[: max(1, min(self.candidate_split_features, p))].astype(int)
        quantiles = np.linspace(0.2, 0.8, int(max(2, self.n_thresholds)))
        min_leaf = max(8, int(self.min_leaf_frac * len(tr_idx)))

        best = {
            "mse": base_val_mse,
            "feature": None,
            "threshold": None,
            "left": (base_intercept, base_coef_act.copy()),
            "right": (base_intercept, base_coef_act.copy()),
        }

        for feat in split_candidates:
            thrs = np.unique(np.quantile(X_tr[:, feat], quantiles))
            for thr in thrs:
                left_mask = X_tr[:, feat] <= thr
                right_mask = ~left_mask
                if left_mask.sum() < min_leaf or right_mask.sum() < min_leaf:
                    continue
                left_intercept, left_coef = self._fit_raw_ridge(
                    X_tr[left_mask][:, active], y_tr[left_mask], alpha
                )
                right_intercept, right_coef = self._fit_raw_ridge(
                    X_tr[right_mask][:, active], y_tr[right_mask], alpha
                )
                pred_val = np.empty_like(y_val, dtype=float)
                val_left = X_val[:, feat] <= thr
                pred_val[val_left] = self._predict_raw_linear(
                    X_val[val_left][:, active], left_intercept, left_coef
                )
                pred_val[~val_left] = self._predict_raw_linear(
                    X_val[~val_left][:, active], right_intercept, right_coef
                )
                mse = float(np.mean((y_val - pred_val) ** 2))
                if mse < best["mse"]:
                    best = {
                        "mse": mse,
                        "feature": int(feat),
                        "threshold": float(thr),
                        "left": (left_intercept, left_coef),
                        "right": (right_intercept, right_coef),
                    }

        use_split = (
            best["feature"] is not None
            and (base_val_mse - best["mse"]) > float(self.min_split_gain)
        )

        if use_split:
            feat = int(best["feature"])
            thr = float(best["threshold"])
            left_all = X[:, feat] <= thr
            right_all = ~left_all
            left_intercept, left_coef = self._fit_raw_ridge(X[left_all][:, active], y[left_all], alpha)
            right_intercept, right_coef = self._fit_raw_ridge(X[right_all][:, active], y[right_all], alpha)

            self.use_split_ = True
            self.split_feature_ = feat
            self.split_threshold_ = thr
            self.left_intercept_ = float(left_intercept)
            self.right_intercept_ = float(right_intercept)
            self.left_coef_active_ = np.asarray(left_coef, dtype=float)
            self.right_coef_active_ = np.asarray(right_coef, dtype=float)
        else:
            all_intercept, all_coef = self._fit_raw_ridge(X[:, active], y, alpha)
            self.use_split_ = False
            self.split_feature_ = int(order[0]) if p > 0 else 0
            self.split_threshold_ = 0.0
            self.left_intercept_ = float(all_intercept)
            self.right_intercept_ = float(all_intercept)
            self.left_coef_active_ = np.asarray(all_coef, dtype=float)
            self.right_coef_active_ = np.asarray(all_coef, dtype=float)

        self.active_features_ = np.asarray(active, dtype=int)
        self.negligible_features_ = negligible
        self.global_intercept_ = float(global_intercept)
        self.global_coef_ = np.asarray(global_coef, dtype=float)
        self.sorted_features_ = order.astype(int)
        self.alpha_ = float(alpha)
        self.alpha_holdout_mse_ = float(alpha_val_mse)
        self.validation_mse_no_split_ = float(base_val_mse)
        self.validation_mse_best_split_ = float(best["mse"])
        self.dominant_feature_ = int(order[0]) if p > 0 else 0
        return self

    def predict(self, X):
        check_is_fitted(self, ["active_features_", "left_intercept_", "left_coef_active_"])
        X = np.asarray(X, dtype=float)
        X_act = X[:, self.active_features_]
        if not self.use_split_:
            return self._predict_raw_linear(X_act, self.left_intercept_, self.left_coef_active_)

        pred = np.empty(X.shape[0], dtype=float)
        left = X[:, self.split_feature_] <= self.split_threshold_
        if np.any(left):
            pred[left] = self._predict_raw_linear(X_act[left], self.left_intercept_, self.left_coef_active_)
        if np.any(~left):
            pred[~left] = self._predict_raw_linear(X_act[~left], self.right_intercept_, self.right_coef_active_)
        return pred

    def _format_equation(self, intercept, coef_active):
        prec = int(self.display_precision)
        terms = [f"{float(intercept):+.{prec}f}"]
        for local_i, feat in enumerate(self.active_features_):
            terms.append(f"{float(coef_active[local_i]):+.{prec}f}*x{int(feat)}")
        return " ".join(terms)

    def __str__(self):
        check_is_fitted(self, ["active_features_", "left_intercept_", "left_coef_active_"])
        lines = [
            "Split Sparse Linear Simulator",
            "Prediction uses at most one threshold check and a short linear equation.",
            f"Active features: {', '.join(f'x{int(j)}' for j in self.active_features_)}",
        ]
        if len(self.negligible_features_) > 0:
            lines.append("Negligible features: " + ", ".join(f"x{int(j)}" for j in self.negligible_features_))

        if self.use_split_:
            lines.extend(
                [
                    "",
                    f"Rule: if x{self.split_feature_} <= {self.split_threshold_:.5f}",
                    "  y = " + self._format_equation(self.left_intercept_, self.left_coef_active_),
                    "Else",
                    "  y = " + self._format_equation(self.right_intercept_, self.right_coef_active_),
                    "",
                    "Simulation: choose the branch with the threshold rule, then multiply-and-sum active terms plus intercept.",
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "Single equation (no split kept):",
                    "  y = " + self._format_equation(self.left_intercept_, self.left_coef_active_),
                    "",
                    "Simulation: multiply each active feature by its coefficient, sum, then add intercept.",
                ]
            )

        lines.append("")
        lines.append(f"Dominant global feature: x{self.dominant_feature_}")
        lines.append(f"Selected ridge alpha: {self.alpha_:.6g}")
        return "\n".join(lines)


import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SplitSparseLinearSimulatorV1.__module__ = "interpretable_regressor"

model_shorthand_name = "SplitSparseLinearSimulatorV1"
model_description = "One-threshold piecewise sparse linear simulator: holdout-selected ridge shrinkage, tiny active feature set, and optional two-region linear equations with an explicit branch rule"
model_defs = [(model_shorthand_name, SplitSparseLinearSimulatorV1())]

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
