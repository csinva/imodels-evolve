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


class RidgePlusStepRuleRegressor(BaseEstimator, RegressorMixin):
    """
    Dense ridge backbone in raw feature space plus one optional step-rule residual.

    y = b0 + sum_j b_j * x_j
      + (dl if x_k <= t else dh)
    """

    def __init__(
        self,
        alpha_grid=(1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0),
        val_frac=0.2,
        min_val_samples=120,
        max_step_features=12,
        step_quantiles=(0.15, 0.3, 0.5, 0.7, 0.85),
        min_leaf_samples=30,
        step_shrink=0.35,
        min_gain_step=0.004,
        coef_prune_abs=1e-6,
        coef_decimals=5,
        random_state=42,
    ):
        self.alpha_grid = alpha_grid
        self.val_frac = val_frac
        self.min_val_samples = min_val_samples
        self.max_step_features = max_step_features
        self.step_quantiles = step_quantiles
        self.min_leaf_samples = min_leaf_samples
        self.step_shrink = step_shrink
        self.min_gain_step = min_gain_step
        self.coef_prune_abs = coef_prune_abs
        self.coef_decimals = coef_decimals
        self.random_state = random_state

    @staticmethod
    def _ridge_closed_form(Z, y, alpha):
        p = Z.shape[1]
        reg = float(alpha) * np.eye(p)
        reg[0, 0] = 0.0
        A = Z.T @ Z + reg
        b = Z.T @ y
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(A) @ b

    def _split_idx(self, n):
        if n < int(self.min_val_samples) + 20:
            idx = np.arange(n)
            return idx, idx
        rng = np.random.RandomState(self.random_state)
        perm = rng.permutation(n)
        n_val = max(int(round(float(self.val_frac) * n)), int(self.min_val_samples))
        n_val = min(n_val, n // 2)
        return perm[n_val:], perm[:n_val]

    def _impute(self, X):
        X = np.asarray(X, dtype=float).copy()
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(self.feature_medians_, np.where(bad)[1])
        return X

    @staticmethod
    def _to_raw_equation(beta_std, x_mean, x_std):
        coef_std = beta_std[1:]
        coef_raw = coef_std / x_std
        intercept_raw = float(beta_std[0] - np.sum((coef_std * x_mean) / x_std))
        return intercept_raw, coef_raw

    def _fit_ridge_alpha(self, Xtr, ytr, Xval, yval, x_mean, x_std):
        Xtr_s = (Xtr - x_mean) / x_std
        Xval_s = (Xval - x_mean) / x_std
        Ztr = np.column_stack([np.ones(Xtr_s.shape[0]), Xtr_s])
        Zval = np.column_stack([np.ones(Xval_s.shape[0]), Xval_s])

        best_alpha = float(self.alpha_grid[0])
        best_beta = self._ridge_closed_form(Ztr, ytr, best_alpha)
        best_mse = float(np.mean((yval - Zval @ best_beta) ** 2))

        for alpha in self.alpha_grid[1:]:
            beta = self._ridge_closed_form(Ztr, ytr, float(alpha))
            mse = float(np.mean((yval - Zval @ beta) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_alpha = float(alpha)
                best_beta = beta

        intercept_raw, coef_raw = self._to_raw_equation(best_beta, x_mean, x_std)
        return intercept_raw, coef_raw, best_alpha, best_mse

    @staticmethod
    def _apply_step(x_col, threshold, left_offset, right_offset):
        return np.where(x_col <= threshold, left_offset, right_offset)

    def _search_step(self, Xtr, ytr, Xval, yval, pred_tr, pred_val, coef_raw):
        base_mse = float(np.mean((yval - pred_val) ** 2))
        best = {
            "mse": base_mse,
            "feature": -1,
            "threshold": 0.0,
            "left": 0.0,
            "right": 0.0,
            "used": False,
        }

        n_features = Xtr.shape[1]
        n_scan = min(int(self.max_step_features), n_features)
        feat_order = np.argsort(np.abs(coef_raw))[::-1][:n_scan]

        residual_tr = ytr - pred_tr
        for j in feat_order:
            xtr_j = Xtr[:, j]
            xval_j = Xval[:, j]
            thresholds = [float(np.quantile(xtr_j, q)) for q in self.step_quantiles]
            thresholds = sorted(set(thresholds))
            for t in thresholds:
                left = xtr_j <= t
                right = ~left
                if left.sum() < int(self.min_leaf_samples) or right.sum() < int(self.min_leaf_samples):
                    continue
                left_offset = float(self.step_shrink) * float(np.mean(residual_tr[left]))
                right_offset = float(self.step_shrink) * float(np.mean(residual_tr[right]))
                pred_val_step = pred_val + self._apply_step(xval_j, t, left_offset, right_offset)
                mse = float(np.mean((yval - pred_val_step) ** 2))
                if mse < best["mse"]:
                    best = {
                        "mse": mse,
                        "feature": int(j),
                        "threshold": float(t),
                        "left": float(left_offset),
                        "right": float(right_offset),
                        "used": True,
                    }

        use_step = best["used"] and best["mse"] < base_mse * (1.0 - float(self.min_gain_step))
        best["used"] = bool(use_step)
        return best, base_mse

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]

        self.feature_medians_ = np.nanmedian(X, axis=0)
        X = self._impute(X)

        tr_idx, val_idx = self._split_idx(X.shape[0])
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xval, yval = X[val_idx], y[val_idx]

        x_mean = np.mean(Xtr, axis=0)
        x_std = np.std(Xtr, axis=0)
        x_std[x_std < 1e-12] = 1.0

        intercept, coef, alpha, _ = self._fit_ridge_alpha(Xtr, ytr, Xval, yval, x_mean, x_std)
        pred_tr = intercept + Xtr @ coef
        pred_val = intercept + Xval @ coef

        step_info, _ = self._search_step(Xtr, ytr, Xval, yval, pred_tr, pred_val, coef)

        # Refit the final dense backbone on all data using selected alpha.
        x_mean_all = np.mean(X, axis=0)
        x_std_all = np.std(X, axis=0)
        x_std_all[x_std_all < 1e-12] = 1.0
        X_all_s = (X - x_mean_all) / x_std_all
        Zall = np.column_stack([np.ones(X.shape[0]), X_all_s])
        beta_all = self._ridge_closed_form(Zall, y, alpha)
        intercept_all, coef_all = self._to_raw_equation(beta_all, x_mean_all, x_std_all)

        self.alpha_ = float(alpha)
        self.x_mean_ = x_mean_all
        self.x_std_ = x_std_all

        q = int(self.coef_decimals)
        coef_all[np.abs(coef_all) < float(self.coef_prune_abs)] = 0.0
        self.coef_ = np.round(coef_all, q)
        self.intercept_ = float(np.round(intercept_all, q))

        self.use_step_ = bool(step_info["used"])
        if self.use_step_:
            j = int(step_info["feature"])
            t = float(step_info["threshold"])
            residual_all = y - (self.intercept_ + X @ self.coef_)
            left = X[:, j] <= t
            right = ~left
            if left.sum() >= int(self.min_leaf_samples) and right.sum() >= int(self.min_leaf_samples):
                left_offset = float(self.step_shrink) * float(np.mean(residual_all[left]))
                right_offset = float(self.step_shrink) * float(np.mean(residual_all[right]))
            else:
                left_offset, right_offset = 0.0, 0.0
            self.step_feature_ = j
            self.step_threshold_ = float(np.round(t, q))
            self.step_left_offset_ = float(np.round(left_offset, q))
            self.step_right_offset_ = float(np.round(right_offset, q))
        else:
            self.step_feature_ = -1
            self.step_threshold_ = 0.0
            self.step_left_offset_ = 0.0
            self.step_right_offset_ = 0.0

        self.feature_importance_ = np.abs(self.coef_)
        self.feature_rank_ = np.argsort(self.feature_importance_)[::-1]
        self.fitted_mse_ = float(np.mean((y - self.predict(X)) ** 2))
        return self

    def predict(self, X):
        check_is_fitted(self, ["coef_", "intercept_", "use_step_"])
        X = self._impute(X)
        yhat = self.intercept_ + X @ self.coef_
        if self.use_step_:
            yhat = yhat + self._apply_step(
                X[:, int(self.step_feature_)],
                float(self.step_threshold_),
                float(self.step_left_offset_),
                float(self.step_right_offset_),
            )
        return yhat

    def __str__(self):
        check_is_fitted(self, ["coef_", "intercept_", "use_step_"])
        lines = [
            "RidgePlusStepRuleRegressor",
            f"Selected ridge alpha: {self.alpha_:.5g}",
            "",
            "Prediction equation (raw features):",
            f"  y = {self.intercept_:+.5f}",
        ]
        for j, c in enumerate(self.coef_):
            if abs(float(c)) > 0.0:
                lines.append(f"    + ({float(c):+.5f}) * x{j}")

        if self.use_step_:
            lines.extend([
                "",
                "Residual step rule:",
                f"  if x{self.step_feature_} <= {self.step_threshold_:.5f}: add {self.step_left_offset_:+.5f}",
                f"  else: add {self.step_right_offset_:+.5f}",
            ])
        else:
            lines.extend(["", "Residual step rule: not used."])

        lines.append("")
        lines.append("Feature importance (absolute linear coefficient):")
        for j in self.feature_rank_[: min(10, self.n_features_in_)]:
            lines.append(f"  x{int(j)}: {self.feature_importance_[int(j)]:.5f}")
        near_zero = [f"x{j}" for j, v in enumerate(self.feature_importance_) if v < 1e-8]
        if near_zero:
            lines.append("Features with near-zero effect: " + ", ".join(near_zero))
        lines.append(f"Training MSE: {self.fitted_mse_:.6f}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
RidgePlusStepRuleRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "RidgeStepResidualV1"
model_description = "Dense ridge equation in raw features plus one optional single-feature residual step rule chosen by validation gain"
model_defs = [(model_shorthand_name, RidgePlusStepRuleRegressor())]

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
