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


class GCVRidgeEquationRegressor(BaseEstimator, RegressorMixin):
    """
    Dense linear regressor fit with closed-form ridge and GCV alpha selection.

    The final prediction equation is expressed directly in raw input features:
      y = intercept + sum_j coef_j * x_j

    This keeps the model easy to simulate while improving numerical stability
    over plain OLS on collinear datasets.
    """

    def __init__(
        self,
        alpha_grid=(1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0),
        coef_decimals=5,
        tiny_coef_threshold=0.0,
    ):
        self.alpha_grid = alpha_grid
        self.coef_decimals = coef_decimals
        self.tiny_coef_threshold = tiny_coef_threshold

    def _impute(self, X):
        X = np.asarray(X, dtype=float).copy()
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(self.feature_medians_, np.where(bad)[1])
        return X

    @staticmethod
    def _ridge_from_svd(U, s, Vt, yc, alpha):
        # beta = V * diag(s / (s^2 + alpha)) * U^T y
        ut_y = U.T @ yc
        shrink = s / (s * s + float(alpha))
        return Vt.T @ (shrink * ut_y)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        self.n_features_in_ = p

        self.feature_medians_ = np.nanmedian(X, axis=0)
        X = self._impute(X)

        self.x_mean_ = np.mean(X, axis=0)
        self.x_scale_ = np.std(X, axis=0)
        self.x_scale_[self.x_scale_ < 1e-12] = 1.0

        Xs = (X - self.x_mean_) / self.x_scale_
        y_mean = float(np.mean(y))
        yc = y - y_mean

        U, s, Vt = np.linalg.svd(Xs, full_matrices=False)
        n_float = float(n)

        best = None
        for alpha in self.alpha_grid:
            beta_std = self._ridge_from_svd(U, s, Vt, yc, alpha)
            resid = yc - (Xs @ beta_std)
            rss = float(np.dot(resid, resid))
            df = float(np.sum((s * s) / (s * s + float(alpha))))
            denom = max((1.0 - df / n_float) ** 2, 1e-12)
            gcv = (rss / n_float) / denom
            if best is None or gcv < best["gcv"]:
                best = {"alpha": float(alpha), "beta_std": beta_std, "gcv": gcv}

        beta_std = best["beta_std"]
        coef_raw = beta_std / self.x_scale_
        intercept_raw = y_mean - float(np.dot(coef_raw, self.x_mean_))

        coef_raw = np.asarray(coef_raw, dtype=float)
        coef_raw[np.abs(coef_raw) < float(self.tiny_coef_threshold)] = 0.0

        q = int(self.coef_decimals)
        self.alpha_ = float(best["alpha"])
        self.intercept_ = float(np.round(intercept_raw, q))
        self.coef_ = np.round(coef_raw, q)

        self.feature_importance_ = np.abs(self.coef_)
        self.feature_rank_ = np.argsort(self.feature_importance_)[::-1]
        pred = self.predict(X)
        self.fitted_mse_ = float(np.mean((y - pred) ** 2))
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "coef_"])
        X = self._impute(X)
        return self.intercept_ + X @ self.coef_

    def __str__(self):
        check_is_fitted(self, ["intercept_", "coef_", "feature_rank_"])
        active = [(j, float(c)) for j, c in enumerate(self.coef_) if abs(float(c)) > 0.0]

        eq_terms = [f"{self.intercept_:+0.5f}"]
        for j, c in active:
            eq_terms.append(f"{c:+0.5f}*x{j}")

        lines = [
            "GCVRidgeEquationRegressor",
            f"Selected alpha (GCV): {self.alpha_:.6g}",
            "",
            "Prediction equation:",
            "  y = " + " ".join(eq_terms),
            "",
            f"Active features ({len(active)}/{self.n_features_in_}):",
            "  " + (", ".join(f"x{j}" for j, _ in active) if active else "none"),
            "",
            "Coefficients by absolute effect (top 12):",
        ]
        for j in self.feature_rank_[: min(12, self.n_features_in_)]:
            lines.append(f"  x{int(j)}: coef={self.coef_[int(j)]:+0.5f}  |coef|={self.feature_importance_[int(j)]:0.5f}")

        zeroed = [f"x{j}" for j, c in enumerate(self.coef_) if abs(float(c)) == 0.0]
        if zeroed:
            lines.append("Zeroed features: " + ", ".join(zeroed))
        lines.append(f"Training MSE: {self.fitted_mse_:.6f}")
        return "\n".join(lines)


class SparseAdditiveStumpBoostRegressor(BaseEstimator, RegressorMixin):
    """
    Compact additive regressor built from a short list of one-feature threshold rules.

    Model form:
      y = bias + sum_t [left_t if x[j_t] <= thr_t else right_t]

    Rules are learned greedily on residuals with small shrinkage.
    This adds limited nonlinearity while staying easy to simulate from text.
    """

    def __init__(
        self,
        max_terms=3,
        n_thresholds=9,
        learning_rate=0.8,
        min_improvement=1e-5,
        coef_decimals=5,
    ):
        self.max_terms = max_terms
        self.n_thresholds = n_thresholds
        self.learning_rate = learning_rate
        self.min_improvement = min_improvement
        self.coef_decimals = coef_decimals

    def _impute(self, X):
        X = np.asarray(X, dtype=float).copy()
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(self.feature_medians_, np.where(bad)[1])
        return X

    @staticmethod
    def _candidate_thresholds(col, n_thresholds):
        qs = np.linspace(0.1, 0.9, int(n_thresholds))
        thr = np.quantile(col, qs)
        thr = np.asarray(np.unique(thr), dtype=float)
        return thr

    @staticmethod
    def _best_stump_for_feature(col, resid, thresholds):
        n = col.shape[0]
        if n == 0 or thresholds.size == 0:
            return None

        order = np.argsort(col)
        xs = col[order]
        rs = resid[order]
        csum = np.cumsum(rs)
        total = csum[-1]

        best = None
        for thr in thresholds:
            left_n = int(np.searchsorted(xs, thr, side="right"))
            right_n = n - left_n
            if left_n < 5 or right_n < 5:
                continue

            left_sum = float(csum[left_n - 1])
            right_sum = float(total - left_sum)
            left_val = left_sum / left_n
            right_val = right_sum / right_n

            pred = np.where(col <= thr, left_val, right_val)
            mse = float(np.mean((resid - pred) ** 2))
            if best is None or mse < best["mse"]:
                best = {
                    "threshold": float(thr),
                    "left_value": float(left_val),
                    "right_value": float(right_val),
                    "mse": mse,
                }
        return best

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        self.n_features_in_ = p

        self.feature_medians_ = np.nanmedian(X, axis=0)
        X = self._impute(X)

        self.bias_ = float(np.mean(y))
        pred = np.full(n, self.bias_, dtype=float)
        self.rules_ = []

        prev_mse = float(np.mean((y - pred) ** 2))
        for _ in range(int(self.max_terms)):
            resid = y - pred
            best_rule = None

            for j in range(p):
                col = X[:, j]
                thresholds = self._candidate_thresholds(col, self.n_thresholds)
                stump = self._best_stump_for_feature(col, resid, thresholds)
                if stump is None:
                    continue
                rule_pred = np.where(col <= stump["threshold"], stump["left_value"], stump["right_value"])
                mse = float(np.mean((resid - rule_pred) ** 2))
                if best_rule is None or mse < best_rule["mse"]:
                    best_rule = {
                        "feature": int(j),
                        "threshold": float(stump["threshold"]),
                        "left_value": float(stump["left_value"]),
                        "right_value": float(stump["right_value"]),
                        "mse": mse,
                    }

            if best_rule is None:
                break

            j = best_rule["feature"]
            thr = best_rule["threshold"]
            lv = self.learning_rate * best_rule["left_value"]
            rv = self.learning_rate * best_rule["right_value"]
            contrib = np.where(X[:, j] <= thr, lv, rv)
            new_pred = pred + contrib
            new_mse = float(np.mean((y - new_pred) ** 2))
            if prev_mse - new_mse < float(self.min_improvement):
                break

            self.rules_.append({
                "feature": j,
                "threshold": float(thr),
                "left_value": float(lv),
                "right_value": float(rv),
            })
            pred = new_pred
            prev_mse = new_mse

        q = int(self.coef_decimals)
        self.bias_ = float(np.round(self.bias_, q))
        rounded_rules = []
        for r in self.rules_:
            rounded_rules.append({
                "feature": r["feature"],
                "threshold": float(np.round(r["threshold"], q)),
                "left_value": float(np.round(r["left_value"], q)),
                "right_value": float(np.round(r["right_value"], q)),
            })
        self.rules_ = rounded_rules

        usage = np.zeros(self.n_features_in_, dtype=float)
        for r in self.rules_:
            usage[r["feature"]] += abs(r["left_value"]) + abs(r["right_value"])
        self.feature_importance_ = usage
        self.feature_rank_ = np.argsort(usage)[::-1]
        self.fitted_mse_ = float(np.mean((y - self.predict(X)) ** 2))
        return self

    def predict(self, X):
        check_is_fitted(self, ["bias_", "rules_"])
        X = self._impute(X)
        out = np.full(X.shape[0], self.bias_, dtype=float)
        for r in self.rules_:
            j = r["feature"]
            thr = r["threshold"]
            lv = r["left_value"]
            rv = r["right_value"]
            out += np.where(X[:, j] <= thr, lv, rv)
        return out

    def __str__(self):
        check_is_fitted(self, ["bias_", "rules_", "feature_rank_"])
        lines = [
            "SparseAdditiveStumpBoostRegressor",
            f"Bias: {self.bias_:+0.5f}",
            "",
            "Prediction program:",
            "  y = bias",
        ]
        for k, r in enumerate(self.rules_, 1):
            j = int(r["feature"])
            thr = r["threshold"]
            lv = r["left_value"]
            rv = r["right_value"]
            lines.append(
                f"  rule{k}: if x{j} <= {thr:+0.5f} then add {lv:+0.5f} else add {rv:+0.5f}"
            )

        lines.extend(
            [
                "",
                f"Rules used: {len(self.rules_)} (max {self.max_terms})",
                "Feature usage by absolute contribution (top 12):",
            ]
        )
        for j in self.feature_rank_[: min(12, self.n_features_in_)]:
            lines.append(f"  x{int(j)}: score={self.feature_importance_[int(j)]:0.5f}")
        lines.append(f"Training MSE: {self.fitted_mse_:.6f}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
GCVRidgeEquationRegressor.__module__ = "interpretable_regressor"
SparseAdditiveStumpBoostRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseAdditiveStumpBoostV1"
model_description = "Greedy residual additive model with a short list of one-feature threshold rules and explicit if-then prediction program"
model_defs = [(model_shorthand_name, SparseAdditiveStumpBoostRegressor())]

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
