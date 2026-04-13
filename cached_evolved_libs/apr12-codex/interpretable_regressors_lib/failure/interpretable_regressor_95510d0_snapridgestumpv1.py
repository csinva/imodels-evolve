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


class SnapRidgeStumpRegressor(BaseEstimator, RegressorMixin):
    """
    Linear ridge backbone with snapped coefficients plus one residual stump:
      y = b + sum_j w_j * x_j + (x_s <= t ? c_left : c_right)
    """

    def __init__(
        self,
        ridge_alpha=0.8,
        coef_snap=0.1,
        coef_tol=2e-3,
        top_stump_features=6,
        stump_shrink=0.85,
        min_stump_gain=2e-4,
    ):
        self.ridge_alpha = ridge_alpha
        self.coef_snap = coef_snap
        self.coef_tol = coef_tol
        self.top_stump_features = top_stump_features
        self.stump_shrink = stump_shrink
        self.min_stump_gain = min_stump_gain

    def _impute(self, X):
        X = np.asarray(X, dtype=float).copy()
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(self.feature_medians_, np.where(bad)[1])
        return X

    @staticmethod
    def _snap(v, step):
        return float(np.round(v / step) * step)

    @staticmethod
    def _ridge_closed_form(Xs, y, alpha):
        y_mean = float(np.mean(y))
        yc = y - y_mean
        gram = Xs.T @ Xs
        rhs = Xs.T @ yc
        coef = np.linalg.solve(gram + float(alpha) * np.eye(Xs.shape[1]), rhs)
        return y_mean, coef

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.feature_medians_ = np.nanmedian(X, axis=0)
        X = self._impute(X)

        self.feature_means_ = np.mean(X, axis=0)
        self.feature_scales_ = np.std(X, axis=0)
        self.feature_scales_[self.feature_scales_ < 1e-8] = 1.0
        Xs = (X - self.feature_means_) / self.feature_scales_

        y_mean, coef_s = self._ridge_closed_form(Xs, y, self.ridge_alpha)
        coef_raw = coef_s / self.feature_scales_
        intercept_raw = y_mean - float(np.dot(self.feature_means_, coef_raw))

        coef_q = np.array([self._snap(v, self.coef_snap) for v in coef_raw], dtype=float)
        coef_q[np.abs(coef_q) < self.coef_tol] = 0.0
        intercept_q = float(np.mean(y - X @ coef_q))

        pred_lin = intercept_q + X @ coef_q
        mse_lin = float(np.mean((y - pred_lin) ** 2))

        order = np.argsort(np.abs(coef_raw))[::-1]
        cand_feats = order[: min(self.top_stump_features, self.n_features_in_)]
        best = None
        residual = y - pred_lin
        for j in cand_feats:
            xj = X[:, int(j)]
            for q in (0.2, 0.35, 0.5, 0.65, 0.8):
                t = float(np.quantile(xj, q))
                left = xj <= t
                right = ~left
                if left.sum() < 20 or right.sum() < 20:
                    continue
                c_left = float(np.mean(residual[left])) * self.stump_shrink
                c_right = float(np.mean(residual[right])) * self.stump_shrink
                pred_try = pred_lin + np.where(left, c_left, c_right)
                mse_try = float(np.mean((y - pred_try) ** 2))
                if best is None or mse_try < best["mse"]:
                    best = {
                        "mse": mse_try,
                        "j": int(j),
                        "t": t,
                        "c_left": self._snap(c_left, self.coef_snap),
                        "c_right": self._snap(c_right, self.coef_snap),
                    }

        self.intercept_ = float(intercept_q)
        self.linear_coef_ = coef_q.copy()
        self.linear_intercept_raw_ = float(intercept_raw)

        use_stump = best is not None and (mse_lin - best["mse"]) > self.min_stump_gain
        if use_stump:
            self.stump_feature_ = int(best["j"])
            self.stump_threshold_ = float(best["t"])
            self.stump_left_value_ = float(best["c_left"])
            self.stump_right_value_ = float(best["c_right"])
        else:
            self.stump_feature_ = -1
            self.stump_threshold_ = 0.0
            self.stump_left_value_ = 0.0
            self.stump_right_value_ = 0.0

        fi = np.abs(self.linear_coef_)
        if self.stump_feature_ >= 0:
            fi[self.stump_feature_] += 0.5 * (
                abs(self.stump_left_value_) + abs(self.stump_right_value_)
            )
        self.feature_importance_ = fi
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "linear_coef_", "stump_feature_", "feature_importance_"])
        X = self._impute(X)
        yhat = self.intercept_ + X @ self.linear_coef_
        if self.stump_feature_ >= 0:
            yhat = yhat + np.where(
                X[:, self.stump_feature_] <= self.stump_threshold_,
                self.stump_left_value_,
                self.stump_right_value_,
            )
        return yhat

    def __str__(self):
        check_is_fitted(self, ["intercept_", "linear_coef_", "feature_importance_"])
        lines = [
            "SnapRidgeStumpRegressor",
            "Computation recipe:",
            "1) linear_part = intercept + sum_j (coef_j * xj)",
            "2) if stump rule exists, add branch adjustment",
            "",
            f"intercept = {self.intercept_:+.4f}",
            "linear coefficients:",
        ]
        active = []
        for j, c in enumerate(self.linear_coef_):
            if abs(float(c)) >= self.coef_tol:
                active.append((abs(float(c)), j, float(c)))
        active.sort(key=lambda t: -t[0])
        for _, j, c in active:
            lines.append(f"  x{j}: {c:+.4f}")
        near_zero = [f"x{j}" for j in range(self.n_features_in_) if abs(self.linear_coef_[j]) < self.coef_tol]
        if near_zero:
            lines.append("near-zero linear effect: " + ", ".join(near_zero))

        if self.stump_feature_ >= 0:
            lines.extend([
                "",
                "stump adjustment:",
                f"  if x{self.stump_feature_} <= {self.stump_threshold_:.4f}: add {self.stump_left_value_:+.4f}",
                f"  else: add {self.stump_right_value_:+.4f}",
            ])
        else:
            lines.extend(["", "stump adjustment: none"])

        lines.append("")
        lines.append("final equation:")
        eq_terms = [f"{self.intercept_:+.4f}"] + [
            f"({c:+.4f} * x{j})" for _, j, c in active
        ]
        lines.append("  y = " + " + ".join(eq_terms))
        if self.stump_feature_ >= 0:
            lines.append(
                f"      + (x{self.stump_feature_} <= {self.stump_threshold_:.4f} ? "
                f"{self.stump_left_value_:+.4f} : {self.stump_right_value_:+.4f})"
            )
        return "\n".join(lines)



# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SnapRidgeStumpRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SnapRidgeStumpV1"
model_description = "Standardized ridge backbone with quantized arithmetic coefficients plus one residual decision-stump offset for compact nonlinearity"
model_defs = [(model_shorthand_name, SnapRidgeStumpRegressor())]


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
