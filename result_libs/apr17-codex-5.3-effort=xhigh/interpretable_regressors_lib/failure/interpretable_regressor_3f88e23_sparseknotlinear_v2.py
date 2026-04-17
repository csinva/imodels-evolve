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


class SparseKnotLinearRegressor(BaseEstimator, RegressorMixin):
    """Sparse global linear model plus one learned hinge knot."""

    def __init__(
        self,
        max_linear_features=6,
        ridge_lambda=1e-3,
        max_knot_features=8,
        min_rel_gain=0.01,
        negligible_feature_eps=5e-3,
    ):
        self.max_linear_features = max_linear_features
        self.ridge_lambda = ridge_lambda
        self.max_knot_features = max_knot_features
        self.min_rel_gain = min_rel_gain
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

    def _fit_sparse_ridge(self, X, y, feature_idx):
        if len(feature_idx) == 0:
            return float(np.mean(y)), np.zeros(0, dtype=float)

        Xs = X[:, feature_idx]
        x_mean = np.mean(Xs, axis=0)
        x_std = np.std(Xs, axis=0)
        x_std[x_std < 1e-12] = 1.0
        Z = (Xs - x_mean) / x_std

        y_mean = float(np.mean(y))
        y_centered = y - y_mean
        p = Z.shape[1]

        gram = Z.T @ Z + self.ridge_lambda * np.eye(p)
        rhs = Z.T @ y_centered
        coef_std = np.linalg.solve(gram, rhs)
        coef_raw = coef_std / x_std
        intercept = float(y_mean - np.dot(coef_raw, x_mean))
        return intercept, coef_raw

    def _predict_linear(self, X):
        if len(self.linear_feature_idx_) == 0:
            return np.full(X.shape[0], self.intercept_, dtype=float)
        return self.intercept_ + X[:, self.linear_feature_idx_] @ self.linear_coef_

    def _hinge(self, x_col):
        if self.knot_direction_ == "above":
            return np.maximum(0.0, x_col - self.knot_threshold_)
        return np.maximum(0.0, self.knot_threshold_ - x_col)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        corr = np.array([self._safe_abs_corr(X[:, j], y) for j in range(n_features)])
        ranked = np.argsort(corr)[::-1]
        k = min(self.max_linear_features, n_features)
        self.linear_feature_idx_ = np.sort(ranked[:k].astype(int))

        self.intercept_, self.linear_coef_ = self._fit_sparse_ridge(X, y, self.linear_feature_idx_)
        base_pred = self._predict_linear(X)
        base_resid = y - base_pred
        base_sse = float(np.dot(base_resid, base_resid))

        self.use_knot_ = False
        self.knot_feature_ = None
        self.knot_threshold_ = None
        self.knot_direction_ = None
        self.knot_coef_ = 0.0

        best = None
        knot_candidates = ranked[: min(self.max_knot_features, n_features)]
        min_gain = self.min_rel_gain * (base_sse + 1e-12)

        for j in knot_candidates:
            xj = X[:, int(j)]
            thresholds = np.unique(np.round(np.quantile(xj, [0.15, 0.3, 0.5, 0.7, 0.85]), 8))
            for threshold in thresholds:
                for direction in ("above", "below"):
                    if direction == "above":
                        h = np.maximum(0.0, xj - threshold)
                    else:
                        h = np.maximum(0.0, threshold - xj)

                    if float(np.std(h)) < 1e-8:
                        continue

                    h_center = h - np.mean(h)
                    denom = float(np.dot(h_center, h_center) + self.ridge_lambda * n_samples)
                    beta = float(np.dot(base_resid, h_center) / denom)
                    pred = base_pred + beta * h
                    resid = y - pred
                    sse = float(np.dot(resid, resid))

                    if best is None or sse < best["sse"]:
                        best = {
                            "feature": int(j),
                            "threshold": float(threshold),
                            "direction": direction,
                            "coef": beta,
                            "sse": sse,
                        }

        if best is not None and (base_sse - best["sse"]) >= min_gain:
            self.use_knot_ = True
            self.knot_feature_ = best["feature"]
            self.knot_threshold_ = best["threshold"]
            self.knot_direction_ = best["direction"]
            self.knot_coef_ = float(best["coef"])

        contrib = np.zeros(n_features, dtype=float)
        if len(self.linear_feature_idx_):
            c_linear = np.mean(
                np.abs(X[:, self.linear_feature_idx_] * self.linear_coef_.reshape(1, -1)),
                axis=0,
            )
            contrib[self.linear_feature_idx_] = c_linear

        if self.use_knot_:
            h = self._hinge(X[:, self.knot_feature_])
            contrib[self.knot_feature_] += float(np.mean(np.abs(self.knot_coef_ * h)))

        self.feature_importances_ = contrib
        return self

    def predict(self, X):
        check_is_fitted(self, ["linear_feature_idx_", "intercept_", "linear_coef_", "use_knot_"])
        X = np.asarray(X, dtype=float)
        pred = self._predict_linear(X)
        if self.use_knot_:
            pred = pred + self.knot_coef_ * self._hinge(X[:, self.knot_feature_])
        return pred

    def __str__(self):
        check_is_fitted(self, ["linear_feature_idx_", "feature_importances_"])
        active_linear = [f"x{j}" for j in self.linear_feature_idx_]

        equation_terms = [f"{self.intercept_:+.5f}"]
        for j, coef in zip(self.linear_feature_idx_, self.linear_coef_):
            equation_terms.append(f"{coef:+.5f}*x{j}")
        if self.use_knot_:
            if self.knot_direction_ == "above":
                hinge_txt = f"max(0, x{self.knot_feature_} - {self.knot_threshold_:.5f})"
            else:
                hinge_txt = f"max(0, {self.knot_threshold_:.5f} - x{self.knot_feature_})"
            equation_terms.append(f"{self.knot_coef_:+.5f}*{hinge_txt}")

        order = np.argsort(self.feature_importances_)[::-1]
        top = [f"x{j}:{self.feature_importances_[j]:.4f}" for j in order[: min(10, self.n_features_in_)]]
        negligible = [
            f"x{j}"
            for j in range(self.n_features_in_)
            if self.feature_importances_[j] <= self.negligible_feature_eps
        ]

        lines = [
            "Sparse Knot Linear Regressor (global equation + one hinge)",
            f"Active linear features ({len(active_linear)}): " + (", ".join(active_linear) if active_linear else "none"),
            "Model equation:",
            "  y = " + " ".join(equation_terms),
        ]

        if self.use_knot_:
            lines.append("Knot term:")
            lines.append(f"  H = {hinge_txt}")
            lines.append(f"  Add {self.knot_coef_:+.5f} * H")
        else:
            lines.append("Knot term: none selected")

        lines.extend([
            "",
            "Most influential features (mean absolute contribution):",
            "  " + ", ".join(top),
            "Features with negligible effect:",
            "  " + (", ".join(negligible) if negligible else "none"),
            "",
            "Prediction recipe: compute the linear equation, then add the single knot term if present.",
        ])
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseKnotLinearRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseKnotLinear_v2"
model_description = "Sparse ridge linear backbone with a single data-driven hinge knot term for compact threshold-aware equations"
model_defs = [(model_shorthand_name, SparseKnotLinearRegressor())]


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
