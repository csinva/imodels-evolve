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
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class SingleSplitSparseLinearRegressor(BaseEstimator, RegressorMixin):
    """
    One-rule sparse model:
    - Fit a sparse global linear model.
    - Optionally split once on a feature threshold.
    - Fit separate sparse linear equations in the two leaves.
    """

    def __init__(
        self,
        max_terms=6,
        max_terms_leaf=5,
        top_split_features=6,
        split_quantiles=(0.25, 0.5, 0.75),
        min_leaf_frac=0.15,
        min_gain=0.01,
        complexity_penalty=0.002,
        random_state=42,
    ):
        self.max_terms = max_terms
        self.max_terms_leaf = max_terms_leaf
        self.top_split_features = top_split_features
        self.split_quantiles = split_quantiles
        self.min_leaf_frac = min_leaf_frac
        self.min_gain = min_gain
        self.complexity_penalty = complexity_penalty
        self.random_state = random_state

    def _impute(self, X):
        X = np.asarray(X, dtype=float)
        X = X.copy()
        nan_mask = ~np.isfinite(X)
        if nan_mask.any():
            X[nan_mask] = np.take(self.feature_medians_, np.where(nan_mask)[1])
        return X

    def _corr_abs(self, a, b):
        ac = a - a.mean()
        bc = b - b.mean()
        denom = (ac.std() + 1e-12) * (bc.std() + 1e-12)
        return abs(np.mean(ac * bc) / denom)

    def _nonlinear_score(self, xj, y):
        return self._corr_abs(xj, y) + 0.7 * self._corr_abs(xj * xj, y)

    def _fit_sparse_linear(self, X, y, max_terms):
        x_mean = X.mean(axis=0)
        x_std = X.std(axis=0)
        x_std[x_std < 1e-12] = 1.0
        Xs = (X - x_mean) / x_std

        selector = LassoCV(
            cv=3,
            random_state=self.random_state,
            n_alphas=30,
            max_iter=4000,
        )
        selector.fit(Xs, y)

        dense_coef = selector.coef_ / x_std
        dense_intercept = selector.intercept_ - float(np.dot(dense_coef, x_mean))

        active = np.where(np.abs(dense_coef) > 1e-7)[0]
        if active.size == 0:
            active = np.array([int(np.argmax(np.abs(dense_coef)))], dtype=int)
        if active.size > max_terms:
            order = np.argsort(np.abs(dense_coef[active]))[::-1]
            active = active[order[:max_terms]]

        reg = LinearRegression()
        reg.fit(X[:, active], y)
        coef = np.zeros(X.shape[1], dtype=float)
        coef[active] = reg.coef_.astype(float)
        intercept = float(reg.intercept_)

        preds = intercept + X @ coef
        mse = float(np.mean((y - preds) ** 2))
        # fallback if refit becomes unstable
        if not np.isfinite(mse) or mse > float(np.mean((y - (dense_intercept + X @ dense_coef)) ** 2)) * 1.15:
            coef = dense_coef
            intercept = dense_intercept
            preds = intercept + X @ coef
            mse = float(np.mean((y - preds) ** 2))
            active = np.where(np.abs(coef) > 1e-7)[0]
        return {
            "intercept": intercept,
            "coef": coef,
            "active": active,
            "mse": mse,
        }

    def _equation_string(self, intercept, coef):
        active = np.where(np.abs(coef) > 1e-7)[0]
        if active.size == 0:
            return f"y = {intercept:.4f}"
        order = active[np.argsort(np.abs(coef[active]))[::-1]]
        terms = [f"{coef[j]:+.4f}*x{j}" for j in order]
        return f"y = {intercept:.4f} " + " ".join(terms)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.feature_medians_ = np.nanmedian(X, axis=0)
        X = self._impute(X)

        global_model = self._fit_sparse_linear(X, y, self.max_terms)
        best = {
            "use_split": False,
            "score": global_model["mse"] + self.complexity_penalty * len(global_model["active"]),
            "global": global_model,
        }

        n = X.shape[0]
        min_leaf = max(10, int(self.min_leaf_frac * n))
        nonlinear_scores = np.array([self._nonlinear_score(X[:, j], y) for j in range(self.n_features_in_)])
        split_order = np.argsort(nonlinear_scores)[::-1][: min(self.top_split_features, self.n_features_in_)]

        for feat in split_order:
            thresholds = np.unique(np.quantile(X[:, feat], self.split_quantiles))
            for thr in thresholds:
                left_mask = X[:, feat] <= thr
                right_mask = ~left_mask
                n_left = int(left_mask.sum())
                n_right = int(right_mask.sum())
                if n_left < min_leaf or n_right < min_leaf:
                    continue
                left_model = self._fit_sparse_linear(X[left_mask], y[left_mask], self.max_terms_leaf)
                right_model = self._fit_sparse_linear(X[right_mask], y[right_mask], self.max_terms_leaf)

                preds = np.empty_like(y, dtype=float)
                preds[left_mask] = left_model["intercept"] + X[left_mask] @ left_model["coef"]
                preds[right_mask] = right_model["intercept"] + X[right_mask] @ right_model["coef"]
                mse = float(np.mean((y - preds) ** 2))
                complexity = (
                    len(left_model["active"]) + len(right_model["active"]) + 1
                )
                score = mse + self.complexity_penalty * complexity
                if score < best["score"] * (1.0 - self.min_gain):
                    best = {
                        "use_split": True,
                        "score": score,
                        "split_feature": int(feat),
                        "split_threshold": float(thr),
                        "left": left_model,
                        "right": right_model,
                        "n_left": n_left,
                        "n_right": n_right,
                    }

        self.use_split_ = bool(best["use_split"])
        if self.use_split_:
            self.split_feature_ = best["split_feature"]
            self.split_threshold_ = best["split_threshold"]
            self.left_model_ = best["left"]
            self.right_model_ = best["right"]
            self.leaf_fraction_ = (best["n_left"] / n, best["n_right"] / n)
            weighted_imp = (
                self.leaf_fraction_[0] * np.abs(self.left_model_["coef"])
                + self.leaf_fraction_[1] * np.abs(self.right_model_["coef"])
            )
            self.feature_importance_ = weighted_imp
        else:
            self.global_model_ = best["global"]
            self.feature_importance_ = np.abs(self.global_model_["coef"])
        return self

    def predict(self, X):
        check_is_fitted(self, ["feature_importance_"])
        X = self._impute(X)
        if getattr(self, "use_split_", False):
            left_mask = X[:, self.split_feature_] <= self.split_threshold_
            preds = np.empty(X.shape[0], dtype=float)
            preds[left_mask] = self.left_model_["intercept"] + X[left_mask] @ self.left_model_["coef"]
            preds[~left_mask] = self.right_model_["intercept"] + X[~left_mask] @ self.right_model_["coef"]
            return preds
        return self.global_model_["intercept"] + X @ self.global_model_["coef"]

    def __str__(self):
        check_is_fitted(self, ["feature_importance_"])
        lines = ["SingleSplitSparseLinearRegressor:"]
        if getattr(self, "use_split_", False):
            lines.append(
                f"Rule: if x{self.split_feature_} <= {self.split_threshold_:.4f}, use left equation; else use right equation."
            )
            lines.append("Left leaf equation:")
            lines.append("  " + self._equation_string(self.left_model_["intercept"], self.left_model_["coef"]))
            lines.append("Right leaf equation:")
            lines.append("  " + self._equation_string(self.right_model_["intercept"], self.right_model_["coef"]))
            lines.append(
                f"Leaf coverage: left={self.leaf_fraction_[0]:.2%}, right={self.leaf_fraction_[1]:.2%}"
            )
        else:
            lines.append("No split selected; using one sparse linear equation.")
            lines.append("  " + self._equation_string(self.global_model_["intercept"], self.global_model_["coef"]))

        lines.append("")
        lines.append("Feature importance (weighted abs coefficients):")
        rank = np.argsort(self.feature_importance_)[::-1]
        for j in rank:
            lines.append(f"  x{j}: {self.feature_importance_[j]:.4f}")
        near_zero = [f"x{j}" for j, v in enumerate(self.feature_importance_) if v < 1e-4]
        if near_zero:
            lines.append("Features with near-zero effect: " + ", ".join(near_zero))
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SingleSplitSparseLinearRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "OneSplitSparseLinV1"
model_description = "Sparse linear model with optional single threshold split into two sparse leaf equations"
model_defs = [(model_shorthand_name, SingleSplitSparseLinearRegressor())]


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
