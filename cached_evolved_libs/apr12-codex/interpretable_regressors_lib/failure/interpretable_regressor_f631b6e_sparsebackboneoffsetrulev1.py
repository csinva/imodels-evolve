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


class SparseBackboneOffsetRuleRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse linear backbone with one optional threshold residual offset rule.

    Prediction form:
      y = b + sum_j w_j*x_j + (offset_left if x_k <= t else offset_right)
    """

    def __init__(
        self,
        max_terms=6,
        top_rule_features=6,
        split_quantiles=(0.2, 0.35, 0.5, 0.65, 0.8),
        min_leaf_frac=0.12,
        min_relative_gain=0.01,
        random_state=42,
    ):
        self.max_terms = max_terms
        self.top_rule_features = top_rule_features
        self.split_quantiles = split_quantiles
        self.min_leaf_frac = min_leaf_frac
        self.min_relative_gain = min_relative_gain
        self.random_state = random_state

    def _impute(self, X):
        X = np.asarray(X, dtype=float).copy()
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(self.feature_medians_, np.where(bad)[1])
        return X

    @staticmethod
    def _corr_abs(a, b):
        ac = a - np.mean(a)
        bc = b - np.mean(b)
        denom = np.sqrt(np.dot(ac, ac) * np.dot(bc, bc)) + 1e-12
        return abs(float(np.dot(ac, bc) / denom))

    def _fit_sparse_backbone(self, X, y):
        from sklearn.linear_model import LassoCV, LinearRegression

        xm = np.mean(X, axis=0)
        xs = np.std(X, axis=0)
        xs[xs < 1e-12] = 1.0
        Xs = (X - xm) / xs

        l1 = LassoCV(cv=3, random_state=self.random_state, n_alphas=40, max_iter=4000)
        l1.fit(Xs, y)

        raw_coef = l1.coef_ / xs
        raw_intercept = float(l1.intercept_ - np.dot(raw_coef, xm))
        active = np.where(np.abs(raw_coef) > 1e-7)[0]
        if active.size == 0:
            active = np.array([int(np.argmax(np.abs(raw_coef)))], dtype=int)
        if active.size > int(self.max_terms):
            order = np.argsort(np.abs(raw_coef[active]))[::-1]
            active = active[order[: int(self.max_terms)]]

        ols = LinearRegression()
        ols.fit(X[:, active], y)
        coef = np.zeros(X.shape[1], dtype=float)
        coef[active] = ols.coef_.astype(float)
        intercept = float(ols.intercept_)
        pred = intercept + X @ coef
        mse = float(np.mean((y - pred) ** 2))

        mse_raw = float(np.mean((y - (raw_intercept + X @ raw_coef)) ** 2))
        if (not np.isfinite(mse)) or mse > 1.1 * mse_raw:
            coef = raw_coef
            intercept = raw_intercept
            pred = intercept + X @ coef
            mse = mse_raw
            active = np.where(np.abs(coef) > 1e-7)[0]
        return intercept, coef, active, pred, mse

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        self.n_features_in_ = p

        self.feature_medians_ = np.nanmedian(X, axis=0)
        self.feature_medians_ = np.where(np.isfinite(self.feature_medians_), self.feature_medians_, 0.0)
        X = self._impute(X)

        intercept, coef, active, base_pred, base_mse = self._fit_sparse_backbone(X, y)
        resid = y - base_pred

        min_leaf = max(12, int(float(self.min_leaf_frac) * n))
        if active.size > 0:
            cand_features = active[np.argsort(np.abs(coef[active]))[::-1][: int(self.top_rule_features)]]
        else:
            scores = np.array([self._corr_abs(X[:, j], resid) for j in range(p)])
            cand_features = np.argsort(scores)[::-1][: int(self.top_rule_features)]

        best = {
            "use_rule": False,
            "mse": base_mse,
            "feature": None,
            "threshold": None,
            "offset_left": 0.0,
            "offset_right": 0.0,
        }
        for feat in cand_features:
            thresholds = np.unique(np.quantile(X[:, int(feat)], self.split_quantiles))
            for thr in thresholds:
                left = X[:, int(feat)] <= float(thr)
                right = ~left
                n_left, n_right = int(np.sum(left)), int(np.sum(right))
                if n_left < min_leaf or n_right < min_leaf:
                    continue
                off_left = float(np.mean(resid[left]))
                off_right = float(np.mean(resid[right]))
                pred = base_pred.copy()
                pred[left] += off_left
                pred[right] += off_right
                mse = float(np.mean((y - pred) ** 2))
                if mse < best["mse"]:
                    best = {
                        "use_rule": True,
                        "mse": mse,
                        "feature": int(feat),
                        "threshold": float(thr),
                        "offset_left": off_left,
                        "offset_right": off_right,
                    }

        self.intercept_ = float(intercept)
        self.coef_ = coef.astype(float)
        self.active_features_ = np.where(np.abs(self.coef_) > 1e-7)[0]
        self.base_mse_ = float(base_mse)

        rel_gain = (base_mse - best["mse"]) / (abs(base_mse) + 1e-12)
        self.use_rule_ = bool(best["use_rule"] and rel_gain >= float(self.min_relative_gain))
        if self.use_rule_:
            self.rule_feature_ = int(best["feature"])
            self.rule_threshold_ = float(best["threshold"])
            self.offset_left_ = float(best["offset_left"])
            self.offset_right_ = float(best["offset_right"])
            self.final_mse_ = float(best["mse"])
        else:
            self.rule_feature_ = -1
            self.rule_threshold_ = 0.0
            self.offset_left_ = 0.0
            self.offset_right_ = 0.0
            self.final_mse_ = float(base_mse)

        imp = np.abs(self.coef_).copy()
        if self.use_rule_:
            imp[self.rule_feature_] += 0.5 * (abs(self.offset_left_) + abs(self.offset_right_))
        self.feature_importance_ = imp
        self.feature_rank_ = np.argsort(self.feature_importance_)[::-1]
        return self

    def predict(self, X):
        check_is_fitted(self, ["coef_", "intercept_", "feature_importance_"])
        X = self._impute(X)
        pred = self.intercept_ + X @ self.coef_
        if self.use_rule_:
            left = X[:, self.rule_feature_] <= self.rule_threshold_
            pred[left] += self.offset_left_
            pred[~left] += self.offset_right_
        return pred

    def __str__(self):
        check_is_fitted(self, ["coef_", "intercept_", "feature_rank_"])
        lines = ["SparseBackboneOffsetRuleRegressor", "", "Prediction rule:"]
        active = np.where(np.abs(self.coef_) > 1e-7)[0]
        if active.size == 0:
            eq = f"y = {self.intercept_:+.6f}"
        else:
            order = active[np.argsort(np.abs(self.coef_[active]))[::-1]]
            terms = [f"{self.coef_[int(j)]:+.6f}*x{int(j)}" for j in order]
            eq = "y = " + f"{self.intercept_:+.6f} " + " ".join(terms)
        lines.append("  " + eq)
        if self.use_rule_:
            lines.append(
                f"  if x{self.rule_feature_} <= {self.rule_threshold_:.6f}: add {self.offset_left_:+.6f}"
            )
            lines.append(
                f"  else: add {self.offset_right_:+.6f}"
            )
        else:
            lines.append("  no threshold offset rule selected")

        lines.extend(["", "Feature importance (descending):"])
        for j in self.feature_rank_:
            lines.append(f"  x{int(j)}: {self.feature_importance_[int(j)]:.6f}")
        near_zero = [f"x{j}" for j, v in enumerate(np.abs(self.coef_)) if v < 1e-7]
        if near_zero:
            lines.append("Near-zero linear features: " + ", ".join(near_zero))
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys

_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseBackboneOffsetRuleRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseBackboneOffsetRuleV1"
model_description = "Sparse linear backbone with one optional threshold-based residual offset rule for simple piecewise calibration"
model_defs = [(model_shorthand_name, SparseBackboneOffsetRuleRegressor())]


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
