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


class SparseRuleResidualRegressor(BaseEstimator, RegressorMixin):
    """
    Compact sparse rule-linear regressor:
    1) Fit a sparse linear backbone.
    2) Add a few one-feature threshold residual rules.
    """

    def __init__(
        self,
        max_linear_terms=6,
        max_rules=3,
        rule_feature_pool=8,
        min_rule_samples=0.10,
        min_rule_gain=1e-4,
        random_state=42,
    ):
        self.max_linear_terms = max_linear_terms
        self.max_rules = max_rules
        self.rule_feature_pool = rule_feature_pool
        self.min_rule_samples = min_rule_samples
        self.min_rule_gain = min_rule_gain
        self.random_state = random_state

    def _impute(self, X):
        X = np.asarray(X, dtype=float).copy()
        nan_mask = ~np.isfinite(X)
        if nan_mask.any():
            X[nan_mask] = np.take(self.feature_medians_, np.where(nan_mask)[1])
        return X

    def _corr_abs(self, a, b):
        ac = a - np.mean(a)
        bc = b - np.mean(b)
        return abs(float(np.mean(ac * bc) / ((np.std(ac) + 1e-12) * (np.std(bc) + 1e-12))))

    def _rank_features(self, X, y):
        scores = np.array([self._corr_abs(X[:, j], y) for j in range(X.shape[1])], dtype=float)
        return np.argsort(scores)[::-1]

    def _rule_mask(self, X, rule):
        if rule["op"] == ">":
            return X[:, rule["j"]] > rule["threshold"]
        return X[:, rule["j"]] <= rule["threshold"]

    def _rule_text(self, rule):
        return f"I(x{rule['j']} {rule['op']} {rule['threshold']:.3f})"

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.feature_medians_ = np.nanmedian(X, axis=0)
        X = self._impute(X)

        x_mean = np.mean(X, axis=0)
        x_std = np.std(X, axis=0)
        x_std[x_std < 1e-12] = 1.0
        Xs = (X - x_mean) / x_std

        selector = LassoCV(cv=3, random_state=self.random_state, n_alphas=30, max_iter=4000)
        selector.fit(Xs, y)
        coef_unscaled = selector.coef_ / x_std
        active = np.where(np.abs(coef_unscaled) > 1e-8)[0]
        if active.size == 0:
            active = np.array([int(np.argmax(np.abs(coef_unscaled)))], dtype=int)
        if active.size > self.max_linear_terms:
            keep = np.argsort(np.abs(coef_unscaled[active]))[::-1][: self.max_linear_terms]
            active = active[keep]
        self.active_linear_features_ = np.array(sorted(active.tolist()), dtype=int)

        reg = LinearRegression()
        reg.fit(X[:, self.active_linear_features_], y)
        self.intercept_ = float(reg.intercept_)
        self.linear_coef_ = reg.coef_.astype(float)

        y_hat = self.intercept_ + X[:, self.active_linear_features_] @ self.linear_coef_
        residual = y - y_hat

        ranked = self._rank_features(X, residual)
        pool = list(ranked[: min(self.rule_feature_pool, self.n_features_in_)].astype(int))
        for j in self.active_linear_features_:
            if int(j) not in pool:
                pool.append(int(j))

        self.rules_ = []
        n = X.shape[0]
        min_count = max(5, int(self.min_rule_samples * n))
        quantiles = [0.2, 0.35, 0.5, 0.65, 0.8]

        for _ in range(self.max_rules):
            resid_var = float(np.mean(residual ** 2))
            best = None
            best_gain = 0.0
            for j in pool:
                xj = X[:, j]
                thresholds = np.unique(np.quantile(xj, quantiles))
                for thr in thresholds:
                    for op in (">", "<="):
                        rule = {"j": int(j), "threshold": float(thr), "op": op, "coef": 0.0}
                        mask = self._rule_mask(X, rule)
                        count = int(np.sum(mask))
                        if count < min_count or count > n - min_count:
                            continue
                        coef = float(np.mean(residual[mask]))
                        if abs(coef) < 1e-8:
                            continue
                        new_residual = residual.copy()
                        new_residual[mask] = new_residual[mask] - coef
                        gain = resid_var - float(np.mean(new_residual ** 2))
                        if gain > best_gain:
                            best_gain = gain
                            best = {"j": int(j), "threshold": float(thr), "op": op, "coef": coef}

            if best is None or best_gain < self.min_rule_gain:
                break
            mask = self._rule_mask(X, best)
            residual[mask] = residual[mask] - best["coef"]
            self.rules_.append(best)

        fi = np.zeros(self.n_features_in_, dtype=float)
        for j, c in zip(self.active_linear_features_, self.linear_coef_):
            fi[int(j)] += abs(float(c))
        for rule in self.rules_:
            fi[int(rule["j"])] += abs(float(rule["coef"]))
        self.feature_importance_ = fi
        return self

    def predict(self, X):
        check_is_fitted(self, ["feature_importance_", "active_linear_features_", "linear_coef_", "rules_", "intercept_"])
        X = self._impute(X)
        y_hat = np.full(X.shape[0], self.intercept_, dtype=float)
        if self.active_linear_features_.size > 0:
            y_hat += X[:, self.active_linear_features_] @ self.linear_coef_
        for rule in self.rules_:
            mask = self._rule_mask(X, rule)
            y_hat[mask] += rule["coef"]
        return y_hat

    def __str__(self):
        check_is_fitted(self, ["feature_importance_", "active_linear_features_", "linear_coef_", "rules_", "intercept_"])
        lines = ["SparseRuleResidualRegressor", "Prediction formula:", f"y = {self.intercept_:+.4f}"]
        linear_terms = sorted(
            zip(self.active_linear_features_, self.linear_coef_),
            key=lambda x: abs(float(x[1])),
            reverse=True,
        )
        for j, c in linear_terms:
            lines.append(f"  {float(c):+.4f} * x{int(j)}")
        for rule in sorted(self.rules_, key=lambda r: abs(float(r["coef"])), reverse=True):
            lines.append(f"  {float(rule['coef']):+.4f} * {self._rule_text(rule)}")
        lines.append("")
        lines.append("Feature importance (sum of absolute term coefficients):")
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
SparseRuleResidualRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseRuleResidualV1"
model_description = "Sparse linear backbone with a few additive threshold residual rules learned greedily for compact nonlinear corrections"
model_defs = [(model_shorthand_name, SparseRuleResidualRegressor())]


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
