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


class SparseLinearHingeRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse linear model with a few residual hinge terms:
    1) Fit a sparse linear equation.
    2) Add a tiny number of one-feature hinge corrections for nonlinearity.
    """

    def __init__(
        self,
        max_linear_terms=7,
        max_hinge_terms=2,
        hinge_feature_pool=8,
        quantiles=(0.2, 0.4, 0.6, 0.8),
        min_hinge_gain=5e-4,
        random_state=42,
    ):
        self.max_linear_terms = max_linear_terms
        self.max_hinge_terms = max_hinge_terms
        self.hinge_feature_pool = hinge_feature_pool
        self.quantiles = quantiles
        self.min_hinge_gain = min_hinge_gain
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
        denom = (np.std(ac) + 1e-12) * (np.std(bc) + 1e-12)
        return abs(float(np.mean(ac * bc) / denom))

    def _hinge_value(self, X, term):
        xj = X[:, term["j"]]
        if term["kind"] == "pos":
            return np.maximum(0.0, xj - term["threshold"])
        return np.maximum(0.0, term["threshold"] - xj)

    def _hinge_text(self, term):
        if term["kind"] == "pos":
            return f"max(0, x{term['j']} - {term['threshold']:.3f})"
        return f"max(0, {term['threshold']:.3f} - x{term['j']})"

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

        selector = LassoCV(cv=3, random_state=self.random_state, n_alphas=30, max_iter=5000)
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

        pred = self.intercept_ + X[:, self.active_linear_features_] @ self.linear_coef_
        residual = y - pred

        nonlinear_scores = []
        for j in range(self.n_features_in_):
            xj = X[:, j]
            score = self._corr_abs(xj, residual) + 0.7 * self._corr_abs(xj * xj, residual)
            nonlinear_scores.append(score)
        ranked = np.argsort(np.asarray(nonlinear_scores))[::-1]
        pool = list(ranked[: min(self.hinge_feature_pool, self.n_features_in_)].astype(int))
        for j in self.active_linear_features_:
            if int(j) not in pool:
                pool.append(int(j))

        self.hinge_terms_ = []
        for _ in range(self.max_hinge_terms):
            base_mse = float(np.mean(residual ** 2))
            best = None
            best_gain = 0.0
            for j in pool:
                xj = X[:, int(j)]
                thresholds = np.unique(np.quantile(xj, self.quantiles))
                for thr in thresholds:
                    for kind in ("pos", "neg"):
                        candidate = {"j": int(j), "threshold": float(thr), "kind": kind}
                        z = self._hinge_value(X, candidate)
                        z_norm = float(np.dot(z, z))
                        if z_norm < 1e-10:
                            continue
                        coef = float(np.dot(residual, z) / z_norm)
                        if abs(coef) < 1e-8:
                            continue
                        new_residual = residual - coef * z
                        gain = base_mse - float(np.mean(new_residual ** 2))
                        if gain > best_gain:
                            best_gain = gain
                            best = {
                                "j": int(j),
                                "threshold": float(thr),
                                "kind": kind,
                                "coef": float(coef),
                            }
            if best is None or best_gain < self.min_hinge_gain:
                break
            residual = residual - best["coef"] * self._hinge_value(X, best)
            self.hinge_terms_.append(best)

        fi = np.zeros(self.n_features_in_, dtype=float)
        for j, c in zip(self.active_linear_features_, self.linear_coef_):
            fi[int(j)] += abs(float(c))
        for term in self.hinge_terms_:
            fi[int(term["j"])] += abs(float(term["coef"]))
        self.feature_importance_ = fi
        return self

    def predict(self, X):
        check_is_fitted(self, ["feature_importance_", "active_linear_features_", "linear_coef_", "hinge_terms_", "intercept_"])
        X = self._impute(X)
        y_hat = np.full(X.shape[0], self.intercept_, dtype=float)
        if self.active_linear_features_.size > 0:
            y_hat += X[:, self.active_linear_features_] @ self.linear_coef_
        for term in self.hinge_terms_:
            y_hat += term["coef"] * self._hinge_value(X, term)
        return y_hat

    def __str__(self):
        check_is_fitted(self, ["feature_importance_", "active_linear_features_", "linear_coef_", "hinge_terms_", "intercept_"])
        lines = ["SparseLinearHingeRegressor", "Prediction formula:", f"y = {self.intercept_:+.4f}"]
        linear_terms = sorted(
            zip(self.active_linear_features_, self.linear_coef_),
            key=lambda x: abs(float(x[1])),
            reverse=True,
        )
        for j, c in linear_terms:
            lines.append(f"  {float(c):+.4f} * x{int(j)}")
        for term in sorted(self.hinge_terms_, key=lambda t: abs(float(t["coef"])), reverse=True):
            lines.append(f"  {float(term['coef']):+.4f} * {self._hinge_text(term)}")
        lines.append(f"Total active terms: {len(linear_terms) + len(self.hinge_terms_)}")
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
SparseLinearHingeRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseLinHingeV2"
model_description = "Sparse L1-selected linear equation plus up to two greedy hinge residual terms for compact nonlinear correction"
model_defs = [(model_shorthand_name, SparseLinearHingeRegressor())]


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
