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
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class HingeLinearRegressor(BaseEstimator, RegressorMixin):
    """
    Adaptive linear model with hinge (ReLU) basis functions.

    For each feature x_j, creates hinge features: max(0, x_j - t_k) for
    several data-driven thresholds t_k (quantiles). This creates a
    piecewise-linear basis that captures nonlinearities while keeping the
    model as a linear combination of interpretable terms.

    The __str__ shows a clean linear equation over the original features
    and hinge terms, which the LLM can compute exactly for point predictions.
    """

    def __init__(self, n_hinges=3, max_input_features=15):
        self.n_hinges = n_hinges
        self.max_input_features = max_input_features

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]

        # Feature selection: keep top features by correlation
        n_feat = min(X.shape[1], self.max_input_features)
        if X.shape[1] > n_feat:
            corrs = np.array([abs(np.corrcoef(X[:, j], y)[0, 1])
                              if np.std(X[:, j]) > 1e-10 else 0
                              for j in range(X.shape[1])])
            self.selected_features_ = np.sort(np.argsort(corrs)[-n_feat:])
        else:
            self.selected_features_ = np.arange(X.shape[1])

        X_sel = X[:, self.selected_features_]
        n_sel = X_sel.shape[1]

        # Build hinge features
        self.hinge_thresholds_ = {}
        feature_cols = [X_sel]  # Start with original features
        feature_names = [f"x{j}" for j in self.selected_features_]

        quantiles = np.linspace(0.2, 0.8, self.n_hinges)

        for i in range(n_sel):
            xj = X_sel[:, i]
            if np.std(xj) < 1e-10:
                continue
            thresholds = np.quantile(xj, quantiles)
            # Remove duplicates
            thresholds = np.unique(thresholds)
            self.hinge_thresholds_[i] = thresholds

            for t in thresholds:
                hinge_col = np.maximum(0, xj - t).reshape(-1, 1)
                feature_cols.append(hinge_col)
                feature_names.append(f"max(0, x{self.selected_features_[i]}-{t:.2f})")

        self.augmented_feature_names_ = feature_names
        X_aug = np.hstack(feature_cols)

        # Fit Ridge on augmented features
        self.ridge_ = RidgeCV(cv=3)
        self.ridge_.fit(X_aug, y)

        return self

    def _augment(self, X):
        """Transform X into augmented feature matrix."""
        X_sel = X[:, self.selected_features_]
        cols = [X_sel]

        for i, thresholds in self.hinge_thresholds_.items():
            xj = X_sel[:, i]
            for t in thresholds:
                cols.append(np.maximum(0, xj - t).reshape(-1, 1))

        return np.hstack(cols)

    def predict(self, X):
        check_is_fitted(self, "ridge_")
        X = np.asarray(X, dtype=np.float64)
        X_aug = self._augment(X)
        return self.ridge_.predict(X_aug)

    def __str__(self):
        check_is_fitted(self, "ridge_")

        coefs = self.ridge_.coef_
        intercept = self.ridge_.intercept_
        names = self.augmented_feature_names_

        # Show as a linear equation
        lines = [
            f"Hinge Linear Regression (Ridge, alpha={self.ridge_.alpha_:.4g}):",
            f"  y = (linear combination of features and hinge terms)",
            "",
            "Coefficients:",
        ]

        # Sort by absolute coefficient value
        active = [(names[i], coefs[i]) for i in range(len(coefs)) if abs(coefs[i]) > 1e-6]
        active.sort(key=lambda x: abs(x[1]), reverse=True)

        for n, c in active:
            lines.append(f"  {n}: {c:.4f}")

        lines.append(f"  intercept: {intercept:.4f}")

        # Also show the equation form for small models
        if len(active) <= 25:
            terms = [f"{c:.4f}*{n}" for n, c in active]
            eq = " + ".join(terms) + f" + {intercept:.4f}"
            lines.insert(1, f"  y = {eq}")

        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
HingeLinearRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "HingeLinear_v1"
model_description = "Ridge on original features + max(0, x-t) hinge terms at quantile thresholds"
model_defs = [(model_shorthand_name, HingeLinearRegressor())]


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
