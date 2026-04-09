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
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
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


class SparsePolyRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse polynomial regressor: generates degree-2 polynomial features
    (including interactions), standardizes them, then fits Lasso (L1) to
    select a sparse set of terms. The result is a compact polynomial equation.

    The __str__ output shows only the active (non-zero) terms as a clean
    equation, which is highly readable by an LLM — similar to linear models
    that score ~77% on interpretability tests, but with better predictive power
    from the interaction and quadratic terms.
    """

    def __init__(self, max_features_in=15):
        self.max_features_in = max_features_in

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]

        # Limit input features for tractability
        n_feat = min(X.shape[1], self.max_features_in)
        if X.shape[1] > n_feat:
            # Select top features by correlation with y
            corrs = np.array([abs(np.corrcoef(X[:, j], y)[0, 1]) if np.std(X[:, j]) > 1e-10 else 0 for j in range(X.shape[1])])
            self.selected_features_ = np.argsort(corrs)[-n_feat:]
            self.selected_features_.sort()
            X = X[:, self.selected_features_]
        else:
            self.selected_features_ = np.arange(X.shape[1])

        # Generate polynomial features
        self.poly_ = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
        X_poly = self.poly_.fit_transform(X)

        # Standardize
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_poly)

        # Fit Lasso for sparsity
        self.lasso_ = LassoCV(cv=3, max_iter=5000, random_state=42)
        self.lasso_.fit(X_scaled, y)

        # Store feature names for display
        orig_names = [f"x{i}" for i in self.selected_features_]
        self.poly_feature_names_ = self.poly_.get_feature_names_out(orig_names)

        return self

    def predict(self, X):
        check_is_fitted(self, "lasso_")
        X = np.asarray(X, dtype=np.float64)
        X = X[:, self.selected_features_]
        X_poly = self.poly_.transform(X)
        X_scaled = self.scaler_.transform(X_poly)
        return self.lasso_.predict(X_scaled)

    def __str__(self):
        check_is_fitted(self, "lasso_")

        coefs = self.lasso_.coef_
        intercept = self.lasso_.intercept_
        names = self.poly_feature_names_

        # Get active (non-zero) terms
        active = [(names[i], coefs[i]) for i in range(len(coefs)) if abs(coefs[i]) > 1e-8]
        active.sort(key=lambda x: abs(x[1]), reverse=True)

        # Build equation string
        lines = [
            f"Sparse Polynomial Regression (Lasso, alpha={self.lasso_.alpha_:.4g}):",
        ]

        # Show equation
        if len(active) <= 20:
            terms = [f"{c:.4f}*{n}" for n, c in active]
            eq = " + ".join(terms) + f" + {intercept:.4f}"
            lines.append(f"  y = {eq}")
        else:
            lines.append(f"  y = (see coefficients below) + {intercept:.4f}")

        lines.append("")
        lines.append(f"Active terms ({len(active)} non-zero out of {len(coefs)}):")
        for n, c in active:
            lines.append(f"  {n}: {c:.4f}")

        zeroed = [names[i] for i in range(len(coefs)) if abs(coefs[i]) <= 1e-8]
        if zeroed:
            lines.append(f"  Zero terms (excluded): {len(zeroed)} features")

        lines.append(f"  intercept: {intercept:.4f}")

        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparsePolyRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "SparsePoly_v1"
model_description = "Lasso on degree-2 polynomial features (interactions + quadratics) with feature selection"
model_defs = [(model_shorthand_name, SparsePolyRegressor())]


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
