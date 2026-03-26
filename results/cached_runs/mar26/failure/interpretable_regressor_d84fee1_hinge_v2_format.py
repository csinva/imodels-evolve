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

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LassoCV
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
    Sparse piecewise-linear model using hinge basis functions.

    For each feature x_j, creates basis functions:
      - x_j (linear)
      - max(0, x_j - t) for learned thresholds t (positive hinges)
      - max(0, t - x_j) for learned thresholds t (negative hinges)

    Then fits LassoCV on this expanded feature set for automatic sparsity.
    The __str__() shows an explicit equation with only active terms,
    making it trivially traceable by an LLM.
    """

    def __init__(self, n_knots=3, max_terms=15):
        self.n_knots = n_knots
        self.max_terms = max_terms

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape
        self.n_features_ = n_features

        # Compute knot positions from quantiles
        self.knots_ = {}
        for j in range(n_features):
            quantiles = np.linspace(15, 85, self.n_knots)
            self.knots_[j] = np.percentile(X[:, j], quantiles)

        # Build expanded feature matrix
        Z, self.term_names_ = self._build_features(X)

        # Fit sparse linear model
        self.lasso_ = LassoCV(cv=min(5, n_samples), max_iter=5000, random_state=42)
        self.lasso_.fit(Z, y)

        # Record active terms
        self.coef_ = self.lasso_.coef_
        self.intercept_ = float(self.lasso_.intercept_)
        self.active_mask_ = np.abs(self.coef_) > 1e-8
        self.n_active_ = int(self.active_mask_.sum())

        return self

    def _build_features(self, X):
        """Build hinge basis expansion."""
        n_samples, n_features = X.shape
        columns = []
        names = []

        for j in range(n_features):
            xj = X[:, j]
            fname = f"x{j}"

            # Linear term
            columns.append(xj)
            names.append(fname)

            # Hinge terms at knot positions
            for t in self.knots_[j]:
                # Positive hinge: max(0, x - t)
                columns.append(np.maximum(0, xj - t))
                if t >= 0:
                    names.append(f"max(0, {fname}-{t:.2f})")
                else:
                    names.append(f"max(0, {fname}+{abs(t):.2f})")

                # Negative hinge: max(0, t - x)
                columns.append(np.maximum(0, t - xj))
                if t >= 0:
                    names.append(f"max(0, {t:.2f}-{fname})")
                else:
                    names.append(f"max(0, -{abs(t):.2f}-{fname})")

        return np.column_stack(columns), names

    def predict(self, X):
        check_is_fitted(self, "lasso_")
        X = np.asarray(X, dtype=np.float64)
        Z, _ = self._build_features(X)
        return self.lasso_.predict(Z)

    def __str__(self):
        check_is_fitted(self, "lasso_")
        lines = [
            "Sparse Piecewise-Linear Regression Model:",
            f"  y = {self.intercept_:.4f}",
        ]

        # Show active terms as equation
        active_terms = []
        for i, (name, coef) in enumerate(zip(self.term_names_, self.coef_)):
            if abs(coef) > 1e-8:
                active_terms.append((name, coef))

        if not active_terms:
            lines.append(f"  (constant model, no active features)")
        else:
            for name, coef in active_terms:
                sign = "+" if coef > 0 else "-"
                lines[1] += f" {sign} {abs(coef):.4f}*{name}"

            lines.append("")
            lines.append(f"Active terms ({len(active_terms)} of {len(self.term_names_)}):")
            for name, coef in sorted(active_terms, key=lambda x: -abs(x[1])):
                lines.append(f"  {coef:+.4f} * {name}")

            # Show which features are used vs excluded
            used_features = set()
            for name, coef in active_terms:
                for j in range(self.n_features_):
                    if f"x{j}" in name:
                        used_features.add(j)
                        break

            unused = [f"x{j}" for j in range(self.n_features_) if j not in used_features]
            if unused:
                lines.append(f"\nFeatures with zero effect (excluded): {', '.join(unused)}")

        lines.append(f"\nTo predict: substitute feature values into the equation above.")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
HingeLinearRegressor.__module__ = "interpretable_regressor"


# ---------------------------------------------------------------------------
# Evaluation (do not edit any of the evaluation functions, only the names and model descriptions below this line)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()

    model_defs = [("HingeLinearV2", HingeLinearRegressor())]

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
    model_name = "HingeLinearV2"
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
    from collections import defaultdict
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
    mean_rank = avg_rank.get(model_name, float("nan"))

    upsert_overall_results([{
        "commit":                             git_hash,
        "mean_rank":                          f"{mean_rank:.2f}" if not np.isnan(mean_rank) else "",
        "frac_interpretability_tests_passed": f"{n_passed / total:.4f}" if total > 0 else "",
        "status":                             "",
        "model_name":                         "HingeLinearV2",
        "description":                        "Sparse hinge-linear model with improved knot formatting",
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

