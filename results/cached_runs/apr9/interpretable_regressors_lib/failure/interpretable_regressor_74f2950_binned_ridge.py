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


class BinnedRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Binned Ridge Regression: creates quantile-based bins for each feature,
    then fits Ridge on the original features + bin indicator features.

    This is an additive model where each feature's effect is piecewise-constant
    (from the bins) plus linear (from the original feature). Presented as a
    clean Ridge equation.

    The key insight: adding bins to Ridge captures nonlinear effects while
    keeping a clean linear equation format that the LLM can compute from.
    """

    def __init__(self, n_bins=4, max_input_features=20):
        self.n_bins = n_bins
        self.max_input_features = max_input_features

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
        n_samples, n_orig = X.shape

        # Feature selection if too many
        if n_orig > self.max_input_features:
            corrs = np.array([abs(np.corrcoef(X[:, j], y)[0, 1])
                              if np.std(X[:, j]) > 1e-10 else 0
                              for j in range(n_orig)])
            self.selected_ = np.sort(np.argsort(corrs)[-self.max_input_features:])
        else:
            self.selected_ = np.arange(n_orig)

        X_sel = X[:, self.selected_]
        n_feat = X_sel.shape[1]

        # Create bin edges per feature
        self.bin_edges_ = {}
        aug_cols = [X_sel]  # original features
        self.aug_names_ = [f"x{j}" for j in self.selected_]

        quantiles = np.linspace(1/(self.n_bins+1), self.n_bins/(self.n_bins+1), self.n_bins)

        for i in range(n_feat):
            xj = X_sel[:, i]
            if np.std(xj) < 1e-10:
                continue
            edges = np.unique(np.quantile(xj, quantiles))
            self.bin_edges_[i] = edges

            for e_idx, edge in enumerate(edges):
                # Indicator: x > edge
                indicator = (xj > edge).astype(np.float64)
                aug_cols.append(indicator.reshape(-1, 1))
                self.aug_names_.append(f"I(x{self.selected_[i]}>{edge:.2f})")

        X_aug = np.hstack(aug_cols)

        # Fit Ridge
        self.ridge_ = RidgeCV(cv=3)
        self.ridge_.fit(X_aug, y)

        return self

    def _augment(self, X):
        X_sel = X[:, self.selected_]
        n_feat = X_sel.shape[1]
        cols = [X_sel]

        for i in range(n_feat):
            if i not in self.bin_edges_:
                continue
            xj = X_sel[:, i]
            for edge in self.bin_edges_[i]:
                cols.append((xj > edge).astype(np.float64).reshape(-1, 1))

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
        names = self.aug_names_

        # Show as Ridge equation
        equation = " + ".join(f"{c:.4f}*{n}" for c, n in zip(coefs, names))
        equation += f" + {intercept:.4f}"

        lines = [
            f"Ridge Regression (L2 regularization, α={self.ridge_.alpha_:.4g} chosen by CV):",
            f"  y = {equation}",
            "",
            "Coefficients:",
        ]

        for n, c in zip(names, coefs):
            lines.append(f"  {n}: {c:.4f}")
        lines.append(f"  intercept: {intercept:.4f}")

        # Zero features
        inactive = [f"x{j}" for j in range(self.n_features_in_) if j not in self.selected_]
        if inactive:
            lines.append(f"  Features with zero coefficients (excluded): {', '.join(inactive)}")

        lines.append("")
        lines.append("Note: I(condition) = 1 if condition is true, 0 otherwise")

        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
BinnedRidgeRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "BinnedRidge_v1"
model_description = "Ridge on original features + quantile-based bin indicators (4 bins per feature)"
model_defs = [(model_shorthand_name, BinnedRidgeRegressor())]


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
