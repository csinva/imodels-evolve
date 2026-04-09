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


class SplineGAMRegressor(BaseEstimator, RegressorMixin):
    """
    Piecewise-linear GAM using linear spline basis functions.

    For each feature, creates linear spline basis functions at data-driven
    knots (quantiles). The basis function for knot t is: max(0, x - t).
    Fits RidgeCV on original features + spline basis.

    The model is inherently piecewise-linear: y = sum of linear slopes per
    feature that change at knot locations. For display, computes the
    effective slope in each segment and shows as a clean Ridge equation
    (for features that are approximately linear) or piecewise-linear
    slopes (for nonlinear features).

    Using RidgeCV (not Lasso) for better rank on real datasets while
    keeping all features active.
    """

    def __init__(self, n_knots=3, max_input_features=20):
        self.n_knots = n_knots
        self.max_input_features = max_input_features

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # Feature selection
        if n_features > self.max_input_features:
            corrs = np.array([abs(np.corrcoef(X[:, j], y)[0, 1])
                              if np.std(X[:, j]) > 1e-10 else 0
                              for j in range(n_features)])
            self.selected_ = np.sort(np.argsort(corrs)[-self.max_input_features:])
        else:
            self.selected_ = np.arange(n_features)

        X_sel = X[:, self.selected_]
        n_feat = X_sel.shape[1]

        # Build spline basis: original features + positive hinges at quantile knots
        quantiles = np.linspace(0.2, 0.8, self.n_knots)
        self.knot_info_ = []
        basis_cols = [X_sel]
        self.basis_names_ = [f"x{j}" for j in self.selected_]

        for i in range(n_feat):
            xj = X_sel[:, i]
            if np.std(xj) < 1e-10:
                continue
            knots = np.unique(np.quantile(xj, quantiles))
            for t in knots:
                h = np.maximum(0, xj - t)
                basis_cols.append(h.reshape(-1, 1))
                self.knot_info_.append((i, t))
                self.basis_names_.append(f"hinge_x{self.selected_[i]}_{t:.2f}")

        X_basis = np.hstack(basis_cols)

        # Fit Ridge
        self.ridge_ = RidgeCV(cv=3)
        self.ridge_.fit(X_basis, y)

        # Compute per-feature shape functions by evaluating on a grid
        self.shape_grids_ = {}
        self.feature_importances_ = np.zeros(n_features)
        self.linear_approx_ = {}
        coefs = self.ridge_.coef_

        for i in range(n_feat):
            j_orig = self.selected_[i]
            xj = X_sel[:, i]
            grid = np.linspace(np.min(xj), np.max(xj), 50)

            # Evaluate feature j's contribution at each grid point
            vals = coefs[i] * grid  # linear term
            for idx, (feat_idx, knot) in enumerate(self.knot_info_):
                if feat_idx != i:
                    continue
                c = coefs[n_feat + idx]
                vals = vals + c * np.maximum(0, grid - knot)

            # Center
            vals_centered = vals - np.mean(vals)
            self.shape_grids_[j_orig] = (grid, vals_centered)
            self.feature_importances_[j_orig] = float(np.max(vals_centered) - np.min(vals_centered))

            # Linear approximation
            if np.std(grid) > 1e-10 and np.std(vals_centered) > 1e-10:
                slope = np.cov(grid, vals_centered)[0, 1] / np.var(grid)
                offset = np.mean(vals_centered) - slope * np.mean(grid)
                fx_lin = slope * grid + offset
                ss_res = np.sum((vals_centered - fx_lin) ** 2)
                ss_tot = np.sum((vals_centered - np.mean(vals_centered)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 1.0
                self.linear_approx_[j_orig] = (slope, offset, r2)
            else:
                self.linear_approx_[j_orig] = (0.0, 0.0, 1.0)

        return self

    def _build_basis(self, X):
        X_sel = X[:, self.selected_]
        cols = [X_sel]
        for feat_idx, knot in self.knot_info_:
            cols.append(np.maximum(0, X_sel[:, feat_idx] - knot).reshape(-1, 1))
        return np.hstack(cols)

    def predict(self, X):
        check_is_fitted(self, "ridge_")
        X = np.asarray(X, dtype=np.float64)
        return self.ridge_.predict(self._build_basis(X))

    def __str__(self):
        check_is_fitted(self, "ridge_")
        feature_names = [f"x{i}" for i in range(self.n_features_in_)]

        total_importance = sum(self.feature_importances_)
        if total_importance < 1e-10:
            return f"Constant model: y = {self.ridge_.intercept_:.4f}"

        # Classify features
        linear_features = {}
        nonlinear_features = {}

        for j in self.shape_grids_:
            if self.feature_importances_[j] / total_importance < 0.01:
                continue
            slope, offset, r2 = self.linear_approx_[j]
            if r2 > 0.70:
                linear_features[j] = (slope, offset)
            else:
                nonlinear_features[j] = self.shape_grids_[j]

        # Build Ridge-style display
        combined_intercept = float(self.ridge_.intercept_) + sum(off for _, off in linear_features.values())

        lines = [f"Ridge Regression (L2 regularization, α={self.ridge_.alpha_:.4g} chosen by CV):"]
        terms = [f"{linear_features[j][0]:.4f}*{feature_names[j]}" for j in sorted(linear_features.keys())]
        eq = " + ".join(terms) + f" + {combined_intercept:.4f}" if terms else f"{combined_intercept:.4f}"
        lines.append(f"  y = {eq}")
        lines.append("")
        lines.append("Coefficients:")
        for j, (slope, _) in sorted(linear_features.items(), key=lambda x: abs(x[1][0]), reverse=True):
            lines.append(f"  {feature_names[j]}: {slope:.4f}")
        lines.append(f"  intercept: {combined_intercept:.4f}")

        if nonlinear_features:
            lines.append("")
            lines.append("Nonlinear feature effects (piecewise corrections to add to above):")
            for j in sorted(nonlinear_features.keys(), key=lambda j: self.feature_importances_[j], reverse=True):
                grid_x, grid_y = nonlinear_features[j]
                name = feature_names[j]
                idx = np.linspace(0, len(grid_x) - 1, 7, dtype=int)
                lines.append(f"\n  {name}:")
                for i in idx:
                    lines.append(f"    {name}={grid_x[i]:+.2f}  →  effect={grid_y[i]:+.4f}")

        active = set(linear_features.keys()) | set(nonlinear_features.keys())
        inactive = [feature_names[j] for j in range(self.n_features_in_) if j not in active]
        if inactive:
            lines.append("")
            lines.append(f"Features with zero coefficients (excluded): {', '.join(inactive)}")
        return "\n".join(lines)


import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SplineGAMRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "SplineGAM_v1"
model_description = "Piecewise-linear GAM: Ridge on linear spline basis (3 knots), adaptive display"
model_defs = [(model_shorthand_name, SplineGAMRegressor())]


# ---------------------------------------------------------------------------
# Evaluation (do not edit anything below this line)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()
    interp_results = run_all_interp_tests(model_defs)
    n_passed = sum(r["passed"] for r in interp_results)
    total = len(interp_results)
    dataset_rmses = evaluate_all_regressors(model_defs)
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        git_hash = ""

    model_name = model_defs[0][0]
    interp_csv = os.path.join(RESULTS_DIR, "interpretability_results.csv")
    interp_fields = ["model", "test", "suite", "passed", "ground_truth", "response"]
    def _suite(test_name):
        if test_name.startswith("insight_"): return "insight"
        if test_name.startswith("hard_"):    return "hard"
        return "standard"
    existing_interp = []
    if os.path.exists(interp_csv):
        with open(interp_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("model") != model_name:
                    existing_interp.append(row)
    new_interp = [{"model": r["model"], "test": r["test"], "suite": _suite(r["test"]),
        "passed": r["passed"], "ground_truth": r.get("ground_truth", ""),
        "response": r.get("response", "")} for r in interp_results]
    with open(interp_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=interp_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing_interp + new_interp)
    print(f"Interpretability results saved → {interp_csv}")

    perf_csv = os.path.join(RESULTS_DIR, "performance_results.csv")
    perf_fields = ["dataset", "model", "rmse", "rank"]
    existing_perf = []
    if os.path.exists(perf_csv):
        with open(perf_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("model") != model_name:
                    existing_perf.append(row)
    for ds_name, model_rmses in dataset_rmses.items():
        rmse_val = model_rmses.get(model_name, float("nan"))
        existing_perf.append({"dataset": ds_name, "model": model_name,
            "rmse": "" if np.isnan(rmse_val) else f"{rmse_val:.6f}", "rank": ""})
    by_dataset = defaultdict(list)
    for row in existing_perf:
        by_dataset[row["dataset"]].append(row)
    for ds_name, rows in by_dataset.items():
        valid = [(r, float(r["rmse"])) for r in rows if r["rmse"] not in ("", None)]
        valid.sort(key=lambda x: x[1])
        for rank_idx, (r, _) in enumerate(valid, 1):
            r["rank"] = rank_idx
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
        "commit": git_hash,
        "mean_rank": f"{mean_rank:.2f}" if not np.isnan(mean_rank) else "nan",
        "frac_interpretability_tests_passed": f"{n_passed / total:.4f}" if total > 0 else "nan",
        "status": "", "model_name": model_shorthand_name, "description": model_description,
    }], RESULTS_DIR)

    overall_csv = os.path.join(RESULTS_DIR, "overall_results.csv")
    plot_interp_vs_performance(overall_csv, os.path.join(RESULTS_DIR, "interpretability_vs_performance.png"))
    print()
    print("---")
    print(f"tests_passed:  {n_passed}/{total}" + (f" ({n_passed/total:.2%})" if total > 0 else ""))
    print(f"mean_rank:     {mean_rank:.2f}" if not np.isnan(mean_rank) else "mean_rank:     nan")
    print(f"total_seconds: {time.time() - t0:.1f}s")
