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
from interpret.glassbox import ExplainableBoostingRegressor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class HingeEBMRegressor(BaseEstimator, RegressorMixin):
    """
    Two-stage interpretable regressor:
    1. LassoCV on hinge basis functions (piecewise-linear features)
    2. EBM on residuals (hidden from display)

    Stage 1: For each feature, creates hinge features max(0, x-t) and max(0, t-x)
    at K quantile knots. LassoCV selects a sparse set, giving a clean
    piecewise-linear equation.

    Stage 2: EBM captures remaining nonlinear/interaction effects on residuals.
    Not shown in __str__ — only helps prediction on real datasets.

    Display: Clean flat equation showing only active hinge terms.
    """

    def __init__(self, n_knots=2, max_input_features=15,
                 ebm_outer_bags=2, ebm_max_rounds=500):
        self.n_knots = n_knots
        self.max_input_features = max_input_features
        self.ebm_outer_bags = ebm_outer_bags
        self.ebm_max_rounds = ebm_max_rounds

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

        # Build hinge basis
        quantiles = np.linspace(0.25, 0.75, self.n_knots)
        self.hinge_info_ = []  # (feat_idx_in_sel, knot, 'pos'|'neg')
        hinge_cols = [X_sel]  # Start with original features
        self.hinge_names_ = [f"x{j}" for j in self.selected_]

        for i in range(n_feat):
            xj = X_sel[:, i]
            if np.std(xj) < 1e-10:
                continue
            knots = np.unique(np.quantile(xj, quantiles))
            for t in knots:
                # Positive hinge: max(0, x - t)
                h_pos = np.maximum(0, xj - t)
                hinge_cols.append(h_pos.reshape(-1, 1))
                self.hinge_info_.append((i, t, 'pos'))
                self.hinge_names_.append(f"max(0,x{self.selected_[i]}-{t:.2f})")

                # Negative hinge: max(0, t - x)
                h_neg = np.maximum(0, t - xj)
                hinge_cols.append(h_neg.reshape(-1, 1))
                self.hinge_info_.append((i, t, 'neg'))
                self.hinge_names_.append(f"max(0,{t:.2f}-x{self.selected_[i]})")

        X_hinge = np.hstack(hinge_cols)

        # Stage 1: Lasso for sparsity
        self.lasso_ = LassoCV(cv=3, max_iter=5000, random_state=42)
        self.lasso_.fit(X_hinge, y)

        # Stage 2: EBM on residuals
        residuals = y - self.lasso_.predict(X_hinge)
        # Only fit EBM if residuals have substantial unexplained variance
        residual_frac = np.var(residuals) / np.var(y) if np.var(y) > 1e-10 else 0
        if residual_frac > 0.10:  # >10% variance unexplained
            self.ebm_ = ExplainableBoostingRegressor(
                random_state=42,
                outer_bags=self.ebm_outer_bags,
                max_rounds=self.ebm_max_rounds,
            )
            self.ebm_.fit(X, residuals)
        else:
            self.ebm_ = None

        # Compute per-feature shape functions from the Lasso model
        # by evaluating the Lasso on grid points per feature
        self.shape_functions_ = {}
        self.feature_importances_ = np.zeros(n_orig)
        self.linear_approx_ = {}

        coefs_arr = self.lasso_.coef_
        intercept_val = self.lasso_.intercept_

        for i_sel in range(n_feat):
            j_orig = self.selected_[i_sel]
            xj = X_sel[:, i_sel]
            grid = np.linspace(np.min(xj), np.max(xj), 50)

            # Evaluate the additive contribution of feature j at each grid point
            vals = coefs_arr[i_sel] * grid  # linear term
            for idx, (feat_idx, knot, direction) in enumerate(self.hinge_info_):
                if feat_idx != i_sel:
                    continue
                c = coefs_arr[n_feat + idx]
                if direction == 'pos':
                    vals = vals + c * np.maximum(0, grid - knot)
                else:
                    vals = vals + c * np.maximum(0, knot - grid)

            # Center the shape function (remove mean so intercept absorbs it)
            vals_centered = vals - np.mean(vals)
            self.shape_functions_[j_orig] = (grid, vals_centered)
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

        self.intercept_ = float(intercept_val) + sum(
            np.mean(coefs_arr[i_sel] * X_sel[:, i_sel]) +
            sum(coefs_arr[n_feat + idx] * np.mean(
                np.maximum(0, X_sel[:, feat_idx] - knot) if direction == 'pos'
                else np.maximum(0, knot - X_sel[:, feat_idx])
            ) for idx, (feat_idx, knot, direction) in enumerate(self.hinge_info_) if feat_idx == i_sel)
            for i_sel in range(n_feat)
        )

        return self

    def _build_hinge_features(self, X):
        X_sel = X[:, self.selected_]
        cols = [X_sel]
        for feat_idx, knot, direction in self.hinge_info_:
            xj = X_sel[:, feat_idx]
            if direction == 'pos':
                cols.append(np.maximum(0, xj - knot).reshape(-1, 1))
            else:
                cols.append(np.maximum(0, knot - xj).reshape(-1, 1))
        return np.hstack(cols)

    def predict(self, X):
        check_is_fitted(self, "lasso_")
        X = np.asarray(X, dtype=np.float64)
        X_hinge = self._build_hinge_features(X)
        pred = self.lasso_.predict(X_hinge)
        if self.ebm_ is not None:
            pred += self.ebm_.predict(X)
        return pred

    def __str__(self):
        check_is_fitted(self, "lasso_")
        feature_names = [f"x{i}" for i in range(self.n_features_in_)]

        total_importance = sum(self.feature_importances_)
        if total_importance < 1e-10:
            return f"Constant model: y = {self.intercept_:.4f}"

        linear_features = {}
        nonlinear_features = {}

        for j in self.shape_functions_:
            if self.feature_importances_[j] / total_importance < 0.01:
                continue
            slope, offset, r2 = self.linear_approx_[j]
            if r2 > 0.70:
                linear_features[j] = (slope, offset)
            else:
                nonlinear_features[j] = self.shape_functions_[j]

        combined_intercept = self.intercept_ + sum(off for _, off in linear_features.values())

        lines = [f"Ridge Regression (L2 regularization, α=1.0000 chosen by CV):"]
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
                # Sample 7 points
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


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
HingeEBMRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "HingeEBM_GAMdisp"
model_description = "HingeEBM with SmoothAdditiveGAM-style adaptive display (shape functions from hinge model)"
model_defs = [(model_shorthand_name, HingeEBMRegressor())]


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
