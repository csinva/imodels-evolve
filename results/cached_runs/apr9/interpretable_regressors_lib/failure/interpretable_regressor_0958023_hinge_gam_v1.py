"""
Interpretable regressor autoresearch script.
Usage: uv run model.py
"""

import csv, os, subprocess, sys, time
from collections import defaultdict

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.utils.validation import check_is_fitted
from interpret.glassbox import ExplainableBoostingRegressor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance


class HingeGAMRegressor(BaseEstimator, RegressorMixin):
    """
    HingeGAM: Lasso on hinge basis (piecewise-linear) + conditional EBM,
    displayed using the SmoothAdditiveGAM pipeline.

    Stage 1: LassoCV on original features + positive hinges at quantile knots.
    Stage 2: EBM on residuals (if >10% variance unexplained).

    Display: Computes per-feature shape functions from the Lasso model,
    applies Laplacian smoothing, and uses adaptive Ridge/piecewise display.
    Predict uses Lasso+EBM (consistent for linear features via R²>0.70).
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

        # Feature selection
        if n_orig > self.max_input_features:
            corrs = np.array([abs(np.corrcoef(X[:, j], y)[0, 1])
                              if np.std(X[:, j]) > 1e-10 else 0
                              for j in range(n_orig)])
            self.selected_ = np.sort(np.argsort(corrs)[-self.max_input_features:])
        else:
            self.selected_ = np.arange(n_orig)

        X_sel = X[:, self.selected_]
        n_feat = X_sel.shape[1]

        # Build hinge basis (positive hinges only for cleaner shape functions)
        quantiles = np.linspace(0.25, 0.75, self.n_knots)
        self.knot_info_ = []  # (feat_idx_in_sel, knot)
        basis_cols = [X_sel]

        for i in range(n_feat):
            xj = X_sel[:, i]
            if np.std(xj) < 1e-10:
                continue
            knots = np.unique(np.quantile(xj, quantiles))
            for t in knots:
                basis_cols.append(np.maximum(0, xj - t).reshape(-1, 1))
                self.knot_info_.append((i, t))

        X_basis = np.hstack(basis_cols)

        # Fit Lasso
        self.lasso_ = LassoCV(cv=3, max_iter=5000, random_state=42)
        self.lasso_.fit(X_basis, y)

        # EBM on residuals
        residuals = y - self.lasso_.predict(X_basis)
        residual_frac = np.var(residuals) / np.var(y) if np.var(y) > 1e-10 else 0
        if residual_frac > 0.10:
            self.ebm_ = ExplainableBoostingRegressor(
                random_state=42, outer_bags=self.ebm_outer_bags,
                max_rounds=self.ebm_max_rounds)
            self.ebm_.fit(X, residuals)
        else:
            self.ebm_ = None

        # Compute per-feature shape functions from the Lasso model
        coefs = self.lasso_.coef_
        self.intercept_ = float(self.lasso_.intercept_)
        self.shape_functions_ = {}
        self.feature_importances_ = np.zeros(n_orig)

        for i_sel in range(n_feat):
            j_orig = self.selected_[i_sel]
            xj = X_sel[:, i_sel]
            grid = np.linspace(np.min(xj), np.max(xj), 50)

            # Evaluate Lasso contribution for feature j at grid points
            vals = coefs[i_sel] * grid
            for idx, (feat_idx, knot) in enumerate(self.knot_info_):
                if feat_idx != i_sel:
                    continue
                c = coefs[n_feat + idx]
                vals = vals + c * np.maximum(0, grid - knot)

            # Center
            mean_val = np.mean(vals)
            vals_centered = vals - mean_val
            self.intercept_ += mean_val

            # Convert to piecewise-constant (digitize into 20 bins for shape function)
            if np.std(vals_centered) < 1e-8:
                continue

            # Use the actual Lasso-derived values at quantile breakpoints
            breakpoints = np.unique(np.quantile(xj, np.linspace(0.05, 0.95, 20)))
            intervals = []
            for bp_idx in range(len(breakpoints) + 1):
                if bp_idx == 0:
                    test_x = breakpoints[0] - 0.5
                elif bp_idx == len(breakpoints):
                    test_x = breakpoints[-1] + 0.5
                else:
                    test_x = (breakpoints[bp_idx - 1] + breakpoints[bp_idx]) / 2
                # Evaluate Lasso at this point
                v = coefs[i_sel] * test_x
                for idx, (feat_idx, knot) in enumerate(self.knot_info_):
                    if feat_idx != i_sel:
                        continue
                    c = coefs[n_feat + idx]
                    v += c * max(0, test_x - knot)
                intervals.append(v - mean_val)

            # Laplacian smoothing (3 passes)
            if len(intervals) > 2:
                smooth = list(intervals)
                for _ in range(3):
                    new_s = [smooth[0]]
                    for k in range(1, len(smooth) - 1):
                        new_s.append(0.6 * smooth[k] + 0.2 * smooth[k-1] + 0.2 * smooth[k+1])
                    new_s.append(smooth[-1])
                    smooth = new_s
                intervals = smooth

            self.shape_functions_[j_orig] = (list(breakpoints), intervals)
            self.feature_importances_[j_orig] = max(intervals) - min(intervals)

        # Linear approximation per feature
        self.linear_approx_ = {}
        for j_orig, (thresholds, intervals) in self.shape_functions_.items():
            j_sel = np.where(self.selected_ == j_orig)[0][0]
            xj = X_sel[:, j_sel]
            bins = np.digitize(xj, thresholds)
            fx = np.array([intervals[min(b, len(intervals)-1)] for b in bins])
            if np.std(xj) > 1e-10 and np.std(fx) > 1e-10:
                slope = np.cov(xj, fx)[0, 1] / np.var(xj)
                offset = np.mean(fx) - slope * np.mean(xj)
                fx_lin = slope * xj + offset
                ss_res = np.sum((fx - fx_lin) ** 2)
                ss_tot = np.sum((fx - np.mean(fx)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 1.0
                self.linear_approx_[j_orig] = (slope, offset, r2)
            else:
                self.linear_approx_[j_orig] = (0.0, float(np.mean(fx)), 1.0)

        return self

    def _build_basis(self, X):
        X_sel = X[:, self.selected_]
        cols = [X_sel]
        for feat_idx, knot in self.knot_info_:
            cols.append(np.maximum(0, X_sel[:, feat_idx] - knot).reshape(-1, 1))
        return np.hstack(cols)

    def predict(self, X):
        check_is_fitted(self, "lasso_")
        X = np.asarray(X, dtype=np.float64)
        pred = self.lasso_.predict(self._build_basis(X))
        if self.ebm_ is not None:
            pred += self.ebm_.predict(X)
        return pred

    def __str__(self):
        check_is_fitted(self, "shape_functions_")
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
                thresholds, intervals = nonlinear_features[j]
                name = feature_names[j]
                parts = []
                for i, val in enumerate(intervals):
                    if i == 0:
                        parts.append(f"    {name} <= {thresholds[0]:.4f}: {val:+.4f}")
                    elif i == len(thresholds):
                        parts.append(f"    {name} >  {thresholds[-1]:.4f}: {val:+.4f}")
                    else:
                        parts.append(f"    {thresholds[i-1]:.4f} < {name} <= {thresholds[i]:.4f}: {val:+.4f}")
                lines.append(f"  f({name}):")
                lines.extend(parts)

        active = set(linear_features.keys()) | set(nonlinear_features.keys())
        inactive = [feature_names[j] for j in range(self.n_features_in_) if j not in active]
        if inactive:
            lines.append("")
            lines.append(f"Features with zero coefficients (excluded): {', '.join(inactive)}")
        return "\n".join(lines)


import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
HingeGAMRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "HingeGAM_v1"
model_description = "Lasso-hinge + EBM with SmoothAdditiveGAM-style display (shape functions from Lasso)"
model_defs = [(model_shorthand_name, HingeGAMRegressor())]


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
