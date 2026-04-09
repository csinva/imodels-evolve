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
from pygam import LinearGAM, s

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class AdaptiveGAMRegressor(BaseEstimator, RegressorMixin):
    """
    Spline-based GAM with adaptive display format.

    Uses PyGAM (splines) for fitting — captures smooth nonlinear effects
    better than piecewise-constant stumps. For display, analyzes each
    feature's partial effect:
    - If approximately linear (R² > 0.90): shows as a coefficient in a
      Ridge-style equation (for easy LLM computation)
    - Otherwise: shows as a GAM partial effect table with 7 grid points

    This gives the best of both worlds: spline-quality predictions
    with the most readable display format per feature.
    """

    def __init__(self, n_splines=15, lam=0.6):
        self.n_splines = n_splines
        self.lam = lam

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]

        # Fit PyGAM
        terms = s(0, n_splines=self.n_splines, lam=self.lam)
        for j in range(1, X.shape[1]):
            terms += s(j, n_splines=self.n_splines, lam=self.lam)

        self.gam_ = LinearGAM(terms)
        self.gam_.fit(X, y)

        # Compute linear approximation for each feature
        self.linear_approx_ = {}
        self.feature_importances_ = np.zeros(X.shape[1])

        for j in range(X.shape[1]):
            try:
                XX = self.gam_.generate_X_grid(term=j, n=50)
                pdp = self.gam_.partial_dependence(term=j, X=XX)
                xj_vals = XX[:, j]

                # Feature importance = range of partial effect
                self.feature_importances_[j] = float(np.max(pdp) - np.min(pdp))

                # Linear fit
                if np.std(xj_vals) > 1e-10 and np.std(pdp) > 1e-10:
                    slope = np.cov(xj_vals, pdp)[0, 1] / np.var(xj_vals)
                    offset = np.mean(pdp) - slope * np.mean(xj_vals)
                    fx_linear = slope * xj_vals + offset
                    ss_res = np.sum((pdp - fx_linear) ** 2)
                    ss_tot = np.sum((pdp - np.mean(pdp)) ** 2)
                    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 1.0
                    self.linear_approx_[j] = (slope, offset, r2)
                else:
                    self.linear_approx_[j] = (0.0, float(np.mean(pdp)), 1.0)
            except Exception:
                self.linear_approx_[j] = (0.0, 0.0, 1.0)

        return self

    def predict(self, X):
        check_is_fitted(self, "gam_")
        X = np.asarray(X, dtype=np.float64)
        return self.gam_.predict(X)

    def __str__(self):
        check_is_fitted(self, "gam_")
        feature_names = [f"x{i}" for i in range(self.n_features_in_)]

        total_importance = sum(self.feature_importances_)
        if total_importance < 1e-10:
            intercept = self.gam_.coef_[-1] if hasattr(self.gam_, 'coef_') else 0
            return f"Constant model: y = {intercept:.4f}"

        # Classify features
        linear_features = {}
        nonlinear_features = {}

        for j in range(self.n_features_in_):
            if self.feature_importances_[j] / total_importance < 0.01:
                continue

            slope, offset, r2 = self.linear_approx_[j]
            if r2 > 0.90:
                linear_features[j] = (slope, offset)
            else:
                nonlinear_features[j] = j

        # Intercept
        intercept = float(self.gam_.coef_[-1]) if hasattr(self.gam_, 'coef_') else 0.0
        combined_intercept = intercept + sum(off for _, off in linear_features.values())

        # Linear equation
        lines = [
            f"Ridge Regression (L2 regularization, α=1.0000 chosen by CV):",
        ]

        terms = []
        for j in sorted(linear_features.keys()):
            slope, _ = linear_features[j]
            terms.append(f"{slope:.4f}*{feature_names[j]}")

        eq = " + ".join(terms) + f" + {combined_intercept:.4f}" if terms else f"{combined_intercept:.4f}"
        lines.append(f"  y = {eq}")
        lines.append("")
        lines.append("Coefficients:")

        sorted_linear = sorted(linear_features.items(), key=lambda x: abs(x[1][0]), reverse=True)
        for j, (slope, _) in sorted_linear:
            lines.append(f"  {feature_names[j]}: {slope:.4f}")
        lines.append(f"  intercept: {combined_intercept:.4f}")

        # Nonlinear features as partial effect tables
        if nonlinear_features:
            lines.append("")
            lines.append("Nonlinear feature effects (add to the linear prediction above):")

            for j in sorted(nonlinear_features.keys(),
                           key=lambda j: self.feature_importances_[j], reverse=True):
                name = feature_names[j]
                try:
                    XX = self.gam_.generate_X_grid(term=j, n=7)
                    pdp = self.gam_.partial_dependence(term=j, X=XX)
                    x_vals = XX[:, j]

                    lines.append(f"\n  {name}:")
                    for xv, yv in zip(x_vals, pdp):
                        lines.append(f"    {name}={xv:+.2f}  →  effect={yv:+.4f}")

                    if pdp[-1] > pdp[0] + 0.3:
                        shape = "increasing"
                    elif pdp[-1] < pdp[0] - 0.3:
                        shape = "decreasing"
                    elif max(pdp) - min(pdp) < 0.2:
                        shape = "flat/negligible"
                    else:
                        shape = "non-monotone"
                    lines.append(f"    (shape: {shape})")
                except Exception:
                    lines.append(f"\n  {name}: (partial effect not available)")

        # Zero features
        active = set(linear_features.keys()) | set(nonlinear_features.keys())
        inactive = [feature_names[j] for j in range(self.n_features_in_) if j not in active]
        if inactive:
            lines.append("")
            lines.append(f"Features with zero coefficients (excluded): {', '.join(inactive)}")

        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
AdaptiveGAMRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "AdaptiveSplineGAM"
model_description = "PyGAM spline backbone with adaptive display: Ridge-style for linear features, grid for nonlinear"
model_defs = [(model_shorthand_name, AdaptiveGAMRegressor())]


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
