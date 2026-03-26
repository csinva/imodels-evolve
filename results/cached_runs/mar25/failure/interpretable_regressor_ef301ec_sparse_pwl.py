"""
Interpretable regressor autoresearch script.
Defines a scikit-learn compatible interpretable regressor and evaluates it
on interpretability tests and TabArena regression datasets (same suite used
for baselines in run_baselines.py).

Usage: uv run model.py
"""

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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "eval"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class InterpretableRegressor(BaseEstimator, RegressorMixin):
    """
    Feature-selected Ridge regression with piecewise-linear feature transforms.

    1. For each feature, create both the original feature and a ReLU transform
       (max(0, x - median)) to capture nonlinearity.
    2. Use LassoCV to select sparse features.
    3. Refit with Ridge on selected features for stable coefficients.

    The __str__() output mimics OLS format: explicit linear equation + coefficients table.
    """

    def __init__(self, max_features=10):
        self.max_features = max_features

    def fit(self, X, y):
        from sklearn.linear_model import RidgeCV
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
        n_orig = X.shape[1]

        # Create piecewise-linear features: original + ReLU(x - median)
        self.medians_ = np.median(X, axis=0)
        X_aug = self._augment(X)
        self.aug_names_ = []
        for i in range(n_orig):
            self.aug_names_.append(f"x{i}")
        for i in range(n_orig):
            self.aug_names_.append(f"relu(x{i}-{self.medians_[i]:.2f})")

        # Step 1: Lasso for feature selection
        lasso = LassoCV(cv=3, random_state=42, max_iter=5000)
        lasso.fit(X_aug, y)
        importance = np.abs(lasso.coef_)
        # Select top features by importance
        n_select = min(self.max_features, (importance > 1e-8).sum())
        n_select = max(n_select, 1)
        top_idx = np.argsort(importance)[::-1][:n_select]
        self.selected_idx_ = np.sort(top_idx)
        self.selected_names_ = [self.aug_names_[i] for i in self.selected_idx_]

        # Step 2: Ridge on selected features
        X_sel = X_aug[:, self.selected_idx_]
        self.ridge_ = RidgeCV(cv=3)
        self.ridge_.fit(X_sel, y)

        self.coef_ = self.ridge_.coef_
        self.intercept_ = float(self.ridge_.intercept_)
        return self

    def _augment(self, X):
        relu_part = np.maximum(0, X - self.medians_[np.newaxis, :])
        return np.hstack([X, relu_part])

    def predict(self, X):
        check_is_fitted(self, "ridge_")
        X = np.asarray(X, dtype=np.float64)
        X_aug = self._augment(X)
        X_sel = X_aug[:, self.selected_idx_]
        return self.ridge_.predict(X_sel)

    def __str__(self):
        check_is_fitted(self, "ridge_")
        names = self.selected_names_
        coefs = self.coef_

        # Build equation string
        equation_parts = []
        for c, n in zip(coefs, names):
            equation_parts.append(f"{c:.4f}*{n}")
        equation = " + ".join(equation_parts) + f" + {self.intercept_:.4f}"

        lines = [
            f"Sparse Piecewise-Linear Regression:  y = {equation}",
            "",
            "Coefficients:",
        ]
        for n, c in zip(names, coefs):
            lines.append(f"  {n}: {c:.4f}")
        lines.append(f"  intercept: {self.intercept_:.4f}")

        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
InterpretableRegressor.__module__ = "interpretable_regressor"


# ---------------------------------------------------------------------------
# Evaluation (do not edit any of the evaluation functions, only the names and model descriptions below this line)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()

    MODEL_NAME = "SparsePWL"
    model_defs = [(MODEL_NAME, InterpretableRegressor())]

    # Interpretability tests
    interp_results = run_all_interp_tests(model_defs)
    n_passed = sum(r["passed"] for r in interp_results)
    total = len(interp_results)

    # prediction performance (RMSE)
    dataset_rmses = evaluate_all_regressors(model_defs)
    rmse_vals = [v[MODEL_NAME] for v in dataset_rmses.values()
                 if not np.isnan(v.get(MODEL_NAME, float("nan")))]
    mean_rmse = float(np.mean(rmse_vals)) if rmse_vals else float("nan")

    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        git_hash = ""

    upsert_overall_results([{
        "commit":                             git_hash,
        "mean_rmse":                          f"{mean_rmse:.6f}" if not np.isnan(mean_rmse) else "",
        "frac_interpretability_tests_passed": f"{n_passed / total:.4f}",
        "status":                             "",
        "model_name":                         MODEL_NAME,
        "description":                        "Sparse piecewise-linear: Lasso feature selection + ReLU transforms + Ridge refit",
    }], RESULTS_DIR)

    # --- Plot ---
    overall_csv = os.path.join(RESULTS_DIR, "overall_results.csv")
    plot_interp_vs_performance(
        overall_csv,
        os.path.join(RESULTS_DIR, "interpretability_vs_performance.png"),
    )

    print()
    print("---")
    print(f"tests_passed:  {n_passed}/{total} ({n_passed/total:.2%})")
    print(f"mean_rmse:     {mean_rmse:.4f}")
    print(f"total_seconds: {time.time() - t0:.1f}s")

