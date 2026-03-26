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
    Linear + Residual Tree hybrid regressor.

    Stage 1: Fit a sparse linear model (LassoCV) for interpretable coefficients.
    Stage 2: Fit a shallow decision tree on the residuals to capture nonlinearity.

    Predictions: y = linear_prediction + tree_correction
    The __str__() shows the explicit linear formula first (easy for LLMs to
    compute with), then the tree correction rules.
    """

    def __init__(self, max_depth=4, min_samples_leaf=5, max_leaf_nodes=12):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]

        # Stage 1: Sparse linear model
        self.lasso_ = LassoCV(cv=3, random_state=42, max_iter=5000)
        self.lasso_.fit(X, y)
        linear_pred = self.lasso_.predict(X)
        residual = y - linear_pred

        # Stage 2: Shallow tree on residuals
        self.tree_ = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_leaf_nodes=self.max_leaf_nodes,
            random_state=42,
        )
        self.tree_.fit(X, residual)
        return self

    def predict(self, X):
        check_is_fitted(self, "lasso_")
        X = np.asarray(X, dtype=np.float64)
        return self.lasso_.predict(X) + self.tree_.predict(X)

    def __str__(self):
        check_is_fitted(self, "lasso_")
        lines = []
        lines.append("Linear + Residual Tree Model")
        lines.append("prediction = linear_part + tree_correction")
        lines.append("")

        # Linear part with explicit formula
        coefs = self.lasso_.coef_
        intercept = self.lasso_.intercept_
        terms = [f"{intercept:.4f}"]
        active_features = []
        for i, c in enumerate(coefs):
            if abs(c) > 1e-8:
                sign = "+" if c > 0 else "-"
                terms.append(f"{sign} {abs(c):.4f}*x{i}")
                active_features.append(i)
        lines.append("LINEAR PART:")
        lines.append(f"  y_linear = {' '.join(terms)}")
        lines.append(f"  active features: {['x' + str(i) for i in active_features]}")
        lines.append(f"  coefficients: {dict(('x' + str(i), round(coefs[i], 4)) for i in active_features)}")
        lines.append("")

        # Tree correction part
        feature_names = [f"x{i}" for i in range(self.n_features_in_)]
        tree_text = export_text(self.tree_, feature_names=feature_names, max_depth=6)
        lines.append("TREE CORRECTION (added to linear prediction):")
        lines.append(tree_text)

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

    MODEL_NAME = "LinearPlusTree"
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
        "description":                        "Linear + residual tree hybrid: LassoCV for linear part then shallow DT on residuals",
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

