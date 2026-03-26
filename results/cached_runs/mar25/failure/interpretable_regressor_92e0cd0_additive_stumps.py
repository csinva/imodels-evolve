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
    Greedy additive decision stumps with feature selection.

    Builds an additive model by greedily adding shallow decision stumps
    (depth-1 trees), each time picking the feature whose stump best reduces
    the current residual. After each stump is added, the residual is updated.
    This produces a sparse, additive, interpretable model whose __str__()
    shows a simple formula: intercept + stump_1(x_i) + stump_2(x_j) + ...

    Each stump is just: if x_i <= threshold: value_left, else: value_right
    This makes it trivial for an LLM to trace any prediction.
    """

    def __init__(self, n_stumps=6, min_samples_leaf=10):
        self.n_stumps = n_stumps
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
        self.intercept_ = float(np.mean(y))
        residual = y - self.intercept_

        self.stumps_ = []  # list of (feature_idx, threshold, value_left, value_right)

        for _ in range(self.n_stumps):
            best_mse = np.inf
            best_stump = None

            for j in range(X.shape[1]):
                col = X[:, j]
                # Try a set of candidate thresholds
                percentiles = np.percentile(col, np.linspace(10, 90, 17))
                for threshold in percentiles:
                    left_mask = col <= threshold
                    right_mask = ~left_mask
                    n_left = left_mask.sum()
                    n_right = right_mask.sum()
                    if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                        continue
                    val_left = residual[left_mask].mean()
                    val_right = residual[right_mask].mean()
                    pred = np.where(left_mask, val_left, val_right)
                    mse = np.mean((residual - pred) ** 2)
                    if mse < best_mse:
                        best_mse = mse
                        best_stump = (j, float(threshold), float(val_left), float(val_right))

            if best_stump is None:
                break

            self.stumps_.append(best_stump)
            j, threshold, val_left, val_right = best_stump
            pred = np.where(X[:, j] <= threshold, val_left, val_right)
            residual = residual - pred

        return self

    def predict(self, X):
        check_is_fitted(self, "stumps_")
        X = np.asarray(X, dtype=np.float64)
        pred = np.full(X.shape[0], self.intercept_)
        for j, threshold, val_left, val_right in self.stumps_:
            pred += np.where(X[:, j] <= threshold, val_left, val_right)
        return pred

    def __str__(self):
        check_is_fitted(self, "stumps_")
        lines = []
        lines.append("Additive Stump Model")
        lines.append(f"prediction = {self.intercept_:.4f}", )

        # Identify which features are used
        used_features = sorted(set(j for j, _, _, _ in self.stumps_))
        lines.append(f"active features: {['x' + str(i) for i in used_features]}")
        lines.append("")
        lines.append("Rules (applied additively):")

        for i, (j, threshold, val_left, val_right) in enumerate(self.stumps_):
            lines.append(f"  + if x{j} <= {threshold:.4f}: {val_left:+.4f}, else: {val_right:+.4f}")

        # Also show the equivalent computation step by step
        lines.append("")
        lines.append("To compute a prediction:")
        lines.append(f"  Start with {self.intercept_:.4f}")
        for i, (j, threshold, val_left, val_right) in enumerate(self.stumps_):
            lines.append(f"  Step {i+1}: add {val_left:+.4f} if x{j} <= {threshold:.4f}, else add {val_right:+.4f}")

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

    MODEL_NAME = "AdditiveStumps"
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
        "description":                        "Greedy additive decision stumps: sparse additive model of if-else rules",
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

