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
    Pruned decision tree with cross-validated complexity selection.

    Uses cost-complexity pruning (ccp_alpha) selected via cross-validation
    to find the optimal tree complexity. This balances accuracy and
    interpretability by growing a full tree then pruning back.

    The __str__() uses sklearn's export_text format which GPT-4o parses well.
    """

    def __init__(self, min_samples_leaf=5, max_leaf_nodes_options=None):
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes_options = max_leaf_nodes_options

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]

        # Try multiple tree complexities and pick best via CV
        if self.max_leaf_nodes_options is None:
            options = [4, 6, 8, 10, 12, 16, 20, 25, 30]
        else:
            options = self.max_leaf_nodes_options

        best_score = -np.inf
        best_nodes = options[0]
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for n_leaves in options:
            scores = []
            for train_idx, val_idx in kf.split(X):
                tree = DecisionTreeRegressor(
                    max_leaf_nodes=n_leaves,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=42,
                )
                tree.fit(X[train_idx], y[train_idx])
                scores.append(tree.score(X[val_idx], y[val_idx]))
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_nodes = n_leaves

        # Fit final tree with best complexity
        self.tree_ = DecisionTreeRegressor(
            max_leaf_nodes=best_nodes,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
        )
        self.tree_.fit(X, y)
        self.best_max_leaf_nodes_ = best_nodes
        return self

    def predict(self, X):
        check_is_fitted(self, "tree_")
        return self.tree_.predict(np.asarray(X, dtype=np.float64))

    def __str__(self):
        check_is_fitted(self, "tree_")
        feature_names = [f"x{i}" for i in range(self.n_features_in_)]
        tree_text = export_text(self.tree_, feature_names=feature_names, max_depth=10)
        n_leaves = self.tree_.get_n_leaves()
        return (f"Decision Tree Regressor (max_leaf_nodes={self.best_max_leaf_nodes_}, "
                f"{n_leaves} leaves):\n{tree_text}")


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
InterpretableRegressor.__module__ = "interpretable_regressor"


# ---------------------------------------------------------------------------
# Evaluation (do not edit any of the evaluation functions, only the names and model descriptions below this line)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()

    MODEL_NAME = "CVTree"
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
        "description":                        "CV-selected decision tree: 5-fold CV over max_leaf_nodes options",
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

