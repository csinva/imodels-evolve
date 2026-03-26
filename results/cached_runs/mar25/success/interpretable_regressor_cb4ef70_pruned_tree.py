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

    def __init__(self, max_leaf_nodes=16, min_samples_leaf=5):
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]

        # Grow a tree with max_leaf_nodes constraint, then prune via CV
        base_tree = DecisionTreeRegressor(
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
        )
        # Get cost complexity pruning path
        full_tree = DecisionTreeRegressor(
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
        )
        full_tree.fit(X, y)
        path = full_tree.cost_complexity_pruning_path(X, y)
        ccp_alphas = path.ccp_alphas

        # Cross-validate to find best alpha
        if len(ccp_alphas) > 20:
            # Subsample alphas for speed
            indices = np.linspace(0, len(ccp_alphas) - 1, 20, dtype=int)
            ccp_alphas = ccp_alphas[indices]

        best_alpha = 0.0
        best_score = -np.inf
        kf = KFold(n_splits=3, shuffle=True, random_state=42)

        for alpha in ccp_alphas:
            scores = []
            for train_idx, val_idx in kf.split(X):
                tree = DecisionTreeRegressor(
                    ccp_alpha=alpha,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=42,
                )
                tree.fit(X[train_idx], y[train_idx])
                scores.append(tree.score(X[val_idx], y[val_idx]))
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_alpha = alpha

        # Fit final tree with best alpha
        self.tree_ = DecisionTreeRegressor(
            ccp_alpha=best_alpha,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
        )
        self.tree_.fit(X, y)
        self.ccp_alpha_ = best_alpha
        return self

    def predict(self, X):
        check_is_fitted(self, "tree_")
        return self.tree_.predict(np.asarray(X, dtype=np.float64))

    def __str__(self):
        check_is_fitted(self, "tree_")
        feature_names = [f"x{i}" for i in range(self.n_features_in_)]
        tree_text = export_text(self.tree_, feature_names=feature_names, max_depth=10)
        n_leaves = self.tree_.get_n_leaves()
        return (f"Decision Tree Regressor (pruned, {n_leaves} leaves, "
                f"ccp_alpha={self.ccp_alpha_:.6f}):\n{tree_text}")


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
InterpretableRegressor.__module__ = "interpretable_regressor"


# ---------------------------------------------------------------------------
# Evaluation (do not edit any of the evaluation functions, only the names and model descriptions below this line)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()

    MODEL_NAME = "PrunedTree"
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
        "description":                        "CV-pruned decision tree with cost-complexity pruning",
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

