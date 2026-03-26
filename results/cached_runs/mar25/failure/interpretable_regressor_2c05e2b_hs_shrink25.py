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

    def __init__(self, min_samples_leaf=5, shrinkage_lambda=1.0):
        self.min_samples_leaf = min_samples_leaf
        self.shrinkage_lambda = shrinkage_lambda

    def _apply_hierarchical_shrinkage(self, tree_model, X, y):
        """Apply hierarchical shrinkage: shrink leaf values toward parent averages.

        For each node, the shrunk value is a weighted average of the node's
        own mean and the cumulative path means, weighted by sample counts.
        This is similar to HSTree from imodels.
        """
        tree = tree_model.tree_
        n_nodes = tree.node_count
        children_left = tree.children_left
        children_right = tree.children_right
        n_samples = tree.n_node_samples
        values = tree.value.copy()  # shape (n_nodes, 1, 1)

        lam = self.shrinkage_lambda

        # Compute parent map
        parent = np.full(n_nodes, -1, dtype=int)
        for i in range(n_nodes):
            if children_left[i] != -1:
                parent[children_left[i]] = i
                parent[children_right[i]] = i

        # For each node, compute shrunk value by walking path from root
        shrunk_values = np.zeros(n_nodes)
        shrunk_values[0] = values[0, 0, 0]

        # BFS to compute shrunk values
        from collections import deque
        queue = deque([0])
        while queue:
            node = queue.popleft()
            if parent[node] == -1:
                # Root: no shrinkage
                shrunk_values[node] = values[node, 0, 0]
            else:
                p = parent[node]
                # Shrink toward parent value
                node_val = values[node, 0, 0]
                parent_val = shrunk_values[p]
                n = n_samples[node]
                # Hierarchical shrinkage formula
                shrunk_values[node] = parent_val + (node_val - values[p, 0, 0]) * n / (n + lam)

            if children_left[node] != -1:
                queue.append(children_left[node])
                queue.append(children_right[node])

        # Update leaf values in the tree
        for i in range(n_nodes):
            if children_left[i] == -1:  # leaf
                values[i, 0, 0] = shrunk_values[i]

        tree.value[:] = values

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]

        # CV over max_leaf_nodes AND shrinkage_lambda
        # Keep trees small (≤20 leaves) for interpretability
        leaf_options = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25]
        lambda_options = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]

        best_score = -np.inf
        best_params = (leaf_options[0], lambda_options[0])
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for n_leaves in leaf_options:
            for lam in lambda_options:
                scores = []
                for train_idx, val_idx in kf.split(X):
                    tree = DecisionTreeRegressor(
                        max_leaf_nodes=n_leaves,
                        min_samples_leaf=self.min_samples_leaf,
                        random_state=42,
                    )
                    tree.fit(X[train_idx], y[train_idx])
                    if lam > 0:
                        self.shrinkage_lambda = lam
                        self._apply_hierarchical_shrinkage(tree, X[train_idx], y[train_idx])
                    scores.append(tree.score(X[val_idx], y[val_idx]))
                mean_score = np.mean(scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = (n_leaves, lam)

        best_nodes, best_lam = best_params
        self.shrinkage_lambda = best_lam
        self.tree_ = DecisionTreeRegressor(
            max_leaf_nodes=best_nodes,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
        )
        self.tree_.fit(X, y)
        if best_lam > 0:
            self._apply_hierarchical_shrinkage(self.tree_, X, y)
        self.best_max_leaf_nodes_ = best_nodes
        self.best_lambda_ = best_lam
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

    MODEL_NAME = "HSShrinkTree25"
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
        "description":                        "CV-tuned tree with hierarchical shrinkage (max 25 leaves)",
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

