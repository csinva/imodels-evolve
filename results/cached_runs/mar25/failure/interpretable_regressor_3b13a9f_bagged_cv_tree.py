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

    def __init__(self, min_samples_leaf=5, n_trees=3):
        self.min_samples_leaf = min_samples_leaf
        self.n_trees = n_trees

    def _cv_best_leaves(self, X, y, seed=42):
        """Find best max_leaf_nodes via 5-fold CV."""
        options = [4, 6, 8, 10, 12, 16, 20, 25, 30]
        best_score = -np.inf
        best_nodes = options[0]
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        for n_leaves in options:
            scores = []
            for train_idx, val_idx in kf.split(X):
                tree = DecisionTreeRegressor(
                    max_leaf_nodes=n_leaves,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=seed,
                )
                tree.fit(X[train_idx], y[train_idx])
                scores.append(tree.score(X[val_idx], y[val_idx]))
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_nodes = n_leaves
        return best_nodes

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
        n = X.shape[0]

        self.trees_ = []
        rng = np.random.RandomState(42)

        for i in range(self.n_trees):
            # Bootstrap sample
            idx = rng.choice(n, size=n, replace=True)
            X_boot, y_boot = X[idx], y[idx]

            best_nodes = self._cv_best_leaves(X_boot, y_boot, seed=42 + i)
            tree = DecisionTreeRegressor(
                max_leaf_nodes=best_nodes,
                min_samples_leaf=self.min_samples_leaf,
                random_state=42 + i,
            )
            tree.fit(X_boot, y_boot)
            self.trees_.append((best_nodes, tree))

        return self

    def predict(self, X):
        check_is_fitted(self, "trees_")
        X = np.asarray(X, dtype=np.float64)
        preds = np.array([tree.predict(X) for _, tree in self.trees_])
        return preds.mean(axis=0)

    def __str__(self):
        check_is_fitted(self, "trees_")
        feature_names = [f"x{i}" for i in range(self.n_features_in_)]
        lines = [f"Ensemble of {self.n_trees} Decision Trees (prediction = average of all trees):"]
        for i, (n_leaves, tree) in enumerate(self.trees_):
            tree_text = export_text(tree, feature_names=feature_names, max_depth=10)
            actual_leaves = tree.get_n_leaves()
            lines.append(f"\n--- Tree {i+1} (max_leaf_nodes={n_leaves}, {actual_leaves} leaves) ---")
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

    MODEL_NAME = "BaggedCVTree"
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
        "description":                        "Bagged ensemble of 3 CV-tuned decision trees averaged for prediction",
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

