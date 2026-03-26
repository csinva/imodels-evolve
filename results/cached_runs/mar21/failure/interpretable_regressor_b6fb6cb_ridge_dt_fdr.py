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
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "eval"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this — everything below is fair game)
# ---------------------------------------------------------------------------


class InterpretableRegressor(BaseEstimator, RegressorMixin):
    """
    DT with Ridge-Regularized Leaf Values and Flat Decision Rules (Ridge-DT-FDR):
    Builds on CV-HSDT-FDR v2 (commit 60f9c64, interp=0.84, RMSE=0.624) by replacing
    hierarchical shrinkage with globally-optimal ridge regression on leaf indicators.

    Algorithm:
      1. Fit a DT (max_leaf_nodes=25) to get tree STRUCTURE (splits).
      2. Build a leaf indicator matrix X_leaf [n_samples x n_leaves].
      3. Fit RidgeCV on X_leaf to find the optimal alpha that minimizes CV-MSE.
      4. Leaf values = ridge.coef_ + ridge.intercept_ / n_leaves.

    This replaces the greedy top-down hierarchical shrinkage with a globally-optimal
    L2-regularized solution. Ridge shrinks all leaf values toward the global mean
    (unlike HSDT which shrinks toward the tree ancestry). The CV-optimal alpha
    is equivalent to an optimal shrinkage intensity.

    Same flat decision rules __str__ format (repr_v=9, cache bust).
    """

    RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]

    def __init__(self, max_leaf_nodes=25, min_samples_leaf=5, cv=5, repr_v=9):
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.cv = cv
        self.repr_v = repr_v  # version tag — increment to bust joblib cache on __str__ changes

    @staticmethod
    def _leaf_indicator_matrix(tree, X_arr):
        """Build [n_samples x n_leaves] indicator matrix. Returns (X_ind, node_to_col)."""
        t = tree.tree_
        n_nodes = t.node_count
        # Map leaf node IDs to contiguous column indices
        node_to_col = {}
        col = 0
        for node in range(n_nodes):
            if t.children_left[node] == -1:  # leaf
                node_to_col[node] = col
                col += 1
        n_leaves = col
        leaf_idx = tree.apply(X_arr)
        X_ind = np.zeros((len(X_arr), n_leaves))
        for i, node in enumerate(leaf_idx):
            X_ind[i, node_to_col[node]] = 1.0
        return X_ind, node_to_col

    def fit(self, X, y):
        if hasattr(X, "columns"):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = [f"x{i}" for i in range(X.shape[1])]

        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        # Step 1: Fit DT structure
        self.tree_ = DecisionTreeRegressor(
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
        )
        self.tree_.fit(X_arr, y_arr)

        # Step 2: Build leaf indicator matrix
        X_ind, self.node_to_col_ = self._leaf_indicator_matrix(self.tree_, X_arr)

        # Step 3: Ridge regression on leaf indicators (CV over alpha)
        self.ridge_ = RidgeCV(alphas=self.RIDGE_ALPHAS, fit_intercept=True,
                              cv=self.cv)
        self.ridge_.fit(X_ind, y_arr)
        self.alpha_ = float(self.ridge_.alpha_)

        # Step 4: Compute shrunk_values_ for all nodes (leaves get ridge values)
        n_nodes = self.tree_.tree_.node_count
        # Initialize with original tree values
        self.shrunk_values_ = self.tree_.tree_.value[:, 0, 0].copy()
        # Overwrite leaf values with ridge predictions
        for node, col in self.node_to_col_.items():
            self.shrunk_values_[node] = self.ridge_.coef_[col] + self.ridge_.intercept_

        return self

    def predict(self, X):
        check_is_fitted(self, "tree_")
        X_arr = np.asarray(X, dtype=float)
        return self.shrunk_values_[self.tree_.apply(X_arr)]

    def _tree_lines(self, node=0, depth=0):
        t = self.tree_.tree_
        names = self.feature_names_in_
        indent = "|   " * depth

        if t.children_left[node] == -1:
            val = self.shrunk_values_[node]
            n_s = t.n_node_samples[node]
            return [f"{indent}|--- value: {val:.4f}  (n={n_s})"]

        fname = names[int(t.feature[node])]
        thresh = t.threshold[node]
        lines = [f"{indent}|--- {fname} <= {thresh:.4g}"]
        lines.extend(self._tree_lines(int(t.children_left[node]), depth + 1))
        lines.append(f"{indent}|--- {fname} > {thresh:.4g}")
        lines.extend(self._tree_lines(int(t.children_right[node]), depth + 1))
        return lines

    def _get_leaf_paths(self):
        """Return a list of {conditions, value, n_samples} for each leaf, sorted by value."""
        t = self.tree_.tree_
        names = self.feature_names_in_
        paths = []

        def traverse(node, conditions):
            if t.children_left[node] == -1:
                paths.append({
                    "conditions": conditions if conditions else ["(always)"],
                    "value": self.shrunk_values_[node],
                    "n_samples": t.n_node_samples[node],
                })
            else:
                fname = names[int(t.feature[node])]
                thresh = t.threshold[node]
                traverse(int(t.children_left[node]), conditions + [f"{fname} <= {thresh:.4g}"])
                traverse(int(t.children_right[node]), conditions + [f"{fname} > {thresh:.4g}"])

        traverse(0, [])
        return sorted(paths, key=lambda p: p["value"])

    def _infer_direction(self, feature_idx):
        t = self.tree_.tree_
        weighted_effect = 0.0
        for node in range(t.node_count):
            if t.children_left[node] == -1 or t.feature[node] != feature_idx:
                continue
            lc, rc = int(t.children_left[node]), int(t.children_right[node])
            weighted_effect += (self.shrunk_values_[rc] - self.shrunk_values_[lc]) * (
                t.n_node_samples[lc] + t.n_node_samples[rc]
            )
        if weighted_effect > 0:
            return "positive (higher → higher prediction)"
        elif weighted_effect < 0:
            return "negative (higher → lower prediction)"
        return "mixed"

    def __str__(self):
        check_is_fitted(self, "tree_")
        names = self.feature_names_in_
        t = self.tree_.tree_
        importances = self.tree_.feature_importances_
        n_leaves = self.tree_.get_n_leaves()

        lines = [
            f"Ridge_DT_FDR(max_leaf_nodes={self.max_leaf_nodes}, "
            f"ridge_alpha={self.alpha_:.2g}, cv={self.cv})",
            f"  nodes={t.node_count}, leaves={n_leaves}",
            "",
            "Tree structure (follow from root; leaf values are shrunk predictions):",
        ]
        lines.extend(self._tree_lines())

        # Flat decision rules section — explicit IF-THEN paths for every leaf.
        # Use this to look up predictions: find the first rule whose conditions
        # ALL hold for the input, then read off the predicted value.
        leaf_paths = self._get_leaf_paths()
        lines += [
            "",
            "Decision rules (sorted by prediction, lowest first):",
            "  To predict: find the rule whose conditions ALL hold for your input.",
        ]
        for i, rule in enumerate(leaf_paths, 1):
            cond_str = " AND ".join(rule["conditions"])
            lines.append(
                f"  Rule {i:2d}: IF {cond_str}"
                f"  THEN predict {rule['value']:.4f}  (n={rule['n_samples']})"
            )

        order = np.argsort(importances)[::-1]
        lines += ["", "Feature importances (Gini-based, higher = more important):"]
        for rank, fi in enumerate(order):
            if importances[fi] > 1e-6:
                direction = self._infer_direction(fi)
                lines.append(
                    f"  {rank+1:2d}. {names[fi]:<25s}  {importances[fi]:.4f}  "
                    f"(net effect: {direction})"
                )

        unused = [names[fi] for fi in range(len(names)) if importances[fi] <= 1e-6]
        if unused:
            lines.append(f"\nFeatures not used in tree (zero importance): {', '.join(unused)}")

        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
InterpretableRegressor.__module__ = "interpretable_regressor"


# ---------------------------------------------------------------------------
# Evaluation (do not edit anything below this line)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()

    model_defs = [("InterpretableRegressor", InterpretableRegressor())]

    # Interpretability tests
    interp_results = run_all_interp_tests(model_defs)
    n_passed = sum(r["passed"] for r in interp_results)
    total = len(interp_results)
    std  = sum(r["passed"] for r in interp_results if r["test"] in {t.__name__ for t in ALL_TESTS})
    hard = sum(r["passed"] for r in interp_results if r["test"] in {t.__name__ for t in HARD_TESTS})
    ins  = sum(r["passed"] for r in interp_results if r["test"] in {t.__name__ for t in INSIGHT_TESTS})

    # TabArena RMSE
    dataset_rmses = evaluate_all_regressors(model_defs)
    rmse_vals = [v["InterpretableRegressor"] for v in dataset_rmses.values()
                 if not np.isnan(v.get("InterpretableRegressor", float("nan")))]
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
        "description":                        "InterpretableRegressor",
    }], RESULTS_DIR)

    print()
    print("---")
    print(f"tests_passed:  {n_passed}/{total} ({n_passed/total:.2%})  "
          f"[std {std}/8  hard {hard}/5  insight {ins}/5]")
    print(f"mean_rmse:     {mean_rmse:.4f}")
    print(f"total_seconds: {time.time() - t0:.1f}s")
