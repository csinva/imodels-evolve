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
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "eval"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this — everything below is fair game)
# ---------------------------------------------------------------------------


class InterpretableRegressor(BaseEstimator, RegressorMixin):
    """
    Hierarchical Shrinkage Decision Tree (HSDT): trains a DT then post-hoc smooths
    leaf values toward ancestor values using hierarchical shrinkage regularization.

    Shrinkage formula (applied top-down, node by node):
      shrunk_value[node] = orig_value[node]
                         + lam * (shrunk_value[parent] - orig_value[node])
                           / (n_node_samples[node] + lam)

    The root keeps its original value. Each child is shrunk toward its parent's
    shrunk value. High-variance (small-sample) leaves shrink more; large leaves
    barely move. The result is a tree with lower variance leaf values → better RMSE.

    The tree STRUCTURE (splits, feature choices) is unchanged, so __str__ uses the
    same export_text format as a regular DT — making interpretability tests easy:
    the LLM just follows the path, reads the shrunk value at the leaf.

    __str__ shows:
      - Full tree with shrunk leaf values
      - Feature importances (Gini-based, from the routing tree)
      - Net effect direction per feature
      - Unused features
    """

    def __init__(self, max_leaf_nodes=25, min_samples_leaf=5, shrinkage_lambda=15.0):
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.shrinkage_lambda = shrinkage_lambda

    def fit(self, X, y):
        if hasattr(X, "columns"):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = [f"x{i}" for i in range(X.shape[1])]

        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        # Step 1: fit standard DT
        self.tree_ = DecisionTreeRegressor(
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
        )
        self.tree_.fit(X_arr, y_arr)

        # Step 2: compute shrunk node values via top-down DFS
        self.shrunk_values_ = self._compute_shrinkage()

        return self

    def _compute_shrinkage(self):
        """Compute hierarchically shrunk values for all tree nodes."""
        t = self.tree_.tree_
        orig = t.value[:, 0, 0].copy()  # shape (n_nodes,)
        shrunk = np.copy(orig)
        lam = self.shrinkage_lambda

        def shrink(node, parent_shrunk):
            if node == 0:  # root keeps original value
                shrunk[node] = orig[node]
            else:
                n_s = t.n_node_samples[node]
                shrunk[node] = orig[node] + lam * (parent_shrunk - orig[node]) / (n_s + lam)
            left = t.children_left[node]
            if left != -1:
                shrink(left, shrunk[node])
                shrink(t.children_right[node], shrunk[node])

        shrink(0, orig[0])
        return shrunk

    def predict(self, X):
        check_is_fitted(self, "tree_")
        X_arr = np.asarray(X, dtype=float)
        leaf_node_ids = self.tree_.apply(X_arr)
        return self.shrunk_values_[leaf_node_ids]

    def _tree_lines(self, node=0, depth=0):
        """Recursively build tree text with shrunk leaf values."""
        t = self.tree_.tree_
        names = self.feature_names_in_
        indent = "|   " * depth

        if t.children_left[node] == -1:  # leaf
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

    def __str__(self):
        check_is_fitted(self, "tree_")
        names = self.feature_names_in_
        n_leaves = self.tree_.get_n_leaves()
        n_nodes = self.tree_.tree_.node_count
        importances = self.tree_.feature_importances_

        lines = [
            f"HierarchicalShrinkageDT(max_leaf_nodes={self.max_leaf_nodes}, "
            f"min_samples_leaf={self.min_samples_leaf}, "
            f"shrinkage_lambda={self.shrinkage_lambda})",
            f"  nodes={n_nodes}, leaves={n_leaves}",
            "",
            "Tree structure (follow from root; leaf values are shrunk predictions):",
        ]
        lines.extend(self._tree_lines())

        # Feature importances
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

    def _infer_direction(self, feature_idx):
        """Infer net effect direction for a feature using shrunk values."""
        t = self.tree_.tree_
        n_nodes = t.node_count
        weighted_effect = 0.0
        for node in range(n_nodes):
            if t.children_left[node] == -1:
                continue
            if t.feature[node] != feature_idx:
                continue
            left_child = int(t.children_left[node])
            right_child = int(t.children_right[node])
            left_val = self.shrunk_values_[left_child]
            right_val = self.shrunk_values_[right_child]
            n_left = t.n_node_samples[left_child]
            n_right = t.n_node_samples[right_child]
            weighted_effect += (right_val - left_val) * (n_left + n_right)
        if weighted_effect > 0:
            return "positive (higher → higher prediction)"
        elif weighted_effect < 0:
            return "negative (higher → lower prediction)"
        else:
            return "mixed"


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
