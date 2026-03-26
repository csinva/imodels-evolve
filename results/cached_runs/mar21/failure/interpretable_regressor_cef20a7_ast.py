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
    Additive Shallow Trees (AST): gradient-boosted ensemble of small depth-2 trees.

    Fits n_trees depth-2 decision trees sequentially on residuals.
    Prediction = base_value + sum_i(learning_rate * tree_i.predict(X))

    Each tree has at most 4 leaves (depth=2) — small enough for the LLM to trace exactly.
    n_trees is kept small (default 7) so the LLM can reason about all contributions.

    __str__ shows:
      - base_value (mean)
      - Each tree explicitly with if/else structure and leaf contributions
        (each shown as a scaled prediction contribution, not raw value)
      - Feature importances aggregated across all trees
      - Net direction of effect per feature
      - Unused features
    """

    def __init__(self, n_trees=7, max_depth=2, learning_rate=0.4, min_samples_leaf=10):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        if hasattr(X, "columns"):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = [f"x{i}" for i in range(X.shape[1])]

        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        self.base_value_ = float(np.mean(y_arr))
        residual = y_arr - self.base_value_

        self.trees_ = []
        for _ in range(self.n_trees):
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=42,
            )
            tree.fit(X_arr, residual)
            pred = tree.predict(X_arr)
            residual -= self.learning_rate * pred
            self.trees_.append(tree)

        return self

    def predict(self, X):
        check_is_fitted(self, "trees_")
        X_arr = np.asarray(X, dtype=float)
        pred = np.full(len(X_arr), self.base_value_)
        for tree in self.trees_:
            pred += self.learning_rate * tree.predict(X_arr)
        return pred

    def _tree_contribution_lines(self, tree, depth=0, node=0):
        """Recursively build a compact tree string showing scaled contributions."""
        t = tree.tree_
        names = self.feature_names_in_
        indent = "    " * depth

        if t.children_left[node] == -1:  # leaf
            val = float(t.value[node][0][0]) * self.learning_rate
            n_s = t.n_node_samples[node]
            return [f"{indent}contribution = {val:+.4f}  (n={n_s})"]

        fname = names[int(t.feature[node])]
        thresh = t.threshold[node]
        lines = [f"{indent}if {fname} <= {thresh:.4g}:"]
        lines.extend(self._tree_contribution_lines(tree, depth + 1, int(t.children_left[node])))
        lines.append(f"{indent}else:  # {fname} > {thresh:.4g}")
        lines.extend(self._tree_contribution_lines(tree, depth + 1, int(t.children_right[node])))
        return lines

    def _tree_feat_importance(self, tree):
        """Per-feature sum of |contribution changes| across all splits in this tree."""
        t = tree.tree_
        imp = np.zeros(len(self.feature_names_in_))
        for node in range(t.node_count):
            if t.children_left[node] == -1:
                continue
            fi = int(t.feature[node])
            lc, rc = int(t.children_left[node]), int(t.children_right[node])
            lv = float(t.value[lc][0][0]) * self.learning_rate
            rv = float(t.value[rc][0][0]) * self.learning_rate
            imp[fi] += abs(rv - lv) * (t.n_node_samples[lc] + t.n_node_samples[rc])
        return imp

    def __str__(self):
        check_is_fitted(self, "trees_")
        names = self.feature_names_in_

        lines = [
            f"AdditiveShallowTrees(n_trees={self.n_trees}, max_depth={self.max_depth}, "
            f"learning_rate={self.learning_rate}, min_samples_leaf={self.min_samples_leaf})",
            f"  base_value = {self.base_value_:.4f}",
            f"  Prediction = base_value + contribution from Tree 1 + contribution from Tree 2 + ...",
            "",
        ]

        # Show each tree
        total_imp = np.zeros(len(names))
        for i, tree in enumerate(self.trees_):
            n_leaves = tree.get_n_leaves()
            lines.append(f"Tree {i+1} (depth={self.max_depth}, leaves={n_leaves}):")
            lines.extend(self._tree_contribution_lines(tree))
            lines.append("")
            total_imp += self._tree_feat_importance(tree)

        # Feature importances
        order = np.argsort(total_imp)[::-1]
        norm = total_imp.sum() or 1.0

        # Net direction: weighted sum of (right - left) contributions
        net_dir = np.zeros(len(names))
        for tree in self.trees_:
            t = tree.tree_
            for node in range(t.node_count):
                if t.children_left[node] == -1:
                    continue
                fi = int(t.feature[node])
                lc, rc = int(t.children_left[node]), int(t.children_right[node])
                lv = float(t.value[lc][0][0]) * self.learning_rate
                rv = float(t.value[rc][0][0]) * self.learning_rate
                n_both = t.n_node_samples[lc] + t.n_node_samples[rc]
                net_dir[fi] += (rv - lv) * n_both

        lines.append("Feature importances (sum of |contribution gap| across all trees × sample count):")
        for rank, fi in enumerate(order):
            if total_imp[fi] > 1e-9:
                if net_dir[fi] > 0:
                    direction = "positive (higher → higher prediction)"
                elif net_dir[fi] < 0:
                    direction = "negative (higher → lower prediction)"
                else:
                    direction = "mixed"
                lines.append(
                    f"  {rank+1:2d}. {names[fi]:<25s}  {total_imp[fi]/norm:.4f}  "
                    f"(net effect: {direction})"
                )

        used = {names[fi] for fi in range(len(names)) if total_imp[fi] > 1e-9}
        unused = [f for f in names if f not in used]
        if unused:
            lines.append(f"\nFeatures not used in any tree (zero importance): {', '.join(unused)}")

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
