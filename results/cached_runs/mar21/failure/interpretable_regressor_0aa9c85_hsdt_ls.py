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
    Hierarchical Shrinkage DT with Leaf Summary (HSDT-LS):
    Standard DT with post-hoc hierarchical shrinkage + an explicit leaf lookup table
    in __str__ to facilitate point predictions and counterfactual reasoning.

    Shrinkage formula (top-down):
      shrunk[node] = orig[node] + lam * (shrunk[parent] - orig[node]) / (n_samples + lam)

    __str__ shows THREE sections:
      1. Full tree with shrunk leaf values (for path tracing)
      2. Leaf Lookup Table: each leaf listed with its path conditions and shrunk prediction
         (helps LLM compute point predictions by checking which leaf conditions are met)
      3. Feature importances with net direction and unused features
    """

    def __init__(self, max_leaf_nodes=35, min_samples_leaf=5, shrinkage_lambda=20.0):
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

        self.tree_ = DecisionTreeRegressor(
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
        )
        self.tree_.fit(X_arr, y_arr)
        self.shrunk_values_ = self._compute_shrinkage()

        # Build leaf-to-path mapping for the Leaf Lookup Table
        self.leaf_paths_ = self._extract_leaf_paths()

        return self

    def _compute_shrinkage(self):
        t = self.tree_.tree_
        orig = t.value[:, 0, 0].copy()
        shrunk = np.copy(orig)
        lam = self.shrinkage_lambda

        def shrink(node, parent_shrunk):
            if node == 0:
                shrunk[node] = orig[node]
            else:
                n_s = t.n_node_samples[node]
                shrunk[node] = orig[node] + lam * (parent_shrunk - orig[node]) / (n_s + lam)
            left = t.children_left[node]
            if left != -1:
                shrink(int(left), shrunk[node])
                shrink(int(t.children_right[node]), shrunk[node])

        shrink(0, orig[0])
        return shrunk

    def _extract_leaf_paths(self):
        """Return dict: node_idx → list of (feature_name, direction, threshold) conditions."""
        t = self.tree_.tree_
        names = self.feature_names_in_
        paths = {}

        def dfs(node, conditions):
            if t.children_left[node] == -1:  # leaf
                paths[node] = list(conditions)
                return
            fname = names[int(t.feature[node])]
            thresh = t.threshold[node]
            dfs(int(t.children_left[node]), conditions + [(fname, "<=", thresh)])
            dfs(int(t.children_right[node]), conditions + [(fname, ">", thresh)])

        dfs(0, [])
        return paths

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

        # ---- Section 1: Full tree ----
        lines = [
            f"HierarchicalShrinkageDT_LS(max_leaf_nodes={self.max_leaf_nodes}, "
            f"shrinkage_lambda={self.shrinkage_lambda})",
            f"  nodes={t.node_count}, leaves={self.tree_.get_n_leaves()}",
            "",
            "Section 1 — Tree structure (follow path from root; leaf values are predictions):",
        ]
        lines.extend(self._tree_lines())

        # ---- Section 2: Leaf Lookup Table ----
        lines += [
            "",
            "Section 2 — Leaf Lookup Table (find the leaf whose conditions match your sample):",
            "  (prediction = value at matching leaf; all conditions for a leaf must be satisfied)",
        ]
        # Sort leaves by prediction value for easy scanning
        leaf_nodes = [(node, self.shrunk_values_[node], conds)
                      for node, conds in self.leaf_paths_.items()]
        leaf_nodes.sort(key=lambda x: x[1])  # sort by prediction
        for i, (node, val, conds) in enumerate(leaf_nodes):
            n_s = t.n_node_samples[node]
            cond_str = " AND ".join(
                f"{fname} {op} {thresh:.4g}" for fname, op, thresh in conds
            ) if conds else "always (root leaf)"
            lines.append(f"  LEAF {i+1:2d}: pred={val:.4f} (n={n_s}) | conditions: {cond_str}")

        # ---- Section 3: Feature importances ----
        order = np.argsort(importances)[::-1]
        lines += ["", "Section 3 — Feature importances (Gini-based, higher = more important):"]
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
