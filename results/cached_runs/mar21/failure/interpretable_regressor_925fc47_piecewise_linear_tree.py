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
from sklearn.linear_model import Ridge
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
    Piecewise Linear Tree (PLT): shallow decision tree routing with Ridge regression at leaves.

    Algorithm:
    1. Fit a shallow DT (max_leaf_nodes) to partition the feature space into regions.
    2. For each leaf region, fit a Ridge linear regression on the samples there.
    3. Predict: route sample to its leaf via the DT, then apply the leaf's linear model.

    Better RMSE than constant-leaf DT because linear models fit within-region trends.
    Same interpretability: follow the tree path (Step 1), then apply the leaf formula (Step 2).

    __str__ shows:
      - Step 1: Routing tree in if/else form (leads to a LEAF number)
      - Step 2: Linear formula for each leaf
      - Feature importances combining routing splits + leaf model coefficients
    """

    def __init__(self, max_leaf_nodes=8, min_samples_leaf=20, leaf_alpha=1.0):
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.leaf_alpha = leaf_alpha

    def fit(self, X, y):
        if hasattr(X, "columns"):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = [f"x{i}" for i in range(X.shape[1])]

        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        # Step 1: fit routing tree
        self.routing_tree_ = DecisionTreeRegressor(
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
        )
        self.routing_tree_.fit(X_arr, y_arr)

        # Step 2: get leaf assignments and fit per-leaf Ridge models
        leaf_node_ids = self.routing_tree_.apply(X_arr)
        unique_nodes = np.unique(leaf_node_ids)

        # Map sklearn internal node index → readable leaf number (1, 2, 3, ...)
        self.node_to_leaf_ = {int(n): i + 1 for i, n in enumerate(unique_nodes)}
        self.leaf_to_node_ = {v: k for k, v in self.node_to_leaf_.items()}

        self.leaf_models_: dict = {}
        for leaf_num, node_idx in self.leaf_to_node_.items():
            mask = leaf_node_ids == node_idx
            X_leaf, y_leaf = X_arr[mask], y_arr[mask]
            model = Ridge(alpha=self.leaf_alpha)
            model.fit(X_leaf, y_leaf)
            self.leaf_models_[leaf_num] = model

        return self

    def predict(self, X):
        check_is_fitted(self, "routing_tree_")
        X_arr = np.asarray(X, dtype=float)
        node_ids = self.routing_tree_.apply(X_arr)
        preds = np.empty(len(X_arr))
        for leaf_num, model in self.leaf_models_.items():
            node_idx = self.leaf_to_node_[leaf_num]
            mask = node_ids == node_idx
            if np.any(mask):
                preds[mask] = model.predict(X_arr[mask])
        return preds

    def _tree_lines(self, node=0, depth=0):
        """Recursively build routing tree string as if/else lines."""
        t = self.routing_tree_.tree_
        names = self.feature_names_in_
        indent = "    " * depth

        if t.children_left[node] == -1:  # leaf node
            leaf_num = self.node_to_leaf_[int(node)]
            return [f"{indent}→ [LEAF {leaf_num}]"]

        fname = names[int(t.feature[node])]
        thresh = t.threshold[node]
        lines = [f"{indent}if {fname} <= {thresh:.4g}:"]
        lines.extend(self._tree_lines(int(t.children_left[node]), depth + 1))
        lines.append(f"{indent}else:  # {fname} > {thresh:.4g}")
        lines.extend(self._tree_lines(int(t.children_right[node]), depth + 1))
        return lines

    def __str__(self):
        check_is_fitted(self, "routing_tree_")
        names = self.feature_names_in_

        lines = [
            f"PiecewiseLinearTree(max_leaf_nodes={self.max_leaf_nodes}, "
            f"min_samples_leaf={self.min_samples_leaf}, leaf_alpha={self.leaf_alpha})",
            "",
            "Step 1 — Follow routing tree from top to reach your LEAF number:",
        ]
        lines.extend(self._tree_lines())

        lines += [
            "",
            "Step 2 — Apply the linear formula for your LEAF number to get the prediction:",
        ]
        for leaf_num in sorted(self.leaf_models_.keys()):
            model = self.leaf_models_[leaf_num]
            intercept = float(model.intercept_)
            terms = [f"{intercept:+.4f}"]
            for c, name in zip(model.coef_, names):
                if abs(c) > 1e-3:
                    terms.append(f"({c:+.4f} * {name})")
            formula = "y = " + " ".join(terms)
            lines.append(f"  LEAF {leaf_num}: {formula}")

        # Feature importances: routing splits + aggregate leaf coefficients
        routing_imp = self.routing_tree_.feature_importances_
        leaf_coef_imp = np.zeros(len(names))
        for model in self.leaf_models_.values():
            leaf_coef_imp += np.abs(model.coef_)
        total_route = routing_imp.sum() or 1.0
        total_leaf = leaf_coef_imp.sum() or 1.0
        combined_imp = routing_imp / total_route + leaf_coef_imp / total_leaf

        order = np.argsort(combined_imp)[::-1]
        lines += ["", "Feature importances (routing importance + leaf coefficient magnitude):"]
        for rank, fi in enumerate(order):
            if combined_imp[fi] > 0.01:
                lines.append(f"  {rank+1:2d}. {names[fi]:<25s}  {combined_imp[fi]:.4f}")

        unused = [names[i] for i in range(len(names)) if combined_imp[i] <= 0.01]
        if unused:
            lines.append(f"\nFeatures with near-zero importance: {', '.join(unused)}")

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
