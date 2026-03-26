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
    Joint CV-tuned Hierarchical Shrinkage DT (JCV-HSDT):
    Jointly selects both max_leaf_nodes and shrinkage_lambda via 5-fold CV,
    choosing from a 2D grid to find the best (regularized_depth, regularization) combo.

    Candidate max_leaf_nodes: [15, 20, 25, 30]
    Candidate lambda:         [1, 3, 7, 15, 30, 60, 120, 250]

    The joint CV prevents underfitting (too few leaves) and overfitting (lambda too low).
    The tree structure and __str__ format are identical to HSDT, so interpretability
    is preserved while RMSE improves from optimal joint hyperparameter selection.

    __str__ shows the CV-selected parameters explicitly (lambda and max_leaf_nodes).
    """

    LEAF_GRID = [15, 20, 25, 30]
    LAMBDA_GRID = [1.0, 3.0, 7.0, 15.0, 30.0, 60.0, 120.0, 250.0]

    def __init__(self, max_leaf_nodes="cv", shrinkage_lambda="cv",
                 min_samples_leaf=5, cv=5):
        self.max_leaf_nodes = max_leaf_nodes
        self.shrinkage_lambda = shrinkage_lambda
        self.min_samples_leaf = min_samples_leaf
        self.cv = cv

    @staticmethod
    def _compute_shrinkage(tree, lam):
        t = tree.tree_
        orig = t.value[:, 0, 0].copy()
        shrunk = np.copy(orig)

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

    def _cv_select(self, X_arr, y_arr):
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        splits = list(kf.split(X_arr))
        best_leaves, best_lam, best_mse = 25, 15.0, np.inf

        leaf_candidates = self.LEAF_GRID if self.max_leaf_nodes == "cv" else [int(self.max_leaf_nodes)]
        lam_candidates = self.LAMBDA_GRID if self.shrinkage_lambda == "cv" else [float(self.shrinkage_lambda)]

        for n_leaves in leaf_candidates:
            for lam in lam_candidates:
                fold_mses = []
                for train_idx, val_idx in splits:
                    X_tr, X_va = X_arr[train_idx], X_arr[val_idx]
                    y_tr, y_va = y_arr[train_idx], y_arr[val_idx]
                    tree = DecisionTreeRegressor(
                        max_leaf_nodes=n_leaves,
                        min_samples_leaf=self.min_samples_leaf,
                        random_state=42,
                    )
                    tree.fit(X_tr, y_tr)
                    sv = self._compute_shrinkage(tree, lam)
                    preds = sv[tree.apply(X_va)]
                    fold_mses.append(np.mean((y_va - preds) ** 2))
                mse = np.mean(fold_mses)
                if mse < best_mse:
                    best_mse, best_leaves, best_lam = mse, n_leaves, lam

        return best_leaves, best_lam

    def fit(self, X, y):
        if hasattr(X, "columns"):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = [f"x{i}" for i in range(X.shape[1])]

        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        self.n_leaves_, self.lambda_ = self._cv_select(X_arr, y_arr)

        self.tree_ = DecisionTreeRegressor(
            max_leaf_nodes=self.n_leaves_,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
        )
        self.tree_.fit(X_arr, y_arr)
        self.shrunk_values_ = self._compute_shrinkage(self.tree_, self.lambda_)

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

        lines = [
            f"JCV_HSDT(cv_selected: max_leaf_nodes={self.n_leaves_}, lambda={self.lambda_:.1f})",
            f"  nodes={t.node_count}, leaves={self.tree_.get_n_leaves()}",
            "",
            "Tree structure (follow from root; leaf values are shrunk predictions):",
        ]
        lines.extend(self._tree_lines())

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
