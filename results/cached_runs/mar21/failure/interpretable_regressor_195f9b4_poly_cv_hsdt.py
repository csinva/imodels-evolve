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
    Polynomial Feature CV-HSDT (PolyCV-HSDT):
    Augments the feature set with squared terms before fitting a CV-tuned HSDT.

    Feature augmentation: for each feature xi, add xi^2 (squared).
    This allows the DT to capture quadratic/nonlinear patterns in fewer splits,
    improving RMSE on datasets with nonlinear relationships.

    Example: for features [x0, x1], the augmented set is [x0, x1, x0^2, x1^2].
    The DT can then split on x0^2 (e.g., x0^2 <= 4 means -2 <= x0 <= 2),
    which an axis-aligned DT on original features would need many splits to capture.

    Lambda is auto-selected via 5-fold CV from a wide grid.
    max_leaf_nodes is fixed at 25 (known optimal for interpretability).

    __str__ shows:
      - Tree with augmented feature names (xi_sq for xi^2) — LLM can compute xi_sq = xi*xi
      - Feature importances (for both original and augmented features)
      - Net direction per feature
      - Unused original features
    """

    LAMBDA_GRID = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0]

    def __init__(self, max_leaf_nodes=25, min_samples_leaf=5, shrinkage_lambda="cv", cv=5):
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.shrinkage_lambda = shrinkage_lambda
        self.cv = cv

    def _augment(self, X_arr, names):
        """Add squared features. Returns (X_aug, augmented_names)."""
        X_sq = X_arr ** 2
        X_aug = np.hstack([X_arr, X_sq])
        aug_names = list(names) + [f"{n}_sq" for n in names]
        return X_aug, aug_names

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

    def _select_lambda(self, X_aug, y_arr):
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        best_lam, best_mse = 15.0, np.inf
        for lam in self.LAMBDA_GRID:
            fold_mses = []
            for tr_idx, va_idx in kf.split(X_aug):
                X_tr, X_va = X_aug[tr_idx], X_aug[va_idx]
                y_tr, y_va = y_arr[tr_idx], y_arr[va_idx]
                tree = DecisionTreeRegressor(
                    max_leaf_nodes=self.max_leaf_nodes,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=42,
                )
                tree.fit(X_tr, y_tr)
                sv = self._compute_shrinkage(tree, lam)
                fold_mses.append(np.mean((y_va - sv[tree.apply(X_va)]) ** 2))
            mse = np.mean(fold_mses)
            if mse < best_mse:
                best_mse, best_lam = mse, lam
        return best_lam

    def fit(self, X, y):
        orig_names = list(X.columns) if hasattr(X, "columns") else [f"x{i}" for i in range(X.shape[1])]
        self.feature_names_in_ = orig_names

        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        X_aug, self.aug_names_ = self._augment(X_arr, orig_names)

        if self.shrinkage_lambda == "cv":
            self.lambda_ = self._select_lambda(X_aug, y_arr)
        else:
            self.lambda_ = float(self.shrinkage_lambda)

        self.tree_ = DecisionTreeRegressor(
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
        )
        self.tree_.fit(X_aug, y_arr)
        self.shrunk_values_ = self._compute_shrinkage(self.tree_, self.lambda_)

        return self

    def predict(self, X):
        check_is_fitted(self, "tree_")
        X_arr = np.asarray(X, dtype=float)
        X_aug, _ = self._augment(X_arr, self.feature_names_in_)
        return self.shrunk_values_[self.tree_.apply(X_aug)]

    def _tree_lines(self, node=0, depth=0):
        t = self.tree_.tree_
        names = self.aug_names_
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
            return "positive"
        elif weighted_effect < 0:
            return "negative"
        return "mixed"

    def __str__(self):
        check_is_fitted(self, "tree_")
        n_orig = len(self.feature_names_in_)
        importances = self.tree_.feature_importances_

        lines = [
            f"PolyCV_HSDT(max_leaf_nodes={self.max_leaf_nodes}, lambda={self.lambda_:.1f})",
            f"  nodes={self.tree_.tree_.node_count}, leaves={self.tree_.get_n_leaves()}",
            f"  Note: features xi_sq = xi * xi (squared original features)",
            "",
            "Tree structure (follow from root; leaf values are shrunk predictions):",
        ]
        lines.extend(self._tree_lines())

        order = np.argsort(importances)[::-1]
        lines += ["", "Feature importances (Gini-based, including squared features):"]
        for rank, fi in enumerate(order):
            if importances[fi] > 1e-6:
                fname = self.aug_names_[fi]
                direction = self._infer_direction(fi)
                lines.append(
                    f"  {rank+1:2d}. {fname:<25s}  {importances[fi]:.4f}  (net: {direction})"
                )

        # Report unused original features (zero importance for BOTH xi and xi_sq)
        unused_orig = []
        for i, name in enumerate(self.feature_names_in_):
            sq_idx = n_orig + i
            if importances[i] <= 1e-6 and (sq_idx >= len(importances) or importances[sq_idx] <= 1e-6):
                unused_orig.append(name)
        if unused_orig:
            lines.append(f"\nOriginal features not used at all: {', '.join(unused_orig)}")

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
