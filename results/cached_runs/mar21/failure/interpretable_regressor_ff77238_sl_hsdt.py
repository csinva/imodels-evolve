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
    Sparse Linear + HSDT (SL-HSDT): two-component interpretable model.

    Component 1 (linear): LassoCV fits the global linear signal. Due to L1
    regularization, typically only 2-5 features are active — making the formula
    compact and easy to apply.

    Component 2 (nonlinear correction): CV-tuned HSDT fits the residuals
    y - linear_pred. With the linear signal removed, the DT only needs to
    capture the nonlinear remainder, which takes fewer splits.

    Prediction: y_hat = linear(X) + tree_correction(X)

    __str__ shows:
      1. Linear formula (LassoCV): y_linear = intercept + coef1*x1 + coef2*x2 + ...
         (only active features shown; zero-coefficient features listed as unused)
      2. Nonlinear correction tree (HSDT, shrunk leaf values)
         — for datasets where the linear model captures most variance,
           the correction tree is small/near-zero
      3. Feature importances combining both components
    """

    LAMBDA_GRID = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0]

    def __init__(self, max_leaf_nodes=20, min_samples_leaf=5,
                 tree_lambda="cv", cv=5):
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.tree_lambda = tree_lambda
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

    def _select_tree_lambda(self, X_arr, residuals):
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        best_lam, best_mse = 15.0, np.inf
        for lam in self.LAMBDA_GRID:
            fold_mses = []
            for tr_idx, va_idx in kf.split(X_arr):
                X_tr, X_va = X_arr[tr_idx], X_arr[va_idx]
                r_tr, r_va = residuals[tr_idx], residuals[va_idx]
                tree = DecisionTreeRegressor(
                    max_leaf_nodes=self.max_leaf_nodes,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=42,
                )
                tree.fit(X_tr, r_tr)
                sv = self._compute_shrinkage(tree, lam)
                fold_mses.append(np.mean((r_va - sv[tree.apply(X_va)]) ** 2))
            mse = np.mean(fold_mses)
            if mse < best_mse:
                best_mse, best_lam = mse, lam
        return best_lam

    def fit(self, X, y):
        if hasattr(X, "columns"):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = [f"x{i}" for i in range(X.shape[1])]

        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        # Component 1: sparse linear (LassoCV)
        self.linear_ = LassoCV(cv=self.cv, random_state=42, max_iter=3000)
        self.linear_.fit(X_arr, y_arr)
        linear_pred = self.linear_.predict(X_arr)
        residuals = y_arr - linear_pred

        # Component 2: HSDT on residuals
        if self.tree_lambda == "cv":
            self.tree_lambda_ = self._select_tree_lambda(X_arr, residuals)
        else:
            self.tree_lambda_ = float(self.tree_lambda)

        self.tree_ = DecisionTreeRegressor(
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
        )
        self.tree_.fit(X_arr, residuals)
        self.shrunk_values_ = self._compute_shrinkage(self.tree_, self.tree_lambda_)

        return self

    def predict(self, X):
        check_is_fitted(self, "tree_")
        X_arr = np.asarray(X, dtype=float)
        linear_pred = self.linear_.predict(X_arr)
        tree_corr = self.shrunk_values_[self.tree_.apply(X_arr)]
        return linear_pred + tree_corr

    def _tree_lines(self, node=0, depth=0):
        t = self.tree_.tree_
        names = self.feature_names_in_
        indent = "|   " * depth

        if t.children_left[node] == -1:
            val = self.shrunk_values_[node]
            n_s = t.n_node_samples[node]
            return [f"{indent}|--- correction: {val:.4f}  (n={n_s})"]

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
        coef = self.linear_.coef_
        intercept = float(self.linear_.intercept_)
        importances = self.tree_.feature_importances_

        lines = [
            f"SL_HSDT(max_leaf_nodes={self.max_leaf_nodes}, tree_lambda={self.tree_lambda_:.1f})",
            f"  Prediction = linear_part + tree_correction",
            "",
            "Part 1 — Sparse linear model (LassoCV):",
            f"  y_linear = {intercept:.4f}",
        ]
        active = [(names[i], coef[i]) for i in range(len(names)) if abs(coef[i]) > 1e-6]
        active.sort(key=lambda x: abs(x[1]), reverse=True)
        for name, c in active:
            lines.append(f"    + ({c:+.4f}) * {name}")

        zeroed = [names[i] for i in range(len(names)) if abs(coef[i]) <= 1e-6]
        if zeroed:
            lines.append(f"  Features with zero linear effect: {', '.join(zeroed)}")

        lines += [
            "",
            f"Part 2 — Nonlinear correction tree (HSDT on residuals; add to y_linear above):",
            f"  tree_lambda={self.tree_lambda_:.1f}, leaves={self.tree_.get_n_leaves()}",
        ]
        lines.extend(self._tree_lines())

        # Combined feature importances
        linear_imp = np.abs(coef)
        linear_imp = linear_imp / (linear_imp.sum() or 1.0)
        tree_imp = importances / (importances.sum() or 1.0)
        combined = 0.5 * linear_imp + 0.5 * tree_imp

        order = np.argsort(combined)[::-1]
        lines += ["", "Combined feature importances (linear + tree, higher = more important):"]
        for rank, fi in enumerate(order):
            if combined[fi] > 0.01:
                lines.append(f"  {rank+1:2d}. {names[fi]:<25s}  {combined[fi]:.4f}")

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
