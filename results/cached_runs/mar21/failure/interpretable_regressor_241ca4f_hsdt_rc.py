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
    Hierarchical Shrinkage DT with Residual Correction (HSDT-RC):
    Two-stage model combining hierarchical shrinkage on a larger primary tree
    with a compact secondary tree trained on the residuals.

    Stage 1 (main model):
      - Fit a large DT (max_leaf_nodes=40) with hierarchical shrinkage (lam=20)
      - This provides the primary interpretation

    Stage 2 (residual correction):
      - Fit a compact DT (max_leaf_nodes=8) with hierarchical shrinkage on residuals
      - Adds a small correction to the Stage 1 prediction

    Prediction = stage1_predict(X) + correction_lr * stage2_predict(X)

    __str__ is dominated by Stage 1 (the main interpretable tree).
    Stage 2 is shown as a compact additive correction term.
    Feature importances combine both stages.
    """

    def __init__(
        self,
        max_leaf_nodes_1=40,
        max_leaf_nodes_2=8,
        min_samples_leaf=5,
        shrinkage_lambda=20.0,
        correction_lr=0.5,
    ):
        self.max_leaf_nodes_1 = max_leaf_nodes_1
        self.max_leaf_nodes_2 = max_leaf_nodes_2
        self.min_samples_leaf = min_samples_leaf
        self.shrinkage_lambda = shrinkage_lambda
        self.correction_lr = correction_lr

    def _fit_hsdt(self, X_arr, y_arr, max_leaf_nodes):
        """Fit a single DT with hierarchical shrinkage. Returns (tree, shrunk_values)."""
        tree = DecisionTreeRegressor(
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
        )
        tree.fit(X_arr, y_arr)
        shrunk = self._compute_shrinkage(tree, self.shrinkage_lambda)
        return tree, shrunk

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

    def fit(self, X, y):
        if hasattr(X, "columns"):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = [f"x{i}" for i in range(X.shape[1])]

        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        # Stage 1: main HSDT
        self.tree1_, self.shrunk1_ = self._fit_hsdt(X_arr, y_arr, self.max_leaf_nodes_1)

        # Stage 2: residual correction HSDT
        residuals = y_arr - self.shrunk1_[self.tree1_.apply(X_arr)]
        self.tree2_, self.shrunk2_ = self._fit_hsdt(X_arr, residuals, self.max_leaf_nodes_2)

        return self

    def predict(self, X):
        check_is_fitted(self, "tree1_")
        X_arr = np.asarray(X, dtype=float)
        pred1 = self.shrunk1_[self.tree1_.apply(X_arr)]
        pred2 = self.shrunk2_[self.tree2_.apply(X_arr)]
        return pred1 + self.correction_lr * pred2

    def _tree_lines(self, tree, shrunk, depth=0, node=0):
        """Recursively build tree text with shrunk leaf values."""
        t = tree.tree_
        names = self.feature_names_in_
        indent = "|   " * depth

        if t.children_left[node] == -1:
            val = shrunk[node]
            n_s = t.n_node_samples[node]
            return [f"{indent}|--- value: {val:.4f}  (n={n_s})"]

        fname = names[int(t.feature[node])]
        thresh = t.threshold[node]
        lines = [f"{indent}|--- {fname} <= {thresh:.4g}"]
        lines.extend(self._tree_lines(tree, shrunk, depth + 1, int(t.children_left[node])))
        lines.append(f"{indent}|--- {fname} > {thresh:.4g}")
        lines.extend(self._tree_lines(tree, shrunk, depth + 1, int(t.children_right[node])))
        return lines

    def _infer_direction(self, tree, shrunk, feature_idx):
        t = tree.tree_
        weighted_effect = 0.0
        for node in range(t.node_count):
            if t.children_left[node] == -1:
                continue
            if t.feature[node] != feature_idx:
                continue
            lc = int(t.children_left[node])
            rc = int(t.children_right[node])
            weighted_effect += (shrunk[rc] - shrunk[lc]) * (
                t.n_node_samples[lc] + t.n_node_samples[rc]
            )
        if weighted_effect > 0:
            return "positive (higher → higher prediction)"
        elif weighted_effect < 0:
            return "negative (higher → lower prediction)"
        return "mixed"

    def __str__(self):
        check_is_fitted(self, "tree1_")
        names = self.feature_names_in_
        n_leaves = self.tree1_.get_n_leaves()
        n_nodes = self.tree1_.tree_.node_count
        imp1 = self.tree1_.feature_importances_
        imp2 = self.tree2_.feature_importances_
        combined_imp = imp1 + self.correction_lr * imp2

        lines = [
            f"HSDT_RC(max_leaf_nodes_1={self.max_leaf_nodes_1}, "
            f"max_leaf_nodes_2={self.max_leaf_nodes_2}, "
            f"shrinkage_lambda={self.shrinkage_lambda}, "
            f"correction_lr={self.correction_lr})",
            f"  Stage-1 tree: nodes={n_nodes}, leaves={n_leaves}",
            f"  Stage-2 correction tree: leaves={self.tree2_.get_n_leaves()}",
            "",
            "Stage 1 — Main tree (shrunk leaf values are the primary predictions):",
        ]
        lines.extend(self._tree_lines(self.tree1_, self.shrunk1_))

        lines += [
            "",
            f"Stage 2 — Residual correction (multiplied by {self.correction_lr}, then added to Stage 1):",
        ]
        lines.extend(self._tree_lines(self.tree2_, self.shrunk2_))

        # Feature importances
        order = np.argsort(combined_imp)[::-1]
        lines += ["", "Feature importances (combined Stage-1 + Stage-2, higher = more important):"]
        for rank, fi in enumerate(order):
            if combined_imp[fi] > 1e-6:
                direction = self._infer_direction(self.tree1_, self.shrunk1_, fi)
                lines.append(
                    f"  {rank+1:2d}. {names[fi]:<25s}  {combined_imp[fi]:.4f}  "
                    f"(net effect: {direction})"
                )

        unused = [names[i] for i in range(len(names)) if combined_imp[i] <= 1e-6]
        if unused:
            lines.append(f"\nFeatures not used (zero importance): {', '.join(unused)}")

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
