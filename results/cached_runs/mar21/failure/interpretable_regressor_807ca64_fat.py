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
    FIGS-style Additive Tree Regressor (FAT): gradient-boosted shallow trees
    represented in the FIGS output format.

    Fits n_trees depth-2 decision trees on residuals, with learning_rate scaling.
    A constant "Tree 0" holds the mean prediction (bias). The remaining trees
    each contribute a small additive correction.

    The __str__ format closely mirrors imodels.FIGSRegressor:
      - Header: "Predictions are made by summing the Val reached in each tree."
      - Tree 0: constant (bias = mean y)
      - Tree k: shown with FIGS-style indentation, split labels, "Val: X (leaf)"
      - Trees separated by "  +"

    Feature importances and net directions are also shown for quick lookups.
    """

    def __init__(self, n_trees=5, max_depth=2, learning_rate=0.4, min_samples_leaf=10):
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

        self.bias_ = float(np.mean(y_arr))
        residual = y_arr - self.bias_

        self.trees_ = []
        # Precomputed scaled leaf values for each tree (for clean __str__ display)
        self.scaled_vals_: list = []

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
            # Store scaled leaf values (actual contribution per leaf)
            self.scaled_vals_.append(tree.tree_.value[:, 0, 0] * self.learning_rate)

        return self

    def predict(self, X):
        check_is_fitted(self, "trees_")
        X_arr = np.asarray(X, dtype=float)
        pred = np.full(len(X_arr), self.bias_)
        for tree, sv in zip(self.trees_, self.scaled_vals_):
            leaf_ids = tree.apply(X_arr)
            pred += sv[leaf_ids]
        return pred

    def _figs_tree_lines(self, tree, sv, depth=0, node=0):
        """Recursively build FIGS-style tree lines."""
        t = tree.tree_
        names = self.feature_names_in_
        indent = "\t" * depth

        if t.children_left[node] == -1:  # leaf
            val = sv[node]
            return [f"{indent}Val: {val:.4f} (leaf)"]

        fname = names[int(t.feature[node])]
        thresh = t.threshold[node]
        lines = [f"{indent}{fname} <= {thresh:.4g} (split)"]
        lines.extend(self._figs_tree_lines(tree, sv, depth + 1, int(t.children_left[node])))
        lines.extend(self._figs_tree_lines(tree, sv, depth + 1, int(t.children_right[node])))
        return lines

    def __str__(self):
        check_is_fitted(self, "trees_")
        names = self.feature_names_in_

        lines = [
            "FIGS-style Additive Tree Regressor:",
            "\tPredictions are made by summing the Val reached by traversing each tree.",
            "\tTree 0 is a constant (mean prediction). Each subsequent tree adds a correction.",
            "",
            f"Tree 0 (constant, no splits):",
            f"\tVal: {self.bias_:.4f} (bias — mean prediction)",
        ]

        for i, (tree, sv) in enumerate(zip(self.trees_, self.scaled_vals_)):
            lines.append("")
            lines.append("\t+")
            lines.append(f"Tree {i+1} (root):")
            lines.extend(self._figs_tree_lines(tree, sv))

        # Feature importances
        total_imp = np.zeros(len(names))
        net_dir = np.zeros(len(names))

        for tree, sv in zip(self.trees_, self.scaled_vals_):
            t = tree.tree_
            for node in range(t.node_count):
                if t.children_left[node] == -1:
                    continue
                fi = int(t.feature[node])
                lc, rc = int(t.children_left[node]), int(t.children_right[node])
                gap = abs(sv[rc] - sv[lc])
                n_both = t.n_node_samples[lc] + t.n_node_samples[rc]
                total_imp[fi] += gap * n_both
                net_dir[fi] += (sv[rc] - sv[lc]) * n_both

        norm = total_imp.sum() or 1.0
        order = np.argsort(total_imp)[::-1]
        lines += ["", "Feature importances (higher = more important):"]
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
            lines.append(f"\nFeatures not used in any tree: {', '.join(unused)}")

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
