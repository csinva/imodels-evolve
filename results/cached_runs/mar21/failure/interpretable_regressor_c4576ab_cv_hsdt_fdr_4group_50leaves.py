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
    CV-HSDT with 4-Group Hierarchical Decision Rules (CV-HSDT-FDR-4Group):
    50-leaf tree + HSDT shrinkage, with decision rules split into 4 groups by
    the root AND secondary split conditions (~12-13 rules per group).

    Three-step lookup makes the LLM's job tractable even with 50 leaves:
      1. Check root condition → left (L) or right (R) branch
      2. Check secondary condition within that branch → LL/LR or RL/RR group
      3. Scan only ~12 rules in that group to find the matching prediction

    Shrinkage formula (top-down):
      shrunk[node] = orig[node] + lam * (shrunk[parent] - orig[node]) / (n_samples + lam)

    Lambda grid: [1, 3, 7, 15, 30, 60]. 50 leaves for improved RMSE.
    repr_v=17 to bust joblib cache.
    """

    LAMBDA_GRID = [1.0, 3.0, 7.0, 15.0, 30.0, 60.0]

    def __init__(self, max_leaf_nodes=50, min_samples_leaf=5, shrinkage_lambda="cv", cv=5,
                 repr_v=17):
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.shrinkage_lambda = shrinkage_lambda
        self.cv = cv
        self.repr_v = repr_v  # version tag — increment to bust joblib cache on __str__ changes

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

    def _select_lambda(self, X_arr, y_arr):
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        best_lam, best_mse = None, np.inf
        for lam in self.LAMBDA_GRID:
            fold_mses = []
            for tr_idx, va_idx in kf.split(X_arr):
                X_tr, X_va = X_arr[tr_idx], X_arr[va_idx]
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
        if hasattr(X, "columns"):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = [f"x{i}" for i in range(X.shape[1])]

        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        if self.shrinkage_lambda == "cv":
            self.lambda_ = self._select_lambda(X_arr, y_arr)
        else:
            self.lambda_ = float(self.shrinkage_lambda)

        self.tree_ = DecisionTreeRegressor(
            max_leaf_nodes=self.max_leaf_nodes,
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

    def _get_hierarchical_leaf_paths(self):
        """Return leaves split into 4 groups by root + secondary split, each sorted by value."""
        t = self.tree_.tree_
        names = self.feature_names_in_

        root_feat = names[int(t.feature[0])]
        root_thresh = t.threshold[0]
        root_left_node = int(t.children_left[0])
        root_right_node = int(t.children_right[0])

        def get_secondary(node):
            if t.children_left[node] == -1:
                return None, None
            return names[int(t.feature[node])], t.threshold[node]

        left_feat, left_thresh = get_secondary(root_left_node)
        right_feat, right_thresh = get_secondary(root_right_node)

        all_paths = []

        def traverse(node, conditions):
            if t.children_left[node] == -1:
                all_paths.append({
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

        root_left_cond = f"{root_feat} <= {root_thresh:.4g}"
        left_paths = [p for p in all_paths if p["conditions"][0] == root_left_cond]
        right_paths = [p for p in all_paths if p["conditions"][0] != root_left_cond]

        if left_feat is not None:
            left_left_cond = f"{left_feat} <= {left_thresh:.4g}"
            ll = sorted([p for p in left_paths if len(p["conditions"]) > 1 and p["conditions"][1] == left_left_cond],
                        key=lambda p: p["value"])
            lr = sorted([p for p in left_paths if not (len(p["conditions"]) > 1 and p["conditions"][1] == left_left_cond)],
                        key=lambda p: p["value"])
        else:
            ll = sorted(left_paths, key=lambda p: p["value"])
            lr = []
            left_feat, left_thresh = root_feat, root_thresh

        if right_feat is not None:
            right_left_cond = f"{right_feat} <= {right_thresh:.4g}"
            rl = sorted([p for p in right_paths if len(p["conditions"]) > 1 and p["conditions"][1] == right_left_cond],
                        key=lambda p: p["value"])
            rr = sorted([p for p in right_paths if not (len(p["conditions"]) > 1 and p["conditions"][1] == right_left_cond)],
                        key=lambda p: p["value"])
        else:
            rl = sorted(right_paths, key=lambda p: p["value"])
            rr = []
            right_feat, right_thresh = root_feat, root_thresh

        return (root_feat, root_thresh,
                left_feat, left_thresh, ll, lr,
                right_feat, right_thresh, rl, rr)

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
            f"CV_HSDT_FDR_4Group(max_leaf_nodes={self.max_leaf_nodes}, "
            f"selected_lambda={self.lambda_:.1f}, cv={self.cv})",
            f"  nodes={t.node_count}, leaves={n_leaves}",
            "",
            "Tree structure (follow from root; leaf values are shrunk predictions):",
        ]
        lines.extend(self._tree_lines())

        # 4-group hierarchical lookup: root → secondary → scan ~12 rules per group.
        (root_feat, root_thresh,
         left_feat, left_thresh, ll_paths, lr_paths,
         right_feat, right_thresh, rl_paths, rr_paths) = self._get_hierarchical_leaf_paths()
        lines += [
            "",
            "Decision rules (4-group hierarchical lookup: check root → check secondary → scan group):",
            f"  Root split:             {root_feat} <= {root_thresh:.4g}",
            f"  Left subtree secondary: {left_feat} <= {left_thresh:.4g}",
            f"  Right subtree secondary:{right_feat} <= {right_thresh:.4g}",
        ]
        if ll_paths:
            lines.append(f"\n  Group LL — {root_feat} <= {root_thresh:.4g} AND {left_feat} <= {left_thresh:.4g} ({len(ll_paths)} rules, sorted by prediction):")
            for i, rule in enumerate(ll_paths, 1):
                rest = " AND ".join(rule["conditions"][2:]) if len(rule["conditions"]) > 2 else "(no further conditions)"
                lines.append(f"    LL{i:02d}: IF {rest}  THEN predict {rule['value']:.4f}  (n={rule['n_samples']})")
        if lr_paths:
            lines.append(f"\n  Group LR — {root_feat} <= {root_thresh:.4g} AND {left_feat} > {left_thresh:.4g} ({len(lr_paths)} rules, sorted by prediction):")
            for i, rule in enumerate(lr_paths, 1):
                rest = " AND ".join(rule["conditions"][2:]) if len(rule["conditions"]) > 2 else "(no further conditions)"
                lines.append(f"    LR{i:02d}: IF {rest}  THEN predict {rule['value']:.4f}  (n={rule['n_samples']})")
        if rl_paths:
            lines.append(f"\n  Group RL — {root_feat} > {root_thresh:.4g} AND {right_feat} <= {right_thresh:.4g} ({len(rl_paths)} rules, sorted by prediction):")
            for i, rule in enumerate(rl_paths, 1):
                rest = " AND ".join(rule["conditions"][2:]) if len(rule["conditions"]) > 2 else "(no further conditions)"
                lines.append(f"    RL{i:02d}: IF {rest}  THEN predict {rule['value']:.4f}  (n={rule['n_samples']})")
        if rr_paths:
            lines.append(f"\n  Group RR — {root_feat} > {root_thresh:.4g} AND {right_feat} > {right_thresh:.4g} ({len(rr_paths)} rules, sorted by prediction):")
            for i, rule in enumerate(rr_paths, 1):
                rest = " AND ".join(rule["conditions"][2:]) if len(rule["conditions"]) > 2 else "(no further conditions)"
                lines.append(f"    RR{i:02d}: IF {rest}  THEN predict {rule['value']:.4f}  (n={rule['n_samples']})")

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
