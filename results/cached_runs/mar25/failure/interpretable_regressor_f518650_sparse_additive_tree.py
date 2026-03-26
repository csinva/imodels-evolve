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
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "eval"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class InterpretableRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse Additive Tree Model (custom GAM).

    Fits one shallow decision tree per feature (univariate), selects the top-k
    most important features via variance of predictions, then combines them
    additively with a global intercept. The __str__() output shows explicit
    per-feature tree rules, making it easy for humans/LLMs to trace predictions.

    Must implement: fit(X, y), predict(X), and __str__().
    """

    def __init__(self, max_depth=3, min_samples_leaf=5, max_features=8,
                 n_boost_rounds=5):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_boost_rounds = n_boost_rounds

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]

        # Stage 1: Greedy boosted feature selection
        # Fit univariate trees one at a time, always picking the feature
        # that best reduces residual variance (forward stagewise additive)
        self.intercept_ = float(np.mean(y))
        residual = y - self.intercept_

        self.feature_trees_ = []  # list of (feature_idx, tree)
        self.selected_features_ = []
        used_features = set()

        for boosting_round in range(self.n_boost_rounds):
            best_score = -np.inf
            best_feat = None
            best_tree = None

            for j in range(X.shape[1]):
                if len(used_features) >= self.max_features and j not in used_features:
                    continue
                tree = DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=42,
                )
                tree.fit(X[:, j:j+1], residual)
                pred = tree.predict(X[:, j:j+1])
                # Score = reduction in residual variance
                score = np.var(residual) - np.var(residual - pred)
                if score > best_score:
                    best_score = score
                    best_feat = j
                    best_tree = tree

            if best_tree is None or best_score <= 0:
                break

            self.feature_trees_.append((best_feat, best_tree))
            used_features.add(best_feat)
            if best_feat not in self.selected_features_:
                self.selected_features_.append(best_feat)
            residual = residual - best_tree.predict(X[:, best_feat:best_feat+1])

        return self

    def predict(self, X):
        check_is_fitted(self, "feature_trees_")
        X = np.asarray(X, dtype=np.float64)
        pred = np.full(X.shape[0], self.intercept_)
        for feat_idx, tree in self.feature_trees_:
            pred += tree.predict(X[:, feat_idx:feat_idx+1])
        return pred

    def __str__(self):
        check_is_fitted(self, "feature_trees_")
        lines = []
        lines.append(f"Sparse Additive Tree Model")
        lines.append(f"intercept = {self.intercept_:.4f}")
        lines.append(f"prediction = intercept + sum of per-feature tree contributions")
        lines.append(f"selected features: {['x' + str(i) for i in self.selected_features_]}")
        lines.append("")

        for round_idx, (feat_idx, tree) in enumerate(self.feature_trees_):
            fname = f"x{feat_idx}"
            feature_names = [fname]
            tree_text = export_text(tree, feature_names=feature_names, max_depth=6)
            lines.append(f"--- Component {round_idx+1}: f_{round_idx+1}({fname}) ---")
            lines.append(tree_text)

        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
InterpretableRegressor.__module__ = "interpretable_regressor"


# ---------------------------------------------------------------------------
# Evaluation (do not edit any of the evaluation functions, only the names and model descriptions below this line)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()

    MODEL_NAME = "SparseAdditiveTree"
    model_defs = [(MODEL_NAME, InterpretableRegressor())]

    # Interpretability tests
    interp_results = run_all_interp_tests(model_defs)
    n_passed = sum(r["passed"] for r in interp_results)
    total = len(interp_results)

    # prediction performance (RMSE)
    dataset_rmses = evaluate_all_regressors(model_defs)
    rmse_vals = [v[MODEL_NAME] for v in dataset_rmses.values()
                 if not np.isnan(v.get(MODEL_NAME, float("nan")))]
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
        "model_name":                         MODEL_NAME,
        "description":                        "Sparse additive tree model: boosted per-feature shallow trees with greedy feature selection",
    }], RESULTS_DIR)

    # --- Plot ---
    overall_csv = os.path.join(RESULTS_DIR, "overall_results.csv")
    plot_interp_vs_performance(
        overall_csv,
        os.path.join(RESULTS_DIR, "interpretability_vs_performance.png"),
    )

    print()
    print("---")
    print(f"tests_passed:  {n_passed}/{total} ({n_passed/total:.2%})")
    print(f"mean_rmse:     {mean_rmse:.4f}")
    print(f"total_seconds: {time.time() - t0:.1f}s")

