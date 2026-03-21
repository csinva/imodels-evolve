"""
Interpretable regressor autoresearch script.
Defines a scikit-learn compatible interpretable regressor and evaluates it
on interpretability tests (same suite used for baselines in run_baselines.py).

Usage: uv run model.py
"""

import os
import sys
import time

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "eval"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this — everything below is fair game)
# ---------------------------------------------------------------------------

MAX_DEPTH = 4
MIN_SAMPLES_LEAF = 10


class InterpretableRegressor(BaseEstimator, RegressorMixin):
    """
    Interpretable scikit-learn compatible regressor.

    This is the baseline: a shallow decision tree.
    The agent may modify this class freely — algorithm, hyperparameters, features, etc.
    Must implement: fit(X, y), predict(X).
    """

    def __init__(self, max_depth=MAX_DEPTH, min_samples_leaf=MIN_SAMPLES_LEAF):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        self.tree_ = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
        )
        self.tree_.fit(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self, "tree_")
        return self.tree_.predict(X)

    def __str__(self):
        check_is_fitted(self, "tree_")
        n_leaves = self.tree_.get_n_leaves()
        n_nodes = self.tree_.tree_.node_count
        return (
            f"InterpretableRegressor(max_depth={self.max_depth}, "
            f"min_samples_leaf={self.min_samples_leaf})\n"
            f"  nodes={n_nodes}, leaves={n_leaves}"
        )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()

    model_defs = [("InterpretableRegressor", InterpretableRegressor())]
    interp_results = run_all_interp_tests(model_defs)

    n_passed = sum(r["passed"] for r in interp_results)
    total = len(interp_results)
    std  = sum(r["passed"] for r in interp_results if r["test"] in {t.__name__ for t in ALL_TESTS})
    hard = sum(r["passed"] for r in interp_results if r["test"] in {t.__name__ for t in HARD_TESTS})
    ins  = sum(r["passed"] for r in interp_results if r["test"] in {t.__name__ for t in INSIGHT_TESTS})

    print()
    print("---")
    print(f"tests_passed:  {n_passed}/{total} ({n_passed/total:.2%})  "
          f"[std {std}/8  hard {hard}/5  insight {ins}/5]")
    print(f"total_seconds: {time.time() - t0:.1f}s")
