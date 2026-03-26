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
    Greedy Additive Stumps (GAS): ensemble of depth-1 decision trees (stumps)
    fitted greedily on residuals (gradient boosting with stumps).

    Prediction: y_hat = base_value + sum_i(learning_rate * stump_i.predict(X))

    The __str__ shows:
      - All rules as simple if/else conditions with their contributions,
        sorted by |left_contribution - right_contribution| (impact magnitude)
      - Aggregated feature importances across all stumps
      - Net direction of effect per feature (higher value → higher/lower prediction)
      - Unused features
    """

    def __init__(self, n_stumps=25, learning_rate=0.4, min_samples_leaf=10):
        self.n_stumps = n_stumps
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        if hasattr(X, "columns"):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = [f"x{i}" for i in range(X.shape[1])]

        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        self.base_value_ = float(np.mean(y_arr))
        residual = y_arr - self.base_value_

        self.stumps_ = []
        for _ in range(self.n_stumps):
            stump = DecisionTreeRegressor(
                max_depth=1,
                min_samples_leaf=self.min_samples_leaf,
                random_state=42,
            )
            stump.fit(X_arr, residual)
            pred = stump.predict(X_arr)
            residual -= self.learning_rate * pred
            self.stumps_.append(stump)

        return self

    def predict(self, X):
        check_is_fitted(self, "stumps_")
        X_arr = np.asarray(X, dtype=float)
        pred = np.full(X_arr.shape[0], self.base_value_)
        for stump in self.stumps_:
            pred += self.learning_rate * stump.predict(X_arr)
        return pred

    def _stump_info(self, stump):
        """Extract feature, threshold, and scaled leaf values from a depth-1 tree."""
        t = stump.tree_
        feature_idx = int(t.feature[0])
        threshold = float(t.threshold[0])
        left_val = float(t.value[t.children_left[0]][0][0]) * self.learning_rate
        right_val = float(t.value[t.children_right[0]][0][0]) * self.learning_rate
        impact = abs(right_val - left_val)
        return {
            "feature_idx": feature_idx,
            "feature_name": self.feature_names_in_[feature_idx],
            "threshold": threshold,
            "left_val": left_val,
            "right_val": right_val,
            "impact": impact,
        }

    def __str__(self):
        check_is_fitted(self, "stumps_")

        infos = [self._stump_info(s) for s in self.stumps_]
        infos_sorted = sorted(infos, key=lambda x: x["impact"], reverse=True)

        lines = [
            f"GreedyAdditiveStumps(n_stumps={self.n_stumps}, "
            f"learning_rate={self.learning_rate}, "
            f"min_samples_leaf={self.min_samples_leaf})",
            f"  base_value = {self.base_value_:.4f}  "
            f"(prediction = base_value + sum of contributions below)",
            "",
            "Rules (each is an independent if/else; apply ALL rules and sum contributions):",
        ]

        for i, info in enumerate(infos_sorted):
            fname = info["feature_name"]
            thresh = info["threshold"]
            lv = info["left_val"]
            rv = info["right_val"]
            lines.append(
                f"  Rule {i+1:2d}: if {fname} <= {thresh:.4g}: "
                f"contribution={lv:+.4f}  |  else: contribution={rv:+.4f}"
            )

        lines.append("")

        # Aggregated feature importances
        feat_imp: dict = {}
        feat_direction: dict = {}  # weighted sum: right_val - left_val, weighted by impact
        for info in infos:
            fname = info["feature_name"]
            feat_imp[fname] = feat_imp.get(fname, 0.0) + info["impact"]
            feat_direction[fname] = (
                feat_direction.get(fname, 0.0)
                + (info["right_val"] - info["left_val"]) * info["impact"]
            )

        total_imp = sum(feat_imp.values()) or 1.0
        feat_imp_sorted = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)

        lines.append(
            "Feature importances (sum of |contribution gap| across rules; higher = more important):"
        )
        for fname, imp in feat_imp_sorted:
            if imp > 1e-9:
                d = feat_direction[fname]
                if d > 0:
                    direction = "positive (higher value → higher prediction)"
                elif d < 0:
                    direction = "negative (higher value → lower prediction)"
                else:
                    direction = "mixed"
                lines.append(
                    f"  {fname:<25s}  importance={imp/total_imp:.4f}  "
                    f"net direction: {direction}"
                )

        used = set(feat_imp.keys())
        unused = [f for f in self.feature_names_in_ if f not in used]
        if unused:
            lines.append(
                f"\nFeatures not used (zero importance): {', '.join(unused)}"
            )

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
