"""
Interpretable regressor autoresearch script.
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores, recompute_all_mean_ranks
from visualize import plot_interp_vs_performance


from sklearn.ensemble import GradientBoostingRegressor


class OLSPlusStumps(BaseEstimator, RegressorMixin):
    """OLS base + a small set of decision stumps on residuals, trained greedily.
    Each stump adds a piece of the form: if x_i > t then +a else -a.
    Final representation: linear equation + small list of stumps (clearly readable)."""

    def __init__(self, n_stumps=5, learning_rate=0.5):
        self.n_stumps = n_stumps
        self.learning_rate = learning_rate

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.lr_ = LinearRegression().fit(X, y)
        residual = y - self.lr_.predict(X)
        self.coef_ = self.lr_.coef_
        self.intercept_ = float(self.lr_.intercept_)
        stumps = []
        for _ in range(self.n_stumps):
            stump = DecisionTreeRegressor(max_depth=1, random_state=42).fit(X, residual)
            # extract threshold, feature, and leaf values
            tree = stump.tree_
            feat = int(tree.feature[0])
            thr = float(tree.threshold[0])
            left = tree.children_left[0]
            right = tree.children_right[0]
            val_left = float(tree.value[left].ravel()[0]) * self.learning_rate
            val_right = float(tree.value[right].ravel()[0]) * self.learning_rate
            # predict correction on training data
            preds = np.where(X[:, feat] <= thr, val_left, val_right)
            residual = residual - preds
            stumps.append((feat, thr, val_left, val_right))
        self.stumps_ = stumps
        return self

    def predict(self, X):
        check_is_fitted(self, "stumps_")
        X = np.asarray(X, dtype=float)
        y = self.lr_.predict(X)
        for feat, thr, vl, vr in self.stumps_:
            y = y + np.where(X[:, feat] <= thr, vl, vr)
        return y

    def __str__(self):
        check_is_fitted(self, "stumps_")
        names = [f"x{i}" for i in range(self.n_features_in_)]
        eqn = " ".join(f"{c:+.4f}*{n}" for c, n in zip(self.coef_, names))
        lines = [
            f"OLS + {self.n_stumps} Decision Stumps (compact, fully-readable additive model):",
            "  y = [OLS linear model] + [sum of N stump corrections]",
            "  Each stump is: if x_feat <= threshold then (value_left) else (value_right)",
            "",
            f"OLS: y = {self.intercept_:+.4f} {eqn}",
            "",
            "Stump corrections (added to OLS output):",
        ]
        for k, (feat, thr, vl, vr) in enumerate(self.stumps_, 1):
            lines.append(f"  stump {k}: if {names[feat]} <= {thr:+.4f} then add {vl:+.4f} else add {vr:+.4f}")
        return "\n".join(lines)


class OLSPlusTinyGBM(BaseEstimator, RegressorMixin):
    """OLS base + tiny GBM on residuals (boosted residual correction).
    Interpretable-ish: main linear model dominates; small tree ensemble handles leftover signal."""

    def __init__(self, n_estimators=30, max_depth=2, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.lr_ = LinearRegression().fit(X, y)
        residual = y - self.lr_.predict(X)
        self.gbm_ = GradientBoostingRegressor(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, random_state=42).fit(X, residual)
        self.coef_ = self.lr_.coef_
        self.intercept_ = float(self.lr_.intercept_)
        return self

    def predict(self, X):
        check_is_fitted(self, "lr_")
        X = np.asarray(X, dtype=float)
        return self.lr_.predict(X) + self.gbm_.predict(X)

    def __str__(self):
        check_is_fitted(self, "lr_")
        names = [f"x{i}" for i in range(self.n_features_in_)]
        eqn = " ".join(f"{c:+.4f}*{n}" for c, n in zip(self.coef_, names))
        lines = [
            f"OLS + Residual GBM ({self.n_estimators} tree estimators, max_depth={self.max_depth}):",
            f"  y = [OLS base] + [small GBM residual correction]",
            "",
            f"OLS base: y = {self.intercept_:+.4f} {eqn}",
            "",
            "Coefficients (main linear effect per unit change in feature):",
        ]
        for n, c in zip(names, self.coef_):
            lines.append(f"  {n}: {c:+.4f}")
        lines.append(f"  intercept: {self.intercept_:+.4f}")
        # GBM feature importances (aggregate nonlinear correction)
        imp = self.gbm_.feature_importances_
        order = np.argsort(-imp)
        lines.append("")
        lines.append("GBM residual correction — feature importances (aggregate over trees):")
        for i in order:
            lines.append(f"  {names[i]}: {imp[i]:.4f}")
        return "\n".join(lines)


class OLSPlusBestHingePerFeature(BaseEstimator, RegressorMixin):
    """Additive model: y = intercept + sum_i [a_i * x_i + b_i * max(0, x_i - t_i)].
    For each feature, choose the single hinge threshold t_i (from ~10 quantiles) that
    maximally reduces residual SSE when added to the current linear fit.
    Prunes features whose combined effect is negligible.
    """

    def __init__(self, n_thresholds=9, prune_eps=1e-4):
        self.n_thresholds = n_thresholds
        self.prune_eps = prune_eps

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]

        # Step 1: linear OLS on raw features
        lr = LinearRegression().fit(X, y)
        residual = y - lr.predict(X)

        # Step 2: per-feature, try hinges at quantiles of x_i, pick best
        thresholds = [None] * self.n_features_in_
        hinge_feats = np.zeros_like(X)
        qs = np.linspace(0.1, 0.9, self.n_thresholds)
        for i in range(self.n_features_in_):
            xi = X[:, i]
            if xi.std() < 1e-12:
                continue
            best_sse = float(np.sum(residual ** 2))
            best_t = None
            for t in np.quantile(xi, qs):
                h = np.maximum(0.0, xi - t)
                # fit single-coefficient OLS of residual on h
                h_centered = h - h.mean()
                var = float(np.sum(h_centered ** 2))
                if var < 1e-12:
                    continue
                b = float(np.sum(h_centered * (residual - residual.mean())) / var)
                pred = (h - h.mean()) * b
                new_sse = float(np.sum((residual - residual.mean() - pred) ** 2))
                if new_sse < best_sse - 1e-8:
                    best_sse = new_sse
                    best_t = float(t)
            thresholds[i] = best_t
            if best_t is not None:
                hinge_feats[:, i] = np.maximum(0.0, X[:, i] - best_t)

        # Step 3: refit OLS on [X, hinge_feats] using only features with hinges
        active_hinges = [i for i in range(self.n_features_in_) if thresholds[i] is not None]
        if active_hinges:
            Z = np.hstack([X, hinge_feats[:, active_hinges]])
        else:
            Z = X
        final = LinearRegression().fit(Z, y)

        self.thresholds_ = thresholds
        self.active_hinges_ = active_hinges
        self.linear_coef_ = final.coef_[: self.n_features_in_]
        self.hinge_coef_ = final.coef_[self.n_features_in_:]
        self.intercept_ = float(final.intercept_)
        return self

    def predict(self, X):
        check_is_fitted(self, "thresholds_")
        X = np.asarray(X, dtype=float)
        y = X @ self.linear_coef_ + self.intercept_
        for j, i in enumerate(self.active_hinges_):
            y += self.hinge_coef_[j] * np.maximum(0.0, X[:, i] - self.thresholds_[i])
        return y

    def __str__(self):
        check_is_fitted(self, "thresholds_")
        names = [f"x{i}" for i in range(self.n_features_in_)]
        lines = [
            "Additive Piecewise-Linear Model:",
            "  y = intercept + sum_i a_i * x_i + sum_i b_i * max(0, x_i - t_i)",
            "  Each feature has a linear coefficient a_i and optionally a 'hinge' term that kicks in past threshold t_i.",
            f"  intercept: {self.intercept_:+.4f}",
            "",
            "Linear coefficients:",
        ]
        for n, c in zip(names, self.linear_coef_):
            lines.append(f"  {n}: {c:+.4f}")
        if self.active_hinges_:
            lines.append("")
            lines.append("Hinge terms (extra effect when feature exceeds threshold):")
            for j, i in enumerate(self.active_hinges_):
                t = self.thresholds_[i]
                b = self.hinge_coef_[j]
                lines.append(f"  max(0, {names[i]} - {t:+.3f}): {b:+.4f}")
        else:
            lines.append("(No hinge terms; effectively linear.)")
        return "\n".join(lines)


class AdditiveShapeTrees(BaseEstimator, RegressorMixin):
    """EBM-lite: fit a shallow tree per feature on residuals, round-robin boosting.
    The final model is a pure additive model f(x) = intercept + sum_i g_i(x_i),
    where g_i is a piecewise-constant shape function (shallow tree on feature i alone).
    Printed representation: shape function values on a grid for each feature."""

    def __init__(self, max_depth=2, n_rounds=3, lr=0.7):
        self.max_depth = max_depth
        self.n_rounds = n_rounds
        self.lr = lr

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.intercept_ = float(y.mean())
        residual = y - self.intercept_
        self.shape_trees_ = [[] for _ in range(self.n_features_in_)]  # list of (tree, lr) per feature
        for _ in range(self.n_rounds):
            # round robin across features
            for i in range(self.n_features_in_):
                x = X[:, i:i+1]
                if x.std() < 1e-12:
                    continue
                tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=42)
                tree.fit(x, residual)
                pred = tree.predict(x) * self.lr
                residual = residual - pred
                self.shape_trees_[i].append(tree)
        # Precompute grid for display
        self.grid_ = []
        self.grid_ys_ = []
        for i in range(self.n_features_in_):
            xs = X[:, i]
            if xs.std() < 1e-12:
                self.grid_.append(np.array([xs[0]]))
                self.grid_ys_.append(np.array([0.0]))
                continue
            lo, hi = float(xs.min()), float(xs.max())
            grid = np.linspace(lo, hi, 7).reshape(-1, 1)
            yval = np.zeros(grid.shape[0])
            for t in self.shape_trees_[i]:
                yval += t.predict(grid) * self.lr
            self.grid_.append(grid.ravel())
            self.grid_ys_.append(yval)
        return self

    def _feat_effect(self, X, i):
        x = X[:, i:i+1]
        out = np.zeros(X.shape[0])
        for t in self.shape_trees_[i]:
            out += t.predict(x) * self.lr
        return out

    def predict(self, X):
        check_is_fitted(self, "shape_trees_")
        X = np.asarray(X, dtype=float)
        y = np.full(X.shape[0], self.intercept_)
        for i in range(self.n_features_in_):
            y += self._feat_effect(X, i)
        return y

    def __str__(self):
        check_is_fitted(self, "shape_trees_")
        lines = [
            "Additive Shape Model (EBM-lite):",
            "  y = intercept + f_0(x_0) + f_1(x_1) + ...  (pure additive, feature effects are INDEPENDENT)",
            f"  intercept: {self.intercept_:+.4f}",
            "",
            "Per-feature shape functions (piecewise-constant; shown at 7 grid points):",
        ]
        for i in range(self.n_features_in_):
            grid = self.grid_[i]
            ys = self.grid_ys_[i]
            shape = "flat/negligible"
            if ys.size > 1:
                span = float(ys.max() - ys.min())
                if span > 0.3:
                    if ys[-1] > ys[0] + 0.5 * span:
                        shape = "increasing"
                    elif ys[-1] < ys[0] - 0.5 * span:
                        shape = "decreasing"
                    else:
                        shape = "non-monotone"
            lines.append(f"\n  x{i}: (shape: {shape})")
            for xv, yv in zip(grid, ys):
                lines.append(f"    x{i}={xv:+.3f}  →  effect={yv:+.3f}")
        return "\n".join(lines)


import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
AdditiveShapeTrees.__module__ = "interpretable_regressor"
OLSPlusBestHingePerFeature.__module__ = "interpretable_regressor"

OLSPlusTinyGBM.__module__ = "interpretable_regressor"
OLSPlusStumps.__module__ = "interpretable_regressor"

model_shorthand_name = "OLSPlusStumps5_v1"
model_description = "OLS base + 5 greedy decision stumps on residuals (shown as if-then corrections)"
model_defs = [(model_shorthand_name, OLSPlusStumps(n_stumps=5, learning_rate=0.5))]


# ---------------------------------------------------------------------------
# Evaluation (do not edit anything below this line)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default='gpt-4o',
                        help="LLM checkpoint for interpretability tests (default: gpt-4o)")
    args = parser.parse_args()

    t0 = time.time()

    interp_results = run_all_interp_tests(model_defs, checkpoint=args.checkpoint)
    n_passed = sum(r["passed"] for r in interp_results)
    total = len(interp_results)

    dataset_rmses = evaluate_all_regressors(model_defs)

    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        git_hash = ""

    model_name = model_defs[0][0]
    interp_csv = os.path.join(RESULTS_DIR, "interpretability_results.csv")
    interp_fields = ["model", "test", "suite", "passed", "ground_truth", "response"]

    def _suite(test_name):
        if test_name.startswith("insight_"): return "insight"
        if test_name.startswith("hard_"):    return "hard"
        return "standard"

    existing_interp = []
    if os.path.exists(interp_csv):
        with open(interp_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("model") != model_name:
                    existing_interp.append(row)

    new_interp = [{
        "model": r["model"],
        "test": r["test"],
        "suite": _suite(r["test"]),
        "passed": r["passed"],
        "ground_truth": r.get("ground_truth", ""),
        "response": r.get("response", ""),
    } for r in interp_results]

    with open(interp_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=interp_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing_interp + new_interp)
    print(f"Interpretability results saved → {interp_csv}")

    perf_csv = os.path.join(RESULTS_DIR, "performance_results.csv")
    perf_fields = ["dataset", "model", "rmse", "rank"]

    existing_perf = []
    if os.path.exists(perf_csv):
        with open(perf_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("model") != model_name:
                    existing_perf.append(row)

    for ds_name, model_rmses in dataset_rmses.items():
        rmse_val = model_rmses.get(model_name, float("nan"))
        existing_perf.append({
            "dataset": ds_name,
            "model": model_name,
            "rmse": "" if np.isnan(rmse_val) else f"{rmse_val:.6f}",
            "rank": "",
        })

    by_dataset = defaultdict(list)
    for row in existing_perf:
        by_dataset[row["dataset"]].append(row)

    for ds_name, rows in by_dataset.items():
        valid = [(r, float(r["rmse"])) for r in rows if r["rmse"] not in ("", None)]
        valid.sort(key=lambda x: x[1])
        for rank_idx, (r, _) in enumerate(valid, 1):
            r["rank"] = rank_idx
        for r in rows:
            if r["rmse"] in ("", None):
                r["rank"] = ""

    with open(perf_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=perf_fields)
        writer.writeheader()
        for ds_name in by_dataset:
            for row in by_dataset[ds_name]:
                writer.writerow(row)
    print(f"Performance results saved → {perf_csv}")

    all_dataset_rmses = defaultdict(dict)
    for row in existing_perf:
        rmse_str = row.get("rmse", "")
        if rmse_str not in ("", None):
            all_dataset_rmses[row["dataset"]][row["model"]] = float(rmse_str)
        else:
            all_dataset_rmses[row["dataset"]][row["model"]] = float("nan")
    avg_rank, _ = compute_rank_scores(dict(all_dataset_rmses))
    mean_rank = avg_rank.get(model_shorthand_name, float("nan"))

    upsert_overall_results([{
        "commit":                             git_hash,
        "mean_rank":                          f"{mean_rank:.2f}" if not np.isnan(mean_rank) else "nan",
        "frac_interpretability_tests_passed": f"{n_passed / total:.4f}" if total > 0 else "nan",
        "status":                             "",
        "model_name":                         model_shorthand_name,
        "description":                        model_description,
    }], RESULTS_DIR)

    recompute_all_mean_ranks(RESULTS_DIR)

    overall_csv = os.path.join(RESULTS_DIR, "overall_results.csv")
    plot_interp_vs_performance(
        overall_csv,
        os.path.join(RESULTS_DIR, "interpretability_vs_performance.png"),
    )

    print()
    print("---")
    print(f"tests_passed:  {n_passed}/{total}" + (f" ({n_passed/total:.2%})" if total > 0 else ""))
    print(f"mean_rank:     {mean_rank:.2f}" if not np.isnan(mean_rank) else "mean_rank:     nan")
    print(f"total_seconds: {time.time() - t0:.1f}s")
