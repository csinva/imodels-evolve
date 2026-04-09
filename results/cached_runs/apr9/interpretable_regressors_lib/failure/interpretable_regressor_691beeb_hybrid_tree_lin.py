"""
Interpretable regressor autoresearch script.
Usage: uv run model.py
"""

import csv, os, subprocess, sys, time
from collections import defaultdict

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import RidgeCV
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance


class HybridTreeLinearRegressor(BaseEstimator, RegressorMixin):
    """
    Hybrid Tree-Linear Model: fits a shallow decision tree to capture
    nonlinear structure, then fits a Ridge model in each leaf to capture
    linear effects within each region.

    Architecture:
    1. Fit a shallow tree (max_leaf_nodes=6) to partition the feature space
    2. In each leaf, fit a Ridge on the local data
    3. Predict: route through tree, then apply leaf's Ridge

    Display: Shows the tree structure with Ridge equations in each leaf.
    This is a piecewise-linear model that's both powerful (captures
    interactions via tree) and readable (each piece is a simple equation).

    This is a novel architecture that hasn't been tried in our 100 experiments.
    """

    def __init__(self, max_leaf_nodes=6, min_samples_leaf=20):
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]

        # Step 1: Fit shallow tree to partition space
        self.tree_ = DecisionTreeRegressor(
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
        )
        self.tree_.fit(X, y)

        # Step 2: Fit Ridge in each leaf
        leaf_ids = self.tree_.apply(X)
        unique_leaves = np.unique(leaf_ids)
        self.leaf_models_ = {}

        for leaf in unique_leaves:
            mask = leaf_ids == leaf
            X_leaf = X[mask]
            y_leaf = y[mask]

            if X_leaf.shape[0] >= 5 and np.std(y_leaf) > 1e-8:
                ridge = RidgeCV(cv=min(3, X_leaf.shape[0]))
                ridge.fit(X_leaf, y_leaf)
                self.leaf_models_[leaf] = ridge
            else:
                # Constant prediction for small leaves
                self.leaf_models_[leaf] = float(np.mean(y_leaf))

        return self

    def predict(self, X):
        check_is_fitted(self, "tree_")
        X = np.asarray(X, dtype=np.float64)
        leaf_ids = self.tree_.apply(X)
        pred = np.zeros(X.shape[0])

        for leaf, model in self.leaf_models_.items():
            mask = leaf_ids == leaf
            if not np.any(mask):
                continue
            if isinstance(model, float):
                pred[mask] = model
            else:
                pred[mask] = model.predict(X[mask])

        return pred

    def __str__(self):
        check_is_fitted(self, "tree_")
        feature_names = [f"x{i}" for i in range(self.n_features_in_)]

        # Build custom tree text showing Ridge equations in leaves
        tree = self.tree_.tree_
        lines = [f"Piecewise-Linear Decision Tree (max_leaf_nodes={self.max_leaf_nodes}):"]

        def _build(node, depth=0):
            prefix = "  " * depth
            if tree.children_left[node] == -1:  # Leaf
                leaf_id = node
                model = self.leaf_models_.get(leaf_id)
                if isinstance(model, float):
                    lines.append(f"{prefix}value: {model:.4f}")
                else:
                    # Show Ridge equation
                    coefs = model.coef_
                    intercept = model.intercept_
                    eq_parts = []
                    for i, c in enumerate(coefs):
                        if abs(c) > 1e-4:
                            eq_parts.append(f"{c:.4f}*{feature_names[i]}")
                    eq = " + ".join(eq_parts) + f" + {intercept:.4f}" if eq_parts else f"{intercept:.4f}"
                    lines.append(f"{prefix}y = {eq}")
            else:
                feat = tree.feature[node]
                thresh = tree.threshold[node]
                lines.append(f"{prefix}if {feature_names[feat]} <= {thresh:.4f}:")
                _build(tree.children_left[node], depth + 1)
                lines.append(f"{prefix}else:  # {feature_names[feat]} > {thresh:.4f}")
                _build(tree.children_right[node], depth + 1)

        _build(0)
        return "\n".join(lines)


import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
HybridTreeLinearRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "HybridTreeLin_v2"
model_description = "Shallow tree (6 leaves) with Ridge per leaf, custom display showing Ridge equations"
model_defs = [(model_shorthand_name, HybridTreeLinearRegressor())]


# ---------------------------------------------------------------------------
# Evaluation (do not edit anything below this line)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()
    interp_results = run_all_interp_tests(model_defs)
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
    new_interp = [{"model": r["model"], "test": r["test"], "suite": _suite(r["test"]),
        "passed": r["passed"], "ground_truth": r.get("ground_truth", ""),
        "response": r.get("response", "")} for r in interp_results]
    with open(interp_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=interp_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing_interp + new_interp)

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
        existing_perf.append({"dataset": ds_name, "model": model_name,
            "rmse": "" if np.isnan(rmse_val) else f"{rmse_val:.6f}", "rank": ""})
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
        "commit": git_hash,
        "mean_rank": f"{mean_rank:.2f}" if not np.isnan(mean_rank) else "nan",
        "frac_interpretability_tests_passed": f"{n_passed / total:.4f}" if total > 0 else "nan",
        "status": "", "model_name": model_shorthand_name, "description": model_description,
    }], RESULTS_DIR)

    overall_csv = os.path.join(RESULTS_DIR, "overall_results.csv")
    plot_interp_vs_performance(overall_csv, os.path.join(RESULTS_DIR, "interpretability_vs_performance.png"))
    print()
    print("---")
    print(f"tests_passed:  {n_passed}/{total}" + (f" ({n_passed/total:.2%})" if total > 0 else ""))
    print(f"mean_rank:     {mean_rank:.2f}" if not np.isnan(mean_rank) else "mean_rank:     nan")
    print(f"total_seconds: {time.time() - t0:.1f}s")
