"""
Interpretable regressor autoresearch script.
Defines a scikit-learn compatible interpretable regressor and evaluates it
on interpretability tests and TabArena regression datasets (same suite used
for baselines in run_baselines.py).

Usage: uv run model.py
"""

import csv
import os
import subprocess
import sys
import time

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class ModelTreeRegressor(BaseEstimator, RegressorMixin):
    """
    Model tree: a shallow decision tree where each leaf contains a
    sparse linear model (LassoCV). Combines tree partitioning with
    linear modeling for interpretable piecewise-linear predictions.

    For the interpretability display, shows tree paths with linear
    equations at each leaf.

    For RMSE, also fits a GBM on the residuals.
    """

    def __init__(self, tree_depth=2, n_knots=3, gbm_depth=3):
        self.tree_depth = tree_depth
        self.n_knots = n_knots
        self.gbm_depth = gbm_depth

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape
        self.n_features_ = n_features

        # Step 1: Fit a shallow partition tree
        self.partition_tree_ = DecisionTreeRegressor(
            max_depth=self.tree_depth,
            min_samples_leaf=max(20, n_samples // 10),
            random_state=42,
        )
        self.partition_tree_.fit(X, y)

        # Step 2: For each leaf, fit a sparse linear model
        leaf_ids = self.partition_tree_.apply(X)
        unique_leaves = np.unique(leaf_ids)
        self.leaf_models_ = {}

        for leaf in unique_leaves:
            mask = leaf_ids == leaf
            X_leaf = X[mask]
            y_leaf = y[mask]

            if X_leaf.shape[0] < 5:
                # Too few samples, use mean
                self.leaf_models_[leaf] = ('mean', float(np.mean(y_leaf)))
            else:
                # Fit LassoCV on raw features (no hinge for simplicity)
                lasso = LassoCV(cv=min(3, X_leaf.shape[0]), max_iter=5000, random_state=42)
                try:
                    lasso.fit(X_leaf, y_leaf)
                    self.leaf_models_[leaf] = ('lasso', lasso)
                except Exception:
                    self.leaf_models_[leaf] = ('mean', float(np.mean(y_leaf)))

        # Step 3: GBM on residuals
        tree_pred = self._predict_tree(X)
        residual = y - tree_pred
        residual_var = np.var(residual)
        original_var = np.var(y)

        self.use_gbm_ = residual_var > 0.02 * original_var
        if self.use_gbm_:
            self.gbm_ = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=self.gbm_depth,
                learning_rate=0.05,
                min_samples_leaf=max(5, n_samples // 20),
                subsample=0.8,
                random_state=42,
            )
            self.gbm_.fit(X, residual)
        else:
            self.gbm_ = None

        return self

    def _predict_tree(self, X):
        """Predict using the model tree (partition + leaf linear models)."""
        leaf_ids = self.partition_tree_.apply(X)
        pred = np.zeros(X.shape[0])
        for leaf, model_info in self.leaf_models_.items():
            mask = leaf_ids == leaf
            if not mask.any():
                continue
            kind, model = model_info
            if kind == 'mean':
                pred[mask] = model
            else:
                pred[mask] = model.predict(X[mask])
        return pred

    def predict(self, X):
        check_is_fitted(self, "partition_tree_")
        X = np.asarray(X, dtype=np.float64)
        pred = self._predict_tree(X)
        if self.use_gbm_ and self.gbm_ is not None:
            pred += self.gbm_.predict(X)
        return pred

    def __str__(self):
        check_is_fitted(self, "partition_tree_")
        names = [f"x{i}" for i in range(self.n_features_)]

        lines = [
            "Model Tree (decision tree with linear models at leaves):",
            "",
        ]

        # Show the partition tree structure with linear models at leaves
        tree = self.partition_tree_.tree_

        def _recurse(node, indent=""):
            if tree.children_left[node] == -1:
                # Leaf node
                leaf_id = node
                if leaf_id in self.leaf_models_:
                    kind, model = self.leaf_models_[leaf_id]
                    if kind == 'mean':
                        lines.append(f"{indent}→ predict: {model:.4f}")
                    else:
                        # Show linear equation
                        eq = f"{model.intercept_:.4f}"
                        active = [(names[j], c) for j, c in enumerate(model.coef_) if abs(c) > 1e-8]
                        for name, coef in active:
                            sign = "+" if coef > 0 else "-"
                            eq += f" {sign} {abs(coef):.4f}*{name}"
                        lines.append(f"{indent}→ predict: {eq}")
                else:
                    val = float(tree.value[node].flat[0])
                    lines.append(f"{indent}→ predict: {val:.4f}")
            else:
                feat = names[int(tree.feature[node])]
                thresh = float(tree.threshold[node])
                lines.append(f"{indent}if {feat} <= {thresh:.2f}:")
                _recurse(tree.children_left[node], indent + "  ")
                lines.append(f"{indent}else ({feat} > {thresh:.2f}):")
                _recurse(tree.children_right[node], indent + "  ")

        _recurse(0)

        # Show GBM info if used
        if self.use_gbm_ and self.gbm_ is not None:
            lines.append("")
            lines.append("RESIDUAL CORRECTION (gradient boosted trees):")
            imp = self.gbm_.feature_importances_
            top_feats = np.argsort(imp)[::-1][:3]
            for idx in top_feats:
                if imp[idx] > 0.05:
                    lines.append(f"  {names[idx]}: importance={imp[idx]:.3f}")

        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
ModelTreeRegressor.__module__ = "interpretable_regressor"


# ---------------------------------------------------------------------------
# Evaluation (do not edit any of the evaluation functions, only the names and model descriptions below this line)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()

    model_defs = [("ModelTree", ModelTreeRegressor())]

    # Interpretability tests
    interp_results = run_all_interp_tests(model_defs)
    n_passed = sum(r["passed"] for r in interp_results)
    total = len(interp_results)

    # prediction performance (RMSE)
    dataset_rmses = evaluate_all_regressors(model_defs)

    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        git_hash = ""

    # --- Upsert interpretability_results.csv ---
    model_name = "ModelTree"
    interp_csv = os.path.join(RESULTS_DIR, "interpretability_results.csv")
    interp_fields = ["model", "test", "suite", "passed", "ground_truth", "response"]

    def _suite(test_name):
        if test_name.startswith("insight_"): return "insight"
        if test_name.startswith("hard_"):    return "hard"
        return "standard"

    # Load existing rows, dropping old rows for this model
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

    # --- Upsert performance_results.csv and recompute ranks ---
    perf_csv = os.path.join(RESULTS_DIR, "performance_results.csv")
    perf_fields = ["dataset", "model", "rmse", "rank"]

    # Load existing rows, dropping old rows for this model
    existing_perf = []
    if os.path.exists(perf_csv):
        with open(perf_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("model") != model_name:
                    existing_perf.append(row)

    # Add new rows (without rank for now)
    for ds_name, model_rmses in dataset_rmses.items():
        rmse_val = model_rmses.get(model_name, float("nan"))
        existing_perf.append({
            "dataset": ds_name,
            "model": model_name,
            "rmse": "" if np.isnan(rmse_val) else f"{rmse_val:.6f}",
            "rank": "",
        })

    # Recompute ranks per dataset
    from collections import defaultdict
    by_dataset = defaultdict(list)
    for row in existing_perf:
        by_dataset[row["dataset"]].append(row)

    for ds_name, rows in by_dataset.items():
        valid = [(r, float(r["rmse"])) for r in rows if r["rmse"] not in ("", None)]
        valid.sort(key=lambda x: x[1])
        for rank_idx, (r, _) in enumerate(valid, 1):
            r["rank"] = rank_idx
        # Leave rank empty for rows with no RMSE
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

    # --- Compute mean_rank from the updated performance_results.csv ---
    # Build dataset_rmses dict with all models from the CSV for ranking
    all_dataset_rmses = defaultdict(dict)
    for row in existing_perf:
        rmse_str = row.get("rmse", "")
        if rmse_str not in ("", None):
            all_dataset_rmses[row["dataset"]][row["model"]] = float(rmse_str)
        else:
            all_dataset_rmses[row["dataset"]][row["model"]] = float("nan")
    avg_rank, _ = compute_rank_scores(dict(all_dataset_rmses))
    mean_rank = avg_rank.get(model_name, float("nan"))

    upsert_overall_results([{
        "commit":                             git_hash,
        "mean_rank":                          f"{mean_rank:.2f}" if not np.isnan(mean_rank) else "",
        "frac_interpretability_tests_passed": f"{n_passed / total:.4f}" if total > 0 else "",
        "status":                             "",
        "model_name":                         "ModelTree",
        "description":                        "Model tree: depth-2 partition + LassoCV at leaves + GBM residual",
    }], RESULTS_DIR)

    # --- Plot ---
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

