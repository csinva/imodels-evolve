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
from collections import defaultdict

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class ResidualTreeSparseLinearRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse linear backbone plus a tiny residual tree.
    - Stage 1: fit a ridge model and keep only the strongest linear terms.
    - Stage 2: fit a shallow residual tree and tune its shrinkage on validation.
    """

    def __init__(
        self,
        alpha_grid=(0.001, 0.01, 0.1, 1.0, 3.0, 10.0),
        holdout_frac=0.2,
        min_linear_terms=3,
        max_linear_terms=8,
        linear_mass_keep=0.985,
        tree_max_leaf_nodes=4,
        tree_min_samples_leaf=20,
        tree_shrinkage_grid=(0.0, 0.2, 0.4, 0.7, 1.0),
        min_tree_rel_gain=0.003,
        random_state=0,
    ):
        self.alpha_grid = alpha_grid
        self.holdout_frac = holdout_frac
        self.min_linear_terms = min_linear_terms
        self.max_linear_terms = max_linear_terms
        self.linear_mass_keep = linear_mass_keep
        self.tree_max_leaf_nodes = tree_max_leaf_nodes
        self.tree_min_samples_leaf = tree_min_samples_leaf
        self.tree_shrinkage_grid = tree_shrinkage_grid
        self.min_tree_rel_gain = min_tree_rel_gain
        self.random_state = random_state

    @staticmethod
    def _mse(y_true, y_pred):
        d = y_true - y_pred
        return float(np.mean(d * d))

    @staticmethod
    def _ridge_fit(D, y, alpha):
        n, p = D.shape
        A = D.T @ D + float(alpha) * np.eye(p, dtype=float)
        A[0, 0] -= float(alpha)  # no regularization on intercept
        b = D.T @ y
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(A) @ b

    def _fit_best_alpha(self, Dtr, ytr, Dval, yval):
        best = None
        for alpha in self.alpha_grid:
            w = self._ridge_fit(Dtr, ytr, alpha=float(alpha))
            mse = self._mse(yval, Dval @ w)
            if best is None or mse < best["mse"]:
                best = {"mse": float(mse), "alpha": float(alpha), "w": np.asarray(w, dtype=float)}
        return best

    @staticmethod
    def _safe_std(X):
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std = np.where(std > 1e-12, std, 1.0)
        return mean.astype(float), std.astype(float)

    @staticmethod
    def _standardize(X, mean, std):
        return (X - mean) / std

    @staticmethod
    def _leaf_count(tree):
        return int(np.sum(tree.tree_.children_left == -1))

    def _fit_ridge_standardized(self, Xtr, ytr, Xval, yval):
        mean, std = self._safe_std(Xtr)
        Xtrz = self._standardize(Xtr, mean, std)
        Xvalz = self._standardize(Xval, mean, std)
        Dtr = np.column_stack([np.ones(len(Xtrz), dtype=float), Xtrz])
        Dval = np.column_stack([np.ones(len(Xvalz), dtype=float), Xvalz])
        fit = self._fit_best_alpha(Dtr, ytr, Dval, yval)

        intercept_z = float(fit["w"][0])
        coef_z = np.asarray(fit["w"][1:], dtype=float)
        coef_raw = coef_z / std
        intercept_raw = intercept_z - float(np.dot(coef_raw, mean))
        return {
            "alpha": float(fit["alpha"]),
            "mse": float(fit["mse"]),
            "intercept": float(intercept_raw),
            "coef": np.asarray(coef_raw, dtype=float),
        }

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        self.n_features_in_ = p

        rng = np.random.RandomState(int(self.random_state))
        n_val = max(20, int(float(self.holdout_frac) * n))
        if n - n_val < 20:
            n_val = max(1, n // 5)
        perm = rng.permutation(n)
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]
        if len(tr_idx) < 10:
            tr_idx = perm
            val_idx = perm[: max(1, min(10, n // 4))]

        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xval, yval = X[val_idx], y[val_idx]

        # Stage 1: dense standardized ridge, then keep strongest coefficient-mass features.
        dense_fit = self._fit_ridge_standardized(Xtr, ytr, Xval, yval)
        dense_coef = np.asarray(dense_fit["coef"], dtype=float)
        order = np.argsort(-np.abs(dense_coef))
        abs_sorted = np.abs(dense_coef[order])
        mass = np.cumsum(abs_sorted) / max(float(np.sum(abs_sorted)), 1e-12)

        max_keep = min(int(self.max_linear_terms), p)
        min_keep = min(max_keep, max(1, int(self.min_linear_terms)))
        keep_n = max_keep
        for i in range(len(order)):
            if i + 1 >= min_keep and mass[i] >= float(self.linear_mass_keep):
                keep_n = i + 1
                break
        keep_n = max(min_keep, min(max_keep, keep_n))
        linear_features = np.sort(order[:keep_n].astype(int))

        # Stage 1b: refit compact ridge on selected features only.
        Xtr_lin = Xtr[:, linear_features]
        Xval_lin = Xval[:, linear_features]
        compact_fit = self._fit_ridge_standardized(Xtr_lin, ytr, Xval_lin, yval)
        lin_intercept = float(compact_fit["intercept"])
        lin_coef = np.asarray(compact_fit["coef"], dtype=float)
        lin_val_pred = lin_intercept + Xval_lin @ lin_coef
        lin_val_mse = self._mse(yval, lin_val_pred)

        # Stage 2: fit tiny residual tree and pick validation shrinkage.
        residual_tr = ytr - (lin_intercept + Xtr_lin @ lin_coef)
        tree = DecisionTreeRegressor(
            max_leaf_nodes=int(self.tree_max_leaf_nodes),
            min_samples_leaf=int(self.tree_min_samples_leaf),
            random_state=int(self.random_state),
        )
        tree.fit(Xtr, residual_tr)
        tree_val_pred = tree.predict(Xval)

        best_tree_weight = 0.0
        best_val_mse = float(lin_val_mse)
        for w in self.tree_shrinkage_grid:
            w = float(w)
            mse = self._mse(yval, lin_val_pred + w * tree_val_pred)
            if mse < best_val_mse:
                best_val_mse = float(mse)
                best_tree_weight = w

        rel_gain = (lin_val_mse - best_val_mse) / max(lin_val_mse, 1e-12)
        if rel_gain < float(self.min_tree_rel_gain):
            best_tree_weight = 0.0
            best_val_mse = float(lin_val_mse)

        # Refit linear part on full data with same selected features.
        full_fit = self._fit_ridge_standardized(X[:, linear_features], y, X[:, linear_features], y)
        self.intercept_ = float(full_fit["intercept"])
        self.linear_features_ = np.asarray(linear_features, dtype=int)
        self.linear_coef_ = np.asarray(full_fit["coef"], dtype=float)
        self.linear_alpha_ = float(full_fit["alpha"])

        # Refit residual tree on full data residuals.
        residual_full = y - (self.intercept_ + X[:, self.linear_features_] @ self.linear_coef_)
        self.residual_tree_ = DecisionTreeRegressor(
            max_leaf_nodes=int(self.tree_max_leaf_nodes),
            min_samples_leaf=int(self.tree_min_samples_leaf),
            random_state=int(self.random_state),
        )
        self.residual_tree_.fit(X, residual_full)
        self.tree_weight_ = float(best_tree_weight)
        self.validation_mse_ = float(best_val_mse)
        self.linear_validation_mse_ = float(lin_val_mse)

        active = set(int(j) for j in self.linear_features_)
        if self.tree_weight_ != 0.0:
            tree_obj = self.residual_tree_.tree_
            for node in range(tree_obj.node_count):
                feat = int(tree_obj.feature[node])
                if feat >= 0:
                    active.add(feat)

        self.active_features_ = np.asarray(sorted(active), dtype=int)
        self.inactive_features_ = np.asarray([j for j in range(p) if j not in active], dtype=int)
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "linear_features_", "linear_coef_", "residual_tree_", "tree_weight_"])
        X = np.asarray(X, dtype=float)
        out = self.intercept_ + X[:, self.linear_features_] @ self.linear_coef_
        if self.tree_weight_ != 0.0:
            out = out + self.tree_weight_ * self.residual_tree_.predict(X)
        return np.asarray(out, dtype=float)

    def _tree_rules_str(self):
        if self.tree_weight_ == 0.0:
            return ["Residual tree contribution: disabled (validation gain too small)."]

        tree = self.residual_tree_.tree_
        lines = [
            f"Residual tree contribution: add {self.tree_weight_:+.6f} * leaf_value",
            f"Residual tree leaves: {self._leaf_count(self.residual_tree_)}",
            "Residual tree rules:",
        ]

        def _walk(node, depth):
            left = int(tree.children_left[node])
            right = int(tree.children_right[node])
            indent = "  " * depth
            if left == -1 and right == -1:
                leaf_val = float(tree.value[node][0][0])
                lines.append(f"{indent}then leaf_value = {leaf_val:+.6f}")
                return
            feat = int(tree.feature[node])
            thr = float(tree.threshold[node])
            lines.append(f"{indent}if x{feat} <= {thr:.6f}:")
            _walk(left, depth + 1)
            lines.append(f"{indent}else:")
            _walk(right, depth + 1)

        _walk(0, 0)
        return lines

    def __str__(self):
        check_is_fitted(self, ["intercept_", "linear_features_", "linear_coef_", "residual_tree_", "tree_weight_"])
        lines = [
            "Residual Tree Sparse Linear Regressor",
            f"linear_ridge_alpha={self.linear_alpha_:.6g}",
            "Prediction = sparse linear equation + shrunken residual tree.",
        ]

        eq = f"y = {self.intercept_:+.6f}"
        for j, c in zip(self.linear_features_, self.linear_coef_):
            eq += f" {float(c):+.6f}*x{int(j)}"
        lines.append(eq)

        lines.append("Sparse linear backbone terms:")
        ranked_linear = sorted(zip(self.linear_features_, self.linear_coef_), key=lambda z: -abs(float(z[1])))
        for j, c in ranked_linear:
            lines.append(f"  x{int(j)}: {float(c):+.6f}")
        lines.extend(self._tree_rules_str())

        if len(self.inactive_features_) > 0:
            lines.append("Features with negligible total effect include: " + ", ".join(f"x{int(j)}" for j in self.inactive_features_))
        lines.append(f"Linear-only validation MSE: {self.linear_validation_mse_:.6f}")
        lines.append(f"Validation MSE: {self.validation_mse_:.6f}")
        linear_ops = int(1 + len(self.linear_features_))
        tree_ops = int(2 * max(0, self._leaf_count(self.residual_tree_) - 1) + 2) if self.tree_weight_ != 0.0 else 0
        ops = linear_ops + tree_ops
        lines.append(f"Approx arithmetic operations: {ops}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
ResidualTreeSparseLinearRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "ResidualTreeSparseLinearV1"
model_description = "Coefficient-mass sparse ridge backbone in raw-feature equation form plus a validation-shrunk 4-leaf residual tree for lightweight nonlinearity"
model_defs = [(model_shorthand_name, ResidualTreeSparseLinearRegressor())]

# Evaluation (do not edit anything below this line)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()

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
    model_name = model_defs[0][0]
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
    mean_rank = avg_rank.get(model_shorthand_name, float("nan"))

    upsert_overall_results([{
        "commit":                             git_hash,
        "mean_rank":                          f"{mean_rank:.2f}" if not np.isnan(mean_rank) else "nan",
        "frac_interpretability_tests_passed": f"{n_passed / total:.4f}" if total > 0 else "nan",
        "status":                             "",
        "model_name":                         model_shorthand_name,
        "description":                        model_description,
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
