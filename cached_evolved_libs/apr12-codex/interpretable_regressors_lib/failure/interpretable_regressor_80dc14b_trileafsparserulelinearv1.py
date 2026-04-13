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
from sklearn.linear_model import LassoCV
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class TriLeafSparseRuleLinearRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse linear backbone plus a tiny residual rule tree (at most 3 leaves).

    Final form:
      y = b + sum_j w_j x_j + r(x)
    where r(x) is a compact two-level threshold rule with region-specific offsets.
    """

    def __init__(
        self,
        max_terms=10,
        top_split_features=10,
        split_quantiles=(0.15, 0.3, 0.5, 0.7, 0.85),
        min_leaf_frac=0.08,
        min_gain_frac=0.01,
        coef_tol=1e-3,
    ):
        self.max_terms = max_terms
        self.top_split_features = top_split_features
        self.split_quantiles = split_quantiles
        self.min_leaf_frac = min_leaf_frac
        self.min_gain_frac = min_gain_frac
        self.coef_tol = coef_tol

    def _impute(self, X):
        X = np.asarray(X, dtype=float).copy()
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(self.feature_medians_, np.where(bad)[1])
        return X

    def _fit_sparse_linear(self, X, y):
        x_mean = np.mean(X, axis=0)
        x_std = np.std(X, axis=0)
        x_std[x_std < 1e-12] = 1.0
        Xs = (X - x_mean) / x_std

        lasso = LassoCV(cv=3, n_alphas=48, max_iter=7000, random_state=42)
        lasso.fit(Xs, y)
        coef_dense = lasso.coef_ / x_std
        intercept_dense = float(lasso.intercept_ - np.dot(coef_dense, x_mean))

        active = np.where(np.abs(coef_dense) > 1e-8)[0]
        if active.size == 0:
            active = np.array([int(np.argmax(np.abs(coef_dense)))], dtype=int)
        if active.size > int(self.max_terms):
            order = np.argsort(np.abs(coef_dense[active]))[::-1]
            active = active[order[: int(self.max_terms)]]

        coef = np.zeros(X.shape[1], dtype=float)
        A = np.column_stack([np.ones(X.shape[0]), X[:, active]])
        beta, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        intercept = float(beta[0])
        coef[active] = beta[1:]

        pred_sparse = intercept + X @ coef
        mse_sparse = float(np.mean((y - pred_sparse) ** 2))

        pred_dense = intercept_dense + X @ coef_dense
        mse_dense = float(np.mean((y - pred_dense) ** 2))

        if mse_dense < mse_sparse * 0.95:
            return intercept_dense, coef_dense
        return intercept, coef

    @staticmethod
    def _offset_mse(residual, mask):
        if mask.sum() == 0:
            return np.inf, 0.0
        c = float(np.mean(residual[mask]))
        mse = float(np.mean((residual[mask] - c) ** 2))
        return mse, c

    def _find_best_split(self, X, residual, mask, candidate_features, min_leaf):
        idx = np.where(mask)[0]
        if idx.size < 2 * min_leaf:
            return None
        best = None
        n = float(idx.size)
        for j in candidate_features:
            xj = X[idx, j]
            thresholds = np.unique(np.quantile(xj, self.split_quantiles))
            for thr in thresholds:
                left_local = xj <= thr
                n_left = int(left_local.sum())
                n_right = int(idx.size - n_left)
                if n_left < min_leaf or n_right < min_leaf:
                    continue
                left_idx = idx[left_local]
                right_idx = idx[~left_local]
                mse_l, c_l = self._offset_mse(residual, np.isin(np.arange(X.shape[0]), left_idx))
                mse_r, c_r = self._offset_mse(residual, np.isin(np.arange(X.shape[0]), right_idx))
                weighted_mse = (n_left * mse_l + n_right * mse_r) / n
                if best is None or weighted_mse < best["mse"]:
                    best = {
                        "feature": int(j),
                        "threshold": float(thr),
                        "mse": float(weighted_mse),
                        "left_const": float(c_l),
                        "right_const": float(c_r),
                    }
        return best

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.feature_medians_ = np.nanmedian(X, axis=0)
        X = self._impute(X)

        self.base_intercept_, self.linear_coef_ = self._fit_sparse_linear(X, y)
        base_pred = self.base_intercept_ + X @ self.linear_coef_
        residual = y - base_pred
        base_res_mse = float(np.mean(residual ** 2))

        corr_scores = np.array(
            [
                abs(float(np.corrcoef(X[:, j], residual)[0, 1])) if np.std(X[:, j]) > 1e-12 else 0.0
                for j in range(self.n_features_in_)
            ],
            dtype=float,
        )
        candidate_features = np.argsort(corr_scores)[::-1][: min(int(self.top_split_features), self.n_features_in_)]

        n = X.shape[0]
        min_leaf = max(10, int(self.min_leaf_frac * n))

        root_mask = np.ones(n, dtype=bool)
        root_split = self._find_best_split(X, residual, root_mask, candidate_features, min_leaf)

        self.has_rules_ = False
        self.root_feature_ = -1
        self.root_threshold_ = 0.0
        self.left_const_ = 0.0
        self.right_const_ = 0.0
        self.second_on_left_ = False
        self.second_feature_ = -1
        self.second_threshold_ = 0.0
        self.second_low_const_ = 0.0
        self.second_high_const_ = 0.0

        if root_split is None:
            self.linear_coef_[np.abs(self.linear_coef_) < self.coef_tol] = 0.0
            self.feature_importance_ = np.abs(self.linear_coef_)
            self.selected_feature_order_ = np.argsort(self.feature_importance_)[::-1]
            return self

        best_mse = root_split["mse"]
        best_struct = ("root_only", root_split, None, None)

        root_left = X[:, root_split["feature"]] <= root_split["threshold"]
        root_right = ~root_left

        for branch_name, branch_mask in [("left", root_left), ("right", root_right)]:
            if int(branch_mask.sum()) < 2 * min_leaf:
                continue
            split2 = self._find_best_split(X, residual, branch_mask, candidate_features, min_leaf)
            if split2 is None:
                continue

            other_mask = ~branch_mask
            mse_other, c_other = self._offset_mse(residual, other_mask)

            idx = np.where(branch_mask)[0]
            branch_x = X[idx, split2["feature"]]
            low_local = branch_x <= split2["threshold"]
            low_idx = idx[low_local]
            high_idx = idx[~low_local]
            mse_low, c_low = self._offset_mse(residual, np.isin(np.arange(n), low_idx))
            mse_high, c_high = self._offset_mse(residual, np.isin(np.arange(n), high_idx))

            combined = (
                float(low_idx.size) * mse_low
                + float(high_idx.size) * mse_high
                + float(other_mask.sum()) * mse_other
            ) / float(n)

            if combined < best_mse:
                best_mse = combined
                best_struct = (
                    "two_level",
                    root_split,
                    (branch_name, split2),
                    (float(c_other), float(c_low), float(c_high)),
                )

        gain = base_res_mse - best_mse
        if gain > float(self.min_gain_frac) * (np.var(y) + 1e-12):
            self.has_rules_ = True
            root = best_struct[1]
            self.root_feature_ = int(root["feature"])
            self.root_threshold_ = float(root["threshold"])
            self.left_const_ = float(root["left_const"])
            self.right_const_ = float(root["right_const"])

            if best_struct[0] == "two_level":
                branch_name, split2 = best_struct[2]
                c_other, c_low, c_high = best_struct[3]
                self.second_on_left_ = bool(branch_name == "left")
                self.second_feature_ = int(split2["feature"])
                self.second_threshold_ = float(split2["threshold"])
                self.second_low_const_ = float(c_low)
                self.second_high_const_ = float(c_high)
                if self.second_on_left_:
                    self.left_const_ = 0.0
                    self.right_const_ = float(c_other)
                else:
                    self.left_const_ = float(c_other)
                    self.right_const_ = 0.0

        self.linear_coef_[np.abs(self.linear_coef_) < self.coef_tol] = 0.0
        for attr in ["left_const_", "right_const_", "second_low_const_", "second_high_const_"]:
            if abs(getattr(self, attr)) < self.coef_tol:
                setattr(self, attr, 0.0)

        self.feature_importance_ = np.abs(self.linear_coef_)
        if self.has_rules_:
            self.feature_importance_[self.root_feature_] += abs(self.left_const_) + abs(self.right_const_)
            if self.second_feature_ >= 0:
                self.feature_importance_[self.second_feature_] += abs(self.second_low_const_) + abs(self.second_high_const_)
        self.selected_feature_order_ = np.argsort(self.feature_importance_)[::-1]
        return self

    def _residual_rule(self, X):
        out = np.zeros(X.shape[0], dtype=float)
        if not self.has_rules_:
            return out
        root_left = X[:, self.root_feature_] <= self.root_threshold_
        out[root_left] += self.left_const_
        out[~root_left] += self.right_const_

        if self.second_feature_ >= 0:
            if self.second_on_left_:
                branch = root_left
            else:
                branch = ~root_left
            idx = np.where(branch)[0]
            if idx.size:
                low = X[idx, self.second_feature_] <= self.second_threshold_
                out[idx[low]] += self.second_low_const_
                out[idx[~low]] += self.second_high_const_
        return out

    def predict(self, X):
        check_is_fitted(self, ["base_intercept_", "linear_coef_", "has_rules_"])
        X = self._impute(X)
        return self.base_intercept_ + X @ self.linear_coef_ + self._residual_rule(X)

    def __str__(self):
        check_is_fitted(self, ["base_intercept_", "linear_coef_", "has_rules_"])
        lines = [
            "TriLeafSparseRuleLinearRegressor",
            "Exact prediction equation:",
            f"  base = {self.base_intercept_:+.6f}",
        ]

        active = np.where(np.abs(self.linear_coef_) >= self.coef_tol)[0]
        for j in active[np.argsort(np.abs(self.linear_coef_[active]))[::-1]]:
            lines.append(f"       {self.linear_coef_[j]:+.6f} * x{j}")
        if active.size == 0:
            lines.append("       (no active linear terms)")

        if not self.has_rules_:
            lines.append("  residual_rule = 0")
        elif self.second_feature_ < 0:
            lines.append(
                f"  residual_rule = {self.left_const_:+.6f} if x{self.root_feature_} <= {self.root_threshold_:+.6f} "
                f"else {self.right_const_:+.6f}"
            )
        else:
            if self.second_on_left_:
                lines.append(
                    f"  if x{self.root_feature_} <= {self.root_threshold_:+.6f}: "
                    f"residual_rule = ({self.second_low_const_:+.6f} if x{self.second_feature_} <= {self.second_threshold_:+.6f} "
                    f"else {self.second_high_const_:+.6f})"
                )
                lines.append(f"  else: residual_rule = {self.right_const_:+.6f}")
            else:
                lines.append(f"  if x{self.root_feature_} <= {self.root_threshold_:+.6f}: residual_rule = {self.left_const_:+.6f}")
                lines.append(
                    f"  else: residual_rule = ({self.second_low_const_:+.6f} if x{self.second_feature_} <= {self.second_threshold_:+.6f} "
                    f"else {self.second_high_const_:+.6f})"
                )

        lines.append("  prediction = base + residual_rule")
        lines.append("")
        lines.append("Feature summary (sorted by total attributed effect):")
        for j in self.selected_feature_order_[: min(12, self.n_features_in_)]:
            lines.append(f"  x{int(j)}: linear={self.linear_coef_[int(j)]:+.6f}, importance={self.feature_importance_[int(j)]:.6f}")
        inactive = [f"x{j}" for j in range(self.n_features_in_) if self.feature_importance_[j] < self.coef_tol]
        if inactive:
            lines.append("Features with near-zero effect: " + ", ".join(inactive))
        lines.append("To simulate: compute base from linear terms, then add residual_rule from the threshold conditions.")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
TriLeafSparseRuleLinearRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "TriLeafSparseRuleLinearV1"
model_description = "Sparse global linear equation plus compact two-level residual threshold rules with at most three regions"
model_defs = [(model_shorthand_name, TriLeafSparseRuleLinearRegressor())]

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
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
