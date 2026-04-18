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
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class RuleGatedSparseLinearRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse linear regressor with one optional gating split.

    Model family:
      - global sparse linear equation
      - or one threshold gate: if x_j <= t use sparse linear equation A, else B

    The solver is greedy forward selection with least-squares refits, followed by
    coefficient quantization for easier manual simulation.
    """

    def __init__(
        self,
        max_global_terms=5,
        max_terms_per_side=3,
        n_split_features=4,
        split_quantiles=(0.35, 0.5, 0.65),
        min_leaf_frac=0.2,
        min_relative_gain=0.01,
        split_gain=0.03,
        coef_tol=1e-6,
        quantization=0.02,
    ):
        self.max_global_terms = max_global_terms
        self.max_terms_per_side = max_terms_per_side
        self.n_split_features = n_split_features
        self.split_quantiles = split_quantiles
        self.min_leaf_frac = min_leaf_frac
        self.min_relative_gain = min_relative_gain
        self.split_gain = split_gain
        self.coef_tol = coef_tol
        self.quantization = quantization

    @staticmethod
    def _round_to_step(values, step):
        if step <= 0:
            return values
        return step * np.round(values / step)

    def _fit_sparse_linear(self, Xc, target, max_terms):
        n_samples, n_features = Xc.shape
        max_terms = min(int(max_terms), n_features)
        if max_terms <= 0:
            return [], np.zeros(0, dtype=float), np.zeros(n_samples, dtype=float), float(np.mean(target ** 2))

        x_norms = np.linalg.norm(Xc, axis=0) + 1e-12
        selected = []
        current_pred = np.zeros(n_samples, dtype=float)
        prev_mse = float(np.mean((target - current_pred) ** 2))

        for step in range(max_terms):
            resid = target - current_pred
            resid_norm = float(np.linalg.norm(resid)) + 1e-12
            corrs = np.abs((Xc.T @ resid) / (x_norms * resid_norm))

            best_feat = None
            best_pred = None
            best_mse = prev_mse
            best_beta = None

            for feat in np.argsort(corrs)[::-1]:
                feat = int(feat)
                if feat in selected:
                    continue
                trial_feats = selected + [feat]
                D = Xc[:, trial_feats]
                beta, *_ = np.linalg.lstsq(D, target, rcond=None)
                trial_pred = D @ beta
                mse = float(np.mean((target - trial_pred) ** 2))
                if mse < best_mse - 1e-12:
                    best_mse = mse
                    best_feat = feat
                    best_pred = trial_pred
                    best_beta = np.asarray(beta, dtype=float)
                break

            if best_feat is None:
                break

            rel_gain = (prev_mse - best_mse) / (abs(prev_mse) + 1e-12)
            if step > 0 and rel_gain < self.min_relative_gain:
                break

            selected.append(best_feat)
            current_pred = best_pred
            prev_mse = best_mse

        if not selected:
            return [], np.zeros(0, dtype=float), np.zeros(n_samples, dtype=float), float(np.mean(target ** 2))

        D = Xc[:, selected]
        beta, *_ = np.linalg.lstsq(D, target, rcond=None)
        beta = np.asarray(beta, dtype=float)

        if self.quantization > 0:
            beta = self._round_to_step(beta, float(self.quantization))

        keep = np.abs(beta) > self.coef_tol
        if not np.all(keep):
            selected = [f for f, k in zip(selected, keep) if k]
            beta = beta[keep]
            if selected:
                D = Xc[:, selected]
                beta, *_ = np.linalg.lstsq(D, target, rcond=None)
                beta = np.asarray(beta, dtype=float)
                if self.quantization > 0:
                    beta = self._round_to_step(beta, float(self.quantization))
                keep = np.abs(beta) > self.coef_tol
                selected = [f for f, k in zip(selected, keep) if k]
                beta = beta[keep]

        if not selected:
            return [], np.zeros(0, dtype=float), np.zeros(n_samples, dtype=float), float(np.mean(target ** 2))

        pred = Xc[:, selected] @ beta
        mse = float(np.mean((target - pred) ** 2))
        return selected, beta, pred, mse

    def _raw_from_centered(self, coef_centered):
        intercept_raw = self.y_mean_ - float(np.dot(self.x_mean_, coef_centered))
        if self.quantization > 0:
            intercept_raw = float(self._round_to_step(np.array([intercept_raw]), float(self.quantization))[0])
        return intercept_raw, coef_centered

    def _fit_branch(self, Xc, target, mask, max_terms):
        if mask.sum() == 0:
            coef = np.zeros(self.n_features_in_, dtype=float)
            intercept, coef = self._raw_from_centered(coef)
            return intercept, coef, np.full(Xc.shape[0], intercept, dtype=float), float(np.mean((self.y_mean_ + target - intercept) ** 2))

        selected, beta, _, _ = self._fit_sparse_linear(Xc[mask], target[mask], max_terms=max_terms)
        coef_centered = np.zeros(self.n_features_in_, dtype=float)
        for feat, b in zip(selected, beta):
            coef_centered[int(feat)] = float(b)
        intercept, coef_raw = self._raw_from_centered(coef_centered)
        pred_all = intercept + Xc @ coef_raw + np.dot(self.x_mean_, coef_raw)
        mse_mask = float(np.mean((self.y_mean_ + target[mask] - pred_all[mask]) ** 2))
        return intercept, coef_raw, pred_all, mse_mask

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n_samples, n_features = X.shape

        self.n_features_in_ = n_features
        self.x_mean_ = X.mean(axis=0)
        self.y_mean_ = float(y.mean())
        Xc = X - self.x_mean_
        target = y - self.y_mean_

        selected, beta, _, _ = self._fit_sparse_linear(Xc, target, max_terms=self.max_global_terms)
        coef_centered = np.zeros(n_features, dtype=float)
        for feat, b in zip(selected, beta):
            coef_centered[int(feat)] = float(b)
        self.global_intercept_, self.global_coef_ = self._raw_from_centered(coef_centered)
        global_pred = self.global_intercept_ + X @ self.global_coef_
        global_mse = float(np.mean((y - global_pred) ** 2))

        self.use_split_ = False
        self.split_feature_ = None
        self.split_threshold_ = None
        self.left_intercept_ = self.global_intercept_
        self.right_intercept_ = self.global_intercept_
        self.left_coef_ = self.global_coef_.copy()
        self.right_coef_ = self.global_coef_.copy()

        # Screen potential split features by centered correlation with target.
        x_norms = np.linalg.norm(Xc, axis=0) + 1e-12
        target_norm = float(np.linalg.norm(target)) + 1e-12
        corrs = np.abs((Xc.T @ target) / (x_norms * target_norm))
        split_features = [int(i) for i in np.argsort(corrs)[::-1][: min(int(self.n_split_features), n_features)]]

        min_leaf = max(20, int(float(self.min_leaf_frac) * n_samples))
        best_split = None
        best_mse = global_mse

        for feat in split_features:
            thresholds = np.unique(np.quantile(X[:, feat], self.split_quantiles))
            for thr in thresholds:
                left_mask = X[:, feat] <= float(thr)
                right_mask = ~left_mask
                if left_mask.sum() < min_leaf or right_mask.sum() < min_leaf:
                    continue

                left_intercept, left_coef, _, _ = self._fit_branch(Xc, target, left_mask, self.max_terms_per_side)
                right_intercept, right_coef, _, _ = self._fit_branch(Xc, target, right_mask, self.max_terms_per_side)

                pred = np.empty(n_samples, dtype=float)
                pred[left_mask] = left_intercept + X[left_mask] @ left_coef
                pred[right_mask] = right_intercept + X[right_mask] @ right_coef
                mse = float(np.mean((y - pred) ** 2))

                if mse < best_mse:
                    best_mse = mse
                    best_split = (
                        int(feat),
                        float(thr),
                        float(left_intercept),
                        left_coef.copy(),
                        float(right_intercept),
                        right_coef.copy(),
                    )

        if best_split is not None:
            rel_gain = (global_mse - best_mse) / (abs(global_mse) + 1e-12)
            if rel_gain >= self.split_gain:
                self.use_split_ = True
                (
                    self.split_feature_,
                    self.split_threshold_,
                    self.left_intercept_,
                    self.left_coef_,
                    self.right_intercept_,
                    self.right_coef_,
                ) = best_split

        # Expose unified coefficients for compatibility.
        if self.use_split_:
            coef_avg = 0.5 * (self.left_coef_ + self.right_coef_)
            self.intercept_ = float(0.5 * (self.left_intercept_ + self.right_intercept_))
            self.coef_ = coef_avg
            self.feature_importance_ = np.maximum(np.abs(self.left_coef_), np.abs(self.right_coef_))
        else:
            self.intercept_ = self.global_intercept_
            self.coef_ = self.global_coef_.copy()
            self.feature_importance_ = np.abs(self.coef_)

        self.selected_features_ = sorted(int(i) for i in np.where(self.feature_importance_ > self.coef_tol)[0])
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "coef_", "use_split_"])
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if not self.use_split_:
            return self.global_intercept_ + X @ self.global_coef_

        pred = np.empty(X.shape[0], dtype=float)
        left_mask = X[:, self.split_feature_] <= self.split_threshold_
        right_mask = ~left_mask
        pred[left_mask] = self.left_intercept_ + X[left_mask] @ self.left_coef_
        pred[right_mask] = self.right_intercept_ + X[right_mask] @ self.right_coef_
        return pred

    @staticmethod
    def _format_equation(intercept, coef, tol):
        active = [(i, c) for i, c in enumerate(coef) if abs(float(c)) > tol]
        active = sorted(active, key=lambda t: abs(float(t[1])), reverse=True)
        if not active:
            return f"y = {intercept:+.4f}"
        terms = [f"{intercept:+.4f}"]
        terms += [f"{float(c):+.4f}*x{i}" for i, c in active]
        return "y = " + " ".join(terms)

    def __str__(self):
        check_is_fitted(self, ["intercept_", "coef_", "feature_importance_"])
        lines = [
            "Rule-Gated Sparse Linear Regressor",
            "Prediction uses at most one threshold rule and sparse linear arithmetic.",
            "",
            "Computation recipe:",
        ]
        if self.use_split_:
            lines.append(f"  1) Check if x{self.split_feature_} <= {self.split_threshold_:.4f}.")
            lines.append("  2) Use the matching branch equation below.")
            lines.append(f"     LEFT  branch equation: {self._format_equation(self.left_intercept_, self.left_coef_, self.coef_tol)}")
            lines.append(f"     RIGHT branch equation: {self._format_equation(self.right_intercept_, self.right_coef_, self.coef_tol)}")
        else:
            lines.append(f"  1) Use equation: {self._format_equation(self.global_intercept_, self.global_coef_, self.coef_tol)}")

        active = sorted(int(i) for i in np.where(self.feature_importance_ > self.coef_tol)[0])
        if active:
            lines.append("")
            lines.append("Active features: " + ", ".join(f"x{i}" for i in active))
        if len(active) < self.n_features_in_ and self.n_features_in_ <= 30:
            inactive = [f"x{i}" for i in range(self.n_features_in_) if i not in set(active)]
            lines.append("Inactive features (zero coefficient): " + ", ".join(inactive))
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
RuleGatedSparseLinearRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "RuleGateSparseLin_v1"
model_description = "Greedy sparse linear regressor with one optional threshold gate selecting between two compact branch equations"
model_defs = [(model_shorthand_name, RuleGatedSparseLinearRegressor())]


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
