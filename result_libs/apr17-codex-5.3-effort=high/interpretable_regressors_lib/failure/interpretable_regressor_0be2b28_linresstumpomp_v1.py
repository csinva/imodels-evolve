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


class LinearResidualStumpRegressor(BaseEstimator, RegressorMixin):
    """
    Compact additive regressor:
      1) sparse linear core found by greedy OMP-style selection
      2) a few residual decision stumps for nonlinear corrections

    This keeps the printed model short enough for LLM simulation while adding
    simple threshold behavior absent in pure linear models.
    """

    def __init__(
        self,
        max_linear_terms=5,
        max_stumps=3,
        feature_screen=20,
        quantiles=(0.15, 0.30, 0.50, 0.70, 0.85),
        min_relative_gain=1e-3,
        min_leaf_frac=0.05,
    ):
        self.max_linear_terms = max_linear_terms
        self.max_stumps = max_stumps
        self.feature_screen = feature_screen
        self.quantiles = quantiles
        self.min_relative_gain = min_relative_gain
        self.min_leaf_frac = min_leaf_frac

    def _fit_sparse_linear(self, X, target):
        n_samples, n_features = X.shape
        if n_features == 0:
            return np.zeros(0, dtype=float), []

        col_norms = np.linalg.norm(X, axis=0) + 1e-12
        selected = []
        current_pred = np.zeros(n_samples, dtype=float)
        prev_mse = float(np.mean((target - current_pred) ** 2))

        max_rounds = min(self.max_linear_terms, n_features)
        for _ in range(max_rounds):
            resid = target - current_pred
            scores = np.abs(X.T @ resid) / col_norms
            if selected:
                scores[np.array(selected, dtype=int)] = -np.inf

            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            if not np.isfinite(best_score) or best_score <= 1e-12:
                break

            trial_selected = selected + [best_idx]
            W = X[:, trial_selected]
            beta, *_ = np.linalg.lstsq(W, target, rcond=None)
            trial_pred = W @ beta
            mse = float(np.mean((target - trial_pred) ** 2))

            rel_gain = (prev_mse - mse) / (abs(prev_mse) + 1e-12)
            if selected and rel_gain < self.min_relative_gain:
                break

            selected = trial_selected
            current_pred = trial_pred
            prev_mse = mse

        coef = np.zeros(n_features, dtype=float)
        if selected:
            W = X[:, selected]
            beta, *_ = np.linalg.lstsq(W, target, rcond=None)
            for i, feat_idx in enumerate(selected):
                coef[int(feat_idx)] = float(beta[i])
        return coef, selected

    @staticmethod
    def _candidate_thresholds(x, quantiles):
        if x.size == 0:
            return []
        qs = np.quantile(x, quantiles)
        uniq = np.unique(np.asarray(qs, dtype=float))
        return [float(v) for v in uniq]

    def _find_best_stump(self, X, resid):
        n_samples, n_features = X.shape
        if n_features == 0:
            return None, float(np.mean(resid ** 2))

        y_norm = float(np.linalg.norm(resid)) + 1e-12
        x_centered = X - X.mean(axis=0, keepdims=True)
        x_norms = np.linalg.norm(x_centered, axis=0) + 1e-12
        corrs = np.abs((x_centered.T @ resid) / (x_norms * y_norm))
        top_k = min(self.feature_screen, n_features)
        candidate_features = np.argsort(corrs)[::-1][:top_k]

        min_leaf = max(5, int(self.min_leaf_frac * n_samples))
        base_mse = float(np.mean(resid ** 2))
        best_mse = base_mse
        best = None

        for feat_idx in candidate_features:
            feat_idx = int(feat_idx)
            xj = X[:, feat_idx]
            for threshold in self._candidate_thresholds(xj, self.quantiles):
                left = xj <= threshold
                n_left = int(left.sum())
                n_right = n_samples - n_left
                if n_left < min_leaf or n_right < min_leaf:
                    continue
                left_val = float(resid[left].mean())
                right_val = float(resid[~left].mean())
                contrib = np.where(left, left_val, right_val)
                mse = float(np.mean((resid - contrib) ** 2))
                if mse < best_mse - 1e-12:
                    best_mse = mse
                    best = (feat_idx, float(threshold), left_val, right_val, contrib)

        return best, best_mse

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        self.n_features_in_ = X.shape[1]

        self.intercept_ = float(y.mean())
        target = y - self.intercept_

        self.linear_coef_, self.linear_selected_ = self._fit_sparse_linear(X, target)
        current_pred = X @ self.linear_coef_
        prev_mse = float(np.mean((target - current_pred) ** 2))

        self.stumps_ = []
        for _ in range(self.max_stumps):
            resid = target - current_pred
            best, best_mse = self._find_best_stump(X, resid)
            if best is None:
                break
            rel_gain = (prev_mse - best_mse) / (abs(prev_mse) + 1e-12)
            if rel_gain < self.min_relative_gain:
                break

            feat_idx, threshold, left_val, right_val, contrib = best
            self.stumps_.append(
                {
                    "feature": int(feat_idx),
                    "threshold": float(threshold),
                    "left": float(left_val),
                    "right": float(right_val),
                }
            )
            current_pred = current_pred + contrib
            prev_mse = best_mse

        feat_importance = np.abs(self.linear_coef_)
        for stump in self.stumps_:
            feat_idx = stump["feature"]
            feat_importance[feat_idx] += abs(stump["right"] - stump["left"])
        self.feature_importance_ = feat_importance
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "linear_coef_", "stumps_"])
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        pred = self.intercept_ + X @ self.linear_coef_
        for stump in self.stumps_:
            feat_idx = stump["feature"]
            threshold = stump["threshold"]
            left_val = stump["left"]
            right_val = stump["right"]
            pred = pred + np.where(X[:, feat_idx] <= threshold, left_val, right_val)
        return pred

    def __str__(self):
        check_is_fitted(self, ["intercept_", "linear_coef_", "stumps_", "feature_importance_"])
        lines = [
            "Linear-Residual Stump Regressor",
            "Prediction rule:",
            "y = intercept + sum_j (coef_j * xj) + sum_k stump_k(x)",
            f"intercept = {self.intercept_:+.6f}",
            "",
            "Linear terms:",
        ]

        nz_idx = np.where(np.abs(self.linear_coef_) > 1e-10)[0]
        if nz_idx.size == 0:
            lines.append("  (none)")
        else:
            order = nz_idx[np.argsort(np.abs(self.linear_coef_[nz_idx]))[::-1]]
            for feat_idx in order:
                coef = float(self.linear_coef_[feat_idx])
                lines.append(f"  {coef:+.6f} * x{int(feat_idx)}")

        lines.append("")
        lines.append("Residual stump rules (each adds a constant):")
        if not self.stumps_:
            lines.append("  (none)")
        else:
            for i, stump in enumerate(self.stumps_, 1):
                feat_idx = stump["feature"]
                threshold = stump["threshold"]
                left_val = stump["left"]
                right_val = stump["right"]
                lines.append(
                    f"  rule{i}: if x{feat_idx} <= {threshold:.6f}, add {left_val:+.6f}; else add {right_val:+.6f}"
                )

        lines.append("")
        lines.append(f"Model size: {len(nz_idx)} linear terms + {len(self.stumps_)} stump rules")
        used = set(int(i) for i in nz_idx.tolist())
        used.update(stump["feature"] for stump in self.stumps_)
        if used:
            lines.append("Active features: " + ", ".join(f"x{i}" for i in sorted(used)))
        unused = [f"x{i}" for i in range(self.n_features_in_) if i not in used]
        if unused and len(unused) <= 30:
            lines.append("Unused features (zero direct effect): " + ", ".join(unused))
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
LinearResidualStumpRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "LinResStumpOMP_v1"
model_description = "Sparse linear OMP core plus a few residual decision-stump corrections for compact nonlinear additivity"
model_defs = [(model_shorthand_name, LinearResidualStumpRegressor())]


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
