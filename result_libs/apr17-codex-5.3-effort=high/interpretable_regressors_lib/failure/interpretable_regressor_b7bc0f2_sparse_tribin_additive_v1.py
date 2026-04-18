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


class SparseTriBinAdditiveRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse additive regressor with two components:
      1) sparse ridge-linear backbone
      2) a few residual single-feature tri-bin corrections

    Each tri-bin correction maps one feature into three regions using quantiles.
    """

    def __init__(
        self,
        ridge=0.2,
        max_linear_terms=6,
        min_linear_rel=0.08,
        max_corrections=2,
        screen_features=16,
        bin_quantiles=(0.33, 0.67),
        correction_shrink=0.9,
        min_correction_gain=1e-4,
        coef_tol=1e-10,
        meaningful_rel=0.12,
    ):
        self.ridge = ridge
        self.max_linear_terms = max_linear_terms
        self.min_linear_rel = min_linear_rel
        self.max_corrections = max_corrections
        self.screen_features = screen_features
        self.bin_quantiles = bin_quantiles
        self.correction_shrink = correction_shrink
        self.min_correction_gain = min_correction_gain
        self.coef_tol = coef_tol
        self.meaningful_rel = meaningful_rel

    @staticmethod
    def _ridge_linear_fit(Z, y, ridge):
        n = Z.shape[0]
        A = np.column_stack([np.ones(n, dtype=float), Z])
        reg = np.eye(A.shape[1], dtype=float)
        reg[0, 0] = 0.0
        lhs = A.T @ A + float(ridge) * reg
        rhs = A.T @ y
        try:
            sol = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        return float(sol[0]), np.asarray(sol[1:], dtype=float)

    @staticmethod
    def _piecewise_values(xj, t1, t2, values):
        out = np.empty_like(xj, dtype=float)
        m0 = xj <= t1
        m1 = (xj > t1) & (xj <= t2)
        m2 = xj > t2
        out[m0] = values[0]
        out[m1] = values[1]
        out[m2] = values[2]
        return out

    def _fit_one_tribin(self, X, residual, feature_idx):
        xj = X[:, feature_idx]
        q_low, q_high = self.bin_quantiles
        t1 = float(np.quantile(xj, q_low))
        t2 = float(np.quantile(xj, q_high))
        if not np.isfinite(t1) or not np.isfinite(t2) or t2 <= t1 + 1e-12:
            return None

        mask0 = xj <= t1
        mask1 = (xj > t1) & (xj <= t2)
        mask2 = xj > t2
        min_count = max(4, int(0.02 * len(xj)))
        if mask0.sum() < min_count or mask1.sum() < min_count or mask2.sum() < min_count:
            return None

        v0 = float(np.mean(residual[mask0]))
        v1 = float(np.mean(residual[mask1]))
        v2 = float(np.mean(residual[mask2]))
        values = np.array([v0, v1, v2], dtype=float) * float(self.correction_shrink)
        contrib = self._piecewise_values(xj, t1, t2, values)
        return {
            "feature": int(feature_idx),
            "t1": t1,
            "t2": t2,
            "values": values,
            "contrib": contrib,
        }

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        self.x_mean_ = X.mean(axis=0)
        self.x_scale_ = X.std(axis=0)
        self.x_scale_[self.x_scale_ < 1e-12] = 1.0
        Z = (X - self.x_mean_) / self.x_scale_

        intercept_std, beta_std = self._ridge_linear_fit(Z, y, self.ridge)
        abs_beta = np.abs(beta_std)
        keep = np.zeros(n_features, dtype=bool)
        if n_features > 0 and np.any(abs_beta > 0):
            k = min(int(self.max_linear_terms), n_features)
            top_idx = np.argsort(abs_beta)[::-1][:k]
            keep[top_idx] = True
            max_abs = float(np.max(abs_beta))
            keep &= abs_beta >= float(self.min_linear_rel) * max_abs
            if not np.any(keep):
                keep[top_idx[:1]] = True

        beta_std_sparse = np.zeros_like(beta_std)
        beta_std_sparse[keep] = beta_std[keep]

        self.linear_coef_ = beta_std_sparse / self.x_scale_
        self.linear_intercept_ = float(intercept_std - np.sum(beta_std_sparse * self.x_mean_ / self.x_scale_))
        pred = self.linear_intercept_ + X @ self.linear_coef_

        corrections = []
        min_abs_gain = float(self.min_correction_gain) * (float(np.var(y)) + 1e-12)
        for _ in range(int(self.max_corrections)):
            residual = y - pred
            base_mse = float(np.mean(residual ** 2))

            xc = X - X.mean(axis=0, keepdims=True)
            yc = residual - float(np.mean(residual))
            xnorm = np.linalg.norm(xc, axis=0) + 1e-12
            ynorm = float(np.linalg.norm(yc)) + 1e-12
            corr = np.abs((xc.T @ yc) / (xnorm * ynorm))
            corr[~np.isfinite(corr)] = 0.0

            screened = np.argsort(corr)[::-1][: min(int(self.screen_features), n_features)]
            best = None
            best_gain = 0.0
            for j in screened:
                candidate = self._fit_one_tribin(X, residual, int(j))
                if candidate is None:
                    continue
                trial_pred = pred + candidate["contrib"]
                gain = base_mse - float(np.mean((y - trial_pred) ** 2))
                if np.isfinite(gain) and gain > best_gain:
                    best_gain = gain
                    best = candidate

            if best is None or best_gain < min_abs_gain:
                break
            pred = pred + best["contrib"]
            corrections.append({
                "feature": best["feature"],
                "t1": float(best["t1"]),
                "t2": float(best["t2"]),
                "values": np.asarray(best["values"], dtype=float),
            })

        intercept_adjust = float(np.mean(y - pred))
        self.intercept_ = self.linear_intercept_ + intercept_adjust
        self.corrections_ = corrections

        self.feature_importance_ = np.abs(self.linear_coef_)
        for corr_term in self.corrections_:
            self.feature_importance_[corr_term["feature"]] += float(np.sum(np.abs(corr_term["values"])))
        self.selected_features_ = sorted(int(i) for i in np.where(self.feature_importance_ > self.coef_tol)[0])
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "linear_coef_", "corrections_"])
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        pred = self.intercept_ + X @ self.linear_coef_
        for corr in self.corrections_:
            pred = pred + self._piecewise_values(
                X[:, corr["feature"]],
                corr["t1"],
                corr["t2"],
                corr["values"],
            )
        return pred

    def __str__(self):
        check_is_fitted(self, ["intercept_", "linear_coef_", "corrections_", "feature_importance_"])
        lines = [
            "Sparse Tri-Bin Additive Regressor",
            "Exact prediction recipe:",
            "  1) Start with intercept.",
            "  2) Add each linear term c*xj.",
            "  3) For each tri-bin rule, add the bin value based on xj.",
            "  4) Return the final number.",
            f"intercept = {self.intercept_:+.5f}",
            "",
            "Linear terms:",
        ]

        linear_pairs = [(j, float(c)) for j, c in enumerate(self.linear_coef_) if abs(float(c)) > self.coef_tol]
        linear_pairs.sort(key=lambda t: abs(t[1]), reverse=True)
        if linear_pairs:
            for idx, (j, c) in enumerate(linear_pairs, 1):
                lines.append(f"  {idx:2d}. add ({c:+.5f}) * x{j}")
        else:
            lines.append("  (none)")

        lines.append("")
        lines.append("Tri-bin corrections:")
        if self.corrections_:
            for i, corr in enumerate(self.corrections_, 1):
                j = int(corr["feature"])
                t1 = float(corr["t1"])
                t2 = float(corr["t2"])
                v0, v1, v2 = [float(v) for v in corr["values"]]
                lines.append(f"  Rule {i} on x{j}:")
                lines.append(f"    if x{j} <= {t1:.5f}: add {v0:+.5f}")
                lines.append(f"    elif x{j} <= {t2:.5f}: add {v1:+.5f}")
                lines.append(f"    else: add {v2:+.5f}")
        else:
            lines.append("  (none)")

        active = [int(i) for i in self.selected_features_]
        lines.append("")
        lines.append("Features used: " + (", ".join(f"x{i}" for i in active) if active else "none"))
        if self.n_features_in_ <= 30 and len(active) < self.n_features_in_:
            active_set = set(active)
            zero_feats = [f"x{i}" for i in range(self.n_features_in_) if i not in active_set]
            lines.append("Zero-contribution features: " + ", ".join(zero_feats))

        if self.feature_importance_.size > 0:
            max_imp = float(np.max(self.feature_importance_))
            if max_imp > 0:
                thr = float(self.meaningful_rel) * max_imp
                meaningful = [f"x{i}" for i in range(self.n_features_in_) if self.feature_importance_[i] >= thr]
                lines.append(
                    "Meaningful features (>= "
                    f"{float(self.meaningful_rel):.2f} * max importance): "
                    + (", ".join(meaningful) if meaningful else "none")
                )
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseTriBinAdditiveRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseTriBinAdditive_v1"
model_description = "Sparse ridge linear backbone plus up to two single-feature tri-bin residual corrections"
model_defs = [(model_shorthand_name, SparseTriBinAdditiveRegressor())]


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
