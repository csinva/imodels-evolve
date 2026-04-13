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


class SparseQuantileGAMRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse additive piecewise-linear regressor.

    For each feature j, we build an additive shape with basis terms:
      z_j, relu(z_j-k1), relu(z_j-k2), relu(z_j-k3)
    where z_j is robustly standardized x_j and k's are global knots.

    Then we:
    1) fit ridge with GCV over all basis terms,
    2) keep only the most important feature blocks,
    3) refit ridge on the reduced additive model.
    """

    def __init__(
        self,
        alpha_grid=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
        knots=(-1.0, 0.0, 1.0),
        max_active_features=8,
        min_group_fraction=0.04,
        coef_prune_tol=1e-4,
        coef_display_tol=0.03,
    ):
        self.alpha_grid = alpha_grid
        self.knots = knots
        self.max_active_features = max_active_features
        self.min_group_fraction = min_group_fraction
        self.coef_prune_tol = coef_prune_tol
        self.coef_display_tol = coef_display_tol

    @staticmethod
    def _sanitize_alpha_grid(alpha_grid):
        a = np.asarray(alpha_grid, dtype=float)
        a = a[np.isfinite(a) & (a > 0)]
        if a.size == 0:
            return np.asarray([1.0], dtype=float)
        return np.unique(a)

    @staticmethod
    def _ridge_gcv(Z, y, alpha_grid):
        y_mean = float(np.mean(y))
        Z_mean = np.mean(Z, axis=0)
        Zc = Z - Z_mean
        yc = y - y_mean
        if Zc.shape[1] == 0:
            return y_mean, np.zeros(0, dtype=float), 1.0

        U, s, Vt = np.linalg.svd(Zc, full_matrices=False)
        Uy = U.T @ yc
        n = float(max(1, Z.shape[0]))

        best = None
        for alpha in alpha_grid:
            alpha = float(alpha)
            filt = s / (s * s + alpha)
            coef = Vt.T @ (filt * Uy)
            pred = y_mean + Zc @ coef
            mse = float(np.mean((y - pred) ** 2))
            dof = float(np.sum((s * s) / (s * s + alpha)))
            denom = max(1e-8, 1.0 - dof / n)
            gcv = mse / (denom * denom)
            if (best is None) or (gcv < best["gcv"]):
                best = {"gcv": gcv, "alpha": alpha, "coef": coef}

        coef = best["coef"]
        intercept = float(y_mean - np.dot(Z_mean, coef))
        return intercept, coef, float(best["alpha"])

    @staticmethod
    def _relu(v):
        return np.maximum(0.0, v)

    def _robust_scale(self, X):
        med = np.median(X, axis=0)
        q25 = np.quantile(X, 0.25, axis=0)
        q75 = np.quantile(X, 0.75, axis=0)
        iqr = q75 - q25
        iqr = np.where(np.abs(iqr) < 1e-8, 1.0, iqr)
        Z = (X - med) / iqr
        return Z, med, iqr

    def _feature_block(self, z_col):
        cols = [z_col]
        for k in self.knots_:
            cols.append(self._relu(z_col - float(k)))
        return np.column_stack(cols)

    def _build_design(self, Z, feature_ids):
        blocks = []
        block_slices = {}
        start = 0
        bsz = 1 + len(self.knots_)
        for j in feature_ids:
            blk = self._feature_block(Z[:, j])
            blocks.append(blk)
            block_slices[int(j)] = slice(start, start + bsz)
            start += bsz
        if blocks:
            D = np.concatenate(blocks, axis=1)
        else:
            D = np.zeros((Z.shape[0], 0), dtype=float)
        return D, block_slices

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        self.n_features_in_ = int(p)

        alpha_grid = self._sanitize_alpha_grid(self.alpha_grid)
        self.knots_ = tuple(float(k) for k in self.knots)

        if p == 0:
            self.intercept_ = float(np.mean(y))
            self.active_features_ = []
            self.feature_coefs_ = {}
            self.medians_ = np.zeros(0, dtype=float)
            self.iqrs_ = np.ones(0, dtype=float)
            self.alpha_ = 1.0
            self.training_mse_ = float(np.mean((y - self.intercept_) ** 2))
            self.is_fitted_ = True
            return self

        Z, med, iqr = self._robust_scale(X)
        D_full, slices_full = self._build_design(Z, np.arange(p))

        _, coef_1, _ = self._ridge_gcv(D_full, y, alpha_grid)

        # Score each feature block and keep strongest groups.
        group_scores = []
        for j in range(p):
            sl = slices_full[j]
            score = float(np.linalg.norm(coef_1[sl]))
            group_scores.append((score, j))
        group_scores.sort(reverse=True)

        max_active = int(max(1, min(p, self.max_active_features)))
        top_score = group_scores[0][0] if group_scores else 0.0
        min_frac = float(max(0.0, self.min_group_fraction))

        kept = []
        for score, j in group_scores[:max_active]:
            if top_score <= 0.0:
                break
            if score >= min_frac * top_score:
                kept.append(int(j))
        if not kept and group_scores:
            kept = [int(group_scores[0][1])]
        kept = sorted(kept)

        D_keep, slices_keep = self._build_design(Z, kept)
        intercept_2, coef_2, alpha_2 = self._ridge_gcv(D_keep, y, alpha_grid)

        # Prune tiny coefficients for readability.
        coef_2 = coef_2.copy()
        coef_2[np.abs(coef_2) < float(max(0.0, self.coef_prune_tol))] = 0.0

        feature_coefs = {}
        for j in kept:
            sl = slices_keep[j]
            feature_coefs[int(j)] = coef_2[sl].copy()

        pred = np.full(n, intercept_2, dtype=float)
        for j in kept:
            blk = self._feature_block(Z[:, j])
            pred += blk @ feature_coefs[j]

        self.intercept_ = float(intercept_2)
        self.alpha_ = float(alpha_2)
        self.active_features_ = kept
        self.feature_coefs_ = feature_coefs
        self.medians_ = med
        self.iqrs_ = iqr
        self.training_mse_ = float(np.mean((y - pred) ** 2))
        self.n_active_features_ = int(len(kept))
        self.is_fitted_ = True
        return self

    def _predict_feature(self, x_col, j):
        z = (x_col - self.medians_[j]) / self.iqrs_[j]
        coef = self.feature_coefs_.get(j)
        if coef is None:
            return np.zeros_like(z, dtype=float)
        out = coef[0] * z
        for t, k in enumerate(self.knots_):
            out += coef[t + 1] * self._relu(z - float(k))
        return out

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        X = np.asarray(X, dtype=float)
        y = np.full(X.shape[0], self.intercept_, dtype=float)
        for j in self.active_features_:
            y += self._predict_feature(X[:, j], j)
        return y

    def _feature_text(self, j):
        coef = self.feature_coefs_.get(j)
        if coef is None:
            return f"f{j}(x{j}) = 0"
        parts = [f"{coef[0]:+.4f}*z{j}"]
        for t, k in enumerate(self.knots_):
            ck = float(coef[t + 1])
            if abs(ck) < 1e-12:
                continue
            parts.append(f"{ck:+.4f}*max(0, z{j}-{float(k):.2f})")
        rhs = " ".join(parts) if parts else "0"
        med = float(self.medians_[j])
        iqr = float(self.iqrs_[j])
        return f"f{j}(x{j}) = {rhs}, where z{j}=(x{j}-{med:.4f})/{iqr:.4f}"

    def __str__(self):
        check_is_fitted(self, "is_fitted_")
        lines = ["Sparse Quantile-GAM Regressor:"]
        lines.append("  prediction = intercept + sum_j f_j(x_j) over active features")
        lines.append(f"  intercept: {self.intercept_:+.4f}")
        lines.append(f"  ridge alpha (GCV): {self.alpha_:.6f}")
        lines.append(f"  training MSE: {self.training_mse_:.6f}")
        lines.append(f"  active features: {self.n_active_features_}/{self.n_features_in_}")

        if not self.active_features_:
            return "\n".join(lines)

        # Show features ordered by total effect magnitude for readability.
        scored = []
        for j in self.active_features_:
            coef = self.feature_coefs_[j]
            scored.append((float(np.sum(np.abs(coef))), j))
        scored.sort(reverse=True)

        tol = float(max(0.0, self.coef_display_tol))
        lines.append(f"  shown feature shapes (sum|coef| >= {tol:.3f}):")
        shown_any = False
        for score, j in scored:
            if score < tol:
                continue
            lines.append(f"    {self._feature_text(j)}")
            shown_any = True
        if not shown_any:
            lines.append("    none above display threshold")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseQuantileGAMRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseQuantileGAM_v1"
model_description = "Sparse additive quantile-spline GAM with robust scaling, per-feature hinge basis blocks, and group-pruned GCV ridge refit"
model_defs = [(model_shorthand_name, SparseQuantileGAMRegressor())]


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

    std_tests = {t.__name__ for t in ALL_TESTS}
    hard_tests = {t.__name__ for t in HARD_TESTS}
    insight_tests = {t.__name__ for t in INSIGHT_TESTS}
    std_passed = sum(r["passed"] for r in interp_results if r["test"] in std_tests)
    hard_passed = sum(r["passed"] for r in interp_results if r["test"] in hard_tests)
    insight_passed = sum(r["passed"] for r in interp_results if r["test"] in insight_tests)
    print(f"[std {std_passed}/{len(std_tests)}  hard {hard_passed}/{len(hard_tests)}  insight {insight_passed}/{len(insight_tests)}]")
    print(f"total_seconds: {time.time() - t0:.1f}s")
