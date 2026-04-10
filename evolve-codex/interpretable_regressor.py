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
from sklearn.linear_model import ElasticNetCV
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class SparseAdaptiveKnotRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse linear + adaptive-knot absolute basis model.

    Model form:
      y = b + sum_j w_j * z_j + sum_k u_k * |z_{j_k} - t_k|

    where z_j are robustly scaled features. Knots t_k are selected from
    quantile candidates by residual correlation, then the full dictionary is
    sparsified with ElasticNetCV and final top terms are retained.
    """

    def __init__(
        self,
        max_nonlinear_features=6,
        knot_quantiles=(0.2, 0.35, 0.5, 0.65, 0.8),
        max_terms=12,
        min_keep_coef=3e-3,
        random_state=42,
    ):
        self.max_nonlinear_features = max_nonlinear_features
        self.knot_quantiles = knot_quantiles
        self.max_terms = max_terms
        self.min_keep_coef = min_keep_coef
        self.random_state = random_state

    @staticmethod
    def _as_2d_float(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def _impute(self, X):
        return np.where(np.isnan(X), self.feature_medians_, X)

    @staticmethod
    def _corr_scores(X, y):
        yc = y - y.mean()
        y_norm = np.linalg.norm(yc) + 1e-12
        scores = np.zeros(X.shape[1], dtype=float)
        for j in range(X.shape[1]):
            xj = X[:, j]
            xj = xj - xj.mean()
            scores[j] = abs(np.dot(xj, yc) / ((np.linalg.norm(xj) + 1e-12) * y_norm))
        return scores

    def _robust_scale(self, X):
        Z = (X - self.feature_medians_) / self.feature_scales_
        return np.clip(Z, -8.0, 8.0)

    def _build_basis(self, Z):
        cols = []
        specs = []

        # Linear terms on all scaled features.
        for j in range(Z.shape[1]):
            cols.append(Z[:, j])
            specs.append(("lin", int(j)))

        # Adaptive-knot absolute terms on a small screened subset.
        for j, knot in self.nonlinear_specs_:
            cols.append(np.abs(Z[:, j] - knot))
            specs.append(("absknot", int(j), float(knot)))

        if len(cols) == 0:
            return np.zeros((Z.shape[0], 1), dtype=float), [("bias_fallback",)]
        return np.column_stack(cols), specs

    def fit(self, X, y):
        X = self._as_2d_float(X)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape

        self.feature_medians_ = np.nanmedian(X, axis=0)
        self.feature_medians_ = np.where(np.isnan(self.feature_medians_), 0.0, self.feature_medians_)
        X_imp = self._impute(X)
        self.feature_scales_ = np.nanmedian(np.abs(X_imp - self.feature_medians_), axis=0)
        self.feature_scales_ = np.where((~np.isfinite(self.feature_scales_)) | (self.feature_scales_ < 1e-6), 1.0, self.feature_scales_)
        Z = self._robust_scale(X_imp)

        # Screen top features by linear correlation in robust-scaled space.
        corr = self._corr_scores(Z, y)
        order = np.argsort(-corr)
        top = [int(j) for j in order[: min(self.max_nonlinear_features, p)]]

        # Residual from dense linear fit; used to choose informative knot locations.
        A_lin = np.column_stack([np.ones(n), Z])
        coef_lin, *_ = np.linalg.lstsq(A_lin, y, rcond=None)
        resid = y - A_lin @ coef_lin

        self.nonlinear_specs_ = []
        for j in top:
            xj = Z[:, j]
            candidates = []
            for q in self.knot_quantiles:
                t = float(np.quantile(xj, q))
                phi = np.abs(xj - t)
                denom = np.linalg.norm(phi) + 1e-12
                score = abs(np.dot(phi, resid)) / denom
                candidates.append((score, t))
            best_t = max(candidates, key=lambda z: z[0])[1]
            self.nonlinear_specs_.append((j, best_t))

        # Fit sparse model on combined dictionary.
        B, self.basis_specs_ = self._build_basis(Z)
        self.b_mean_ = B.mean(axis=0)
        self.b_std_ = B.std(axis=0)
        self.b_std_ = np.where(self.b_std_ < 1e-8, 1.0, self.b_std_)
        Bs = (B - self.b_mean_) / self.b_std_

        self.selector_ = ElasticNetCV(
            l1_ratio=[0.6, 0.8, 0.95, 1.0],
            alphas=np.logspace(-3, 0, 25),
            cv=3,
            random_state=self.random_state,
            max_iter=10000,
            fit_intercept=True,
        )
        self.selector_.fit(Bs, y)
        coef = self.selector_.coef_.copy()
        keep = np.abs(coef) >= self.min_keep_coef

        # Keep only the strongest terms for compactness.
        if keep.sum() > self.max_terms:
            idx = np.argsort(-np.abs(coef))[: self.max_terms]
            new_keep = np.zeros_like(keep, dtype=bool)
            new_keep[idx] = True
            keep = new_keep

        # Debias with least squares on kept standardized terms.
        if keep.sum() == 0:
            self.final_coef_ = np.zeros(Bs.shape[1], dtype=float)
            self.final_intercept_ = float(y.mean())
            self.active_mask_ = keep
            self.n_features_in_ = p
            return self

        Bs_keep = Bs[:, keep]
        A = np.column_stack([np.ones(n), Bs_keep])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        self.final_intercept_ = float(beta[0])
        self.final_coef_ = np.zeros(Bs.shape[1], dtype=float)
        self.final_coef_[keep] = beta[1:]
        self.active_mask_ = keep
        self.n_features_in_ = p
        return self

    def predict(self, X):
        check_is_fitted(self, ["final_coef_", "final_intercept_", "basis_specs_", "b_mean_", "b_std_"])
        X = self._as_2d_float(X)
        X_imp = self._impute(X)
        Z = self._robust_scale(X_imp)
        B, _ = self._build_basis(Z)
        Bs = (B - self.b_mean_) / self.b_std_
        return self.final_intercept_ + Bs @ self.final_coef_

    def __str__(self):
        check_is_fitted(self, ["final_coef_", "basis_specs_", "active_mask_"])

        coef = self.final_coef_.ravel()
        eff_intercept = float(self.final_intercept_ - np.sum(coef * (self.b_mean_ / self.b_std_)))

        lines = [
            "SparseAdaptiveKnotRegressor",
            f"prediction = {eff_intercept:.6f}",
        ]
        active_terms = []
        for idx, (c, spec, use) in enumerate(zip(coef, self.basis_specs_, self.active_mask_)):
            if not use:
                continue
            scale = c / self.b_std_[idx]

            if spec[0] == "lin":
                term = f"x{spec[1]}"
            elif spec[0] == "absknot":
                j, knot = spec[1], spec[2]
                med = self.feature_medians_[j]
                scl = self.feature_scales_[j]
                knot_raw = med + scl * knot
                term = f"|x{j} - {knot_raw:.4g}|"
            else:
                term = "1"
            active_terms.append((abs(scale), f"  {scale:+.6f} * {term}"))
        for _, line in sorted(active_terms, key=lambda t: -t[0]):
            lines.append(line)

        if len(lines) == 2:
            lines.append("  +0.000000 * (no active terms)")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseAdaptiveKnotRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseAdaptiveKnot_v1"
model_description = "Sparse robust linear model with adaptive |x-knot| terms selected by residual screening and compact debiased top-term equation"
model_defs = [(model_shorthand_name, SparseAdaptiveKnotRegressor())]


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
