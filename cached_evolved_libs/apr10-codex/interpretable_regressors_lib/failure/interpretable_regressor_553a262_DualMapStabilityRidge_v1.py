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


class DualMapStabilityRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Interpretable sparse ridge with per-feature transform selection.

    For each feature, choose either a linear standardized map z_j or a signed-root
    map sign(z_j)*sqrt(|z_j|) based on univariate alignment with y, then fit a
    GCV-selected ridge model and keep only stability-screened terms.
    """

    def __init__(
        self,
        alpha_grid=None,
        max_terms=24,
        min_terms=6,
        scale_multipliers=(0.6, 1.0, 1.8),
        stable_sign_frac=0.67,
        rel_strength_tol=0.015,
        coef_display_tol=0.02,
    ):
        self.alpha_grid = alpha_grid
        self.max_terms = max_terms
        self.min_terms = min_terms
        self.scale_multipliers = scale_multipliers
        self.stable_sign_frac = stable_sign_frac
        self.rel_strength_tol = rel_strength_tol
        self.coef_display_tol = coef_display_tol

    @staticmethod
    def _zscore(X):
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        sigma = np.where(sigma < 1e-12, 1.0, sigma)
        return mu, sigma

    @staticmethod
    def _signed_root(z):
        return np.sign(z) * np.sqrt(np.abs(z) + 1e-12)

    @staticmethod
    def _ridge_from_svd(U, s, Vt, y, alpha):
        filt = s / (s * s + float(alpha))
        return Vt.T @ (filt * (U.T @ y))

    def _select_maps(self, Z, y):
        n, p = Z.shape
        yc = y - np.mean(y)
        y_norm = float(np.linalg.norm(yc)) + 1e-12

        map_ids = np.zeros(p, dtype=int)  # 0 -> linear z, 1 -> signed-root
        M = np.empty_like(Z)

        for j in range(p):
            zj = Z[:, j]
            sj = self._signed_root(zj)

            zc = zj - np.mean(zj)
            sc = sj - np.mean(sj)

            corr_z = float(np.dot(zc, yc) / ((np.linalg.norm(zc) + 1e-12) * y_norm))
            corr_s = float(np.dot(sc, yc) / ((np.linalg.norm(sc) + 1e-12) * y_norm))

            if abs(corr_s) > abs(corr_z):
                map_ids[j] = 1
                M[:, j] = sj
            else:
                map_ids[j] = 0
                M[:, j] = zj

        return M, map_ids

    def _gcv_alpha(self, X, y):
        n = X.shape[0]
        U, s, Vt = np.linalg.svd(X, full_matrices=False)

        if self.alpha_grid is None:
            alphas = np.logspace(-5, 4, 19)
        else:
            alphas = np.asarray(self.alpha_grid, dtype=float)

        best = None
        for a in alphas:
            coef = self._ridge_from_svd(U, s, Vt, y, a)
            pred = X @ coef
            mse = float(np.mean((y - pred) ** 2))
            dof = float(np.sum((s * s) / (s * s + float(a))))
            denom = max(1e-7, 1.0 - dof / max(1.0, float(n)))
            gcv = mse / (denom * denom)
            if (best is None) or (gcv < best[0]):
                best = (gcv, float(a), coef, U, s, Vt)
        return best

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape

        if p == 0:
            self.mu_ = np.zeros(0, dtype=float)
            self.sigma_ = np.ones(0, dtype=float)
            self.map_ids_ = np.zeros(0, dtype=int)
            self.active_features_ = np.zeros(0, dtype=int)
            self.coef_ = np.zeros(0, dtype=float)
            self.intercept_ = float(np.mean(y))
            self.alpha_ = 1.0
            self.training_mse_ = float(np.mean((y - self.intercept_) ** 2))
            self.n_features_in_ = 0
            self.is_fitted_ = True
            return self

        mu, sigma = self._zscore(X)
        Z = (X - mu) / sigma
        M, map_ids = self._select_maps(Z, y)

        y_mean = float(np.mean(y))
        yc = y - y_mean
        x_mean = np.mean(M, axis=0)
        Mc = M - x_mean

        gcv, alpha, coef_base, U, s, Vt = self._gcv_alpha(Mc, yc)

        scales = np.asarray(self.scale_multipliers, dtype=float)
        scales = scales[np.isfinite(scales) & (scales > 0)]
        if scales.size == 0:
            scales = np.asarray([1.0], dtype=float)

        coef_path = []
        for mult in scales:
            a = float(alpha * mult)
            coef_path.append(self._ridge_from_svd(U, s, Vt, yc, a))
        coef_path = np.asarray(coef_path)

        sign_votes = np.sign(coef_path)
        pos_frac = np.mean(sign_votes > 0, axis=0)
        neg_frac = np.mean(sign_votes < 0, axis=0)
        sign_stability = np.maximum(pos_frac, neg_frac)

        median_coef = np.median(coef_path, axis=0)
        strength = np.abs(median_coef)
        max_strength = float(np.max(strength)) if strength.size else 0.0

        stable_mask = sign_stability >= float(self.stable_sign_frac)
        strong_mask = strength >= float(self.rel_strength_tol) * max(1e-12, max_strength)
        keep_mask = stable_mask & strong_mask

        order = np.argsort(-strength)
        kmin = int(max(1, min(self.min_terms, p)))
        kmax = int(max(kmin, min(self.max_terms, p)))

        kept = [int(j) for j in order if keep_mask[j]]
        if len(kept) < kmin:
            kept = [int(j) for j in order[:kmin]]
        elif len(kept) > kmax:
            kept = kept[:kmax]

        kept = sorted(kept)
        Mc_keep = Mc[:, kept]

        U2, s2, Vt2 = np.linalg.svd(Mc_keep, full_matrices=False)
        coef_keep = self._ridge_from_svd(U2, s2, Vt2, yc, alpha)

        coef_full = np.zeros(p, dtype=float)
        for local_idx, global_idx in enumerate(kept):
            coef_full[global_idx] = float(coef_keep[local_idx])

        pred = y_mean + (Mc @ coef_full)
        intercept = float(y_mean - np.dot(x_mean, coef_full))

        self.mu_ = mu
        self.sigma_ = sigma
        self.map_ids_ = map_ids
        self.active_features_ = np.asarray(kept, dtype=int)
        self.coef_ = coef_full
        self.intercept_ = intercept
        self.alpha_ = float(alpha)
        self.gcv_score_ = float(gcv)
        self.training_mse_ = float(np.mean((y - pred) ** 2))
        self.n_features_in_ = int(p)
        self.is_fitted_ = True
        return self

    def _mapped_features(self, X):
        Z = (X - self.mu_) / self.sigma_
        M = Z.copy()
        root_cols = np.where(self.map_ids_ == 1)[0]
        if root_cols.size > 0:
            M[:, root_cols] = self._signed_root(Z[:, root_cols])
        return M

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        X = np.asarray(X, dtype=float)
        M = self._mapped_features(X)
        return self.intercept_ + M @ self.coef_

    def __str__(self):
        check_is_fitted(self, "is_fitted_")
        lines = ["Dual-Map Stability Ridge Regressor:"]
        lines.append("  prediction = intercept + sum_j w_j * phi_j((xj-mu_j)/sigma_j)")
        lines.append("  where phi_j(z) is either linear z or signed_root(z)")
        lines.append(f"  ridge alpha (GCV): {self.alpha_:.6f}")
        lines.append(f"  intercept: {self.intercept_:+.4f}")
        lines.append(f"  active terms: {len(self.active_features_)} / {self.n_features_in_}")
        lines.append(f"  training MSE: {self.training_mse_:.6f}")

        tol = float(max(0.0, self.coef_display_tol))
        terms = []
        for j in self.active_features_:
            c = float(self.coef_[int(j)])
            if abs(c) < tol:
                continue
            map_name = "z" if int(self.map_ids_[int(j)]) == 0 else "signed_root(z)"
            terms.append((j, c, map_name))

        if not terms:
            lines.append("  terms: none above display threshold")
        else:
            lines.append("  terms (sorted by |weight|):")
            for j, c, map_name in sorted(terms, key=lambda t: -abs(t[1])):
                lines.append(f"    {c:+.4f} * phi_{j}[{map_name}] with z=(x{j}-mu{j})/sigma{j}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
DualMapStabilityRidgeRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "DualMapStabilityRidge_v1"
model_description = "Per-feature linear vs signed-root transform selection, GCV ridge, and sign-stability sparse screening"
model_defs = [(model_shorthand_name, DualMapStabilityRidgeRegressor())]


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
