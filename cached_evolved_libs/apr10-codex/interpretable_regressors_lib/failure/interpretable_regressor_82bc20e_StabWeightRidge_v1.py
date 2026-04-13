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


class StabilityWeightedRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Custom interpretable linear model with adaptive uncertainty penalties.
    1) choose a global ridge alpha by GCV on standardized features,
    2) estimate coefficient uncertainty from bootstrap ridge fits,
    3) convert uncertainty into per-feature penalty weights,
    4) fit a final weighted-ridge equation in raw feature space.
    """

    def __init__(
        self,
        alpha_grid=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
        n_bootstraps=9,
        sample_fraction=0.8,
        stability_floor=0.2,
        min_penalty_weight=0.35,
        max_penalty_weight=3.5,
        blend_with_uniform=0.25,
        blend_with_base=0.2,
        coef_display_tol=0.01,
        random_state=42,
    ):
        self.alpha_grid = alpha_grid
        self.n_bootstraps = n_bootstraps
        self.sample_fraction = sample_fraction
        self.stability_floor = stability_floor
        self.min_penalty_weight = min_penalty_weight
        self.max_penalty_weight = max_penalty_weight
        self.blend_with_uniform = blend_with_uniform
        self.blend_with_base = blend_with_base
        self.coef_display_tol = coef_display_tol
        self.random_state = random_state

    @staticmethod
    def _standardize(X):
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        sigma = np.where(sigma < 1e-12, 1.0, sigma)
        return mu, sigma

    def _sanitize_alpha_grid(self):
        alphas = np.asarray(self.alpha_grid, dtype=float)
        alphas = alphas[np.isfinite(alphas) & (alphas > 0)]
        if alphas.size == 0:
            return np.asarray([1.0], dtype=float)
        return np.unique(alphas)

    @staticmethod
    def _solve_centered_weighted_ridge(Xc, yc, alpha, penalty_weights):
        p = Xc.shape[1]
        if p == 0:
            return np.zeros(0, dtype=float)
        w = np.asarray(penalty_weights, dtype=float).ravel()
        if w.size != p:
            w = np.ones(p, dtype=float)
        A = Xc.T @ Xc + float(alpha) * np.diag(w)
        b = Xc.T @ yc
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(A) @ b

    def _fit_weighted_ridge_standardized(self, Xs, y, alpha, penalty_weights):
        x_mean = np.mean(Xs, axis=0)
        y_mean = float(np.mean(y))
        Xc = Xs - x_mean
        yc = y - y_mean
        coef = self._solve_centered_weighted_ridge(Xc, yc, alpha, penalty_weights)
        intercept = float(y_mean - np.dot(x_mean, coef))
        return intercept, coef

    def _choose_alpha_gcv(self, Xs, y, alphas):
        if Xs.shape[1] == 0:
            return 1.0
        x_mean = np.mean(Xs, axis=0)
        y_mean = float(np.mean(y))
        Xc = Xs - x_mean
        yc = y - y_mean
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        Uy = U.T @ yc
        n = float(max(1, Xs.shape[0]))

        best_alpha = float(alphas[0])
        best_score = None
        for a in alphas:
            filt = s / (s * s + float(a))
            coef = Vt.T @ (filt * Uy)
            pred = y_mean + Xc @ coef
            mse = float(np.mean((y - pred) ** 2))
            dof = float(np.sum((s * s) / (s * s + float(a))))
            denom = max(1e-8, 1.0 - dof / n)
            gcv = mse / (denom * denom)
            if (best_score is None) or (gcv < best_score):
                best_score = gcv
                best_alpha = float(a)
        return best_alpha

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        self.n_features_in_ = int(p)

        if p == 0:
            self.intercept_ = float(np.mean(y))
            self.coef_ = np.zeros(0, dtype=float)
            self.alpha_ = 1.0
            self.penalty_weights_ = np.zeros(0, dtype=float)
            self.stability_score_ = np.zeros(0, dtype=float)
            self.training_mse_ = float(np.mean((y - self.intercept_) ** 2))
            self.is_fitted_ = True
            return self

        mu, sigma = self._standardize(X)
        Xs = (X - mu) / sigma
        alphas = self._sanitize_alpha_grid()
        alpha = self._choose_alpha_gcv(Xs, y, alphas)

        _, base_coef_std = self._fit_weighted_ridge_standardized(
            Xs, y, alpha, np.ones(p, dtype=float)
        )

        rng = np.random.RandomState(self.random_state)
        n_boot = int(max(1, self.n_bootstraps))
        n_sub = int(max(4, min(n, round(float(self.sample_fraction) * n))))
        coef_boot = np.zeros((n_boot, p), dtype=float)
        for b in range(n_boot):
            idx = rng.choice(n, size=n_sub, replace=True)
            _, coef_b = self._fit_weighted_ridge_standardized(
                Xs[idx], y[idx], alpha, np.ones(p, dtype=float)
            )
            coef_boot[b] = coef_b

        coef_mean = np.mean(coef_boot, axis=0)
        coef_std = np.std(coef_boot, axis=0)
        stability = np.abs(coef_mean) / (coef_std + 1e-8)

        floor = float(max(1e-6, self.stability_floor))
        penalty = 1.0 / (stability + floor)
        penalty /= max(1e-8, float(np.mean(penalty)))
        penalty = np.clip(
            penalty,
            float(self.min_penalty_weight),
            float(self.max_penalty_weight),
        )
        mix = float(np.clip(self.blend_with_uniform, 0.0, 1.0))
        penalty = (1.0 - mix) * penalty + mix * np.ones_like(penalty)

        intercept_std, adapt_coef_std = self._fit_weighted_ridge_standardized(Xs, y, alpha, penalty)
        blend = float(np.clip(self.blend_with_base, 0.0, 1.0))
        coef_std_final = (1.0 - blend) * adapt_coef_std + blend * base_coef_std

        coef_raw = coef_std_final / sigma
        intercept_raw = float(intercept_std - np.dot(coef_std_final, mu / sigma))
        pred = intercept_raw + X @ coef_raw

        self.mu_ = mu
        self.sigma_ = sigma
        self.alpha_ = float(alpha)
        self.intercept_ = intercept_raw
        self.coef_ = coef_raw
        self.penalty_weights_ = penalty
        self.stability_score_ = stability
        self.training_mse_ = float(np.mean((y - pred) ** 2))
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        X = np.asarray(X, dtype=float)
        return self.intercept_ + X @ self.coef_

    def __str__(self):
        check_is_fitted(self, "is_fitted_")
        lines = ["Stability-Weighted Ridge Regressor:"]
        lines.append("  prediction = intercept + sum_j b_j * xj")
        lines.append(f"  global ridge alpha (GCV): {self.alpha_:.6f}")
        lines.append(f"  intercept: {self.intercept_:+.4f}")
        lines.append(f"  training MSE: {self.training_mse_:.6f}")

        tol = float(max(0.0, self.coef_display_tol))
        terms = []
        for j, c in enumerate(self.coef_):
            if abs(float(c)) >= tol:
                terms.append(
                    (
                        abs(float(c)),
                        int(j),
                        float(c),
                        float(self.stability_score_[j]),
                        float(self.penalty_weights_[j]),
                    )
                )
        terms.sort(reverse=True)
        lines.append(f"  shown terms (|coef| >= {tol:.3f}): {len(terms)} / {self.n_features_in_}")
        if not terms:
            lines.append("  linear terms: none above display threshold")
        else:
            lines.append("  linear terms (largest absolute coefficients):")
            for _, j, c, s, w in terms[:18]:
                lines.append(f"    {c:+.4f} * x{j}   [stability {s:.3f}, penalty {w:.3f}]")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
StabilityWeightedRidgeRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "StabWeightRidge_v1"
model_description = "Adaptive weighted-ridge equation: bootstrap uncertainty defines per-feature penalties, then blended with base ridge for stable dense linear predictions"
model_defs = [(model_shorthand_name, StabilityWeightedRidgeRegressor())]


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
