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


class StableSparseLOORidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Linear regressor built from scratch:
    1) choose ridge alpha with exact LOOCV from SVD,
    2) prune weak features by AICc-like objective, then refit ridge.
    """

    def __init__(
        self,
        alpha_grid=None,
        complexity_penalty=0.004,
        min_features=3,
        max_features=14,
        prune_screen=20,
        max_prune_steps=25,
        prune_tol=1e-5,
        coef_display_tol=0.02,
    ):
        self.alpha_grid = alpha_grid
        self.complexity_penalty = complexity_penalty
        self.min_features = min_features
        self.max_features = max_features
        self.prune_screen = prune_screen
        self.max_prune_steps = max_prune_steps
        self.prune_tol = prune_tol
        self.coef_display_tol = coef_display_tol

    @staticmethod
    def _zscore(X):
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        sigma = np.where(sigma < 1e-12, 1.0, sigma)
        return mu, sigma

    @staticmethod
    def _ridge_fit(X, y, alpha):
        n, p = X.shape
        if p == 0:
            intercept = float(np.mean(y))
            pred = np.full(n, intercept, dtype=float)
            return intercept, np.zeros(0, dtype=float), pred

        x_mean = np.mean(X, axis=0)
        y_mean = float(np.mean(y))
        Xc = X - x_mean
        yc = y - y_mean

        A = Xc.T @ Xc + float(alpha) * np.eye(p)
        b = Xc.T @ yc
        coef = np.linalg.solve(A, b)
        intercept = float(y_mean - np.dot(x_mean, coef))
        pred = intercept + X @ coef
        return intercept, coef, pred

    def _choose_alpha_loocv(self, X, y):
        n, p = X.shape
        if p == 0:
            return 1.0

        x_mean = np.mean(X, axis=0)
        y_mean = float(np.mean(y))
        Xc = X - x_mean
        yc = y - y_mean

        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        Uy = U.T @ yc
        s2 = s ** 2
        U2 = U ** 2

        if self.alpha_grid is None:
            alphas = np.logspace(-4, 4, 19)
        else:
            alphas = np.asarray(self.alpha_grid, dtype=float)

        best_alpha = float(alphas[0])
        best_score = np.inf

        for a in alphas:
            filt = s2 / (s2 + a)
            yhat = U @ (filt * Uy)
            resid = yc - yhat
            h_diag = U2 @ filt
            loo_resid = resid / np.maximum(1e-6, 1.0 - h_diag)
            score = float(np.mean(loo_resid ** 2))
            if score < best_score:
                best_score = score
                best_alpha = float(a)

        return best_alpha

    def _objective(self, y, pred, k):
        n = max(1, y.shape[0])
        mse = float(np.mean((y - pred) ** 2))
        # Small complexity surcharge to prefer concise equations.
        return mse + float(self.complexity_penalty) * (k * np.log1p(n) / n)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape

        mu, sigma = self._zscore(X)
        Xs = (X - mu) / sigma

        alpha = self._choose_alpha_loocv(Xs, y)
        _, coef_std_full, pred_full = self._ridge_fit(Xs, y, alpha)

        active = np.where(np.abs(coef_std_full) > 1e-12)[0].tolist()
        if len(active) == 0:
            active = [int(np.argmax(np.var(Xs, axis=0)))] if p > 0 else []

        kmin = int(max(1, min(self.min_features, p))) if p > 0 else 0
        kmax = int(max(kmin, min(self.max_features, p))) if p > 0 else 0

        if len(active) > kmax:
            keep = np.argsort(-np.abs(coef_std_full))[:kmax]
            active = sorted(int(i) for i in keep)

        X_act = Xs[:, active] if active else np.zeros((n, 0), dtype=float)
        _, coef_act, pred = self._ridge_fit(X_act, y, alpha)
        best_obj = self._objective(y, pred, len(active))

        for _ in range(int(max(0, self.max_prune_steps))):
            if len(active) <= kmin:
                break

            strength = np.abs(coef_act)
            screen = int(max(1, min(self.prune_screen, len(active))))
            local_order = np.argsort(strength)[:screen]

            best_step = None
            for li in local_order:
                drop_pos = int(li)
                cand_active = active[:drop_pos] + active[drop_pos + 1 :]
                X_cand = Xs[:, cand_active] if cand_active else np.zeros((n, 0), dtype=float)
                _, coef_cand, pred_cand = self._ridge_fit(X_cand, y, alpha)
                obj_cand = self._objective(y, pred_cand, len(cand_active))
                if (best_step is None) or (obj_cand < best_step[0]):
                    best_step = (obj_cand, cand_active, coef_cand, pred_cand)

            if best_step is None:
                break
            if best_step[0] >= best_obj - float(self.prune_tol):
                break

            best_obj = best_step[0]
            active = best_step[1]
            coef_act = best_step[2]
            pred = best_step[3]

        X_final = Xs[:, active] if active else np.zeros((n, 0), dtype=float)
        intercept_std, coef_std_final, pred_final = self._ridge_fit(X_final, y, alpha)

        coef_std_full_final = np.zeros(p, dtype=float)
        for j_local, j_global in enumerate(active):
            coef_std_full_final[int(j_global)] = float(coef_std_final[j_local])

        coef_raw = coef_std_full_final / sigma
        intercept_raw = float(intercept_std - np.dot(coef_std_full_final, mu / sigma))

        self.alpha_ = float(alpha)
        self.mu_ = mu
        self.sigma_ = sigma
        self.intercept_ = intercept_raw
        self.coef_ = coef_raw
        self.active_features_ = np.asarray(active, dtype=int)
        self.training_mse_ = float(np.mean((y - pred_final) ** 2))
        self.n_features_in_ = int(p)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        X = np.asarray(X, dtype=float)
        return self.intercept_ + X @ self.coef_

    def __str__(self):
        check_is_fitted(self, "is_fitted_")
        lines = ["Stable Sparse LOO-Ridge Regressor:"]
        lines.append("  prediction = intercept + sum_j b_j * xj")
        lines.append(f"  ridge alpha (LOOCV): {self.alpha_:.5f}")
        lines.append(f"  intercept: {self.intercept_:+.4f}")
        lines.append(f"  active features: {len(self.active_features_)} / {self.n_features_in_}")
        lines.append(f"  training MSE: {self.training_mse_:.6f}")

        tol = float(max(0.0, self.coef_display_tol))
        active_terms = [
            (j, float(c))
            for j, c in enumerate(self.coef_)
            if abs(float(c)) >= tol
        ]
        if not active_terms:
            lines.append("  linear terms: none above display threshold")
        else:
            lines.append("  linear terms:")
            for j, c in sorted(active_terms, key=lambda t: -abs(t[1])):
                lines.append(f"    {c:+.4f} * x{j}")
        return "\n".join(lines)



# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
StableSparseLOORidgeRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "StableSparseLOORidge_v1"
model_description = "Custom LOOCV ridge from SVD with stability-aware backward pruning and compact linear equation refit"
model_defs = [(model_shorthand_name, StableSparseLOORidgeRegressor())]


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
