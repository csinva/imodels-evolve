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


class AdaptiveHuberRidgeMapV1(BaseEstimator, RegressorMixin):
    """
    Robust adaptive weighted ridge in raw-feature equation form.

    - Robustly scale features (median/MAD) and clip extreme z-scores.
    - Build adaptive ridge penalties from a pilot ridge fit.
    - Choose global shrinkage alpha by deterministic K-fold CV.
    - Return one explicit linear equation in original feature units.
    """

    def __init__(
        self,
        alpha_grid=None,
        n_folds=3,
        clip_z=4.5,
        pilot_alpha=1.0,
        adapt_gamma=0.7,
        adapt_tau=0.08,
        seed=42,
        eps=1e-12,
    ):
        self.alpha_grid = alpha_grid
        self.n_folds = n_folds
        self.clip_z = clip_z
        self.pilot_alpha = pilot_alpha
        self.adapt_gamma = adapt_gamma
        self.adapt_tau = adapt_tau
        self.seed = seed
        self.eps = eps

    @staticmethod
    def _robust_scale(X):
        med = np.median(X, axis=0)
        mad = np.median(np.abs(X - med), axis=0) * 1.4826
        std = np.std(X, axis=0)
        scale = np.where(mad > 1e-12, mad, np.where(std > 1e-12, std, 1.0))
        return med.astype(float), scale.astype(float)

    def _ridge_solve(self, Xs, y, alpha, penalty_weights):
        n, p = Xs.shape
        D = np.column_stack([np.ones(n, dtype=float), Xs])
        reg_diag = np.concatenate([[0.0], np.asarray(penalty_weights, dtype=float)])
        A = D.T @ D + float(alpha) * np.diag(reg_diag)
        b = D.T @ y
        try:
            theta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            theta = np.linalg.pinv(A) @ b
        return np.asarray(theta, dtype=float)

    def _make_folds(self, n):
        k = int(max(2, min(self.n_folds, n)))
        idx = np.arange(n)
        rng = np.random.RandomState(self.seed)
        rng.shuffle(idx)
        return np.array_split(idx, k)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        self.n_features_in_ = p

        med, scale = self._robust_scale(X)
        Xs = (X - med) / scale
        Xs = np.clip(Xs, -float(self.clip_z), float(self.clip_z))

        theta_pilot = self._ridge_solve(
            Xs,
            y,
            float(self.pilot_alpha),
            np.ones(p, dtype=float),
        )
        beta_pilot = np.asarray(theta_pilot[1:], dtype=float)

        penalty = 1.0 / ((np.abs(beta_pilot) + float(self.adapt_tau)) ** float(self.adapt_gamma))
        penalty = penalty / max(float(np.mean(penalty)), self.eps)
        penalty = np.clip(penalty, 0.25, 6.0)

        alphas = (
            np.asarray(self.alpha_grid, dtype=float)
            if self.alpha_grid is not None
            else np.logspace(-4, 2, 12)
        )
        folds = self._make_folds(n)
        best_alpha = float(alphas[0])
        best_cv_mse = None
        for alpha in alphas:
            mse_sum = 0.0
            valid_folds = 0
            for val_idx in folds:
                tr_mask = np.ones(n, dtype=bool)
                tr_mask[val_idx] = False
                if tr_mask.sum() < 2:
                    continue
                theta = self._ridge_solve(Xs[tr_mask], y[tr_mask], float(alpha), penalty)
                pred_val = theta[0] + Xs[val_idx] @ theta[1:]
                mse_sum += float(np.mean((y[val_idx] - pred_val) ** 2))
                valid_folds += 1
            if valid_folds == 0:
                continue
            cv_mse = mse_sum / valid_folds
            if (best_cv_mse is None) or (cv_mse < best_cv_mse):
                best_cv_mse = cv_mse
                best_alpha = float(alpha)

        theta = self._ridge_solve(Xs, y, best_alpha, penalty)
        coef_std = np.asarray(theta[1:], dtype=float)
        self.coef_ = coef_std / scale
        self.intercept_ = float(theta[0] - np.dot(self.coef_, med))
        self.alpha_ = float(best_alpha)
        self.cv_mse_ = float(best_cv_mse) if best_cv_mse is not None else float("nan")
        self.penalty_weights_ = np.asarray(penalty, dtype=float)

        abs_coef = np.abs(self.coef_)
        mass = abs_coef / max(float(np.sum(abs_coef)), self.eps)
        self.meaningful_features_ = np.where(mass >= 0.06)[0].astype(int)
        if self.meaningful_features_.size == 0:
            self.meaningful_features_ = np.array([int(np.argmax(abs_coef))], dtype=int)
        self.negligible_features_ = np.where(mass < 0.01)[0].astype(int)
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "coef_"])
        X = np.asarray(X, dtype=float)
        return self.intercept_ + X @ self.coef_

    def __str__(self):
        check_is_fitted(self, ["intercept_", "coef_", "alpha_"])
        lines = [
            "Adaptive Huberized Ridge Map Regressor",
            "Exact prediction equation in raw features:",
        ]

        terms = [f"{self.intercept_:+.6f}"]
        active = np.where(np.abs(self.coef_) > 1e-12)[0]
        for j in active:
            terms.append(f"{float(self.coef_[j]):+.6f}*x{int(j)}")
        lines.append("  y = " + " ".join(terms))

        lines.append("")
        lines.append("Coefficients (sorted by absolute magnitude):")
        for j in np.argsort(-np.abs(self.coef_)):
            lines.append(f"  x{int(j)}: {float(self.coef_[j]):+.6f}")

        lines.append("")
        lines.append(f"Selected alpha (CV): {self.alpha_:.6g}")
        lines.append(f"Mean CV MSE: {self.cv_mse_:.6f}")
        lines.append("Meaningful features: " + ", ".join(f"x{int(i)}" for i in self.meaningful_features_))
        if len(self.negligible_features_) > 0:
            lines.append("Negligible features: " + ", ".join(f"x{int(i)}" for i in self.negligible_features_))
        return "\n".join(lines)


import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
AdaptiveHuberRidgeMapV1.__module__ = "interpretable_regressor"

model_shorthand_name = "AdaptiveHuberRidgeMapV1"
model_description = "From-scratch robust median-MAD scaled and clipped adaptive weighted ridge with K-fold-selected global shrinkage, expressed as one explicit raw-feature linear equation"
model_defs = [(model_shorthand_name, AdaptiveHuberRidgeMapV1())]

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
