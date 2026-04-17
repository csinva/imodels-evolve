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


class ScoreSplineLinearRegressor(BaseEstimator, RegressorMixin):
    """Custom CV ridge linear model with an optional one-knot hinge on model score."""

    def __init__(
        self,
        alpha_grid=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
        cv_folds=4,
        val_frac=0.2,
        hinge_quantiles=(0.15, 0.3, 0.5, 0.7, 0.85),
        min_hinge_rel_gain=0.01,
        random_state=0,
    ):
        self.alpha_grid = alpha_grid
        self.cv_folds = cv_folds
        self.val_frac = val_frac
        self.hinge_quantiles = hinge_quantiles
        self.min_hinge_rel_gain = min_hinge_rel_gain
        self.random_state = random_state

    @staticmethod
    def _safe_standardize(X):
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        sigma = np.where(sigma < 1e-12, 1.0, sigma)
        return mu, sigma

    @staticmethod
    def _ridge_fit_standardized(Xs, y, alpha):
        n_samples, n_features = Xs.shape
        X_aug = np.hstack([np.ones((n_samples, 1)), Xs])
        reg = np.eye(n_features + 1, dtype=float)
        reg[0, 0] = 0.0  # do not penalize intercept
        gram = X_aug.T @ X_aug + float(alpha) * reg
        rhs = X_aug.T @ y
        beta = np.linalg.solve(gram, rhs)
        intercept = float(beta[0])
        coef = np.asarray(beta[1:], dtype=float)
        return intercept, coef

    def _choose_alpha(self, Xs, y):
        n_samples = Xs.shape[0]
        if n_samples < 4:
            return float(self.alpha_grid[0])

        k = int(max(2, min(int(self.cv_folds), n_samples)))
        rng = np.random.RandomState(self.random_state)
        perm = rng.permutation(n_samples)
        folds = np.array_split(perm, k)

        best_alpha = float(self.alpha_grid[0])
        best_mse = float("inf")

        for alpha in self.alpha_grid:
            mses = []
            for i in range(k):
                val_idx = folds[i]
                if len(val_idx) == 0:
                    continue
                train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
                if len(train_idx) == 0:
                    continue

                b0, w = self._ridge_fit_standardized(Xs[train_idx], y[train_idx], float(alpha))
                pred = b0 + Xs[val_idx] @ w
                mse = float(np.mean((y[val_idx] - pred) ** 2))
                mses.append(mse)

            if mses:
                mean_mse = float(np.mean(mses))
                if mean_mse < best_mse:
                    best_mse = mean_mse
                    best_alpha = float(alpha)

        return best_alpha

    def _fit_hinge_on_score(self, score, y, y_hat):
        n = len(score)
        if n <= 5:
            return 0.0, 0.0, 0.0, 0.0

        n_val = max(1, int(round(float(self.val_frac) * n)))
        n_val = min(n_val, n - 1)
        rng = np.random.RandomState(self.random_state + 1)
        perm = rng.permutation(n)
        train_idx = perm[:-n_val]
        val_idx = perm[-n_val:]

        s_tr = score[train_idx]
        s_va = score[val_idx]
        r_tr = y[train_idx] - y_hat[train_idx]
        r_va = y[val_idx] - y_hat[val_idx]

        base_val_mse = float(np.mean(r_va ** 2))
        best_gain = 0.0
        best_t = 0.0
        best_g = 0.0
        best_val_mse = base_val_mse

        thresholds = [float(np.quantile(s_tr, q)) for q in self.hinge_quantiles]
        seen = set()
        unique_thresholds = []
        for t in thresholds:
            key = round(t, 8)
            if key not in seen:
                seen.add(key)
                unique_thresholds.append(t)

        for t in unique_thresholds:
            h_tr = np.maximum(0.0, s_tr - t)
            h_va = np.maximum(0.0, s_va - t)

            denom = float(h_tr @ h_tr)
            if denom < 1e-12:
                continue

            gamma = float((h_tr @ r_tr) / denom)
            pred_resid_va = gamma * h_va
            val_mse = float(np.mean((r_va - pred_resid_va) ** 2))
            rel_gain = (base_val_mse - val_mse) / max(abs(base_val_mse), 1e-12)

            if rel_gain > best_gain:
                best_gain = rel_gain
                best_t = float(t)
                best_g = gamma
                best_val_mse = val_mse

        if best_gain < float(self.min_hinge_rel_gain):
            return 0.0, 0.0, 0.0, base_val_mse

        return best_t, best_g, best_gain, best_val_mse

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape
        if n_features == 0:
            raise ValueError("No features provided")
        self.n_features_in_ = n_features

        self.x_mean_, self.x_scale_ = self._safe_standardize(X)
        Xs = (X - self.x_mean_) / self.x_scale_

        self.alpha_ = self._choose_alpha(Xs, y)
        intercept_s, coef_s = self._ridge_fit_standardized(Xs, y, self.alpha_)

        self.coef_ = coef_s / self.x_scale_
        self.intercept_ = float(intercept_s - np.sum((coef_s * self.x_mean_) / self.x_scale_))

        score = self.intercept_ + X @ self.coef_
        t, g, gain, hinge_val_mse = self._fit_hinge_on_score(score, y, score)
        self.hinge_threshold_ = float(t)
        self.hinge_gamma_ = float(g)
        self.hinge_rel_gain_ = float(gain)
        self.hinge_val_mse_ = float(hinge_val_mse)

        importances = np.abs(self.coef_).copy()
        max_imp = float(np.max(importances)) if np.max(importances) > 0 else 0.0
        self.feature_importances_ = importances / max_imp if max_imp > 0 else importances
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "coef_", "hinge_threshold_", "hinge_gamma_"])
        X = np.asarray(X, dtype=float)
        score = self.intercept_ + X @ self.coef_
        if abs(self.hinge_gamma_) > 0.0:
            score = score + self.hinge_gamma_ * np.maximum(0.0, score - self.hinge_threshold_)
        return score

    def __str__(self):
        check_is_fitted(
            self,
            [
                "intercept_",
                "coef_",
                "alpha_",
                "feature_importances_",
                "hinge_threshold_",
                "hinge_gamma_",
                "hinge_rel_gain_",
            ],
        )

        lines = [
            "Score-Spline Linear Regressor",
            "Step 1: linear score s = intercept + sum_j coef_j * x_j",
            "Step 2: prediction y = s + gamma * max(0, s - threshold)",
            f"intercept = {self.intercept_:+.6f}",
            f"ridge_alpha = {self.alpha_:.6g}",
            "",
            "Raw-feature coefficients:",
        ]

        for j, c in enumerate(self.coef_):
            lines.append(f"x{j}: coef={float(c):+.6f}, importance={float(self.feature_importances_[j]):.3f}")

        lines.append("")
        if abs(self.hinge_gamma_) > 0.0:
            lines.append(
                f"Score hinge correction: gamma={self.hinge_gamma_:+.6f}, threshold={self.hinge_threshold_:+.6f}, rel_val_gain={self.hinge_rel_gain_:.4f}"
            )
        else:
            lines.append("Score hinge correction: none (disabled)")

        lines.append("")
        lines.append("Simulation recipe: compute s from raw x-values, then apply the one-line hinge correction.")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys

_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])

# Update the model shorthand name and description below to reflect the class above and any changes you make.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
ScoreSplineLinearRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "ScoreSplineLinear_v1"
model_description = "From-scratch CV ridge on raw features plus optional single hinge correction on the linear score selected by validation gain"
model_defs = [(model_shorthand_name, ScoreSplineLinearRegressor())]


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

    # --- Recompute global rank summary from updated performance_results.csv ---
    # Build dataset -> {model: rmse}
    perf_table = defaultdict(dict)
    with open(perf_csv, newline="") as f:
        for row in csv.DictReader(f):
            ds = row["dataset"]
            m = row["model"]
            rmse_s = row.get("rmse", "")
            if rmse_s in ("", None):
                perf_table[ds][m] = float("nan")
            else:
                try:
                    perf_table[ds][m] = float(rmse_s)
                except ValueError:
                    perf_table[ds][m] = float("nan")

    avg_rank, _ = compute_rank_scores(perf_table)
    mean_rank = avg_rank.get(model_name, float("nan"))

    # --- Upsert overall_results.csv ---
    overall_rows = [{
        "commit": git_hash,
        "mean_rank": f"{mean_rank:.2f}" if np.isfinite(mean_rank) else "",
        "frac_interpretability_tests_passed": f"{(n_passed / total):.4f}" if total else "",
        "status": "",  # fill manually after reviewing
        "model_name": model_name,
        "description": model_description,
    }]
    upsert_overall_results(overall_rows, RESULTS_DIR)

    # --- Plot update ---
    overall_csv = os.path.join(RESULTS_DIR, "overall_results.csv")
    plot_interp_vs_performance(
        overall_csv,
        os.path.join(RESULTS_DIR, "interpretability_vs_performance.png"),
    )

    # Print compact summary
    std_names = {t.__name__ for t in ALL_TESTS}
    hard_names = {t.__name__ for t in HARD_TESTS}
    ins_names = {t.__name__ for t in INSIGHT_TESTS}
    n_std = sum(r["passed"] for r in interp_results if r["test"] in std_names)
    n_hard = sum(r["passed"] for r in interp_results if r["test"] in hard_names)
    n_ins = sum(r["passed"] for r in interp_results if r["test"] in ins_names)

    print("\n---")
    print(f"tests_passed:  {n_passed}/{total} ({(n_passed/total):.2%})  "
          f"[std {n_std}/{len(std_names)}  hard {n_hard}/{len(hard_names)}  insight {n_ins}/{len(ins_names)}]")
    print(f"total_seconds: {time.time() - t0:.1f}s")
