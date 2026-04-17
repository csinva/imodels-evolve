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


class CrossFitHingeRidgeV1(BaseEstimator, RegressorMixin):
    """
    From-scratch 3-fold CV ridge with optional single-feature hinge correction.

    The hinge term is only enabled when cross-validation shows a consistent,
    meaningful gain over the linear backbone.
    """

    def __init__(
        self,
        alpha_grid=None,
        n_folds=3,
        hinge_lambda=1.0,
        min_cv_relative_gain=0.015,
        eps=1e-12,
    ):
        self.alpha_grid = alpha_grid
        self.n_folds = n_folds
        self.hinge_lambda = hinge_lambda
        self.min_cv_relative_gain = min_cv_relative_gain
        self.eps = eps

    @staticmethod
    def _safe_scale(X):
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        sigma = np.where(sigma > 1e-12, sigma, 1.0)
        return mu.astype(float), sigma.astype(float)

    @staticmethod
    def _kfold_indices(n, n_folds):
        n_folds = int(max(2, min(n_folds, n)))
        all_idx = np.arange(n, dtype=int)
        return np.array_split(all_idx, n_folds)

    def _solve_ridge_std(self, X_std, y, alpha):
        n, p = X_std.shape
        D = np.column_stack([np.ones(n, dtype=float), X_std])
        reg = np.zeros(p + 1, dtype=float)
        reg[1:] = float(alpha)
        A = D.T @ D + np.diag(reg)
        b = D.T @ y
        try:
            theta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            theta = np.linalg.pinv(A) @ b
        return float(theta[0]), np.asarray(theta[1:], dtype=float)

    @staticmethod
    def _hinge_col(x, threshold, sign):
        return np.maximum(0.0, float(sign) * (x - float(threshold)))

    def _fit_hinge_coef(self, residual, z):
        denom = float(np.dot(z, z) + float(self.hinge_lambda))
        if denom <= 0:
            return 0.0
        return float(np.dot(z, residual) / denom)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        self.n_features_in_ = p

        alphas = (
            np.asarray(self.alpha_grid, dtype=float)
            if self.alpha_grid is not None
            else np.logspace(-4, 3, 14)
        )

        folds = self._kfold_indices(n, self.n_folds)

        best_alpha = None
        best_cv_mse = None
        for alpha in alphas:
            mse_folds = []
            for val_idx in folds:
                tr_mask = np.ones(n, dtype=bool)
                tr_mask[val_idx] = False
                tr_idx = np.where(tr_mask)[0]
                Xtr, ytr = X[tr_idx], y[tr_idx]
                Xval, yval = X[val_idx], y[val_idx]

                mu, sigma = self._safe_scale(Xtr)
                Xtr_std = (Xtr - mu) / sigma
                Xval_std = (Xval - mu) / sigma
                b0, w_std = self._solve_ridge_std(Xtr_std, ytr, alpha=float(alpha))
                pred = b0 + Xval_std @ w_std
                mse_folds.append(float(np.mean((yval - pred) ** 2)))

            cv_mse = float(np.mean(mse_folds))
            if (best_cv_mse is None) or (cv_mse < best_cv_mse):
                best_cv_mse = cv_mse
                best_alpha = float(alpha)

        mu, sigma = self._safe_scale(X)
        X_std = (X - mu) / sigma
        b0_std, w_std = self._solve_ridge_std(X_std, y, alpha=best_alpha)
        coef_raw = w_std / sigma
        intercept_raw = float(b0_std - np.dot(coef_raw, mu))
        pred_linear = intercept_raw + X @ coef_raw
        mse_linear = float(np.mean((y - pred_linear) ** 2))

        top_j = int(np.argmax(np.abs(coef_raw)))
        q_lo, q_med, q_hi = np.quantile(X[:, top_j], [0.3, 0.5, 0.7])
        hinge_candidates = [
            (float(q_lo), 1.0),
            (float(q_lo), -1.0),
            (float(q_med), 1.0),
            (float(q_med), -1.0),
            (float(q_hi), 1.0),
            (float(q_hi), -1.0),
        ]

        best_hinge = None
        best_hinge_cv = None
        for threshold, sign in hinge_candidates:
            mse_folds = []
            gains = []
            for val_idx in folds:
                tr_mask = np.ones(n, dtype=bool)
                tr_mask[val_idx] = False
                tr_idx = np.where(tr_mask)[0]
                Xtr, ytr = X[tr_idx], y[tr_idx]
                Xval, yval = X[val_idx], y[val_idx]

                mu_f, sigma_f = self._safe_scale(Xtr)
                Xtr_std = (Xtr - mu_f) / sigma_f
                Xval_std = (Xval - mu_f) / sigma_f
                b_f, w_f_std = self._solve_ridge_std(Xtr_std, ytr, alpha=best_alpha)
                w_f_raw = w_f_std / sigma_f
                b_f_raw = float(b_f - np.dot(w_f_raw, mu_f))

                base_tr = b_f_raw + Xtr @ w_f_raw
                base_val = b_f_raw + Xval @ w_f_raw
                z_tr = self._hinge_col(Xtr[:, top_j], threshold=threshold, sign=sign)
                z_val = self._hinge_col(Xval[:, top_j], threshold=threshold, sign=sign)
                g = self._fit_hinge_coef(ytr - base_tr, z_tr)
                pred_val = base_val + g * z_val

                mse_base = float(np.mean((yval - base_val) ** 2))
                mse_new = float(np.mean((yval - pred_val) ** 2))
                mse_folds.append(mse_new)
                gains.append((mse_base - mse_new) / max(mse_base, self.eps))

            mean_gain = float(np.mean(gains))
            consistent = bool(np.median(gains) > 0.0)
            cv_mse = float(np.mean(mse_folds))
            if consistent and mean_gain >= float(self.min_cv_relative_gain):
                if (best_hinge_cv is None) or (cv_mse < best_hinge_cv):
                    best_hinge_cv = cv_mse
                    best_hinge = (float(threshold), float(sign), mean_gain)

        if best_hinge is None:
            threshold = 0.0
            sign = 1.0
            g_coef = 0.0
            mse_final = mse_linear
            hinge_active = False
            hinge_gain = 0.0
        else:
            threshold, sign, hinge_gain = best_hinge
            z_full = self._hinge_col(X[:, top_j], threshold=threshold, sign=sign)
            g_coef = self._fit_hinge_coef(y - pred_linear, z_full)
            pred_full = pred_linear + g_coef * z_full
            mse_final = float(np.mean((y - pred_full) ** 2))
            hinge_active = True

        self.intercept_ = float(intercept_raw)
        self.coef_ = np.asarray(coef_raw, dtype=float)
        self.alpha_ = float(best_alpha)
        self.cv_mse_linear_ = float(best_cv_mse)
        self.train_mse_linear_ = float(mse_linear)
        self.train_mse_final_ = float(mse_final)
        self.hinge_feature_ = int(top_j)
        self.hinge_threshold_ = float(threshold)
        self.hinge_sign_ = float(sign)
        self.hinge_coef_ = float(g_coef)
        self.hinge_active_ = bool(hinge_active and abs(g_coef) > 1e-10)
        self.hinge_cv_relative_gain_ = float(hinge_gain) if self.hinge_active_ else 0.0

        mass = np.abs(self.coef_)
        total = float(np.sum(mass)) + self.eps
        normalized = mass / total
        self.meaningful_features_ = np.where(normalized >= 0.06)[0].astype(int)
        if self.meaningful_features_.size == 0:
            self.meaningful_features_ = np.array([int(np.argmax(mass))], dtype=int)
        self.negligible_features_ = np.where(normalized < 0.01)[0].astype(int)
        return self

    def predict(self, X):
        check_is_fitted(
            self,
            [
                "intercept_",
                "coef_",
                "hinge_feature_",
                "hinge_threshold_",
                "hinge_sign_",
                "hinge_coef_",
                "hinge_active_",
            ],
        )
        X = np.asarray(X, dtype=float)
        yhat = self.intercept_ + X @ self.coef_
        if self.hinge_active_:
            z = self._hinge_col(
                X[:, self.hinge_feature_],
                threshold=self.hinge_threshold_,
                sign=self.hinge_sign_,
            )
            yhat = yhat + self.hinge_coef_ * z
        return yhat

    def __str__(self):
        check_is_fitted(
            self,
            ["intercept_", "coef_", "hinge_feature_", "hinge_threshold_", "hinge_sign_", "hinge_coef_", "hinge_active_"],
        )
        lines = [
            "CrossFit Hinge-Ridge Regressor",
            "Exact raw-feature equation:",
        ]
        terms = [f"{self.intercept_:+.6f}"]
        for j, c in enumerate(self.coef_):
            terms.append(f"{float(c):+.6f}*x{int(j)}")
        if self.hinge_active_:
            if self.hinge_sign_ > 0:
                hinge_txt = f"max(0, x{self.hinge_feature_} - ({self.hinge_threshold_:+.6f}))"
            else:
                hinge_txt = f"max(0, ({self.hinge_threshold_:+.6f}) - x{self.hinge_feature_})"
            terms.append(f"{self.hinge_coef_:+.6f}*{hinge_txt}")
        lines.append("  y = " + " ".join(terms))
        lines.append("")
        lines.append(f"Ridge alpha selected by 3-fold CV: {self.alpha_:.6g}")
        lines.append(f"CV MSE linear backbone: {self.cv_mse_linear_:.6f}")
        lines.append(f"Train MSE linear backbone: {self.train_mse_linear_:.6f}")
        lines.append(f"Train MSE final model: {self.train_mse_final_:.6f}")

        lines.append("")
        lines.append("Linear coefficients (largest absolute magnitude first):")
        for j in np.argsort(-np.abs(self.coef_)):
            lines.append(f"  x{int(j)}: {float(self.coef_[j]):+.6f}")

        lines.append("")
        if self.hinge_active_:
            lines.append("Hinge correction: active")
            lines.append(f"  feature: x{self.hinge_feature_}")
            lines.append(f"  threshold: {self.hinge_threshold_:+.6f}")
            lines.append(f"  sign: {'positive-side' if self.hinge_sign_ > 0 else 'negative-side'}")
            lines.append(f"  coefficient: {self.hinge_coef_:+.6f}")
            lines.append(f"  CV relative gain: {self.hinge_cv_relative_gain_:.2%}")
        else:
            lines.append("Hinge correction: inactive (linear model retained)")

        if len(self.meaningful_features_) > 0:
            lines.append("Meaningful features: " + ", ".join(f"x{int(i)}" for i in self.meaningful_features_))
        if len(self.negligible_features_) > 0:
            lines.append("Negligible features: " + ", ".join(f"x{int(i)}" for i in self.negligible_features_))
        return "\n".join(lines)


import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
CrossFitHingeRidgeV1.__module__ = "interpretable_regressor"

model_shorthand_name = "CrossFitHingeRidgeV1"
model_description = "From-scratch 3-fold CV ridge backbone with a single dominant-feature hinge correction activated only when cross-validated gain is consistent and meaningful"
model_defs = [(model_shorthand_name, CrossFitHingeRidgeV1())]

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
