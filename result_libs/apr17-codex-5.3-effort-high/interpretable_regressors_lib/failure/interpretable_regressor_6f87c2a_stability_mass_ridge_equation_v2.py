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


class StabilityMassRidgeEquationV2(BaseEstimator, RegressorMixin):
    """
    Closed-form CV ridge in standardized space, mapped back to a raw-feature equation.

    Then applies conservative validation-gated top-k sparsification so the final
    equation stays simulatable without incurring large predictive regressions.
    """

    def __init__(
        self,
        alpha_grid=None,
        cv_folds=5,
        val_fraction=0.2,
        k_candidates=(5, 8, 12, 20, 35),
        mse_tolerance=0.015,
        random_state=42,
        display_precision=5,
    ):
        self.alpha_grid = alpha_grid
        self.cv_folds = cv_folds
        self.val_fraction = val_fraction
        self.k_candidates = k_candidates
        self.mse_tolerance = mse_tolerance
        self.random_state = random_state
        self.display_precision = display_precision

    @staticmethod
    def _safe_scale(X):
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        sigma = np.where(sigma > 1e-12, sigma, 1.0)
        return mu.astype(float), sigma.astype(float)

    @staticmethod
    def _solve_ridge(D, y, alpha):
        reg = np.eye(D.shape[1], dtype=float)
        reg[0, 0] = 0.0
        A = D.T @ D + float(alpha) * reg
        b = D.T @ y
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(A) @ b

    @classmethod
    def _fit_linear(cls, Xs, y, alpha):
        n = Xs.shape[0]
        D = np.column_stack([np.ones(n, dtype=float), Xs])
        return cls._solve_ridge(D, y, alpha)

    @staticmethod
    def _predict_linear(Xs, theta):
        return float(theta[0]) + Xs @ np.asarray(theta[1:], dtype=float)

    def _select_alpha_cv(self, Xs, y):
        alphas = np.asarray(self.alpha_grid, dtype=float) if self.alpha_grid is not None else np.logspace(-6, 3, 16)
        n = Xs.shape[0]
        if n < 20:
            return float(alphas[0]), float("nan")

        n_folds = int(max(2, min(int(self.cv_folds), 6)))
        rng = np.random.RandomState(self.random_state)
        idx = np.arange(n)
        rng.shuffle(idx)
        folds = np.array_split(idx, n_folds)

        best_alpha = float(alphas[len(alphas) // 2])
        best_mse = float("inf")
        for alpha in alphas:
            mses = []
            for k in range(n_folds):
                va = folds[k]
                tr = np.concatenate([folds[j] for j in range(n_folds) if j != k])
                if len(tr) < 8 or len(va) < 4:
                    continue
                theta = self._fit_linear(Xs[tr], y[tr], float(alpha))
                pred = self._predict_linear(Xs[va], theta)
                mses.append(float(np.mean((y[va] - pred) ** 2)))
            if not mses:
                continue
            mse = float(np.mean(mses))
            if mse < best_mse:
                best_mse = mse
                best_alpha = float(alpha)
        return best_alpha, best_mse

    @staticmethod
    def _split_indices(n, val_fraction, seed):
        rng = np.random.RandomState(seed)
        idx = np.arange(n)
        rng.shuffle(idx)
        n_val = max(16, int(float(val_fraction) * n))
        n_val = min(max(1, n - 8), n_val)
        return idx[n_val:], idx[:n_val]

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        self.n_features_in_ = p

        mu, sigma = self._safe_scale(X)
        Xs = (X - mu) / sigma

        alpha, cv_mse = self._select_alpha_cv(Xs, y)
        theta_dense = self._fit_linear(Xs, y, alpha)

        dense_coef_std = theta_dense[1:]
        order = np.argsort(-np.abs(dense_coef_std))

        tr_idx, va_idx = self._split_indices(n, self.val_fraction, self.random_state)
        X_tr = Xs[tr_idx]
        y_tr = y[tr_idx]
        X_va = Xs[va_idx]
        y_va = y[va_idx]

        def fit_subset(idx):
            theta_sub = self._fit_linear(X_tr[:, idx], y_tr, alpha)
            pred_va = self._predict_linear(X_va[:, idx], theta_sub)
            mse_va = float(np.mean((y_va - pred_va) ** 2))
            return theta_sub, mse_va

        theta_full_tr = self._fit_linear(X_tr, y_tr, alpha)
        mse_full = float(np.mean((y_va - self._predict_linear(X_va, theta_full_tr)) ** 2))

        l1 = np.abs(dense_coef_std)
        denom = float(np.sum(l1)) if float(np.sum(l1)) > 1e-12 else 1.0
        csum = np.cumsum(l1[order]) / denom
        mass_k = int(np.searchsorted(csum, 0.995) + 1)

        k_pool = {p, max(1, mass_k)}
        for k in self.k_candidates:
            k_pool.add(max(1, min(p, int(k))))
        k_pool = sorted(k_pool)

        candidates = []
        for k in k_pool:
            idx = np.sort(order[:k])
            theta_sub, mse_sub = fit_subset(idx)
            candidates.append((int(k), idx, theta_sub, mse_sub))

        best_mse = min(m for _, _, _, m in candidates)
        allowed = [c for c in candidates if c[3] <= best_mse * (1.0 + float(self.mse_tolerance))]
        k_sel, idx_sel, _, mse_sel = min(allowed, key=lambda t: t[0])

        use_dense = mse_full <= mse_sel * (1.0 + 0.002)
        if use_dense:
            theta_final_std = self._fit_linear(Xs, y, alpha)
            val_mse_final = mse_full
            selection_mode = "dense"
        else:
            idx_final = np.sort(idx_sel)
            theta_sub_final = self._fit_linear(Xs[:, idx_final], y, alpha)
            theta_final_std = np.zeros(p + 1, dtype=float)
            theta_final_std[0] = theta_sub_final[0]
            theta_final_std[1:][idx_final] = theta_sub_final[1:]
            val_mse_final = mse_sel
            selection_mode = "sparse"

        coef_std = theta_final_std[1:]
        coef_raw = coef_std / sigma
        intercept_raw = float(theta_final_std[0] - np.dot(coef_raw, mu))

        self.feature_mean_ = mu
        self.feature_scale_ = sigma
        self.alpha_ = float(alpha)
        self.alpha_cv_mse_ = float(cv_mse)
        self.linear_coef_raw_ = coef_raw.astype(float)
        self.intercept_raw_ = intercept_raw
        self.selected_features_ = np.asarray(np.where(np.abs(self.linear_coef_raw_) > 1e-12)[0], dtype=int)
        self.selection_mode_ = selection_mode
        self.selected_k_ = int(k_sel)
        self.validation_mse_sparse_best_ = float(best_mse)
        self.validation_mse_selected_ = float(val_mse_final)
        self.validation_mse_dense_ = float(mse_full)

        abs_coef = np.abs(self.linear_coef_raw_)
        self.dominant_feature_ = int(np.argmax(abs_coef)) if p > 0 else 0

        return self

    def predict(self, X):
        check_is_fitted(self, ["linear_coef_raw_", "intercept_raw_", "selected_features_"])
        X = np.asarray(X, dtype=float)
        return float(self.intercept_raw_) + X @ self.linear_coef_raw_

    def _format_equation(self):
        prec = int(self.display_precision)
        parts = [f"{self.intercept_raw_:+.{prec}f}"]
        for j, c in enumerate(self.linear_coef_raw_):
            if abs(float(c)) > 1e-12:
                parts.append(f"{float(c):+.{prec}f}*x{j}")
        return " ".join(parts)

    def __str__(self):
        check_is_fitted(self, ["selected_features_", "linear_coef_raw_", "intercept_raw_"])
        n = int(self.n_features_in_)
        prec = int(self.display_precision)

        active_set = set(int(i) for i in self.selected_features_.tolist())
        negligible = [j for j in range(n) if j not in active_set]

        lines = [
            "Stability-Mass Ridge Equation Regressor V2",
            "Prediction is one explicit raw-feature linear equation.",
            "y = " + self._format_equation(),
            "",
            "How to simulate:",
            "1) Multiply each listed coefficient by its feature value xj.",
            "2) Sum all terms and add the intercept.",
            "",
            "Active features: " + (", ".join(f"x{j}" for j in self.selected_features_) if len(self.selected_features_) else "none"),
            "Negligible features: " + (", ".join(f"x{j}" for j in negligible) if negligible else "none"),
            "",
            f"Dominant global feature: x{self.dominant_feature_}",
            f"Ridge alpha (CV): {self.alpha_:.6g}",
            f"Selection mode: {self.selection_mode_}",
            f"Validation MSE dense={self.validation_mse_dense_:.{prec}f}, selected={self.validation_mse_selected_:.{prec}f}, sparse_best={self.validation_mse_sparse_best_:.{prec}f}",
        ]
        return "\n".join(lines)


import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
StabilityMassRidgeEquationV2.__module__ = "interpretable_regressor"

model_shorthand_name = "StabilityMassRidgeEquationV2"
model_description = "From-scratch CV ridge mapped to an explicit raw-feature equation, with conservative validation-gated top-k mass sparsification for simulatable linear predictions"
model_defs = [(model_shorthand_name, StabilityMassRidgeEquationV2())]

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
