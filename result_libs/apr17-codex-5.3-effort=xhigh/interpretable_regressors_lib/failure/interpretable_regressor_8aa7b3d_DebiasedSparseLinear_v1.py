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


class DebiasedSparseLinearRegressor(BaseEstimator, RegressorMixin):
    """From-scratch CV lasso via proximal gradient + OLS debias on selected features."""

    def __init__(
        self,
        lambda_grid=(1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1),
        cv_folds=4,
        max_iter=300,
        tol=1e-6,
        max_active_terms=12,
        random_state=0,
    ):
        self.lambda_grid = lambda_grid
        self.cv_folds = cv_folds
        self.max_iter = max_iter
        self.tol = tol
        self.max_active_terms = max_active_terms
        self.random_state = random_state

    @staticmethod
    def _standardize(X):
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        sigma = np.where(sigma < 1e-12, 1.0, sigma)
        return mu, sigma

    @staticmethod
    def _soft_threshold(v, thresh):
        return np.sign(v) * np.maximum(np.abs(v) - thresh, 0.0)

    def _fit_lasso_pg(self, Xs, yc, lam):
        n, p = Xs.shape
        w = np.zeros(p, dtype=float)

        # Lipschitz constant of gradient for (1/2n)||y-Xw||^2
        # L = lambda_max(X^T X)/n
        svals = np.linalg.svd(Xs, full_matrices=False, compute_uv=False)
        L = float((svals[0] ** 2) / max(n, 1)) if svals.size > 0 else 1.0
        step = 1.0 / max(L, 1e-8)

        for _ in range(int(self.max_iter)):
            r = Xs @ w - yc
            grad = (Xs.T @ r) / max(n, 1)
            w_new = self._soft_threshold(w - step * grad, step * float(lam))
            if np.max(np.abs(w_new - w)) < float(self.tol):
                w = w_new
                break
            w = w_new
        return w

    def _cv_choose_lambda(self, Xs, y):
        n = Xs.shape[0]
        if n < 6:
            return float(self.lambda_grid[0])

        k = int(max(2, min(int(self.cv_folds), n)))
        rng = np.random.RandomState(self.random_state)
        perm = rng.permutation(n)
        folds = np.array_split(perm, k)

        best_lam = float(self.lambda_grid[0])
        best_mse = float("inf")

        for lam in self.lambda_grid:
            fold_mses = []
            for i in range(k):
                val_idx = folds[i]
                if len(val_idx) == 0:
                    continue
                tr_idx = np.concatenate([folds[j] for j in range(k) if j != i])
                if len(tr_idx) == 0:
                    continue

                Xtr = Xs[tr_idx]
                ytr = y[tr_idx]
                Xva = Xs[val_idx]
                yva = y[val_idx]

                ytr_mean = float(np.mean(ytr))
                yc_tr = ytr - ytr_mean
                w = self._fit_lasso_pg(Xtr, yc_tr, float(lam))
                pred_va = ytr_mean + Xva @ w
                fold_mses.append(float(np.mean((yva - pred_va) ** 2)))

            if fold_mses:
                mse = float(np.mean(fold_mses))
                if mse < best_mse:
                    best_mse = mse
                    best_lam = float(lam)

        return best_lam

    @staticmethod
    def _safe_ols(X, y):
        if X.shape[1] == 0:
            return np.zeros(0, dtype=float), float(np.mean(y))

        X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
        gram = X_aug.T @ X_aug
        rhs = X_aug.T @ y
        try:
            beta = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(gram) @ rhs
        return np.asarray(beta[1:], dtype=float), float(beta[0])

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        if p == 0:
            raise ValueError("No features provided")

        self.n_features_in_ = p
        self.x_mean_, self.x_scale_ = self._standardize(X)
        Xs = (X - self.x_mean_) / self.x_scale_

        self.lambda_ = self._cv_choose_lambda(Xs, y)
        y_mean = float(np.mean(y))
        yc = y - y_mean
        w_lasso = self._fit_lasso_pg(Xs, yc, self.lambda_)

        active = np.flatnonzero(np.abs(w_lasso) > 1e-8)
        if active.size > int(self.max_active_terms):
            order = np.argsort(np.abs(w_lasso[active]))[::-1]
            active = active[order[: int(self.max_active_terms)]]
            active.sort()

        self.active_features_ = np.asarray(active, dtype=int)

        coef_raw = np.zeros(p, dtype=float)
        if self.active_features_.size == 0:
            self.intercept_ = y_mean
            self.coef_ = coef_raw
        else:
            Xa = X[:, self.active_features_]
            w_active, b_active = self._safe_ols(Xa, y)
            coef_raw[self.active_features_] = w_active
            self.intercept_ = float(b_active)
            self.coef_ = coef_raw

        abs_coef = np.abs(self.coef_)
        max_abs = float(np.max(abs_coef)) if np.max(abs_coef) > 0 else 0.0
        self.feature_importances_ = abs_coef / max_abs if max_abs > 0 else abs_coef

        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "coef_", "active_features_"])
        X = np.asarray(X, dtype=float)
        return self.intercept_ + X @ self.coef_

    def __str__(self):
        check_is_fitted(self, ["intercept_", "coef_", "lambda_", "active_features_", "feature_importances_"])

        lines = [
            "Debiased Sparse Linear Regressor",
            "Prediction rule:",
        ]

        if self.active_features_.size == 0:
            lines.append(f"y = {self.intercept_:+.6f}")
            lines.append("(no active feature terms)")
        else:
            terms = [f"{float(self.coef_[j]):+.6f}*x{int(j)}" for j in self.active_features_]
            lines.append(f"y = {self.intercept_:+.6f} " + " ".join(terms))

        lines.append("")
        lines.append(f"lasso_lambda = {self.lambda_:.6g}")
        lines.append(f"active_features = {', '.join(f'x{int(j)}' for j in self.active_features_) if self.active_features_.size else 'none'}")
        lines.append("Feature summary (sorted by absolute coefficient):")

        order = np.argsort(np.abs(self.coef_))[::-1]
        for j in order:
            c = float(self.coef_[j])
            if abs(c) < 1e-12 and self.active_features_.size <= 15:
                continue
            lines.append(
                f"x{int(j)}: coef={c:+.6f}, importance={float(self.feature_importances_[j]):.3f}"
            )

        lines.append("")
        lines.append("Simulation recipe: multiply each shown coefficient by its feature value, sum terms, then add intercept.")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys

_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])

# Update the model shorthand name and description below to reflect the class above and any changes you make.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
DebiasedSparseLinearRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "DebiasedSparseLinear_v1"
model_description = "From-scratch proximal-gradient L1 sparse linear model with CV lambda selection, OLS debias refit on active terms, and capped active-feature equation"
model_defs = [(model_shorthand_name, DebiasedSparseLinearRegressor())]


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
