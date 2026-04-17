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


class QuantizedSparseRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    GCV ridge with adaptive sparsity and coefficient quantization.

    The model first fits ridge on standardized features, keeps the smallest feature
    set that explains most coefficient mass, then chooses a quantization step on a
    validation split to make the final equation easier to simulate.
    """

    def __init__(
        self,
        alpha_grid=None,
        keep_mass=0.98,
        min_active=3,
        max_active=16,
        val_frac=0.2,
        quant_steps=(0.0, 0.01, 0.02, 0.05, 0.1),
        complexity_penalty=0.002,
        random_state=42,
    ):
        self.alpha_grid = alpha_grid
        self.keep_mass = keep_mass
        self.min_active = min_active
        self.max_active = max_active
        self.val_frac = val_frac
        self.quant_steps = quant_steps
        self.complexity_penalty = complexity_penalty
        self.random_state = random_state

    @staticmethod
    def _safe_scale(X):
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        sigma = np.where(sigma < 1e-8, 1.0, sigma)
        return mu, sigma

    @staticmethod
    def _ridge_fit(X, y, alpha):
        n, p = X.shape
        D = np.hstack([np.ones((n, 1), dtype=float), X])
        reg = np.zeros(p + 1, dtype=float)
        reg[1:] = max(float(alpha), 0.0)
        A = D.T @ D + np.diag(reg)
        b = D.T @ y
        try:
            beta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(A) @ b
        return float(beta[0]), np.asarray(beta[1:], dtype=float)

    @staticmethod
    def _mse(y_true, y_pred):
        return float(np.mean((y_true - y_pred) ** 2))

    def _select_alpha_gcv(self, X, y):
        n = X.shape[0]
        if self.alpha_grid is None:
            grid = np.logspace(-4, 2, 11)
        else:
            grid = np.asarray(self.alpha_grid, dtype=float)
        grid = np.maximum(grid, 1e-10)

        y_centered = y - np.mean(y)
        U, s, _ = np.linalg.svd(X, full_matrices=False)
        Uy = U.T @ y_centered
        s2 = s ** 2

        best_alpha = float(grid[0])
        best_gcv = float("inf")
        for alpha in grid:
            shrink = s2 / (s2 + alpha)
            yhat_centered = U @ (shrink * Uy)
            resid = y_centered - yhat_centered
            mse = float(np.mean(resid ** 2))
            df = float(np.sum(shrink))
            denom = max((1.0 - df / max(n, 1)) ** 2, 1e-8)
            gcv = mse / denom
            if gcv < best_gcv:
                best_gcv = gcv
                best_alpha = float(alpha)
        return best_alpha, best_gcv

    def _select_active(self, coef_std):
        p = len(coef_std)
        order = np.argsort(np.abs(coef_std))[::-1]
        mass = np.abs(coef_std).sum()
        if mass <= 1e-12:
            return np.array([int(order[0])], dtype=int)

        cum = 0.0
        selected = []
        for j in order:
            selected.append(int(j))
            cum += abs(float(coef_std[j]))
            if len(selected) >= int(self.max_active):
                break
            if len(selected) >= int(self.min_active) and (cum / mass) >= float(self.keep_mass):
                break
        if not selected:
            selected = [int(order[0])]
        selected = sorted(set(selected))
        return np.asarray(selected, dtype=int)

    @staticmethod
    def _quantize(arr, step):
        arr = np.asarray(arr, dtype=float)
        if step <= 0:
            return arr.copy()
        return np.round(arr / step) * step

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        self.n_features_in_ = p

        mu, sigma = self._safe_scale(X)
        Xs = (X - mu) / sigma

        rng = np.random.RandomState(self.random_state)
        perm = rng.permutation(n)
        n_val = max(20, int(self.val_frac * n))
        if n - n_val < 20:
            n_val = max(1, n // 5)
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]
        if len(tr_idx) < 20:
            tr_idx = perm
            val_idx = perm

        Xtr = Xs[tr_idx]
        ytr = y[tr_idx]
        Xval = Xs[val_idx]
        yval = y[val_idx]

        alpha, gcv_score = self._select_alpha_gcv(Xtr, ytr)
        b_dense, w_dense = self._ridge_fit(Xtr, ytr, alpha)
        active = self._select_active(w_dense)

        Xa_tr = Xtr[:, active]
        Xa_val = Xval[:, active]
        b_sparse, w_sparse = self._ridge_fit(Xa_tr, ytr, alpha)
        pred_sparse = b_sparse + Xa_val @ w_sparse
        mse_sparse = self._mse(yval, pred_sparse)

        best = None
        for step in np.asarray(self.quant_steps, dtype=float):
            wq = self._quantize(w_sparse, float(step))
            bq = float(self._quantize(np.array([b_sparse]), float(step))[0])
            pred = bq + Xa_val @ wq
            mse = self._mse(yval, pred)
            nonzero = int(np.sum(np.abs(wq) > 1e-12))
            score = mse + float(self.complexity_penalty) * nonzero + 0.3 * float(self.complexity_penalty) * float(step > 0)
            if (best is None) or (score < best["score"]):
                best = {
                    "step": float(step),
                    "b": bq,
                    "w": np.asarray(wq, dtype=float),
                    "mse": mse,
                    "nonzero": nonzero,
                    "score": score,
                }

        w_std_full = np.zeros(p, dtype=float)
        w_std_full[active] = best["w"]

        coef_raw = w_std_full / sigma
        intercept_raw = float(best["b"] - np.dot(coef_raw, mu))

        active_raw = np.where(np.abs(coef_raw) > 1e-12)[0]
        if len(active_raw) == 0:
            j = int(np.argmax(np.abs(coef_raw))) if p > 0 else 0
            if p > 0:
                active_raw = np.array([j], dtype=int)

        self.intercept_ = intercept_raw
        self.coef_ = np.asarray(coef_raw, dtype=float)
        self.active_features_ = np.asarray(active_raw, dtype=int)
        self.alpha_ = alpha
        self.gcv_score_ = gcv_score
        self.val_mse_sparse_ = mse_sparse
        self.val_mse_quantized_ = float(best["mse"])
        self.quantization_step_ = float(best["step"])
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "coef_", "active_features_"])
        X = np.asarray(X, dtype=float)
        return self.intercept_ + X @ self.coef_

    @staticmethod
    def _linear_terms(coef):
        active = [j for j in range(len(coef)) if abs(float(coef[j])) > 1e-12]
        active = sorted(active, key=lambda j: -abs(float(coef[j])))
        return [f"{float(coef[j]):+.6f}*x{j}" for j in active], active

    def __str__(self):
        check_is_fitted(self, ["intercept_", "coef_", "active_features_"])
        terms, active = self._linear_terms(self.coef_)
        expr = f"{float(self.intercept_):.6f}"
        if terms:
            expr += " " + " ".join(terms)
        lines = [
            "Quantized Sparse Ridge Regressor",
            f"Chosen ridge alpha: {self.alpha_:.6g}",
            f"Coefficient quantization step: {self.quantization_step_:.4f}",
            "Raw-feature prediction equation:",
            f"y = {expr}",
            f"Active features: {', '.join(f'x{j}' for j in active) if active else '(none)'}",
        ]
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
QuantizedSparseRidgeRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "QuantizedSparseRidgeV1"
model_description = "GCV ridge with coefficient-mass feature selection and validation-chosen coefficient quantization to produce a compact, simulation-friendly explicit equation"
model_defs = [(model_shorthand_name, QuantizedSparseRidgeRegressor())]


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
