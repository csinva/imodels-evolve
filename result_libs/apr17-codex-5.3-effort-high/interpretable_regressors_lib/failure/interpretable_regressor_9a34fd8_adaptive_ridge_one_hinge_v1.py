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


class AdaptiveRidgeHingeRegressor(BaseEstimator, RegressorMixin):
    """
    Dense ridge backbone with a single optional hinge correction.

    1) Fit a standardized ridge model with alpha selected by GCV.
    2) Find the most influential feature and test one hinge basis
       max(0, x_j - knot) on a validation split.
    3) Keep the hinge only when it improves validation MSE meaningfully.
    """

    def __init__(
        self,
        alpha_grid=None,
        val_frac=0.2,
        knot_quantiles=(0.25, 0.4, 0.5, 0.6, 0.75),
        min_hinge_gain=0.005,
        random_state=42,
    ):
        self.alpha_grid = alpha_grid
        self.val_frac = val_frac
        self.knot_quantiles = knot_quantiles
        self.min_hinge_gain = min_hinge_gain
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

    @staticmethod
    def _ridge_with_basis(X, y, alpha, basis=None):
        if basis is None:
            D = np.hstack([np.ones((X.shape[0], 1), dtype=float), X])
            reg = np.zeros(X.shape[1] + 1, dtype=float)
            reg[1:] = float(alpha)
        else:
            b = np.asarray(basis, dtype=float).reshape(-1, 1)
            D = np.hstack([np.ones((X.shape[0], 1), dtype=float), X, b])
            reg = np.zeros(X.shape[1] + 2, dtype=float)
            reg[1:] = float(alpha)

        A = D.T @ D + np.diag(reg)
        rhs = D.T @ y
        try:
            beta = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(A) @ rhs
        return D, beta

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
        _, beta_base = self._ridge_with_basis(Xtr, ytr, alpha, basis=None)
        b_base = float(beta_base[0])
        w_base = np.asarray(beta_base[1:], dtype=float)
        base_pred = b_base + Xval @ w_base
        mse_base = self._mse(yval, base_pred)

        pivot_idx = int(np.argmax(np.abs(w_base)))
        xj_tr = Xtr[:, pivot_idx]
        xj_val = Xval[:, pivot_idx]

        best = {
            "use_hinge": False,
            "knot": 0.0,
            "beta": beta_base,
            "mse": mse_base,
        }

        for q in self.knot_quantiles:
            knot = float(np.quantile(xj_tr, float(q)))
            hinge_tr = np.maximum(0.0, xj_tr - knot)
            _, beta_h = self._ridge_with_basis(Xtr, ytr, alpha, basis=hinge_tr)
            pred_h = (
                float(beta_h[0])
                + Xval @ np.asarray(beta_h[1 : 1 + p], dtype=float)
                + float(beta_h[-1]) * np.maximum(0.0, xj_val - knot)
            )
            mse_h = self._mse(yval, pred_h)
            if mse_h < best["mse"]:
                best = {
                    "use_hinge": True,
                    "knot": knot,
                    "beta": np.asarray(beta_h, dtype=float),
                    "mse": mse_h,
                }

        rel_gain = (mse_base - best["mse"]) / max(mse_base, 1e-10)
        use_hinge = bool(best["use_hinge"] and rel_gain >= float(self.min_hinge_gain))

        if use_hinge:
            beta = np.asarray(best["beta"], dtype=float)
            b_std = float(beta[0])
            w_std = np.asarray(beta[1 : 1 + p], dtype=float)
            hinge_w_std = float(beta[-1])
            knot_std = float(best["knot"])
        else:
            b_std = b_base
            w_std = w_base
            hinge_w_std = 0.0
            knot_std = 0.0

        coef_raw = w_std / sigma
        intercept_raw = float(b_std - np.dot(coef_raw, mu))

        if use_hinge:
            knot_raw = float(mu[pivot_idx] + sigma[pivot_idx] * knot_std)
            hinge_w_raw = float(hinge_w_std / sigma[pivot_idx])
        else:
            knot_raw = 0.0
            hinge_w_raw = 0.0

        active_raw = np.where(np.abs(coef_raw) > 1e-12)[0]
        self.alpha_ = alpha
        self.gcv_score_ = gcv_score
        self.val_mse_base_ = mse_base
        self.val_mse_final_ = float(best["mse"]) if use_hinge else mse_base
        self.hinge_gain_ = float(rel_gain)

        self.intercept_ = intercept_raw
        self.coef_ = np.asarray(coef_raw, dtype=float)
        self.active_features_ = np.asarray(active_raw, dtype=int)
        self.hinge_feature_ = int(pivot_idx) if use_hinge else -1
        self.hinge_knot_ = knot_raw
        self.hinge_weight_ = hinge_w_raw
        self.use_hinge_ = bool(use_hinge)
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "coef_", "active_features_"])
        X = np.asarray(X, dtype=float)
        y = self.intercept_ + X @ self.coef_
        if self.use_hinge_ and self.hinge_feature_ >= 0:
            y = y + self.hinge_weight_ * np.maximum(0.0, X[:, self.hinge_feature_] - self.hinge_knot_)
        return y

    @staticmethod
    def _linear_terms(coef):
        active = [j for j in range(len(coef)) if abs(float(coef[j])) > 1e-12]
        active = sorted(active, key=lambda j: -abs(float(coef[j])))
        return [f"{float(coef[j]):+.6f}*x{j}" for j in active], active

    def __str__(self):
        check_is_fitted(self, ["intercept_", "coef_", "active_features_", "use_hinge_"])
        terms, active = self._linear_terms(self.coef_)
        expr = f"{float(self.intercept_):.6f}"
        if terms:
            expr += " " + " ".join(terms)
        if self.use_hinge_ and self.hinge_feature_ >= 0 and abs(self.hinge_weight_) > 1e-12:
            expr += (
                f" {float(self.hinge_weight_):+.6f}"
                f"*max(0, x{int(self.hinge_feature_)}-{float(self.hinge_knot_):.6f})"
            )
        lines = [
            "Adaptive Ridge + One Hinge Regressor",
            f"Chosen ridge alpha: {self.alpha_:.6g}",
            f"Validation base MSE: {self.val_mse_base_:.6f}",
            f"Validation final MSE: {self.val_mse_final_:.6f}",
            f"Hinge relative gain: {self.hinge_gain_:.4f}",
            "Raw-feature prediction equation:",
            f"y = {expr}",
            f"Linear active features: {', '.join(f'x{j}' for j in active) if active else '(none)'}",
            (
                f"Hinge term: on x{self.hinge_feature_} with knot {self.hinge_knot_:.6f}"
                if self.use_hinge_
                else "Hinge term: not used"
            ),
        ]
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
AdaptiveRidgeHingeRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "AdaptiveRidgeOneHingeV1"
model_description = "Dense GCV ridge backbone with optional single-feature hinge correction retained only when validation gain is meaningful, while preserving an explicit simulatable equation"
model_defs = [(model_shorthand_name, AdaptiveRidgeHingeRegressor())]


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
