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


class GCVSparseHingeRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Compact ridge-style linear model with two explicit interpretability constraints:
      1) hard sparsification of the linear terms
      2) at most one hinge correction, only if it improves validation MSE
    """

    def __init__(
        self,
        alpha_grid=None,
        max_active_features=6,
        coef_rel_threshold=0.12,
        val_frac=0.2,
        hinge_alpha_scale=0.8,
        min_hinge_gain=0.01,
        random_state=42,
    ):
        self.alpha_grid = alpha_grid
        self.max_active_features = max_active_features
        self.coef_rel_threshold = coef_rel_threshold
        self.val_frac = val_frac
        self.hinge_alpha_scale = hinge_alpha_scale
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
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
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

    def _sparsify(self, coef_std):
        p = len(coef_std)
        abs_w = np.abs(coef_std)
        order = np.argsort(abs_w)[::-1]
        if p == 0:
            return np.array([], dtype=int)
        top = abs_w[order[0]]
        thresh = max(float(self.coef_rel_threshold) * top, 1e-10)
        keep = [int(j) for j in order if abs_w[j] >= thresh]
        keep = keep[: max(1, int(self.max_active_features))]
        if not keep:
            keep = [int(order[0])]
        return np.array(sorted(set(keep)), dtype=int)

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
        b0, w0 = self._ridge_fit(Xtr, ytr, alpha)
        active = self._sparsify(w0)

        Xa_tr = Xtr[:, active]
        Xa_val = Xval[:, active]
        b_lin, w_lin = self._ridge_fit(Xa_tr, ytr, alpha)
        pred_val_lin = b_lin + Xa_val @ w_lin
        mse_lin = self._mse(yval, pred_val_lin)

        self.use_hinge_ = False
        best_hinge = None
        best_mse = mse_lin

        # Test one-knot hinge on top linear features only; keep the equation compact.
        top_local = np.argsort(np.abs(w_lin))[::-1][: min(2, len(active))]
        for local_j in top_local:
            xj_tr = Xa_tr[:, local_j]
            xj_val = Xa_val[:, local_j]
            for knot in np.unique(np.quantile(xj_tr, [0.25, 0.5, 0.75])):
                h_tr = np.maximum(0.0, xj_tr - float(knot))
                h_val = np.maximum(0.0, xj_val - float(knot))

                Xaug_tr = np.column_stack([Xa_tr, h_tr])
                Xaug_val = np.column_stack([Xa_val, h_val])
                b_aug, w_aug = self._ridge_fit(
                    Xaug_tr, ytr, alpha=max(alpha * float(self.hinge_alpha_scale), 1e-10)
                )
                pred_val = b_aug + Xaug_val @ w_aug
                mse_aug = self._mse(yval, pred_val)
                if mse_aug < best_mse:
                    best_mse = mse_aug
                    best_hinge = {
                        "local_j": int(local_j),
                        "knot_std": float(knot),
                        "b": float(b_aug),
                        "w_lin": np.asarray(w_aug[:-1], dtype=float),
                        "w_hinge": float(w_aug[-1]),
                    }

        rel_gain = (mse_lin - best_mse) / max(mse_lin, 1e-12)
        if best_hinge is not None and rel_gain >= float(self.min_hinge_gain):
            self.use_hinge_ = True
            b_std = best_hinge["b"]
            w_std = best_hinge["w_lin"]
        else:
            b_std = b_lin
            w_std = w_lin

        coef_raw = np.zeros(p, dtype=float)
        if len(active) > 0:
            coef_raw[active] = w_std / sigma[active]
        intercept_raw = float(b_std - np.dot(coef_raw, mu))

        if self.use_hinge_:
            local_j = best_hinge["local_j"]
            global_j = int(active[local_j])
            knot_std = best_hinge["knot_std"]
            thr_raw = float(mu[global_j] + sigma[global_j] * knot_std)
            w_h = float(best_hinge["w_hinge"])
            hinge_coef_raw = float(w_h / sigma[global_j])
            self.hinge_feature_ = global_j
            self.hinge_threshold_ = thr_raw
            self.hinge_coef_ = hinge_coef_raw

        self.intercept_ = intercept_raw
        self.coef_ = coef_raw
        self.active_features_ = active
        self.alpha_ = alpha
        self.gcv_score_ = gcv_score
        self.val_mse_linear_ = mse_lin
        self.val_mse_best_ = best_mse
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "coef_", "active_features_"])
        X = np.asarray(X, dtype=float)
        yhat = self.intercept_ + X @ self.coef_
        if getattr(self, "use_hinge_", False):
            j = int(self.hinge_feature_)
            yhat = yhat + float(self.hinge_coef_) * np.maximum(0.0, X[:, j] - float(self.hinge_threshold_))
        return yhat

    @staticmethod
    def _linear_terms(coef, max_terms=8):
        active = [j for j in range(len(coef)) if abs(float(coef[j])) > 1e-10]
        active = sorted(active, key=lambda j: -abs(float(coef[j])))[: int(max_terms)]
        return [f"{float(coef[j]):+.6f}*x{j}" for j in active], active

    def __str__(self):
        check_is_fitted(self, ["intercept_", "coef_", "active_features_"])
        terms, active = self._linear_terms(self.coef_)
        expr = f"{float(self.intercept_):.6f}"
        if terms:
            expr += " " + " ".join(terms)
        if getattr(self, "use_hinge_", False):
            expr += (
                f" {float(self.hinge_coef_):+.6f}*max(0, x{int(self.hinge_feature_)} - "
                f"{float(self.hinge_threshold_):.6f})"
            )

        lines = [
            "GCV Sparse Hinge Ridge Regressor",
            "Raw-feature equation:",
            f"y = {expr}",
            f"Active linear features: {', '.join(f'x{j}' for j in active) if active else '(none)'}",
            f"Chosen ridge alpha: {self.alpha_:.6g}",
        ]
        if getattr(self, "use_hinge_", False):
            lines.append(
                f"Hinge correction: x{self.hinge_feature_} above {self.hinge_threshold_:.6f}"
            )
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
GCVSparseHingeRidgeRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "GCVSparseHingeRidgeV1"
model_description = "GCV-selected sparse ridge equation on standardized features with hard coefficient pruning and optional single-knot hinge correction accepted only on validation gain"
model_defs = [(model_shorthand_name, GCVSparseHingeRidgeRegressor())]


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
