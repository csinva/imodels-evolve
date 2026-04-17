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


class ValGatedSparseHingeRidgeV1(BaseEstimator, RegressorMixin):
    """
    From-scratch ridge map with validation-gated sparsity and one-knot hinge:
    1) fit dense closed-form ridge (alpha by GCV)
    2) fit sparse ridge on top-|coef| features
    3) select dense vs sparse by validation MSE
    4) optionally add one hinge term on dominant feature if validation improves.
    """

    def __init__(
        self,
        alpha_grid=None,
        val_fraction=0.2,
        max_active_features=8,
        sparse_tolerance=1.01,
        hinge_ridge=1.0,
        hinge_gain=0.01,
        random_state=42,
        eps=1e-12,
    ):
        self.alpha_grid = alpha_grid
        self.val_fraction = val_fraction
        self.max_active_features = max_active_features
        self.sparse_tolerance = sparse_tolerance
        self.hinge_ridge = hinge_ridge
        self.hinge_gain = hinge_gain
        self.random_state = random_state
        self.eps = eps

    @staticmethod
    def _safe_scale(x):
        s = np.std(x, axis=0)
        return np.where(s > 1e-12, s, 1.0)

    def _solve_ridge_gcv(self, Xz, yc):
        n, _ = Xz.shape
        U, sing, Vt = np.linalg.svd(Xz, full_matrices=False)
        UTy = U.T @ yc
        s2 = sing ** 2

        alphas = np.logspace(-5, 4, 20) if self.alpha_grid is None else np.asarray(self.alpha_grid, dtype=float)
        best = None
        for alpha in alphas:
            filt = sing / (s2 + alpha)
            beta = Vt.T @ (filt * UTy)
            resid = yc - Xz @ beta
            mse = float(np.mean(resid ** 2))
            hat_trace = float(np.sum(s2 / (s2 + alpha)))
            denom = (1.0 - hat_trace / max(n, 1)) ** 2
            gcv = mse / max(denom, self.eps)
            if (best is None) or (gcv < best[0]):
                best = (gcv, float(alpha), beta)
        return best[1], best[2]

    @staticmethod
    def _hinge(x, t):
        return np.maximum(0.0, x - t)

    @staticmethod
    def _build_split(n, val_fraction, random_state):
        n_val = int(max(20, round(n * val_fraction)))
        n_val = min(max(n_val, 1), max(n - 1, 1))
        rng = np.random.RandomState(random_state)
        idx = np.arange(n)
        rng.shuffle(idx)
        val_idx = idx[:n_val]
        tr_idx = idx[n_val:]
        if tr_idx.size == 0:
            tr_idx = idx[:-1]
            val_idx = idx[-1:]
        return tr_idx, val_idx

    def _fit_linear_on_indices(self, X, y, feat_idx):
        x_mean = np.mean(X, axis=0)
        x_scale = self._safe_scale(X)
        Xz = (X - x_mean) / x_scale
        y_mean = float(np.mean(y))
        yc = y - y_mean

        Xsel = Xz[:, feat_idx]
        alpha, beta_sel_z = self._solve_ridge_gcv(Xsel, yc)

        coef = np.zeros(X.shape[1], dtype=float)
        coef[feat_idx] = beta_sel_z / x_scale[feat_idx]
        intercept = float(y_mean - np.dot(coef, x_mean))
        return alpha, coef, intercept

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        self.n_features_in_ = p

        tr_idx, val_idx = self._build_split(n, float(self.val_fraction), int(self.random_state))
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xval, yval = X[val_idx], y[val_idx]

        # Dense linear candidate.
        all_idx = np.arange(p, dtype=int)
        dense_alpha, dense_coef, dense_intercept = self._fit_linear_on_indices(Xtr, ytr, all_idx)
        dense_val_pred = dense_intercept + Xval @ dense_coef
        dense_val_mse = float(np.mean((yval - dense_val_pred) ** 2))

        # Sparse linear candidate based on dense coefficient magnitudes.
        abs_coef = np.abs(dense_coef)
        k = int(min(max(2, self.max_active_features), p))
        active_idx = np.argsort(-abs_coef)[:k]
        sparse_alpha, sparse_coef, sparse_intercept = self._fit_linear_on_indices(Xtr, ytr, active_idx)
        sparse_val_pred = sparse_intercept + Xval @ sparse_coef
        sparse_val_mse = float(np.mean((yval - sparse_val_pred) ** 2))

        choose_sparse = sparse_val_mse <= dense_val_mse * float(self.sparse_tolerance)
        chosen_coef = sparse_coef if choose_sparse else dense_coef
        chosen_intercept = sparse_intercept if choose_sparse else dense_intercept
        chosen_alpha = sparse_alpha if choose_sparse else dense_alpha
        chosen_active = np.where(np.abs(chosen_coef) > 1e-12)[0].astype(int)

        # Optional one-knot hinge correction, validation-gated.
        if chosen_active.size == 0:
            dom = int(np.argmax(abs_coef))
        else:
            dom = int(chosen_active[np.argmax(np.abs(chosen_coef[chosen_active]))])

        base_tr_pred = chosen_intercept + Xtr @ chosen_coef
        base_val_pred = chosen_intercept + Xval @ chosen_coef
        base_val_mse = float(np.mean((yval - base_val_pred) ** 2))

        q_grid = np.quantile(Xtr[:, dom], [0.2, 0.4, 0.6, 0.8])
        q_grid = np.unique(q_grid)

        best_hinge = (base_val_mse, 0.0, 0.0)
        for t in q_grid:
            htr = self._hinge(Xtr[:, dom], float(t))
            hval = self._hinge(Xval[:, dom], float(t))
            denom = float(htr @ htr + self.hinge_ridge)
            gamma = float(htr @ (ytr - base_tr_pred)) / max(denom, self.eps)
            val_pred = base_val_pred + gamma * hval
            val_mse = float(np.mean((yval - val_pred) ** 2))
            if val_mse < best_hinge[0]:
                best_hinge = (val_mse, float(t), gamma)

        hinge_gain = (base_val_mse - best_hinge[0]) / max(base_val_mse, self.eps)
        use_hinge = bool(hinge_gain >= float(self.hinge_gain))

        self.alpha_ = float(chosen_alpha)
        self.coef_ = chosen_coef
        self.intercept_ = float(chosen_intercept)
        self.used_sparse_backbone_ = bool(choose_sparse)
        self.dominant_feature_ = dom
        self.hinge_knot_ = float(best_hinge[1])
        self.hinge_coef_ = float(best_hinge[2]) if use_hinge else 0.0
        self.use_hinge_ = use_hinge
        self.val_mse_linear_ = float(base_val_mse)
        self.val_mse_final_ = float(best_hinge[0] if use_hinge else base_val_mse)
        self.hinge_relative_gain_ = float(hinge_gain if use_hinge else 0.0)
        self.active_features_ = np.where(np.abs(self.coef_) > 1e-12)[0].astype(int)
        self.negligible_features_ = np.where(np.abs(self.coef_) <= 1e-12)[0].astype(int)
        return self

    def predict(self, X):
        check_is_fitted(self, ["coef_", "intercept_", "dominant_feature_", "hinge_knot_", "hinge_coef_"])
        X = np.asarray(X, dtype=float)
        yhat = self.intercept_ + X @ self.coef_
        if self.use_hinge_:
            yhat = yhat + self.hinge_coef_ * self._hinge(X[:, self.dominant_feature_], self.hinge_knot_)
        return yhat

    def __str__(self):
        check_is_fitted(self, ["coef_", "intercept_", "dominant_feature_", "hinge_knot_", "hinge_coef_"])
        lines = [
            "Validation-Gated Sparse Hinge Ridge Regressor",
            "Prediction equation:",
            "  y_lin = intercept + sum_j coef_j * xj",
            f"  intercept = {self.intercept_:+.6f}",
            f"  ridge alpha (GCV) = {self.alpha_:.6g}",
            f"  backbone chosen by validation: {'sparse' if self.used_sparse_backbone_ else 'dense'}",
            "",
            "Coefficients in raw-feature space:",
        ]

        order = np.argsort(-np.abs(self.coef_))
        for j in order:
            lines.append(f"  x{int(j)}: {float(self.coef_[j]):+.6f}")

        if self.use_hinge_:
            j = int(self.dominant_feature_)
            t = float(self.hinge_knot_)
            g = float(self.hinge_coef_)
            lines.extend([
                "",
                "One-knot hinge correction:",
                f"  correction = ({g:+.6f}) * max(0, x{j} - ({t:+.6f}))",
                "  final y = y_lin + correction",
            ])
        else:
            lines.extend([
                "",
                "No hinge correction retained (validation gain too small).",
                "Final y = y_lin",
            ])

        if len(self.active_features_) > 0:
            lines.append("Active features: " + ", ".join(f"x{int(i)}" for i in self.active_features_))
        if len(self.negligible_features_) > 0:
            lines.append("Negligible features: " + ", ".join(f"x{int(i)}" for i in self.negligible_features_))

        lines.append(f"Validation MSE (linear backbone): {self.val_mse_linear_:.6f}")
        lines.append(f"Validation MSE (final model): {self.val_mse_final_:.6f}")
        if self.use_hinge_:
            lines.append(f"Relative validation MSE gain from hinge: {self.hinge_relative_gain_:.2%}")
        return "\n".join(lines)


import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
ValGatedSparseHingeRidgeV1.__module__ = "interpretable_regressor"

model_shorthand_name = "ValGatedSparseHingeRidgeV1"
model_description = "From-scratch ridge equation that validation-selects dense vs top-k sparse backbone, then optionally adds one dominant-feature hinge correction only when validation gain is meaningful"
model_defs = [(model_shorthand_name, ValGatedSparseHingeRidgeV1())]

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
