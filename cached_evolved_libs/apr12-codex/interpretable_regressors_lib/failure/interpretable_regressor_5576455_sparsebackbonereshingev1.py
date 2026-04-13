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
from sklearn.linear_model import LassoCV
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class SparseBackboneResidualHingeRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse linear equation plus a few residual hinge terms.

    Workflow:
    1) Fit a sparse linear backbone using L1 selection.
    2) Add up to a small number of hinge terms max(0, x_j - t) greedily on residuals.
    3) Refit one compact ridge equation on the selected basis.
    """

    def __init__(
        self,
        max_linear_terms=8,
        max_hinges=2,
        lasso_cv=3,
        lasso_n_alphas=36,
        alpha_grid=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
        knot_quantiles=(0.15, 0.3, 0.5, 0.7, 0.85),
        val_frac=0.2,
        min_val_samples=80,
        min_hinge_gain=0.002,
        min_coef_abs=1e-7,
        random_state=42,
    ):
        self.max_linear_terms = max_linear_terms
        self.max_hinges = max_hinges
        self.lasso_cv = lasso_cv
        self.lasso_n_alphas = lasso_n_alphas
        self.alpha_grid = alpha_grid
        self.knot_quantiles = knot_quantiles
        self.val_frac = val_frac
        self.min_val_samples = min_val_samples
        self.min_hinge_gain = min_hinge_gain
        self.min_coef_abs = min_coef_abs
        self.random_state = random_state

    @staticmethod
    def _ridge_closed_form(Z, y, alpha):
        p = Z.shape[1]
        reg = float(alpha) * np.eye(p)
        reg[0, 0] = 0.0
        return np.linalg.solve(Z.T @ Z + reg, Z.T @ y)

    def _impute(self, X):
        X = np.asarray(X, dtype=float).copy()
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(self.feature_medians_, np.where(bad)[1])
        return X

    def _split_idx(self, n):
        if n < int(self.min_val_samples) + 20:
            idx = np.arange(n)
            return idx, idx
        rng = np.random.RandomState(self.random_state)
        perm = rng.permutation(n)
        n_val = max(int(round(float(self.val_frac) * n)), int(self.min_val_samples))
        n_val = min(n_val, n // 2)
        return perm[n_val:], perm[:n_val]

    def _fit_sparse_linear(self, Xtr, ytr, Xval, yval):
        p = Xtr.shape[1]
        x_mean = Xtr.mean(axis=0)
        x_std = Xtr.std(axis=0)
        x_std[x_std < 1e-12] = 1.0

        Xs_tr = (Xtr - x_mean) / x_std
        selector = LassoCV(
            cv=int(self.lasso_cv),
            random_state=self.random_state,
            n_alphas=int(self.lasso_n_alphas),
            max_iter=8000,
        )
        selector.fit(Xs_tr, ytr)

        dense_coef = selector.coef_ / x_std
        dense_intercept = float(selector.intercept_ - np.dot(dense_coef, x_mean))
        active = np.where(np.abs(dense_coef) > 1e-8)[0]
        if active.size == 0:
            active = np.array([int(np.argmax(np.abs(dense_coef)))], dtype=int)
        if active.size > int(self.max_linear_terms):
            order = np.argsort(np.abs(dense_coef[active]))[::-1]
            active = active[order[: int(self.max_linear_terms)]]

        # Simple OLS refit on selected features for cleaner coefficients.
        Ztr = np.column_stack([np.ones(Xtr.shape[0]), Xtr[:, active]])
        beta = np.linalg.lstsq(Ztr, ytr, rcond=None)[0]
        intercept = float(beta[0])
        coef = np.zeros(p, dtype=float)
        coef[active] = beta[1:]

        pred_tr = intercept + Xtr @ coef
        pred_val = intercept + Xval @ coef
        mse_tr = float(np.mean((ytr - pred_tr) ** 2))
        mse_val = float(np.mean((yval - pred_val) ** 2))

        # Fallback if OLS refit destabilizes.
        val_fallback = float(np.mean((yval - (dense_intercept + Xval @ dense_coef)) ** 2))
        if not np.isfinite(mse_val) or mse_val > val_fallback * 1.15:
            intercept = dense_intercept
            coef = dense_coef
            active = np.where(np.abs(coef) > 1e-8)[0]
            pred_tr = intercept + Xtr @ coef
            pred_val = intercept + Xval @ coef
            mse_tr = float(np.mean((ytr - pred_tr) ** 2))
            mse_val = float(np.mean((yval - pred_val) ** 2))

        return {
            "intercept": intercept,
            "coef": coef,
            "active": active,
            "pred_tr": pred_tr,
            "pred_val": pred_val,
            "mse_tr": mse_tr,
            "mse_val": mse_val,
        }

    @staticmethod
    def _hinge(x, t):
        return np.maximum(0.0, x - float(t))

    def _candidate_features(self, Xtr, residual):
        x_center = Xtr - Xtr.mean(axis=0)
        denom = Xtr.std(axis=0)
        denom[denom < 1e-12] = 1.0
        rc = residual - residual.mean()
        scores = np.abs((x_center.T @ rc) / (len(residual) * denom))
        order = np.argsort(scores)[::-1]
        topk = min(max(int(self.max_linear_terms) + 4, 6), Xtr.shape[1])
        return [int(j) for j in order[:topk]]

    def _fit(self, X, y):
        n, p = X.shape
        tr_idx, val_idx = self._split_idx(n)
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xval, yval = X[val_idx], y[val_idx]

        base = self._fit_sparse_linear(Xtr, ytr, Xval, yval)
        hinge_terms = []
        cur_pred_tr = base["pred_tr"].copy()
        cur_pred_val = base["pred_val"].copy()
        cur_val_mse = float(np.mean((yval - cur_pred_val) ** 2))

        for _ in range(int(self.max_hinges)):
            residual_tr = ytr - cur_pred_tr
            candidate_feats = self._candidate_features(Xtr, residual_tr)

            best_local = None
            for j in candidate_feats:
                xj_tr = Xtr[:, j]
                xj_val = Xval[:, j]
                knots = np.unique(np.quantile(xj_tr, self.knot_quantiles))
                if knots.size == 0:
                    continue
                for t in knots:
                    htr = self._hinge(xj_tr, t)
                    hval = self._hinge(xj_val, t)
                    denom = float(np.dot(htr, htr) + 1e-8)
                    b = float(np.dot(htr, residual_tr) / denom)
                    trial_pred_val = cur_pred_val + b * hval
                    mse_val = float(np.mean((yval - trial_pred_val) ** 2))
                    if best_local is None or mse_val < best_local["mse_val"]:
                        best_local = {
                            "feature": int(j),
                            "knot": float(t),
                            "coef": float(b),
                            "mse_val": mse_val,
                            "pred_tr": cur_pred_tr + b * htr,
                            "pred_val": trial_pred_val,
                        }

            if best_local is None:
                break

            gain = cur_val_mse - best_local["mse_val"]
            if gain < float(self.min_hinge_gain) * max(cur_val_mse, 1e-8):
                break

            hinge_terms.append((best_local["feature"], best_local["knot"]))
            cur_pred_tr = best_local["pred_tr"]
            cur_pred_val = best_local["pred_val"]
            cur_val_mse = best_local["mse_val"]

        # Joint refit over intercept + active linear + selected hinges.
        linear_active = np.where(np.abs(base["coef"]) > 1e-8)[0]
        linear_active = linear_active[np.argsort(np.abs(base["coef"][linear_active]))[::-1]]
        linear_active = linear_active[: int(self.max_linear_terms)]
        linear_active = np.array(sorted(int(j) for j in linear_active), dtype=int)

        def build_design(A):
            cols = [np.ones(A.shape[0])]
            for j in linear_active:
                cols.append(A[:, int(j)])
            for j, t in hinge_terms:
                cols.append(self._hinge(A[:, int(j)], float(t)))
            return np.column_stack(cols)

        Ztr = build_design(Xtr)
        Zval = build_design(Xval)

        best_alpha = float(self.alpha_grid[0])
        best_beta = self._ridge_closed_form(Ztr, ytr, best_alpha)
        best_mse = float(np.mean((yval - Zval @ best_beta) ** 2))
        for alpha in self.alpha_grid[1:]:
            beta = self._ridge_closed_form(Ztr, ytr, float(alpha))
            mse = float(np.mean((yval - Zval @ beta) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_alpha = float(alpha)
                best_beta = beta

        Zall = build_design(X)
        beta_all = self._ridge_closed_form(Zall, y, best_alpha)

        self.alpha_ = best_alpha
        self.linear_active_ = [int(j) for j in linear_active.tolist()]
        self.hinge_terms_ = [(int(j), float(t)) for j, t in hinge_terms]
        self.intercept_ = float(beta_all[0])

        n_lin = len(self.linear_active_)
        n_hinge = len(self.hinge_terms_)
        self.linear_coefs_ = np.asarray(beta_all[1: 1 + n_lin], dtype=float)
        self.hinge_coefs_ = np.asarray(beta_all[1 + n_lin: 1 + n_lin + n_hinge], dtype=float)

        # Prune near-zero terms for readability and stability.
        keep_lin = [i for i, c in enumerate(self.linear_coefs_) if abs(float(c)) > float(self.min_coef_abs)]
        keep_hinge = [i for i, c in enumerate(self.hinge_coefs_) if abs(float(c)) > float(self.min_coef_abs)]
        self.linear_active_ = [self.linear_active_[i] for i in keep_lin]
        self.linear_coefs_ = np.asarray([self.linear_coefs_[i] for i in keep_lin], dtype=float)
        self.hinge_terms_ = [self.hinge_terms_[i] for i in keep_hinge]
        self.hinge_coefs_ = np.asarray([self.hinge_coefs_[i] for i in keep_hinge], dtype=float)

        imp = np.zeros(p, dtype=float)
        for j, c in zip(self.linear_active_, self.linear_coefs_):
            imp[int(j)] += abs(float(c))
        for (j, _), c in zip(self.hinge_terms_, self.hinge_coefs_):
            imp[int(j)] += abs(float(c))
        self.feature_importance_ = imp
        self.feature_rank_ = np.argsort(imp)[::-1]
        self.fitted_mse_ = float(np.mean((y - self.predict(X)) ** 2))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.feature_medians_ = np.nanmedian(X, axis=0)
        X = self._impute(X)
        self._fit(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "linear_active_", "linear_coefs_", "hinge_terms_", "hinge_coefs_"])
        X = self._impute(X)
        yhat = np.full(X.shape[0], self.intercept_, dtype=float)
        for j, c in zip(self.linear_active_, self.linear_coefs_):
            yhat += float(c) * X[:, int(j)]
        for (j, t), c in zip(self.hinge_terms_, self.hinge_coefs_):
            yhat += float(c) * self._hinge(X[:, int(j)], float(t))
        return yhat

    def __str__(self):
        check_is_fitted(self, ["intercept_", "linear_active_", "linear_coefs_", "hinge_terms_", "hinge_coefs_"])
        lines = [
            "SparseBackboneResidualHingeRegressor",
            f"Ridge alpha: {self.alpha_:.5g}",
            "",
            "Prediction equation (exact):",
            f"  y = {self.intercept_:+.4f}",
        ]

        if len(self.linear_active_) == 0 and len(self.hinge_terms_) == 0:
            lines.append("  (no active terms)")

        for j, c in zip(self.linear_active_, self.linear_coefs_):
            lines.append(f"    + ({float(c):+.4f})*x{int(j)}")

        for (j, t), c in zip(self.hinge_terms_, self.hinge_coefs_):
            lines.append(
                f"    + ({float(c):+.4f})*max(0, x{int(j)}-{float(t):+.4f})"
            )

        lines.append("")
        lines.append("Top feature importance (sum abs coefficients):")
        for j in self.feature_rank_[: min(10, self.n_features_in_)]:
            lines.append(f"  x{int(j)}: {self.feature_importance_[int(j)]:.5f}")

        inactive = [f"x{j}" for j in range(self.n_features_in_) if self.feature_importance_[j] <= 1e-10]
        if inactive:
            lines.append("Features with negligible effect: " + ", ".join(inactive))
        lines.append(f"Training MSE: {self.fitted_mse_:.6f}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseBackboneResidualHingeRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseBackboneResHingeV1"
model_description = "Sparse L1 linear backbone with up to two greedy residual hinge terms and a compact joint ridge refit"
model_defs = [(model_shorthand_name, SparseBackboneResidualHingeRegressor())]

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------

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
