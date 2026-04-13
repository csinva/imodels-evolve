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


class SparseBackboneTwoKnotRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse linear backbone with one optional two-knot piecewise correction.

    The final equation is:
      y = b0 + sum_j b_j * x_j + g0*x_k + g1*max(0, x_k - t1) + g2*max(0, x_k - t2)
    where at most one feature k gets the piecewise term.
    """

    def __init__(
        self,
        max_linear_terms=6,
        lasso_cv=3,
        lasso_n_alphas=32,
        alpha_grid=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
        knot_quantiles=(0.2, 0.35, 0.5, 0.65, 0.8),
        max_piecewise_candidates=4,
        val_frac=0.2,
        min_val_samples=80,
        min_piecewise_gain=0.003,
        min_coef_abs=1e-7,
        random_state=42,
    ):
        self.max_linear_terms = max_linear_terms
        self.lasso_cv = lasso_cv
        self.lasso_n_alphas = lasso_n_alphas
        self.alpha_grid = alpha_grid
        self.knot_quantiles = knot_quantiles
        self.max_piecewise_candidates = max_piecewise_candidates
        self.val_frac = val_frac
        self.min_val_samples = min_val_samples
        self.min_piecewise_gain = min_piecewise_gain
        self.min_coef_abs = min_coef_abs
        self.random_state = random_state

    @staticmethod
    def _hinge(x, t):
        return np.maximum(0.0, x - float(t))

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

    def _select_linear_features(self, Xtr, ytr):
        p = Xtr.shape[1]
        x_mean = Xtr.mean(axis=0)
        x_std = Xtr.std(axis=0)
        x_std[x_std < 1e-12] = 1.0
        Xs = (Xtr - x_mean) / x_std

        lasso = LassoCV(
            cv=int(self.lasso_cv),
            random_state=self.random_state,
            n_alphas=int(self.lasso_n_alphas),
            max_iter=8000,
        )
        lasso.fit(Xs, ytr)
        raw_coef = lasso.coef_ / x_std

        active = np.where(np.abs(raw_coef) > 1e-8)[0]
        if active.size == 0:
            active = np.array([int(np.argmax(np.abs(raw_coef)))], dtype=int)
        if active.size > int(self.max_linear_terms):
            order = np.argsort(np.abs(raw_coef[active]))[::-1]
            active = active[order[: int(self.max_linear_terms)]]
        return np.array(sorted(int(j) for j in active), dtype=int), raw_coef

    def _fit_ridge_with_holdout(self, Ztr, ytr, Zval, yval):
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
        return best_alpha, best_beta, best_mse

    def _build_design(self, X, linear_features, pw_feature, knots):
        cols = [np.ones(X.shape[0])]
        for j in linear_features:
            cols.append(X[:, int(j)])
        if pw_feature is not None:
            xk = X[:, int(pw_feature)]
            cols.append(xk)
            cols.append(self._hinge(xk, knots[0]))
            cols.append(self._hinge(xk, knots[1]))
        return np.column_stack(cols)

    def _candidate_piecewise_features(self, raw_coef, Xtr, ytr):
        order_by_coef = np.argsort(np.abs(raw_coef))[::-1]
        top_coef = [int(j) for j in order_by_coef[: int(self.max_piecewise_candidates)]]

        xc = Xtr - Xtr.mean(axis=0)
        yc = ytr - ytr.mean()
        denom = np.sqrt((xc ** 2).sum(axis=0)) * (np.sqrt((yc ** 2).sum()) + 1e-12)
        corr = np.abs((xc.T @ yc) / (denom + 1e-12))
        order_by_corr = np.argsort(corr)[::-1]
        top_corr = [int(j) for j in order_by_corr[: int(self.max_piecewise_candidates)]]

        merged = []
        for j in top_coef + top_corr:
            if j not in merged:
                merged.append(j)
        return merged[: max(2, int(self.max_piecewise_candidates))]

    def _fit(self, X, y):
        n, p = X.shape
        tr_idx, val_idx = self._split_idx(n)
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xval, yval = X[val_idx], y[val_idx]

        linear_features, raw_coef = self._select_linear_features(Xtr, ytr)
        Ztr_base = self._build_design(Xtr, linear_features, None, None)
        Zval_base = self._build_design(Xval, linear_features, None, None)
        base_alpha, _, base_mse = self._fit_ridge_with_holdout(Ztr_base, ytr, Zval_base, yval)

        best = {
            "mse": base_mse,
            "pw_feature": None,
            "knots": None,
            "linear_features": linear_features,
            "alpha": base_alpha,
        }

        candidate_features = self._candidate_piecewise_features(raw_coef, Xtr, ytr)
        q = np.array(self.knot_quantiles, dtype=float)
        for j in candidate_features:
            lin_wo_j = np.array([idx for idx in linear_features if int(idx) != int(j)], dtype=int)
            xj_tr = Xtr[:, int(j)]
            knots_all = np.unique(np.quantile(xj_tr, q))
            if knots_all.size < 2:
                continue
            for a in range(len(knots_all)):
                for b in range(a + 1, len(knots_all)):
                    t1, t2 = float(knots_all[a]), float(knots_all[b])
                    Ztr = self._build_design(Xtr, lin_wo_j, int(j), (t1, t2))
                    Zval = self._build_design(Xval, lin_wo_j, int(j), (t1, t2))
                    alpha, _, mse = self._fit_ridge_with_holdout(Ztr, ytr, Zval, yval)
                    if mse < best["mse"]:
                        best = {
                            "mse": float(mse),
                            "pw_feature": int(j),
                            "knots": (t1, t2),
                            "linear_features": lin_wo_j,
                            "alpha": float(alpha),
                        }

        rel_gain = (base_mse - best["mse"]) / max(base_mse, 1e-12)
        if best["pw_feature"] is None or rel_gain < float(self.min_piecewise_gain):
            best = {
                "mse": base_mse,
                "pw_feature": None,
                "knots": None,
                "linear_features": linear_features,
                "alpha": base_alpha,
            }

        Zall = self._build_design(X, best["linear_features"], best["pw_feature"], best["knots"])
        beta_all = self._ridge_closed_form(Zall, y, best["alpha"])

        self.alpha_ = float(best["alpha"])
        self.intercept_ = float(beta_all[0])
        self.linear_features_ = [int(j) for j in best["linear_features"]]
        self.linear_coefs_ = np.asarray(beta_all[1: 1 + len(self.linear_features_)], dtype=float)

        offset = 1 + len(self.linear_features_)
        self.pw_feature_ = best["pw_feature"]
        if self.pw_feature_ is None:
            self.pw_knots_ = None
            self.pw_coefs_ = np.zeros(3, dtype=float)
        else:
            self.pw_knots_ = (float(best["knots"][0]), float(best["knots"][1]))
            self.pw_coefs_ = np.asarray(beta_all[offset: offset + 3], dtype=float)

        keep_lin = [i for i, c in enumerate(self.linear_coefs_) if abs(float(c)) > float(self.min_coef_abs)]
        self.linear_features_ = [self.linear_features_[i] for i in keep_lin]
        self.linear_coefs_ = np.asarray([self.linear_coefs_[i] for i in keep_lin], dtype=float)

        if self.pw_feature_ is not None and np.max(np.abs(self.pw_coefs_)) <= float(self.min_coef_abs):
            self.pw_feature_ = None
            self.pw_knots_ = None
            self.pw_coefs_ = np.zeros(3, dtype=float)

        imp = np.zeros(p, dtype=float)
        for j, c in zip(self.linear_features_, self.linear_coefs_):
            imp[int(j)] += abs(float(c))
        if self.pw_feature_ is not None:
            imp[int(self.pw_feature_)] += float(np.sum(np.abs(self.pw_coefs_)))
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
        check_is_fitted(self, ["intercept_", "linear_features_", "linear_coefs_", "pw_feature_", "pw_coefs_"])
        X = self._impute(X)
        yhat = np.full(X.shape[0], self.intercept_, dtype=float)
        for j, c in zip(self.linear_features_, self.linear_coefs_):
            yhat += float(c) * X[:, int(j)]
        if self.pw_feature_ is not None:
            xk = X[:, int(self.pw_feature_)]
            yhat += float(self.pw_coefs_[0]) * xk
            yhat += float(self.pw_coefs_[1]) * self._hinge(xk, self.pw_knots_[0])
            yhat += float(self.pw_coefs_[2]) * self._hinge(xk, self.pw_knots_[1])
        return yhat

    def __str__(self):
        check_is_fitted(self, ["intercept_", "linear_features_", "linear_coefs_", "pw_feature_", "pw_coefs_"])
        lines = [
            "SparseBackboneTwoKnotRegressor",
            f"Ridge alpha: {self.alpha_:.5g}",
            "",
            "Prediction equation (exact):",
            f"  y = {self.intercept_:+.4f}",
        ]

        if len(self.linear_features_) == 0 and self.pw_feature_ is None:
            lines.append("    + 0")

        for j, c in zip(self.linear_features_, self.linear_coefs_):
            lines.append(f"    + ({float(c):+.4f})*x{int(j)}")

        if self.pw_feature_ is not None:
            j = int(self.pw_feature_)
            t1, t2 = self.pw_knots_
            lines.append(f"    + ({float(self.pw_coefs_[0]):+.4f})*x{j}")
            lines.append(f"    + ({float(self.pw_coefs_[1]):+.4f})*max(0, x{j}-{float(t1):+.4f})")
            lines.append(f"    + ({float(self.pw_coefs_[2]):+.4f})*max(0, x{j}-{float(t2):+.4f})")

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
SparseBackboneTwoKnotRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseBackboneTwoKnotV1"
model_description = "Sparse L1 backbone with one validation-selected two-knot piecewise correction on a single feature and exact symbolic equation"
model_defs = [(model_shorthand_name, SparseBackboneTwoKnotRegressor())]

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
