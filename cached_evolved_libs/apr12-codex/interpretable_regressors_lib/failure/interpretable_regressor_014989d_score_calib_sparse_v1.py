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


class ScoreCalibratedSparseRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse linear score with compact nonlinear score calibration.

    Step 1: fit sparse ridge backbone on selected features.
    Step 2: fit a two-knot piecewise-linear calibrator on the backbone score.
    Step 3: optionally add one interaction term if validation error improves.
    """

    def __init__(
        self,
        alpha_min_exp=-3.0,
        alpha_max_exp=2.0,
        alpha_grid_size=16,
        candidate_k=(4, 8, 12),
        val_frac=0.2,
        min_val_samples=50,
        knot_quantiles=(0.2, 0.4, 0.6, 0.8),
        calib_l2=1e-2,
        interaction_top_features=4,
        interaction_min_gain_frac=0.003,
        coef_tol=1e-7,
        random_state=42,
    ):
        self.alpha_min_exp = alpha_min_exp
        self.alpha_max_exp = alpha_max_exp
        self.alpha_grid_size = alpha_grid_size
        self.candidate_k = candidate_k
        self.val_frac = val_frac
        self.min_val_samples = min_val_samples
        self.knot_quantiles = knot_quantiles
        self.calib_l2 = calib_l2
        self.interaction_top_features = interaction_top_features
        self.interaction_min_gain_frac = interaction_min_gain_frac
        self.coef_tol = coef_tol
        self.random_state = random_state

    def _impute(self, X):
        X = np.asarray(X, dtype=float).copy()
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(self.feature_medians_, np.where(bad)[1])
        return X

    def _make_val_split(self, n):
        if n < max(30, int(self.min_val_samples) + 10):
            return None, None
        rng = np.random.RandomState(self.random_state)
        idx = rng.permutation(n)
        val_n = max(int(round(float(self.val_frac) * n)), int(self.min_val_samples))
        val_n = min(val_n, n // 2)
        if val_n < 20:
            return None, None
        return idx[val_n:], idx[:val_n]

    @staticmethod
    def _fit_subset_ridge(X, y, active, alpha):
        if len(active) == 0:
            return dict(active=np.array([], dtype=int), coef=np.zeros(0), intercept=float(np.mean(y)))
        Xa = X[:, active]
        mu = np.mean(Xa, axis=0)
        sigma = np.std(Xa, axis=0)
        sigma[sigma < 1e-12] = 1.0
        Z = (Xa - mu) / sigma
        yc = y - float(np.mean(y))
        k = Z.shape[1]
        beta_std = np.linalg.solve(Z.T @ Z + float(alpha) * np.eye(k), Z.T @ yc)
        coef = beta_std / sigma
        intercept = float(np.mean(y) - np.dot(mu, coef))
        return dict(active=np.asarray(active, dtype=int), coef=np.asarray(coef), intercept=intercept)

    @staticmethod
    def _predict_linear(X, params):
        yhat = np.full(X.shape[0], params["intercept"], dtype=float)
        if len(params["active"]) > 0:
            yhat += X[:, params["active"]] @ params["coef"]
        return yhat

    @staticmethod
    def _fit_calibrator(score, y, t1, t2, l2):
        s = np.asarray(score, dtype=float)
        h1 = np.maximum(0.0, s - float(t1))
        h2 = np.maximum(0.0, s - float(t2))
        D = np.column_stack([np.ones_like(s), s, h1, h2])
        reg = np.diag([0.0, float(l2), float(l2), float(l2)])
        beta = np.linalg.solve(D.T @ D + reg, D.T @ y)
        return beta

    @staticmethod
    def _apply_calibrator(score, beta, t1, t2):
        s = np.asarray(score, dtype=float)
        return (
            beta[0]
            + beta[1] * s
            + beta[2] * np.maximum(0.0, s - float(t1))
            + beta[3] * np.maximum(0.0, s - float(t2))
        )

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        self.n_features_in_ = p

        self.feature_medians_ = np.nanmedian(X, axis=0)
        X = self._impute(X)

        tr_idx, val_idx = self._make_val_split(n)
        if tr_idx is None:
            tr_idx = np.arange(n)
            val_idx = np.arange(n)

        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xval, yval = X[val_idx], y[val_idx]

        # Stage A: choose alpha with full ridge for stable feature ranking.
        alphas = np.logspace(float(self.alpha_min_exp), float(self.alpha_max_exp), int(self.alpha_grid_size))
        full_active = np.arange(p)
        best_alpha = float(alphas[0])
        best_full = None
        best_full_val_mse = np.inf
        for alpha in alphas:
            params = self._fit_subset_ridge(Xtr, ytr, full_active, alpha)
            mse = float(np.mean((yval - self._predict_linear(Xval, params)) ** 2))
            if mse < best_full_val_mse:
                best_full_val_mse = mse
                best_full = params
                best_alpha = float(alpha)
        self.alpha_ = best_alpha

        full_coef = np.zeros(p)
        full_coef[best_full["active"]] = best_full["coef"]
        score_strength = np.abs(full_coef) * np.maximum(np.std(Xtr, axis=0), 1e-12)
        order = np.argsort(score_strength)[::-1]

        # Stage B: choose sparse subset size by validation.
        k_values = sorted({max(1, min(int(k), p)) for k in self.candidate_k})
        best_sparse = None
        best_sparse_val_mse = np.inf
        for k in k_values:
            active = np.sort(order[:k])
            params = self._fit_subset_ridge(Xtr, ytr, active, self.alpha_)
            mse = float(np.mean((yval - self._predict_linear(Xval, params)) ** 2))
            if mse < best_sparse_val_mse:
                best_sparse_val_mse = mse
                best_sparse = params

        self.selected_features_ = [int(j) for j in best_sparse["active"].tolist()]
        self.coef_ = np.asarray(best_sparse["coef"], dtype=float)
        self.intercept_ = float(best_sparse["intercept"])

        # Stage C: calibrate the sparse score with two hinges in score space.
        score_tr = self._predict_linear(Xtr, best_sparse)
        score_val = self._predict_linear(Xval, best_sparse)
        q = np.unique(np.quantile(score_tr, self.knot_quantiles))
        if q.size < 2:
            q = np.array([np.min(score_tr), np.max(score_tr)])
        best_cal = None
        best_cal_mse = np.inf
        for i in range(len(q)):
            for j in range(i + 1, len(q)):
                t1, t2 = float(q[i]), float(q[j])
                beta = self._fit_calibrator(score_tr, ytr, t1, t2, self.calib_l2)
                pred_val = self._apply_calibrator(score_val, beta, t1, t2)
                mse = float(np.mean((yval - pred_val) ** 2))
                if mse < best_cal_mse:
                    best_cal_mse = mse
                    best_cal = dict(beta=beta, t1=t1, t2=t2)

        self.calib_beta_ = np.asarray(best_cal["beta"], dtype=float)
        self.calib_t1_ = float(best_cal["t1"])
        self.calib_t2_ = float(best_cal["t2"])
        self.base_val_mse_ = float(best_cal_mse)

        # Stage D: optional single interaction if it improves validation.
        self.has_interaction_ = False
        self.interaction_pair_ = (-1, -1)
        self.interaction_coef_ = 0.0

        top = [int(j) for j in order[: min(int(self.interaction_top_features), p)]]
        yval_var = float(np.var(yval) + 1e-12)
        pred_val_base = self.predict(Xval)
        resid_tr = ytr - self.predict(Xtr)
        best_int = None
        for a in range(len(top)):
            for b in range(a + 1, len(top)):
                j1, j2 = top[a], top[b]
                ztr = Xtr[:, j1] * Xtr[:, j2]
                zval = Xval[:, j1] * Xval[:, j2]
                denom = float(np.dot(ztr, ztr) + 1e-6 * len(ztr))
                if denom <= 1e-12:
                    continue
                coef = float(np.dot(ztr, resid_tr) / denom)
                pred = pred_val_base + coef * zval
                mse = float(np.mean((yval - pred) ** 2))
                if best_int is None or mse < best_int["mse"]:
                    best_int = dict(mse=mse, j1=j1, j2=j2, coef=coef)

        if best_int is not None:
            gain = self.base_val_mse_ - float(best_int["mse"])
            if gain > float(self.interaction_min_gain_frac) * yval_var and abs(float(best_int["coef"])) > float(self.coef_tol):
                self.has_interaction_ = True
                self.interaction_pair_ = (int(best_int["j1"]), int(best_int["j2"]))
                self.interaction_coef_ = float(best_int["coef"])

        # Refit sparse linear backbone on all data with fixed selected features.
        final_lin = self._fit_subset_ridge(X, y, np.asarray(self.selected_features_, dtype=int), self.alpha_)
        self.selected_features_ = [int(j) for j in final_lin["active"].tolist()]
        self.coef_ = np.asarray(final_lin["coef"], dtype=float)
        self.intercept_ = float(final_lin["intercept"])

        # Refit calibration on all data for final model.
        score_all = self._predict_linear(X, final_lin)
        self.calib_beta_ = self._fit_calibrator(score_all, y, self.calib_t1_, self.calib_t2_, self.calib_l2)

        if self.has_interaction_:
            j1, j2 = self.interaction_pair_
            residual = y - self._apply_calibrator(score_all, self.calib_beta_, self.calib_t1_, self.calib_t2_)
            z = X[:, j1] * X[:, j2]
            denom = float(np.dot(z, z) + 1e-6 * len(z))
            self.interaction_coef_ = float(np.dot(z, residual) / denom)

        # Prune tiny linear coefficients for cleaner equations.
        keep = np.where(np.abs(self.coef_) > float(self.coef_tol))[0]
        self.selected_features_ = [self.selected_features_[int(i)] for i in keep]
        self.coef_ = self.coef_[keep]

        # Feature-importance summary for display.
        imp = np.zeros(p, dtype=float)
        for j, c in zip(self.selected_features_, self.coef_):
            imp[int(j)] += abs(float(c))
        if self.has_interaction_:
            j1, j2 = self.interaction_pair_
            imp[j1] += abs(self.interaction_coef_) * 0.5
            imp[j2] += abs(self.interaction_coef_) * 0.5
        self.feature_importance_ = imp
        self.feature_rank_ = np.argsort(self.feature_importance_)[::-1]
        self.fitted_mse_ = float(np.mean((y - self.predict(X)) ** 2))
        return self

    def _linear_score(self, X):
        yhat = np.full(X.shape[0], self.intercept_, dtype=float)
        if len(self.selected_features_) > 0:
            yhat += X[:, self.selected_features_] @ self.coef_
        return yhat

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "coef_", "selected_features_", "calib_beta_", "calib_t1_", "calib_t2_"])
        X = self._impute(X)
        score = self._linear_score(X)
        yhat = self._apply_calibrator(score, self.calib_beta_, self.calib_t1_, self.calib_t2_)
        if self.has_interaction_:
            j1, j2 = self.interaction_pair_
            yhat += self.interaction_coef_ * (X[:, j1] * X[:, j2])
        return yhat

    def __str__(self):
        check_is_fitted(self, ["intercept_", "coef_", "selected_features_", "calib_beta_", "calib_t1_", "calib_t2_"])
        lines = [
            "ScoreCalibratedSparseRegressor",
            f"Ridge alpha: {self.alpha_:.5g}",
            "Step 1 (sparse score):",
        ]

        if len(self.selected_features_) == 0:
            lines.append(f"  s = {self.intercept_:+.4f}")
        else:
            ordered = np.argsort(np.abs(self.coef_))[::-1]
            terms = [f"({self.coef_[i]:+.4f})*x{self.selected_features_[i]}" for i in ordered]
            lines.append(f"  s = {self.intercept_:+.4f} " + " ".join(terms))

        b0, b1, b2, b3 = [float(v) for v in self.calib_beta_]
        lines.append("Step 2 (piecewise calibration on score s):")
        lines.append(
            "  y = "
            f"{b0:+.4f} + ({b1:+.4f})*s + ({b2:+.4f})*max(0, s-{self.calib_t1_:+.4f}) "
            f"+ ({b3:+.4f})*max(0, s-{self.calib_t2_:+.4f})"
        )

        if self.has_interaction_:
            j1, j2 = self.interaction_pair_
            lines.append(f"Step 3 (interaction correction): + ({self.interaction_coef_:+.4f})*(x{j1}*x{j2})")
        else:
            lines.append("Step 3 (interaction correction): none")

        lines.append("")
        lines.append("Feature importance (sorted):")
        for j in self.feature_rank_[: min(12, self.n_features_in_)]:
            lines.append(f"  x{int(j)}: {self.feature_importance_[int(j)]:.5f}")
        unused = [f"x{j}" for j in range(self.n_features_in_) if self.feature_importance_[j] <= 1e-10]
        if unused:
            lines.append("Features with negligible effect: " + ", ".join(unused))
        lines.append(f"Training MSE: {self.fitted_mse_:.6f}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
ScoreCalibratedSparseRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "ScoreCalibSparseV1"
model_description = "Sparse ridge score with two-knot piecewise score calibration and optional single interaction correction"
model_defs = [(model_shorthand_name, ScoreCalibratedSparseRegressor())]

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
