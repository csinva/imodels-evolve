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


class GlobalRidgeOneHingeRegressor(BaseEstimator, RegressorMixin):
    """
    Closed-form ridge regression over all features plus at most one hinge term.

    Model on raw features:
      y = intercept + sum_j w_j * x_j + a * max(0, s * (x_k - t))
    where s in {-1, +1}. The hinge is only kept if it improves fit enough.
    """

    def __init__(
        self,
        ridge_lambda=0.2,
        hinge_lambda=1e-3,
        thresholds_per_feature=4,
        min_rel_gain=3e-3,
    ):
        self.ridge_lambda = ridge_lambda
        self.hinge_lambda = hinge_lambda
        self.thresholds_per_feature = thresholds_per_feature
        self.min_rel_gain = min_rel_gain

    @staticmethod
    def _safe_scale(X):
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        sigma = np.where(sigma < 1e-8, 1.0, sigma)
        return mu, sigma

    @staticmethod
    def _solve_ridge_design(D, y, reg):
        A = D.T @ D + np.diag(reg)
        b = D.T @ y
        try:
            beta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(A) @ b
        return beta

    @staticmethod
    def _hinge_col(x, thr, sign):
        return np.maximum(0.0, sign * (x - thr))

    def _fit_joint(self, Xs, y, hinge_spec=None):
        n, p = Xs.shape
        if hinge_spec is None:
            D = np.hstack([np.ones((n, 1), dtype=float), Xs])
            reg = np.zeros(p + 1, dtype=float)
            reg[1:] = max(float(self.ridge_lambda), 0.0)
            beta = self._solve_ridge_design(D, y, reg)
            pred = D @ beta
            return beta[0], beta[1:], 0.0, pred

        j, thr, sign = hinge_spec
        z = self._hinge_col(Xs[:, j], thr, sign)
        D = np.hstack([np.ones((n, 1), dtype=float), Xs, z[:, None]])
        reg = np.zeros(p + 2, dtype=float)
        reg[1 : 1 + p] = max(float(self.ridge_lambda), 0.0)
        reg[-1] = max(float(self.hinge_lambda), 0.0)
        beta = self._solve_ridge_design(D, y, reg)
        pred = D @ beta
        return beta[0], beta[1 : 1 + p], beta[-1], pred

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        self.n_features_in_ = p

        mu, sigma = self._safe_scale(X)
        Xs = (X - mu) / sigma

        b0, w_std0, a0, pred0 = self._fit_joint(Xs, y, hinge_spec=None)
        mse0 = float(np.mean((y - pred0) ** 2))

        residual = y - pred0
        corr = np.abs(Xs.T @ residual)
        top_k = min(p, 12)
        top_idx = np.argsort(corr)[-top_k:]

        best = None
        best_mse = mse0
        for j in top_idx:
            xj = Xs[:, j]
            qs = np.linspace(0.15, 0.85, max(2, int(self.thresholds_per_feature)))
            thrs = np.unique(np.quantile(xj, qs))
            for thr in thrs:
                for sign in (-1.0, 1.0):
                    b, w_std, a, pred = self._fit_joint(Xs, y, hinge_spec=(int(j), float(thr), float(sign)))
                    mse = float(np.mean((y - pred) ** 2))
                    if mse < best_mse:
                        best_mse = mse
                        best = (int(j), float(thr), float(sign), float(b), np.asarray(w_std), float(a))

        rel_gain = (mse0 - best_mse) / max(mse0, 1e-12)
        if best is None or rel_gain < float(self.min_rel_gain):
            self.hinge_feature_ = None
            self.hinge_threshold_std_ = 0.0
            self.hinge_sign_ = 1.0
            self.hinge_coef_ = 0.0
            b_std = float(b0)
            w_std = np.asarray(w_std0, dtype=float)
            train_mse = mse0
        else:
            self.hinge_feature_ = int(best[0])
            self.hinge_threshold_std_ = float(best[1])
            self.hinge_sign_ = float(best[2])
            b_std = float(best[3])
            w_std = np.asarray(best[4], dtype=float)
            self.hinge_coef_ = float(best[5])
            _, _, _, pred = self._fit_joint(Xs, y, hinge_spec=(self.hinge_feature_, self.hinge_threshold_std_, self.hinge_sign_))
            train_mse = float(np.mean((y - pred) ** 2))

        w_raw = w_std / sigma
        intercept_raw = b_std - float(np.dot(w_raw, mu))

        self.feature_mu_ = mu
        self.feature_sigma_ = sigma
        self.linear_coef_ = np.asarray(w_raw, dtype=float)
        self.intercept_ = float(intercept_raw)
        if self.hinge_feature_ is None:
            self.hinge_threshold_raw_ = 0.0
        else:
            self.hinge_threshold_raw_ = float(mu[self.hinge_feature_] + sigma[self.hinge_feature_] * self.hinge_threshold_std_)
        self.train_mse_ = train_mse
        return self

    def predict(self, X):
        check_is_fitted(
            self,
            ["intercept_", "linear_coef_", "feature_mu_", "feature_sigma_", "hinge_feature_", "hinge_threshold_raw_", "hinge_sign_", "hinge_coef_"],
        )
        X = np.asarray(X, dtype=float)
        yhat = self.intercept_ + X @ self.linear_coef_
        if self.hinge_feature_ is not None and abs(self.hinge_coef_) > 1e-12:
            xj = X[:, self.hinge_feature_]
            yhat += self.hinge_coef_ * self._hinge_col(xj, self.hinge_threshold_raw_, self.hinge_sign_)
        return yhat

    def __str__(self):
        check_is_fitted(self, ["intercept_", "linear_coef_", "hinge_feature_", "hinge_threshold_raw_", "hinge_sign_", "hinge_coef_"])
        lines = [
            "Global Ridge + One Hinge Regressor",
            "Prediction equation on raw features:",
        ]
        eq_terms = [f"{self.intercept_:.6f}"]
        for j, c in enumerate(self.linear_coef_):
            eq_terms.append(f"{c:+.6f}*x{j}")
        if self.hinge_feature_ is not None and abs(self.hinge_coef_) > 1e-6:
            sign_txt = "+" if self.hinge_sign_ > 0 else "-"
            eq_terms.append(
                f"{self.hinge_coef_:+.6f}*max(0, {sign_txt}1*(x{self.hinge_feature_} - {self.hinge_threshold_raw_:.6f}))"
            )
        lines.append("  y = " + " ".join(eq_terms))
        lines.append("")
        lines.append("Linear coefficients:")
        for j, c in sorted(enumerate(self.linear_coef_), key=lambda x: -abs(x[1])):
            lines.append(f"  x{j}: {c:+.6f}")
        lines.append("")
        if self.hinge_feature_ is None or abs(self.hinge_coef_) <= 1e-6:
            lines.append("Hinge term: none")
        else:
            direction = "max(0, x - t)" if self.hinge_sign_ > 0 else "max(0, t - x)"
            lines.append(
                f"Hinge term: {self.hinge_coef_:+.6f} * {direction} on x{self.hinge_feature_} with t={self.hinge_threshold_raw_:.6f}"
            )
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
GlobalRidgeOneHingeRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "GlobalRidgeOneHingeV1"
model_description = "Closed-form ridge over all features with optional single residual-selected hinge basis term jointly refit for lightweight nonlinear correction"
model_defs = [(model_shorthand_name, GlobalRidgeOneHingeRegressor())]


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
