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
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class RobustAdaptiveLinearRegressor(BaseEstimator, RegressorMixin):
    """From-scratch robust linear regressor with feature-adaptive ridge shrinkage."""

    def __init__(
        self,
        alpha_grid=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
        huber_delta_grid=(1.0, 2.0),
        corr_shrink_power=1.0,
        n_splits=4,
        max_iter=10,
        tol=1e-4,
        active_coef_threshold=0.06,
        random_state=0,
    ):
        self.alpha_grid = alpha_grid
        self.huber_delta_grid = huber_delta_grid
        self.corr_shrink_power = corr_shrink_power
        self.n_splits = n_splits
        self.max_iter = max_iter
        self.tol = tol
        self.active_coef_threshold = active_coef_threshold
        self.random_state = random_state

    @staticmethod
    def _safe_standardize(X):
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        sigma = np.where(sigma > 1e-12, sigma, 1.0)
        return (X - mu) / sigma, mu, sigma

    @staticmethod
    def _weighted_ridge_solve(X, y, penalty_diag, sample_weights):
        w = np.sqrt(np.maximum(sample_weights, 1e-12))
        Xw = X * w[:, None]
        yw = y * w

        lhs = Xw.T @ Xw + np.diag(penalty_diag)
        rhs = Xw.T @ yw
        try:
            return np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(lhs) @ rhs

    def _fit_irls(self, Xz, yz, penalty_diag, delta):
        n = Xz.shape[0]
        w = np.ones(n, dtype=float)
        beta = np.zeros(Xz.shape[1], dtype=float)

        for _ in range(max(1, int(self.max_iter))):
            beta_prev = beta.copy()
            beta = self._weighted_ridge_solve(Xz, yz, penalty_diag, w)
            resid = yz - Xz @ beta

            mad = np.median(np.abs(resid - np.median(resid)))
            scale = 1.4826 * mad + 1e-8
            cutoff = float(delta) * scale
            abs_r = np.abs(resid)
            w = np.where(abs_r <= cutoff, 1.0, cutoff / (abs_r + 1e-12))

            if np.max(np.abs(beta - beta_prev)) < self.tol:
                break

        return beta

    def _design_and_penalty(self, X, y, alpha):
        n, p = X.shape
        Xz, x_mu, x_sigma = self._safe_standardize(X)
        y_mu = float(np.mean(y))
        y_std = float(np.std(y))
        y_std = y_std if y_std > 1e-12 else 1.0
        yz = (y - y_mu) / y_std

        yc = yz - np.mean(yz)
        denom = np.sqrt(np.sum(Xz**2, axis=0) * np.sum(yc**2)) + 1e-12
        corr = np.abs((Xz.T @ yc) / denom)

        corr_term = np.maximum(corr, 1e-3)
        adaptive = 1.0 / (corr_term ** float(self.corr_shrink_power))
        adaptive /= np.median(adaptive)

        penalty_diag = float(alpha) * adaptive

        return Xz, yz, penalty_diag, x_mu, x_sigma, y_mu, y_std, corr

    def _cv_score(self, X, y, alpha, delta):
        n = X.shape[0]
        n_splits = min(max(2, int(self.n_splits)), n)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        mses = []
        for tr, va in kf.split(X):
            Xtr, ytr = X[tr], y[tr]
            Xva, yva = X[va], y[va]

            Xz, yz, penalty_diag, x_mu, x_sigma, y_mu, y_std, _ = self._design_and_penalty(Xtr, ytr, alpha)
            beta_std = self._fit_irls(Xz, yz, penalty_diag, delta)

            coef = (y_std / x_sigma) * beta_std
            intercept = y_mu - float(np.dot(coef, x_mu))

            pred = Xva @ coef + intercept
            mses.append(float(np.mean((yva - pred) ** 2)))

        return float(np.mean(mses))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if y.ndim != 1:
            y = y.ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have incompatible shapes")

        n, p = X.shape
        if p == 0:
            raise ValueError("No features provided")

        self.n_features_in_ = p

        alpha_choices = [float(a) for a in self.alpha_grid] or [1.0]
        delta_choices = [float(d) for d in self.huber_delta_grid] or [1.0]

        best = None
        for alpha in alpha_choices:
            for delta in delta_choices:
                mse = self._cv_score(X, y, alpha, delta)
                if best is None or mse < best["mse"]:
                    best = {"mse": mse, "alpha": alpha, "delta": delta}

        self.alpha_ = float(best["alpha"]) if best is not None else 1.0
        self.huber_delta_ = float(best["delta"]) if best is not None else 1.0

        Xz, yz, penalty_diag, x_mu, x_sigma, y_mu, y_std, corr = self._design_and_penalty(
            X, y, self.alpha_
        )
        beta_std = self._fit_irls(Xz, yz, penalty_diag, self.huber_delta_)

        coef = (y_std / x_sigma) * beta_std
        intercept = y_mu - float(np.dot(coef, x_mu))

        self.coef_ = np.asarray(coef, dtype=float)
        self.intercept_ = float(intercept)

        abs_coef = np.abs(self.coef_)
        max_abs = float(np.max(abs_coef)) if abs_coef.size else 0.0
        if max_abs > 0:
            self.feature_importances_ = abs_coef / max_abs
            self.active_features_ = np.flatnonzero(
                abs_coef >= float(self.active_coef_threshold) * max_abs
            ).astype(int)
        else:
            self.feature_importances_ = abs_coef
            self.active_features_ = np.zeros(0, dtype=int)

        self.correlation_signal_ = np.asarray(corr, dtype=float)
        return self

    def predict(self, X):
        check_is_fitted(self, ["coef_", "intercept_", "active_features_"])
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_

    def __str__(self):
        check_is_fitted(
            self,
            [
                "coef_",
                "intercept_",
                "alpha_",
                "huber_delta_",
                "active_features_",
                "feature_importances_",
                "correlation_signal_",
            ],
        )

        lines = [
            "Robust Adaptive Linear Regressor",
            "Prediction equation:",
            f"y = {self.intercept_:+.6f}",
        ]

        for j, c in enumerate(self.coef_):
            lines.append(f"    {c:+.6f} * x{j}")

        active = ", ".join(f"x{int(j)}" for j in self.active_features_) if self.active_features_.size else "none"
        lines.append("")
        lines.append(f"ridge_alpha = {self.alpha_:.6g}")
        lines.append(f"huber_delta = {self.huber_delta_:.6g}")
        lines.append(f"active_features = {active}")
        lines.append("feature_importance_by_abs_coef:")
        for j, s in enumerate(self.feature_importances_):
            lines.append(f"  x{j}: {float(s):.4f}  (corr_signal={float(self.correlation_signal_[j]):.4f})")
        lines.append("")
        lines.append("Simulation recipe: multiply each feature xj by its coefficient, sum all terms, then add intercept.")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys

_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])

# Update the model shorthand name and description below to reflect the class above and any changes you make.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
RobustAdaptiveLinearRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "RobustAdaptiveLinear_v1"
model_description = "From-scratch IRLS robust linear model with CV-selected adaptive diagonal ridge penalties based on feature-target correlation"
model_defs = [(model_shorthand_name, RobustAdaptiveLinearRegressor())]


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
