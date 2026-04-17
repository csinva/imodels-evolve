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
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class SingleIndexSplineRegressor(BaseEstimator, RegressorMixin):
    """Sparse ridge single-index model with a tiny hinge spline on the linear score."""

    def __init__(
        self,
        alpha_grid=(1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
        max_screen_features=20,
        min_screen_features=6,
        knot_quantiles=(0.25, 0.5, 0.75),
        spline_alphas=(0.0, 1e-4, 1e-3, 1e-2, 1e-1),
        spline_cv_folds=4,
        hinge_prune_tol=0.08,
        random_state=0,
    ):
        self.alpha_grid = alpha_grid
        self.max_screen_features = max_screen_features
        self.min_screen_features = min_screen_features
        self.knot_quantiles = knot_quantiles
        self.spline_alphas = spline_alphas
        self.spline_cv_folds = spline_cv_folds
        self.hinge_prune_tol = hinge_prune_tol
        self.random_state = random_state

    @staticmethod
    def _safe_standardize(X):
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        sigma = np.where(sigma < 1e-12, 1.0, sigma)
        return mu, sigma

    @staticmethod
    def _ridge_closed_form(Z, y, alpha):
        n, d = Z.shape
        reg = np.eye(d, dtype=float) * float(alpha)
        reg[0, 0] = 0.0
        lhs = Z.T @ Z + reg
        rhs = Z.T @ y
        try:
            return np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(lhs) @ rhs

    def _screen_features(self, X, y):
        p = X.shape[1]
        if p <= self.max_screen_features:
            return np.arange(p, dtype=int)
        y_centered = y - np.mean(y)
        x_centered = X - np.mean(X, axis=0, keepdims=True)
        denom = np.sqrt(np.sum(x_centered**2, axis=0) * np.sum(y_centered**2)) + 1e-12
        corr = np.abs((x_centered.T @ y_centered) / denom)
        order = np.argsort(corr)[::-1]
        k = max(self.min_screen_features, min(self.max_screen_features, p))
        keep = np.sort(order[:k]).astype(int)
        return keep

    def _build_spline_design(self, score, knots):
        cols = [np.ones_like(score), score]
        for knot in knots:
            cols.append(np.maximum(0.0, score - knot))
        return np.column_stack(cols)

    def _fit_score_spline(self, score, y):
        knots = np.unique(np.quantile(score, self.knot_quantiles))
        Z = self._build_spline_design(score, knots)
        n = len(y)
        n_splits = int(min(max(2, self.spline_cv_folds), n))
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        best_alpha = float(self.spline_alphas[0])
        best_mse = float("inf")
        for alpha in self.spline_alphas:
            fold_mses = []
            for tr, va in kf.split(Z):
                beta = self._ridge_closed_form(Z[tr], y[tr], alpha)
                pred = Z[va] @ beta
                fold_mses.append(float(np.mean((y[va] - pred) ** 2)))
            mse = float(np.mean(fold_mses))
            if mse < best_mse:
                best_mse = mse
                best_alpha = float(alpha)

        beta = self._ridge_closed_form(Z, y, best_alpha)
        linear_scale = abs(float(beta[1])) + 1e-12
        hinge = np.asarray(beta[2:], dtype=float)
        keep_hinge = np.abs(hinge) >= self.hinge_prune_tol * linear_scale
        if hinge.size > 0 and not np.any(keep_hinge):
            keep_hinge[np.argmax(np.abs(hinge))] = True

        kept_knots = knots[keep_hinge] if hinge.size > 0 else np.array([], dtype=float)
        if hinge.size > 0 and np.any(~keep_hinge):
            Z_refit = self._build_spline_design(score, kept_knots)
            beta_refit = self._ridge_closed_form(Z_refit, y, best_alpha)
        else:
            beta_refit = beta
        return {
            "alpha": best_alpha,
            "knots": kept_knots.astype(float),
            "beta": np.asarray(beta_refit, dtype=float),
        }

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        if p == 0:
            raise ValueError("No features provided")

        self.n_features_in_ = p
        self.screened_features_ = self._screen_features(X, y)
        X_screen = X[:, self.screened_features_]
        self.x_mean_, self.x_scale_ = self._safe_standardize(X_screen)
        Xs = (X_screen - self.x_mean_) / self.x_scale_

        self.backbone_ = RidgeCV(alphas=np.array(self.alpha_grid, dtype=float), cv=5)
        self.backbone_.fit(Xs, y)

        coef_scaled = np.asarray(self.backbone_.coef_, dtype=float)
        coef_screen = coef_scaled / self.x_scale_
        intercept = float(self.backbone_.intercept_ - np.dot(self.x_mean_ / self.x_scale_, coef_scaled))
        self.alpha_ = float(self.backbone_.alpha_)

        full_coef = np.zeros(p, dtype=float)
        full_coef[self.screened_features_] = coef_screen
        self.coef_ = full_coef
        self.intercept_ = intercept

        score = self.intercept_ + X @ self.coef_
        self.score_spline_ = self._fit_score_spline(score, y)

        abs_coef = np.abs(self.coef_)
        max_abs = float(np.max(abs_coef)) if np.max(abs_coef) > 0 else 0.0
        self.feature_importances_ = abs_coef / max_abs if max_abs > 0 else abs_coef

        coef_tol = max(1e-10, 0.06 * (max_abs if max_abs > 0 else 1.0))
        self.active_features_ = np.flatnonzero(abs_coef >= coef_tol).astype(int)

        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "coef_", "score_spline_", "active_features_"])
        X = np.asarray(X, dtype=float)
        score = self.intercept_ + X @ self.coef_
        spline = self.score_spline_
        Z = self._build_spline_design(score, spline["knots"])
        return Z @ spline["beta"]

    def __str__(self):
        check_is_fitted(
            self,
            ["intercept_", "coef_", "alpha_", "active_features_", "feature_importances_", "score_spline_"],
        )
        spline = self.score_spline_
        beta = spline["beta"]

        lines = [
            "Single-Index Sparse Spline Regressor",
            "Prediction rule:",
        ]
        lines.append("1) score = intercept + sum_j(coef_j * xj)")
        lines.append("2) y = beta0 + beta1*score + sum_k(gamma_k * max(0, score - knot_k))")

        lines.append("")
        lines.append(f"backbone_ridge_alpha = {self.alpha_:.6g}")
        lines.append(f"spline_ridge_alpha = {spline['alpha']:.6g}")
        lines.append(
            f"active_linear_features = {', '.join(f'x{int(j)}' for j in self.active_features_) if self.active_features_.size else 'none'}"
        )
        lines.append("Linear score coefficients:")
        lines.append(f"intercept: {self.intercept_:+.6f}")

        for j in range(self.coef_.size):
            c = float(self.coef_[j])
            lines.append(f"x{int(j)}: coef={c:+.6f}")

        lines.append("")
        lines.append(f"beta0: {float(beta[0]):+.6f}")
        lines.append(f"beta1: {float(beta[1]):+.6f}")
        if spline["knots"].size:
            lines.append("Spline hinge terms:")
            for idx, (knot, gamma) in enumerate(zip(spline["knots"], beta[2:])):
                lines.append(f"gamma{idx}: {float(gamma):+.6f} at knot={float(knot):+.6f}")
        else:
            lines.append("Spline hinge terms: none")
        lines.append("")
        lines.append("Simulation recipe: compute score from x-values, then apply the short spline equation.")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys

_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])

# Update the model shorthand name and description below to reflect the class above and any changes you make.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
SingleIndexSplineRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "SingleIndexSpline_v2"
model_description = "Correlation-screened ridge linear score with singular-safe tiny CV-regularized hinge spline calibration for compact nonlinear correction"
model_defs = [(model_shorthand_name, SingleIndexSplineRegressor())]


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
