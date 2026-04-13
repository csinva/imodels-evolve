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


class SparseKnotAdditiveRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse additive piecewise-linear equation:
      1) Robust-scale each feature.
      2) For each feature, build a compact basis: linear + 3 quantile hinges.
      3) Greedily add feature groups that most reduce residual error.
      4) Joint ridge refit on selected groups for stable predictions.
    """

    def __init__(
        self,
        max_groups=7,
        quantiles=(0.2, 0.5, 0.8),
        ridge_alpha=0.2,
        min_improvement=1e-4,
        max_display_terms=18,
    ):
        self.max_groups = max_groups
        self.quantiles = quantiles
        self.ridge_alpha = ridge_alpha
        self.min_improvement = min_improvement
        self.max_display_terms = max_display_terms

    def _robust_stats(self, X):
        med = np.median(X, axis=0)
        q25 = np.quantile(X, 0.25, axis=0)
        q75 = np.quantile(X, 0.75, axis=0)
        iqr = q75 - q25
        scale = np.where(iqr < 1e-8, 1.0, iqr)
        return med, scale

    def _standardize(self, X):
        return (X - self.med_) / self.scale_

    def _feature_basis(self, z_col, knots):
        cols = [z_col]
        for k in knots:
            cols.append(np.maximum(0.0, z_col - k))
        B = np.column_stack(cols)
        return B

    def _ridge_fit(self, X, y, alpha):
        if X.size == 0:
            mu = float(np.mean(y)) if y.size else 0.0
            return np.zeros(0, dtype=float), mu
        XtX = X.T @ X
        reg = float(max(alpha, 1e-10)) * np.eye(X.shape[1])
        beta = np.linalg.solve(XtX + reg, X.T @ y)
        return beta, 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape

        self.n_features_in_ = int(p)
        self.feature_names_ = [f"x{j}" for j in range(p)]

        if n == 0:
            self.med_ = np.zeros(p, dtype=float)
            self.scale_ = np.ones(p, dtype=float)
            self.knots_ = [np.zeros(3, dtype=float) for _ in range(p)]
            self.selected_groups_ = []
            self.group_coefs_ = {}
            self.intercept_ = float(np.mean(y)) if y.size else 0.0
            self.training_mse_ = 0.0
            self.is_fitted_ = True
            return self

        self.med_, self.scale_ = self._robust_stats(X)
        Z = self._standardize(X)
        self.knots_ = []
        for j in range(p):
            qvals = np.quantile(Z[:, j], self.quantiles)
            self.knots_.append(np.asarray(qvals, dtype=float))

        group_basis = []
        for j in range(p):
            Bj = self._feature_basis(Z[:, j], self.knots_[j])
            Bj = Bj - np.mean(Bj, axis=0, keepdims=True)
            group_basis.append(Bj)

        y_mean = float(np.mean(y))
        yc = y - y_mean

        selected = []
        residual = yc.copy()
        current_sse = float(np.dot(residual, residual))
        max_groups = int(min(max(1, self.max_groups), p))

        for _ in range(max_groups):
            best_j = None
            best_improvement = 0.0
            for j in range(p):
                if j in selected:
                    continue
                Bj = group_basis[j]
                beta_j, _ = self._ridge_fit(Bj, residual, self.ridge_alpha)
                pred_j = Bj @ beta_j
                new_resid = residual - pred_j
                new_sse = float(np.dot(new_resid, new_resid))
                improvement = current_sse - new_sse
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_j = j

            if best_j is None or best_improvement < float(self.min_improvement) * max(1.0, current_sse):
                break

            selected.append(best_j)

            Xsel = np.hstack([group_basis[j] for j in selected])
            beta_sel, _ = self._ridge_fit(Xsel, yc, self.ridge_alpha)
            pred_sel = Xsel @ beta_sel
            residual = yc - pred_sel
            current_sse = float(np.dot(residual, residual))

        self.selected_groups_ = selected
        self.intercept_ = y_mean
        self.group_coefs_ = {}

        if selected:
            Xsel = np.hstack([group_basis[j] for j in selected])
            beta_sel, _ = self._ridge_fit(Xsel, yc, self.ridge_alpha)
            pos = 0
            for j in selected:
                width = group_basis[j].shape[1]
                self.group_coefs_[j] = beta_sel[pos : pos + width]
                pos += width
            final_pred = y_mean + Xsel @ beta_sel
        else:
            final_pred = np.full(n, y_mean)

        self.training_mse_ = float(np.mean((y - final_pred) ** 2))
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        X = np.asarray(X, dtype=float)
        Z = self._standardize(X)

        yhat = np.full(Z.shape[0], self.intercept_, dtype=float)
        for j in self.selected_groups_:
            Bj = self._feature_basis(Z[:, j], self.knots_[j])
            Bj = Bj - np.mean(Bj, axis=0, keepdims=True)
            yhat += Bj @ self.group_coefs_[j]
        return yhat

    def __str__(self):
        check_is_fitted(self, "is_fitted_")
        lines = ["Sparse Knot Additive Equation Regressor:"]
        lines.append("  prediction = intercept + sum selected_features piecewise_linear_shape(x_j)")
        lines.append(f"  ridge alpha: {self.ridge_alpha:.4g}")
        lines.append(f"  selected features: {len(self.selected_groups_)}/{self.n_features_in_}")
        lines.append(f"  training MSE: {self.training_mse_:.6f}")
        lines.append(f"  intercept: {self.intercept_:+.6f}")

        entries = []
        for j in self.selected_groups_:
            coef = self.group_coefs_[j]
            score = float(np.sum(np.abs(coef)))
            entries.append((score, j, coef, self.knots_[j]))
        entries.sort(reverse=True, key=lambda t: t[0])

        lines.append("  strongest feature-shapes:")
        for _, j, coef, knots in entries[: int(max(1, self.max_display_terms))]:
            lines.append(
                "    x{}: linear={:+.4f}, h({:+.3f})={:+.4f}, h({:+.3f})={:+.4f}, h({:+.3f})={:+.4f}".format(
                    int(j),
                    coef[0],
                    float(knots[0]),
                    coef[1],
                    float(knots[1]),
                    coef[2],
                    float(knots[2]),
                    coef[3],
                )
            )
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseKnotAdditiveRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "SparseKnotAdditive_v1"
model_description = "Sparse additive equation with robust scaling, per-feature quantile-knot hinge bases, greedy group selection, and joint ridge refit"
model_defs = [(model_shorthand_name, SparseKnotAdditiveRegressor())]


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

    std_tests = {t.__name__ for t in ALL_TESTS}
    hard_tests = {t.__name__ for t in HARD_TESTS}
    insight_tests = {t.__name__ for t in INSIGHT_TESTS}
    std_passed = sum(r["passed"] for r in interp_results if r["test"] in std_tests)
    hard_passed = sum(r["passed"] for r in interp_results if r["test"] in hard_tests)
    insight_passed = sum(r["passed"] for r in interp_results if r["test"] in insight_tests)
    print(f"[std {std_passed}/{len(std_tests)}  hard {hard_passed}/{len(hard_tests)}  insight {insight_passed}/{len(insight_tests)}]")
    print(f"total_seconds: {time.time() - t0:.1f}s")
