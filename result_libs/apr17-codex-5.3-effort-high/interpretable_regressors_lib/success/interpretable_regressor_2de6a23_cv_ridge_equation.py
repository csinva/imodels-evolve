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


class CrossValidatedRidgeEquationV1(BaseEstimator, RegressorMixin):
    """
    From-scratch ridge regressor with explicit equation output.

    - Standardizes features for stable optimization.
    - Selects L2 penalty by K-fold CV.
    - Solves ridge in closed form.
    - Maps coefficients back to raw feature units for exact simulatability.
    """

    def __init__(
        self,
        alpha_grid=None,
        n_folds=5,
        seed=42,
        negligible_rel_threshold=0.03,
        negligible_abs_floor=1e-3,
        display_precision=6,
        eps=1e-12,
    ):
        self.alpha_grid = alpha_grid
        self.n_folds = n_folds
        self.seed = seed
        self.negligible_rel_threshold = negligible_rel_threshold
        self.negligible_abs_floor = negligible_abs_floor
        self.display_precision = display_precision
        self.eps = eps

    @staticmethod
    def _safe_standardize(X):
        mean = np.mean(X, axis=0)
        scale = np.std(X, axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        return mean.astype(float), scale.astype(float)

    @staticmethod
    def _ridge_closed_form(Xs, y, alpha):
        n = Xs.shape[0]
        D = np.column_stack([np.ones(n, dtype=float), Xs])
        reg = np.eye(D.shape[1], dtype=float)
        reg[0, 0] = 0.0
        A = D.T @ D + float(alpha) * reg
        b = D.T @ y
        try:
            theta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            theta = np.linalg.pinv(A) @ b
        return np.asarray(theta, dtype=float)

    def _make_folds(self, n):
        k = int(max(2, min(self.n_folds, n)))
        idx = np.arange(n)
        rng = np.random.RandomState(self.seed)
        rng.shuffle(idx)
        return np.array_split(idx, k)

    def _select_alpha_cv(self, Xs, y):
        n = Xs.shape[0]
        alphas = (
            np.asarray(self.alpha_grid, dtype=float)
            if self.alpha_grid is not None
            else np.logspace(-6, 3, 19)
        )
        folds = self._make_folds(n)
        best_alpha = float(alphas[0])
        best_cv_mse = None
        for alpha in alphas:
            mse_sum = 0.0
            valid_folds = 0
            for val_idx in folds:
                tr_mask = np.ones(n, dtype=bool)
                tr_mask[val_idx] = False
                if tr_mask.sum() < 2:
                    continue
                theta = self._ridge_closed_form(Xs[tr_mask], y[tr_mask], float(alpha))
                pred_val = theta[0] + Xs[val_idx] @ theta[1:]
                mse_sum += float(np.mean((y[val_idx] - pred_val) ** 2))
                valid_folds += 1
            if valid_folds == 0:
                continue
            cv_mse = mse_sum / valid_folds
            if (best_cv_mse is None) or (cv_mse < best_cv_mse):
                best_cv_mse = cv_mse
                best_alpha = float(alpha)
        return best_alpha, (float(best_cv_mse) if best_cv_mse is not None else float("nan"))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        self.n_features_in_ = p

        mean, scale = self._safe_standardize(X)
        Xs = (X - mean) / scale

        alpha, cv_mse = self._select_alpha_cv(Xs, y)
        theta = self._ridge_closed_form(Xs, y, alpha)

        coef_std = np.asarray(theta[1:], dtype=float)
        coef_raw = coef_std / scale
        intercept_raw = float(theta[0] - np.dot(coef_raw, mean))

        self.coef_ = coef_raw
        self.intercept_ = intercept_raw
        self.feature_mean_ = mean
        self.feature_scale_ = scale
        self.alpha_ = float(alpha)
        self.cv_mse_ = float(cv_mse)

        abs_coef = np.abs(self.coef_)
        max_abs = float(np.max(abs_coef)) if abs_coef.size > 0 else 0.0
        negligible_thr = max(
            float(self.negligible_abs_floor),
            float(self.negligible_rel_threshold) * max_abs,
        )
        self.negligible_threshold_ = negligible_thr
        self.negligible_features_ = np.where(abs_coef <= negligible_thr)[0].astype(int)
        self.active_features_ = np.where(abs_coef > negligible_thr)[0].astype(int)
        self.sorted_features_ = np.argsort(-abs_coef).astype(int)
        self.dominant_feature_ = int(self.sorted_features_[0]) if p > 0 else 0
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "coef_"])
        X = np.asarray(X, dtype=float)
        return self.intercept_ + X @ self.coef_

    def __str__(self):
        check_is_fitted(self, ["intercept_", "coef_"])
        prec = int(self.display_precision)

        def fmt(v):
            return f"{float(v):+.{prec}f}"

        terms = [fmt(self.intercept_)]
        for j in range(self.n_features_in_):
            terms.append(f"{fmt(self.coef_[j])}*x{j}")

        lines = [
            "Cross-Validated Ridge Equation Regressor",
            "Exact prediction equation in raw features:",
            "  y = " + " ".join(terms),
            "",
            "Simulation recipe: multiply each feature xj by its coefficient, sum all terms, then add intercept.",
            "",
            "Feature coefficients (sorted by absolute magnitude):",
        ]

        for j in self.sorted_features_:
            coef_j = float(self.coef_[j])
            tag = "active" if j in set(self.active_features_.tolist()) else "negligible"
            lines.append(f"  x{int(j)}: {coef_j:+.{prec}f} ({tag})")

        lines.append("")
        lines.append(f"Dominant feature by absolute coefficient: x{self.dominant_feature_}")
        lines.append(f"CV-selected ridge alpha: {self.alpha_:.6g}")
        lines.append(f"Cross-validated MSE: {self.cv_mse_:.6f}")

        if len(self.active_features_) > 0:
            lines.append("Active features: " + ", ".join(f"x{int(i)}" for i in self.active_features_))
        if len(self.negligible_features_) > 0:
            lines.append("Negligible features: " + ", ".join(f"x{int(i)}" for i in self.negligible_features_))
        return "\n".join(lines)


import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
CrossValidatedRidgeEquationV1.__module__ = "interpretable_regressor"

model_shorthand_name = "CrossValidatedRidgeEquationV1"
model_description = "From-scratch standardized closed-form ridge with CV-selected penalty, mapped back to an explicit raw-feature equation plus active/negligible feature summaries for easier simulation"
model_defs = [(model_shorthand_name, CrossValidatedRidgeEquationV1())]

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
