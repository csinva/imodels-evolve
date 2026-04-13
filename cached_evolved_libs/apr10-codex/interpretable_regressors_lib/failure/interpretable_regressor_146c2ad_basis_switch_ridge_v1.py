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


class BasisSwitchRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Interpretable equation model that switches among three basis families:
      1) linear
      2) linear + centered quadratic
      3) linear + two quantile hinge pairs per feature
    Each family is fit with custom GCV ridge; the simplest family within a
    relative tolerance of the best GCV score is selected.
    """

    def __init__(
        self,
        alpha_grid_size=17,
        alpha_min_log10=-5.0,
        alpha_max_log10=3.0,
        gcv_tol=0.03,
        display_top_k=10,
    ):
        self.alpha_grid_size = alpha_grid_size
        self.alpha_min_log10 = alpha_min_log10
        self.alpha_max_log10 = alpha_max_log10
        self.gcv_tol = gcv_tol
        self.display_top_k = display_top_k

    @staticmethod
    def _standardize(X):
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        sigma = np.where(sigma < 1e-10, 1.0, sigma)
        return (X - mu) / sigma, mu, sigma

    def _ridge_gcv(self, B, y):
        n = B.shape[0]
        if B.shape[1] == 0:
            pred = np.full(n, np.mean(y), dtype=float)
            return {
                "coef": np.zeros(0, dtype=float),
                "intercept": float(np.mean(y)),
                "pred": pred,
                "alpha": 0.0,
                "gcv": float(np.mean((y - pred) ** 2)),
                "mse": float(np.mean((y - pred) ** 2)),
                "df": 0.0,
            }

        y_mean = float(np.mean(y))
        yc = y - y_mean

        B_mean = np.mean(B, axis=0)
        Bc = B - B_mean

        U, s, Vt = np.linalg.svd(Bc, full_matrices=False)
        uty = U.T @ yc
        s2 = s * s

        a_lo = float(self.alpha_min_log10)
        a_hi = float(self.alpha_max_log10)
        grid_n = int(max(5, self.alpha_grid_size))
        alphas = np.logspace(a_lo, a_hi, num=grid_n)

        best = None
        for a in alphas:
            shrink = s / (s2 + a)
            coef = Vt.T @ (shrink * uty)
            intercept = y_mean - float(B_mean @ coef)
            pred = B @ coef + intercept
            mse = float(np.mean((y - pred) ** 2))
            df = float(np.sum(s2 / (s2 + a)))
            denom = max(1e-8, (1.0 - (df / max(n, 1))))
            gcv = mse / (denom * denom)
            cur = {
                "coef": coef,
                "intercept": intercept,
                "pred": pred,
                "alpha": float(a),
                "gcv": gcv,
                "mse": mse,
                "df": df,
            }
            if best is None or cur["gcv"] < best["gcv"]:
                best = cur
        return best

    @staticmethod
    def _build_linear_basis(Z):
        names = [f"x{j}" for j in range(Z.shape[1])]
        return Z, names

    @staticmethod
    def _build_quadratic_basis(Z):
        cols = [Z]
        names = [f"x{j}" for j in range(Z.shape[1])]
        z2 = Z * Z
        z2 -= np.mean(z2, axis=0)
        cols.append(z2)
        names.extend([f"(x{j})^2_centered" for j in range(Z.shape[1])])
        return np.hstack(cols), names

    def _build_hinge_basis(self, Z):
        p = Z.shape[1]
        cols = [Z]
        names = [f"x{j}" for j in range(p)]
        knots = np.quantile(Z, [0.25, 0.75], axis=0)
        for j in range(p):
            q1 = float(knots[0, j])
            q3 = float(knots[1, j])
            xj = Z[:, j]
            left = np.maximum(q1 - xj, 0.0)
            right = np.maximum(xj - q3, 0.0)
            left -= np.mean(left)
            right -= np.mean(right)
            cols.append(left[:, None])
            cols.append(right[:, None])
            names.append(f"hinge_left(x{j},q25={q1:.3f})")
            names.append(f"hinge_right(x{j},q75={q3:.3f})")
        return np.hstack(cols), names

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        self.n_features_in_ = int(p)

        if n == 0:
            self.mean_ = float(np.mean(y)) if y.size else 0.0
            self.scale_mean_ = np.zeros(p, dtype=float)
            self.scale_std_ = np.ones(p, dtype=float)
            self.family_ = "linear"
            self.feature_names_ = []
            self.coef_ = np.zeros(0, dtype=float)
            self.intercept_ = self.mean_
            self.training_mse_ = 0.0
            self.alpha_ = 0.0
            self.is_fitted_ = True
            return self

        Z, mu, sigma = self._standardize(X)
        self.scale_mean_ = mu
        self.scale_std_ = sigma

        candidates = []

        B_lin, n_lin = self._build_linear_basis(Z)
        r_lin = self._ridge_gcv(B_lin, y)
        candidates.append(("linear", B_lin, n_lin, r_lin))

        B_quad, n_quad = self._build_quadratic_basis(Z)
        r_quad = self._ridge_gcv(B_quad, y)
        candidates.append(("linear+quadratic", B_quad, n_quad, r_quad))

        B_hinge, n_hinge = self._build_hinge_basis(Z)
        r_hinge = self._ridge_gcv(B_hinge, y)
        candidates.append(("linear+hinge", B_hinge, n_hinge, r_hinge))

        best_gcv = min(item[3]["gcv"] for item in candidates)
        tol = float(max(0.0, self.gcv_tol))
        admissible = [item for item in candidates if item[3]["gcv"] <= best_gcv * (1.0 + tol)]

        complexity = {"linear": 1, "linear+quadratic": 2, "linear+hinge": 3}
        family, B, names, result = min(admissible, key=lambda item: complexity[item[0]])

        self.family_ = family
        self.feature_names_ = names
        self.coef_ = result["coef"]
        self.intercept_ = float(result["intercept"])
        self.alpha_ = float(result["alpha"])
        self.training_mse_ = float(result["mse"])
        self.training_gcv_ = float(result["gcv"])
        self.training_df_ = float(result["df"])
        self.n_basis_terms_ = int(B.shape[1])
        self.is_fitted_ = True
        return self

    def _transform(self, X):
        Z = (X - self.scale_mean_) / self.scale_std_
        if self.family_ == "linear":
            B, _ = self._build_linear_basis(Z)
            return B
        if self.family_ == "linear+quadratic":
            B, _ = self._build_quadratic_basis(Z)
            return B
        B, _ = self._build_hinge_basis(Z)
        return B

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        X = np.asarray(X, dtype=float)
        B = self._transform(X)
        return B @ self.coef_ + self.intercept_

    def __str__(self):
        check_is_fitted(self, "is_fitted_")
        lines = ["Basis-Switch GCV Ridge Regressor:"]
        lines.append("  prediction = intercept + weighted sum of transparent basis terms")
        lines.append(f"  chosen family: {self.family_}")
        lines.append(f"  intercept: {self.intercept_:+.4f}")
        lines.append(f"  ridge alpha (GCV): {self.alpha_:.4g}")
        lines.append(f"  basis terms: {self.n_basis_terms_}")
        lines.append(f"  training MSE: {self.training_mse_:.6f}")
        if self.coef_.size == 0:
            lines.append("  no learned terms")
            return "\n".join(lines)

        top_k = int(max(1, self.display_top_k))
        order = np.argsort(np.abs(self.coef_))[::-1][:top_k]
        lines.append(f"  top |coefficient| terms shown: {len(order)}")
        for idx in order:
            lines.append(f"    {self.feature_names_[idx]}: {self.coef_[idx]:+.4f}")
        if self.coef_.size > top_k:
            lines.append(f"  ... plus {self.coef_.size - top_k} smaller terms")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
BasisSwitchRidgeRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "BasisSwitchRidge_v1"
model_description = "Custom GCV-ridge equation that selects the simplest effective basis among linear, quadratic, and quantile-hinge families"
model_defs = [(model_shorthand_name, BasisSwitchRidgeRegressor())]


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
