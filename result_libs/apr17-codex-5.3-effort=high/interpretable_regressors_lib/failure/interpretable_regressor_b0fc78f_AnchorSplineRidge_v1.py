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


class AnchorSplineRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Ridge backbone + one anchored spline feature.

    The model is:
      y = intercept + sum_j beta_j * xj
          + sum_k gamma_k * max(0, x_anchor - knot_k)

    One anchor feature is selected from top linear contributors if it improves
    validation RMSE by a minimum relative amount.
    """

    def __init__(
        self,
        val_fraction=0.2,
        alphas=(0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
        anchor_top_features=6,
        anchor_quantiles=(0.2, 0.5, 0.8),
        min_anchor_relative_gain=0.005,
        coef_tol=1e-10,
        decimals=6,
        random_state=0,
    ):
        self.val_fraction = val_fraction
        self.alphas = alphas
        self.anchor_top_features = anchor_top_features
        self.anchor_quantiles = anchor_quantiles
        self.min_anchor_relative_gain = min_anchor_relative_gain
        self.coef_tol = coef_tol
        self.decimals = decimals
        self.random_state = random_state

    @staticmethod
    def _rmse(y_true, y_pred):
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    def _split_indices(self, n):
        if n <= 80:
            idx = np.arange(n, dtype=int)
            return idx, idx
        rng = np.random.RandomState(self.random_state)
        order = rng.permutation(n)
        n_val = int(round(float(self.val_fraction) * n))
        n_val = min(max(n_val, 40), n - 40)
        va_idx = order[:n_val]
        tr_idx = order[n_val:]
        if tr_idx.size == 0:
            tr_idx = va_idx
        return tr_idx, va_idx

    @staticmethod
    def _solve_ridge_with_intercept(D, y, alpha):
        n, p = D.shape
        if p == 0:
            return float(np.mean(y)), np.zeros(0, dtype=float)
        scale = np.std(D, axis=0).astype(float)
        scale[scale < 1e-12] = 1.0
        Ds = D / scale
        A = np.column_stack([np.ones(n, dtype=float), Ds])
        reg = np.eye(p + 1, dtype=float)
        reg[0, 0] = 0.0
        lhs = A.T @ A + float(max(alpha, 0.0)) * reg
        rhs = A.T @ y
        try:
            sol = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        intercept = float(sol[0])
        coefs = np.asarray(sol[1:], dtype=float) / scale
        return intercept, coefs

    @staticmethod
    def _anchor_basis(X, feature_idx, knots):
        xj = X[:, int(feature_idx)]
        cols = [np.maximum(0.0, xj - float(t)) for t in np.asarray(knots, dtype=float)]
        return np.column_stack(cols) if cols else np.zeros((X.shape[0], 0), dtype=float)

    def _fit_design(self, X, y, alpha, anchor_feature=None, anchor_knots=None):
        p = X.shape[1]
        if anchor_feature is None or anchor_knots is None or len(anchor_knots) == 0:
            D = X
            intercept, coef = self._solve_ridge_with_intercept(D, y, alpha)
            return intercept, coef, None, np.zeros(0, dtype=float)

        B = self._anchor_basis(X, anchor_feature, anchor_knots)
        D = np.column_stack([X, B])
        intercept, coef = self._solve_ridge_with_intercept(D, y, alpha)
        beta = np.asarray(coef[:p], dtype=float)
        gamma = np.asarray(coef[p:], dtype=float)
        return intercept, beta, int(anchor_feature), gamma

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n_samples, n_features = X.shape
        self.n_features_in_ = int(n_features)

        tr_idx, va_idx = self._split_indices(n_samples)
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xva, yva = X[va_idx], y[va_idx]

        # 1) Select base ridge alpha.
        best_linear = None
        for alpha in self.alphas:
            inter, beta, _, _ = self._fit_design(Xtr, ytr, alpha, None, None)
            pred_va = inter + Xva @ beta
            rmse_va = self._rmse(yva, pred_va)
            if best_linear is None or rmse_va < best_linear["rmse_va"]:
                best_linear = {
                    "alpha": float(alpha),
                    "inter": float(inter),
                    "beta": np.asarray(beta, dtype=float),
                    "rmse_va": float(rmse_va),
                }

        # 2) Try one anchored spline feature.
        x_scale = np.std(Xtr, axis=0) + 1e-12
        contributions = np.abs(best_linear["beta"]) * x_scale
        order = np.argsort(contributions)[::-1]
        top_features = order[: min(int(self.anchor_top_features), n_features)]

        best_aug = {
            "alpha": float(best_linear["alpha"]),
            "inter": float(best_linear["inter"]),
            "beta": np.asarray(best_linear["beta"], dtype=float),
            "anchor_feature": None,
            "anchor_knots": np.zeros(0, dtype=float),
            "gamma": np.zeros(0, dtype=float),
            "rmse_va": float(best_linear["rmse_va"]),
        }

        for feat in top_features:
            knots = np.unique(np.quantile(Xtr[:, int(feat)], np.asarray(self.anchor_quantiles, dtype=float)))
            if knots.size == 0:
                continue
            for alpha in self.alphas:
                inter, beta, anchor_feature, gamma = self._fit_design(
                    Xtr,
                    ytr,
                    alpha,
                    int(feat),
                    knots,
                )
                pred_va = inter + Xva @ beta
                if gamma.size > 0:
                    pred_va = pred_va + self._anchor_basis(Xva, int(feat), knots) @ gamma
                rmse_va = self._rmse(yva, pred_va)
                if rmse_va < best_aug["rmse_va"]:
                    best_aug = {
                        "alpha": float(alpha),
                        "inter": float(inter),
                        "beta": np.asarray(beta, dtype=float),
                        "anchor_feature": anchor_feature,
                        "anchor_knots": np.asarray(knots, dtype=float),
                        "gamma": np.asarray(gamma, dtype=float),
                        "rmse_va": float(rmse_va),
                    }

        rel_gain = (best_linear["rmse_va"] - best_aug["rmse_va"]) / max(best_linear["rmse_va"], 1e-12)
        use_anchor = (
            best_aug["anchor_feature"] is not None
            and rel_gain >= float(self.min_anchor_relative_gain)
        )

        if not use_anchor:
            best_aug["anchor_feature"] = None
            best_aug["anchor_knots"] = np.zeros(0, dtype=float)
            best_aug["gamma"] = np.zeros(0, dtype=float)
            best_aug["alpha"] = float(best_linear["alpha"])

        # 3) Refit selected structure on full data.
        inter_full, beta_full, anchor_feature_full, gamma_full = self._fit_design(
            X,
            y,
            best_aug["alpha"],
            best_aug["anchor_feature"],
            best_aug["anchor_knots"],
        )

        beta_full[np.abs(beta_full) < self.coef_tol] = 0.0
        gamma_full[np.abs(gamma_full) < self.coef_tol] = 0.0
        if gamma_full.size == 0 or np.all(np.abs(gamma_full) <= self.coef_tol):
            anchor_feature_full = None
            gamma_full = np.zeros(0, dtype=float)
            best_aug["anchor_knots"] = np.zeros(0, dtype=float)

        self.alpha_selected_ = float(best_aug["alpha"])
        self.intercept_ = float(inter_full)
        self.linear_coefs_ = np.asarray(beta_full, dtype=float)
        self.anchor_feature_ = anchor_feature_full
        self.anchor_knots_ = np.asarray(best_aug["anchor_knots"], dtype=float)
        self.anchor_coefs_ = np.asarray(gamma_full, dtype=float)

        feature_imp = np.abs(self.linear_coefs_).copy()
        if self.anchor_feature_ is not None and self.anchor_coefs_.size > 0:
            feature_imp[int(self.anchor_feature_)] += float(np.sum(np.abs(self.anchor_coefs_)))
        self.feature_importance_ = feature_imp
        self.selected_features_ = sorted(int(i) for i in np.where(feature_imp > self.coef_tol)[0])
        self.operations_ = int(
            1
            + 2 * int(np.sum(np.abs(self.linear_coefs_) > self.coef_tol))
            + 3 * int(self.anchor_coefs_.size)
        )
        return self

    def predict(self, X):
        check_is_fitted(
            self,
            ["intercept_", "linear_coefs_", "anchor_feature_", "anchor_knots_", "anchor_coefs_"],
        )
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        pred = self.intercept_ + X @ self.linear_coefs_
        if self.anchor_feature_ is not None and self.anchor_coefs_.size > 0:
            pred = pred + self._anchor_basis(X, self.anchor_feature_, self.anchor_knots_) @ self.anchor_coefs_
        return pred

    def __str__(self):
        check_is_fitted(
            self,
            ["intercept_", "linear_coefs_", "anchor_feature_", "anchor_knots_", "anchor_coefs_"],
        )
        dec = int(self.decimals)

        terms = []
        for j, c in enumerate(self.linear_coefs_):
            if abs(float(c)) > self.coef_tol:
                terms.append((f"x{j}", float(c)))

        if self.anchor_feature_ is not None and self.anchor_coefs_.size > 0:
            for t, g in zip(self.anchor_knots_, self.anchor_coefs_):
                if abs(float(g)) > self.coef_tol:
                    name = f"max(0, x{int(self.anchor_feature_)} - {float(t):.{dec}f})"
                    terms.append((name, float(g)))

        terms.sort(key=lambda t: abs(t[1]), reverse=True)
        rhs = [f"{self.intercept_:+.{dec}f}"] + [f"({coef:+.{dec}f})*{name}" for name, coef in terms]

        lines = [
            "Anchor Spline Ridge Regressor",
            "Exact prediction equation:",
            "  y = " + " + ".join(rhs),
            "",
            "Only terms listed above contribute; omitted features have coefficient 0.",
            f"Selected ridge alpha: {self.alpha_selected_:.6g}",
        ]
        if self.anchor_feature_ is None or self.anchor_coefs_.size == 0:
            lines.append("Anchor spline feature: none")
        else:
            lines.append(
                f"Anchor spline feature: x{int(self.anchor_feature_)} with knots "
                + ", ".join(f"{float(t):.{dec}f}" for t in self.anchor_knots_)
            )
        lines.append(f"Approximate arithmetic operations: {self.operations_}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
AnchorSplineRidgeRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "AnchorSplineRidge_v1"
model_description = "Dense ridge backbone with validation-selected single-feature anchored spline residual using three hinge knots"
model_defs = [(model_shorthand_name, AnchorSplineRidgeRegressor())]


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
