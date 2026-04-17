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


class ProjectedSplineResidualRegressor(BaseEstimator, RegressorMixin):
    """Sparse linear model plus a compact spline over a learned 1D projection."""

    def __init__(
        self,
        ridge_alpha=1.0,
        max_active_features=6,
        knot_quantiles=(0.25, 0.5, 0.75),
        negligible_coef_eps=1e-5,
    ):
        self.ridge_alpha = ridge_alpha
        self.max_active_features = max_active_features
        self.knot_quantiles = knot_quantiles
        self.negligible_coef_eps = negligible_coef_eps

    @staticmethod
    def _corr(a, b):
        a0 = a - np.mean(a)
        b0 = b - np.mean(b)
        den = (np.std(a0) + 1e-12) * (np.std(b0) + 1e-12)
        return float(np.mean(a0 * b0) / den)

    @staticmethod
    def _ridge_fit(X, y, alpha, penalize_mask=None):
        p = X.shape[1]
        reg = float(alpha) * np.eye(p)
        if penalize_mask is not None:
            pm = np.asarray(penalize_mask, dtype=float)
            reg = reg * pm[:, None]
        return np.linalg.solve(X.T @ X + reg, X.T @ y)

    def _hinge_basis(self, z, knots):
        cols = [z]
        for k in knots:
            cols.append(np.maximum(0.0, z - k))
        return np.column_stack(cols)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        _, n_features = X.shape
        self.n_features_in_ = n_features

        y_mean = float(np.mean(y))
        self.intercept_ = y_mean
        yc = y - y_mean

        corr = np.array([abs(self._corr(X[:, j], yc)) for j in range(n_features)], dtype=float)
        if np.all(~np.isfinite(corr)):
            corr = np.zeros_like(corr)
        order = np.argsort(np.nan_to_num(corr, nan=0.0))[::-1]
        max_act = max(1, min(int(self.max_active_features), n_features))
        self.active_features_ = np.array([int(j) for j in order[:max_act]], dtype=int)

        Xa = X[:, self.active_features_]

        # Stage 1: learn a sparse projection z = w^T x_active.
        self.projection_coef_ = self._ridge_fit(Xa, yc, self.ridge_alpha)
        z = Xa @ self.projection_coef_

        # Stage 2: fit sparse linear backbone + hinge spline in projected space.
        self.spline_knots_ = np.array(
            [float(np.quantile(z, q)) for q in self.knot_quantiles],
            dtype=float,
        )
        Z = self._hinge_basis(z, self.spline_knots_)
        design = np.column_stack([Xa, Z])
        coef = self._ridge_fit(design, yc, self.ridge_alpha)
        n_linear = Xa.shape[1]
        self.linear_coef_active_ = coef[:n_linear]
        self.spline_coef_ = coef[n_linear:]

        self.linear_coef_active_[np.abs(self.linear_coef_active_) < self.negligible_coef_eps] = 0.0
        self.projection_coef_[np.abs(self.projection_coef_) < self.negligible_coef_eps] = 0.0
        self.spline_coef_[np.abs(self.spline_coef_) < self.negligible_coef_eps] = 0.0

        self.linear_coef_full_ = np.zeros(n_features, dtype=float)
        self.linear_coef_full_[self.active_features_] = self.linear_coef_active_

        imp = np.abs(self.linear_coef_full_)
        if np.max(imp) > 0.0:
            imp = imp / float(np.max(imp))
        self.feature_importances_ = imp

        return self

    def predict(self, X):
        check_is_fitted(
            self,
            [
                "intercept_",
                "active_features_",
                "projection_coef_",
                "linear_coef_active_",
                "spline_coef_",
                "spline_knots_",
            ],
        )
        X = np.asarray(X, dtype=float)
        Xa = X[:, self.active_features_]
        z = Xa @ self.projection_coef_
        Z = self._hinge_basis(z, self.spline_knots_)
        pred = self.intercept_ + Xa @ self.linear_coef_active_ + Z @ self.spline_coef_
        return pred.astype(float)

    def __str__(self):
        check_is_fitted(
            self,
            [
                "intercept_",
                "active_features_",
                "projection_coef_",
                "linear_coef_full_",
                "spline_coef_",
                "spline_knots_",
                "feature_importances_",
            ],
        )

        ranked = np.argsort(self.feature_importances_)[::-1]
        active_ranked = [int(j) for j in ranked if abs(self.linear_coef_full_[j]) >= self.negligible_coef_eps]

        z_terms = []
        for local_idx, feat in enumerate(self.active_features_):
            coef = float(self.projection_coef_[local_idx])
            if abs(coef) >= self.negligible_coef_eps:
                z_terms.append(f"{coef:+.4f}*x{int(feat)}")
        z_expr = " ".join(z_terms) if z_terms else "0.0"

        y_terms = [f"{self.intercept_:+.4f}"]
        for feat in active_ranked:
            y_terms.append(f"{float(self.linear_coef_full_[feat]):+.4f}*x{feat}")
        if abs(float(self.spline_coef_[0])) >= self.negligible_coef_eps:
            y_terms.append(f"{float(self.spline_coef_[0]):+.4f}*z")
        for i, knot in enumerate(self.spline_knots_):
            c = float(self.spline_coef_[1 + i])
            if abs(c) >= self.negligible_coef_eps:
                y_terms.append(f"{c:+.4f}*max(0, z-{float(knot):.4f})")

        top_txt = ", ".join(f"x{j}:{self.feature_importances_[j]:.3f}" for j in active_ranked[:10])
        inactive = [f"x{i}" for i in range(self.n_features_in_) if i not in set(self.active_features_)]

        lines = [
            "Projected Spline Residual Regressor",
            "Prediction uses a sparse linear backbone plus a 1D spline correction.",
            "Step 1 (projection score):",
            f"  z = {z_expr}",
            "Step 2 (prediction equation):",
            "  y = " + " ".join(y_terms),
            "",
            "Top feature influence ranking: " + (top_txt if top_txt else "none"),
            "Inactive features: " + (", ".join(inactive) if inactive else "none"),
        ]
        return "\n".join(lines)
# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])

# Update the model shorthand name and description below to reflect the class above and any changes you make.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
ProjectedSplineResidualRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "ProjectedSplineResidual_v1"
model_description = "Sparse screened linear backbone with a global projected hinge-spline correction and explicit two-step symbolic equation"
model_defs = [(model_shorthand_name, ProjectedSplineResidualRegressor())]


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
