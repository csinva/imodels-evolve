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


class ResidualSnapSplineRegressor(BaseEstimator, RegressorMixin):
    """
    Two-stage interpretable equation:
      1) dense ridge linear backbone on centered raw features
      2) sparse residual correction with per-feature quantile hinge triplets

    The second stage selects only a few features via residual-gain screening.
    Hinge basis columns are centered with training-set means (stored), so
    prediction-time behavior is stable and deterministic.
    """

    def __init__(
        self,
        max_spline_features=3,
        alpha_grid=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
        hinge_quantiles=(0.2, 0.5, 0.8),
        min_gain_ratio=0.01,
        prune_group_ratio=0.1,
        coef_display_tol=0.03,
    ):
        self.max_spline_features = max_spline_features
        self.alpha_grid = alpha_grid
        self.hinge_quantiles = hinge_quantiles
        self.min_gain_ratio = min_gain_ratio
        self.prune_group_ratio = prune_group_ratio
        self.coef_display_tol = coef_display_tol

    @staticmethod
    def _ridge_fit(X, y, alpha):
        if X.shape[1] == 0:
            return np.zeros(0, dtype=float)
        A = X.T @ X + float(alpha) * np.eye(X.shape[1], dtype=float)
        return np.linalg.solve(A, X.T @ y)

    @staticmethod
    def _gcv_alpha(X, y, alpha_grid):
        if X.shape[1] == 0:
            return 1.0
        n = X.shape[0]
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        Uy = U.T @ y
        best_alpha = float(alpha_grid[0])
        best_score = None
        for a in alpha_grid:
            a = float(a)
            filt = s / (s * s + a)
            w = Vt.T @ (filt * Uy)
            pred = X @ w
            mse = float(np.mean((y - pred) ** 2))
            dof = float(np.sum((s * s) / (s * s + a)))
            denom = max(1e-8, 1.0 - dof / max(1.0, float(n)))
            gcv = mse / (denom * denom)
            if (best_score is None) or (gcv < best_score):
                best_score = gcv
                best_alpha = a
        return best_alpha

    def _sanitize_alpha_grid(self):
        alphas = np.asarray(self.alpha_grid, dtype=float)
        alphas = alphas[np.isfinite(alphas) & (alphas > 0)]
        if alphas.size == 0:
            return np.asarray([1.0], dtype=float)
        return alphas

    def _feature_hinges(self, x, knots):
        cols = []
        for t in knots:
            cols.append(np.maximum(0.0, x - t))
        return np.column_stack(cols) if cols else np.zeros((x.shape[0], 0), dtype=float)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        self.n_features_in_ = int(p)

        if p == 0:
            self.x_mean_ = np.zeros(0, dtype=float)
            self.linear_coef_ = np.zeros(0, dtype=float)
            self.linear_alpha_ = 1.0
            self.spline_alpha_ = 1.0
            self.selected_features_ = np.zeros(0, dtype=int)
            self.spline_knots_ = np.zeros((0, 0), dtype=float)
            self.spline_coef_ = np.zeros(0, dtype=float)
            self.spline_basis_means_ = np.zeros(0, dtype=float)
            self.intercept_ = float(np.mean(y))
            self.training_mse_ = float(np.mean((y - self.intercept_) ** 2))
            self.is_fitted_ = True
            return self

        x_mean = np.mean(X, axis=0)
        Xc = X - x_mean
        y_mean = float(np.mean(y))
        yc = y - y_mean

        alphas = self._sanitize_alpha_grid()
        alpha_lin = self._gcv_alpha(Xc, yc, alphas)
        w_lin = self._ridge_fit(Xc, yc, alpha_lin)
        resid = yc - Xc @ w_lin

        max_feats = int(max(0, min(int(self.max_spline_features), p)))
        q = np.asarray(self.hinge_quantiles, dtype=float)
        q = q[np.isfinite(q) & (q > 0) & (q < 1)]
        q = np.unique(np.clip(q, 1e-4, 1.0 - 1e-4))
        if q.size == 0:
            q = np.asarray([0.5], dtype=float)

        scored = []
        resid_var = float(np.var(resid)) + 1e-12
        for j in range(p):
            knots = np.quantile(X[:, j], q).astype(float)
            H = self._feature_hinges(X[:, j], knots)
            if H.shape[1] == 0:
                continue
            H_mean = np.mean(H, axis=0)
            Hc = H - H_mean
            a_j = self._gcv_alpha(Hc, resid, alphas)
            b_j = self._ridge_fit(Hc, resid, a_j)
            new_resid = resid - Hc @ b_j
            gain = float(np.var(resid) - np.var(new_resid))
            scored.append((gain, int(j), knots))

        scored.sort(key=lambda t: t[0], reverse=True)
        selected = []
        min_gain = float(self.min_gain_ratio) * resid_var
        for gain, j, knots in scored:
            if len(selected) >= max_feats:
                break
            if gain <= min_gain and len(selected) > 0:
                break
            selected.append((j, knots))

        Z_cols = []
        Z_meta = []
        if selected:
            for j, knots in selected:
                H = self._feature_hinges(X[:, j], knots)
                for k, t in enumerate(knots):
                    Z_cols.append(H[:, k])
                    Z_meta.append((int(j), float(t)))
        Z = np.column_stack(Z_cols) if Z_cols else np.zeros((n, 0), dtype=float)
        Z_mean = np.mean(Z, axis=0) if Z.shape[1] else np.zeros(0, dtype=float)
        Zc = Z - Z_mean if Z.shape[1] else Z

        alpha_spline = self._gcv_alpha(Zc, resid, alphas) if Zc.shape[1] else 1.0
        b_spline = self._ridge_fit(Zc, resid, alpha_spline)

        if selected and b_spline.size:
            group_abs = []
            n_knots = q.size
            for g in range(len(selected)):
                seg = b_spline[g * n_knots:(g + 1) * n_knots]
                group_abs.append(float(np.linalg.norm(seg, ord=1)))
            max_group = max(group_abs) if group_abs else 0.0
            keep_groups = [
                g for g, val in enumerate(group_abs)
                if val >= float(self.prune_group_ratio) * max(1e-12, max_group)
            ]
            if len(keep_groups) < len(selected):
                new_Z_meta = []
                keep_cols = []
                for g in keep_groups:
                    base = g * n_knots
                    for k in range(n_knots):
                        keep_cols.append(base + k)
                        new_Z_meta.append(Z_meta[base + k])
                if keep_cols:
                    Z = Z[:, keep_cols]
                    Z_mean = Z_mean[keep_cols]
                    Zc = Z - Z_mean
                    b_spline = self._ridge_fit(Zc, resid, alpha_spline)
                    Z_meta = new_Z_meta
                    keep_features = sorted({Z_meta[i][0] for i in range(len(Z_meta))})
                    selected = [(j, np.quantile(X[:, j], q).astype(float)) for j in keep_features]
                else:
                    Z = np.zeros((n, 0), dtype=float)
                    Z_mean = np.zeros(0, dtype=float)
                    Zc = Z
                    b_spline = np.zeros(0, dtype=float)
                    Z_meta = []
                    selected = []

        pred = y_mean + Xc @ w_lin + (Zc @ b_spline if Zc.shape[1] else 0.0)

        self.x_mean_ = np.asarray(x_mean, dtype=float)
        self.linear_coef_ = np.asarray(w_lin, dtype=float)
        self.linear_alpha_ = float(alpha_lin)
        self.spline_alpha_ = float(alpha_spline)
        self.spline_coef_ = np.asarray(b_spline, dtype=float)
        self.spline_basis_means_ = np.asarray(Z_mean, dtype=float)
        self.spline_meta_ = list(Z_meta)
        self.selected_features_ = np.asarray(sorted({j for j, _ in selected}), dtype=int)
        self.intercept_ = float(y_mean - np.dot(self.x_mean_, self.linear_coef_))
        self.training_mse_ = float(np.mean((y - pred) ** 2))
        self.is_fitted_ = True
        return self

    def _build_spline_matrix(self, X):
        if len(self.spline_meta_) == 0:
            return np.zeros((X.shape[0], 0), dtype=float)
        cols = []
        for j, t in self.spline_meta_:
            cols.append(np.maximum(0.0, X[:, j] - t))
        return np.column_stack(cols)

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        X = np.asarray(X, dtype=float)
        Xc = X - self.x_mean_
        yhat = self.intercept_ + X @ self.linear_coef_
        if self.spline_coef_.size:
            Z = self._build_spline_matrix(X)
            Zc = Z - self.spline_basis_means_
            yhat = yhat + Zc @ self.spline_coef_
        return yhat

    def __str__(self):
        check_is_fitted(self, "is_fitted_")
        lines = ["Residual Snap-Spline Equation Regressor:"]
        lines.append(f"  intercept: {self.intercept_:+.4f}")
        lines.append(f"  linear ridge alpha (GCV): {self.linear_alpha_:.6f}")
        lines.append(f"  spline ridge alpha (GCV): {self.spline_alpha_:.6f}")
        lines.append(f"  spline-selected features: {len(self.selected_features_)}")
        lines.append(f"  training MSE: {self.training_mse_:.6f}")

        lin_terms = []
        for j, c in enumerate(self.linear_coef_):
            if abs(c) >= float(self.coef_display_tol):
                lin_terms.append((abs(float(c)), int(j), float(c)))
        lin_terms.sort(reverse=True)

        if lin_terms:
            lines.append("  linear backbone (largest coefficients):")
            for _, j, c in lin_terms[:12]:
                lines.append(f"    {c:+.4f} * x{j}")

        if self.spline_coef_.size:
            lines.append("  residual hinge corrections:")
            items = []
            for coef, (j, t), m in zip(self.spline_coef_, self.spline_meta_, self.spline_basis_means_):
                if abs(float(coef)) < float(self.coef_display_tol):
                    continue
                items.append((abs(float(coef)), int(j), float(t), float(m), float(coef)))
            items.sort(reverse=True)
            for _, j, t, m, c in items:
                lines.append(f"    {c:+.4f} * (max(0, x{j} - {t:.4f}) - {m:.4f})")

        approx_ops = 1 + int(np.count_nonzero(np.abs(self.linear_coef_) > 0.0)) + 4 * len(self.spline_meta_)
        lines.append(f"  approximate arithmetic operations: ~{approx_ops}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
ResidualSnapSplineRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "ResidualSnapSpline_v1"
model_description = "Two-stage equation: GCV-ridge linear backbone plus sparse residual quantile-hinge triplet corrections with stored training centering"
model_defs = [(model_shorthand_name, ResidualSnapSplineRegressor())]


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
