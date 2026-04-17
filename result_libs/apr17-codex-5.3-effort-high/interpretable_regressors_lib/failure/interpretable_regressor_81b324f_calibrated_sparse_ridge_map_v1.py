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


class CalibratedSparseRidgeMapRegressor(BaseEstimator, RegressorMixin):
    """
    Robust GCV ridge in raw-feature form with one optional hinge correction.

    Design goals:
    - Preserve strong predictive performance with dense linear ridge backbone.
    - Keep simulation easy via a direct raw-feature equation.
    - Add at most one piecewise term if it materially improves validation RMSE.
    """

    def __init__(
        self,
        alpha_grid=None,
        max_hinge_terms=1,
        hinge_quantiles=(0.2, 0.5, 0.8),
        top_hinge_features=6,
        hinge_alpha=0.05,
        min_rel_improve=0.008,
        negligible_effect_ratio=0.03,
        random_state=42,
    ):
        self.alpha_grid = alpha_grid
        self.max_hinge_terms = max_hinge_terms
        self.hinge_quantiles = hinge_quantiles
        self.top_hinge_features = top_hinge_features
        self.hinge_alpha = hinge_alpha
        self.min_rel_improve = min_rel_improve
        self.negligible_effect_ratio = negligible_effect_ratio
        self.random_state = random_state

    @staticmethod
    def _ridge_fit(X, y, alpha):
        n, p = X.shape
        D = np.hstack([np.ones((n, 1), dtype=float), X])
        reg = np.zeros(p + 1, dtype=float)
        reg[1:] = max(float(alpha), 0.0)
        A = D.T @ D + np.diag(reg)
        b = D.T @ y
        try:
            beta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(A) @ b
        return float(beta[0]), np.asarray(beta[1:], dtype=float)

    def _select_alpha_gcv(self, Z, y):
        n = Z.shape[0]
        grid = np.logspace(-5, 3, 20) if self.alpha_grid is None else np.asarray(self.alpha_grid, dtype=float)
        grid = np.maximum(grid, 1e-12)

        y_centered = y - float(np.mean(y))
        U, s, _ = np.linalg.svd(Z, full_matrices=False)
        s2 = s ** 2
        Uy = U.T @ y_centered

        best_alpha, best_gcv = float(grid[0]), float("inf")
        for alpha in grid:
            shrink = s2 / (s2 + alpha)
            yhat_centered = U @ (shrink * Uy)
            resid = y_centered - yhat_centered
            mse = float(np.mean(resid ** 2))
            df = float(np.sum(shrink))
            denom = max((1.0 - df / max(n, 1)) ** 2, 1e-8)
            gcv = mse / denom
            if gcv < best_gcv:
                best_alpha = float(alpha)
                best_gcv = gcv
        return best_alpha, best_gcv

    @staticmethod
    def _build_hinge_column(X, basis):
        xj = X[:, basis["j"]]
        t = basis["t"]
        if basis["kind"] == "hinge_pos":
            return np.maximum(0.0, xj - t)
        if basis["kind"] == "hinge_neg":
            return np.maximum(0.0, t - xj)
        raise ValueError(f"unknown hinge kind: {basis['kind']}")

    def _candidate_hinges(self, X, w_std):
        p = X.shape[1]
        k = min(max(int(self.top_hinge_features), 1), p)
        top = np.argsort(-np.abs(w_std))[:k]
        thresholds = tuple(float(q) for q in self.hinge_quantiles)

        bases = []
        for j in top:
            xj = X[:, int(j)]
            for q in thresholds:
                t = float(np.quantile(xj, q))
                bases.append({"kind": "hinge_pos", "j": int(j), "t": t, "name": f"max(0, x{int(j)} - {t:.4f})"})
                bases.append({"kind": "hinge_neg", "j": int(j), "t": t, "name": f"max(0, {t:.4f} - x{int(j)})"})
        return bases

    def _fit_hinge_residual(self, X, residual, hinges, alpha):
        n = X.shape[0]
        if len(hinges) == 0:
            return 0.0, np.array([]), np.zeros((n, 0)), np.array([]), np.array([])

        H_raw = np.column_stack([self._build_hinge_column(X, h) for h in hinges]).astype(float)
        h_mean = H_raw.mean(axis=0)
        h_std = H_raw.std(axis=0)
        h_std = np.where(h_std < 1e-8, 1.0, h_std)
        H = (H_raw - h_mean) / h_std

        b0, w = self._ridge_fit(H, residual, alpha)
        return b0, w, H, h_mean, h_std

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        self.n_features_in_ = p

        med = np.median(X, axis=0)
        q75 = np.quantile(X, 0.75, axis=0)
        q25 = np.quantile(X, 0.25, axis=0)
        iqr = q75 - q25
        scale = np.where(iqr < 1e-8, np.std(X, axis=0), iqr)
        scale = np.where(scale < 1e-8, 1.0, scale)

        Z = (X - med) / scale
        alpha, gcv = self._select_alpha_gcv(Z, y)
        b_std, w_std = self._ridge_fit(Z, y, alpha)

        coef_raw = w_std / scale
        intercept_raw = float(b_std - np.dot(coef_raw, med))

        y_linear = intercept_raw + X @ coef_raw
        residual = y - y_linear

        rng = np.random.RandomState(self.random_state)
        idx = np.arange(n)
        rng.shuffle(idx)
        n_tr = max(20, int(0.85 * n))
        tr = idx[:n_tr]
        va = idx[n_tr:] if n_tr < n else idx[: max(1, n // 6)]

        candidate_hinges = self._candidate_hinges(X, w_std)
        chosen = []
        base_rmse = float(np.sqrt(np.mean((y[va] - y_linear[va]) ** 2)))
        best_rmse = base_rmse

        for _ in range(max(0, int(self.max_hinge_terms))):
            best_basis = None
            best_basis_rmse = best_rmse

            for cand in candidate_hinges:
                if cand in chosen:
                    continue
                trial = chosen + [cand]
                b0, w_h, _, h_mean, h_std = self._fit_hinge_residual(X[tr], residual[tr], trial, float(self.hinge_alpha))
                H_va = np.column_stack([self._build_hinge_column(X[va], h) for h in trial]).astype(float)
                H_va = (H_va - h_mean) / h_std
                pred = y_linear[va] + b0 + H_va @ w_h
                rmse = float(np.sqrt(np.mean((y[va] - pred) ** 2)))
                if rmse < best_basis_rmse:
                    best_basis_rmse = rmse
                    best_basis = cand

            if best_basis is None or best_basis_rmse >= best_rmse * (1.0 - float(self.min_rel_improve)):
                break
            chosen.append(best_basis)
            best_rmse = best_basis_rmse

        b_h, w_h, _, h_mean, h_std = self._fit_hinge_residual(X, residual, chosen, float(self.hinge_alpha))
        hinge_raw_coef = []
        for i in range(len(chosen)):
            c = float(w_h[i] / h_std[i])
            hinge_raw_coef.append(c)
            b_h -= c * float(h_mean[i])

        effect = np.abs(coef_raw) * np.std(X, axis=0)
        max_effect = float(np.max(effect)) if p > 0 else 0.0
        cutoff = float(self.negligible_effect_ratio) * max(max_effect, 1e-12)
        negligible = np.where(effect <= cutoff)[0].astype(int)
        meaningful = np.where(effect > cutoff)[0].astype(int)

        self.intercept_ = float(intercept_raw + b_h)
        self.coef_ = np.asarray(coef_raw, dtype=float)
        self.hinge_terms_ = chosen
        self.hinge_coefs_ = np.asarray(hinge_raw_coef, dtype=float)
        self.alpha_ = float(alpha)
        self.gcv_score_ = float(gcv)
        self.val_rmse_linear_ = float(base_rmse)
        self.val_rmse_final_ = float(best_rmse)
        self.effect_scores_ = effect
        self.meaningful_features_ = meaningful
        self.negligible_features_ = negligible
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "coef_", "hinge_terms_", "hinge_coefs_"])
        X = np.asarray(X, dtype=float)
        y = self.intercept_ + X @ self.coef_
        for c, h in zip(self.hinge_coefs_, self.hinge_terms_):
            y = y + float(c) * self._build_hinge_column(X, h)
        return y

    def __str__(self):
        check_is_fitted(self, ["intercept_", "coef_", "hinge_terms_", "hinge_coefs_"])

        order = np.argsort(-np.abs(self.coef_))
        eq_terms = [f"{float(self.coef_[j]):+.6f}*x{int(j)}" for j in order if abs(float(self.coef_[j])) > 1e-12]
        eq = f"y = {float(self.intercept_):.6f}"
        if eq_terms:
            eq += " " + " ".join(eq_terms)
        for c, h in zip(self.hinge_coefs_, self.hinge_terms_):
            eq += f" {float(c):+.6f}*{h['name']}"

        meaningful = ", ".join(f"x{int(j)}" for j in self.meaningful_features_) or "(none)"
        negligible = ", ".join(f"x{int(j)}" for j in self.negligible_features_) or "(none)"

        lines = [
            "Calibrated Sparse Ridge Map Regressor",
            f"Ridge alpha (GCV): {self.alpha_:.6g}",
            f"Validation RMSE linear={self.val_rmse_linear_:.6f} final={self.val_rmse_final_:.6f}",
            "Raw-feature prediction equation (exact):",
            eq,
            "Feature coefficients sorted by absolute size:",
        ]
        for j in order:
            lines.append(f"  x{int(j)}: coef={float(self.coef_[j]):+.6f}  effect={float(self.effect_scores_[j]):.6f}")

        if len(self.hinge_terms_) > 0:
            lines.append("Piecewise hinge corrections:")
            for c, h in zip(self.hinge_coefs_, self.hinge_terms_):
                lines.append(f"  {float(c):+.6f} * {h['name']}")
        else:
            lines.append("Piecewise hinge corrections: (none)")

        lines.append(f"Meaningful features: {meaningful}")
        lines.append(f"Negligible features: {negligible}")

        # Rough operation count for compactness-oriented prompts.
        n_linear = int(np.sum(np.abs(self.coef_) > 1e-12))
        n_hinge = len(self.hinge_terms_)
        op_count = 1 + 2 * n_linear + 3 * n_hinge
        lines.append(f"Approx operations to evaluate: {op_count}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
CalibratedSparseRidgeMapRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "CalibratedSparseRidgeMapV1"
model_description = "Robust GCV ridge expressed directly in raw-feature equation form with meaningful/negligible feature partition and at most one validation-gated hinge correction"
model_defs = [(model_shorthand_name, CalibratedSparseRidgeMapRegressor())]

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
