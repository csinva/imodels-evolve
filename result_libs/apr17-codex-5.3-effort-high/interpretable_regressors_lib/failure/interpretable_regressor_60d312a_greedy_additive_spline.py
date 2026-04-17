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


class GreedyAdditiveSplineRegressor(BaseEstimator, RegressorMixin):
    """
    Dense ridge backbone + greedy sparse additive spline residual.

    Step 1) Fit a dense GCV-selected ridge model on standardized features.
    Step 2) Fit residuals with a small set of additive basis terms
            (one-sided hinges + centered quadratics) chosen greedily by
            validation RMSE improvement.
    """

    def __init__(
        self,
        alpha_grid=None,
        basis_alpha=0.2,
        top_linear_features=8,
        max_basis_terms=3,
        quantiles=(0.25, 0.5, 0.75),
        min_rel_improve=0.004,
        random_state=42,
    ):
        self.alpha_grid = alpha_grid
        self.basis_alpha = basis_alpha
        self.top_linear_features = top_linear_features
        self.max_basis_terms = max_basis_terms
        self.quantiles = quantiles
        self.min_rel_improve = min_rel_improve
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
        if self.alpha_grid is None:
            grid = np.logspace(-5, 3, 20)
        else:
            grid = np.asarray(self.alpha_grid, dtype=float)
        grid = np.maximum(grid, 1e-12)

        y_centered = y - np.mean(y)
        U, s, _ = np.linalg.svd(Z, full_matrices=False)
        s2 = s ** 2
        Uy = U.T @ y_centered

        best_alpha = float(grid[0])
        best_gcv = float("inf")
        for alpha in grid:
            shrink = s2 / (s2 + alpha)
            yhat_centered = U @ (shrink * Uy)
            resid = y_centered - yhat_centered
            mse = float(np.mean(resid ** 2))
            df = float(np.sum(shrink))
            denom = max((1.0 - df / max(n, 1)) ** 2, 1e-8)
            gcv = mse / denom
            if gcv < best_gcv:
                best_gcv = gcv
                best_alpha = float(alpha)
        return best_alpha, best_gcv

    @staticmethod
    def _build_basis_column(X, basis):
        kind = basis["kind"]
        j = basis["j"]
        if kind == "hinge_pos":
            return np.maximum(0.0, X[:, j] - basis["t"])
        if kind == "hinge_neg":
            return np.maximum(0.0, basis["t"] - X[:, j])
        if kind == "quad_centered":
            return (X[:, j] - basis["m"]) ** 2
        raise ValueError(f"Unknown basis kind: {kind}")

    def _candidate_bases(self, X, w_lin):
        p = X.shape[1]
        k = min(max(int(self.top_linear_features), 1), p)
        top = np.argsort(-np.abs(w_lin))[:k]
        top = np.asarray(top, dtype=int)
        qs = tuple(float(q) for q in self.quantiles)

        bases = []
        for j in top:
            xj = X[:, j]
            for q in qs:
                t = float(np.quantile(xj, q))
                bases.append({
                    "kind": "hinge_pos",
                    "j": int(j),
                    "t": t,
                    "name": f"max(0, x{j} - {t:.3f})",
                })
                bases.append({
                    "kind": "hinge_neg",
                    "j": int(j),
                    "t": t,
                    "name": f"max(0, {t:.3f} - x{j})",
                })
            m = float(np.mean(xj))
            bases.append({
                "kind": "quad_centered",
                "j": int(j),
                "m": m,
                "name": f"(x{j} - {m:.3f})^2",
            })
        return bases

    def _fit_residual_basis(self, X, residual, bases, alpha):
        n = X.shape[0]
        if len(bases) == 0:
            return 0.0, np.array([]), [], np.zeros((n, 0)), np.array([]), np.array([])

        G_raw = np.column_stack([self._build_basis_column(X, b) for b in bases]).astype(float)
        g_mean = G_raw.mean(axis=0)
        g_std = G_raw.std(axis=0)
        g_std = np.where(g_std < 1e-8, 1.0, g_std)
        G = (G_raw - g_mean) / g_std
        b0, w = self._ridge_fit(G, residual, alpha)
        return b0, w, bases, G, g_mean, g_std

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        self.n_features_in_ = p

        mu = X.mean(axis=0)
        scale = X.std(axis=0)
        scale = np.where(scale < 1e-8, 1.0, scale)
        Z = (X - mu) / scale

        alpha, gcv = self._select_alpha_gcv(Z, y)
        b_lin, w_lin = self._ridge_fit(Z, y, alpha)
        coef_raw = w_lin / scale
        intercept_raw = float(b_lin - np.dot(coef_raw, mu))

        y_lin = intercept_raw + X @ coef_raw
        residual = y - y_lin

        rng = np.random.RandomState(self.random_state)
        idx = np.arange(n)
        rng.shuffle(idx)
        n_tr = max(12, int(0.85 * n))
        tr = idx[:n_tr]
        va = idx[n_tr:] if n_tr < n else idx[: max(1, n // 6)]

        candidate_bases = self._candidate_bases(X, w_lin)
        chosen = []

        base_rmse = float(np.sqrt(np.mean((y[va] - y_lin[va]) ** 2)))
        best_rmse = base_rmse
        max_terms = max(0, int(self.max_basis_terms))

        for _ in range(max_terms):
            best_basis = None
            best_basis_rmse = best_rmse
            for cand in candidate_bases:
                if cand in chosen:
                    continue
                trial = chosen + [cand]
                b0, w_b, _, _, g_mean, g_std = self._fit_residual_basis(
                    X[tr], residual[tr], trial, float(self.basis_alpha)
                )
                G_va = np.column_stack([self._build_basis_column(X[va], b) for b in trial]).astype(float)
                G_va = (G_va - g_mean) / g_std
                pred = y_lin[va] + b0 + G_va @ w_b
                rmse = float(np.sqrt(np.mean((y[va] - pred) ** 2)))
                if rmse < best_basis_rmse:
                    best_basis_rmse = rmse
                    best_basis = cand

            if best_basis is None or best_basis_rmse >= best_rmse * (1.0 - float(self.min_rel_improve)):
                break
            chosen.append(best_basis)
            best_rmse = best_basis_rmse

        b_res, w_res, chosen_bases, _, g_mean, g_std = self._fit_residual_basis(
            X, residual, chosen, float(self.basis_alpha)
        )
        basis_raw_coef = []
        for k_idx in range(len(chosen_bases)):
            c = float(w_res[k_idx] / g_std[k_idx])
            basis_raw_coef.append(c)
            b_res -= c * float(g_mean[k_idx])

        self.alpha_ = float(alpha)
        self.gcv_score_ = float(gcv)
        self.mu_ = mu
        self.scale_ = scale
        self.intercept_ = float(intercept_raw + b_res)
        self.coef_ = np.asarray(coef_raw, dtype=float)
        self.basis_terms_ = chosen_bases
        self.basis_coefs_ = np.asarray(basis_raw_coef, dtype=float)
        self.active_features_ = np.where(np.abs(self.coef_) > 1e-12)[0].astype(int)
        self.val_rmse_linear_ = float(base_rmse)
        self.val_rmse_final_ = float(best_rmse)
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "coef_", "basis_terms_", "basis_coefs_"])
        X = np.asarray(X, dtype=float)
        pred = self.intercept_ + X @ self.coef_
        for c, basis in zip(self.basis_coefs_, self.basis_terms_):
            pred = pred + float(c) * self._build_basis_column(X, basis)
        return pred

    @staticmethod
    def _linear_terms(coef, limit=12):
        active = [j for j in range(len(coef)) if abs(float(coef[j])) > 1e-12]
        active = sorted(active, key=lambda j: -abs(float(coef[j])))
        shown = active[:limit]
        terms = [f"{float(coef[j]):+.6f}*x{j}" for j in shown]
        return terms, active, shown

    def __str__(self):
        check_is_fitted(self, ["intercept_", "coef_", "basis_terms_", "basis_coefs_"])
        terms, active, shown = self._linear_terms(self.coef_)
        expr = f"{float(self.intercept_):.6f}"
        if terms:
            expr += " " + " ".join(terms)
        for c, b in zip(self.basis_coefs_, self.basis_terms_):
            expr += f" {float(c):+.6f}*{b['name']}"

        lines = [
            "Greedy Additive Spline Residual Regressor",
            f"Chosen ridge alpha (GCV): {self.alpha_:.6g}",
            f"GCV score: {self.gcv_score_:.6f}",
            f"Validation RMSE (linear only): {self.val_rmse_linear_:.6f}",
            f"Validation RMSE (final model): {self.val_rmse_final_:.6f}",
            "Raw-feature prediction equation:",
            f"y = {expr}",
            f"Active linear features ({len(active)}): {', '.join(f'x{j}' for j in active) if active else '(none)'}",
            "Linear coefficients (sorted by |coefficient|):",
        ]
        for j in shown:
            lines.append(f"  x{j}: {float(self.coef_[j]):+.6f}")
        if len(active) > len(shown):
            lines.append(f"  ... {len(active) - len(shown)} more linear terms")
        if not active:
            lines.append("  (none)")

        if len(self.basis_terms_) > 0:
            lines.append("Residual additive basis terms:")
            for c, b in zip(self.basis_coefs_, self.basis_terms_):
                lines.append(f"  {float(c):+.6f} * {b['name']}")
        else:
            lines.append("Residual additive basis terms: (none)")

        lines.append(f"Intercept: {float(self.intercept_):+.6f}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
GreedyAdditiveSplineRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "GreedyAdditiveSplineV1"
model_description = "Dense GCV ridge backbone plus greedy-selected sparse additive residual spline terms (one-sided hinges and centered quadratics) on top linear features"
model_defs = [(model_shorthand_name, GreedyAdditiveSplineRegressor())]


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
