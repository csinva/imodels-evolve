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


class AdaptiveElasticHingeMapRegressor(BaseEstimator, RegressorMixin):
    """
    Adaptive weighted-L1 linear model with debiasing and one optional hinge term.

    Steps:
    1) Robustly standardize features.
    2) Fit ridge for stable pilot coefficients.
    3) Run weighted coordinate-descent L1 (adaptive lasso style) for sparsity.
    4) Debias selected features with ridge refit.
    5) Optionally add one hinge correction if validation RMSE improves enough.
    """

    def __init__(
        self,
        ridge_alpha=0.1,
        adaptive_l1=0.03,
        adaptive_gamma=1.0,
        max_cd_iter=120,
        tol=1e-6,
        max_active_features=10,
        max_hinge_terms=1,
        hinge_quantiles=(0.25, 0.5, 0.75),
        hinge_alpha=0.08,
        min_rel_improve=0.008,
        random_state=42,
    ):
        self.ridge_alpha = ridge_alpha
        self.adaptive_l1 = adaptive_l1
        self.adaptive_gamma = adaptive_gamma
        self.max_cd_iter = max_cd_iter
        self.tol = tol
        self.max_active_features = max_active_features
        self.max_hinge_terms = max_hinge_terms
        self.hinge_quantiles = hinge_quantiles
        self.hinge_alpha = hinge_alpha
        self.min_rel_improve = min_rel_improve
        self.random_state = random_state

    @staticmethod
    def _soft_threshold(z, lmbd):
        if z > lmbd:
            return z - lmbd
        if z < -lmbd:
            return z + lmbd
        return 0.0

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

    def _fit_weighted_lasso_cd(self, Z, y_centered, penalty_weights):
        n, p = Z.shape
        beta = np.zeros(p, dtype=float)
        r = y_centered.copy()
        col_norm2 = np.sum(Z * Z, axis=0) + 1e-12
        lmbd = float(self.adaptive_l1) * float(n)

        for _ in range(max(1, int(self.max_cd_iter))):
            max_delta = 0.0
            for j in range(p):
                zj = Z[:, j]
                old = beta[j]
                rho = float(np.dot(zj, r) + col_norm2[j] * old)
                new = self._soft_threshold(rho, lmbd * float(penalty_weights[j])) / col_norm2[j]
                if new != old:
                    r -= zj * (new - old)
                    beta[j] = new
                    d = abs(new - old)
                    if d > max_delta:
                        max_delta = d
            if max_delta < float(self.tol):
                break
        return beta

    @staticmethod
    def _build_hinge_column(X, basis):
        xj = X[:, basis["j"]]
        t = basis["t"]
        if basis["kind"] == "hinge_pos":
            return np.maximum(0.0, xj - t)
        if basis["kind"] == "hinge_neg":
            return np.maximum(0.0, t - xj)
        raise ValueError(f"unknown hinge kind: {basis['kind']}")

    def _fit_hinge_residual(self, X, residual, hinges):
        n = X.shape[0]
        if len(hinges) == 0:
            return 0.0, np.array([]), np.array([]), np.array([])
        H_raw = np.column_stack([self._build_hinge_column(X, h) for h in hinges]).astype(float)
        h_mean = H_raw.mean(axis=0)
        h_std = H_raw.std(axis=0)
        h_std = np.where(h_std < 1e-8, 1.0, h_std)
        H = (H_raw - h_mean) / h_std
        b0, w = self._ridge_fit(H, residual, self.hinge_alpha)
        return b0, w, h_mean, h_std

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        self.n_features_in_ = p

        med = np.median(X, axis=0)
        mad = np.median(np.abs(X - med), axis=0)
        scale = 1.4826 * mad
        scale = np.where(scale < 1e-8, np.std(X, axis=0), scale)
        scale = np.where(scale < 1e-8, 1.0, scale)
        Z = (X - med) / scale

        y_mean = float(np.mean(y))
        yc = y - y_mean

        _, ridge_w = self._ridge_fit(Z, yc, float(self.ridge_alpha))
        penalty_weights = 1.0 / (np.abs(ridge_w) + 1e-4) ** float(self.adaptive_gamma)
        penalty_weights /= np.median(penalty_weights)

        w_sparse = self._fit_weighted_lasso_cd(Z, yc, penalty_weights)

        active = np.where(np.abs(w_sparse) > 1e-10)[0]
        if len(active) == 0:
            active = np.array([int(np.argmax(np.abs(ridge_w)))], dtype=int)

        if len(active) > int(self.max_active_features):
            keep = np.argsort(-np.abs(w_sparse[active]))[: int(self.max_active_features)]
            active = np.sort(active[keep])

        Z_active = Z[:, active]
        b_debias, w_debias = self._ridge_fit(Z_active, yc, float(self.ridge_alpha))

        coef_raw = np.zeros(p, dtype=float)
        coef_raw[active] = w_debias / scale[active]
        intercept_raw = float(y_mean + b_debias - np.dot(coef_raw, med))

        y_linear = intercept_raw + X @ coef_raw
        residual = y - y_linear

        rng = np.random.RandomState(self.random_state)
        idx = np.arange(n)
        rng.shuffle(idx)
        n_tr = max(20, int(0.85 * n))
        tr = idx[:n_tr]
        va = idx[n_tr:] if n_tr < n else idx[: max(1, n // 6)]

        chosen = []
        base_rmse = float(np.sqrt(np.mean((y[va] - y_linear[va]) ** 2)))
        best_rmse = base_rmse
        if len(active) > 0 and int(self.max_hinge_terms) > 0:
            candidates = []
            for j in active:
                xj = X[:, int(j)]
                for q in self.hinge_quantiles:
                    t = float(np.quantile(xj, float(q)))
                    candidates.append({"kind": "hinge_pos", "j": int(j), "t": t, "name": f"max(0, x{int(j)} - {t:.4f})"})
                    candidates.append({"kind": "hinge_neg", "j": int(j), "t": t, "name": f"max(0, {t:.4f} - x{int(j)})"})

            for _ in range(int(self.max_hinge_terms)):
                best_cand = None
                best_cand_rmse = best_rmse
                for cand in candidates:
                    if cand in chosen:
                        continue
                    trial = chosen + [cand]
                    b0, w_h, h_mean, h_std = self._fit_hinge_residual(X[tr], residual[tr], trial)
                    H_va = np.column_stack([self._build_hinge_column(X[va], h) for h in trial]).astype(float)
                    H_va = (H_va - h_mean) / h_std
                    pred = y_linear[va] + b0 + H_va @ w_h
                    rmse = float(np.sqrt(np.mean((y[va] - pred) ** 2)))
                    if rmse < best_cand_rmse:
                        best_cand_rmse = rmse
                        best_cand = cand
                if best_cand is None or best_cand_rmse >= best_rmse * (1.0 - float(self.min_rel_improve)):
                    break
                chosen.append(best_cand)
                best_rmse = best_cand_rmse

        b_h, w_h, h_mean, h_std = self._fit_hinge_residual(X, residual, chosen)
        hinge_raw = []
        for i in range(len(chosen)):
            c = float(w_h[i] / h_std[i])
            hinge_raw.append(c)
            b_h -= c * float(h_mean[i])

        abs_effect = np.abs(coef_raw) * np.std(X, axis=0)
        eff_max = max(float(np.max(abs_effect)), 1e-12)
        meaningful = np.where(abs_effect > 0.03 * eff_max)[0]
        negligible = np.where(abs_effect <= 0.03 * eff_max)[0]

        self.intercept_ = float(intercept_raw + b_h)
        self.coef_ = coef_raw
        self.hinge_terms_ = chosen
        self.hinge_coefs_ = np.asarray(hinge_raw, dtype=float)
        self.active_features_ = np.asarray(active, dtype=int)
        self.meaningful_features_ = np.asarray(meaningful, dtype=int)
        self.negligible_features_ = np.asarray(negligible, dtype=int)
        self.val_rmse_linear_ = float(base_rmse)
        self.val_rmse_final_ = float(best_rmse)
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
        shown = [j for j in order if abs(float(self.coef_[j])) > 1e-12]
        eq = f"y = {float(self.intercept_):.6f}"
        for j in shown:
            eq += f" {float(self.coef_[j]):+.6f}*x{int(j)}"
        for c, h in zip(self.hinge_coefs_, self.hinge_terms_):
            eq += f" {float(c):+.6f}*{h['name']}"

        lines = [
            "Adaptive Elastic Hinge Map Regressor",
            "Exact prediction equation in raw features:",
            eq,
            "Active linear features (sorted by |coefficient|):",
        ]
        for j in shown:
            lines.append(f"  x{int(j)}: coef={float(self.coef_[j]):+.6f}")

        if len(self.hinge_terms_) > 0:
            lines.append("Piecewise corrections:")
            for c, h in zip(self.hinge_coefs_, self.hinge_terms_):
                lines.append(f"  {float(c):+.6f} * {h['name']}")
        else:
            lines.append("Piecewise corrections: (none)")

        meaningful = ", ".join(f"x{int(j)}" for j in self.meaningful_features_) or "(none)"
        negligible = ", ".join(f"x{int(j)}" for j in self.negligible_features_) or "(none)"
        lines.append(f"Meaningful features: {meaningful}")
        lines.append(f"Negligible features: {negligible}")
        ops = 1 + 2 * len(shown) + 3 * len(self.hinge_terms_)
        lines.append(f"Approx operations to evaluate: {ops}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
AdaptiveElasticHingeMapRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "AdaptiveElasticHingeMapV1"
model_description = "Adaptive weighted-L1 sparse linear map with ridge debiasing and one validation-gated hinge residual correction in an explicit raw-feature equation"
model_defs = [(model_shorthand_name, AdaptiveElasticHingeMapRegressor())]

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
