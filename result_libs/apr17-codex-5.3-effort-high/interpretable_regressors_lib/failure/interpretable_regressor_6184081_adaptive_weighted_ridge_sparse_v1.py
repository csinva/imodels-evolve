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


class AdaptiveWeightedRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Adaptive weighted-ridge linear regressor with validation-gated sparsification.

    Steps:
    1) Standardize features.
    2) Fit pilot ridge, then convert pilot magnitudes into feature-wise penalties.
    3) Refit weighted ridge with GCV alpha selection.
    4) Optionally prune tiny coefficients if validation MSE does not worsen.
    """

    def __init__(
        self,
        alpha_grid=(0.0005, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0),
        pilot_alpha=0.1,
        adaptive_power=1.0,
        weight_floor=0.2,
        holdout_frac=0.2,
        sparsity_levels=(0.0, 0.01, 0.02, 0.04, 0.06),
        max_active_features=18,
        random_state=0,
    ):
        self.alpha_grid = alpha_grid
        self.pilot_alpha = pilot_alpha
        self.adaptive_power = adaptive_power
        self.weight_floor = weight_floor
        self.holdout_frac = holdout_frac
        self.sparsity_levels = sparsity_levels
        self.max_active_features = max_active_features
        self.random_state = random_state

    @staticmethod
    def _ridge_with_diag_penalty(X, y, penalty_diag, alpha):
        n, p = X.shape
        D = np.hstack([np.ones((n, 1), dtype=float), X])
        reg = np.zeros(p + 1, dtype=float)
        reg[1:] = float(alpha) * np.asarray(penalty_diag, dtype=float)
        A = D.T @ D + np.diag(reg)
        b = D.T @ y
        try:
            beta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(A) @ b
        return float(beta[0]), np.asarray(beta[1:], dtype=float)

    def _select_alpha_weighted_gcv(self, Z, yc, penalty_diag):
        n, p = Z.shape
        xtx = Z.T @ Z
        W = np.diag(np.asarray(penalty_diag, dtype=float))
        best = (float("inf"), float(self.alpha_grid[0]), np.zeros(p, dtype=float))
        for alpha in self.alpha_grid:
            _, w = self._ridge_with_diag_penalty(Z, yc, penalty_diag, alpha)
            resid = yc - Z @ w
            rss = float(np.dot(resid, resid))
            S = xtx + float(alpha) * W
            try:
                Sinv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                Sinv = np.linalg.pinv(S)
            df = float(np.trace(Sinv @ xtx))
            denom = (n - df) ** 2
            if denom < 1e-10:
                continue
            gcv = rss / denom
            if gcv < best[0]:
                best = (gcv, float(alpha), w)
        return best[1], best[2]

    @staticmethod
    def _mse(y_true, y_pred):
        err = y_true - y_pred
        return float(np.mean(err * err))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        self.n_features_in_ = p

        x_mean = X.mean(axis=0)
        x_std = X.std(axis=0)
        x_std = np.where(x_std < 1e-8, 1.0, x_std)
        Z = (X - x_mean) / x_std
        y_mean = float(y.mean())
        yc = y - y_mean

        # Stable holdout split for pruning decisions.
        rng = np.random.RandomState(int(self.random_state))
        n_val = max(20, int(float(self.holdout_frac) * n))
        if n - n_val < 20:
            n_val = max(1, n // 5)
        perm = rng.permutation(n)
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]
        if len(tr_idx) < 5:
            tr_idx = perm
            val_idx = perm[: max(1, min(10, n // 4))]

        Z_tr, Z_val = Z[tr_idx], Z[val_idx]
        yc_tr, yc_val = yc[tr_idx], yc[val_idx]

        _, pilot_w = self._ridge_with_diag_penalty(
            Z_tr, yc_tr, penalty_diag=np.ones(p, dtype=float), alpha=float(self.pilot_alpha)
        )
        abs_pilot = np.abs(pilot_w)
        scale = float(np.median(abs_pilot[abs_pilot > 1e-12])) if np.any(abs_pilot > 1e-12) else 1.0
        penalty_diag = (abs_pilot / max(scale, 1e-12) + float(self.weight_floor)) ** (-float(self.adaptive_power))
        penalty_diag = penalty_diag / max(float(np.mean(penalty_diag)), 1e-12)

        alpha, w = self._select_alpha_weighted_gcv(Z_tr, yc_tr, penalty_diag)
        intercept_z = float(np.mean(yc_tr - Z_tr @ w))

        # Validation-gated hard-threshold pruning.
        base_pred_val = intercept_z + Z_val @ w
        best_mse = self._mse(yc_val, base_pred_val)
        best_coef = np.asarray(w, dtype=float)

        abs_w = np.abs(w)
        w_max = max(float(abs_w.max()), 1e-12)
        for rel_thr in self.sparsity_levels:
            thr = float(rel_thr) * w_max
            mask = abs_w >= thr
            if np.sum(mask) == 0:
                mask[np.argmax(abs_w)] = True
            if np.sum(mask) > int(self.max_active_features):
                top = np.argsort(-abs_w)[: int(self.max_active_features)]
                m2 = np.zeros_like(mask, dtype=bool)
                m2[top] = True
                mask = m2

            w_masked = np.zeros_like(w)
            w_masked[mask] = w[mask]
            pred_val = intercept_z + Z_val @ w_masked
            mse_val = self._mse(yc_val, pred_val)
            if mse_val <= best_mse * 1.002:
                best_mse = mse_val
                best_coef = w_masked

        coef_raw = best_coef / x_std
        intercept_raw = float(y_mean + intercept_z - np.dot(coef_raw, x_mean))

        abs_raw = np.abs(coef_raw)
        active = np.where(abs_raw > 1e-12)[0]
        if len(active) == 0:
            active = np.array([int(np.argmax(abs_w))], dtype=int)
            coef_raw[active[0]] = w[active[0]] / x_std[active[0]]
            intercept_raw = float(y_mean + intercept_z - np.dot(coef_raw, x_mean))
        inactive = np.array([j for j in range(p) if j not in set(active)], dtype=int)

        self.intercept_ = intercept_raw
        self.coef_ = coef_raw
        self.alpha_ = float(alpha)
        self.penalty_diag_ = np.asarray(penalty_diag, dtype=float)
        self.active_features_ = np.asarray(active, dtype=int)
        self.inactive_features_ = inactive
        self.validation_mse_ = float(best_mse)
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "coef_"])
        X = np.asarray(X, dtype=float)
        return self.intercept_ + X @ self.coef_

    def __str__(self):
        check_is_fitted(self, ["intercept_", "coef_", "active_features_"])
        order = np.argsort(-np.abs(self.coef_))
        shown = [int(j) for j in order if abs(float(self.coef_[j])) > 1e-12]

        eq = f"y = {float(self.intercept_):.6f}"
        for j in shown:
            eq += f" {float(self.coef_[j]):+.6f}*x{j}"

        lines = [
            "Adaptive Weighted-Ridge Sparse Linear Regressor",
            f"Chosen alpha (weighted GCV): {self.alpha_:.6g}",
            "Prediction equation:",
            eq,
            "Active features (sorted by |coefficient|):",
        ]
        for j in shown:
            lines.append(f"  x{j}: coef={float(self.coef_[j]):+.6f}")

        if len(self.inactive_features_) > 0:
            inactive_txt = ", ".join(f"x{int(j)}" for j in self.inactive_features_)
            lines.append(f"Inactive/negligible features: {inactive_txt}")

        lines.append(f"Validation MSE (centered target): {self.validation_mse_:.6f}")
        lines.append(f"Approx arithmetic ops to evaluate: {2 * len(shown)}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
AdaptiveWeightedRidgeRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "AdaptiveWeightedRidgeSparseV1"
model_description = "Adaptive weighted-ridge linear regressor with pilot-coefficient-informed penalties and validation-gated hard-threshold sparsification for a compact explicit equation"
model_defs = [(model_shorthand_name, AdaptiveWeightedRidgeRegressor())]

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
