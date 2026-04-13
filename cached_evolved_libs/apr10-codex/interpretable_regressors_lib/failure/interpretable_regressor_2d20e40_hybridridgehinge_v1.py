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


class HybridRidgeHingeRegressor(BaseEstimator, RegressorMixin):
    """
    Dense linear backbone fit via closed-form ridge (alpha chosen by GCV),
    plus up to a few explicit one-feature hinge correction terms.
    """

    def __init__(
        self,
        alpha_grid=None,
        max_hinges=2,
        hinge_screen=6,
        hinge_quantiles=(0.33, 0.67),
        hinge_min_gain=1e-4,
        hinge_penalty=0.001,
        coef_display_tol=0.02,
    ):
        self.alpha_grid = alpha_grid
        self.max_hinges = max_hinges
        self.hinge_screen = hinge_screen
        self.hinge_quantiles = hinge_quantiles
        self.hinge_min_gain = hinge_min_gain
        self.hinge_penalty = hinge_penalty
        self.coef_display_tol = coef_display_tol

    @staticmethod
    def _safe_scale(x):
        med = np.median(x, axis=0)
        mad = np.median(np.abs(x - med), axis=0)
        scale = 1.4826 * mad
        scale = np.where(scale < 1e-8, 1.0, scale)
        return med, scale

    @staticmethod
    def _safe_corr(x, y):
        xc = x - np.mean(x)
        yc = y - np.mean(y)
        denom = (np.linalg.norm(xc) + 1e-12) * (np.linalg.norm(yc) + 1e-12)
        return float(np.dot(xc, yc) / denom)

    @staticmethod
    def _ridge_fit_centered(Xc, yc, alpha):
        p = Xc.shape[1]
        A = Xc.T @ Xc + float(alpha) * np.eye(p)
        b = Xc.T @ yc
        return np.linalg.solve(A, b)

    def _choose_alpha_gcv(self, Xc, yc):
        n, p = Xc.shape
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        Uy = U.T @ yc

        if self.alpha_grid is None:
            alphas = np.logspace(-3, 3, 15)
        else:
            alphas = np.asarray(self.alpha_grid, dtype=float)

        best_alpha = float(alphas[0])
        best_gcv = np.inf

        s2 = s ** 2
        for a in alphas:
            filt = s2 / (s2 + a)
            yhat = U @ (filt * Uy)
            resid = yc - yhat
            rss = float(np.dot(resid, resid))
            df = float(np.sum(filt))
            denom = float((n - df) ** 2 + 1e-12)
            gcv = rss / denom
            if gcv < best_gcv:
                best_gcv = gcv
                best_alpha = float(a)

        return best_alpha

    @staticmethod
    def _hinge_col(x, threshold, side):
        if side == "right":
            return np.maximum(0.0, x - threshold)
        if side == "left":
            return np.maximum(0.0, threshold - x)
        raise ValueError(f"Unknown hinge side: {side}")

    def _build_hinge_candidates(self, X, residual):
        n, p = X.shape
        corr = np.array([abs(self._safe_corr(X[:, j], residual)) for j in range(p)], dtype=float)
        k = int(max(1, min(int(self.hinge_screen), p)))
        feat_idx = np.argsort(-corr)[:k]

        cands = []
        for j in feat_idx:
            xj = X[:, int(j)]
            for q in self.hinge_quantiles:
                thr = float(np.quantile(xj, float(q)))
                for side in ("right", "left"):
                    h = self._hinge_col(xj, thr, side)
                    if float(np.std(h)) < 1e-8:
                        continue
                    cands.append(
                        {
                            "feature": int(j),
                            "threshold": thr,
                            "side": side,
                            "col": h,
                        }
                    )
        return cands

    @staticmethod
    def _fit_full(X_lin, hinge_cols, y, alpha):
        n = X_lin.shape[0]
        if hinge_cols:
            H = np.column_stack(hinge_cols)
            Z = np.column_stack([X_lin, H])
        else:
            H = np.zeros((n, 0), dtype=float)
            Z = X_lin

        z_mean = np.mean(Z, axis=0)
        y_mean = float(np.mean(y))
        Zc = Z - z_mean
        yc = y - y_mean

        w = HybridRidgeHingeRegressor._ridge_fit_centered(Zc, yc, alpha)
        intercept = float(y_mean - np.dot(z_mean, w))

        p_lin = X_lin.shape[1]
        w_lin = w[:p_lin]
        w_hinge = w[p_lin:]
        pred = intercept + Z @ w
        return intercept, w_lin, w_hinge, pred

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape

        med, scale = self._safe_scale(X)
        Xs = (X - med) / scale

        x_mean = np.mean(Xs, axis=0)
        y_mean = float(np.mean(y))
        Xc = Xs - x_mean
        yc = y - y_mean

        alpha = self._choose_alpha_gcv(Xc, yc)
        beta_std = self._ridge_fit_centered(Xc, yc, alpha)
        intercept = float(y_mean - np.dot(x_mean, beta_std))

        pred = intercept + Xs @ beta_std
        mse_best = float(np.mean((y - pred) ** 2))

        hinge_specs = []
        hinge_cols = []
        used = set()

        max_h = int(max(0, self.max_hinges))
        for _ in range(max_h):
            residual = y - pred
            cands = self._build_hinge_candidates(X, residual)
            best = None

            for c in cands:
                key = (c["feature"], c["side"], round(c["threshold"], 6))
                if key in used:
                    continue

                trial_cols = hinge_cols + [c["col"]]
                _, _, _, pred_try = self._fit_full(Xs, trial_cols, y, alpha)
                mse = float(np.mean((y - pred_try) ** 2))
                obj = mse + float(self.hinge_penalty) * len(trial_cols)

                if (best is None) or (obj < best[0]):
                    best = (obj, mse, c, key, pred_try)

            if best is None:
                break

            gain = mse_best - best[1]
            if gain < float(self.hinge_min_gain):
                break

            used.add(best[3])
            hinge_specs.append(
                {
                    "feature": int(best[2]["feature"]),
                    "threshold": float(best[2]["threshold"]),
                    "side": best[2]["side"],
                }
            )
            hinge_cols.append(best[2]["col"])

            _, beta_std, _, pred = self._fit_full(Xs, hinge_cols, y, alpha)
            mse_best = float(np.mean((y - pred) ** 2))

        intercept, beta_std, hinge_w, _ = self._fit_full(Xs, hinge_cols, y, alpha)

        beta_raw = beta_std / scale
        intercept_raw = float(intercept - np.dot(beta_std, med / scale))

        self.alpha_ = float(alpha)
        self.median_ = med
        self.scale_ = scale
        self.intercept_ = intercept_raw
        self.coef_ = beta_raw
        self.hinge_specs_ = hinge_specs
        self.hinge_weights_ = np.asarray(hinge_w, dtype=float)

        self.n_features_in_ = int(p)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        X = np.asarray(X, dtype=float)

        pred = np.full(X.shape[0], float(self.intercept_), dtype=float)
        pred += X @ self.coef_

        for w, spec in zip(self.hinge_weights_, self.hinge_specs_):
            xj = X[:, int(spec["feature"])]
            h = self._hinge_col(xj, float(spec["threshold"]), spec["side"])
            pred += float(w) * h

        return pred

    def __str__(self):
        check_is_fitted(self, "is_fitted_")
        lines = ["Hybrid Ridge + Hinge Regressor:"]
        lines.append("  prediction = intercept + sum_j b_j*xj + sum_t g_t*hinge_t(x)")
        lines.append(f"  ridge alpha (GCV-chosen): {self.alpha_:.4f}")
        lines.append(f"  intercept: {self.intercept_:+.4f}")

        tol = float(max(0.0, self.coef_display_tol))
        active = [(j, float(c)) for j, c in enumerate(self.coef_) if abs(float(c)) >= tol]
        zeroed = [f"x{j}" for j, c in enumerate(self.coef_) if abs(float(c)) < tol]

        lines.append("  linear terms:")
        if active:
            active_sorted = sorted(active, key=lambda t: -abs(t[1]))
            for j, c in active_sorted:
                lines.append(f"    {c:+.4f} * x{j}")
        else:
            lines.append("    (no active linear terms)")

        if zeroed:
            lines.append(f"  near-zero linear terms (excluded for interpretation): {', '.join(zeroed)}")

        if len(self.hinge_specs_) == 0:
            lines.append("  hinge corrections: none")
        else:
            lines.append("  hinge corrections:")
            for w, spec in sorted(
                zip(self.hinge_weights_, self.hinge_specs_),
                key=lambda t: -abs(float(t[0])),
            ):
                j = int(spec["feature"])
                thr = float(spec["threshold"])
                side = spec["side"]
                if side == "right":
                    form = f"max(0, x{j} - {thr:.3f})"
                else:
                    form = f"max(0, {thr:.3f} - x{j})"
                lines.append(f"    {float(w):+.4f} * {form}")

        return "\n".join(lines)



# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
HybridRidgeHingeRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "HybridRidgeHinge_v1"
model_description = "Dense GCV-selected ridge backbone with up to two explicit one-feature hinge correction terms"
model_defs = [(model_shorthand_name, HybridRidgeHingeRegressor())]


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
