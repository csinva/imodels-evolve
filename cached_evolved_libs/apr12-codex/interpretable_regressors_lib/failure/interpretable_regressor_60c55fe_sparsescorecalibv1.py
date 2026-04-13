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


class SparseScoreCalibratedRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse single-index regressor with optional one-dimensional hinge calibration.

    Model form:
      score = b0 + sum_j w_j * z_j        (z_j are standardized selected features)
      y     = c0 + c1*score + c2*relu(score-k1) + c3*relu(score-k2)   (optional)

    If calibration does not improve validation error, the model falls back to
    the plain sparse linear score equation.
    """

    def __init__(
        self,
        alpha_grid=(1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0),
        alpha_cal=1e-3,
        val_frac=0.2,
        min_val_samples=120,
        max_active_features=8,
        min_gain_calibration=0.01,
        coef_prune_abs=5e-5,
        coef_decimals=5,
        random_state=42,
    ):
        self.alpha_grid = alpha_grid
        self.alpha_cal = alpha_cal
        self.val_frac = val_frac
        self.min_val_samples = min_val_samples
        self.max_active_features = max_active_features
        self.min_gain_calibration = min_gain_calibration
        self.coef_prune_abs = coef_prune_abs
        self.coef_decimals = coef_decimals
        self.random_state = random_state

    @staticmethod
    def _ridge_closed_form(Z, y, alpha):
        p = Z.shape[1]
        reg = float(alpha) * np.eye(p)
        reg[0, 0] = 0.0
        A = Z.T @ Z + reg
        b = Z.T @ y
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(A) @ b

    def _split_idx(self, n):
        if n < int(self.min_val_samples) + 20:
            idx = np.arange(n)
            return idx, idx
        rng = np.random.RandomState(self.random_state)
        perm = rng.permutation(n)
        n_val = max(int(round(float(self.val_frac) * n)), int(self.min_val_samples))
        n_val = min(n_val, n // 2)
        return perm[n_val:], perm[:n_val]

    def _impute(self, X):
        X = np.asarray(X, dtype=float).copy()
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(self.feature_medians_, np.where(bad)[1])
        return X

    @staticmethod
    def _standardize_with_stats(X, mean, scale):
        return (X - mean) / scale

    def _fit_ridge_alpha(self, Xtr, ytr, Xval, yval):
        Ztr = np.column_stack([np.ones(Xtr.shape[0]), Xtr])
        Zval = np.column_stack([np.ones(Xval.shape[0]), Xval])

        best_alpha = float(self.alpha_grid[0])
        best_beta = self._ridge_closed_form(Ztr, ytr, best_alpha)
        best_mse = float(np.mean((yval - Zval @ best_beta) ** 2))
        for alpha in self.alpha_grid[1:]:
            beta = self._ridge_closed_form(Ztr, ytr, float(alpha))
            mse = float(np.mean((yval - Zval @ beta) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_alpha = float(alpha)
                best_beta = beta
        return best_beta, best_alpha, best_mse

    @staticmethod
    def _design_calibration(score, k1, k2):
        return np.column_stack([
            np.ones(score.shape[0]),
            score,
            np.maximum(0.0, score - k1),
            np.maximum(0.0, score - k2),
        ])

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]

        self.feature_medians_ = np.nanmedian(X, axis=0)
        X = self._impute(X)

        tr_idx, val_idx = self._split_idx(X.shape[0])
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xval, yval = X[val_idx], y[val_idx]

        x_mean = np.mean(Xtr, axis=0)
        x_std = np.std(Xtr, axis=0)
        x_std[x_std < 1e-12] = 1.0
        self.x_mean_ = x_mean
        self.x_std_ = x_std

        Xtr_s = self._standardize_with_stats(Xtr, x_mean, x_std)
        Xval_s = self._standardize_with_stats(Xval, x_mean, x_std)
        Xall_s = self._standardize_with_stats(X, x_mean, x_std)

        beta_dense, _, _ = self._fit_ridge_alpha(Xtr_s, ytr, Xval_s, yval)
        dense_coef = beta_dense[1:]

        k = min(int(self.max_active_features), self.n_features_in_)
        order = np.argsort(np.abs(dense_coef))[::-1]
        selected = [int(j) for j in order[:k]]
        if not selected:
            selected = [0]

        Xtr_sel = Xtr_s[:, selected]
        Xval_sel = Xval_s[:, selected]
        Xall_sel = Xall_s[:, selected]

        beta_sel, self.alpha_, val_mse_linear = self._fit_ridge_alpha(Xtr_sel, ytr, Xval_sel, yval)
        intercept_linear = float(beta_sel[0])
        coef_linear = np.asarray(beta_sel[1:], dtype=float)

        score_tr = intercept_linear + Xtr_sel @ coef_linear
        score_val = intercept_linear + Xval_sel @ coef_linear
        score_all = intercept_linear + Xall_sel @ coef_linear

        k1 = float(np.quantile(score_tr, 0.35))
        k2 = float(np.quantile(score_tr, 0.7))
        if k2 <= k1:
            k2 = k1 + max(1e-6, 1e-3 * float(np.std(score_tr) + 1e-12))

        Ztr_cal = self._design_calibration(score_tr, k1, k2)
        Zval_cal = self._design_calibration(score_val, k1, k2)
        gamma = self._ridge_closed_form(Ztr_cal, ytr, float(self.alpha_cal))
        val_pred_cal = Zval_cal @ gamma
        val_mse_cal = float(np.mean((yval - val_pred_cal) ** 2))

        use_cal = val_mse_cal < val_mse_linear * (1.0 - float(self.min_gain_calibration))
        self.use_calibration_ = bool(use_cal)

        q = int(self.coef_decimals)
        self.selected_features_ = selected
        self.intercept_score_ = float(np.round(intercept_linear, q))
        coef_linear[np.abs(coef_linear) < float(self.coef_prune_abs)] = 0.0
        self.coef_score_ = np.round(coef_linear, q)

        if self.use_calibration_:
            Zall_cal = self._design_calibration(score_all, k1, k2)
            gamma_all = self._ridge_closed_form(Zall_cal, y, float(self.alpha_cal))
            gamma_all[np.abs(gamma_all) < float(self.coef_prune_abs)] = 0.0
            self.calib_coef_ = np.round(gamma_all, q)
            self.knot1_ = float(np.round(k1, q))
            self.knot2_ = float(np.round(k2, q))
            self.intercept_ = float(self.calib_coef_[0])
        else:
            self.calib_coef_ = np.array([0.0, 1.0, 0.0, 0.0], dtype=float)
            self.knot1_ = float(np.round(k1, q))
            self.knot2_ = float(np.round(k2, q))
            self.intercept_ = self.intercept_score_

        self.feature_importance_ = np.zeros(self.n_features_in_, dtype=float)
        for loc, j in enumerate(self.selected_features_):
            self.feature_importance_[j] = abs(float(self.coef_score_[loc]))
        self.feature_rank_ = np.argsort(self.feature_importance_)[::-1]
        self.fitted_mse_ = float(np.mean((y - self.predict(X)) ** 2))
        return self

    def _score(self, X):
        X = self._impute(X)
        Xs = self._standardize_with_stats(X, self.x_mean_, self.x_std_)
        Xsel = Xs[:, self.selected_features_]
        return float(self.intercept_score_) + Xsel @ self.coef_score_

    def predict(self, X):
        check_is_fitted(self, ["selected_features_", "coef_score_", "use_calibration_"])
        score = self._score(X)
        if not self.use_calibration_:
            return score
        c0, c1, c2, c3 = [float(v) for v in self.calib_coef_]
        y = (
            c0
            + c1 * score
            + c2 * np.maximum(0.0, score - float(self.knot1_))
            + c3 * np.maximum(0.0, score - float(self.knot2_))
        )
        return y

    def __str__(self):
        check_is_fitted(self, ["selected_features_", "coef_score_", "use_calibration_"])
        lines = [
            "SparseScoreCalibratedRegressor",
            f"Selected ridge alpha (score): {self.alpha_:.5g}",
            "",
            "Step 1: standardized sparse score equation",
            f"  score = {self.intercept_score_:+.5f}",
        ]
        for loc, j in enumerate(self.selected_features_):
            c = float(self.coef_score_[loc])
            if abs(c) > 0.0:
                lines.append(f"    + ({c:+.5f}) * z{int(j)}")
        lines.append("  where z_j = (x_j - mean_j) / std_j")

        if self.use_calibration_:
            c0, c1, c2, c3 = [float(v) for v in self.calib_coef_]
            lines.extend([
                "",
                "Step 2: one-dimensional calibration of score",
                f"  y = {c0:+.5f} + ({c1:+.5f})*score"
                f" + ({c2:+.5f})*relu(score - {self.knot1_:.5f})"
                f" + ({c3:+.5f})*relu(score - {self.knot2_:.5f})",
            ])
        else:
            lines.extend([
                "",
                "Step 2 skipped: calibration disabled (validation gain too small).",
                "  Final prediction: y = score",
            ])

        lines.append("")
        lines.append("Feature importance (absolute score coefficient):")
        for j in self.feature_rank_[: min(10, self.n_features_in_)]:
            lines.append(f"  x{int(j)}: {self.feature_importance_[int(j)]:.5f}")
        near_zero = [f"x{j}" for j, v in enumerate(self.feature_importance_) if v < 1e-6]
        if near_zero:
            lines.append("Features with near-zero effect: " + ", ".join(near_zero))
        lines.append(f"Training MSE: {self.fitted_mse_:.6f}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseScoreCalibratedRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseScoreCalibV1"
model_description = "Sparse standardized ridge score over top features with optional two-knot hinge calibration on the 1D score"
model_defs = [(model_shorthand_name, SparseScoreCalibratedRegressor())]

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------

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
