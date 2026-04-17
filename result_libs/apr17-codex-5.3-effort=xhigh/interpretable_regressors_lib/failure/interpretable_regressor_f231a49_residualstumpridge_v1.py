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
from sklearn.linear_model import RidgeCV
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class ResidualStumpRidgeRegressor(BaseEstimator, RegressorMixin):
    """Ridge backbone plus one residual stump correction learned from data."""

    def __init__(
        self,
        alpha_grid=(1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
        cv_folds=5,
        min_gain=1e-8,
        split_quantiles=(0.2, 0.35, 0.5, 0.65, 0.8),
        random_state=0,
    ):
        self.alpha_grid = alpha_grid
        self.cv_folds = cv_folds
        self.min_gain = min_gain
        self.split_quantiles = split_quantiles
        self.random_state = random_state

    @staticmethod
    def _safe_standardize(X):
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        sigma = np.where(sigma < 1e-12, 1.0, sigma)
        return mu, sigma

    def _fit_best_stump(self, X, residual):
        n, p = X.shape
        base_mse = float(np.mean(residual ** 2))
        best = {
            "feature": -1,
            "threshold": 0.0,
            "left": 0.0,
            "right": 0.0,
            "gain": 0.0,
        }
        if n < 8:
            return best

        for j in range(p):
            xj = X[:, j]
            if np.allclose(xj, xj[0]):
                continue
            qs = np.quantile(xj, self.split_quantiles)
            # Keep a tiny candidate set for speed and robustness.
            thresholds = np.unique(np.concatenate((qs, [np.median(xj)])))
            for thr in thresholds:
                left_mask = xj <= thr
                n_left = int(np.sum(left_mask))
                n_right = n - n_left
                if n_left < 4 or n_right < 4:
                    continue
                left_val = float(np.mean(residual[left_mask]))
                right_val = float(np.mean(residual[~left_mask]))
                corrected = np.where(left_mask, left_val, right_val)
                mse = float(np.mean((residual - corrected) ** 2))
                gain = base_mse - mse
                if gain > best["gain"]:
                    best = {
                        "feature": int(j),
                        "threshold": float(thr),
                        "left": left_val,
                        "right": right_val,
                        "gain": float(gain),
                    }
        if best["gain"] < float(self.min_gain):
            best = {"feature": -1, "threshold": 0.0, "left": 0.0, "right": 0.0, "gain": 0.0}
        return best

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        if p == 0:
            raise ValueError("No features provided")

        self.n_features_in_ = p
        self.x_mean_, self.x_scale_ = self._safe_standardize(X)
        Xs = (X - self.x_mean_) / self.x_scale_

        self.backbone_ = RidgeCV(alphas=np.array(self.alpha_grid, dtype=float), cv=int(self.cv_folds))
        self.backbone_.fit(Xs, y)

        coef_scaled = np.asarray(self.backbone_.coef_, dtype=float)
        self.coef_ = coef_scaled / self.x_scale_
        self.intercept_ = float(self.backbone_.intercept_ - np.dot(self.x_mean_ / self.x_scale_, coef_scaled))
        self.alpha_ = float(self.backbone_.alpha_)

        pred_linear = self.intercept_ + X @ self.coef_
        residual = y - pred_linear
        self.stump_ = self._fit_best_stump(X, residual)

        abs_coef = np.abs(self.coef_)
        max_abs = float(np.max(abs_coef)) if np.max(abs_coef) > 0 else 0.0
        self.feature_importances_ = abs_coef / max_abs if max_abs > 0 else abs_coef

        coef_tol = max(1e-10, 0.05 * (max_abs if max_abs > 0 else 1.0))
        self.active_features_ = np.flatnonzero(abs_coef >= coef_tol).astype(int)

        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "coef_", "stump_", "active_features_"])
        X = np.asarray(X, dtype=float)
        pred = self.intercept_ + X @ self.coef_
        if self.stump_["feature"] >= 0:
            j = self.stump_["feature"]
            pred = pred + np.where(X[:, j] <= self.stump_["threshold"], self.stump_["left"], self.stump_["right"])
        return pred

    def __str__(self):
        check_is_fitted(
            self,
            ["intercept_", "coef_", "alpha_", "active_features_", "feature_importances_", "stump_"],
        )

        lines = [
            "Residual Stump Ridge Regressor",
            "Prediction rule:",
        ]
        lines.append(
            "1) linear_base = intercept + sum_j(coef_j * xj)"
        )
        if self.stump_["feature"] >= 0:
            j = int(self.stump_["feature"])
            lines.append(
                f"2) residual_correction = {self.stump_['left']:+.6f} if x{j} <= {self.stump_['threshold']:+.6f} else {self.stump_['right']:+.6f}"
            )
        else:
            lines.append("2) residual_correction = 0.000000")
        lines.append("3) y = linear_base + residual_correction")

        lines.append("")
        lines.append(f"ridge_alpha = {self.alpha_:.6g}")
        lines.append(f"active_linear_features = {', '.join(f'x{int(j)}' for j in self.active_features_) if self.active_features_.size else 'none'}")
        lines.append("Full linear coefficients (use these for exact simulation):")

        for j in range(self.coef_.size):
            c = float(self.coef_[j])
            lines.append(f"x{int(j)}: coef={c:+.6f}")

        lines.append("")
        lines.append("This is a compact model: one linear equation plus one if-then correction.")
        lines.append("Simulation recipe: compute linear_base, then add residual_correction from the threshold rule.")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys

_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])

# Update the model shorthand name and description below to reflect the class above and any changes you make.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
ResidualStumpRidgeRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "ResidualStumpRidge_v1"
model_description = "Custom model with RidgeCV linear backbone plus one data-driven residual stump correction, exported as exact equation + one threshold rule"
model_defs = [(model_shorthand_name, ResidualStumpRidgeRegressor())]


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
