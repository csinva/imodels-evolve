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


class QuantizedAdaptiveRidgeRegressor(BaseEstimator, RegressorMixin):
    """CV-selected ridge with coefficient pruning, OLS refit, and quantized symbolic coefficients."""

    def __init__(
        self,
        alpha_grid=(0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0),
        cv_folds=3,
        max_nonzero=20,
        coef_rel_thresh=0.03,
        quantization_step=0.05,
        tiny_coef=1e-10,
        random_state=0,
    ):
        self.alpha_grid = alpha_grid
        self.cv_folds = cv_folds
        self.max_nonzero = max_nonzero
        self.coef_rel_thresh = coef_rel_thresh
        self.quantization_step = quantization_step
        self.tiny_coef = tiny_coef
        self.random_state = random_state

    @staticmethod
    def _fit_ridge_raw(X, y, alpha):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        x_mean = np.mean(X, axis=0)
        x_std = np.std(X, axis=0)
        x_std[x_std < 1e-12] = 1.0

        y_mean = float(np.mean(y))
        y_centered = y - y_mean
        Xs = (X - x_mean) / x_std

        p = Xs.shape[1]
        gram = Xs.T @ Xs + float(alpha) * np.eye(p)
        rhs = Xs.T @ y_centered
        try:
            coef_std = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            coef_std = np.linalg.lstsq(gram, rhs, rcond=None)[0]

        coef_raw = coef_std / x_std
        intercept = y_mean - float(x_mean @ coef_raw)
        return coef_raw, intercept

    def _cv_alpha(self, X, y):
        n = X.shape[0]
        k = max(2, min(int(self.cv_folds), n))
        rng = np.random.RandomState(self.random_state)
        order = rng.permutation(n)
        folds = np.array_split(order, k)

        best_alpha = None
        best_mse = float("inf")
        for alpha in self.alpha_grid:
            mse_vals = []
            for i in range(k):
                val_idx = folds[i]
                if val_idx.size == 0:
                    continue
                train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]

                coef, intercept = self._fit_ridge_raw(X_train, y_train, alpha)
                pred = intercept + X_val @ coef
                mse_vals.append(float(np.mean((y_val - pred) ** 2)))

            if not mse_vals:
                continue
            mse = float(np.mean(mse_vals))
            if mse < best_mse:
                best_mse = mse
                best_alpha = float(alpha)

        if best_alpha is None:
            best_alpha = float(self.alpha_grid[0])
        return best_alpha

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        self.selected_alpha_ = self._cv_alpha(X, y)
        raw_coef, _ = self._fit_ridge_raw(X, y, self.selected_alpha_)

        abs_coef = np.abs(raw_coef)
        if n_features == 0:
            raise ValueError("No features provided")

        top_k = max(1, min(int(self.max_nonzero), n_features))
        order = np.argsort(abs_coef)[::-1]
        keep_top = set(int(j) for j in order[:top_k])

        scale = float(np.max(abs_coef)) if np.max(abs_coef) > 0 else 1.0
        keep_thresh = set(int(j) for j in np.where(abs_coef >= float(self.coef_rel_thresh) * scale)[0])
        keep_idx = sorted(keep_top.intersection(keep_thresh))

        if not keep_idx:
            keep_idx = [int(order[0])]

        Xk = X[:, keep_idx]
        design = np.column_stack([np.ones(n_samples, dtype=float), Xk])
        beta = np.linalg.lstsq(design, y, rcond=None)[0]
        intercept = float(beta[0])
        coef_keep = np.asarray(beta[1:], dtype=float)

        step = float(self.quantization_step)
        if step > 0:
            coef_keep = np.round(coef_keep / step) * step

        if np.all(np.abs(coef_keep) <= max(self.tiny_coef, 0.5 * max(step, self.tiny_coef))):
            best_local = int(np.argmax(np.abs(beta[1:]))) if len(beta) > 1 else 0
            coef_keep = np.zeros_like(coef_keep)
            if coef_keep.size > 0:
                coef_keep[best_local] = beta[1:][best_local]

        coef_keep[np.abs(coef_keep) < self.tiny_coef] = 0.0
        intercept = float(np.mean(y) - np.mean(Xk @ coef_keep))

        coef_full = np.zeros(n_features, dtype=float)
        for local_idx, feat_idx in enumerate(keep_idx):
            coef_full[int(feat_idx)] = float(coef_keep[local_idx])

        importances = np.abs(coef_full)
        mx = float(np.max(importances)) if np.max(importances) > 0 else 0.0
        self.feature_importances_ = importances / mx if mx > 0 else importances

        self.intercept_ = intercept
        self.coef_ = coef_full
        self.kept_features_ = np.array(keep_idx, dtype=int)
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "coef_"])
        X = np.asarray(X, dtype=float)
        return self.intercept_ + X @ self.coef_

    def __str__(self):
        check_is_fitted(self, ["intercept_", "coef_", "feature_importances_", "kept_features_", "selected_alpha_"])

        active = [int(j) for j in np.where(np.abs(self.coef_) > self.tiny_coef)[0]]
        ranked = sorted(active, key=lambda j: abs(float(self.coef_[j])), reverse=True)

        eq_terms = [f"{self.intercept_:+.4f}"]
        for j in ranked:
            eq_terms.append(f"{float(self.coef_[j]):+.4f}*x{j}")

        coef_lines = [f"x{j}: coef={float(self.coef_[j]):+.4f}, importance={float(self.feature_importances_[j]):.3f}" for j in ranked]
        inactive = [f"x{j}" for j in range(self.n_features_in_) if j not in set(active)]

        lines = [
            "Quantized Adaptive Ridge Regressor",
            f"Chosen ridge alpha by CV: {self.selected_alpha_:.4g}",
            "Prediction equation (raw features):",
            "  y = " + " ".join(eq_terms),
            "",
            "Active coefficients (largest first):",
            *("  " + line for line in (coef_lines if coef_lines else ["none"])),
            "Inactive (zeroed) features: " + (", ".join(inactive) if inactive else "none"),
        ]
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys

_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])

# Update the model shorthand name and description below to reflect the class above and any changes you make.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
QuantizedAdaptiveRidgeRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "QuantizedAdaptiveRidge_v2"
model_description = "From-scratch CV ridge with singular-safe solve, post-fit pruning, OLS refit, and quantized raw-feature coefficients for simulation-friendly equations"
model_defs = [(model_shorthand_name, QuantizedAdaptiveRidgeRegressor())]


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
