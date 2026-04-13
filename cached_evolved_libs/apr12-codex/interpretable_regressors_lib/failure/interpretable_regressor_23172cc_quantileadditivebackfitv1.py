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


class QuantileAdditiveBackfitRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse additive model learned by coordinate backfitting over per-feature
    quantile bins. Each selected feature contributes an explicit piecewise-
    constant function that is easy to simulate from text.
    """

    def __init__(
        self,
        max_active_features=6,
        n_bins=6,
        n_backfit_rounds=5,
        screening_mix=(1.0, 0.5, 0.35),
        min_bin_count=12,
        smoothing=0.2,
        contribution_tol=1e-3,
        random_state=42,
    ):
        self.max_active_features = max_active_features
        self.n_bins = n_bins
        self.n_backfit_rounds = n_backfit_rounds
        self.screening_mix = screening_mix
        self.min_bin_count = min_bin_count
        self.smoothing = smoothing
        self.contribution_tol = contribution_tol
        self.random_state = random_state

    def _impute(self, X):
        X = np.asarray(X, dtype=float).copy()
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(self.feature_medians_, np.where(bad)[1])
        return X

    @staticmethod
    def _corr_abs(a, b):
        ac = a - float(np.mean(a))
        bc = b - float(np.mean(b))
        denom = (float(np.std(ac)) + 1e-12) * (float(np.std(bc)) + 1e-12)
        return abs(float(np.mean(ac * bc)) / denom)

    def _make_bins(self, x):
        q = np.linspace(0.0, 1.0, int(self.n_bins) + 1)
        edges = np.unique(np.quantile(x, q))
        if edges.size < 3:
            lo = float(np.min(x))
            hi = float(np.max(x))
            if hi <= lo:
                hi = lo + 1.0
            edges = np.array([lo, (lo + hi) * 0.5, hi], dtype=float)
        return edges

    @staticmethod
    def _bucket_ids(x, edges):
        return np.searchsorted(edges[1:-1], x, side="right")

    def _fit_feature_effect(self, resid, ids, n_bins):
        vals = np.zeros(n_bins, dtype=float)
        counts = np.bincount(ids, minlength=n_bins).astype(float)
        sums = np.bincount(ids, weights=resid, minlength=n_bins)
        global_mean = float(np.mean(resid))

        for b in range(n_bins):
            if counts[b] >= int(self.min_bin_count):
                vals[b] = sums[b] / counts[b]
            else:
                vals[b] = global_mean

        if n_bins > 1 and float(self.smoothing) > 0.0:
            smooth_vals = vals.copy()
            for b in range(n_bins):
                left = vals[b - 1] if b > 0 else vals[b]
                right = vals[b + 1] if b + 1 < n_bins else vals[b]
                avg_nb = 0.5 * (left + right)
                smooth_vals[b] = (1.0 - float(self.smoothing)) * vals[b] + float(self.smoothing) * avg_nb
            vals = smooth_vals

        vals -= float(np.mean(vals[ids]))
        return vals

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]

        self.feature_medians_ = np.nanmedian(X, axis=0)
        X = self._impute(X)

        p = X.shape[1]
        w_lin, w_abs, w_sq = self.screening_mix
        screening = np.array([
            float(w_lin) * self._corr_abs(X[:, j], y)
            + float(w_abs) * self._corr_abs(np.abs(X[:, j]), y)
            + float(w_sq) * self._corr_abs(X[:, j] * X[:, j], y)
            for j in range(p)
        ])
        keep = min(int(self.max_active_features), p)
        self.active_features_ = np.argsort(screening)[::-1][:keep]

        self.intercept_ = float(np.mean(y))
        pred = np.full_like(y, self.intercept_, dtype=float)

        self.feature_edges_ = {}
        self.feature_values_ = {}
        self.feature_ids_train_ = {}
        for j in self.active_features_:
            j = int(j)
            edges = self._make_bins(X[:, j])
            ids = self._bucket_ids(X[:, j], edges)
            self.feature_edges_[j] = edges
            self.feature_ids_train_[j] = ids
            self.feature_values_[j] = np.zeros(len(edges) - 1, dtype=float)

        for _ in range(int(self.n_backfit_rounds)):
            for j in self.active_features_:
                j = int(j)
                ids = self.feature_ids_train_[j]
                old = self.feature_values_[j][ids]
                pred -= old
                resid = y - pred

                vals = self._fit_feature_effect(resid, ids, n_bins=len(self.feature_values_[j]))
                self.feature_values_[j] = vals
                pred += vals[ids]

        self.fitted_mse_ = float(np.mean((y - pred) ** 2))
        self.feature_effect_strength_ = np.zeros(p, dtype=float)
        for j in self.active_features_:
            j = int(j)
            ids = self.feature_ids_train_[j]
            self.feature_effect_strength_[j] = float(np.mean(np.abs(self.feature_values_[j][ids])))
        self.selected_feature_order_ = np.argsort(self.feature_effect_strength_)[::-1]
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "active_features_", "feature_edges_", "feature_values_"])
        X = self._impute(X)
        yhat = np.full(X.shape[0], self.intercept_, dtype=float)
        for j in self.active_features_:
            j = int(j)
            edges = self.feature_edges_[j]
            ids = self._bucket_ids(X[:, j], edges)
            yhat += self.feature_values_[j][ids]
        return yhat

    def __str__(self):
        check_is_fitted(self, ["intercept_", "active_features_", "feature_edges_", "feature_values_"])
        lines = [
            "QuantileAdditiveBackfitRegressor",
            "Prediction rule:",
            f"  y = {self.intercept_:+.6f} + sum(feature_piecewise_effects)",
            "",
            "To simulate: for each listed feature, pick the matching bin and add its effect.",
        ]

        shown = 0
        for j in self.selected_feature_order_:
            j = int(j)
            if j not in self.feature_values_:
                continue
            strength = self.feature_effect_strength_[j]
            if strength < float(self.contribution_tol):
                continue
            edges = self.feature_edges_[j]
            vals = self.feature_values_[j]

            lines.append("")
            lines.append(f"Feature x{j} (avg |effect|={strength:.4f}):")
            for b in range(len(vals)):
                lo = edges[b]
                hi = edges[b + 1]
                if b == 0:
                    interval = f"x{j} <= {hi:+.4f}"
                elif b == len(vals) - 1:
                    interval = f"x{j} > {lo:+.4f}"
                else:
                    interval = f"{lo:+.4f} < x{j} <= {hi:+.4f}"
                lines.append(f"  if {interval}: add {vals[b]:+.6f}")
            shown += 1

        if shown == 0:
            lines.append("")
            lines.append("All learned feature effects are negligible; model predicts near intercept.")

        active = set(int(j) for j in self.active_features_)
        negligible = [f"x{j}" for j in range(self.n_features_in_) if j not in active]
        if negligible:
            lines.append("")
            lines.append("Not selected (treated as negligible): " + ", ".join(negligible))
        lines.append(f"Training MSE: {self.fitted_mse_:.6f}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
QuantileAdditiveBackfitRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "QuantileAdditiveBackfitV1"
model_description = "Sparse additive regressor learned by quantile-bin backfitting over top screened features with explicit per-bin contribution tables"
model_defs = [(model_shorthand_name, QuantileAdditiveBackfitRegressor())]

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
