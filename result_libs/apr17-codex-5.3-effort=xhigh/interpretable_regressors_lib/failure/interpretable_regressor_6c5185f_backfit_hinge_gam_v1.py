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


class BackfitHingeAdditiveRegressor(BaseEstimator, RegressorMixin):
    """Sparse additive piecewise-linear regressor fit by residual backfitting."""

    def __init__(
        self,
        ridge_alpha=0.2,
        max_active_features=8,
        n_rounds=20,
        learning_rate=0.35,
        knot_quantiles=(0.25, 0.5, 0.75),
        min_round_gain=5e-4,
        negligible_coef_eps=1e-4,
    ):
        self.ridge_alpha = ridge_alpha
        self.max_active_features = max_active_features
        self.n_rounds = n_rounds
        self.learning_rate = learning_rate
        self.knot_quantiles = knot_quantiles
        self.min_round_gain = min_round_gain
        self.negligible_coef_eps = negligible_coef_eps

    @staticmethod
    def _corr(a, b):
        a0 = a - np.mean(a)
        b0 = b - np.mean(b)
        den = (np.std(a0) + 1e-12) * (np.std(b0) + 1e-12)
        return float(np.mean(a0 * b0) / den)

    @staticmethod
    def _ridge_fit(X, y, alpha, penalize_mask=None):
        p = X.shape[1]
        reg = float(alpha) * np.eye(p)
        if penalize_mask is not None:
            pm = np.asarray(penalize_mask, dtype=float)
            reg = reg * pm[:, None]
        return np.linalg.solve(X.T @ X + reg, X.T @ y)

    def _feature_basis(self, x_col, knots):
        cols = [x_col]
        for k in knots:
            cols.append(np.maximum(0.0, x_col - k))
        return np.column_stack(cols)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        self.feature_knots_ = [
            np.array([float(np.quantile(X[:, j], q)) for q in self.knot_quantiles], dtype=float)
            for j in range(n_features)
        ]
        self.feature_basis_cols_ = [1 + len(self.feature_knots_[j]) for j in range(n_features)]

        y_mean = float(np.mean(y))
        self.intercept_ = y_mean
        pred = np.full(n_samples, self.intercept_, dtype=float)

        self.feature_coef_ = [np.zeros(self.feature_basis_cols_[j], dtype=float) for j in range(n_features)]
        self.feature_active_count_ = np.zeros(n_features, dtype=int)
        self.round_gain_history_ = []

        # Correlation screening for candidate features.
        yc = y - y_mean
        corr = np.array([abs(self._corr(X[:, j], yc)) for j in range(n_features)], dtype=float)
        order = np.argsort(corr)[::-1]
        max_act = max(1, min(int(self.max_active_features), n_features))
        active = [int(j) for j in order[:max_act]]

        # Precompute per-feature basis matrices.
        bases = {
            j: self._feature_basis(X[:, j], self.feature_knots_[j])
            for j in active
        }

        for _ in range(max(0, int(self.n_rounds))):
            residual = y - pred
            best = None
            for j in active:
                B = bases[j]
                # Keep the local basis centered so global intercept remains interpretable.
                Bc = B - np.mean(B, axis=0, keepdims=True)
                coef = self._ridge_fit(Bc, residual, self.ridge_alpha)
                step = float(self.learning_rate) * (Bc @ coef)
                gain = float(np.mean(step ** 2))
                if (best is None) or (gain > best[0]):
                    best = (gain, j, coef, step)

            if best is None or best[0] < float(self.min_round_gain):
                break
            gain, j, coef, step = best
            self.round_gain_history_.append(gain)
            self.feature_coef_[j] += float(self.learning_rate) * coef
            self.feature_active_count_[j] += 1
            pred += step

        # Compress tiny coefficients and compute feature importances.
        imp = np.zeros(n_features, dtype=float)
        for j in range(n_features):
            c = self.feature_coef_[j]
            c[np.abs(c) < self.negligible_coef_eps] = 0.0
            self.feature_coef_[j] = c
            imp[j] = float(np.sum(np.abs(c)))
        if np.max(imp) > 0:
            imp = imp / np.max(imp)
        self.feature_importances_ = imp

        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "feature_coef_", "feature_knots_"])
        X = np.asarray(X, dtype=float)
        pred = np.full(X.shape[0], self.intercept_, dtype=float)
        for j in range(self.n_features_in_):
            coef = self.feature_coef_[j]
            if np.all(np.abs(coef) < self.negligible_coef_eps):
                continue
            B = self._feature_basis(X[:, j], self.feature_knots_[j])
            pred += B @ coef
        return pred

    def _feature_formula_str(self, j):
        coef = self.feature_coef_[j]
        parts = []
        if abs(float(coef[0])) >= self.negligible_coef_eps:
            parts.append(f"{float(coef[0]):+.4f}*x{j}")
        for k_idx, knot in enumerate(self.feature_knots_[j]):
            c = float(coef[1 + k_idx])
            if abs(c) >= self.negligible_coef_eps:
                parts.append(f"{c:+.4f}*max(0, x{j}-{float(knot):.4f})")
        return " ".join(parts)

    def __str__(self):
        check_is_fitted(self, ["intercept_", "feature_coef_", "feature_importances_", "feature_active_count_"])

        eq_parts = [f"{self.intercept_:+.4f}"]
        ranked = np.argsort(self.feature_importances_)[::-1]
        for j in ranked:
            ftxt = self._feature_formula_str(int(j))
            if ftxt:
                eq_parts.append(ftxt)

        ranking_txt = ", ".join(
            f"x{int(j)}:{self.feature_importances_[j]:.3f}" for j in ranked[: min(10, self.n_features_in_)]
        )

        lines = [
            "Backfit Hinge Additive Regressor",
            "Additive prediction rule: y = intercept + sum_j g_j(xj), each g_j is piecewise-linear.",
            "Exact prediction rule:",
            "  y = " + " ".join(eq_parts),
            "",
            "Top feature influence ranking: " + ranking_txt,
        ]

        lines.append("Per-feature components (non-negligible only):")
        for j in ranked:
            ftxt = self._feature_formula_str(int(j))
            if ftxt:
                lines.append(f"  g_{int(j)}(x{int(j)}) = {ftxt}")
        near_zero = [f"x{i}" for i in range(self.n_features_in_) if np.all(np.abs(self.feature_coef_[i]) < self.negligible_coef_eps)]
        lines.append("Inactive features: " + (", ".join(near_zero) if near_zero else "none"))

        return "\n".join(lines)
# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
BackfitHingeAdditiveRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "BackfitHingeGAM_v1"
model_description = "Sparse additive backfitting with per-feature piecewise-linear hinge bases and explicit simulation-ready equation"
model_defs = [(model_shorthand_name, BackfitHingeAdditiveRegressor())]


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
