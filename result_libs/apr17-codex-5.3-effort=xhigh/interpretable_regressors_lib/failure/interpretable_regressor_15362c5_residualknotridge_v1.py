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


class ResidualizedKnotRegressor(BaseEstimator, RegressorMixin):
    """Ridge backbone + sparse residual hinge corrections."""

    def __init__(
        self,
        ridge_lambda=1e-3,
        max_hinges=6,
        hinge_search_features=12,
        max_linear_terms_small=12,
        small_feature_cutoff=25,
        min_rel_gain=5e-4,
        negligible_feature_eps=5e-3,
    ):
        self.ridge_lambda = ridge_lambda
        self.max_hinges = max_hinges
        self.hinge_search_features = hinge_search_features
        self.max_linear_terms_small = max_linear_terms_small
        self.small_feature_cutoff = small_feature_cutoff
        self.min_rel_gain = min_rel_gain
        self.negligible_feature_eps = negligible_feature_eps

    def _safe_abs_corr(self, a, b):
        a_std = float(np.std(a))
        b_std = float(np.std(b))
        if a_std < 1e-12 or b_std < 1e-12:
            return 0.0
        c = np.corrcoef(a, b)[0, 1]
        if not np.isfinite(c):
            return 0.0
        return float(abs(c))

    def _ridge_solve(self, X, y):
        if X.shape[1] == 0:
            return np.zeros(0, dtype=float)
        p = X.shape[1]
        gram = X.T @ X + self.ridge_lambda * np.eye(p)
        rhs = X.T @ y
        return np.linalg.solve(gram, rhs)

    def _hinge_values(self, z_col, knot, direction):
        if direction == "pos":
            return np.maximum(0.0, z_col - knot)
        return np.maximum(0.0, knot - z_col)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        self.x_mean_ = np.mean(X, axis=0)
        self.x_scale_ = np.std(X, axis=0)
        self.x_scale_[self.x_scale_ < 1e-12] = 1.0
        Z = (X - self.x_mean_) / self.x_scale_

        self.y_mean_ = float(np.mean(y))
        y_centered = y - self.y_mean_

        beta_dense = self._ridge_solve(Z, y_centered)
        base_pred = Z @ beta_dense
        base_residual = y_centered - base_pred
        baseline_sse = float(np.dot(base_residual, base_residual))

        if n_features <= self.small_feature_cutoff:
            abs_beta = np.abs(beta_dense)
            keep_linear_count = min(self.max_linear_terms_small, n_features)
            keep_linear_idx = np.argsort(abs_beta)[::-1][:keep_linear_count]
            keep_linear_idx = np.sort(keep_linear_idx.astype(int))
        else:
            keep_linear_idx = np.arange(n_features, dtype=int)

        selected_hinges = []
        hinge_cols = np.zeros((n_samples, 0), dtype=float)
        used_terms = set()
        current_pred = base_pred.copy()
        current_sse = float(np.dot(y_centered - current_pred, y_centered - current_pred))
        min_abs_gain = self.min_rel_gain * (baseline_sse + 1e-12)

        for _ in range(self.max_hinges):
            residual = y_centered - current_pred
            corr_scores = np.array([self._safe_abs_corr(Z[:, j], residual) for j in range(n_features)])
            ranked = np.argsort(corr_scores)[::-1][: min(self.hinge_search_features, n_features)]

            best_term = None
            best_col = None
            best_score = -1.0
            for j in ranked:
                z_col = Z[:, int(j)]
                knots = np.unique(np.round(np.quantile(z_col, [0.2, 0.4, 0.6, 0.8]), 6))
                for knot in knots:
                    for direction in ("pos", "neg"):
                        term_key = (int(j), float(knot), direction)
                        if term_key in used_terms:
                            continue
                        col = self._hinge_values(z_col, float(knot), direction)
                        score = self._safe_abs_corr(col, residual)
                        if score > best_score:
                            best_score = score
                            best_term = term_key
                            best_col = col

            if best_term is None:
                break

            trial_hinge_cols = (
                np.column_stack([hinge_cols, best_col]) if hinge_cols.size else best_col.reshape(-1, 1)
            )
            design = np.column_stack([Z[:, keep_linear_idx], trial_hinge_cols])
            coef = self._ridge_solve(design, y_centered)
            trial_pred = design @ coef
            trial_sse = float(np.dot(y_centered - trial_pred, y_centered - trial_pred))
            gain = current_sse - trial_sse
            if gain < min_abs_gain:
                break

            selected_hinges.append(best_term)
            hinge_cols = trial_hinge_cols
            used_terms.add(best_term)
            current_pred = trial_pred
            current_sse = trial_sse

        design = (
            np.column_stack([Z[:, keep_linear_idx], hinge_cols])
            if hinge_cols.size
            else Z[:, keep_linear_idx]
        )
        coef = self._ridge_solve(design, y_centered)
        n_linear = len(keep_linear_idx)
        self.linear_idx_ = keep_linear_idx
        self.linear_coef_ = coef[:n_linear]
        self.hinge_terms_ = selected_hinges
        self.hinge_coef_ = coef[n_linear:]
        self.intercept_ = self.y_mean_

        feature_contrib = np.zeros(n_features, dtype=float)
        if n_linear:
            linear_contrib = np.mean(np.abs(Z[:, keep_linear_idx] * self.linear_coef_.reshape(1, -1)), axis=0)
            feature_contrib[keep_linear_idx] += linear_contrib
        for c, (j, knot, direction) in zip(self.hinge_coef_, self.hinge_terms_):
            col = self._hinge_values(Z[:, j], knot, direction)
            feature_contrib[j] += float(np.mean(np.abs(c * col)))
        self.feature_importances_ = feature_contrib
        return self

    def predict(self, X):
        check_is_fitted(self, ["x_mean_", "x_scale_", "linear_idx_", "linear_coef_", "intercept_"])
        X = np.asarray(X, dtype=float)
        Z = (X - self.x_mean_) / self.x_scale_

        pred = np.full(X.shape[0], self.intercept_, dtype=float)
        if len(self.linear_idx_):
            pred += Z[:, self.linear_idx_] @ self.linear_coef_
        for c, (j, knot, direction) in zip(self.hinge_coef_, self.hinge_terms_):
            pred += c * self._hinge_values(Z[:, j], knot, direction)
        return pred

    def _raw_parameters(self):
        raw_linear = []
        intercept_raw = float(self.intercept_)
        for j, c in zip(self.linear_idx_, self.linear_coef_):
            slope = float(c / self.x_scale_[j])
            intercept_raw -= slope * float(self.x_mean_[j])
            raw_linear.append((int(j), slope))

        raw_hinges = []
        for c, (j, knot, direction) in zip(self.hinge_coef_, self.hinge_terms_):
            threshold_raw = float(self.x_mean_[j] + self.x_scale_[j] * knot)
            hinge_weight = float(c / self.x_scale_[j])
            raw_hinges.append((int(j), threshold_raw, direction, hinge_weight))
        return intercept_raw, raw_linear, raw_hinges

    def __str__(self):
        check_is_fitted(self, ["linear_idx_", "linear_coef_", "hinge_terms_", "hinge_coef_"])
        intercept_raw, raw_linear, raw_hinges = self._raw_parameters()

        pieces = [f"{intercept_raw:+.5f}"]
        for j, slope in raw_linear:
            pieces.append(f"{slope:+.5f}*x{j}")
        for j, threshold, direction, w in raw_hinges:
            if direction == "pos":
                pieces.append(f"{w:+.5f}*max(0, x{j} - {threshold:.5f})")
            else:
                pieces.append(f"{w:+.5f}*max(0, {threshold:.5f} - x{j})")

        order = np.argsort(self.feature_importances_)[::-1]
        ranked = [f"x{j}:{self.feature_importances_[j]:.4f}" for j in order[: min(10, self.n_features_in_)]]
        negligible = [
            f"x{j}"
            for j in range(self.n_features_in_)
            if self.feature_importances_[j] <= self.negligible_feature_eps
        ]

        lines = [
            "Residualized Knot Regressor (ridge backbone + sparse hinge corrections)",
            f"Linear terms: {len(raw_linear)}  |  Hinge terms: {len(raw_hinges)}",
            "Prediction formula:",
            "  y = " + " ".join(pieces),
            "",
            "Most influential features (mean absolute contribution):",
            "  " + ", ".join(ranked),
            "Features with negligible effect:",
            "  " + (", ".join(negligible) if negligible else "none"),
            "",
            "Hinge glossary: max(0, xj - t) is zero below t and linear above t.",
        ]
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
ResidualizedKnotRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "ResidualKnotRidge_v1"
model_description = "Ridge linear backbone with greedily selected residual hinge corrections and compact equation-oriented string output"
model_defs = [(model_shorthand_name, ResidualizedKnotRegressor())]


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
