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


class SparseAdditiveKnotRegressor(BaseEstimator, RegressorMixin):
    """Sparse additive model with one data-driven hinge knot per selected feature."""

    def __init__(
        self,
        max_active_features=4,
        candidate_quantiles=(0.2, 0.35, 0.5, 0.65, 0.8),
        ridge_lambda=1e-2,
        min_rel_gain=0.008,
        negligible_feature_eps=5e-3,
    ):
        self.max_active_features = max_active_features
        self.candidate_quantiles = candidate_quantiles
        self.ridge_lambda = ridge_lambda
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

    def _fit_ridge(self, Phi, y):
        if Phi.shape[1] == 0:
            return float(np.mean(y)), np.zeros(0, dtype=float)
        x_mean = np.mean(Phi, axis=0)
        x_std = np.std(Phi, axis=0)
        x_std[x_std < 1e-12] = 1.0
        Z = (Phi - x_mean) / x_std
        y_mean = float(np.mean(y))
        y_c = y - y_mean
        gram = Z.T @ Z + self.ridge_lambda * np.eye(Z.shape[1], dtype=float)
        coef_std = np.linalg.solve(gram, Z.T @ y_c)
        coef = coef_std / x_std
        intercept = float(y_mean - np.dot(coef, x_mean))
        return intercept, coef

    def _feature_block(self, x_col, knot):
        return np.column_stack([x_col, np.maximum(0.0, x_col - knot)])

    def _build_design(self, X, feature_specs):
        if len(feature_specs) == 0:
            return np.zeros((X.shape[0], 0), dtype=float)
        blocks = [self._feature_block(X[:, spec["feature"]], spec["knot"]) for spec in feature_specs]
        return np.hstack(blocks)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        _, n_features = X.shape
        self.n_features_in_ = n_features

        corr = np.array([self._safe_abs_corr(X[:, j], y) for j in range(n_features)], dtype=float)
        ranked = list(np.argsort(corr)[::-1].astype(int))

        specs = []
        design = self._build_design(X, specs)
        intercept, coef = self._fit_ridge(design, y)
        pred = intercept + (design @ coef if design.shape[1] else 0.0)
        current_sse = float(np.dot(y - pred, y - pred))

        candidate_features = ranked[: min(n_features, 10)]
        for _ in range(min(self.max_active_features, len(candidate_features))):
            best = None
            min_gain_abs = self.min_rel_gain * (current_sse + 1e-12)

            for feat in candidate_features:
                if any(spec["feature"] == feat for spec in specs):
                    continue
                xj = X[:, feat]
                knots = np.unique(np.round(np.quantile(xj, self.candidate_quantiles), 8))
                for knot in knots:
                    cand_specs = specs + [{"feature": int(feat), "knot": float(knot)}]
                    cand_design = self._build_design(X, cand_specs)
                    cand_intercept, cand_coef = self._fit_ridge(cand_design, y)
                    cand_pred = cand_intercept + cand_design @ cand_coef
                    cand_sse = float(np.dot(y - cand_pred, y - cand_pred))
                    gain = current_sse - cand_sse
                    if gain > min_gain_abs and (best is None or cand_sse < best["sse"]):
                        best = {
                            "specs": cand_specs,
                            "design": cand_design,
                            "intercept": cand_intercept,
                            "coef": cand_coef,
                            "pred": cand_pred,
                            "sse": cand_sse,
                        }

            if best is None:
                break
            specs = best["specs"]
            design = best["design"]
            intercept = best["intercept"]
            coef = best["coef"]
            pred = best["pred"]
            current_sse = best["sse"]

        self.feature_specs_ = specs
        self.intercept_ = float(intercept)
        self.term_coef_ = coef.astype(float, copy=True)

        self.feature_importances_ = np.zeros(n_features, dtype=float)
        if len(specs) > 0:
            for idx, spec in enumerate(specs):
                linear_col = design[:, 2 * idx]
                hinge_col = design[:, 2 * idx + 1]
                w_lin = float(self.term_coef_[2 * idx])
                w_hinge = float(self.term_coef_[2 * idx + 1])
                contrib = np.mean(np.abs(w_lin * linear_col + w_hinge * hinge_col))
                self.feature_importances_[spec["feature"]] = float(contrib)
        return self

    def predict(self, X):
        check_is_fitted(self, ["feature_specs_", "intercept_", "term_coef_"])
        X = np.asarray(X, dtype=float)
        design = self._build_design(X, self.feature_specs_)
        return self.intercept_ + (design @ self.term_coef_ if design.shape[1] else 0.0)

    def __str__(self):
        check_is_fitted(self, ["feature_specs_", "intercept_", "term_coef_", "feature_importances_"])
        active = [spec["feature"] for spec in self.feature_specs_]
        lines = [
            "Sparse Additive Knot Regressor",
            "Model form: y = intercept + sum_j [a_j*xj + b_j*max(0, xj - knot_j)]",
            f"Intercept: {self.intercept_:+.5f}",
            f"Active features ({len(active)}): " + (", ".join(f"x{j}" for j in active) if active else "none"),
            "",
            "Per-feature effects:",
        ]

        equation_terms = [f"{self.intercept_:+.5f}"]
        for idx, spec in enumerate(self.feature_specs_):
            feat = spec["feature"]
            knot = spec["knot"]
            a = float(self.term_coef_[2 * idx])
            b = float(self.term_coef_[2 * idx + 1])
            left_slope = a
            right_slope = a + b
            equation_terms.append(f"{a:+.5f}*x{feat}")
            equation_terms.append(f"{b:+.5f}*max(0, x{feat} - {knot:.5f})")
            lines.append(
                f"  x{feat}: knot={knot:.5f}, left_slope={left_slope:+.5f}, right_slope={right_slope:+.5f}"
            )

        if len(self.feature_specs_) == 0:
            lines.append("  (no active features; constant model)")

        order = np.argsort(self.feature_importances_)[::-1]
        top = [f"x{j}:{self.feature_importances_[j]:.4f}" for j in order[: min(10, self.n_features_in_)]]
        negligible = [f"x{j}" for j in range(self.n_features_in_) if self.feature_importances_[j] <= self.negligible_feature_eps]

        lines.extend([
            "",
            "Full equation:",
            "  y = " + " ".join(equation_terms),
            "",
            "Most influential features (mean absolute contribution):",
            "  " + ", ".join(top),
            "Features with negligible effect:",
            "  " + (", ".join(negligible) if negligible else "none"),
            "",
            "Simulation recipe: evaluate each active feature's piecewise-linear contribution, then add intercept.",
        ])
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseAdditiveKnotRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseAdditiveKnot_v1"
model_description = "Greedy sparse additive one-knot-per-feature model with explicit piecewise slopes and direct simulation equation"
model_defs = [(model_shorthand_name, SparseAdditiveKnotRegressor())]


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
