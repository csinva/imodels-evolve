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


class SparseResidualHybridRegressor(BaseEstimator, RegressorMixin):
    """Full-feature ridge plus one optional residual nonlinear correction term."""

    def __init__(
        self,
        ridge_lambda=0.15,
        max_candidate_features=12,
        candidate_quantiles=(0.15, 0.3, 0.5, 0.7, 0.85),
        min_rel_gain=0.01,
        negligible_coef_eps=0.02,
    ):
        self.ridge_lambda = ridge_lambda
        self.max_candidate_features = max_candidate_features
        self.candidate_quantiles = candidate_quantiles
        self.min_rel_gain = min_rel_gain
        self.negligible_coef_eps = negligible_coef_eps

    def _safe_abs_corr(self, a, b):
        sa = float(np.std(a))
        sb = float(np.std(b))
        if sa < 1e-12 or sb < 1e-12:
            return 0.0
        c = np.corrcoef(a, b)[0, 1]
        if not np.isfinite(c):
            return 0.0
        return float(abs(c))

    def _solve_ridge(self, Z, y):
        z_mean = np.mean(Z, axis=0)
        z_std = np.std(Z, axis=0)
        z_std[z_std < 1e-12] = 1.0
        Zs = (Z - z_mean) / z_std

        y_mean = float(np.mean(y))
        y_c = y - y_mean

        gram = Zs.T @ Zs + self.ridge_lambda * np.eye(Zs.shape[1], dtype=float)
        coef_std = np.linalg.solve(gram, Zs.T @ y_c)
        coef = coef_std / z_std
        intercept = float(y_mean - np.dot(coef, z_mean))
        return intercept, coef

    def _basis_values(self, X, term):
        feat = term["feature"]
        knot = term["knot"]
        x = X[:, feat]
        if term["kind"] == "hinge_pos":
            return np.maximum(0.0, x - knot)
        if term["kind"] == "hinge_neg":
            return np.maximum(0.0, knot - x)
        if term["kind"] == "abs":
            return np.abs(x - knot)
        raise ValueError(f"Unknown basis kind {term['kind']}")

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        self.base_intercept_, self.base_linear_coef_ = self._solve_ridge(X, y)
        base_pred = self.base_intercept_ + X @ self.base_linear_coef_
        residual = y - base_pred

        feat_corr = np.array([self._safe_abs_corr(X[:, j], residual) for j in range(n_features)], dtype=float)
        ranked_feats = np.argsort(feat_corr)[::-1]
        candidate_feats = ranked_feats[: min(self.max_candidate_features, n_features)]

        base_sse = float(np.dot(residual, residual))
        best_term = None
        best_sse = base_sse

        for feat in candidate_feats:
            xj = X[:, int(feat)]
            knots = np.unique(np.round(np.quantile(xj, self.candidate_quantiles), 8))
            for knot in knots:
                for kind in ("hinge_pos", "hinge_neg", "abs"):
                    term = {"kind": kind, "feature": int(feat), "knot": float(knot)}
                    z = self._basis_values(X, term)
                    if float(np.std(z)) < 1e-10:
                        continue

                    Z_try = np.column_stack([X, z])
                    intercept_try, coef_try = self._solve_ridge(Z_try, y)
                    pred_try = intercept_try + Z_try @ coef_try
                    sse = float(np.dot(y - pred_try, y - pred_try))
                    if sse < best_sse:
                        best_sse = sse
                        best_term = term

        rel_gain = (base_sse - best_sse) / (base_sse + 1e-12)
        if best_term is not None and rel_gain >= self.min_rel_gain:
            z = self._basis_values(X, best_term)
            Z_full = np.column_stack([X, z])
            self.intercept_, coef_full = self._solve_ridge(Z_full, y)
            self.linear_coef_ = coef_full[:n_features]
            self.extra_coef_ = float(coef_full[n_features])
            self.extra_term_ = best_term
        else:
            self.intercept_ = self.base_intercept_
            self.linear_coef_ = self.base_linear_coef_
            self.extra_coef_ = 0.0
            self.extra_term_ = None

        abs_linear = np.abs(self.linear_coef_)
        self.feature_importances_ = abs_linear.copy()
        if self.extra_term_ is not None:
            self.feature_importances_[self.extra_term_["feature"]] += abs(self.extra_coef_)

        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "linear_coef_", "extra_term_", "extra_coef_"])
        X = np.asarray(X, dtype=float)
        pred = self.intercept_ + X @ self.linear_coef_
        if self.extra_term_ is not None:
            pred += self.extra_coef_ * self._basis_values(X, self.extra_term_)
        return pred

    def _extra_term_str(self):
        if self.extra_term_ is None:
            return None
        coef = self.extra_coef_
        feat = self.extra_term_["feature"]
        knot = self.extra_term_["knot"]
        kind = self.extra_term_["kind"]
        if kind == "hinge_pos":
            return f"{coef:+.5f}*max(0, x{feat}-{knot:.5f})"
        if kind == "hinge_neg":
            return f"{coef:+.5f}*max(0, {knot:.5f}-x{feat})"
        return f"{coef:+.5f}*abs(x{feat}-{knot:.5f})"

    def __str__(self):
        check_is_fitted(self, ["intercept_", "linear_coef_", "feature_importances_", "extra_term_", "extra_coef_"])

        order = np.argsort(np.abs(self.linear_coef_))[::-1]
        active = [int(j) for j in order if abs(self.linear_coef_[j]) > self.negligible_coef_eps]
        negligible = [f"x{j}" for j in range(self.n_features_in_) if abs(self.linear_coef_[j]) <= self.negligible_coef_eps]

        equation_terms = [f"{self.intercept_:+.5f}"]
        for j in active:
            equation_terms.append(f"{self.linear_coef_[j]:+.5f}*x{j}")
        extra_str = self._extra_term_str()
        if extra_str is not None:
            equation_terms.append(extra_str)

        top_importance = np.argsort(self.feature_importances_)[::-1]
        top_text = ", ".join(
            f"x{j}:{self.feature_importances_[j]:.4f}" for j in top_importance[: min(10, self.n_features_in_)]
        )

        lines = [
            "Sparse Residual Hybrid Regressor",
            "Model form: y = intercept + linear terms + optional one-term nonlinear correction",
            "Compact equation:",
            "  y = " + " ".join(equation_terms),
            "",
            "Active linear coefficients (sorted by absolute magnitude):",
        ]
        if active:
            for j in active:
                lines.append(f"  x{j}: {self.linear_coef_[j]:+.5f}")
        else:
            lines.append("  (none)")

        lines.extend([
            "",
            "Nonlinear correction term:",
            f"  {extra_str}" if extra_str is not None else "  (none)",
            "",
            "Most influential features (linear coefficient magnitude + correction attribution):",
            "  " + top_text,
            "Features with negligible linear effect:",
            "  " + (", ".join(negligible) if negligible else "none"),
            "",
            "Simulation recipe: plug feature values into the compact equation directly.",
        ])
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseResidualHybridRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseResidualHybrid_v1"
model_description = "Full-feature ridge baseline with one greedily selected residual nonlinear basis term (hinge or absolute knot) and compact equation output"
model_defs = [(model_shorthand_name, SparseResidualHybridRegressor())]


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
