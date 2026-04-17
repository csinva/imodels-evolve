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
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class SparseHingeAdditiveRegressor(BaseEstimator, RegressorMixin):
    """Sparse additive model with linear and hinge basis terms.

    The model standardizes each feature, creates a compact dictionary of
    linear and one-sided hinge terms, uses L1 selection, then refits a small
    linear model on selected terms.
    """

    def __init__(self, max_features=10, max_terms=14, n_thresholds=2):
        self.max_features = max_features
        self.max_terms = max_terms
        self.n_thresholds = n_thresholds

    def _standardize(self, X):
        return (X - self.feature_means_) / self.feature_scales_

    def _corr_abs(self, x, y):
        x_std = x.std()
        y_std = y.std()
        if x_std < 1e-12 or y_std < 1e-12:
            return 0.0
        return abs(np.corrcoef(x, y)[0, 1])

    def _build_dictionary(self, Z, y):
        n_features = Z.shape[1]
        corrs = np.array([self._corr_abs(Z[:, j], y) for j in range(n_features)])
        ranked_features = np.argsort(corrs)[::-1]
        selected_features = ranked_features[: min(self.max_features, n_features)]

        candidate_cols = []
        candidate_terms = []

        for j in selected_features:
            z = Z[:, j]
            candidate_cols.append(z)
            candidate_terms.append(("linear", int(j), None))

            if self.n_thresholds > 0:
                quantiles = np.linspace(0.25, 0.75, self.n_thresholds)
                for q in quantiles:
                    t = float(np.quantile(z, q))
                    candidate_cols.append(np.maximum(0.0, z - t))
                    candidate_terms.append(("pos_hinge", int(j), t))
                    candidate_cols.append(np.maximum(0.0, t - z))
                    candidate_terms.append(("neg_hinge", int(j), t))

        X_dict = np.column_stack(candidate_cols)

        # Keep only the strongest dictionary terms by univariate relevance.
        term_scores = np.array([self._corr_abs(X_dict[:, k], y) for k in range(X_dict.shape[1])])
        max_candidates = max(4, min(X_dict.shape[1], self.max_terms * 4))
        keep = np.argsort(term_scores)[::-1][:max_candidates]

        return X_dict[:, keep], [candidate_terms[k] for k in keep]

    def _term_values(self, Z, term):
        kind, j, t = term
        z = Z[:, j]
        if kind == "linear":
            return z
        if kind == "pos_hinge":
            return np.maximum(0.0, z - t)
        return np.maximum(0.0, t - z)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        self.feature_means_ = X.mean(axis=0)
        self.feature_scales_ = X.std(axis=0)
        self.feature_scales_[self.feature_scales_ < 1e-12] = 1.0

        Z = self._standardize(X)
        X_dict, dict_terms = self._build_dictionary(Z, y)

        selector = LassoCV(cv=3, n_alphas=30, random_state=42, max_iter=8000)
        selector.fit(X_dict, y)
        raw_coef = selector.coef_

        active = np.where(np.abs(raw_coef) > 1e-7)[0]
        if len(active) == 0:
            active = np.array([int(np.argmax(np.abs(raw_coef)))])

        if len(active) > self.max_terms:
            top = np.argsort(np.abs(raw_coef[active]))[::-1][: self.max_terms]
            active = active[top]

        X_sel = X_dict[:, active]
        terms_sel = [dict_terms[k] for k in active]

        refit = LinearRegression()
        refit.fit(X_sel, y)

        self.intercept_ = float(refit.intercept_)
        self.coef_ = refit.coef_.astype(float)
        self.terms_ = terms_sel
        self.alpha_ = float(selector.alpha_)

        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "coef_", "terms_"])
        X = np.asarray(X, dtype=float)
        Z = self._standardize(X)
        X_sel = np.column_stack([self._term_values(Z, t) for t in self.terms_])
        return self.intercept_ + X_sel @ self.coef_

    def __str__(self):
        check_is_fitted(self, ["intercept_", "coef_", "terms_"])

        lines = [
            "Sparse Hinge Additive Regressor:",
            "  Standardization: z_j = (x_j - mean_j) / scale_j",
            f"  L1 selection alpha: {self.alpha_:.5f}",
            f"  Prediction: y = {self.intercept_:.4f} + sum_k c_k * term_k",
            "",
            "Selected terms:",
        ]

        for k, (coef, term) in enumerate(zip(self.coef_, self.terms_), 1):
            kind, j, t = term
            mean_j = self.feature_means_[j]
            scale_j = self.feature_scales_[j]
            z_expr = f"(x{j} - {mean_j:.3f})/{scale_j:.3f}"
            if kind == "linear":
                term_expr = z_expr
            elif kind == "pos_hinge":
                term_expr = f"max(0, {z_expr} - {t:.3f})"
            else:
                term_expr = f"max(0, {t:.3f} - {z_expr})"
            lines.append(f"  c{k}={coef:+.4f}  term{k}={term_expr}")

        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseHingeAdditiveRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseHingeAdd_v1"
model_description = "Sparse additive regressor with standardized linear and hinge basis terms selected by L1, then OLS refit"
model_defs = [(model_shorthand_name, SparseHingeAdditiveRegressor())]


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
