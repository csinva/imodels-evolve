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
from sklearn.linear_model import LassoCV
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class SparseAdditiveHingeRegressor(BaseEstimator, RegressorMixin):
    """
    Interpretable scikit-learn compatible regressor.

    Sparse additive regressor over human-readable basis terms:
      - linear terms: x_j
      - hinge terms: max(0, x_j - t), max(0, t - x_j)
    Thresholds are selected from feature quantiles and terms are pruned to keep
    the final equation compact and simulatable.

    Must implement: fit(X, y), predict(X), and __str__().
    """

    def __init__(
        self,
        threshold_quantiles=(0.25, 0.5, 0.75),
        max_feature_families=12,
        max_terms=8,
        cv=3,
        random_state=42,
    ):
        self.threshold_quantiles = threshold_quantiles
        self.max_feature_families = max_feature_families
        self.max_terms = max_terms
        self.cv = cv
        self.random_state = random_state

    def _feature_score(self, x_col, y):
        centered_y = y - np.mean(y)
        y_norm = np.linalg.norm(centered_y) + 1e-12

        scores = []
        x_centered = x_col - np.mean(x_col)
        scores.append(abs(np.dot(x_centered, centered_y)) / ((np.linalg.norm(x_centered) + 1e-12) * y_norm))

        for q in self.threshold_quantiles:
            t = float(np.quantile(x_col, q))
            h_pos = np.maximum(0.0, x_col - t)
            h_neg = np.maximum(0.0, t - x_col)
            h_pos -= np.mean(h_pos)
            h_neg -= np.mean(h_neg)
            scores.append(abs(np.dot(h_pos, centered_y)) / ((np.linalg.norm(h_pos) + 1e-12) * y_norm))
            scores.append(abs(np.dot(h_neg, centered_y)) / ((np.linalg.norm(h_neg) + 1e-12) * y_norm))

        return float(np.max(scores))

    def _build_basis(self, X, selected_features):
        cols = []
        terms = []

        for j in selected_features:
            xj = X[:, j]
            cols.append(xj)
            terms.append(("linear", int(j), None))

            for q in self.threshold_quantiles:
                t = float(np.quantile(xj, q))
                cols.append(np.maximum(0.0, xj - t))
                terms.append(("hinge_pos", int(j), t))
                cols.append(np.maximum(0.0, t - xj))
                terms.append(("hinge_neg", int(j), t))

        if not cols:
            return np.zeros((X.shape[0], 0), dtype=float), []
        return np.column_stack(cols), terms

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n_samples, n_features = X.shape

        family_scores = np.array([self._feature_score(X[:, j], y) for j in range(n_features)])
        k = min(int(self.max_feature_families), n_features)
        selected_features = np.argsort(-family_scores)[:k]
        self.selected_features_ = np.array(selected_features, dtype=int)

        Phi, terms = self._build_basis(X, self.selected_features_)
        self.intercept_ = float(np.mean(y))

        if Phi.shape[1] == 0:
            self.active_terms_ = []
            self.coef_ = np.zeros(0, dtype=float)
            self.is_fitted_ = True
            return self

        lasso = LassoCV(
            cv=self.cv,
            random_state=self.random_state,
            n_alphas=40,
            max_iter=5000,
        )
        lasso.fit(Phi, y)

        coef = lasso.coef_.copy()
        intercept = float(lasso.intercept_)
        nonzero = np.flatnonzero(np.abs(coef) > 1e-8)

        if nonzero.size == 0:
            self.active_terms_ = []
            self.coef_ = np.zeros(0, dtype=float)
            self.intercept_ = intercept
            self.is_fitted_ = True
            return self

        if nonzero.size > int(self.max_terms):
            keep_order = np.argsort(-np.abs(coef[nonzero]))[: int(self.max_terms)]
            nonzero = nonzero[keep_order]

        Phi_active = Phi[:, nonzero]
        beta, *_ = np.linalg.lstsq(
            np.column_stack([np.ones(n_samples), Phi_active]),
            y,
            rcond=None,
        )

        self.intercept_ = float(beta[0])
        self.coef_ = np.asarray(beta[1:], dtype=float)
        self.active_terms_ = [terms[int(i)] for i in nonzero]
        self.alpha_ = float(lasso.alpha_)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        X = np.asarray(X, dtype=float)
        pred = np.full(X.shape[0], self.intercept_, dtype=float)

        for c, (kind, j, t) in zip(self.coef_, self.active_terms_):
            xj = X[:, j]
            if kind == "linear":
                pred += c * xj
            elif kind == "hinge_pos":
                pred += c * np.maximum(0.0, xj - t)
            else:
                pred += c * np.maximum(0.0, t - xj)
        return pred

    def __str__(self):
        check_is_fitted(self, "is_fitted_")

        lines = [
            "Sparse Additive Hinge Regressor:",
            "  y = intercept + sum_k coef_k * term_k",
            f"  intercept: {self.intercept_:+.4f}",
        ]

        if hasattr(self, "alpha_"):
            lines.append(f"  lasso_alpha: {self.alpha_:.4g}")

        if not self.active_terms_:
            lines.append("  No active terms selected.")
            return "\n".join(lines)

        lines.append("  Active terms:")
        for idx, (coef, (kind, j, t)) in enumerate(zip(self.coef_, self.active_terms_), 1):
            if kind == "linear":
                term_txt = f"x{j}"
            elif kind == "hinge_pos":
                term_txt = f"max(0, x{j} - {t:.4f})"
            else:
                term_txt = f"max(0, {t:.4f} - x{j})"
            lines.append(f"    {idx}. {coef:+.4f} * {term_txt}")

        equation = f"{self.intercept_:+.4f}"
        for coef, (kind, j, t) in zip(self.coef_, self.active_terms_):
            if kind == "linear":
                term_txt = f"x{j}"
            elif kind == "hinge_pos":
                term_txt = f"max(0, x{j} - {t:.4f})"
            else:
                term_txt = f"max(0, {t:.4f} - x{j})"
            equation += f" {coef:+.4f}*{term_txt}"
        lines.append(f"  Equation: y = {equation}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseAdditiveHingeRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseAddHinge_v1"
model_description = "Sparse additive hinge basis with quantile thresholds, L1 selection, and top-term refit"
model_defs = [(model_shorthand_name, SparseAdditiveHingeRegressor())]


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
