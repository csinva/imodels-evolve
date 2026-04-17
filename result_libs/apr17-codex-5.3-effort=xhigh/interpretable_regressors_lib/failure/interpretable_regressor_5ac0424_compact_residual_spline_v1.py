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


class CompactResidualSplineRegressor(BaseEstimator, RegressorMixin):
    """Compact sparse linear model with a few greedily selected hinge atoms."""

    def __init__(
        self,
        max_linear_features=4,
        max_hinges=2,
        ridge_lambda=5e-3,
        candidate_quantiles=(0.2, 0.4, 0.6, 0.8),
        max_candidate_features=8,
        min_rel_gain=0.01,
        negligible_feature_eps=5e-3,
    ):
        self.max_linear_features = max_linear_features
        self.max_hinges = max_hinges
        self.ridge_lambda = ridge_lambda
        self.candidate_quantiles = candidate_quantiles
        self.max_candidate_features = max_candidate_features
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

    def _fit_ridge_design(self, Phi, y):
        if Phi.shape[1] == 0:
            return float(np.mean(y)), np.zeros(0, dtype=float)

        x_mean = np.mean(Phi, axis=0)
        x_std = np.std(Phi, axis=0)
        x_std[x_std < 1e-12] = 1.0
        Z = (Phi - x_mean) / x_std

        y_mean = float(np.mean(y))
        y_centered = y - y_mean
        p = Z.shape[1]

        gram = Z.T @ Z + self.ridge_lambda * np.eye(p)
        rhs = Z.T @ y_centered
        coef_std = np.linalg.solve(gram, rhs)
        coef = coef_std / x_std
        intercept = float(y_mean - np.dot(coef, x_mean))
        return intercept, coef

    def _make_hinge(self, x_col, threshold, direction):
        if direction == "above":
            return np.maximum(0.0, x_col - threshold)
        return np.maximum(0.0, threshold - x_col)

    def _build_design(self, X, terms):
        if len(terms) == 0:
            return np.zeros((X.shape[0], 0), dtype=float)
        cols = []
        for term in terms:
            if term["kind"] == "linear":
                cols.append(X[:, term["feature"]])
            else:
                cols.append(
                    self._make_hinge(
                        X[:, term["feature"]],
                        term["threshold"],
                        term["direction"],
                    )
                )
        return np.column_stack(cols)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        corr = np.array([self._safe_abs_corr(X[:, j], y) for j in range(n_features)])
        ranked = np.argsort(corr)[::-1]
        linear_k = min(self.max_linear_features, n_features)
        linear_idx = np.sort(ranked[:linear_k].astype(int))
        terms = [{"kind": "linear", "feature": int(j)} for j in linear_idx]

        design = self._build_design(X, terms)
        intercept, coef = self._fit_ridge_design(design, y)
        pred = intercept + (design @ coef if design.shape[1] else 0.0)
        prev_sse = float(np.dot(y - pred, y - pred))
        min_gain_abs = self.min_rel_gain * (prev_sse + 1e-12)

        candidate_features = ranked[: min(self.max_candidate_features, n_features)]
        self.hinge_terms_ = []

        for _ in range(self.max_hinges):
            residual = y - pred
            best_term = None
            best_proxy_sse = prev_sse

            for j in candidate_features:
                xj = X[:, int(j)]
                thresholds = np.unique(np.round(np.quantile(xj, self.candidate_quantiles), 8))
                for threshold in thresholds:
                    for direction in ("above", "below"):
                        h = self._make_hinge(xj, float(threshold), direction)
                        if float(np.std(h)) < 1e-8:
                            continue

                        h_center = h - np.mean(h)
                        denom = float(np.dot(h_center, h_center) + self.ridge_lambda * n_samples)
                        beta = float(np.dot(residual, h_center) / denom)
                        proxy_residual = residual - beta * h
                        proxy_sse = float(np.dot(proxy_residual, proxy_residual))
                        if proxy_sse < best_proxy_sse:
                            best_proxy_sse = proxy_sse
                            best_term = {
                                "kind": "hinge",
                                "feature": int(j),
                                "threshold": float(threshold),
                                "direction": direction,
                            }

            if best_term is None or (prev_sse - best_proxy_sse) < min_gain_abs:
                break

            candidate_terms = terms + [best_term]
            candidate_design = self._build_design(X, candidate_terms)
            cand_intercept, cand_coef = self._fit_ridge_design(candidate_design, y)
            cand_pred = cand_intercept + candidate_design @ cand_coef
            cand_sse = float(np.dot(y - cand_pred, y - cand_pred))

            if (prev_sse - cand_sse) < min_gain_abs:
                break

            terms = candidate_terms
            design = candidate_design
            intercept = cand_intercept
            coef = cand_coef
            pred = cand_pred
            prev_sse = cand_sse
            self.hinge_terms_.append(best_term)

        self.terms_ = terms
        self.intercept_ = float(intercept)
        self.term_coef_ = coef.astype(float, copy=True)

        contrib = np.zeros(n_features, dtype=float)
        if design.shape[1]:
            for term, weight, col in zip(self.terms_, self.term_coef_, design.T):
                contrib[term["feature"]] += float(np.mean(np.abs(weight * col)))
        self.feature_importances_ = contrib
        return self

    def predict(self, X):
        check_is_fitted(self, ["terms_", "intercept_", "term_coef_"])
        X = np.asarray(X, dtype=float)
        design = self._build_design(X, self.terms_)
        return self.intercept_ + (design @ self.term_coef_ if design.shape[1] else 0.0)

    def __str__(self):
        check_is_fitted(self, ["terms_", "term_coef_", "feature_importances_"])
        active_features = sorted({t["feature"] for t in self.terms_})

        equation_terms = [f"{self.intercept_:+.5f}"]
        for term, coef in zip(self.terms_, self.term_coef_):
            if term["kind"] == "linear":
                equation_terms.append(f"{coef:+.5f}*x{term['feature']}")
            else:
                if term["direction"] == "above":
                    hinge_txt = f"max(0, x{term['feature']} - {term['threshold']:.5f})"
                else:
                    hinge_txt = f"max(0, {term['threshold']:.5f} - x{term['feature']})"
                equation_terms.append(f"{coef:+.5f}*{hinge_txt}")

        order = np.argsort(self.feature_importances_)[::-1]
        top = [f"x{j}:{self.feature_importances_[j]:.4f}" for j in order[: min(10, self.n_features_in_)]]
        negligible = [
            f"x{j}"
            for j in range(self.n_features_in_)
            if self.feature_importances_[j] <= self.negligible_feature_eps
        ]

        lines = [
            "Compact Residual Spline Regressor (sparse linear + selected hinges)",
            f"Total terms: {len(self.terms_)}",
            f"Active features ({len(active_features)}): " + (", ".join(f"x{j}" for j in active_features) if active_features else "none"),
            "Model equation:",
            "  y = " + " ".join(equation_terms),
        ]

        hinge_count = sum(1 for t in self.terms_ if t["kind"] == "hinge")
        lines.append(f"Hinge terms selected: {hinge_count}")

        lines.extend([
            "",
            "Most influential features (mean absolute contribution):",
            "  " + ", ".join(top),
            "Features with negligible effect:",
            "  " + (", ".join(negligible) if negligible else "none"),
            "",
            "Prediction recipe: compute the equation directly from left to right.",
        ])
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
CompactResidualSplineRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "CompactResidualSpline_v1"
model_description = "Greedy compact equation: sparse linear backbone plus up to two residual hinge atoms with joint ridge refits"
model_defs = [(model_shorthand_name, CompactResidualSplineRegressor())]


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
