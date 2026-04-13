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


class HingeSplineRuleRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse linear-hinge regressor with explicit human-readable rules.

    The model builds a compact basis over raw features:
      - linear terms x_j
      - one-sided hinge terms max(0, x_j - t) and max(0, t - x_j)
        at a small set of data-driven thresholds t

    A small ridge path picks stable coefficients, then we keep only the most
    influential terms and refit once for a short final equation.
    """

    def __init__(
        self,
        screening_features=10,
        max_terms=8,
        n_thresholds=3,
        alpha_grid=(0.02, 0.08, 0.3, 1.0, 3.0),
        min_abs_coef=0.03,
    ):
        self.screening_features = screening_features
        self.max_terms = max_terms
        self.n_thresholds = n_thresholds
        self.alpha_grid = alpha_grid
        self.min_abs_coef = min_abs_coef

    @staticmethod
    def _safe_corr(x, y):
        xc = x - np.mean(x)
        yc = y - np.mean(y)
        denom = (np.linalg.norm(xc) + 1e-12) * (np.linalg.norm(yc) + 1e-12)
        return float(np.dot(xc, yc) / denom)

    @staticmethod
    def _ridge_fit(M, y, alpha):
        n = M.shape[0]
        D = np.column_stack([np.ones(n, dtype=float), M])
        reg = float(alpha) * np.eye(D.shape[1], dtype=float)
        reg[0, 0] = 0.0
        beta = np.linalg.solve(D.T @ D + reg, D.T @ y)
        return float(beta[0]), np.asarray(beta[1:], dtype=float)

    def _candidate_thresholds(self, x):
        # Include zero to capture common synthetic threshold tests,
        # plus inner quantiles for data-adaptive flexibility.
        ts = [0.0]
        q_count = max(1, int(self.n_thresholds))
        qs = np.linspace(0.2, 0.8, q_count)
        for q in qs:
            ts.append(float(np.quantile(x, q)))

        ts = np.array(sorted(ts), dtype=float)
        uniq = [ts[0]]
        for v in ts[1:]:
            if abs(v - uniq[-1]) > 1e-9:
                uniq.append(v)
        return uniq

    def _build_design(self, X, features):
        cols = []
        terms = []
        for j in features:
            xj = X[:, j]
            cols.append(xj)
            terms.append(("linear", int(j), None, +1))

            for t in self._candidate_thresholds(xj):
                cols.append(np.maximum(0.0, xj - t))
                terms.append(("hinge", int(j), float(t), +1))

                cols.append(np.maximum(0.0, t - xj))
                terms.append(("hinge", int(j), float(t), -1))

        if len(cols) == 0:
            return np.zeros((X.shape[0], 0), dtype=float), []
        M = np.column_stack(cols).astype(float)
        return M, terms

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape

        k = min(max(1, int(self.screening_features)), p)
        corrs = np.array([abs(self._safe_corr(X[:, j], y)) for j in range(p)], dtype=float)
        screened = list(np.argsort(-corrs)[:k])

        M_full, terms_full = self._build_design(X, screened)
        if M_full.shape[1] == 0:
            self.intercept_ = float(np.mean(y))
            self.terms_ = []
            self.weights_ = np.zeros(0, dtype=float)
            self.n_features_in_ = p
            self.is_fitted_ = True
            return self

        best_loss = float("inf")
        best_intercept = float(np.mean(y))
        best_coef = np.zeros(M_full.shape[1], dtype=float)
        for alpha in self.alpha_grid:
            inter, coef = self._ridge_fit(M_full, y, alpha)
            resid = y - (inter + M_full @ coef)
            mse = float(np.mean(resid ** 2))
            penalty = 0.02 * float(np.mean(np.abs(coef)))
            score = mse + penalty
            if score < best_loss:
                best_loss = score
                best_intercept = inter
                best_coef = coef

        strengths = np.abs(best_coef) * np.std(M_full, axis=0)
        order = np.argsort(-strengths)
        k_keep = min(max(1, int(self.max_terms)), len(order))
        keep = [int(i) for i in order[:k_keep] if strengths[int(i)] > 1e-10]
        if len(keep) == 0:
            keep = [int(order[0])]

        M_keep = M_full[:, keep]

        refit_intercept, refit_coef = self._ridge_fit(M_keep, y, alpha=0.15)

        final_terms = []
        final_weights = []
        for idx_local, w in enumerate(refit_coef):
            if abs(float(w)) >= float(self.min_abs_coef):
                term = terms_full[keep[idx_local]]
                final_terms.append(term)
                final_weights.append(float(w))

        if len(final_terms) == 0:
            strongest = int(np.argmax(np.abs(refit_coef)))
            final_terms = [terms_full[keep[strongest]]]
            final_weights = [float(refit_coef[strongest])]

        self.intercept_ = float(refit_intercept)
        self.terms_ = final_terms
        self.weights_ = np.asarray(final_weights, dtype=float)
        self.n_features_in_ = p
        self.is_fitted_ = True
        return self

    def _eval_term(self, X, term):
        ttype, j, thr, direction = term
        xj = X[:, j]
        if ttype == "linear":
            return xj
        if direction > 0:
            return np.maximum(0.0, xj - thr)
        return np.maximum(0.0, thr - xj)

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        X = np.asarray(X, dtype=float)
        out = np.full(X.shape[0], self.intercept_, dtype=float)
        for w, term in zip(self.weights_, self.terms_):
            out += float(w) * self._eval_term(X, term)
        return out

    def __str__(self):
        check_is_fitted(self, "is_fitted_")
        lines = ["Hinge Spline Rule Regressor:"]
        lines.append("  Prediction = intercept + sum_k w_k * term_k")
        lines.append(f"  intercept: {self.intercept_:+.4f}")

        if len(self.terms_) == 0:
            lines.append("  terms: none")
            return "\n".join(lines)

        abs_order = np.argsort(-np.abs(self.weights_))
        for rank, idx in enumerate(abs_order, 1):
            w = float(self.weights_[idx])
            ttype, j, thr, direction = self.terms_[int(idx)]
            if ttype == "linear":
                expr = f"x{j}"
            else:
                if direction > 0:
                    expr = f"max(0, x{j} - {thr:.4g})"
                else:
                    expr = f"max(0, {thr:.4g} - x{j})"
            lines.append(f"  {rank}. {w:+.4f} * {expr}")

        lines.append("  Example: plug feature values into each term, multiply by weight, then add intercept.")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
HingeSplineRuleRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "HingeSplineRule_v1"
model_description = "Compact linear-plus-hinge rule equation with quantile thresholds, ridge selection, and top-term refit"
model_defs = [(model_shorthand_name, HingeSplineRuleRegressor())]


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

    std_tests = {t.__name__ for t in ALL_TESTS}
    hard_tests = {t.__name__ for t in HARD_TESTS}
    insight_tests = {t.__name__ for t in INSIGHT_TESTS}
    std_passed = sum(r["passed"] for r in interp_results if r["test"] in std_tests)
    hard_passed = sum(r["passed"] for r in interp_results if r["test"] in hard_tests)
    insight_passed = sum(r["passed"] for r in interp_results if r["test"] in insight_tests)
    print(f"[std {std_passed}/{len(std_tests)}  hard {hard_passed}/{len(hard_tests)}  insight {insight_passed}/{len(insight_tests)}]")
    print(f"total_seconds: {time.time() - t0:.1f}s")
