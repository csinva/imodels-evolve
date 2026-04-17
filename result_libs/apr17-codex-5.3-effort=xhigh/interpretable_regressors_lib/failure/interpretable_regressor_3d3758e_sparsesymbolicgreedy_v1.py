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


class SparseSymbolicRegressor(BaseEstimator, RegressorMixin):
    """Sparse symbolic additive regressor with greedy residual matching.

    Candidate term library:
    - linear: x_j
    - square: x_j^2
    - positive/negative hinge: max(0, x_j - t), max(0, t - x_j)
    - pairwise interactions: x_i * x_j

    The model builds a compact equation by selecting terms that best explain
    residuals and refitting coefficients with light ridge stabilization.
    """

    def __init__(
        self,
        max_linear_features=6,
        max_total_terms=8,
        max_interaction_features=4,
        min_abs_corr=0.03,
        ridge_lambda=1e-3,
    ):
        self.max_linear_features = max_linear_features
        self.max_total_terms = max_total_terms
        self.max_interaction_features = max_interaction_features
        self.min_abs_corr = min_abs_corr
        self.ridge_lambda = ridge_lambda

    def _safe_abs_corr(self, a, b):
        a_std = float(np.std(a))
        b_std = float(np.std(b))
        if a_std < 1e-12 or b_std < 1e-12:
            return 0.0
        c = np.corrcoef(a, b)[0, 1]
        if not np.isfinite(c):
            return 0.0
        return float(abs(c))

    def _term_values(self, X, term):
        kind = term[0]
        if kind == "linear":
            _, j = term
            return X[:, j]
        if kind == "square":
            _, j = term
            return X[:, j] ** 2
        if kind == "pos_hinge":
            _, j, t = term
            return np.maximum(0.0, X[:, j] - t)
        if kind == "neg_hinge":
            _, j, t = term
            return np.maximum(0.0, t - X[:, j])
        if kind == "interaction":
            _, i, j = term
            return X[:, i] * X[:, j]
        raise ValueError(f"Unknown term type: {kind}")

    def _fit_ridge_closed_form(self, X_design, y):
        n_terms = X_design.shape[1]
        gram = X_design.T @ X_design
        rhs = X_design.T @ y
        gram = gram + self.ridge_lambda * np.eye(n_terms)
        coef = np.linalg.solve(gram, rhs)
        return coef

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # Feature preselection by absolute correlation with y.
        feature_scores = np.array([self._safe_abs_corr(X[:, j], y) for j in range(n_features)])
        ranked_features = np.argsort(feature_scores)[::-1]
        linear_features = ranked_features[: min(self.max_linear_features, n_features)]

        selected_terms = [("linear", int(j)) for j in linear_features]
        selected_matrix = np.column_stack([self._term_values(X, t) for t in selected_terms])

        # Initial compact linear backbone.
        y_mean = float(np.mean(y))
        y_centered = y - y_mean
        coef = self._fit_ridge_closed_form(selected_matrix, y_centered)

        # Build nonlinear candidate dictionary on top-ranked features.
        nonlinear_candidates = []
        nonlinear_features = ranked_features[: min(self.max_interaction_features, n_features)]
        for j in nonlinear_features:
            j = int(j)
            col = X[:, j]
            med = float(np.median(col))
            nonlinear_candidates.extend([
                ("square", j),
                ("pos_hinge", j, 0.0),
                ("neg_hinge", j, 0.0),
                ("pos_hinge", j, med),
                ("neg_hinge", j, med),
            ])

        for a in range(len(nonlinear_features)):
            for b in range(a + 1, len(nonlinear_features)):
                i = int(nonlinear_features[a])
                j = int(nonlinear_features[b])
                nonlinear_candidates.append(("interaction", i, j))

        remaining = list(nonlinear_candidates)

        # Greedy residual matching pursuit for nonlinear terms.
        while len(selected_terms) < self.max_total_terms and remaining:
            pred = y_mean + selected_matrix @ coef
            residual = y - pred

            best_idx = None
            best_score = -1.0
            for idx, term in enumerate(remaining):
                v = self._term_values(X, term)
                score = self._safe_abs_corr(v, residual)
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is None or best_score < self.min_abs_corr:
                break

            best_term = remaining.pop(best_idx)
            new_col = self._term_values(X, best_term)

            selected_terms.append(best_term)
            selected_matrix = np.column_stack([selected_matrix, new_col])
            coef = self._fit_ridge_closed_form(selected_matrix, y_centered)

        # Prune tiny terms for compactness/readability.
        keep = np.where(np.abs(coef) > 1e-5)[0]
        if len(keep) == 0:
            keep = np.array([int(np.argmax(np.abs(coef)))])

        self.intercept_ = y_mean
        self.terms_ = [selected_terms[k] for k in keep]
        self.coef_ = coef[keep].astype(float)

        # Refit intercept after pruning.
        X_sel = np.column_stack([self._term_values(X, t) for t in self.terms_])
        self.intercept_ = float(np.mean(y - X_sel @ self.coef_))

        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "terms_", "coef_"])
        X = np.asarray(X, dtype=float)

        X_sel = np.column_stack([self._term_values(X, t) for t in self.terms_])
        return self.intercept_ + X_sel @ self.coef_

    def _term_to_str(self, term):
        kind = term[0]
        if kind == "linear":
            return f"x{term[1]}"
        if kind == "square":
            return f"(x{term[1]})^2"
        if kind == "pos_hinge":
            return f"max(0, x{term[1]} - {term[2]:.3f})"
        if kind == "neg_hinge":
            return f"max(0, {term[2]:.3f} - x{term[1]})"
        if kind == "interaction":
            return f"x{term[1]} * x{term[2]}"
        return str(term)

    def __str__(self):
        check_is_fitted(self, ["intercept_", "terms_", "coef_"])

        ops_est = 1 + len(self.terms_)
        lines = [
            "Sparse Symbolic Regressor:",
            f"  Terms: {len(self.terms_)} (estimated arithmetic operations: ~{ops_est})",
            f"  Equation: y = {self.intercept_:+.4f} + sum_k c_k * term_k",
            "",
            "Active terms:",
        ]

        order = np.argsort(np.abs(self.coef_))[::-1]
        for idx in order:
            coef = self.coef_[idx]
            term = self.terms_[idx]
            lines.append(f"  {coef:+.4f} * {self._term_to_str(term)}")

        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseSymbolicRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseSymbolicGreedy_v1"
model_description = "Greedy sparse symbolic regressor using linear, hinge, square, and interaction terms with compact equation output"
model_defs = [(model_shorthand_name, SparseSymbolicRegressor())]


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
