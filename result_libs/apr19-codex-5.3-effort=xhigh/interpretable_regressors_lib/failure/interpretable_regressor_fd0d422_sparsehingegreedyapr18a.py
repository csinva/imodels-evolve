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
from itertools import combinations

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


class SparseBasisGreedyRegressor(BaseEstimator, RegressorMixin):
    """
    Interpretable scikit-learn compatible regressor.

    Greedy sparse basis model with a compact closed-form equation:
      - candidate terms: linear, quadratic, hinges, step indicators, sparse interactions
      - greedy forward selection on normalized candidates
      - small ridge refit on selected raw terms for stable coefficients

    The resulting string is intentionally explicit so an LLM can simulate predictions.
    Must implement: fit(X, y), predict(X), and __str__().
    """

    def __init__(
        self,
        max_terms=7,
        max_candidates=140,
        interaction_features=6,
        quantile_knots=(0.25, 0.5, 0.75),
        include_quadratic=True,
        include_steps=True,
        selection_l2=1e-4,
        final_l2=1e-6,
        min_abs_correlation=1e-3,
        min_improvement=1e-4,
        prune_threshold=1e-5,
    ):
        self.max_terms = max_terms
        self.max_candidates = max_candidates
        self.interaction_features = interaction_features
        self.quantile_knots = quantile_knots
        self.include_quadratic = include_quadratic
        self.include_steps = include_steps
        self.selection_l2 = selection_l2
        self.final_l2 = final_l2
        self.min_abs_correlation = min_abs_correlation
        self.min_improvement = min_improvement
        self.prune_threshold = prune_threshold

    def _eval_term(self, X, term):
        kind = term[0]
        if kind == "linear":
            j = term[1]
            return X[:, j]
        if kind == "quadratic":
            j = term[1]
            return X[:, j] ** 2
        if kind == "hinge_pos":
            j, knot = term[1], term[2]
            return np.maximum(0.0, X[:, j] - knot)
        if kind == "hinge_neg":
            j, knot = term[1], term[2]
            return np.maximum(0.0, knot - X[:, j])
        if kind == "step_gt":
            j, knot = term[1], term[2]
            return (X[:, j] > knot).astype(float)
        if kind == "interaction":
            a, b = term[1], term[2]
            return X[:, a] * X[:, b]
        raise ValueError(f"Unknown term kind: {kind}")

    def _term_to_str(self, term):
        kind = term[0]
        if kind == "linear":
            return f"x{term[1]}"
        if kind == "quadratic":
            return f"(x{term[1]}^2)"
        if kind == "hinge_pos":
            return f"max(0, x{term[1]} - {term[2]:.4f})"
        if kind == "hinge_neg":
            return f"max(0, {term[2]:.4f} - x{term[1]})"
        if kind == "step_gt":
            return f"I(x{term[1]} > {term[2]:.4f})"
        if kind == "interaction":
            return f"(x{term[1]} * x{term[2]})"
        return str(term)

    def _safe_corr(self, a, b):
        a_centered = a - float(np.mean(a))
        b_centered = b - float(np.mean(b))
        denom = float(np.linalg.norm(a_centered) * np.linalg.norm(b_centered))
        if denom <= 1e-12:
            return 0.0
        return float(np.dot(a_centered, b_centered) / denom)

    def _build_candidate_library(self, X, y):
        n_samples, n_features = X.shape
        terms = []
        columns = []

        y_centered = y - float(np.mean(y))
        feature_corr = np.array(
            [abs(self._safe_corr(X[:, j], y_centered)) for j in range(n_features)],
            dtype=float,
        )
        top_for_interactions = np.argsort(feature_corr)[::-1][: min(self.interaction_features, n_features)]

        for j in range(n_features):
            xj = X[:, j]
            terms.append(("linear", j))
            columns.append(xj)

            if self.include_quadratic:
                terms.append(("quadratic", j))
                columns.append(xj**2)

            knots = np.quantile(xj, self.quantile_knots)
            for knot in np.unique(np.round(knots.astype(float), 6)):
                terms.append(("hinge_pos", j, float(knot)))
                columns.append(np.maximum(0.0, xj - knot))

                terms.append(("hinge_neg", j, float(knot)))
                columns.append(np.maximum(0.0, knot - xj))

                if self.include_steps:
                    terms.append(("step_gt", j, float(knot)))
                    columns.append((xj > knot).astype(float))

        for a, b in combinations(top_for_interactions, 2):
            terms.append(("interaction", int(a), int(b)))
            columns.append(X[:, a] * X[:, b])

        if not columns:
            return [], np.zeros((n_samples, 0), dtype=float)

        Phi = np.column_stack(columns).astype(float)
        std = Phi.std(axis=0)
        keep = std > 1e-10
        Phi = Phi[:, keep]
        terms = [t for t, k in zip(terms, keep) if k]
        return terms, Phi

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        self.feature_names_ = [f"x{i}" for i in range(n_features)]

        terms, Phi = self._build_candidate_library(X, y)
        if Phi.shape[1] == 0:
            self.intercept_ = float(np.mean(y))
            self.coef_ = np.array([], dtype=float)
            self.terms_ = []
            self.feature_importance_ = np.zeros(n_features, dtype=float)
            self.inactive_features_ = self.feature_names_
            return self

        y_centered = y - float(np.mean(y))
        phi_mean = Phi.mean(axis=0)
        phi_std = Phi.std(axis=0) + 1e-12
        Phi_norm = (Phi - phi_mean) / phi_std

        corr_scores = np.abs(Phi_norm.T @ y_centered) / max(1, n_samples - 1)
        pool = np.argsort(corr_scores)[::-1][: min(self.max_candidates, Phi.shape[1])]

        selected = []
        residual = y_centered.copy()
        prev_rss = float(residual @ residual)

        for _ in range(self.max_terms):
            candidates = [idx for idx in pool if idx not in selected]
            if not candidates:
                break

            candidate_corrs = np.abs(Phi_norm[:, candidates].T @ residual) / max(1, n_samples - 1)
            best_pos = int(np.argmax(candidate_corrs))
            best_idx = int(candidates[best_pos])
            best_corr = float(candidate_corrs[best_pos])
            if best_corr < self.min_abs_correlation:
                break

            trial = selected + [best_idx]
            B = Phi_norm[:, trial]
            gram = B.T @ B + self.selection_l2 * np.eye(B.shape[1], dtype=float)
            rhs = B.T @ y_centered
            try:
                beta = np.linalg.solve(gram, rhs)
            except np.linalg.LinAlgError:
                beta = np.linalg.lstsq(B, y_centered, rcond=None)[0]

            new_residual = y_centered - B @ beta
            rss = float(new_residual @ new_residual)
            if prev_rss - rss < self.min_improvement * (prev_rss + 1e-12):
                break

            selected = trial
            residual = new_residual
            prev_rss = rss

        if not selected:
            selected = [int(pool[0])]

        def _ridge_with_intercept(A_mat):
            A_aug = np.column_stack([np.ones(n_samples, dtype=float), A_mat])
            reg = np.diag([0.0] + [self.final_l2] * A_mat.shape[1]).astype(float)
            lhs = A_aug.T @ A_aug + reg
            rhs_local = A_aug.T @ y
            try:
                beta_local = np.linalg.solve(lhs, rhs_local)
            except np.linalg.LinAlgError:
                beta_local = np.linalg.lstsq(A_aug, y, rcond=None)[0]
            return float(beta_local[0]), beta_local[1:]

        A = Phi[:, selected]
        intercept, coefs = _ridge_with_intercept(A)

        keep = np.abs(coefs) >= self.prune_threshold
        if np.any(keep):
            selected = [selected[i] for i, k in enumerate(keep) if k]
            coefs = coefs[keep]
            A = Phi[:, selected]
            intercept, coefs = _ridge_with_intercept(A)
        else:
            strongest = int(np.argmax(np.abs(coefs)))
            selected = [selected[strongest]]
            A = Phi[:, selected]
            intercept, coefs = _ridge_with_intercept(A)

        self.intercept_ = float(intercept)
        self.coef_ = np.asarray(coefs, dtype=float)
        self.terms_ = [terms[i] for i in selected]

        feature_importance = np.zeros(n_features, dtype=float)
        for coef, term in zip(self.coef_, self.terms_):
            mag = float(abs(coef))
            if term[0] == "interaction":
                feature_importance[term[1]] += 0.5 * mag
                feature_importance[term[2]] += 0.5 * mag
            else:
                feature_importance[term[1]] += mag
        self.feature_importance_ = feature_importance
        self.inactive_features_ = [
            f"x{i}" for i, score in enumerate(feature_importance) if score < 1e-10
        ]
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "coef_", "terms_", "n_features_in_"])
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}.")

        preds = np.full(X.shape[0], self.intercept_, dtype=float)
        for coef, term in zip(self.coef_, self.terms_):
            preds += coef * self._eval_term(X, term)
        return preds

    def __str__(self):
        check_is_fitted(self, ["intercept_", "coef_", "terms_", "feature_importance_"])

        lines = [
            "Sparse Basis Greedy Regressor",
            "Notation: I(condition) = 1 if condition is true, else 0.",
            f"intercept = {self.intercept_:+.6f}",
            "Prediction equation (sum all active terms):",
        ]

        equation = [f"y = {self.intercept_:+.6f}"]
        for coef, term in zip(self.coef_, self.terms_):
            term_str = self._term_to_str(term)
            lines.append(f"  {coef:+.6f} * {term_str}")
            equation.append(f"{coef:+.6f} * {term_str}")
        lines.append("  " + " ".join(equation))

        order = np.argsort(self.feature_importance_)[::-1]
        top = [
            f"x{i} ({self.feature_importance_[i]:.3f})"
            for i in order
            if self.feature_importance_[i] > 1e-10
        ]
        if top:
            lines.append("Feature influence ranking (higher = more used): " + ", ".join(top))

        if self.inactive_features_:
            lines.append("Inactive original features: " + ", ".join(self.inactive_features_))

        lines.append(
            f"Compactness: {len(self.terms_)} active terms; model is computed by adding these terms to intercept."
        )
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseBasisGreedyRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseHingeGreedyApr18a"
model_description = "Custom sparse basis regressor: greedy-selected linear/quadratic/hinge/step/interaction terms with explicit equation string."
model_defs = [(model_shorthand_name, SparseBasisGreedyRegressor())]


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
