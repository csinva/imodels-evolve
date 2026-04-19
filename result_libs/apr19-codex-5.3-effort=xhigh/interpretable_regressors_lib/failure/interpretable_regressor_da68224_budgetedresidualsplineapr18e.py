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


class BudgetedResidualSplineRegressor(BaseEstimator, RegressorMixin):
    """
    Linear backbone with a tiny budget of nonlinear residual corrections.

    The model keeps a dense-enough linear core for predictive robustness, then
    adds only a few piecewise/rule terms if they provide measurable residual
    gain. The final output is an explicit symbolic equation in raw features.
    """

    def __init__(
        self,
        max_linear_features=18,
        max_nonlinear_terms=3,
        candidate_features=8,
        interaction_pool=4,
        knot_quantiles=(0.2, 0.4, 0.6, 0.8),
        selection_l2=5e-4,
        final_l2=2e-5,
        min_abs_score=5e-4,
        min_relative_improvement=4e-4,
        min_linear_rel_coef=0.015,
        nonlinear_prune_threshold=2e-4,
        inactive_rel_threshold=0.06,
        include_steps=True,
        include_hinge_neg=True,
        include_interactions=True,
    ):
        self.max_linear_features = max_linear_features
        self.max_nonlinear_terms = max_nonlinear_terms
        self.candidate_features = candidate_features
        self.interaction_pool = interaction_pool
        self.knot_quantiles = knot_quantiles
        self.selection_l2 = selection_l2
        self.final_l2 = final_l2
        self.min_abs_score = min_abs_score
        self.min_relative_improvement = min_relative_improvement
        self.min_linear_rel_coef = min_linear_rel_coef
        self.nonlinear_prune_threshold = nonlinear_prune_threshold
        self.inactive_rel_threshold = inactive_rel_threshold
        self.include_steps = include_steps
        self.include_hinge_neg = include_hinge_neg
        self.include_interactions = include_interactions

    def _safe_center_corr(self, a, b):
        ac = a - float(np.mean(a))
        bc = b - float(np.mean(b))
        denom = float(np.linalg.norm(ac) * np.linalg.norm(bc))
        if denom <= 1e-12:
            return 0.0
        return float(np.dot(ac, bc) / denom)

    def _ridge_with_intercept(self, Phi, y, l2):
        n = Phi.shape[0]
        A = np.column_stack([np.ones(n, dtype=float), Phi])
        reg = np.diag([0.0] + [l2] * Phi.shape[1]).astype(float)
        lhs = A.T @ A + reg
        rhs = A.T @ y
        try:
            beta = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(A, y, rcond=None)[0]
        return float(beta[0]), np.asarray(beta[1:], dtype=float)

    def _eval_term(self, X, term):
        kind = term[0]
        if kind == "linear":
            return X[:, term[1]]
        if kind == "hinge_pos":
            return np.maximum(0.0, X[:, term[1]] - term[2])
        if kind == "hinge_neg":
            return np.maximum(0.0, term[2] - X[:, term[1]])
        if kind == "step_gt":
            return (X[:, term[1]] > term[2]).astype(float)
        if kind == "interaction":
            return X[:, term[1]] * X[:, term[2]]
        raise ValueError(f"Unknown term: {term}")

    def _term_to_str(self, term):
        kind = term[0]
        if kind == "linear":
            return f"x{term[1]}"
        if kind == "hinge_pos":
            return f"max(0, x{term[1]} - {term[2]:.6f})"
        if kind == "hinge_neg":
            return f"max(0, {term[2]:.6f} - x{term[1]})"
        if kind == "step_gt":
            return f"I(x{term[1]} > {term[2]:.6f})"
        if kind == "interaction":
            return f"(x{term[1]} * x{term[2]})"
        return str(term)

    def _build_nonlinear_library(self, X, feature_pool):
        n_samples = X.shape[0]
        terms = []
        cols = []

        for j in feature_pool:
            j = int(j)
            xj = X[:, j]
            knots = [0.0]
            for q in self.knot_quantiles:
                knots.append(float(np.quantile(xj, q)))
            knots = sorted(set(np.round(np.asarray(knots, dtype=float), 6)))

            for knot in knots:
                terms.append(("hinge_pos", j, float(knot)))
                cols.append(np.maximum(0.0, xj - float(knot)))

                if self.include_hinge_neg:
                    terms.append(("hinge_neg", j, float(knot)))
                    cols.append(np.maximum(0.0, float(knot) - xj))

                if self.include_steps:
                    terms.append(("step_gt", j, float(knot)))
                    cols.append((xj > float(knot)).astype(float))

        if self.include_interactions:
            inter_top = list(feature_pool[: min(self.interaction_pool, len(feature_pool))])
            for a, b in combinations(inter_top, 2):
                a = int(a)
                b = int(b)
                terms.append(("interaction", a, b))
                cols.append(X[:, a] * X[:, b])

        if not cols:
            return [], np.zeros((n_samples, 0), dtype=float)

        Phi = np.column_stack(cols).astype(float)
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
            raise ValueError("X and y must have the same number of rows")

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        if n_features == 0:
            self.intercept_ = float(np.mean(y))
            self.coef_ = np.array([], dtype=float)
            self.terms_ = []
            self.linear_term_count_ = 0
            self.feature_importance_ = np.array([], dtype=float)
            self.inactive_features_ = []
            self.meaningful_features_ = []
            return self

        # Stage 1: stable ridge ranking on standardized features.
        x_mean = X.mean(axis=0)
        x_scale = X.std(axis=0)
        x_scale[x_scale < 1e-12] = 1.0
        Z = (X - x_mean) / x_scale
        y_centered = y - float(np.mean(y))
        lhs = Z.T @ Z + self.selection_l2 * np.eye(n_features, dtype=float)
        rhs = Z.T @ y_centered
        try:
            w_std = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            w_std = np.linalg.lstsq(Z, y_centered, rcond=None)[0]
        raw_w = w_std / x_scale

        ranked = np.argsort(np.abs(raw_w))[::-1]
        if n_features <= self.max_linear_features:
            linear_features = [int(j) for j in ranked]
        else:
            max_mag = float(np.max(np.abs(raw_w)))
            min_mag = self.min_linear_rel_coef * max(max_mag, 1e-12)
            linear_features = [
                int(j) for j in ranked if abs(raw_w[j]) >= min_mag
            ][: self.max_linear_features]
            if not linear_features:
                linear_features = [int(ranked[0])]

        current_terms = [("linear", j) for j in linear_features]
        current_cols = [X[:, j] for j in linear_features]
        Phi_current = np.column_stack(current_cols).astype(float)
        intercept, coefs = self._ridge_with_intercept(Phi_current, y, self.final_l2)
        residual = y - (intercept + Phi_current @ coefs)
        prev_rss = float(residual @ residual)

        # Stage 2: add a tiny number of residual piecewise/rule corrections.
        feature_pool = [
            int(j) for j in ranked[: min(self.candidate_features, len(ranked))]
        ]
        candidate_terms, Phi_candidates = self._build_nonlinear_library(X, feature_pool)
        remaining = list(range(Phi_candidates.shape[1]))
        nonlinear_added = 0

        while nonlinear_added < self.max_nonlinear_terms and remaining:
            scores = np.array(
                [abs(self._safe_center_corr(Phi_candidates[:, j], residual)) for j in remaining],
                dtype=float,
            )
            best_pos = int(np.argmax(scores))
            best_j = int(remaining[best_pos])
            if float(scores[best_pos]) < self.min_abs_score:
                break

            trial_cols = current_cols + [Phi_candidates[:, best_j]]
            Phi_trial = np.column_stack(trial_cols).astype(float)
            trial_intercept, trial_coefs = self._ridge_with_intercept(
                Phi_trial, y, self.final_l2
            )
            trial_residual = y - (trial_intercept + Phi_trial @ trial_coefs)
            rss = float(trial_residual @ trial_residual)
            rel_gain = (prev_rss - rss) / (prev_rss + 1e-12)
            remaining = [j for j in remaining if j != best_j]
            if rel_gain < self.min_relative_improvement:
                continue

            current_cols = trial_cols
            current_terms.append(candidate_terms[best_j])
            intercept, coefs = trial_intercept, trial_coefs
            residual, prev_rss = trial_residual, rss
            nonlinear_added += 1

        # Prune only tiny nonlinear corrections (keep linear backbone intact).
        linear_count = len(linear_features)
        keep = np.ones(len(current_terms), dtype=bool)
        for idx in range(linear_count, len(current_terms)):
            keep[idx] = abs(coefs[idx]) >= self.nonlinear_prune_threshold
        if np.any(keep):
            current_terms = [t for t, k in zip(current_terms, keep) if k]
            current_cols = [c for c, k in zip(current_cols, keep) if k]
        else:
            current_terms = [current_terms[0]]
            current_cols = [current_cols[0]]

        Phi_final = np.column_stack(current_cols).astype(float)
        intercept, coefs = self._ridge_with_intercept(Phi_final, y, self.final_l2)

        self.intercept_ = float(intercept)
        self.coef_ = np.asarray(coefs, dtype=float)
        self.terms_ = list(current_terms)
        self.linear_term_count_ = int(sum(t[0] == "linear" for t in self.terms_))

        importance = np.zeros(n_features, dtype=float)
        for c, t in zip(self.coef_, self.terms_):
            mag = float(abs(c))
            if t[0] == "interaction":
                importance[t[1]] += 0.5 * mag
                importance[t[2]] += 0.5 * mag
            else:
                importance[t[1]] += mag
        self.feature_importance_ = importance

        max_imp = float(np.max(importance)) if n_features > 0 else 0.0
        cutoff = self.inactive_rel_threshold * max(max_imp, 1e-12)
        meaningful = [f"x{i}" for i in range(n_features) if importance[i] >= cutoff]
        inactive = [f"x{i}" for i in range(n_features) if importance[i] < cutoff]
        self.meaningful_features_ = meaningful
        self.inactive_features_ = inactive
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "coef_", "terms_", "n_features_in_"])

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")

        preds = np.full(X.shape[0], self.intercept_, dtype=float)
        for c, t in zip(self.coef_, self.terms_):
            preds += c * self._eval_term(X, t)
        return preds

    def __str__(self):
        check_is_fitted(
            self,
            [
                "intercept_",
                "coef_",
                "terms_",
                "linear_term_count_",
                "feature_importance_",
                "meaningful_features_",
                "inactive_features_",
            ],
        )

        lines = [
            "Budgeted Residual Spline Regressor",
            "Exact prediction uses raw features x0, x1, ... directly.",
            "Definitions: I(condition)=1 if true else 0.",
            "",
            "Equation (exact):",
        ]

        eq_parts = [f"{self.intercept_:+.6f}"]
        for c, t in zip(self.coef_, self.terms_):
            eq_parts.append(f"{c:+.6f}*{self._term_to_str(t)}")
        lines.append("  y = " + " ".join(eq_parts))

        lines.append("")
        lines.append("Term list:")
        for i, (c, t) in enumerate(zip(self.coef_, self.terms_), 1):
            lines.append(f"  t{i}: {c:+.6f} * {self._term_to_str(t)}")

        if self.meaningful_features_:
            lines.append("")
            lines.append("Meaningful features: " + ", ".join(self.meaningful_features_))

        ordered = np.argsort(self.feature_importance_)[::-1]
        ranked = [f"x{j} ({self.feature_importance_[j]:.3f})" for j in ordered if self.feature_importance_[j] > 1e-12]
        if ranked:
            lines.append("Feature influence ranking: " + ", ".join(ranked))

        if self.inactive_features_:
            lines.append("Features with near-zero influence: " + ", ".join(self.inactive_features_))

        nonlinear_count = len(self.terms_) - self.linear_term_count_
        lines.append(
            f"Compactness: {self.linear_term_count_} linear terms + {nonlinear_count} nonlinear corrections."
        )
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
BudgetedResidualSplineRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "BudgetedResidualSplineApr18e"
model_description = (
    "Dense ridge-ranked linear backbone with a small budget of residual hinge/step/"
    "interaction corrections selected by gain for compact raw-feature equations."
)
model_defs = [
    (
        model_shorthand_name,
        BudgetedResidualSplineRegressor(
            max_linear_features=18,
            max_nonlinear_terms=3,
            candidate_features=8,
            interaction_pool=4,
            knot_quantiles=(0.2, 0.4, 0.6, 0.8),
            selection_l2=5e-4,
            final_l2=2e-5,
            min_abs_score=5e-4,
            min_relative_improvement=4e-4,
            min_linear_rel_coef=0.015,
            nonlinear_prune_threshold=2e-4,
            inactive_rel_threshold=0.06,
            include_steps=True,
            include_hinge_neg=True,
            include_interactions=True,
        ),
    )
]


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
