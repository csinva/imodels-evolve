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


class CompactResidualAdditiveRegressor(BaseEstimator, RegressorMixin):
    """
    Compact sparse additive regressor.

    Design goals:
      - Keep a short closed-form equation that is easy for an LLM to simulate.
      - Preserve predictive performance via a residual-greedy search over a small
        nonlinear library (hinges/quadratics/interactions) when they add value.

    The model always exposes a direct sum of active terms plus an intercept.
    """

    def __init__(
        self,
        max_terms=8,
        nonlinear_features=6,
        interaction_features=4,
        selection_l2=1e-4,
        final_l2=1e-6,
        min_abs_score=1e-3,
        min_relative_improvement=5e-4,
        prune_threshold=2e-4,
    ):
        self.max_terms = max_terms
        self.nonlinear_features = nonlinear_features
        self.interaction_features = interaction_features
        self.selection_l2 = selection_l2
        self.final_l2 = final_l2
        self.min_abs_score = min_abs_score
        self.min_relative_improvement = min_relative_improvement
        self.prune_threshold = prune_threshold

    def _safe_center_corr(self, a, b):
        ac = a - float(np.mean(a))
        bc = b - float(np.mean(b))
        denom = float(np.linalg.norm(ac) * np.linalg.norm(bc))
        if denom <= 1e-12:
            return 0.0
        return float(np.dot(ac, bc) / denom)

    def _eval_term(self, Z, term):
        kind = term[0]
        if kind == "linear":
            j = term[1]
            return Z[:, j]
        if kind == "relu_pos":
            j = term[1]
            return np.maximum(0.0, Z[:, j])
        if kind == "relu_neg":
            j = term[1]
            return np.maximum(0.0, -Z[:, j])
        if kind == "quadratic":
            j = term[1]
            col = Z[:, j] ** 2
            return col - float(np.mean(col))
        if kind == "interaction":
            a, b = term[1], term[2]
            return Z[:, a] * Z[:, b]
        raise ValueError(f"Unknown term: {term}")

    def _term_to_str(self, term):
        kind = term[0]
        if kind == "linear":
            return f"z{term[1]}"
        if kind == "relu_pos":
            return f"max(0, z{term[1]})"
        if kind == "relu_neg":
            return f"max(0, -z{term[1]})"
        if kind == "quadratic":
            return f"(z{term[1]}^2 - mean(z{term[1]}^2))"
        if kind == "interaction":
            return f"(z{term[1]} * z{term[2]})"
        return str(term)

    def _build_library(self, Z, y_centered):
        n_samples, n_features = Z.shape
        terms = []
        cols = []

        # Always include a linear backbone.
        for j in range(n_features):
            terms.append(("linear", j))
            cols.append(Z[:, j])

        # Rank features by signal before adding nonlinear candidates.
        signal = np.array([abs(self._safe_center_corr(Z[:, j], y_centered)) for j in range(n_features)])
        top_nonlin = np.argsort(signal)[::-1][: min(self.nonlinear_features, n_features)]

        for j in top_nonlin:
            terms.append(("relu_pos", int(j)))
            cols.append(np.maximum(0.0, Z[:, j]))

            terms.append(("relu_neg", int(j)))
            cols.append(np.maximum(0.0, -Z[:, j]))

            quad = Z[:, j] ** 2
            terms.append(("quadratic", int(j)))
            cols.append(quad - float(np.mean(quad)))

        top_inter = np.argsort(signal)[::-1][: min(self.interaction_features, n_features)]
        for a, b in combinations(top_inter, 2):
            terms.append(("interaction", int(a), int(b)))
            cols.append(Z[:, a] * Z[:, b])

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

        # z-score features for stable sparse search and transferable thresholding.
        self.x_mean_ = X.mean(axis=0)
        self.x_scale_ = X.std(axis=0)
        self.x_scale_[self.x_scale_ < 1e-12] = 1.0
        Z = (X - self.x_mean_) / self.x_scale_

        self.y_mean_ = float(np.mean(y))
        y_centered = y - self.y_mean_

        terms, Phi = self._build_library(Z, y_centered)
        if Phi.shape[1] == 0:
            self.intercept_ = self.y_mean_
            self.coef_ = np.array([], dtype=float)
            self.terms_ = []
            self.feature_importance_ = np.zeros(n_features, dtype=float)
            self.inactive_features_ = [f"x{i}" for i in range(n_features)]
            return self

        # Normalize candidate columns for fair greedy scoring.
        phi_mean = Phi.mean(axis=0)
        phi_std = Phi.std(axis=0) + 1e-12
        Phi_norm = (Phi - phi_mean) / phi_std

        selected = []
        residual = y_centered.copy()
        prev_rss = float(residual @ residual)

        for _ in range(self.max_terms):
            remaining = [j for j in range(Phi_norm.shape[1]) if j not in selected]
            if not remaining:
                break

            # Score by absolute correlation with residual.
            scores = np.abs(Phi_norm[:, remaining].T @ residual) / max(1, n_samples - 1)
            best_pos = int(np.argmax(scores))
            best_j = int(remaining[best_pos])
            best_score = float(scores[best_pos])
            if best_score < self.min_abs_score:
                break

            trial = selected + [best_j]
            B = Phi_norm[:, trial]
            gram = B.T @ B + self.selection_l2 * np.eye(B.shape[1], dtype=float)
            rhs = B.T @ y_centered
            try:
                beta = np.linalg.solve(gram, rhs)
            except np.linalg.LinAlgError:
                beta = np.linalg.lstsq(B, y_centered, rcond=None)[0]

            new_residual = y_centered - B @ beta
            rss = float(new_residual @ new_residual)
            rel_gain = (prev_rss - rss) / (prev_rss + 1e-12)
            if rel_gain < self.min_relative_improvement:
                break

            selected = trial
            residual = new_residual
            prev_rss = rss

        if not selected:
            # Fallback: keep strongest linear term.
            linear_scores = np.array([abs(self._safe_center_corr(Z[:, j], y_centered)) for j in range(n_features)])
            fallback_feature = int(np.argmax(linear_scores))
            selected = [terms.index(("linear", fallback_feature))]

        A = Phi[:, selected]

        def _ridge_refit(A_local):
            A_aug = np.column_stack([np.ones(n_samples, dtype=float), A_local])
            reg = np.diag([0.0] + [self.final_l2] * A_local.shape[1]).astype(float)
            lhs = A_aug.T @ A_aug + reg
            rhs = A_aug.T @ y
            try:
                beta = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                beta = np.linalg.lstsq(A_aug, y, rcond=None)[0]
            return float(beta[0]), np.asarray(beta[1:], dtype=float)

        intercept, coefs = _ridge_refit(A)

        # Prune tiny terms, then refit for cleaner equations.
        keep = np.abs(coefs) >= self.prune_threshold
        if np.any(keep):
            selected = [selected[i] for i, k in enumerate(keep) if k]
            A = Phi[:, selected]
            intercept, coefs = _ridge_refit(A)
        else:
            strongest = int(np.argmax(np.abs(coefs)))
            selected = [selected[strongest]]
            A = Phi[:, selected]
            intercept, coefs = _ridge_refit(A)

        self.intercept_ = float(intercept)
        self.coef_ = np.asarray(coefs, dtype=float)
        self.terms_ = [terms[j] for j in selected]

        # Attribute nonlinear feature usage back to original x_j.
        importance = np.zeros(n_features, dtype=float)
        for c, t in zip(self.coef_, self.terms_):
            mag = float(abs(c))
            if t[0] == "interaction":
                importance[t[1]] += 0.5 * mag
                importance[t[2]] += 0.5 * mag
            else:
                importance[t[1]] += mag
        self.feature_importance_ = importance
        self.inactive_features_ = [
            f"x{i}" for i in range(n_features) if importance[i] < 1e-10
        ]
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "coef_", "terms_", "x_mean_", "x_scale_", "n_features_in_"])

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")

        Z = (X - self.x_mean_) / self.x_scale_
        preds = np.full(X.shape[0], self.intercept_, dtype=float)
        for c, t in zip(self.coef_, self.terms_):
            preds += c * self._eval_term(Z, t)
        return preds

    def __str__(self):
        check_is_fitted(self, ["intercept_", "coef_", "terms_", "x_mean_", "x_scale_", "feature_importance_"])

        lines = [
            "Compact Residual Additive Regressor",
            "Definition of normalized variables: z_j = (x_j - mean_j) / scale_j",
            "Prediction uses a short additive equation with active terms only.",
            f"intercept = {self.intercept_:+.6f}",
            "",
            "Active terms (add each term's contribution):",
        ]

        equation = [f"y = {self.intercept_:+.6f}"]
        for c, t in zip(self.coef_, self.terms_):
            t_str = self._term_to_str(t)
            lines.append(f"  {c:+.6f} * {t_str}")
            equation.append(f"{c:+.6f} * {t_str}")
        lines.append("")
        lines.append("Equation:")
        lines.append("  " + " ".join(equation))

        order = np.argsort(self.feature_importance_)[::-1]
        used = [f"x{i} ({self.feature_importance_[i]:.3f})" for i in order if self.feature_importance_[i] > 1e-10]
        if used:
            lines.append("")
            lines.append("Feature influence ranking: " + ", ".join(used))

        if self.inactive_features_:
            lines.append("Features with near-zero influence: " + ", ".join(self.inactive_features_))

        # Include normalization constants only for active features to keep the string compact.
        active_idx = sorted({t[1] for t in self.terms_ if t[0] != "interaction"} | {i for t in self.terms_ if t[0] == "interaction" for i in (t[1], t[2])})
        if active_idx:
            lines.append("")
            lines.append("Normalization constants for active features:")
            for j in active_idx:
                lines.append(f"  z{j}: mean={self.x_mean_[j]:+.4f}, scale={self.x_scale_[j]:.4f}")

        lines.append("")
        lines.append(f"Compactness: {len(self.terms_)} active terms.")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
CompactResidualAdditiveRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "CompactResidualAdditiveApr18b"
model_description = (
    "Sparse residual additive regressor: compact normalized linear backbone with "
    "selective hinge/quadratic/interaction corrections via greedy residual fitting."
)
model_defs = [
    (
        model_shorthand_name,
        CompactResidualAdditiveRegressor(
            max_terms=7,
            nonlinear_features=5,
            interaction_features=3,
            min_relative_improvement=8e-4,
            prune_threshold=5e-4,
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
