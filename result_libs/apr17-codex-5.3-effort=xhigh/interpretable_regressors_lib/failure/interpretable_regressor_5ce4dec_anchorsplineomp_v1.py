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


class AnchorSplineRegressor(BaseEstimator, RegressorMixin):
    """Compact regression equation with greedy-selected raw terms.

    Candidate terms include:
    - linear: x_j
    - square: x_j^2
    - hinge: max(0, x_j - t), max(0, t - x_j)
    - low-order interactions: x_i * x_j
    """

    def __init__(
        self,
        max_terms=10,
        max_hinge_features=8,
        max_interaction_features=4,
        min_rel_gain=1e-4,
        ridge_lambda=1e-4,
    ):
        self.max_terms = max_terms
        self.max_hinge_features = max_hinge_features
        self.max_interaction_features = max_interaction_features
        self.min_rel_gain = min_rel_gain
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

    def _fit_ridge(self, X_design, y_centered):
        if X_design.shape[1] == 0:
            return np.zeros(0, dtype=float)
        n_terms = X_design.shape[1]
        gram = X_design.T @ X_design + self.ridge_lambda * np.eye(n_terms)
        rhs = X_design.T @ y_centered
        return np.linalg.solve(gram, rhs)

    def _build_candidates(self, X, ranked_features):
        n_features = X.shape[1]
        candidates = []

        for j in range(n_features):
            candidates.append(("linear", int(j)))

        hinge_features = ranked_features[: min(self.max_hinge_features, n_features)]
        for j in hinge_features:
            j = int(j)
            xj = X[:, j]
            knots = np.quantile(xj, [0.2, 0.5, 0.8])
            knots = np.unique(np.round(knots, 6))
            for t in knots:
                candidates.append(("pos_hinge", j, float(t)))
                candidates.append(("neg_hinge", j, float(t)))
            candidates.append(("square", j))

        interaction_features = ranked_features[: min(self.max_interaction_features, n_features)]
        for a in range(len(interaction_features)):
            for b in range(a + 1, len(interaction_features)):
                i = int(interaction_features[a])
                j = int(interaction_features[b])
                candidates.append(("interaction", i, j))
        return candidates

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        y_mean = float(np.mean(y))
        y_centered = y - y_mean

        # Rank features by direct relevance to seed nonlinear dictionary.
        feature_scores = np.array([self._safe_abs_corr(X[:, j], y_centered) for j in range(n_features)])
        ranked_features = np.argsort(feature_scores)[::-1]
        candidates = self._build_candidates(X, ranked_features)

        selected_terms = []
        selected_matrix = np.zeros((n_samples, 0), dtype=float)
        coef = np.zeros(0, dtype=float)

        baseline_sse = float(np.dot(y_centered, y_centered))
        current_sse = baseline_sse
        remaining = list(candidates)
        min_abs_gain = self.min_rel_gain * (baseline_sse + 1e-12)

        while len(selected_terms) < self.max_terms and remaining:
            residual = y_centered - (selected_matrix @ coef if coef.size else 0.0)

            best_idx = None
            best_score = -1.0
            for idx, term in enumerate(remaining):
                v = self._term_values(X, term)
                score = self._safe_abs_corr(v, residual)
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is None:
                break

            term = remaining.pop(best_idx)
            v = self._term_values(X, term)
            trial_matrix = np.column_stack([selected_matrix, v]) if selected_matrix.size else v.reshape(-1, 1)
            trial_coef = self._fit_ridge(trial_matrix, y_centered)
            trial_residual = y_centered - trial_matrix @ trial_coef
            trial_sse = float(np.dot(trial_residual, trial_residual))
            gain = current_sse - trial_sse
            if gain < min_abs_gain:
                break

            selected_terms.append(term)
            selected_matrix = trial_matrix
            coef = trial_coef
            current_sse = trial_sse

        if coef.size == 0:
            selected_terms = [("linear", int(ranked_features[0]))]
            selected_matrix = self._term_values(X, selected_terms[0]).reshape(-1, 1)
            coef = self._fit_ridge(selected_matrix, y_centered)

        keep = np.where(np.abs(coef) > 1e-6)[0]
        if keep.size == 0:
            keep = np.array([int(np.argmax(np.abs(coef)))])

        self.terms_ = [selected_terms[k] for k in keep]
        self.coef_ = coef[keep].astype(float)
        X_sel = np.column_stack([self._term_values(X, t) for t in self.terms_])
        self.intercept_ = float(np.mean(y - X_sel @ self.coef_))

        # Mean absolute per-feature contribution for readability diagnostics.
        feature_contrib = np.zeros(n_features, dtype=float)
        for c, t in zip(self.coef_, self.terms_):
            if t[0] == "linear" or t[0] == "square":
                feature_contrib[t[1]] += float(np.mean(np.abs(c * self._term_values(X, t))))
            elif t[0] == "pos_hinge" or t[0] == "neg_hinge":
                feature_contrib[t[1]] += float(np.mean(np.abs(c * self._term_values(X, t))))
            elif t[0] == "interaction":
                feature_contrib[t[1]] += 0.5 * float(np.mean(np.abs(c * self._term_values(X, t))))
                feature_contrib[t[2]] += 0.5 * float(np.mean(np.abs(c * self._term_values(X, t))))
        self.feature_importances_ = feature_contrib
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

        pieces = [f"{self.intercept_:+.6f}"]
        for c, t in zip(self.coef_, self.terms_):
            pieces.append(f"{c:+.6f}*({self._term_to_str(t)})")
        equation = " ".join(pieces)

        active_features = set()
        for t in self.terms_:
            if t[0] == "interaction":
                active_features.add(int(t[1]))
                active_features.add(int(t[2]))
            else:
                active_features.add(int(t[1]))
        inactive = [f"x{j}" for j in range(self.n_features_in_) if j not in active_features]

        order = np.argsort(self.feature_importances_)[::-1]
        ranked = [f"x{j}:{self.feature_importances_[j]:.4f}" for j in order[: min(10, self.n_features_in_)]]

        lines = [
            "Anchor Spline Regressor (compact symbolic equation)",
            f"Number of terms: {len(self.terms_)}",
            "Prediction formula:",
            f"  y = {equation}",
            "",
            "Most influential features by average absolute contribution:",
            "  " + ", ".join(ranked),
            "Inactive features with no direct term:",
            "  " + (", ".join(inactive) if inactive else "none"),
            "",
            "Term glossary:",
            "  max(0, xj - t) means positive hinge at threshold t.",
            "  max(0, t - xj) means negative hinge at threshold t.",
        ]
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
AnchorSplineRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "AnchorSplineOMP_v1"
model_description = "Forward-selected compact equation with linear, hinge, square, and limited interaction terms optimized for readable simulation"
model_defs = [(model_shorthand_name, AnchorSplineRegressor())]


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
