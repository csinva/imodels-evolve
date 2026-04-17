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


class SparseBasisPursuitRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse, interpretable basis-pursuit regressor.

    The model builds a dictionary of simple symbolic basis terms and greedily
    selects a compact subset that best explains y using ridge-refit each step.
    Term families:
      - linear: x_i
      - quadratic: x_i^2
      - hinges: max(0, x_i), max(0, -x_i)
      - absolute: |x_i|
      - pairwise interactions: x_i * x_j for top correlated features
    """

    def __init__(
        self,
        max_terms=8,
        min_improvement=1e-4,
        ridge_lambda=1e-4,
        max_interactions=6,
        include_quadratic=True,
        include_hinges=True,
        include_abs=True,
    ):
        self.max_terms = max_terms
        self.min_improvement = min_improvement
        self.ridge_lambda = ridge_lambda
        self.max_interactions = max_interactions
        self.include_quadratic = include_quadratic
        self.include_hinges = include_hinges
        self.include_abs = include_abs

    @staticmethod
    def _term_name(term):
        kind = term[0]
        if kind == "linear":
            return f"x{term[1]}"
        if kind == "square":
            return f"(x{term[1]}^2)"
        if kind == "hinge_pos":
            return f"max(0, x{term[1]})"
        if kind == "hinge_neg":
            return f"max(0, -x{term[1]})"
        if kind == "abs":
            return f"|x{term[1]}|"
        if kind == "interaction":
            return f"(x{term[1]} * x{term[2]})"
        raise ValueError(f"Unknown term kind: {kind}")

    @staticmethod
    def _term_values(X, term):
        kind = term[0]
        if kind == "linear":
            return X[:, term[1]]
        if kind == "square":
            z = X[:, term[1]]
            return z * z
        if kind == "hinge_pos":
            return np.maximum(0.0, X[:, term[1]])
        if kind == "hinge_neg":
            return np.maximum(0.0, -X[:, term[1]])
        if kind == "abs":
            return np.abs(X[:, term[1]])
        if kind == "interaction":
            return X[:, term[1]] * X[:, term[2]]
        raise ValueError(f"Unknown term kind: {kind}")

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        self.y_mean_ = float(np.mean(y))
        y_centered = y - self.y_mean_

        candidate_terms = [("linear", j) for j in range(n_features)]
        if self.include_quadratic:
            candidate_terms.extend(("square", j) for j in range(n_features))
        if self.include_hinges:
            candidate_terms.extend(("hinge_pos", j) for j in range(n_features))
            candidate_terms.extend(("hinge_neg", j) for j in range(n_features))
        if self.include_abs:
            candidate_terms.extend(("abs", j) for j in range(n_features))

        if self.max_interactions > 1 and n_features > 1:
            rel = []
            for j in range(n_features):
                col = X[:, j]
                if np.std(col) < 1e-12:
                    rel.append(0.0)
                    continue
                c = np.corrcoef(col, y_centered)[0, 1]
                rel.append(0.0 if np.isnan(c) else abs(c))
            top_k = min(max(2, self.max_interactions), n_features)
            top_idx = np.argsort(rel)[::-1][:top_k]
            for i in range(len(top_idx)):
                for j in range(i + 1, len(top_idx)):
                    candidate_terms.append(("interaction", int(top_idx[i]), int(top_idx[j])))

        candidate_cols = [self._term_values(X, t) for t in candidate_terms]
        if not candidate_cols:
            self.terms_ = []
            self.coef_ = np.zeros(0)
            self.intercept_ = self.y_mean_
            return self

        selected = []
        selected_coef = np.zeros(0)
        current_mse = float(np.mean(y_centered ** 2))
        residual = y_centered.copy()
        max_steps = min(self.max_terms, len(candidate_terms))

        for _ in range(max_steps):
            best_idx = None
            best_score = -np.inf
            for k, col in enumerate(candidate_cols):
                if k in selected:
                    continue
                denom = np.linalg.norm(col) + 1e-12
                score = abs(float(np.dot(residual, col))) / denom
                if score > best_score:
                    best_score = score
                    best_idx = k

            if best_idx is None:
                break

            trial = selected + [best_idx]
            B = np.column_stack([candidate_cols[t] for t in trial])
            gram = B.T @ B
            rhs = B.T @ y_centered
            ridge = self.ridge_lambda * np.eye(gram.shape[0])
            coef = np.linalg.solve(gram + ridge, rhs)
            pred = B @ coef
            trial_mse = float(np.mean((y_centered - pred) ** 2))

            gain = current_mse - trial_mse
            if gain < self.min_improvement * max(1.0, current_mse):
                break

            selected = trial
            selected_coef = coef
            current_mse = trial_mse
            residual = y_centered - pred

        self.terms_ = [candidate_terms[i] for i in selected]
        self.coef_ = np.asarray(selected_coef, dtype=float)
        self.intercept_ = self.y_mean_
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        X = np.asarray(X, dtype=float)
        if len(self.terms_) == 0:
            return np.full(X.shape[0], self.intercept_, dtype=float)
        B = np.column_stack([self._term_values(X, t) for t in self.terms_])
        return self.intercept_ + B @ self.coef_

    def __str__(self):
        check_is_fitted(self, "coef_")
        lines = [
            "Sparse Basis Pursuit Regressor",
            "Prediction rule:",
        ]
        if len(self.terms_) == 0:
            lines.append(f"y = {self.intercept_:.6f}")
            return "\n".join(lines)

        equation = [f"{self.intercept_:.6f}"]
        for c, t in zip(self.coef_, self.terms_):
            equation.append(f"{c:+.6f}*{self._term_name(t)}")
        lines.append("y = " + " ".join(equation))
        lines.append("")
        lines.append("Active terms:")
        for i, (c, t) in enumerate(zip(self.coef_, self.terms_), 1):
            lines.append(f"{i:2d}. coef={c:+.6f}   term={self._term_name(t)}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseBasisPursuitRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseBasisPursuitV1"
model_description = "Greedy sparse basis pursuit over linear, hinge, quadratic, abs, and limited interaction terms with explicit symbolic equation output"
model_defs = [(model_shorthand_name, SparseBasisPursuitRegressor())]


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
