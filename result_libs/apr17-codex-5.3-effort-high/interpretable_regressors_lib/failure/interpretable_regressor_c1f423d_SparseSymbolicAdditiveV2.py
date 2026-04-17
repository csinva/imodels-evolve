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


class SparseSymbolicAdditiveRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse symbolic additive regressor with greedy forward selection.

    Candidate basis terms:
      - linear: x_j
      - quadratic: x_j^2
      - absolute: |x_j|
      - hinge at zero: max(0, x_j), max(0, -x_j)
      - hinge at median: max(0, x_j - median_j)
      - optional interaction: x_j * x_k (for top correlated features)
    """

    def __init__(
        self,
        max_terms=7,
        ridge_lambda=1e-2,
        min_rel_gain=5e-4,
        top_interaction_features=4,
        include_interactions=True,
    ):
        self.max_terms = max_terms
        self.ridge_lambda = ridge_lambda
        self.min_rel_gain = min_rel_gain
        self.top_interaction_features = top_interaction_features
        self.include_interactions = include_interactions

    @staticmethod
    def _ridge_fit(B, y, lam):
        if B.shape[1] == 0:
            pred = np.zeros(B.shape[0], dtype=float)
            return np.zeros(0, dtype=float), pred, float(np.mean((y - pred) ** 2))
        p = B.shape[1]
        eye = np.sqrt(max(lam, 1e-12)) * np.eye(p)
        A = np.vstack([B, eye])
        b = np.concatenate([y, np.zeros(p, dtype=float)])
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        pred = B @ coef
        mse = float(np.mean((y - pred) ** 2))
        return coef, pred, mse

    @staticmethod
    def _term_column(X, term):
        kind = term["kind"]
        if kind == "lin":
            return X[:, term["j"]]
        if kind == "sq":
            xj = X[:, term["j"]]
            return xj * xj
        if kind == "abs":
            return np.abs(X[:, term["j"]])
        if kind == "hinge_pos":
            return np.maximum(0.0, X[:, term["j"]])
        if kind == "hinge_neg":
            return np.maximum(0.0, -X[:, term["j"]])
        if kind == "hinge_med":
            xj = X[:, term["j"]]
            return np.maximum(0.0, xj - term["knot"])
        if kind == "prod":
            return X[:, term["j"]] * X[:, term["k"]]
        raise ValueError(f"Unknown term kind: {kind}")

    @staticmethod
    def _term_expr(term):
        kind = term["kind"]
        j = term.get("j", -1)
        if kind == "lin":
            return f"x{j}"
        if kind == "sq":
            return f"(x{j}^2)"
        if kind == "abs":
            return f"abs(x{j})"
        if kind == "hinge_pos":
            return f"max(0, x{j})"
        if kind == "hinge_neg":
            return f"max(0, -x{j})"
        if kind == "hinge_med":
            return f"max(0, x{j}-{term['knot']:.6f})"
        if kind == "prod":
            return f"(x{term['j']}*x{term['k']})"
        return str(term)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        self.intercept_ = float(np.mean(y))
        yc = y - self.intercept_
        medians = np.median(X, axis=0)

        candidates = []
        for j in range(n_features):
            candidates.append({"kind": "lin", "j": int(j)})
            candidates.append({"kind": "sq", "j": int(j)})
            candidates.append({"kind": "abs", "j": int(j)})
            candidates.append({"kind": "hinge_pos", "j": int(j)})
            candidates.append({"kind": "hinge_neg", "j": int(j)})
            candidates.append({"kind": "hinge_med", "j": int(j), "knot": float(medians[j])})

        if self.include_interactions and n_features >= 2:
            corr = np.abs(X.T @ yc) / max(1.0, np.linalg.norm(yc))
            top_k = min(self.top_interaction_features, n_features)
            top = np.argsort(corr)[-top_k:]
            for ai in range(len(top)):
                for bi in range(ai + 1, len(top)):
                    j = int(top[ai])
                    k = int(top[bi])
                    candidates.append({"kind": "prod", "j": j, "k": k})

        selected_idx = []
        selected_cols = []
        current_mse = float(np.mean(yc ** 2))

        max_steps = min(self.max_terms, len(candidates))
        for _ in range(max_steps):
            best_idx = None
            best_coef = None
            best_pred = None
            best_mse = current_mse

            for i, term in enumerate(candidates):
                if i in selected_idx:
                    continue
                col = self._term_column(X, term)
                if not np.all(np.isfinite(col)):
                    continue
                trial_cols = selected_cols + [col]
                B = np.column_stack(trial_cols)
                coef, pred, mse = self._ridge_fit(B, yc, self.ridge_lambda)
                if mse < best_mse:
                    best_mse = mse
                    best_idx = i
                    best_coef = coef
                    best_pred = pred

            if best_idx is None:
                break

            rel_gain = (current_mse - best_mse) / max(current_mse, 1e-12)
            if rel_gain < self.min_rel_gain:
                break

            selected_idx.append(best_idx)
            selected_cols.append(self._term_column(X, candidates[best_idx]))
            current_mse = best_mse
            final_coef = best_coef
            final_pred = best_pred

        self.selected_terms_ = [candidates[i] for i in selected_idx]
        self.coef_ = np.asarray(final_coef if selected_idx else np.zeros(0), dtype=float)
        self.train_mse_ = current_mse
        self._last_train_pred_ = np.asarray(final_pred if selected_idx else np.zeros(n_samples), dtype=float)
        return self

    def _design(self, X):
        X = np.asarray(X, dtype=float)
        if not self.selected_terms_:
            return np.zeros((X.shape[0], 0))
        cols = [self._term_column(X, term) for term in self.selected_terms_]
        return np.column_stack(cols)

    def predict(self, X):
        check_is_fitted(self, ["selected_terms_", "coef_", "intercept_"])
        X = np.asarray(X, dtype=float)
        B = self._design(X)
        if B.shape[1] == 0:
            return np.full(X.shape[0], self.intercept_, dtype=float)
        return self.intercept_ + B @ self.coef_

    def __str__(self):
        check_is_fitted(self, ["selected_terms_", "coef_", "intercept_"])
        lines = ["Sparse Symbolic Additive Regressor", "Prediction rule:"]
        if len(self.selected_terms_) == 0:
            lines.append(f"y = {self.intercept_:.6f}")
            return "\n".join(lines)

        expr_parts = [f"{self.intercept_:.6f}"]
        lines.append("y = intercept + sum(term contributions)")
        lines.append("")
        lines.append(f"intercept = {self.intercept_:.6f}")
        lines.append("active terms:")

        for c, t in zip(self.coef_, self.selected_terms_):
            term_expr = self._term_expr(t)
            expr_parts.append(f"{c:+.6f}*{term_expr}")
            lines.append(f"  {c:+.6f} * {term_expr}")

        lines.append("")
        lines.append("equation:")
        lines.append("y = " + " ".join(expr_parts))
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseSymbolicAdditiveRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseSymbolicAdditiveV2"
model_description = "Greedy forward-selected sparse symbolic additive model over linear/quadratic/abs/hinge terms plus limited pairwise interactions with explicit equation output"
model_defs = [(model_shorthand_name, SparseSymbolicAdditiveRegressor())]


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
