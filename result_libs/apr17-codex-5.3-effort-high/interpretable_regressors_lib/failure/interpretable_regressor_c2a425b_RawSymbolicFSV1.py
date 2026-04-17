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


class RawSymbolicRegressor(BaseEstimator, RegressorMixin):
    """
    Compact symbolic regressor on raw x-features.

    Candidate basis terms:
      - linear: x_j
      - hinge+: max(0, x_j)
      - hinge-: max(0, -x_j)
      - abs: |x_j|
      - interaction: x_j * x_k
    """

    def __init__(
        self,
        max_terms=8,
        ridge_lambda=1e-2,
        min_rel_gain=1e-3,
        screen_features=12,
        max_interaction_features=4,
    ):
        self.max_terms = max_terms
        self.ridge_lambda = ridge_lambda
        self.min_rel_gain = min_rel_gain
        self.screen_features = screen_features
        self.max_interaction_features = max_interaction_features

    @staticmethod
    def _ridge_fit(B, y, lam):
        n = B.shape[0]
        if B.shape[1] == 0:
            intercept = float(np.mean(y))
            return intercept, np.zeros(0, dtype=float), np.full(n, intercept, dtype=float)

        Xd = np.hstack([np.ones((n, 1), dtype=float), B])
        p = Xd.shape[1]
        reg = np.sqrt(max(lam, 1e-12)) * np.eye(p, dtype=float)
        reg[0, 0] = 0.0
        A = np.vstack([Xd, reg])
        b = np.concatenate([y, np.zeros(p, dtype=float)])
        beta, *_ = np.linalg.lstsq(A, b, rcond=None)
        intercept = float(beta[0])
        coef = np.asarray(beta[1:], dtype=float)
        pred = intercept + B @ coef
        return intercept, coef, pred

    @staticmethod
    def _term_expr(term):
        kind = term["kind"]
        j = term.get("j")
        if kind == "lin":
            return f"x{j}"
        if kind == "hinge_pos":
            return f"max(0, x{j})"
        if kind == "hinge_neg":
            return f"max(0, -x{j})"
        if kind == "abs":
            return f"|x{j}|"
        if kind == "int":
            return f"(x{term['j']}*x{term['k']})"
        return str(term)

    def _term_col(self, X, term):
        kind = term["kind"]
        if kind == "lin":
            return X[:, term["j"]]
        if kind == "hinge_pos":
            return np.maximum(0.0, X[:, term["j"]])
        if kind == "hinge_neg":
            return np.maximum(0.0, -X[:, term["j"]])
        if kind == "abs":
            return np.abs(X[:, term["j"]])
        if kind == "int":
            return X[:, term["j"]] * X[:, term["k"]]
        raise ValueError(f"Unknown term kind: {kind}")

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        yc = y - float(np.mean(y))
        corr = np.abs(X.T @ yc)
        keep = min(n_features, max(1, self.screen_features))
        main_feats = np.argsort(corr)[-keep:]
        main_feats = np.sort(main_feats.astype(int))
        self.screened_features_ = main_feats

        candidates = []
        for j in main_feats:
            candidates.append({"kind": "lin", "j": int(j)})
            candidates.append({"kind": "hinge_pos", "j": int(j)})
            candidates.append({"kind": "hinge_neg", "j": int(j)})
            candidates.append({"kind": "abs", "j": int(j)})

        top_int = main_feats[np.argsort(corr[main_feats])[::-1][:min(len(main_feats), self.max_interaction_features)]]
        for i in range(len(top_int)):
            for j in range(i + 1, len(top_int)):
                candidates.append({"kind": "int", "j": int(top_int[i]), "k": int(top_int[j])})

        if not candidates:
            self.selected_terms_ = []
            self.coef_ = np.zeros(0, dtype=float)
            self.intercept_ = float(np.mean(y))
            return self

        candidate_cols = [self._term_col(X, t) for t in candidates]
        selected = []
        current_pred = np.full(n_samples, np.mean(y), dtype=float)
        current_mse = float(np.mean((y - current_pred) ** 2))
        best_intercept = float(np.mean(y))
        best_coef = np.zeros(0, dtype=float)

        for _ in range(min(self.max_terms, len(candidates))):
            residual = y - current_pred
            best_idx = None
            best_score = -np.inf
            for i, col in enumerate(candidate_cols):
                if i in selected:
                    continue
                score = abs(float(np.dot(residual, col))) / (np.linalg.norm(col) + 1e-12)
                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx is None:
                break

            trial_idx = selected + [best_idx]
            B = np.column_stack([candidate_cols[t] for t in trial_idx])
            intercept, coef, pred = self._ridge_fit(B, y, self.ridge_lambda)
            trial_mse = float(np.mean((y - pred) ** 2))
            rel_gain = (current_mse - trial_mse) / max(current_mse, 1e-12)
            if rel_gain < self.min_rel_gain:
                break

            selected = trial_idx
            current_pred = pred
            current_mse = trial_mse
            best_intercept = intercept
            best_coef = coef

        self.selected_terms_ = [candidates[i] for i in selected]
        self.coef_ = np.asarray(best_coef, dtype=float)
        self.intercept_ = float(best_intercept)
        self.train_mse_ = float(current_mse)
        return self

    def predict(self, X):
        check_is_fitted(self, ["selected_terms_", "coef_", "intercept_"])
        X = np.asarray(X, dtype=float)
        preds = np.full(X.shape[0], self.intercept_, dtype=float)
        if not self.selected_terms_:
            return preds
        B = np.column_stack([self._term_col(X, t) for t in self.selected_terms_])
        preds += B @ self.coef_
        return preds

    def __str__(self):
        check_is_fitted(self, ["selected_terms_", "coef_", "intercept_"])
        lines = ["Raw Symbolic Regressor", "Prediction rule (compute directly from x-values):"]

        pieces = [f"{self.intercept_:.6f}"]
        for c, t in zip(self.coef_, self.selected_terms_):
            pieces.append(f"{c:+.6f}*{self._term_expr(t)}")
        lines.append("  y = " + " ".join(pieces))

        used = set()
        for t in self.selected_terms_:
            used.add(int(t["j"]))
            if t["kind"] == "int":
                used.add(int(t["k"]))
        inactive = [f"x{j}" for j in range(self.n_features_in_) if j not in used]

        lines.append("")
        lines.append("Active terms:")
        for i, (c, t) in enumerate(zip(self.coef_, self.selected_terms_), 1):
            lines.append(f"  {i:2d}. {c:+.6f} * {self._term_expr(t)}")
        lines.append("")
        lines.append("Inactive features (no direct effect):")
        lines.append("  " + (", ".join(inactive) if inactive else "none"))
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
RawSymbolicRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "RawSymbolicFSV1"
model_description = "Forward-selected compact symbolic model on raw features using linear, signed hinge, absolute-value, and sparse pairwise interaction terms"
model_defs = [(model_shorthand_name, RawSymbolicRegressor())]


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
