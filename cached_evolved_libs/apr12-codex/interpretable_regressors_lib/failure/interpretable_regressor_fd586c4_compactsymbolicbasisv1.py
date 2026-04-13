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


class CompactSymbolicBasisRegressor(BaseEstimator, RegressorMixin):
    """
    Compact symbolic additive model:
      y = intercept + sum_k coef_k * phi_k(x)
    where each basis term phi_k is chosen greedily from a small library of
    human-readable transformations.
    """

    def __init__(
        self,
        max_terms=6,
        candidate_features=6,
        ridge_alpha=1e-3,
        min_gain=1e-4,
        coef_tol=1e-7,
    ):
        self.max_terms = max_terms
        self.candidate_features = candidate_features
        self.ridge_alpha = ridge_alpha
        self.min_gain = min_gain
        self.coef_tol = coef_tol

    def _impute(self, X):
        X = np.asarray(X, dtype=float).copy()
        mask = ~np.isfinite(X)
        if mask.any():
            X[mask] = np.take(self.feature_medians_, np.where(mask)[1])
        return X

    @staticmethod
    def _evaluate_term(X, term):
        kind = term["kind"]
        if kind == "linear":
            return X[:, term["j"]]
        if kind == "square":
            z = X[:, term["j"]]
            return z * z
        if kind == "abs":
            return np.abs(X[:, term["j"]])
        if kind == "hinge_pos":
            return np.maximum(0.0, X[:, term["j"]] - term["t"])
        if kind == "hinge_neg":
            return np.maximum(0.0, term["t"] - X[:, term["j"]])
        if kind == "interaction":
            return X[:, term["j"]] * X[:, term["k"]]
        raise ValueError(f"Unknown term kind: {kind}")

    def _build_candidates(self, X, y_center):
        p = X.shape[1]
        corr = np.abs(np.mean(X * y_center[:, None], axis=0))
        feat_order = np.argsort(corr)[::-1]
        top_feats = feat_order[: min(self.candidate_features, p)].tolist()
        if not top_feats:
            top_feats = [0]

        terms = []
        seen = set()

        def add(term):
            key = tuple(sorted(term.items()))
            if key not in seen:
                seen.add(key)
                terms.append(term)

        for j in top_feats:
            xj = X[:, j]
            t = float(np.median(xj))
            add({"kind": "linear", "j": int(j)})
            add({"kind": "square", "j": int(j)})
            add({"kind": "abs", "j": int(j)})
            add({"kind": "hinge_pos", "j": int(j), "t": t})
            add({"kind": "hinge_neg", "j": int(j), "t": t})

        for i in range(min(3, len(top_feats))):
            for j in range(i + 1, min(3, len(top_feats))):
                add({"kind": "interaction", "j": int(top_feats[i]), "k": int(top_feats[j])})

        valid_terms = []
        columns = []
        for term in terms:
            z = self._evaluate_term(X, term)
            if float(np.std(z)) > 1e-10:
                valid_terms.append(term)
                columns.append(z)

        Z = np.column_stack(columns) if columns else np.zeros((X.shape[0], 0), dtype=float)
        return valid_terms, Z

    def _ridge_fit_centered(self, A, y):
        if A.shape[1] == 0:
            intercept = float(np.mean(y))
            return intercept, np.zeros(0, dtype=float), np.zeros(0, dtype=float)

        y_mean = float(np.mean(y))
        y_c = y - y_mean
        a_mean = np.mean(A, axis=0)
        A_c = A - a_mean
        gram = A_c.T @ A_c
        rhs = A_c.T @ y_c
        coef = np.linalg.solve(gram + self.ridge_alpha * np.eye(A_c.shape[1]), rhs)
        intercept = y_mean - float(np.dot(a_mean, coef))
        return intercept, coef, a_mean

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]

        self.feature_medians_ = np.nanmedian(X, axis=0)
        X = self._impute(X)

        y_center = y - float(np.mean(y))
        self.candidate_terms_, Z = self._build_candidates(X, y_center)

        selected = []
        remaining = list(range(len(self.candidate_terms_)))
        best_mse = float(np.mean((y - np.mean(y)) ** 2))
        best_intercept = float(np.mean(y))
        best_coef = np.zeros(0, dtype=float)

        for _ in range(self.max_terms):
            if not remaining:
                break
            residual = y - (best_intercept + (Z[:, selected] @ best_coef if selected else 0.0))
            best_idx = None
            best_score = -1.0
            for idx in remaining:
                z = Z[:, idx]
                zc = z - float(np.mean(z))
                denom = float(np.sqrt(np.mean(zc * zc))) + 1e-12
                score = abs(float(np.mean(zc * residual))) / denom
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx is None:
                break

            trial_selected = selected + [best_idx]
            A = Z[:, trial_selected]
            intercept, coef, _ = self._ridge_fit_centered(A, y)
            pred = intercept + A @ coef
            mse = float(np.mean((y - pred) ** 2))

            if (best_mse - mse) < self.min_gain:
                break
            selected = trial_selected
            remaining.remove(best_idx)
            best_mse = mse
            best_intercept = intercept
            best_coef = coef

        if not selected and Z.shape[1] > 0:
            scores = np.abs(np.mean((Z - np.mean(Z, axis=0)) * y_center[:, None], axis=0))
            selected = [int(np.argmax(scores))]
            A = Z[:, selected]
            best_intercept, best_coef, _ = self._ridge_fit_centered(A, y)

        self.intercept_ = float(best_intercept)
        self.terms_ = []
        self.coef_ = []
        for idx, coef in zip(selected, best_coef):
            if abs(float(coef)) > self.coef_tol:
                self.terms_.append(self.candidate_terms_[idx])
                self.coef_.append(float(coef))
        self.coef_ = np.array(self.coef_, dtype=float)

        if self.terms_:
            A = np.column_stack([self._evaluate_term(X, t) for t in self.terms_])
            self.intercept_, self.coef_, _ = self._ridge_fit_centered(A, y)
        else:
            self.intercept_ = float(np.mean(y))
            self.coef_ = np.zeros(0, dtype=float)

        fi = np.zeros(self.n_features_in_, dtype=float)
        for term, coef in zip(self.terms_, self.coef_):
            mag = abs(float(coef))
            fi[term["j"]] += mag
            if term["kind"] == "interaction":
                fi[term["k"]] += mag
        self.feature_importance_ = fi
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "terms_", "coef_", "feature_importance_"])
        X = self._impute(X)
        y_hat = np.full(X.shape[0], self.intercept_, dtype=float)
        for term, coef in zip(self.terms_, self.coef_):
            y_hat += coef * self._evaluate_term(X, term)
        return y_hat

    @staticmethod
    def _term_to_str(term):
        kind = term["kind"]
        j = term["j"]
        if kind == "linear":
            return f"x{j}"
        if kind == "square":
            return f"(x{j}^2)"
        if kind == "abs":
            return f"|x{j}|"
        if kind == "hinge_pos":
            return f"max(0, x{j} - {term['t']:.4f})"
        if kind == "hinge_neg":
            return f"max(0, {term['t']:.4f} - x{j})"
        if kind == "interaction":
            return f"(x{j} * x{term['k']})"
        return str(term)

    def __str__(self):
        check_is_fitted(self, ["intercept_", "terms_", "coef_", "feature_importance_"])
        lines = [
            "CompactSymbolicBasisRegressor",
            "Prediction equation:",
            f"y = {self.intercept_:+.5f}",
        ]
        order = np.argsort(np.abs(self.coef_))[::-1] if self.coef_.size else []
        for idx in order:
            lines.append(f"  {self.coef_[idx]:+.5f} * {self._term_to_str(self.terms_[idx])}")

        lines.append("")
        lines.append(f"Model size: {len(self.terms_)} symbolic terms")
        lines.append("Active features (by total absolute coefficient contribution):")
        active = [j for j in np.argsort(self.feature_importance_)[::-1]
                  if self.feature_importance_[j] > self.coef_tol]
        for j in active:
            lines.append(f"  x{j}: {self.feature_importance_[j]:.5f}")
        inactive = [f"x{j}" for j in range(self.n_features_in_) if self.feature_importance_[j] <= self.coef_tol]
        if inactive:
            lines.append(f"Zero/near-zero features: {', '.join(inactive)}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
CompactSymbolicBasisRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "CompactSymbolicBasisV1"
model_description = "Greedy-selected compact symbolic additive regressor using linear, square, absolute, hinge, and sparse interaction basis terms with ridge refit"
model_defs = [(model_shorthand_name, CompactSymbolicBasisRegressor())]


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
