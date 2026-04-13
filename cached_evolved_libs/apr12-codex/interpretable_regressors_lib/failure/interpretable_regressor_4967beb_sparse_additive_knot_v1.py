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


class SparseAdditiveKnotRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse additive one-knot model:
      y = intercept + sum_j [a_j * x_j + b_j * max(0, x_j - t_j)]

    Each selected feature gets at most one hinge with learned threshold t_j.
    This keeps the equation compact and directly simulatable while allowing
    threshold-like nonlinear behavior.
    """

    def __init__(
        self,
        max_active_features=4,
        candidate_pool=10,
        threshold_quantiles=(0.15, 0.3, 0.5, 0.7, 0.85),
        min_gain=2e-3,
        ridge_alpha=3e-2,
        coef_tol=2e-3,
        random_state=42,
    ):
        self.max_active_features = max_active_features
        self.candidate_pool = candidate_pool
        self.threshold_quantiles = threshold_quantiles
        self.min_gain = min_gain
        self.ridge_alpha = ridge_alpha
        self.coef_tol = coef_tol
        self.random_state = random_state

    def _impute(self, X):
        X = np.asarray(X, dtype=float).copy()
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(self.feature_medians_, np.where(bad)[1])
        return X

    @staticmethod
    def _abs_corr(x, y):
        xc = x - np.mean(x)
        yc = y - np.mean(y)
        den = (np.linalg.norm(xc) * np.linalg.norm(yc)) + 1e-12
        return abs(float(np.dot(xc, yc) / den))

    def _solve_ridge(self, A, y):
        gram = A.T @ A
        reg = float(self.ridge_alpha) * np.eye(gram.shape[0])
        reg[0, 0] = 0.0
        return np.linalg.solve(gram + reg, A.T @ y)

    def _feature_best_atom(self, xj, residual):
        best = None
        base_mse = float(np.mean(residual ** 2))
        for q in self.threshold_quantiles:
            t = float(np.quantile(xj, q))
            h = np.maximum(0.0, xj - t)
            A = np.column_stack([np.ones(xj.shape[0]), xj, h])
            beta = self._solve_ridge(A, residual)
            fit = A @ beta
            mse = float(np.mean((residual - fit) ** 2))
            gain = base_mse - mse
            if (best is None) or (gain > best["gain"]):
                best = {
                    "threshold": t,
                    "lin_coef": float(beta[1]),
                    "hinge_coef": float(beta[2]),
                    "gain": gain,
                }
        return best

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.feature_medians_ = np.nanmedian(X, axis=0)
        X = self._impute(X)

        p = X.shape[1]
        scores = np.array([self._abs_corr(X[:, j], y) for j in range(p)])
        pool = np.argsort(scores)[::-1][: min(int(self.candidate_pool), p)]

        intercept = float(np.mean(y))
        residual = y - intercept

        selected = []
        used = set()
        for _ in range(min(int(self.max_active_features), len(pool))):
            best_j = None
            best_atom = None
            for j in pool:
                j = int(j)
                if j in used:
                    continue
                atom = self._feature_best_atom(X[:, j], residual)
                if atom is None:
                    continue
                if (best_atom is None) or (atom["gain"] > best_atom["gain"]):
                    best_atom = atom
                    best_j = j

            if best_j is None or best_atom is None or best_atom["gain"] < self.min_gain:
                break

            used.add(best_j)
            selected.append({
                "feature": int(best_j),
                "threshold": float(best_atom["threshold"]),
                "lin_coef": float(best_atom["lin_coef"]),
                "hinge_coef": float(best_atom["hinge_coef"]),
            })

            xj = X[:, best_j]
            residual = residual - (
                best_atom["lin_coef"] * xj +
                best_atom["hinge_coef"] * np.maximum(0.0, xj - best_atom["threshold"])
            )

        # Joint refit for selected atoms for stable coefficients.
        A_cols = [np.ones(X.shape[0])]
        atom_meta = []
        for atom in selected:
            j = atom["feature"]
            t = atom["threshold"]
            xj = X[:, j]
            hj = np.maximum(0.0, xj - t)
            A_cols.append(xj)
            A_cols.append(hj)
            atom_meta.append((j, t))

        A = np.column_stack(A_cols)
        beta = self._solve_ridge(A, y)

        self.intercept_ = float(beta[0])
        self.terms_ = []
        idx = 1
        for j, t in atom_meta:
            lin_c = float(beta[idx])
            h_c = float(beta[idx + 1])
            idx += 2
            if abs(lin_c) <= self.coef_tol and abs(h_c) <= self.coef_tol:
                continue
            self.terms_.append({
                "feature": int(j),
                "threshold": float(t),
                "lin_coef": lin_c,
                "hinge_coef": h_c,
            })

        self.coef_ = np.zeros(self.n_features_in_, dtype=float)
        fi = np.zeros(self.n_features_in_, dtype=float)
        for term in self.terms_:
            j = term["feature"]
            self.coef_[j] += term["lin_coef"]
            fi[j] += abs(term["lin_coef"]) + abs(term["hinge_coef"])

        self.feature_importance_ = fi
        self.selected_feature_order_ = np.argsort(fi)[::-1]
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "terms_", "feature_importance_"])
        X = self._impute(X)
        yhat = np.full(X.shape[0], self.intercept_, dtype=float)
        for term in self.terms_:
            j = term["feature"]
            xj = X[:, j]
            yhat += term["lin_coef"] * xj
            yhat += term["hinge_coef"] * np.maximum(0.0, xj - term["threshold"])
        return yhat

    def __str__(self):
        check_is_fitted(self, ["intercept_", "terms_", "feature_importance_"])
        lines = [
            "SparseAdditiveKnotRegressor",
            "Prediction rule:",
            f"  y = {self.intercept_:+.6f}",
        ]
        if not self.terms_:
            lines.append("  (no active feature terms)")
            return "\n".join(lines)

        lines.append("  + sum of active feature terms:")
        for term in self.terms_:
            j = term["feature"]
            lines.append(
                "    "
                f"({term['lin_coef']:+.6f}*x{j}) + "
                f"({term['hinge_coef']:+.6f}*max(0, x{j}-{term['threshold']:.6f}))"
            )
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseAdditiveKnotRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseAdditiveKnotV1"
model_description = "Sparse additive one-knot equation with up to four features, each with linear plus learned-threshold hinge effect"
model_defs = [(model_shorthand_name, SparseAdditiveKnotRegressor())]


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
