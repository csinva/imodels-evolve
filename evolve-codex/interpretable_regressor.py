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
from sklearn.linear_model import LassoCV
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class SparseSymbolicInteractionRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse symbolic regression with screened interpretable basis functions.

    Basis dictionary:
      - a small set of original features x_j
      - optional |x_j - median_j| nonlinearity terms
      - a tiny set of pairwise interactions (x_i - med_i)*(x_j - med_j)

    Final model is a sparse linear combination fit with L1 regularization.
    """

    def __init__(
        self,
        max_main_features=6,
        max_abs_features=3,
        max_interactions=2,
        interaction_pool=6,
        min_keep_coef=1e-3,
        random_state=42,
    ):
        self.max_main_features = max_main_features
        self.max_abs_features = max_abs_features
        self.max_interactions = max_interactions
        self.interaction_pool = interaction_pool
        self.min_keep_coef = min_keep_coef
        self.random_state = random_state

    @staticmethod
    def _as_2d_float(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def _impute(self, X):
        return np.where(np.isnan(X), self.feature_medians_, X)

    @staticmethod
    def _corr_scores(X, y):
        yc = y - y.mean()
        y_norm = np.linalg.norm(yc) + 1e-12
        scores = np.zeros(X.shape[1], dtype=float)
        for j in range(X.shape[1]):
            xj = X[:, j]
            xj = xj - xj.mean()
            scores[j] = abs(np.dot(xj, yc) / ((np.linalg.norm(xj) + 1e-12) * y_norm))
        return scores

    def _build_basis(self, X):
        cols = []
        specs = []

        # Linear terms.
        for j in self.main_features_:
            cols.append(X[:, j])
            specs.append(("lin", int(j)))

        # Absolute-deviation terms to capture simple U-shaped behavior.
        for j in self.abs_features_:
            cols.append(np.abs(X[:, j] - self.feature_medians_[j]))
            specs.append(("abs", int(j)))

        # Pairwise centered interactions.
        for i, j in self.interactions_:
            cols.append((X[:, i] - self.feature_medians_[i]) * (X[:, j] - self.feature_medians_[j]))
            specs.append(("int", int(i), int(j)))

        if len(cols) == 0:
            return np.zeros((X.shape[0], 1), dtype=float), [("bias_fallback",)]
        return np.column_stack(cols), specs

    def fit(self, X, y):
        X = self._as_2d_float(X)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape

        self.feature_medians_ = np.nanmedian(X, axis=0)
        self.feature_medians_ = np.where(np.isnan(self.feature_medians_), 0.0, self.feature_medians_)
        X_imp = self._impute(X)

        # Feature screening from linear correlation.
        corr = self._corr_scores(X_imp, y)
        order = np.argsort(-corr)

        k_main = min(self.max_main_features, p)
        self.main_features_ = [int(j) for j in order[:k_main]]

        # Small abs-feature subset from strongest screened main features.
        k_abs = min(self.max_abs_features, len(self.main_features_))
        self.abs_features_ = self.main_features_[:k_abs]

        # Residual-based interaction screening using only top interaction_pool features.
        pool = [int(j) for j in order[: min(self.interaction_pool, p)]]
        self.interactions_ = []
        if self.max_interactions > 0 and len(pool) >= 2:
            # Residual from simple least-squares on main linear terms.
            Z_main = X_imp[:, self.main_features_] if self.main_features_ else np.zeros((n, 0), dtype=float)
            if Z_main.shape[1] > 0:
                A = np.column_stack([np.ones(n), Z_main])
                coef_main, *_ = np.linalg.lstsq(A, y, rcond=None)
                resid = y - A @ coef_main
            else:
                resid = y - y.mean()

            candidates = []
            for a in range(len(pool)):
                for b in range(a + 1, len(pool)):
                    i, j = pool[a], pool[b]
                    term = (X_imp[:, i] - self.feature_medians_[i]) * (X_imp[:, j] - self.feature_medians_[j])
                    denom = np.linalg.norm(term) + 1e-12
                    score = abs(np.dot(term, resid)) / denom
                    candidates.append((score, i, j))
            candidates.sort(reverse=True, key=lambda t: t[0])
            self.interactions_ = [(int(i), int(j)) for _, i, j in candidates[: self.max_interactions]]

        # Fit sparse model on the dictionary.
        Z, self.basis_specs_ = self._build_basis(X_imp)
        self.z_mean_ = Z.mean(axis=0)
        self.z_std_ = Z.std(axis=0)
        self.z_std_ = np.where(self.z_std_ < 1e-8, 1.0, self.z_std_)
        Zs = (Z - self.z_mean_) / self.z_std_

        self.linear_ = LassoCV(
            cv=3,
            random_state=self.random_state,
            max_iter=8000,
            n_alphas=60,
            fit_intercept=True,
        )
        self.linear_.fit(Zs, y)

        # Drop tiny terms for compactness and easier textual simulation.
        coef = self.linear_.coef_.copy()
        keep = np.abs(coef) >= self.min_keep_coef
        self.active_mask_ = keep
        self.n_features_in_ = p
        return self

    def predict(self, X):
        check_is_fitted(self, ["linear_", "basis_specs_", "z_mean_", "z_std_"])
        X = self._as_2d_float(X)
        X_imp = self._impute(X)
        Z, _ = self._build_basis(X_imp)
        Zs = (Z - self.z_mean_) / self.z_std_
        return self.linear_.predict(Zs)

    def __str__(self):
        check_is_fitted(self, ["linear_", "basis_specs_", "active_mask_"])

        coef = self.linear_.coef_.ravel()
        eff_intercept = float(self.linear_.intercept_ - np.sum(coef * (self.z_mean_ / self.z_std_)))

        lines = [
            "SparseSymbolicInteractionRegressor",
            f"prediction = {eff_intercept:.6f}",
        ]

        for idx, (c, spec, use) in enumerate(zip(coef, self.basis_specs_, self.active_mask_)):
            if not use:
                continue
            scale = c / self.z_std_[idx]

            if spec[0] == "lin":
                term = f"x{spec[1]}"
            elif spec[0] == "abs":
                med = self.feature_medians_[spec[1]]
                term = f"|x{spec[1]} - {med:.4g}|"
            elif spec[0] == "int":
                i, j = spec[1], spec[2]
                mi = self.feature_medians_[i]
                mj = self.feature_medians_[j]
                term = f"(x{i} - {mi:.4g})*(x{j} - {mj:.4g})"
            else:
                term = "1"

            lines.append(f"  {scale:+.6f} * {term}")

        if len(lines) == 2:
            lines.append("  +0.000000 * (no active terms)")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseSymbolicInteractionRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseSymbolicInteract_v1"
model_description = "Sparse symbolic dictionary: screened linear terms + |x-med| nonlinearities + residual-screened pairwise interactions via L1 fit"
model_defs = [(model_shorthand_name, SparseSymbolicInteractionRegressor())]


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
