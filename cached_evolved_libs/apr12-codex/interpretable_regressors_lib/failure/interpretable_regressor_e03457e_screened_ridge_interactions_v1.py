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


class ScreenedRidgeInteractionsRegressor(BaseEstimator, RegressorMixin):
    """
    Dense ridge backbone with a tiny, validation-selected interaction set.

    The model remains a single explicit arithmetic equation:
      y = intercept + sum_j w_j * x_j + sum_(i,k) v_(i,k) * (x_i * x_k)
    """

    def __init__(
        self,
        alpha_grid=(1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0),
        val_frac=0.2,
        min_val_samples=120,
        max_screen_features=10,
        max_interactions=2,
        min_interaction_gain=1e-4,
        coef_prune_abs=1e-6,
        coef_decimals=5,
        random_state=42,
    ):
        self.alpha_grid = alpha_grid
        self.val_frac = val_frac
        self.min_val_samples = min_val_samples
        self.max_screen_features = max_screen_features
        self.max_interactions = max_interactions
        self.min_interaction_gain = min_interaction_gain
        self.coef_prune_abs = coef_prune_abs
        self.coef_decimals = coef_decimals
        self.random_state = random_state

    @staticmethod
    def _ridge_closed_form(Z, y, alpha):
        p = Z.shape[1]
        reg = float(alpha) * np.eye(p)
        reg[0, 0] = 0.0  # do not penalize intercept
        return np.linalg.solve(Z.T @ Z + reg, Z.T @ y)

    def _impute(self, X):
        X = np.asarray(X, dtype=float).copy()
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(self.feature_medians_, np.where(bad)[1])
        return X

    def _split_idx(self, n):
        if n < int(self.min_val_samples) + 20:
            idx = np.arange(n)
            return idx, idx
        rng = np.random.RandomState(self.random_state)
        perm = rng.permutation(n)
        n_val = max(int(round(float(self.val_frac) * n)), int(self.min_val_samples))
        n_val = min(n_val, n // 2)
        return perm[n_val:], perm[:n_val]

    def _build_interactions(self, X, pairs):
        if not pairs:
            return np.zeros((X.shape[0], 0), dtype=float)
        cols = [(X[:, i] * X[:, j])[:, None] for i, j in pairs]
        return np.hstack(cols)

    def _fit_alpha_with_holdout(self, Xtr, ytr, Xval, yval, pairs):
        Itr = self._build_interactions(Xtr, pairs)
        Ival = self._build_interactions(Xval, pairs)
        Dtr = np.hstack([Xtr, Itr]) if Itr.shape[1] > 0 else Xtr
        Dval = np.hstack([Xval, Ival]) if Ival.shape[1] > 0 else Xval
        Ztr = np.column_stack([np.ones(Dtr.shape[0]), Dtr])
        Zval = np.column_stack([np.ones(Dval.shape[0]), Dval])

        best_alpha = float(self.alpha_grid[0])
        best_beta = self._ridge_closed_form(Ztr, ytr, best_alpha)
        best_mse = float(np.mean((yval - Zval @ best_beta) ** 2))

        for alpha in self.alpha_grid[1:]:
            beta = self._ridge_closed_form(Ztr, ytr, float(alpha))
            mse = float(np.mean((yval - Zval @ beta) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_alpha = float(alpha)
                best_beta = beta
        return best_beta, best_alpha, best_mse

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.feature_medians_ = np.nanmedian(X, axis=0)
        X = self._impute(X)

        tr_idx, val_idx = self._split_idx(X.shape[0])
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xval, yval = X[val_idx], y[val_idx]

        # Start from a dense linear ridge backbone.
        beta_lin, alpha_lin, mse_lin = self._fit_alpha_with_holdout(Xtr, ytr, Xval, yval, pairs=[])
        linear_coef = np.asarray(beta_lin[1: 1 + self.n_features_in_], dtype=float)
        top_k = min(int(self.max_screen_features), self.n_features_in_)
        screened = np.argsort(np.abs(linear_coef))[::-1][:top_k]
        candidate_pairs = [(int(screened[a]), int(screened[b]))
                           for a in range(len(screened)) for b in range(a + 1, len(screened))]

        selected_pairs = []
        current_mse = float(mse_lin)
        current_alpha = float(alpha_lin)
        for _ in range(int(self.max_interactions)):
            best_pair = None
            best_candidate_mse = current_mse
            best_candidate_alpha = current_alpha
            for pair in candidate_pairs:
                if pair in selected_pairs:
                    continue
                _, alpha_cand, mse_cand = self._fit_alpha_with_holdout(
                    Xtr, ytr, Xval, yval, pairs=selected_pairs + [pair]
                )
                if mse_cand < best_candidate_mse:
                    best_pair = pair
                    best_candidate_mse = float(mse_cand)
                    best_candidate_alpha = float(alpha_cand)
            gain = current_mse - best_candidate_mse
            if best_pair is None or gain < float(self.min_interaction_gain):
                break
            selected_pairs.append(best_pair)
            current_mse = best_candidate_mse
            current_alpha = best_candidate_alpha

        self.interaction_pairs_ = selected_pairs
        self.alpha_ = float(current_alpha)

        # Refit exact final equation on all training samples using selected alpha.
        I = self._build_interactions(X, self.interaction_pairs_)
        D = np.hstack([X, I]) if I.shape[1] > 0 else X
        Z = np.column_stack([np.ones(D.shape[0]), D])
        beta = self._ridge_closed_form(Z, y, self.alpha_)

        q = int(self.coef_decimals)
        linear_coef = np.asarray(beta[1: 1 + self.n_features_in_], dtype=float)
        inter_coef = np.asarray(beta[1 + self.n_features_in_:], dtype=float)

        linear_coef[np.abs(linear_coef) < float(self.coef_prune_abs)] = 0.0
        inter_coef[np.abs(inter_coef) < float(self.coef_prune_abs)] = 0.0

        self.intercept_ = float(np.round(beta[0], q))
        self.coef_ = np.round(linear_coef, q)
        self.interaction_coef_ = np.round(inter_coef, q)
        self.selected_features_ = [int(j) for j in np.where(np.abs(self.coef_) > 0.0)[0]]
        if len(self.selected_features_) == 0:
            j = int(np.argmax(np.abs(linear_coef)))
            self.selected_features_ = [j]
            self.coef_[j] = float(np.round(linear_coef[j], q))

        # Rank by absolute linear effect + shared interaction mass.
        importance = np.abs(self.coef_).astype(float)
        for c, (i, j) in zip(self.interaction_coef_, self.interaction_pairs_):
            importance[i] += abs(float(c))
            importance[j] += abs(float(c))
        self.feature_importance_ = importance
        self.feature_rank_ = np.argsort(self.feature_importance_)[::-1]
        self.fitted_mse_ = float(np.mean((y - self.predict(X)) ** 2))
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "coef_", "interaction_pairs_", "interaction_coef_"])
        X = self._impute(X)
        y = self.intercept_ + X @ self.coef_
        if len(self.interaction_pairs_) > 0:
            for c, (i, j) in zip(self.interaction_coef_, self.interaction_pairs_):
                y = y + float(c) * (X[:, i] * X[:, j])
        return y

    def __str__(self):
        check_is_fitted(self, ["intercept_", "coef_", "interaction_pairs_", "interaction_coef_"])
        lines = [
            "ScreenedRidgeInteractionsRegressor",
            f"Selected alpha: {self.alpha_:.5g}",
            "",
            "Exact prediction equation used by predict:",
            f"  y = {self.intercept_:+.5f}",
        ]
        for j in self.selected_features_:
            lines.append(f"    + ({float(self.coef_[j]):+.5f})*x{int(j)}")
        for c, (i, j) in zip(self.interaction_coef_, self.interaction_pairs_):
            if abs(float(c)) > 0.0:
                lines.append(f"    + ({float(c):+.5f})*(x{int(i)}*x{int(j)})")

        lines.append("")
        lines.append("Computation recipe: start at intercept, then add each listed linear and interaction term.")
        lines.append("Any feature not listed has zero linear coefficient.")
        if len(self.interaction_pairs_) == 0:
            lines.append("No interaction terms are active.")
        lines.append("")
        lines.append("Top feature importance (abs linear + attached interaction mass):")
        for j in self.feature_rank_[: min(10, self.n_features_in_)]:
            lines.append(f"  x{int(j)}: {self.feature_importance_[int(j)]:.5f}")
        lines.append(f"Training MSE: {self.fitted_mse_:.6f}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
ScreenedRidgeInteractionsRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "ScreenedRidgeInteractionsV1"
model_description = "Dense closed-form ridge with holdout alpha selection and validation-screened pairwise interaction terms in one explicit arithmetic equation"
model_defs = [(model_shorthand_name, ScreenedRidgeInteractionsRegressor())]

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------

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
