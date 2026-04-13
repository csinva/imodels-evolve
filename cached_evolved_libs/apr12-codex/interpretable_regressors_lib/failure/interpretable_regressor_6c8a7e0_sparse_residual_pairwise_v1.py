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


class SparseResidualPairwiseRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse additive regressor with residual pairwise interactions.

    Model:
      y = b0 + sum_j b_j * x_j + sum_(a,b) g_ab * (x_a * x_b)
    """

    def __init__(
        self,
        corr_screen=28,
        max_linear_terms=12,
        max_interactions=4,
        interaction_pool_size=8,
        min_interaction_gain=0.004,
        alpha_grid=(1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0),
        coef_prune_abs=1e-5,
        coef_decimals=4,
        random_state=42,
    ):
        self.corr_screen = corr_screen
        self.max_linear_terms = max_linear_terms
        self.max_interactions = max_interactions
        self.interaction_pool_size = interaction_pool_size
        self.min_interaction_gain = min_interaction_gain
        self.alpha_grid = alpha_grid
        self.coef_prune_abs = coef_prune_abs
        self.coef_decimals = coef_decimals
        self.random_state = random_state

    @staticmethod
    def _ridge_closed_form(Z, y, alpha):
        p = Z.shape[1]
        reg = float(alpha) * np.eye(p)
        reg[0, 0] = 0.0
        A = Z.T @ Z + reg
        b = Z.T @ y
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(A) @ b

    def _impute(self, X):
        X = np.asarray(X, dtype=float).copy()
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(self.feature_medians_, np.where(bad)[1])
        return X

    @staticmethod
    def _corr_abs(a, b):
        ac = a - np.mean(a)
        bc = b - np.mean(b)
        denom = (np.std(ac) + 1e-12) * (np.std(bc) + 1e-12)
        return abs(float(np.mean(ac * bc) / denom))

    def _fit_alpha(self, Xtr, ytr, Xva, yva, linear_idx, interactions):
        Ztr = self._design(Xtr, linear_idx, interactions)
        Zva = self._design(Xva, linear_idx, interactions)
        best = None
        for alpha in self.alpha_grid:
            beta = self._ridge_closed_form(Ztr, ytr, alpha)
            pred = Zva @ beta
            mse = float(np.mean((yva - pred) ** 2))
            if best is None or mse < best["mse"]:
                best = {"alpha": float(alpha), "beta": beta, "mse": mse}
        return best

    @staticmethod
    def _design(X, linear_idx, interactions):
        cols = [np.ones(X.shape[0], dtype=float)]
        for j in linear_idx:
            cols.append(X[:, int(j)])
        for a, b in interactions:
            cols.append(X[:, int(a)] * X[:, int(b)])
        return np.column_stack(cols)

    def _predict_from_parts(self, X, intercept, linear_idx, linear_coef, interactions, interaction_coef):
        pred = np.full(X.shape[0], float(intercept), dtype=float)
        if len(linear_idx) > 0:
            pred += X[:, linear_idx] @ linear_coef
        for (a, b), c in zip(interactions, interaction_coef):
            pred += float(c) * (X[:, a] * X[:, b])
        return pred

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        self.n_features_in_ = p

        self.feature_medians_ = np.nanmedian(X, axis=0)
        X = self._impute(X)

        rng = np.random.RandomState(self.random_state)
        idx = rng.permutation(n)
        n_tr = max(int(0.8 * n), min(220, n))
        tr_idx = idx[:n_tr]
        va_idx = idx[n_tr:] if n_tr < n else idx[:n_tr]
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xva, yva = X[va_idx], y[va_idx]

        corr = np.array([self._corr_abs(Xtr[:, j], ytr) for j in range(p)])
        n_screen = min(max(int(self.corr_screen), int(self.max_linear_terms)), p)
        screened = np.argsort(corr)[::-1][:n_screen]
        n_lin = min(int(self.max_linear_terms), screened.size)
        linear_idx = np.sort(screened[:n_lin]).astype(int)

        linear_fit = self._fit_alpha(Xtr, ytr, Xva, yva, linear_idx, interactions=[])
        base_pred_tr = self._design(Xtr, linear_idx, []) @ linear_fit["beta"]
        residual_tr = ytr - base_pred_tr
        base_val_mse = float(np.mean((yva - (self._design(Xva, linear_idx, []) @ linear_fit["beta"])) ** 2))

        pool_size = min(int(self.interaction_pool_size), linear_idx.size)
        pool = linear_idx[np.argsort(corr[linear_idx])[::-1][:pool_size]]
        pair_candidates = []
        for i in range(len(pool)):
            for j in range(i + 1, len(pool)):
                a, b = int(pool[i]), int(pool[j])
                score = self._corr_abs(Xtr[:, a] * Xtr[:, b], residual_tr)
                pair_candidates.append(((a, b), score))
        pair_candidates.sort(key=lambda x: x[1], reverse=True)

        interactions = []
        best_val_mse = base_val_mse
        for pair, _ in pair_candidates[:20]:
            if len(interactions) >= int(self.max_interactions):
                break
            trial_interactions = interactions + [pair]
            fit_trial = self._fit_alpha(Xtr, ytr, Xva, yva, linear_idx, trial_interactions)
            if fit_trial["mse"] < best_val_mse * (1.0 - float(self.min_interaction_gain)):
                interactions = trial_interactions
                best_val_mse = fit_trial["mse"]

        final_fit = self._fit_alpha(X, y, X, y, linear_idx, interactions)
        beta = final_fit["beta"]

        q = int(self.coef_decimals)
        self.alpha_ = float(final_fit["alpha"])
        self.intercept_ = float(np.round(beta[0], q))
        linear_coef = np.array(beta[1:1 + len(linear_idx)], dtype=float)
        interaction_coef = np.array(beta[1 + len(linear_idx):], dtype=float)

        linear_coef[np.abs(linear_coef) < float(self.coef_prune_abs)] = 0.0
        interaction_coef[np.abs(interaction_coef) < float(self.coef_prune_abs)] = 0.0
        self.linear_features_ = linear_idx.astype(int)
        self.linear_coef_ = np.round(linear_coef, q)
        self.interactions_ = [(int(a), int(b)) for (a, b) in interactions]
        self.interaction_coef_ = np.round(interaction_coef, q)

        importance = np.zeros(p, dtype=float)
        for j, c in zip(self.linear_features_, self.linear_coef_):
            importance[int(j)] += abs(float(c))
        for (a, b), c in zip(self.interactions_, self.interaction_coef_):
            v = 0.5 * abs(float(c))
            importance[int(a)] += v
            importance[int(b)] += v
        self.feature_importance_ = importance
        self.feature_rank_ = np.argsort(self.feature_importance_)[::-1]
        self.fitted_mse_ = float(np.mean((y - self.predict(X)) ** 2))
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "linear_features_", "linear_coef_", "interactions_", "interaction_coef_"])
        X = self._impute(X)
        return self._predict_from_parts(
            X,
            self.intercept_,
            self.linear_features_,
            self.linear_coef_,
            self.interactions_,
            self.interaction_coef_,
        )

    def __str__(self):
        check_is_fitted(self, ["intercept_", "linear_features_", "linear_coef_", "interactions_", "interaction_coef_"])
        terms = [f"{self.intercept_:+0.4f}"]
        for j, c in zip(self.linear_features_, self.linear_coef_):
            if abs(float(c)) > 0.0:
                terms.append(f"{float(c):+0.4f}*x{int(j)}")
        for (a, b), c in zip(self.interactions_, self.interaction_coef_):
            if abs(float(c)) > 0.0:
                terms.append(f"{float(c):+0.4f}*x{int(a)}*x{int(b)}")

        lines = [
            "SparseResidualPairwiseRegressor",
            f"Ridge alpha: {self.alpha_:.4g}",
            "",
            "Prediction equation:",
            "  y = " + " ".join(terms),
            "",
            f"Active linear features ({int(np.sum(np.abs(self.linear_coef_) > 0))}): "
            + ", ".join(f"x{int(j)}" for j, c in zip(self.linear_features_, self.linear_coef_) if abs(float(c)) > 0.0),
        ]
        if self.interactions_:
            active_pairs = [
                f"x{a}*x{b}" for (a, b), c in zip(self.interactions_, self.interaction_coef_) if abs(float(c)) > 0.0
            ]
            lines.append(f"Active pairwise terms ({len(active_pairs)}): " + ", ".join(active_pairs))
        else:
            lines.append("Active pairwise terms (0): none")

        lines.append("")
        lines.append("Feature importance (descending):")
        for j in self.feature_rank_[: min(10, self.n_features_in_)]:
            lines.append(f"  x{int(j)}: {self.feature_importance_[int(j)]:.4f}")
        near_zero = [f"x{j}" for j, v in enumerate(self.feature_importance_) if v < 1e-6]
        if near_zero:
            lines.append("Near-zero effect features: " + ", ".join(near_zero))
        lines.append(f"Training MSE: {self.fitted_mse_:.6f}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseResidualPairwiseRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseResidualPairwiseV1"
model_description = "Sparse screened linear backbone with holdout-gated residual pairwise interaction terms and explicit compact equation"
model_defs = [(model_shorthand_name, SparseResidualPairwiseRegressor())]

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
