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


class SplitAdjustedRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Shared-backbone model tree:
      1) Fit one global ridge equation.
      2) Optionally split once on one feature.
      3) Add compact leaf-specific residual linear adjustments.
    """

    def __init__(
        self,
        alpha_grid=(0.003, 0.01, 0.03, 0.1, 0.3, 1.0),
        split_quantiles=(0.2, 0.35, 0.5, 0.65, 0.8),
        top_split_features=8,
        max_delta_terms=3,
        min_leaf_frac=0.15,
        min_gain_frac=0.0015,
        complexity_penalty=0.0008,
        leaf_l2=0.02,
        coef_tol=1e-4,
        random_state=42,
    ):
        self.alpha_grid = alpha_grid
        self.split_quantiles = split_quantiles
        self.top_split_features = top_split_features
        self.max_delta_terms = max_delta_terms
        self.min_leaf_frac = min_leaf_frac
        self.min_gain_frac = min_gain_frac
        self.complexity_penalty = complexity_penalty
        self.leaf_l2 = leaf_l2
        self.coef_tol = coef_tol
        self.random_state = random_state

    def _impute(self, X):
        X = np.asarray(X, dtype=float).copy()
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(self.feature_medians_, np.where(bad)[1])
        return X

    @staticmethod
    def _fit_ridge_standardized(X, y, alpha):
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        sigma[sigma < 1e-10] = 1.0
        Xs = (X - mu) / sigma
        y_mean = float(np.mean(y))
        yc = y - y_mean
        p = X.shape[1]
        gram = Xs.T @ Xs
        beta_s = np.linalg.solve(gram + float(alpha) * np.eye(p), Xs.T @ yc)
        coef = beta_s / sigma
        intercept = y_mean - float(np.dot(mu, coef))
        return float(intercept), coef.astype(float)

    def _choose_alpha(self, X, y):
        n = X.shape[0]
        if n < 40:
            return float(self.alpha_grid[0])
        rng = np.random.RandomState(self.random_state)
        idx = np.arange(n)
        rng.shuffle(idx)
        cut = max(int(0.8 * n), n - 40)
        tr = idx[:cut]
        va = idx[cut:]
        if va.size < 12:
            return float(self.alpha_grid[0])

        best_alpha = float(self.alpha_grid[0])
        best_mse = np.inf
        for a in self.alpha_grid:
            intercept, coef = self._fit_ridge_standardized(X[tr], y[tr], a)
            pred = intercept + X[va] @ coef
            mse = float(np.mean((y[va] - pred) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_alpha = float(a)
        return best_alpha

    def _corr_abs(self, a, b):
        ac = a - float(np.mean(a))
        bc = b - float(np.mean(b))
        denom = (float(np.std(ac)) + 1e-12) * (float(np.std(bc)) + 1e-12)
        return abs(float(np.mean(ac * bc)) / denom)

    def _split_score(self, xj, y):
        return self._corr_abs(xj, y) + 0.6 * self._corr_abs(xj * xj, y) + 0.4 * self._corr_abs(np.abs(xj), y)

    def _fit_leaf_delta(self, X_leaf, residual):
        n_leaf, p = X_leaf.shape
        if n_leaf < 3:
            return {"intercept": float(np.mean(residual)), "coef": np.zeros(p), "active": np.array([], dtype=int)}

        scores = np.array([self._corr_abs(X_leaf[:, j], residual) for j in range(p)])
        order = np.argsort(scores)[::-1]
        active = order[: min(int(self.max_delta_terms), p)]
        active = np.array([j for j in active if scores[j] > 1e-6], dtype=int)
        if active.size == 0:
            return {"intercept": float(np.mean(residual)), "coef": np.zeros(p), "active": active}

        Z = np.column_stack([np.ones(n_leaf), X_leaf[:, active]])
        reg = float(self.leaf_l2) * n_leaf
        eye = np.eye(Z.shape[1])
        eye[0, 0] = 0.0
        beta = np.linalg.solve(Z.T @ Z + reg * eye, Z.T @ residual)

        coef = np.zeros(p, dtype=float)
        coef[active] = beta[1:]
        return {
            "intercept": float(beta[0]),
            "coef": coef,
            "active": np.where(np.abs(coef) > self.coef_tol)[0],
        }

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.feature_medians_ = np.nanmedian(X, axis=0)
        X = self._impute(X)

        self.alpha_ = self._choose_alpha(X, y)
        self.intercept_, self.coef_ = self._fit_ridge_standardized(X, y, self.alpha_)
        base_pred = self.intercept_ + X @ self.coef_
        residual = y - base_pred
        base_mse = float(np.mean((y - base_pred) ** 2))
        y_var = float(np.var(y)) + 1e-12

        n = X.shape[0]
        min_leaf = max(12, int(float(self.min_leaf_frac) * n))
        candidate_order = np.argsort(np.array([self._split_score(X[:, j], y) for j in range(self.n_features_in_)]))[::-1]
        candidate_features = candidate_order[: min(int(self.top_split_features), self.n_features_in_)]

        self.use_split_ = False
        best_score = base_mse
        best_payload = None
        for j in candidate_features:
            xj = X[:, j]
            thresholds = np.unique(np.quantile(xj, self.split_quantiles))
            for thr in thresholds:
                left = xj <= float(thr)
                right = ~left
                n_left = int(np.sum(left))
                n_right = int(np.sum(right))
                if n_left < min_leaf or n_right < min_leaf:
                    continue

                left_delta = self._fit_leaf_delta(X[left], residual[left])
                right_delta = self._fit_leaf_delta(X[right], residual[right])

                pred = base_pred.copy()
                pred[left] += left_delta["intercept"] + X[left] @ left_delta["coef"]
                pred[right] += right_delta["intercept"] + X[right] @ right_delta["coef"]
                mse = float(np.mean((y - pred) ** 2))
                complexity = 1 + len(left_delta["active"]) + len(right_delta["active"])
                score = mse + float(self.complexity_penalty) * complexity * y_var

                if score < best_score:
                    best_score = score
                    best_payload = {
                        "feature": int(j),
                        "threshold": float(thr),
                        "left": left_delta,
                        "right": right_delta,
                        "frac_left": n_left / n,
                        "frac_right": n_right / n,
                        "mse": mse,
                    }

        if best_payload is not None and (base_mse - best_payload["mse"]) > float(self.min_gain_frac) * y_var:
            self.use_split_ = True
            self.split_feature_ = int(best_payload["feature"])
            self.split_threshold_ = float(best_payload["threshold"])
            self.left_delta_ = best_payload["left"]
            self.right_delta_ = best_payload["right"]
            self.leaf_fraction_ = (float(best_payload["frac_left"]), float(best_payload["frac_right"]))

        self.coef_[np.abs(self.coef_) < self.coef_tol] = 0.0
        self.feature_importance_ = np.abs(self.coef_).copy()
        if self.use_split_:
            self.feature_importance_ += self.leaf_fraction_[0] * np.abs(self.left_delta_["coef"])
            self.feature_importance_ += self.leaf_fraction_[1] * np.abs(self.right_delta_["coef"])
            self.feature_importance_[self.split_feature_] += 0.5 * (
                abs(self.left_delta_["intercept"]) + abs(self.right_delta_["intercept"])
            )
        self.selected_feature_order_ = np.argsort(self.feature_importance_)[::-1]
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "coef_", "use_split_"])
        X = self._impute(X)
        yhat = self.intercept_ + X @ self.coef_
        if self.use_split_:
            left = X[:, self.split_feature_] <= self.split_threshold_
            right = ~left
            yhat[left] += self.left_delta_["intercept"] + X[left] @ self.left_delta_["coef"]
            yhat[right] += self.right_delta_["intercept"] + X[right] @ self.right_delta_["coef"]
        return yhat

    def _equation(self, intercept, coef):
        terms = [f"{intercept:+.6f}"]
        order = np.argsort(np.abs(coef))[::-1]
        for j in order:
            if abs(coef[j]) >= self.coef_tol:
                terms.append(f"{coef[j]:+.6f}*x{int(j)}")
        return "y = " + " ".join(terms)

    def __str__(self):
        check_is_fitted(self, ["intercept_", "coef_", "use_split_"])
        lines = [
            "SplitAdjustedRidgeRegressor",
            "Global backbone equation:",
            f"  {self._equation(self.intercept_, self.coef_)}",
        ]
        if self.use_split_:
            lines.append("")
            lines.append(
                f"Split rule: if x{self.split_feature_} <= {self.split_threshold_:+.6f}, use LEFT adjustment; else RIGHT adjustment."
            )
            lines.append("LEFT adjustment (added to global prediction):")
            lines.append(f"  {self._equation(self.left_delta_['intercept'], self.left_delta_['coef'])}")
            lines.append("RIGHT adjustment (added to global prediction):")
            lines.append(f"  {self._equation(self.right_delta_['intercept'], self.right_delta_['coef'])}")
            lines.append(f"Leaf coverage: left={self.leaf_fraction_[0]:.2%}, right={self.leaf_fraction_[1]:.2%}")
        else:
            lines.append("No split selected; prediction is the global equation only.")

        lines.append("")
        lines.append("Feature summary (sorted by total effect magnitude):")
        for j in self.selected_feature_order_[: min(12, self.n_features_in_)]:
            lines.append(f"  x{int(j)}: linear={self.coef_[int(j)]:+.6f}, importance={self.feature_importance_[int(j)]:.6f}")

        rel_cut = 0.06 * (np.max(np.abs(self.coef_)) + 1e-12)
        negligible = [f"x{j}" for j in range(self.n_features_in_) if abs(self.coef_[j]) <= rel_cut]
        if negligible:
            lines.append("Likely negligible features (tiny linear coefficients): " + ", ".join(negligible))
        if self.use_split_:
            lines.append("To simulate: compute global equation, then add the left/right adjustment from the split rule.")
        else:
            lines.append("To simulate: compute the global equation directly.")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SplitAdjustedRidgeRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SplitAdjustedRidgeV1"
model_description = "Global ridge backbone with one optional threshold split and compact leaf-specific residual linear adjustments"
model_defs = [(model_shorthand_name, SplitAdjustedRidgeRegressor())]

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
