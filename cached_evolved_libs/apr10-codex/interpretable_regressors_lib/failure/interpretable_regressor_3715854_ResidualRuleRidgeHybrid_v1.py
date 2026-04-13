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


class ResidualRuleRidgeHybridRegressor(BaseEstimator, RegressorMixin):
    """
    Hybrid interpretable regressor:
    1) robust-scaled ridge backbone (closed-form GCV alpha selection),
    2) a short residual-boosted sequence of one-feature decision stumps.

    Prediction:
      y_hat = intercept + z @ w + sum_m rule_m(z_jm)
    where each rule_m is:
      if z_j <= t: add delta_left
      else:        add delta_right
    """

    def __init__(
        self,
        alpha_grid=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
        max_rules=8,
        learning_rate=0.35,
        min_leaf_frac=0.12,
        candidate_quantiles=(0.15, 0.35, 0.50, 0.65, 0.85),
        feature_subsample=12,
        min_rule_improvement=1e-4,
        coef_display_tol=0.03,
    ):
        self.alpha_grid = alpha_grid
        self.max_rules = max_rules
        self.learning_rate = learning_rate
        self.min_leaf_frac = min_leaf_frac
        self.candidate_quantiles = candidate_quantiles
        self.feature_subsample = feature_subsample
        self.min_rule_improvement = min_rule_improvement
        self.coef_display_tol = coef_display_tol

    @staticmethod
    def _sanitize_alpha_grid(alpha_grid):
        a = np.asarray(alpha_grid, dtype=float)
        a = a[np.isfinite(a) & (a > 0)]
        if a.size == 0:
            return np.asarray([1.0], dtype=float)
        return np.unique(a)

    @staticmethod
    def _ridge_gcv(Z, y, alpha_grid):
        y_mean = float(np.mean(y))
        Z_mean = np.mean(Z, axis=0)
        Zc = Z - Z_mean
        yc = y - y_mean
        if Zc.shape[1] == 0:
            return y_mean, np.zeros(0, dtype=float), 1.0

        U, s, Vt = np.linalg.svd(Zc, full_matrices=False)
        Uy = U.T @ yc
        n = float(max(1, Z.shape[0]))

        best = None
        for alpha in alpha_grid:
            alpha = float(alpha)
            filt = s / (s * s + alpha)
            coef = Vt.T @ (filt * Uy)
            pred = y_mean + Zc @ coef
            mse = float(np.mean((y - pred) ** 2))
            dof = float(np.sum((s * s) / (s * s + alpha)))
            denom = max(1e-8, 1.0 - dof / n)
            gcv = mse / (denom * denom)
            if (best is None) or (gcv < best["gcv"]):
                best = {"gcv": gcv, "alpha": alpha, "coef": coef}

        coef = best["coef"]
        intercept = float(y_mean - np.dot(Z_mean, coef))
        return intercept, coef, float(best["alpha"])

    def _robust_scale(self, X):
        med = np.median(X, axis=0)
        q25 = np.quantile(X, 0.25, axis=0)
        q75 = np.quantile(X, 0.75, axis=0)
        iqr = q75 - q25
        iqr = np.where(np.abs(iqr) < 1e-8, 1.0, iqr)
        Z = (X - med) / iqr
        return Z, med, iqr

    def _candidate_features(self, Z, residual):
        p = Z.shape[1]
        if p <= 1:
            return np.arange(p, dtype=int)
        corr = np.zeros(p, dtype=float)
        rnorm = float(np.sqrt(np.sum(residual * residual)) + 1e-12)
        for j in range(p):
            zj = Z[:, j]
            znorm = float(np.sqrt(np.sum(zj * zj)) + 1e-12)
            corr[j] = abs(float(np.dot(zj, residual))) / (znorm * rnorm)
        k = int(max(1, min(p, self.feature_subsample)))
        idx = np.argpartition(corr, -k)[-k:]
        return np.sort(idx.astype(int))

    def _fit_best_stump(self, Z, residual):
        n, _ = Z.shape
        min_leaf = int(max(2, min(n - 2, int(np.ceil(self.min_leaf_frac * n)))))
        if n < 2 * min_leaf:
            return None

        base_mse = float(np.mean(residual * residual))
        best = None
        q_grid = np.asarray(self.candidate_quantiles, dtype=float)
        q_grid = q_grid[(q_grid > 0.0) & (q_grid < 1.0)]
        if q_grid.size == 0:
            q_grid = np.asarray([0.5], dtype=float)

        for j in self._candidate_features(Z, residual):
            z = Z[:, j]
            thresholds = np.unique(np.quantile(z, q_grid))
            if thresholds.size == 0:
                continue
            for t in thresholds:
                left = z <= t
                n_left = int(np.sum(left))
                n_right = n - n_left
                if n_left < min_leaf or n_right < min_leaf:
                    continue
                right = ~left
                left_mean = float(np.mean(residual[left]))
                right_mean = float(np.mean(residual[right]))
                update = np.where(left, left_mean, right_mean)
                step = float(self.learning_rate) * update
                new_mse = float(np.mean((residual - step) ** 2))
                improvement = base_mse - new_mse
                if (best is None) or (improvement > best["improvement"]):
                    best = {
                        "feature": int(j),
                        "threshold": float(t),
                        "delta_left": float(self.learning_rate * left_mean),
                        "delta_right": float(self.learning_rate * right_mean),
                        "improvement": float(improvement),
                    }
        return best

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        self.n_features_in_ = int(p)
        alpha_grid = self._sanitize_alpha_grid(self.alpha_grid)

        if p == 0:
            self.intercept_ = float(np.mean(y))
            self.coef_ = np.zeros(0, dtype=float)
            self.medians_ = np.zeros(0, dtype=float)
            self.iqrs_ = np.ones(0, dtype=float)
            self.rules_ = []
            self.alpha_ = 1.0
            self.training_mse_ = float(np.mean((y - self.intercept_) ** 2))
            self.is_fitted_ = True
            return self

        Z, med, iqr = self._robust_scale(X)
        intercept, coef, alpha = self._ridge_gcv(Z, y, alpha_grid)
        pred = intercept + Z @ coef

        rules = []
        max_rules = int(max(0, self.max_rules))
        min_gain = float(max(0.0, self.min_rule_improvement))
        for _ in range(max_rules):
            residual = y - pred
            stump = self._fit_best_stump(Z, residual)
            if stump is None or stump["improvement"] < min_gain:
                break
            j = stump["feature"]
            t = stump["threshold"]
            dl = stump["delta_left"]
            dr = stump["delta_right"]
            update = np.where(Z[:, j] <= t, dl, dr)
            pred = pred + update
            rules.append(stump)

        self.intercept_ = float(intercept)
        self.alpha_ = float(alpha)
        self.coef_ = coef
        self.rules_ = rules
        self.medians_ = med
        self.iqrs_ = iqr
        self.training_mse_ = float(np.mean((y - pred) ** 2))
        self.n_active_features_ = int(np.sum(np.abs(self.coef_) > 1e-10))
        self.n_rules_ = int(len(rules))
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        X = np.asarray(X, dtype=float)
        Z = (X - self.medians_) / self.iqrs_
        y = self.intercept_ + Z @ self.coef_
        for rule in self.rules_:
            j = rule["feature"]
            t = rule["threshold"]
            y += np.where(Z[:, j] <= t, rule["delta_left"], rule["delta_right"])
        return y

    def __str__(self):
        check_is_fitted(self, "is_fitted_")
        lines = ["Residual Rule + Ridge Hybrid Regressor:"]
        lines.append("  prediction = intercept + linear backbone + residual stump rules")
        lines.append(f"  intercept: {self.intercept_:+.4f}")
        lines.append(f"  ridge alpha (GCV): {self.alpha_:.5g}")
        lines.append(f"  training MSE: {self.training_mse_:.6f}")
        lines.append(f"  active linear features: {self.n_active_features_}/{self.n_features_in_}")
        lines.append(f"  residual rules: {self.n_rules_}")

        abs_coef = np.abs(self.coef_)
        tol = float(max(0.0, self.coef_display_tol))
        idx = [int(i) for i in np.argsort(-abs_coef) if abs_coef[i] >= tol]
        if idx:
            lines.append(f"  linear terms shown (|coef| >= {tol:.3f}):")
            for j in idx[:8]:
                med = float(self.medians_[j])
                iqr = float(self.iqrs_[j])
                lines.append(
                    f"    {self.coef_[j]:+.4f}*z{j}, where z{j}=(x{j}-{med:.4f})/{iqr:.4f}"
                )
        else:
            lines.append("  linear terms shown: none above threshold")

        if not self.rules_:
            lines.append("  rules: none")
            return "\n".join(lines)

        lines.append("  residual rules:")
        for i, r in enumerate(self.rules_, 1):
            j = int(r["feature"])
            t = float(r["threshold"])
            dl = float(r["delta_left"])
            dr = float(r["delta_right"])
            lines.append(
                f"    r{i}: if z{j} <= {t:+.4f} add {dl:+.4f}, else add {dr:+.4f}"
            )
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
ResidualRuleRidgeHybridRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "ResidualRuleRidgeHybrid_v1"
model_description = "Robust-scaled GCV-ridge backbone plus short residual-boosted one-feature stump rules"
model_defs = [(model_shorthand_name, ResidualRuleRidgeHybridRegressor())]


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

    std_tests = {t.__name__ for t in ALL_TESTS}
    hard_tests = {t.__name__ for t in HARD_TESTS}
    insight_tests = {t.__name__ for t in INSIGHT_TESTS}
    std_passed = sum(r["passed"] for r in interp_results if r["test"] in std_tests)
    hard_passed = sum(r["passed"] for r in interp_results if r["test"] in hard_tests)
    insight_passed = sum(r["passed"] for r in interp_results if r["test"] in insight_tests)
    print(f"[std {std_passed}/{len(std_tests)}  hard {hard_passed}/{len(hard_tests)}  insight {insight_passed}/{len(insight_tests)}]")
    print(f"total_seconds: {time.time() - t0:.1f}s")
