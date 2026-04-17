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


class ResidualRuleLinearRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse linear backbone plus short residual step-rule list.

    Prediction:
      y = intercept + sum_j coef_j * x_j + sum_r weight_r * I(condition_r)
    """

    def __init__(
        self,
        max_linear_terms=10,
        max_rules=5,
        linear_ridge=5e-2,
        rule_shrinkage=20.0,
        min_rule_rel_gain=2e-3,
        n_thresholds=7,
        min_support=0.08,
    ):
        self.max_linear_terms = max_linear_terms
        self.max_rules = max_rules
        self.linear_ridge = linear_ridge
        self.rule_shrinkage = rule_shrinkage
        self.min_rule_rel_gain = min_rule_rel_gain
        self.n_thresholds = n_thresholds
        self.min_support = min_support

    @staticmethod
    def _safe_std(v):
        s = np.std(v)
        return float(s if s > 1e-12 else 1.0)

    @staticmethod
    def _ridge_with_intercept(B, y, lam):
        n = B.shape[0]
        if B.shape[1] == 0:
            intercept = float(np.mean(y))
            pred = np.full(n, intercept, dtype=float)
            return intercept, np.zeros(0, dtype=float), pred

        ones = np.ones((n, 1), dtype=float)
        Xd = np.hstack([ones, B])
        p = Xd.shape[1]
        reg = np.sqrt(max(lam, 1e-12)) * np.eye(p)
        reg[0, 0] = 0.0
        A = np.vstack([Xd, reg])
        b = np.concatenate([y, np.zeros(p, dtype=float)])
        beta, *_ = np.linalg.lstsq(A, b, rcond=None)
        intercept = float(beta[0])
        coef = np.asarray(beta[1:], dtype=float)
        pred = intercept + B @ coef
        return intercept, coef, pred

    def _rule_mask(self, X, rule):
        col = X[:, rule["j"]]
        if rule["op"] == ">":
            return col > rule["t"]
        return col <= rule["t"]

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        self.x_mean_ = np.mean(X, axis=0)
        self.x_scale_ = np.array([self._safe_std(X[:, j]) for j in range(n_features)], dtype=float)
        Xn = (X - self.x_mean_) / self.x_scale_

        yc = y - float(np.mean(y))
        corr = np.abs(Xn.T @ yc)
        k = min(max(1, self.max_linear_terms), n_features)
        active = np.argsort(corr)[-k:]
        active = np.sort(active)
        self.active_linear_features_ = active.astype(int)

        B = Xn[:, self.active_linear_features_]
        intercept, coef, pred = self._ridge_with_intercept(B, y, self.linear_ridge)
        self.intercept_ = float(intercept)
        self.linear_coef_ = coef
        residual = y - pred

        self.rules_ = []
        current_mse = float(np.mean(residual ** 2))
        quantiles = np.linspace(0.1, 0.9, max(3, self.n_thresholds))

        for _ in range(self.max_rules):
            if current_mse <= 1e-12:
                break

            feature_order = np.argsort(np.abs(Xn.T @ residual))[::-1]
            feature_pool = feature_order[: min(n_features, max(10, self.max_linear_terms))]
            best = None

            for j in feature_pool:
                values = Xn[:, j]
                thresholds = np.unique(np.quantile(values, quantiles))
                for t in thresholds:
                    for op in (">", "<="):
                        mask = values > t if op == ">" else values <= t
                        support = float(np.mean(mask))
                        if support < self.min_support or support > 1.0 - self.min_support:
                            continue

                        n_on = int(np.sum(mask))
                        raw_w = float(np.mean(residual[mask]))
                        shrink = n_on / (n_on + self.rule_shrinkage)
                        w = raw_w * shrink
                        if abs(w) < 1e-10:
                            continue

                        new_residual = residual - w * mask.astype(float)
                        new_mse = float(np.mean(new_residual ** 2))
                        gain = current_mse - new_mse
                        if best is None or gain > best["gain"]:
                            best = {
                                "j": int(j),
                                "op": op,
                                "t": float(t),
                                "weight": float(w),
                                "gain": float(gain),
                            }

            if best is None:
                break

            rel_gain = best["gain"] / max(current_mse, 1e-12)
            if rel_gain < self.min_rule_rel_gain:
                break

            self.rules_.append({k: best[k] for k in ("j", "op", "t", "weight")})
            mask = self._rule_mask(Xn, self.rules_[-1])
            residual = residual - self.rules_[-1]["weight"] * mask.astype(float)
            current_mse = float(np.mean(residual ** 2))

        self.train_mse_ = current_mse
        return self

    def predict(self, X):
        check_is_fitted(
            self,
            ["x_mean_", "x_scale_", "active_linear_features_", "linear_coef_", "intercept_", "rules_"],
        )
        X = np.asarray(X, dtype=float)
        Xn = (X - self.x_mean_) / self.x_scale_
        preds = np.full(X.shape[0], self.intercept_, dtype=float)

        if len(self.active_linear_features_) > 0:
            B = Xn[:, self.active_linear_features_]
            preds += B @ self.linear_coef_

        for rule in self.rules_:
            mask = self._rule_mask(Xn, rule)
            preds += rule["weight"] * mask.astype(float)
        return preds

    def __str__(self):
        check_is_fitted(
            self,
            ["active_linear_features_", "linear_coef_", "intercept_", "rules_", "x_mean_", "x_scale_"],
        )
        lines = ["Residual Rule Linear Regressor", "prediction = linear_backbone + residual_step_rules", ""]
        lines.append("Linear backbone (features are z-scored: z_j=(x_j-mean_j)/std_j):")
        lines.append(f"  intercept = {self.intercept_:.6f}")

        if len(self.active_linear_features_) == 0:
            lines.append("  no active linear terms")
        else:
            order = np.argsort(np.abs(self.linear_coef_))[::-1]
            for idx in order:
                j = int(self.active_linear_features_[idx])
                c = float(self.linear_coef_[idx])
                lines.append(
                    f"  {c:+.6f} * z{j}   [mean={self.x_mean_[j]:.6f}, std={self.x_scale_[j]:.6f}]"
                )

        lines.append("")
        lines.append("Residual step rules (additive offsets):")
        if len(self.rules_) == 0:
            lines.append("  none")
        else:
            for r_i, rule in enumerate(self.rules_, 1):
                lines.append(
                    f"  Rule {r_i}: if z{rule['j']} {rule['op']} {rule['t']:.6f} "
                    f"then add {rule['weight']:+.6f}"
                )

        lines.append("")
        lines.append("How to predict: compute linear part, then add each rule offset whose condition is true.")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
ResidualRuleLinearRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "ResidualRuleLinearV1"
model_description = "Sparse z-scored linear backbone plus greedy one-dimensional residual step rules with explicit threshold-and-offset representation"
model_defs = [(model_shorthand_name, ResidualRuleLinearRegressor())]


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
