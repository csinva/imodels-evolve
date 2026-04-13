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


class CalibratedRidgeRuleRegressor(BaseEstimator, RegressorMixin):
    """
    Closed-form ridge with simple alpha selection plus one optional residual rule.

    Prediction form:
      y = intercept + sum_j coef_j * x_j + rule_adjustment(x_rule)
    where rule_adjustment is either 0 or a single threshold-based offset.
    """

    def __init__(
        self,
        alpha_grid=(0.03, 0.1, 0.3, 1.0, 3.0, 10.0),
        coef_tol=1e-3,
        max_display_terms=12,
        top_rule_features=6,
        min_rule_gain=0.003,
        min_leaf_frac=0.12,
        random_state=42,
    ):
        self.alpha_grid = alpha_grid
        self.coef_tol = coef_tol
        self.max_display_terms = max_display_terms
        self.top_rule_features = top_rule_features
        self.min_rule_gain = min_rule_gain
        self.min_leaf_frac = min_leaf_frac
        self.random_state = random_state

    def _impute(self, X):
        X = np.asarray(X, dtype=float).copy()
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(self.feature_medians_, np.where(bad)[1])
        return X

    @staticmethod
    def _ridge_fit(Xs, y, alpha):
        y_mean = float(np.mean(y))
        yc = y - y_mean
        gram = Xs.T @ Xs
        rhs = Xs.T @ yc
        beta = np.linalg.solve(gram + float(alpha) * np.eye(Xs.shape[1]), rhs)
        return y_mean, beta

    def _fit_dense_ridge(self, X, y, alpha):
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        sigma[sigma < 1e-10] = 1.0
        Xs = (X - mu) / sigma
        y_mean, beta_s = self._ridge_fit(Xs, y, alpha)
        coef = beta_s / sigma
        intercept = y_mean - float(np.dot(mu, coef))
        pred = intercept + X @ coef
        mse = float(np.mean((y - pred) ** 2))
        return intercept, coef, mse

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.feature_medians_ = np.nanmedian(X, axis=0)
        X = self._impute(X)

        rng = np.random.RandomState(self.random_state)
        n = X.shape[0]
        idx = np.arange(n)
        rng.shuffle(idx)
        cut = max(int(0.8 * n), 20)
        tr_idx = idx[:cut]
        va_idx = idx[cut:] if cut < n else idx[:0]
        if va_idx.size < 10:
            tr_idx = idx
            va_idx = idx

        best_alpha = None
        best_val_mse = None
        for alpha in self.alpha_grid:
            inter, coef, _ = self._fit_dense_ridge(X[tr_idx], y[tr_idx], alpha)
            val_pred = inter + X[va_idx] @ coef
            val_mse = float(np.mean((y[va_idx] - val_pred) ** 2))
            if best_val_mse is None or val_mse < best_val_mse:
                best_val_mse = val_mse
                best_alpha = float(alpha)

        self.alpha_ = float(best_alpha)
        inter, coef, mse_linear = self._fit_dense_ridge(X, y, self.alpha_)

        self.intercept_ = float(inter)
        self.linear_coef_ = coef.astype(float)
        pred_linear = self.intercept_ + X @ self.linear_coef_
        residual = y - pred_linear

        ranking = np.argsort(np.abs(self.linear_coef_))[::-1]
        cand_feats = ranking[: min(self.top_rule_features, self.n_features_in_)]
        min_leaf = max(10, int(self.min_leaf_frac * n))
        best_rule = None
        for j in cand_feats:
            xj = X[:, int(j)]
            for q in (0.15, 0.30, 0.50, 0.70, 0.85):
                thr = float(np.quantile(xj, q))
                left = xj <= thr
                right = ~left
                if int(left.sum()) < min_leaf or int(right.sum()) < min_leaf:
                    continue
                left_adj = float(np.mean(residual[left]))
                right_adj = float(np.mean(residual[right]))
                pred = pred_linear + np.where(left, left_adj, right_adj)
                mse = float(np.mean((y - pred) ** 2))
                if best_rule is None or mse < best_rule["mse"]:
                    best_rule = {
                        "mse": mse,
                        "feature": int(j),
                        "threshold": thr,
                        "left_adj": left_adj,
                        "right_adj": right_adj,
                    }

        use_rule = best_rule is not None and (mse_linear - best_rule["mse"]) > self.min_rule_gain
        if use_rule:
            self.rule_feature_ = int(best_rule["feature"])
            self.rule_threshold_ = float(best_rule["threshold"])
            self.rule_left_adjust_ = float(best_rule["left_adj"])
            self.rule_right_adjust_ = float(best_rule["right_adj"])
        else:
            self.rule_feature_ = -1
            self.rule_threshold_ = 0.0
            self.rule_left_adjust_ = 0.0
            self.rule_right_adjust_ = 0.0

        fi = np.abs(self.linear_coef_)
        if self.rule_feature_ >= 0:
            fi[self.rule_feature_] += 0.5 * (
                abs(self.rule_left_adjust_) + abs(self.rule_right_adjust_)
            )
        self.feature_importance_ = fi
        return self

    def predict(self, X):
        check_is_fitted(self, ["intercept_", "linear_coef_", "rule_feature_", "feature_importance_"])
        X = self._impute(X)
        yhat = self.intercept_ + X @ self.linear_coef_
        if self.rule_feature_ >= 0:
            yhat = yhat + np.where(
                X[:, self.rule_feature_] <= self.rule_threshold_,
                self.rule_left_adjust_,
                self.rule_right_adjust_,
            )
        return yhat

    def __str__(self):
        check_is_fitted(self, ["intercept_", "linear_coef_", "rule_feature_", "feature_importance_"])
        rank = np.argsort(np.abs(self.linear_coef_))[::-1]
        active = [j for j in rank if abs(self.linear_coef_[j]) >= self.coef_tol]
        shown = active[: self.max_display_terms]
        near_zero = [j for j in range(self.n_features_in_) if abs(self.linear_coef_[j]) < self.coef_tol]

        lines = [
            "CalibratedRidgeRuleRegressor",
            "Computation steps:",
            "1) Start with linear equation y_lin = intercept + sum(coef_j * xj).",
            "2) If the rule is active, add exactly one branch-specific adjustment.",
            "",
            f"selected_alpha = {self.alpha_:.4f}",
            f"intercept = {self.intercept_:+.6f}",
            "linear coefficients (largest first):",
        ]
        for j in shown:
            lines.append(f"  x{j}: {self.linear_coef_[j]:+.6f}")
        if len(active) > len(shown):
            lines.append(f"  ... {len(active) - len(shown)} additional small active terms omitted")
        if near_zero:
            lines.append("near-zero effect features: " + ", ".join(f"x{j}" for j in near_zero))

        if self.rule_feature_ >= 0:
            lines.extend([
                "",
                "single residual rule:",
                f"  if x{self.rule_feature_} <= {self.rule_threshold_:.6f}: add {self.rule_left_adjust_:+.6f}",
                f"  else: add {self.rule_right_adjust_:+.6f}",
            ])
        else:
            lines.extend(["", "single residual rule: none"])

        eq_terms = [f"{self.intercept_:+.6f}"] + [
            f"({self.linear_coef_[j]:+.6f}*x{j})" for j in shown
        ]
        lines.extend([
            "",
            "compact equation using displayed terms:",
            "  y ~= " + " + ".join(eq_terms),
        ])
        return "\n".join(lines)



# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
CalibratedRidgeRuleRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "CalibratedRidgeRuleV1"
model_description = "Custom closed-form ridge with holdout-selected alpha and an optional single residual threshold rule for compact nonlinearity"
model_defs = [(model_shorthand_name, CalibratedRidgeRuleRegressor())]


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
