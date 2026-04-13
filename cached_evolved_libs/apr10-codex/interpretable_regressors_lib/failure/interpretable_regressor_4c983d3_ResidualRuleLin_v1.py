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
    Greedy symbolic regressor with two stages:
      1) sparse linear backbone selected via forward correlation
      2) sparse binary threshold rules fitted on residuals
    """

    def __init__(
        self,
        candidate_features=20,
        max_linear_terms=5,
        max_rule_terms=3,
        quantiles=(0.33, 0.67),
        min_residual_gain=1e-4,
    ):
        self.candidate_features = candidate_features
        self.max_linear_terms = max_linear_terms
        self.max_rule_terms = max_rule_terms
        self.quantiles = quantiles
        self.min_residual_gain = min_residual_gain

    @staticmethod
    def _safe_corr_score(v, target):
        vc = v - np.mean(v)
        tc = target - np.mean(target)
        return float(abs(np.dot(vc, tc)) / ((np.linalg.norm(vc) + 1e-12) * (np.linalg.norm(tc) + 1e-12)))

    @staticmethod
    def _fit_with_terms(X, y, linear_terms, rule_terms):
        cols = [np.ones(X.shape[0], dtype=float)]
        for j in linear_terms:
            cols.append(X[:, j])
        for j, t in rule_terms:
            cols.append((X[:, j] > t).astype(float))
        design = np.column_stack(cols)
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        pred = design @ beta
        rss = float(np.sum((y - pred) ** 2))
        return beta, pred, rss

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        _, n_features = X.shape

        base_scores = np.array([self._safe_corr_score(X[:, j], y) for j in range(n_features)])
        pool_k = min(int(self.candidate_features), n_features)
        pool = [int(j) for j in np.argsort(-base_scores)[:pool_k]]
        self.feature_pool_ = np.asarray(pool, dtype=int)

        linear_terms = []
        rule_terms = []
        beta, pred, best_rss = self._fit_with_terms(X, y, linear_terms, rule_terms)

        for _ in range(int(self.max_linear_terms)):
            remaining = [j for j in pool if j not in linear_terms]
            if not remaining:
                break

            residual = y - pred
            best_j = max(remaining, key=lambda j: self._safe_corr_score(X[:, j], residual))

            trial_linear = linear_terms + [best_j]
            trial_beta, trial_pred, trial_rss = self._fit_with_terms(X, y, trial_linear, rule_terms)
            gain = (best_rss - trial_rss) / (best_rss + 1e-12)
            if gain < self.min_residual_gain:
                break

            linear_terms = trial_linear
            beta, pred, best_rss = trial_beta, trial_pred, trial_rss

        rule_candidates = []
        for j in pool:
            xj = X[:, j]
            for q in self.quantiles:
                t = float(np.quantile(xj, q))
                rule_candidates.append((int(j), t))

        for _ in range(int(self.max_rule_terms)):
            if not rule_candidates:
                break

            residual = y - pred
            best_rule = max(
                rule_candidates,
                key=lambda rt: self._safe_corr_score((X[:, rt[0]] > rt[1]).astype(float), residual),
            )

            trial_rules = rule_terms + [best_rule]
            trial_beta, trial_pred, trial_rss = self._fit_with_terms(X, y, linear_terms, trial_rules)
            gain = (best_rss - trial_rss) / (best_rss + 1e-12)
            if gain < self.min_residual_gain:
                break

            rule_terms = trial_rules
            beta, pred, best_rss = trial_beta, trial_pred, trial_rss
            rule_candidates = [rt for rt in rule_candidates if rt != best_rule]

        self.linear_terms_ = linear_terms
        self.rule_terms_ = rule_terms
        self.intercept_ = float(beta[0])
        self.linear_coefs_ = np.asarray(beta[1 : 1 + len(linear_terms)], dtype=float)
        self.rule_coefs_ = np.asarray(beta[1 + len(linear_terms) :], dtype=float)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        X = np.asarray(X, dtype=float)
        pred = np.full(X.shape[0], self.intercept_, dtype=float)

        for c, j in zip(self.linear_coefs_, self.linear_terms_):
            pred += c * X[:, j]
        for c, (j, t) in zip(self.rule_coefs_, self.rule_terms_):
            pred += c * (X[:, j] > t).astype(float)
        return pred

    def __str__(self):
        check_is_fitted(self, "is_fitted_")
        lines = [
            "Residual Rule Linear Regressor:",
            "  y = intercept + sum(a_j * x_j) + sum(b_k * I[x_j > t_k])",
            f"  intercept: {self.intercept_:+.4f}",
        ]

        if self.linear_terms_:
            lines.append("  Linear terms:")
            for idx, (c, j) in enumerate(zip(self.linear_coefs_, self.linear_terms_), 1):
                lines.append(f"    {idx}. {c:+.4f} * x{j}")
        else:
            lines.append("  Linear terms: none")

        if self.rule_terms_:
            lines.append("  Residual rule terms:")
            for idx, (c, (j, t)) in enumerate(zip(self.rule_coefs_, self.rule_terms_), 1):
                lines.append(f"    {idx}. {c:+.4f} * I[x{j} > {t:.4f}]")
        else:
            lines.append("  Residual rule terms: none")

        eq = f"{self.intercept_:+.4f}"
        for c, j in zip(self.linear_coefs_, self.linear_terms_):
            eq += f" {c:+.4f}*x{j}"
        for c, (j, t) in zip(self.rule_coefs_, self.rule_terms_):
            eq += f" {c:+.4f}*I[x{j}>{t:.4f}]"
        lines.append(f"  Equation: y = {eq}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
ResidualRuleLinearRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "ResidualRuleLin_v1"
model_description = "Greedy sparse linear backbone with residual quantile-threshold rule corrections"
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
