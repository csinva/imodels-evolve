"""
Interpretable regressor autoresearch script.
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores, recompute_all_mean_ranks
from visualize import plot_interp_vs_performance


class ThresholdPruneOLS(BaseEstimator, RegressorMixin):
    """Full OLS, then prune features with |std-coef| < 1% of max, refit OLS."""

    def __init__(self, threshold_frac=0.01):
        self.threshold_frac = threshold_frac

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]
        mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        sigma_safe = np.where(sigma > 1e-12, sigma, 1.0)
        Xs = (X - mu) / sigma_safe
        lr_std = LinearRegression().fit(Xs, y)
        std_coef = np.abs(lr_std.coef_)
        max_std = std_coef.max() if std_coef.size else 0.0
        keep = std_coef >= max(self.threshold_frac * max_std, 1e-10)
        if not keep.any():
            keep = np.array([True] * self.n_features_in_)
        self.selected_ = np.where(keep)[0].tolist()
        lr = LinearRegression().fit(X[:, self.selected_], y)
        self.coef_ = lr.coef_
        self.intercept_ = float(lr.intercept_)
        return self

    def predict(self, X):
        check_is_fitted(self, "selected_")
        X = np.asarray(X, dtype=float)
        return X[:, self.selected_] @ self.coef_ + self.intercept_

    def __str__(self):
        check_is_fitted(self, "selected_")
        names = [f"x{i}" for i in range(self.n_features_in_)]
        eqn_parts = [f"{c:+.4f}*{names[i]}" for i, c in zip(self.selected_, self.coef_)]
        eqn = " ".join(eqn_parts) if eqn_parts else "0"
        lines = [
            "OLS Linear Regression (clean; near-zero coefficients pruned):",
            f"  y = {self.intercept_:+.4f} {eqn}",
            "",
            "Coefficients (exact linear effects per unit change in feature):",
        ]
        for i, c in zip(self.selected_, self.coef_):
            lines.append(f"  {names[i]}: {c:+.4f}")
        excluded = [names[i] for i in range(self.n_features_in_) if i not in self.selected_]
        if excluded:
            lines.append(f"  excluded (negligible effect): {', '.join(excluded)}")
        lines.append(f"  intercept: {self.intercept_:+.4f}")
        return "\n".join(lines)


import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
ThresholdPruneOLS.__module__ = "interpretable_regressor"

model_shorthand_name = "ThresholdPruneOLS_1pct_v2"
model_description = "OLS, prune features with standardized coef below 1% of max, OLS refit"
model_defs = [(model_shorthand_name, ThresholdPruneOLS(threshold_frac=0.01))]


# ---------------------------------------------------------------------------
# Evaluation (do not edit anything below this line)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default='gpt-4o',
                        help="LLM checkpoint for interpretability tests (default: gpt-4o)")
    args = parser.parse_args()

    t0 = time.time()

    interp_results = run_all_interp_tests(model_defs, checkpoint=args.checkpoint)
    n_passed = sum(r["passed"] for r in interp_results)
    total = len(interp_results)

    dataset_rmses = evaluate_all_regressors(model_defs)

    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        git_hash = ""

    model_name = model_defs[0][0]
    interp_csv = os.path.join(RESULTS_DIR, "interpretability_results.csv")
    interp_fields = ["model", "test", "suite", "passed", "ground_truth", "response"]

    def _suite(test_name):
        if test_name.startswith("insight_"): return "insight"
        if test_name.startswith("hard_"):    return "hard"
        return "standard"

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

    perf_csv = os.path.join(RESULTS_DIR, "performance_results.csv")
    perf_fields = ["dataset", "model", "rmse", "rank"]

    existing_perf = []
    if os.path.exists(perf_csv):
        with open(perf_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("model") != model_name:
                    existing_perf.append(row)

    for ds_name, model_rmses in dataset_rmses.items():
        rmse_val = model_rmses.get(model_name, float("nan"))
        existing_perf.append({
            "dataset": ds_name,
            "model": model_name,
            "rmse": "" if np.isnan(rmse_val) else f"{rmse_val:.6f}",
            "rank": "",
        })

    by_dataset = defaultdict(list)
    for row in existing_perf:
        by_dataset[row["dataset"]].append(row)

    for ds_name, rows in by_dataset.items():
        valid = [(r, float(r["rmse"])) for r in rows if r["rmse"] not in ("", None)]
        valid.sort(key=lambda x: x[1])
        for rank_idx, (r, _) in enumerate(valid, 1):
            r["rank"] = rank_idx
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

    recompute_all_mean_ranks(RESULTS_DIR)

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
