"""
Run interpretability tests and performance regression evaluation on a unified
set of regressors, then produce a combined performance plot.

Usage: uv run run_baselines.py
Outputs (all under results/):
  interpretability_results.csv
  performance_results.csv               — per-dataset per-model RMSE
  overall_results.csv                — summary of overall scores (for leaderboard)
  interpretability_vs_performance.png
"""

import csv
import os
import subprocess
import sys
import time

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LassoCV, LinearRegression, RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from pygam import LinearGAM
from imodels import (
    FIGSRegressor,
    HSTreeRegressorCV,
    RuleFitRegressor,
)
from interpret.glassbox import ExplainableBoostingRegressor
from tabpfn import TabPFNRegressor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import ALL_TESTS, HARD_TESTS, INSIGHT_TESTS, run_all_interp_tests
from performance_eval import evaluate_all_regressors, compute_rank_scores, RESULTS_DIR, upsert_overall_results
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

REGRESSOR_DEFS = [
    ("PyGAM",      LinearGAM(n_splines=10)),
    ("DT_mini",    DecisionTreeRegressor(max_leaf_nodes=8,  random_state=42)),
    ("DT_large",   DecisionTreeRegressor(max_leaf_nodes=20, random_state=42)),
    ("OLS",        LinearRegression()),
    ("LassoCV",    LassoCV(cv=3)),
    ("RidgeCV",    RidgeCV(cv=3)),
    ("RF",         RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)),
    ("GBM",        GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)),
    ("MLP",        MLPRegressor(random_state=42)),
    ("FIGS_mini",    FIGSRegressor(max_rules=8,  random_state=42)),
    ("FIGS_large",   FIGSRegressor(max_rules=20, random_state=42)),
    ("RuleFit",      RuleFitRegressor(max_rules=20, random_state=42)),
    ("HSTree_mini",  HSTreeRegressorCV(max_leaf_nodes=8,  random_state=42)),
    ("HSTree_large", HSTreeRegressorCV(max_leaf_nodes=20, random_state=42)),
    ("EBM",          ExplainableBoostingRegressor(random_state=42, outer_bags=3, max_rounds=1000)),
    ("TabPFN",       TabPFNRegressor(device="cpu", random_state=42)),
]

# Human-readable descriptions for each model
MODEL_DESCRIPTIONS = {
    "PyGAM":       "generalized additive model with 10 splines per feature and default settings",
    "DT_mini":     "small decision tree with up to 8 max_leaf_nodes",
    "DT_large":    "large decision tree with up to 20 max_leaf_nodes",
    "OLS":         "ordinary least squares linear regression",
    "LassoCV":     "Lasso linear model with cross-validation to select the regularization parameter",
    "RidgeCV":     "Ridge linear model with cross-validation to select the regularization parameter",
    "RF":          "random forest with 50 tree estimators, each with max_depth of 5",
    "GBM":         "gradient boosting machine with 100 tree estimators, each with max_depth of 3",
    "MLP":         "multi-layer perceptron with default hidden layer size, ReLU activation, and Adam solver",
    "FIGS_mini":   "small FIGS with up to 8 max_rules",
    "FIGS_large":  "large FIGS with up to 20 max_rules",
    "RuleFit":     "RuleFit with up to 20 max_rules",
    "HSTree_mini": "small HSTree with up to 8 max_leaf_nodes",
    "HSTree_large":"large HSTree with up to 20 max_leaf_nodes",
    "EBM":         "explainable boosting machine (InterpretML) with 3 outer bags and 1000 max rounds",
    "TabPFN":      "TabPFN foundation model for tabular data",
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _suite(test_name):
    if test_name.startswith("insight_"): return "insight"
    if test_name.startswith("hard_"):    return "hard"
    return "standard"


if __name__ == "__main__":
    t0 = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Interpretability tests ---
    print("\n" + "="*60)
    print("  INTERPRETABILITY TESTS")
    print("="*60)
    interp_results = run_all_interp_tests(REGRESSOR_DEFS)

    model_names = list(dict.fromkeys(r["model"] for r in interp_results))
    interp_scores = {m: sum(r["passed"] for r in interp_results if r["model"] == m)
                        / len([r for r in interp_results if r["model"] == m])
                     for m in model_names}

    print("\n\nInterpretability scores (all tests):")
    for name, score in sorted(interp_scores.items(), key=lambda x: -x[1]):
        subset = [r for r in interp_results if r["model"] == name]
        n = sum(r["passed"] for r in subset)
        std  = sum(r["passed"] for r in subset if r["test"] in {t.__name__ for t in ALL_TESTS})
        hard = sum(r["passed"] for r in subset if r["test"] in {t.__name__ for t in HARD_TESTS})
        ins  = sum(r["passed"] for r in subset if r["test"] in {t.__name__ for t in INSIGHT_TESTS})
        print(f"  {name:<15}: {n:2d}/{len(subset)} ({score:.2%})  "
              f"[std {std}/8  hard {hard}/5  insight {ins}/5]")

    csv_path = os.path.join(RESULTS_DIR, "interpretability_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "test", "suite", "passed",
                                               "ground_truth", "response"],
                                extrasaction="ignore")
        writer.writeheader()
        for r in interp_results:
            writer.writerow({**r, "suite": _suite(r["test"])})
    print(f"Per-test results saved → {csv_path}")

    # --- performance evaluation ---
    print("\n" + "="*60)
    print("  performance EVALUATION")
    print("="*60)
    dataset_rmses = evaluate_all_regressors(REGRESSOR_DEFS)
    avg_rank, avg_rmse = compute_rank_scores(dataset_rmses)

    print("\n\nperformance summary (sorted by avg rank):")
    for name, rank in sorted(avg_rank.items(), key=lambda x: x[1]):
        print(f"  {name:<15}: avg_rank={rank:.2f}  mean_rmse={avg_rmse.get(name, float('nan')):.4f}")

    performance_csv = os.path.join(RESULTS_DIR, "performance_results.csv")
    with open(performance_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "model", "rmse", "rank"])
        for ds_name, model_rmses in dataset_rmses.items():
            valid = [(n, v) for n, v in model_rmses.items() if not np.isnan(v)]
            rank_map = {n: r + 1 for r, (n, _) in enumerate(
                sorted(valid, key=lambda x: x[1]))}  # ascending: lower RMSE = better
            for name, rmse in model_rmses.items():
                rank = rank_map.get(name, "")
                writer.writerow([ds_name, name, "" if np.isnan(rmse) else f"{rmse:.6f}", rank])
    print(f"Per-dataset results saved → {performance_csv}")

    # --- Overall results CSV ---
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        git_hash = ""

    overall_rows = [{
        "commit":                             'baseline',
        "mean_rank":                          f"{avg_rank[mname]:.2f}" if mname in avg_rank else "nan",
        "frac_interpretability_tests_passed": f"{interp_scores[mname]:.4f}" if mname in interp_scores else "nan",
        "status":                             "baseline",
        "model_name":                         mname,
        "description":                        MODEL_DESCRIPTIONS.get(mname, mname),
    } for mname in model_names]
    upsert_overall_results(overall_rows, RESULTS_DIR)

    # --- Plot ---
    overall_csv = os.path.join(RESULTS_DIR, "overall_results.csv")
    plot_interp_vs_performance(
        overall_csv,
        os.path.join(RESULTS_DIR, "interpretability_vs_performance.png"),
    )

    print(f"\nTotal time: {time.time() - t0:.1f}s")

    # also run the interpretable_regressor.py script here
    os.system("uv run interpretable_regressor.py")