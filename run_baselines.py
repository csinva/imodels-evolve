"""
Run interpretability tests and TabArena regression evaluation on a unified
set of regressors, then produce a combined performance plot.

Usage: uv run run_baselines.py
Outputs (all under results/):
  all_scores.json                    — interpretability fraction per model
  interpretability_per_test_results.csv
  tabarena_scores.json               — avg rank + mean RMSE per model
  tabarena_results.csv               — per-dataset per-model RMSE
  interpretability_vs_performance.png
"""

import csv
import json
import os
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LassoCV, LinearRegression, RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "eval"))
from interp_eval import ALL_TESTS, HARD_TESTS, INSIGHT_TESTS, run_all_interp_tests
from performance import evaluate_all_regressors, compute_rank_scores, RESULTS_DIR, upsert_overall_results

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

REGRESSOR_DEFS = [
    ("DT_mini",    DecisionTreeRegressor(max_leaf_nodes=8,  random_state=42)),
    ("DT_large",   DecisionTreeRegressor(max_leaf_nodes=20, random_state=42)),
    ("OLS",        LinearRegression()),
    ("LassoCV",    LassoCV(cv=3)),
    ("RidgeCV",    RidgeCV(cv=3)),
    ("RF",         RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)),
    ("GBM",        GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)),
    ("MLP",        MLPRegressor(random_state=42)),
]

try:
    from pygam import LinearGAM
    REGRESSOR_DEFS = [("GAM", LinearGAM(n_splines=10))] + REGRESSOR_DEFS
except ImportError:
    pass

try:
    from imodels import (
        FIGSRegressor,
        HSTreeRegressorCV,
        RuleFitRegressor,
        TreeGAMRegressor,
    )
    REGRESSOR_DEFS += [
        ("FIGS_mini",    FIGSRegressor(max_rules=8,  random_state=42)),
        ("FIGS_large",   FIGSRegressor(max_rules=20, random_state=42)),
        ("RuleFit",      RuleFitRegressor(max_rules=20, random_state=42)),
        ("HSTree_mini",  HSTreeRegressorCV(max_leaf_nodes=8,  random_state=42)),
        ("HSTree_large", HSTreeRegressorCV(max_leaf_nodes=20, random_state=42)),
        ("TreeGAM",      TreeGAMRegressor(n_boosting_rounds=5, max_leaf_nodes=4, random_state=42)),
    ]
except ImportError:
    pass

MODEL_GROUPS = {
    "black-box": {"RF", "GBM", "MLP"},
    "imodels":   {"FIGS_mini", "FIGS_large", "RuleFit", "HSTree_mini", "HSTree_large", "TreeGAM"},
    "linear":    {"OLS", "LASSO", "LassoCV", "RidgeCV"},
    "tree":      {"DT_mini", "DT_large"},
    "gam":       {"GAM"},
}
GROUP_COLORS = {
    "black-box": "#e74c3c",
    "imodels":   "#27ae60",
    "linear":    "#2980b9",
    "tree":      "#e67e22",
    "gam":       "#8e44ad",
}

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _model_color(name):
    for group, members in MODEL_GROUPS.items():
        if name in members:
            return GROUP_COLORS[group]
    return "#7f8c8d"


def plot_interp_vs_tabarena(interp_results, tabarena_csv_path, out_path):
    """Scatter: x=TabArena mean RMSE (±1 SEM across datasets), y=interpretability tests passed."""
    from adjustText import adjust_text

    tabarena_rmses = {}
    with open(tabarena_csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["rmse"]:
                tabarena_rmses.setdefault(row["model"], []).append(float(row["rmse"]))

    mean_rmse = {m: float(np.mean(v)) for m, v in tabarena_rmses.items()}
    sem_rmse  = {m: float(np.std(v) / np.sqrt(len(v))) for m, v in tabarena_rmses.items()}

    model_names = list(dict.fromkeys(r["model"] for r in interp_results))
    n_passed = {n: sum(r["passed"] for r in interp_results if r["model"] == n)
                for n in model_names}

    names  = [n for n in model_names if n in mean_rmse]
    x      = np.array([mean_rmse[n] for n in names])
    x_err  = np.array([sem_rmse[n]  for n in names])
    y      = np.array([n_passed[n] for n in names])
    colors = [_model_color(n) for n in names]

    fig, ax = plt.subplots(figsize=(10, 6))
    for xi, yi, xe, color in zip(x, y, x_err, colors):
        ax.errorbar(xi, yi, xerr=xe, fmt="o", color=color,
                    ecolor=color, elinewidth=1.2, capsize=4,
                    markersize=8, markeredgecolor="white", markeredgewidth=0.6,
                    zorder=5)

    texts = [ax.text(xi, yi, name, fontsize=8.5) for xi, yi, name in zip(x, y, names)]
    ax.set_xlim(left=min(x) - 0.05, right=1.1) 
    adjust_text(texts, x=x, y=y, ax=ax,
                arrowprops=dict(arrowstyle="-", color="grey", lw=0.6))

    ax.set_xlabel("TabArena Mean RMSE", fontsize=10)
    ax.set_ylabel("Interpretability Tests Passed (out of 18)", fontsize=10)
    ax.set_title("Interpretability vs. TabArena Performance", fontsize=12, fontweight="bold")
    
    ax.grid(True, alpha=0.3)

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=GROUP_COLORS[g], markersize=9,
               label=g.replace("-", " ").title())
        for g in GROUP_COLORS if any(n in MODEL_GROUPS[g] for n in names)
    ]
    ax.legend(handles=legend_handles, fontsize=9) #, loc="upper left")
    

    # plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")


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

    with open(os.path.join(RESULTS_DIR, "all_scores.json"), "w") as f:
        json.dump({"interpretability": interp_scores}, f, indent=2)

    csv_path = os.path.join(RESULTS_DIR, "interpretability_per_test_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "test", "suite", "passed",
                                               "ground_truth", "response"],
                                extrasaction="ignore")
        writer.writeheader()
        for r in interp_results:
            writer.writerow({**r, "suite": _suite(r["test"])})
    print(f"Per-test results saved → {csv_path}")

    # --- TabArena evaluation ---
    print("\n" + "="*60)
    print("  TABARENA EVALUATION")
    print("="*60)
    dataset_rmses = evaluate_all_regressors(REGRESSOR_DEFS)
    avg_rank, avg_rmse = compute_rank_scores(dataset_rmses)

    print("\n\nTabArena summary (sorted by avg rank):")
    for name, rank in sorted(avg_rank.items(), key=lambda x: x[1]):
        print(f"  {name:<15}: avg_rank={rank:.2f}  mean_rmse={avg_rmse.get(name, float('nan')):.4f}")

    with open(os.path.join(RESULTS_DIR, "tabarena_scores.json"), "w") as f:
        json.dump({"tabarena_avg_rank": avg_rank, "tabarena_mean_rmse": avg_rmse,
                   "tabarena_per_dataset": dataset_rmses}, f, indent=2)

    tabarena_csv = os.path.join(RESULTS_DIR, "tabarena_results.csv")
    with open(tabarena_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "model", "rmse", "rank"])
        for ds_name, model_rmses in dataset_rmses.items():
            valid = [(n, v) for n, v in model_rmses.items() if not np.isnan(v)]
            rank_map = {n: r + 1 for r, (n, _) in enumerate(
                sorted(valid, key=lambda x: x[1]))}  # ascending: lower RMSE = better
            for name, rmse in model_rmses.items():
                rank = rank_map.get(name, "")
                writer.writerow([ds_name, name, "" if np.isnan(rmse) else f"{rmse:.6f}", rank])
    print(f"Per-dataset results saved → {tabarena_csv}")

    # --- Overall results CSV ---
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        git_hash = ""

    overall_rows = [{
        "commit":                             'baseline',
        "mean_rmse":                          f"{avg_rmse[mname]:.6f}" if mname in avg_rmse else "",
        "frac_interpretability_tests_passed": f"{interp_scores[mname]:.4f}",
        "status":                             "baseline",
        "description":                        mname,
    } for mname in model_names]
    upsert_overall_results(overall_rows, RESULTS_DIR)

    # --- Plot ---
    plot_interp_vs_tabarena(
        interp_results, tabarena_csv,
        os.path.join(RESULTS_DIR, "interpretability_vs_performance.png"),
    )

    print(f"\nTotal time: {time.time() - t0:.1f}s")
