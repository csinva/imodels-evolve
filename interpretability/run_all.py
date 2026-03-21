"""
Run ALL interpretability tests on a unified set of regressors.
Produces a bar chart of interpretability scores broken down by test suite.

Usage: uv run run_all.py
"""

import csv
import json
import time
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from joblib import Memory

import imodelsx.llm
from interpretability import (
    ALL_TESTS, HARD_TESTS, INSIGHT_TESTS, _HAS_IMODELS,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
_memory = Memory(location=os.path.join(RESULTS_DIR, "cache"), verbose=0)

# Registry populated at module level so the cached function can look up tests by name.
_ALL_TEST_FNS = {fn.__name__: fn for fn in ALL_TESTS + HARD_TESTS + INSIGHT_TESTS}


@_memory.cache
def _run_one_test(model_name, test_fn_name, model):
    """Run a single interpretability test and cache the result.

    Cache key is effectively (model_name, test_fn_name) since model params
    are determined by model_name. joblib also hashes `model` as a safety net.
    """
    llm = imodelsx.llm.get_llm("gpt-4o")
    test_fn = _ALL_TEST_FNS[test_fn_name]
    try:
        result = test_fn(model, llm)
    except AssertionError as e:
        result = dict(test=test_fn_name, passed=False, error=f"Assertion: {e}", response=None)
    except Exception as e:
        result = dict(test=test_fn_name, passed=False, error=str(e), response=None)
    result["model"] = model_name
    result.setdefault("test", test_fn_name)
    return result

# ---------------------------------------------------------------------------
# Model registry: (name, regressor)
# ---------------------------------------------------------------------------

MODEL_DEFS = []

from pygam import LinearGAM
MODEL_DEFS.append(("GAM", LinearGAM(n_splines=10)))

MODEL_DEFS += [
    ("DT_mini",    DecisionTreeRegressor(max_leaf_nodes=8,  random_state=42)),
    ("DT_large",   DecisionTreeRegressor(max_leaf_nodes=20, random_state=42)),
    ("OLS",        LinearRegression()),
    ("LASSO",      Lasso(alpha=0.1)),
    ("LassoCV",    LassoCV(cv=5)),
    ("RidgeCV",    RidgeCV()),
    ("RF",         RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)),
    ("GBM",        GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)),
    ("MLP",        MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=1000,
                                random_state=42, learning_rate_init=0.01)),
]

from imodels import (
    FIGSRegressor,
    HSTreeRegressor,
    RuleFitRegressor,
    TreeGAMRegressor,
)
MODEL_DEFS += [
    ("FIGS_mini",   FIGSRegressor(max_rules=8,  random_state=42)),
    ("FIGS_large",  FIGSRegressor(max_rules=20, random_state=42)),
    ("RuleFit",     RuleFitRegressor(max_rules=20, random_state=42)),
    ("HSTree_mini",  HSTreeRegressor(max_leaf_nodes=8,  random_state=42)),
    ("HSTree_large", HSTreeRegressor(max_leaf_nodes=20, random_state=42)),
    ("TreeGAM",     TreeGAMRegressor(n_boosting_rounds=5, max_leaf_nodes=4, random_state=42)),
]


# ---------------------------------------------------------------------------
# Interpretability tests
# ---------------------------------------------------------------------------

def run_all_interp_tests(model_defs):
    """Run standard + hard + insight tests on all regressors, with per-test caching."""
    all_results = []
    for name, reg in model_defs:
        print(f"\n{'='*60}")
        print(f"  Model: {name}")
        print("=" * 60)
        for test_list, label in [
            (ALL_TESTS,     "standard"),
            (HARD_TESTS,    "hard"),
            (INSIGHT_TESTS, "insight"),
        ]:
            print(f"\n  [{label}]")
            suite_results = []
            for test_fn in test_list:
                result = _run_one_test(name, test_fn.__name__, reg)
                status = "PASS" if result["passed"] else "FAIL"
                resp = (result.get("response") or "")[:80].replace("\n", " ")
                print(f"  [{status}] {result['test']}")
                print(f"         ground_truth : {result.get('ground_truth', '')}")
                print(f"         llm_response : {resp}")
                suite_results.append(result)
            n_passed = sum(r["passed"] for r in suite_results)
            print(f"\n  → {n_passed}/{len(test_list)} passed")
            all_results.extend(suite_results)
    return all_results


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

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


def _model_color(name):
    for group, members in MODEL_GROUPS.items():
        if name in members:
            return GROUP_COLORS[group]
    return "#7f8c8d"


def plot_interp_vs_test_rank(interp_results, out_path):
    """Scatter: x=mean per-test rank (lower=better), y=total tests passed, with SEM error bars.

    For each of the 18 tests, models are ranked 1..N by pass/fail (average rank for ties).
    The x-axis shows each model's mean rank across all tests; error bars show ±1 SEM.
    """
    from scipy.stats import rankdata
    from adjustText import adjust_text

    model_names = list(dict.fromkeys(r["model"] for r in interp_results))
    all_tests   = list(dict.fromkeys(r["test"]  for r in interp_results))

    # per_test_ranks[model] = list of ranks (one per test)
    per_test_ranks = {name: [] for name in model_names}
    for test_name in all_tests:
        scores = {r["model"]: float(r["passed"])
                  for r in interp_results if r["test"] == test_name}
        if not scores:
            continue
        names_in_test = list(scores.keys())
        vals = np.array([scores[n] for n in names_in_test])
        # Rank descending (higher score = better = lower rank number)
        ranks = rankdata(-vals, method="average")
        for name, rank in zip(names_in_test, ranks):
            per_test_ranks[name].append(rank)

    mean_rank = {n: float(np.mean(v)) for n, v in per_test_ranks.items() if v}
    sem_rank  = {n: float(np.std(v) / np.sqrt(len(v))) for n, v in per_test_ranks.items() if v}
    n_passed  = {n: sum(r["passed"] for r in interp_results if r["model"] == n)
                 for n in model_names}

    names  = [n for n in model_names if n in mean_rank]
    x      = np.array([mean_rank[n] for n in names])
    x_err  = np.array([sem_rank[n]  for n in names])
    y      = np.array([n_passed[n]  for n in names])
    colors = [_model_color(n) for n in names]

    fig, ax = plt.subplots(figsize=(10, 6))
    for xi, yi, xe, color in zip(x, y, x_err, colors):
        ax.errorbar(xi, yi, xerr=xe, fmt="o", color=color,
                    ecolor=color, elinewidth=1.2, capsize=4,
                    markersize=8, markeredgecolor="white", markeredgewidth=0.6,
                    zorder=5)

    texts = [ax.text(xi, yi, name, fontsize=8.5)
             for xi, yi, name in zip(x, y, names)]
    adjust_text(texts, x=x, y=y, ax=ax,
                arrowprops=dict(arrowstyle="-", color="grey", lw=0.6))

    ax.set_xlabel("Mean Per-Test Rank (lower = better, ±1 SEM)", fontsize=10)
    ax.set_ylabel("Total Tests Passed (out of 18)", fontsize=10)
    ax.set_title("Interpretability: Tests Passed vs. Mean Test Rank", fontsize=12, fontweight="bold")
    ax.invert_xaxis()  # leftmost = best rank
    ax.grid(True, alpha=0.3)

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=GROUP_COLORS[g], markersize=9,
               label=g.replace("-", " ").title())
        for g in GROUP_COLORS if any(n in MODEL_GROUPS[g] for n in names)
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="lower left")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()

    interp_results = run_all_interp_tests(MODEL_DEFS)

    model_names = list(dict.fromkeys(r["model"] for r in interp_results))
    interp_scores = {}
    for mname in model_names:
        subset = [r for r in interp_results if r["model"] == mname]
        interp_scores[mname] = sum(r["passed"] for r in subset) / len(subset)

    print("\n\nInterpretability scores (all tests):")
    for name, score in sorted(interp_scores.items(), key=lambda x: -x[1]):
        subset = [r for r in interp_results if r["model"] == name]
        n = sum(r["passed"] for r in subset)
        total = len(subset)
        std  = sum(r["passed"] for r in subset if r["test"] in {t.__name__ for t in ALL_TESTS})
        hard = sum(r["passed"] for r in subset if r["test"] in {t.__name__ for t in HARD_TESTS})
        ins  = sum(r["passed"] for r in subset if r["test"] in {t.__name__ for t in INSIGHT_TESTS})
        print(f"  {name:<15}: {n:2d}/{total} ({score:.2%})  "
              f"[std {std}/8  hard {hard}/5  insight {ins}/5]")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    scores_path = os.path.join(RESULTS_DIR, "all_scores.json")
    with open(scores_path, "w") as f:
        json.dump({"interpretability": interp_scores}, f, indent=2)
    print(f"\nScores saved → {scores_path}")

    # Save per-test per-model results to CSV
    csv_path = os.path.join(RESULTS_DIR, "per_test_results.csv")
    csv_cols = ["model", "test", "suite", "passed", "ground_truth", "response"]
    def _suite(test_name):
        if test_name.startswith("insight_"): return "insight"
        if test_name.startswith("hard_"):    return "hard"
        return "standard"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_cols, extrasaction="ignore")
        writer.writeheader()
        for r in interp_results:
            writer.writerow({**r, "suite": _suite(r["test"])})
    print(f"Per-test results saved → {csv_path}")

    plot_path = os.path.join(RESULTS_DIR, "interpretability_vs_performance.png")
    plot_interp_vs_test_rank(interp_results, plot_path)

    print(f"\nTotal time: {time.time() - t0:.1f}s")
