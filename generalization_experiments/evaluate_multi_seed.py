"""
Multi-seed evaluation of interpretability tests for both gpt-4o and gpt-5.4.

Runs all interpretability tests 3 times with different LLM seeds for each
checkpoint, then averages results to reduce randomness. Performance evaluation
is run once (deterministic, doesn't depend on LLM).

Saves results to new_results_gpt_5_4/ with averaged scores and comparison plots.

Usage: uv run evaluate_multi_seed.py
"""

import csv
import importlib.util
import os
import re
import sys
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import r2_score

# Add src/ to path
SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "evolve", "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "evolve"))

from performance_eval import (
    MAX_SAMPLES, MAX_FEATURES, MIN_SAMPLES, MIN_FEATURES,
    subsample_dataset, OVERALL_CSV_COLS,
)
from interp_eval import get_model_str, ask_llm, _safe_clone
from visualize import plot_interp_vs_performance

import imodelsx.llm

# Baseline model imports
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV as SkLassoCV, LinearRegression, RidgeCV as SkRidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from pygam import LinearGAM
from imodels import FIGSRegressor, HSTreeRegressorCV, RuleFitRegressor
from interpret.glassbox import ExplainableBoostingRegressor
from tabpfn import TabPFNRegressor

# Import test functions from evaluate_new_generalization
from evaluate_new_generalization import (
    ALL_NEW_TESTS,
    NEW_STANDARD_TESTS, NEW_HARD_TESTS, NEW_INSIGHT_TESTS,
    NEW_DISCRIM_TESTS, NEW_SIMULATABILITY_TESTS,
    _ALL_NEW_TEST_FNS,
    load_all_models,
    evaluate_new_performance,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHECKPOINTS = ["gpt-4o", "gpt-5.4"]
SEEDS = [1, 2, 3]

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "new_results_gpt_5_4"
RESULTS_DIR.mkdir(exist_ok=True)

BASELINE_DEFS = [
    ("PyGAM",      LinearGAM(n_splines=10)),
    ("DT_mini",    DecisionTreeRegressor(max_leaf_nodes=8,  random_state=42)),
    ("DT_large",   DecisionTreeRegressor(max_leaf_nodes=20, random_state=42)),
    ("OLS",        LinearRegression()),
    ("LassoCV",    SkLassoCV(cv=3)),
    ("RidgeCV",    SkRidgeCV(cv=3)),
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

BASELINE_DESCRIPTIONS = {
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
# Seeded LLM wrapper
# ---------------------------------------------------------------------------

class SeededLLM:
    """Wrapper that forces a specific seed on every LLM call."""
    def __init__(self, llm, seed):
        self._llm = llm
        self._seed = seed

    def __call__(self, *args, **kwargs):
        kwargs['seed'] = self._seed
        return self._llm(*args, **kwargs)


# ---------------------------------------------------------------------------
# Cached test runner with checkpoint + seed
# ---------------------------------------------------------------------------

from joblib import Memory
_multi_cache = Memory(location=os.path.join(str(RESULTS_DIR), "cache_multi_seed"), verbose=0)


@_multi_cache.cache
def _run_one_test_seeded(model_name, test_fn_name, model, checkpoint, seed):
    """Run a single interpretability test with a specific checkpoint and seed."""
    llm = SeededLLM(imodelsx.llm.get_llm(checkpoint), seed)
    test_fn = _ALL_NEW_TEST_FNS[test_fn_name]
    try:
        result = test_fn(model, llm)
    except AssertionError as e:
        result = dict(test=test_fn_name, passed=False, error=f"Assertion: {e}", response=None)
    except Exception as e:
        result = dict(test=test_fn_name, passed=False, error=str(e), response=None)
    result["model"] = model_name
    result.setdefault("test", test_fn_name)
    return result


def run_interp_tests_seeded(model_defs_simple, checkpoint, seed):
    """Run all interpretability tests on all models with a specific checkpoint and seed."""
    from joblib import Parallel, delayed

    tasks = [(name, reg, test_fn)
             for name, reg in model_defs_simple
             for test_fn in ALL_NEW_TESTS]

    print(f"\n  Running {len(tasks)} tests with checkpoint={checkpoint}, seed={seed}...")

    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_run_one_test_seeded)(name, test_fn.__name__, reg, checkpoint, seed)
        for name, reg, test_fn in tasks
    )

    # Summary
    n_passed = sum(r["passed"] for r in results)
    print(f"  Done: {n_passed}/{len(results)} passed "
          f"({n_passed/len(results)*100:.1f}%) [checkpoint={checkpoint}, seed={seed}]")

    return results


# ---------------------------------------------------------------------------
# Suite label helper
# ---------------------------------------------------------------------------

def _suite(test_name):
    if test_name.startswith("new_insight_"): return "insight"
    if test_name.startswith("new_hard_"):    return "hard"
    if test_name.startswith("new_discrim_"): return "discrim"
    if test_name.startswith("new_simulatability_"): return "simulatability"
    return "standard"


# ---------------------------------------------------------------------------
# Comparison visualization
# ---------------------------------------------------------------------------

def plot_multi_seed_comparison(avg_results, out_dir):
    """Create comprehensive comparison visualizations for multi-seed results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # avg_results: dict of checkpoint -> {model_name: {"mean_rank": ..., "interp_mean": ..., "interp_std": ...}}

    # --- Plot 1: Side-by-side scatter with error bars ---
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    colors = {"gpt-4o": "steelblue", "gpt-5.4": "coral"}
    markers = {"gpt-4o": "o", "gpt-5.4": "^"}

    ax = axes[0]
    for ckpt in CHECKPOINTS:
        data = avg_results[ckpt]
        names = list(data.keys())
        x = [data[n]["interp_mean"] for n in names]
        y = [data[n]["mean_rank"] for n in names]
        xerr = [data[n]["interp_std"] for n in names]
        ax.errorbar(x, y, xerr=xerr, fmt=markers[ckpt], color=colors[ckpt],
                    alpha=0.6, label=ckpt, markersize=6, capsize=2, linewidth=0.5)
    ax.set_xlabel("Fraction Interpretability Tests Passed (avg over 3 seeds)")
    ax.set_ylabel("Mean Rank (lower is better)")
    ax.set_title("Interpretability vs Performance (Multi-Seed Average)")
    ax.invert_yaxis()
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Plot 2: Paired comparison of interp scores ---
    ax = axes[1]
    common_models = sorted(set(avg_results["gpt-4o"].keys()) & set(avg_results["gpt-5.4"].keys()))
    if common_models:
        x_4o = [avg_results["gpt-4o"][m]["interp_mean"] for m in common_models]
        y_54 = [avg_results["gpt-5.4"][m]["interp_mean"] for m in common_models]
        xerr = [avg_results["gpt-4o"][m]["interp_std"] for m in common_models]
        yerr = [avg_results["gpt-5.4"][m]["interp_std"] for m in common_models]

        ax.errorbar(x_4o, y_54, xerr=xerr, yerr=yerr, fmt='o', color='teal',
                    alpha=0.6, markersize=5, capsize=2, linewidth=0.5)
        lims = [0, max(max(x_4o), max(y_54)) * 1.1]
        ax.plot(lims, lims, '--', color='gray', alpha=0.5, label="y=x (no change)")
        ax.set_xlabel("Interp Score (gpt-4o, avg 3 seeds)")
        ax.set_ylabel("Interp Score (gpt-5.4, avg 3 seeds)")
        ax.set_title("Interpretability: gpt-4o vs gpt-5.4 (with std dev)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Label outliers
        for i, m in enumerate(common_models):
            diff = y_54[i] - x_4o[i]
            if abs(diff) > 0.12:
                ax.annotate(m, (x_4o[i], y_54[i]), fontsize=6, alpha=0.7)

    # --- Plot 3: Per-model interp difference with error bars ---
    ax = axes[2]
    if common_models:
        diffs = []
        for m in common_models:
            d4o = avg_results["gpt-4o"][m]
            d54 = avg_results["gpt-5.4"][m]
            diff = d54["interp_mean"] - d4o["interp_mean"]
            # Combined std (approximate)
            std = np.sqrt(d4o["interp_std"]**2 + d54["interp_std"]**2)
            diffs.append((m, diff, std))

        diffs.sort(key=lambda x: x[1])
        n_show = min(15, len(diffs))
        show = diffs[:n_show] + diffs[-n_show:]
        # deduplicate
        seen = set()
        show_dedup = []
        for item in show:
            if item[0] not in seen:
                seen.add(item[0])
                show_dedup.append(item)
        show = show_dedup

        bar_colors = ["coral" if d < 0 else "steelblue" for _, d, _ in show]
        y_pos = range(len(show))
        ax.barh(y_pos, [d for _, d, _ in show], xerr=[s for _, _, s in show],
                color=bar_colors, alpha=0.7, capsize=2)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([m for m, _, _ in show], fontsize=6)
        ax.set_xlabel("Change in Interp Score (gpt-5.4 - gpt-4o)")
        ax.set_title("Biggest Changes (with std dev error bars)")
        ax.axvline(x=0, color='gray', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    out_path = os.path.join(out_dir, "comparison_multi_seed.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved -> {out_path}")

    # --- Plot 4: Per-seed variability ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for idx, ckpt in enumerate(CHECKPOINTS):
        ax = axes[idx]
        data = avg_results[ckpt]
        names = sorted(data.keys(), key=lambda n: -data[n]["interp_mean"])
        means = [data[n]["interp_mean"] for n in names]
        stds = [data[n]["interp_std"] for n in names]

        # Show top 20 by interp score
        n_show = min(25, len(names))
        ax.barh(range(n_show), means[:n_show], xerr=stds[:n_show],
                color=colors[ckpt], alpha=0.7, capsize=2)
        ax.set_yticks(range(n_show))
        ax.set_yticklabels(names[:n_show], fontsize=6)
        ax.set_xlabel("Interp Score (mean +/- std over 3 seeds)")
        ax.set_title(f"Top {n_show} models — {ckpt}")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    out_path = os.path.join(out_dir, "per_seed_variability.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Variability plot saved -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()

    # --- Load models ---
    print("=" * 60)
    print("  Loading models")
    print("=" * 60)
    model_defs_full = load_all_models()
    print(f"\nLoaded {len(model_defs_full)} evolved models.\n")

    for bname, bmodel in BASELINE_DEFS:
        model_defs_full.append((bname, bmodel, BASELINE_DESCRIPTIONS.get(bname, bname)))

    model_defs_simple = [(name, model) for name, model, _ in model_defs_full]
    print(f"Total models: {len(model_defs_simple)}\n")

    # --- Run interpretability tests for each checkpoint x seed ---
    # all_interp_results[checkpoint][seed] = list of result dicts
    all_interp_results = {ckpt: {} for ckpt in CHECKPOINTS}

    for ckpt in CHECKPOINTS:
        print("\n" + "=" * 60)
        print(f"  INTERPRETABILITY TESTS — {ckpt}")
        print("=" * 60)
        for seed in SEEDS:
            all_interp_results[ckpt][seed] = run_interp_tests_seeded(
                model_defs_simple, ckpt, seed
            )

    # --- Compute per-model averaged interp scores ---
    # avg_interp[checkpoint][model_name] = {"interp_mean": ..., "interp_std": ..., "per_seed": [...]}
    avg_interp = {ckpt: {} for ckpt in CHECKPOINTS}
    baseline_names = {bname for bname, _ in BASELINE_DEFS}

    for ckpt in CHECKPOINTS:
        for name, _, _ in model_defs_full:
            seed_scores = []
            for seed in SEEDS:
                results = all_interp_results[ckpt][seed]
                model_results = [r for r in results if r["model"] == name]
                if model_results:
                    frac = sum(r["passed"] for r in model_results) / len(model_results)
                    seed_scores.append(frac)
            if seed_scores:
                avg_interp[ckpt][name] = {
                    "interp_mean": float(np.mean(seed_scores)),
                    "interp_std": float(np.std(seed_scores)),
                    "per_seed": seed_scores,
                }

    # --- Performance evaluation (run once, deterministic) ---
    print("\n" + "=" * 60)
    print("  PERFORMANCE EVALUATION (16 OpenML datasets)")
    print("=" * 60)
    dataset_rmses = evaluate_new_performance(model_defs_simple)

    # Compute ranks
    def _lenient_rank_scores(dataset_rmses):
        all_model_names = set()
        for d in dataset_rmses.values():
            all_model_names.update(d.keys())
        ranks_per_model = {n: [] for n in all_model_names}
        rmse_per_model = {n: [] for n in all_model_names}
        for ds_name, model_rmses in dataset_rmses.items():
            valid = [(n, v) for n, v in model_rmses.items() if not np.isnan(v)]
            sorted_models = sorted(valid, key=lambda x: x[1])
            rank_map = {n: r + 1 for r, (n, _) in enumerate(sorted_models)}
            for name in all_model_names:
                if name in model_rmses and not np.isnan(model_rmses[name]):
                    ranks_per_model[name].append(rank_map[name])
                    rmse_per_model[name].append(model_rmses[name])
        avg_rank = {n: float(np.mean(v)) if v else float("nan")
                    for n, v in ranks_per_model.items()}
        avg_rmse = {n: float(np.mean(v)) if v else float("nan")
                    for n, v in rmse_per_model.items()}
        return avg_rank, avg_rmse

    avg_rank, avg_rmse = _lenient_rank_scores(dataset_rmses)

    # --- Build avg_results combining interp + perf for visualization ---
    avg_results = {ckpt: {} for ckpt in CHECKPOINTS}
    for ckpt in CHECKPOINTS:
        for name in avg_interp[ckpt]:
            avg_results[ckpt][name] = {
                **avg_interp[ckpt][name],
                "mean_rank": avg_rank.get(name, float("nan")),
            }

    # --- Save per-seed interpretability_results.csv ---
    for ckpt in CHECKPOINTS:
        ckpt_tag = ckpt.replace(".", "").replace("-", "")
        for seed in SEEDS:
            csv_path = str(RESULTS_DIR / f"interpretability_results_{ckpt_tag}_seed{seed}.csv")
            results = all_interp_results[ckpt][seed]
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f,
                    fieldnames=["model", "test", "suite", "passed", "ground_truth", "response"],
                    extrasaction="ignore")
                writer.writeheader()
                for r in results:
                    writer.writerow({
                        "model": r["model"], "test": r["test"],
                        "suite": _suite(r["test"]), "passed": r["passed"],
                        "ground_truth": r.get("ground_truth", ""),
                        "response": r.get("response", ""),
                    })
            print(f"  Saved: {csv_path}")

    # --- Save performance_results.csv ---
    perf_csv = str(RESULTS_DIR / "performance_results.csv")
    perf_rows = []
    for ds_name, model_rmses in dataset_rmses.items():
        for name, rmse_val in model_rmses.items():
            perf_rows.append({
                "dataset": ds_name, "model": name,
                "rmse": "" if np.isnan(rmse_val) else f"{rmse_val:.6f}",
                "rank": "",
            })
    by_dataset = defaultdict(list)
    for row in perf_rows:
        by_dataset[row["dataset"]].append(row)
    for ds_name, rows in by_dataset.items():
        valid = [(r, float(r["rmse"])) for r in rows if r["rmse"] not in ("", None)]
        valid.sort(key=lambda x: x[1])
        for rank_idx, (r, _) in enumerate(valid, 1):
            r["rank"] = rank_idx
    with open(perf_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "model", "rmse", "rank"])
        writer.writeheader()
        for ds_name in by_dataset:
            for row in by_dataset[ds_name]:
                writer.writerow(row)
    print(f"\nPerformance results saved -> {perf_csv}")

    # --- Save overall_results.csv for each checkpoint (averaged interp) ---
    for ckpt in CHECKPOINTS:
        ckpt_tag = ckpt.replace(".", "").replace("-", "")
        overall_rows = []
        for name, model, desc in model_defs_full:
            rank = avg_rank.get(name, float("nan"))
            interp_data = avg_interp[ckpt].get(name, {})
            frac = interp_data.get("interp_mean", float("nan"))
            frac_std = interp_data.get("interp_std", float("nan"))
            overall_rows.append({
                "commit": "baseline" if name in baseline_names else "",
                "mean_rank": f"{rank:.2f}" if not np.isnan(rank) else "nan",
                "frac_interpretability_tests_passed": f"{frac:.4f}" if not np.isnan(frac) else "nan",
                "status": "baseline" if name in baseline_names else "",
                "model_name": name,
                "description": desc,
            })

        csv_path = str(RESULTS_DIR / f"overall_results_{ckpt_tag}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=OVERALL_CSV_COLS)
            writer.writeheader()
            writer.writerows(overall_rows)
        print(f"Overall results ({ckpt}) saved -> {csv_path}")

        # Standard interp vs perf plot for this checkpoint
        plot_interp_vs_performance(
            csv_path,
            str(RESULTS_DIR / f"interpretability_vs_performance_{ckpt_tag}.png"),
        )

    # --- Also save a combined overall_results.csv (gpt-5.4 averaged) as the main file ---
    overall_rows_main = []
    for name, model, desc in model_defs_full:
        rank = avg_rank.get(name, float("nan"))
        interp_data = avg_interp["gpt-5.4"].get(name, {})
        frac = interp_data.get("interp_mean", float("nan"))
        overall_rows_main.append({
            "commit": "baseline" if name in baseline_names else "",
            "mean_rank": f"{rank:.2f}" if not np.isnan(rank) else "nan",
            "frac_interpretability_tests_passed": f"{frac:.4f}" if not np.isnan(frac) else "nan",
            "status": "baseline" if name in baseline_names else "",
            "model_name": name,
            "description": desc,
        })
    main_csv = str(RESULTS_DIR / "overall_results.csv")
    with open(main_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OVERALL_CSV_COLS)
        writer.writeheader()
        writer.writerows(overall_rows_main)
    print(f"\nMain overall results (gpt-5.4 avg) saved -> {main_csv}")
    plot_interp_vs_performance(main_csv, str(RESULTS_DIR / "interpretability_vs_performance.png"))

    # --- Comparison visualizations ---
    print("\n" + "=" * 60)
    print("  COMPARISON VISUALIZATIONS")
    print("=" * 60)
    plot_multi_seed_comparison(avg_results, str(RESULTS_DIR))

    # --- Summary table ---
    print("\n" + "=" * 60)
    print("  SUMMARY (averaged over 3 seeds)")
    print("=" * 60)
    print(f"\n  {'Model':<25} {'Rank':>6}  {'gpt-4o interp':>15}  {'gpt-5.4 interp':>15}")
    print("  " + "-" * 68)
    sorted_models = sorted(
        [name for name, _, _ in model_defs_full],
        key=lambda n: avg_rank.get(n, 999)
    )
    for name in sorted_models:
        rank = avg_rank.get(name, float("nan"))
        i4o = avg_interp.get("gpt-4o", {}).get(name, {})
        i54 = avg_interp.get("gpt-5.4", {}).get(name, {})
        s4o = f"{i4o.get('interp_mean', float('nan')):.3f}+/-{i4o.get('interp_std', 0):.3f}" if i4o else "n/a"
        s54 = f"{i54.get('interp_mean', float('nan')):.3f}+/-{i54.get('interp_std', 0):.3f}" if i54 else "n/a"
        bl = " [B]" if name in baseline_names else ""
        print(f"  {name:<25} {rank:6.2f}  {s4o:>15}  {s54:>15}{bl}")

    print(f"\nTotal time: {time.time() - t0:.1f}s")
