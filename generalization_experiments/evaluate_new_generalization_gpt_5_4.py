"""
Evaluate all models (baselines + interpretable_regressors_lib) on interpretability
tests using the gpt-5.4 LLM checkpoint instead of gpt-4o.

Saves results to new_results_gpt_5_4/ in the same format as original_results/.
Also produces comparison visualizations against original results.

Usage: uv run evaluate_new_generalization_gpt_5_4.py
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
from sklearn.base import clone
from sklearn.metrics import r2_score

# Add src/ to path so we can import shared modules
SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "evolve", "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "evolve"))

from performance_eval import (
    MAX_SAMPLES, MAX_FEATURES, MIN_SAMPLES, MIN_FEATURES, SUBSAMPLE_SEED,
    subsample_dataset, compute_rank_scores, OVERALL_CSV_COLS,
)
from interp_eval import get_model_str, ask_llm, _safe_clone
from visualize import plot_interp_vs_performance

import openml
import imodelsx.llm

# Baseline model imports
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LassoCV as SkLassoCV, LinearRegression, RidgeCV as SkRidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from pygam import LinearGAM
from imodels import FIGSRegressor, HSTreeRegressorCV, RuleFitRegressor
from interpret.glassbox import ExplainableBoostingRegressor
from tabpfn import TabPFNRegressor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GPT_5_4_CHECKPOINT = "gpt-5.4"

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_DIR = SCRIPT_DIR / "interpretable_regressors_lib" / "success"
RESULTS_DIR = SCRIPT_DIR / "new_results_gpt_5_4"
RESULTS_DIR.mkdir(exist_ok=True)
ORIGINAL_RESULTS_DIR = SCRIPT_DIR / "original_results"

# New OpenML dataset IDs from suite 335 (abalone removed)
NEW_OPENML_IDS = [
    44065, 44066, 44068, 44069, 45048, 45041,
    45043, 45047, 45045, 45046, 44055, 44056,
    44059, 44061, 44062, 44063,
]

_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "imodels-evolve")

# ---------------------------------------------------------------------------
# Baseline model definitions (same as run_baselines.py)
# ---------------------------------------------------------------------------

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
# Load models from lib
# ---------------------------------------------------------------------------

def load_all_models():
    """Import each .py file in LIB_DIR and extract (model_shorthand_name, model_instance, description)."""
    model_defs = []
    seen_names = set()
    for py_file in sorted(LIB_DIR.glob("*.py")):
        mod_name = py_file.stem
        spec = importlib.util.spec_from_file_location(mod_name, py_file)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception as e:
            print(f"WARNING: failed to load {py_file.name}: {e}")
            continue

        for attr_name in dir(mod):
            obj = getattr(mod, attr_name)
            if isinstance(obj, type) and hasattr(obj, 'fit') and hasattr(obj, 'predict'):
                obj.__module__ = mod_name

        name = getattr(mod, "model_shorthand_name", None)
        desc = getattr(mod, "model_description", "")
        defs = getattr(mod, "model_defs", None)
        if name and defs and name not in seen_names:
            seen_names.add(name)
            model_defs.append((name, defs[0][1], desc))
            print(f"  Loaded: {name}")
    return model_defs


# ---------------------------------------------------------------------------
# Performance evaluation (16 OpenML datasets)
# ---------------------------------------------------------------------------

def _load_openml_dataset_by_id(dataset_id):
    """Load an OpenML dataset by ID, returning (X_train, X_test, y_train, y_test)."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OrdinalEncoder
    import pandas as pd

    path = os.path.join(_CACHE_DIR, f"openml_id_{dataset_id}.parquet")
    if not os.path.exists(path):
        os.makedirs(_CACHE_DIR, exist_ok=True)
        openml.config.cache_directory = os.path.join(_CACHE_DIR, "openml")
        dataset = openml.datasets.get_dataset(dataset_id, download_data=True)
        X, y, _, attribute_names = dataset.get_data(target=dataset.default_target_attribute)
        df = pd.DataFrame(X, columns=attribute_names)
        df["__target__"] = y
        df.to_parquet(path, index=False)
    else:
        df = pd.read_parquet(path)

    y_raw = df["__target__"].values
    X_raw = df.drop(columns=["__target__"])
    y = pd.to_numeric(pd.Series(y_raw), errors="coerce").values.astype(float)

    valid = ~np.isnan(y)
    y = y[valid]
    X_raw = X_raw[valid]

    cat_cols = X_raw.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X_raw.columns if c not in cat_cols]

    X_tr, X_te, y_tr, y_te = train_test_split(X_raw, y, test_size=0.2, random_state=42)

    for col in num_cols:
        median = X_tr[col].median()
        X_tr[col] = pd.to_numeric(X_tr[col], errors="coerce").fillna(median)
        X_te[col] = pd.to_numeric(X_te[col], errors="coerce").fillna(median)

    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32)
        X_tr[cat_cols] = enc.fit_transform(X_tr[cat_cols].astype(str))
        X_te[cat_cols] = enc.transform(X_te[cat_cols].astype(str))

    return X_tr.astype(np.float32).values, X_te.astype(np.float32).values, y_tr, y_te


def get_new_datasets():
    """Yield (name, X_train, X_test, y_train, y_test) for the 16 new OpenML datasets."""
    for did in NEW_OPENML_IDS:
        try:
            X_tr, X_te, y_tr, y_te = _load_openml_dataset_by_id(did)
            yield (f"openml/{did}", X_tr, X_te, y_tr, y_te)
        except Exception as e:
            print(f"  WARNING: skipping openml ID {did}: {e}")


def _eval_one_dataset(ds_name, X_train, X_test, y_train, y_test, model_defs_simple):
    """Evaluate all regressors on one dataset. Returns (ds_name, {model_name: rmse})."""
    import gc
    from sklearn.metrics import mean_squared_error

    X_train, X_test, y_train, y_test = subsample_dataset(X_train, X_test, y_train, y_test)

    if len(X_train) < MIN_SAMPLES:
        print(f"  Skipping {ds_name}: only {len(X_train)} training samples")
        return ds_name, {}
    if X_train.shape[1] < MIN_FEATURES:
        print(f"  Skipping {ds_name}: only {X_train.shape[1]} features")
        return ds_name, {}

    y_mean = float(y_train.mean())
    y_std = float(y_train.std())
    if y_std > 0:
        y_train = (y_train - y_mean) / y_std
        y_test = (y_test - y_mean) / y_std

    print(f"\n  Dataset: {ds_name} — {X_train.shape[1]} features, "
          f"{len(X_train)} train samples")
    model_rmses = {}
    for name, reg in model_defs_simple:
        for attempt in range(2):
            try:
                m = deepcopy(reg)
                m.fit(X_train, y_train)
                preds = m.predict(X_test)
                rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
                model_rmses[name] = rmse
                print(f"    {name:<25}: {rmse:.4f}")
                break
            except Exception as e:
                if attempt == 0:
                    gc.collect()
                    print(f"    {name:<25}: RETRY (attempt 1 failed: {e})")
                else:
                    print(f"    {name:<25}: ERROR — {e}")
                    model_rmses[name] = float("nan")
        gc.collect()
    return ds_name, model_rmses


def evaluate_performance(model_defs_simple):
    """Evaluate all models on the 16 new OpenML datasets."""
    datasets = list(get_new_datasets())
    print(f"\nEvaluating on {len(datasets)} new datasets...")
    results = []
    for ds_name, X_tr, X_te, y_tr, y_te in datasets:
        results.append(
            _eval_one_dataset(ds_name, X_tr, X_te, y_tr, y_te, model_defs_simple)
        )
    return dict(results)


# ---------------------------------------------------------------------------
# Interpretability tests using gpt-5.4
# ---------------------------------------------------------------------------
# We reuse the NEW interpretability tests from evaluate_new_generalization.py
# but run them with the gpt-5.4 checkpoint.

from evaluate_new_generalization import (
    ALL_NEW_TESTS,
    NEW_STANDARD_TESTS, NEW_HARD_TESTS, NEW_INSIGHT_TESTS,
    NEW_DISCRIM_TESTS, NEW_SIMULATABILITY_TESTS,
    _ALL_NEW_TEST_FNS,
)

from joblib import Memory
_gpt54_cache = Memory(location=os.path.join(str(RESULTS_DIR), "cache"), verbose=0)


@_gpt54_cache.cache
def _run_one_test_gpt54(model_name, test_fn_name, model):
    llm = imodelsx.llm.get_llm(GPT_5_4_CHECKPOINT)
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


def run_all_interp_tests_gpt54(model_defs_simple):
    """Run all interpretability tests on all models using gpt-5.4."""
    from joblib import Parallel, delayed

    tasks = [(name, reg, test_fn)
             for name, reg in model_defs_simple
             for test_fn in ALL_NEW_TESTS]

    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_run_one_test_gpt54)(name, test_fn.__name__, reg)
        for name, reg, test_fn in tasks
    )

    # Print results
    for name, reg in model_defs_simple:
        print(f"\n{'='*60}\n  Model: {name}\n{'='*60}")
        for test_list, label in [
            (NEW_STANDARD_TESTS, "standard"),
            (NEW_HARD_TESTS, "hard"),
            (NEW_INSIGHT_TESTS, "insight"),
            (NEW_DISCRIM_TESTS, "discrim"),
            (NEW_SIMULATABILITY_TESTS, "simulatability"),
        ]:
            print(f"\n  [{label}]")
            suite_results = [r for r in results if r["model"] == name
                             and r["test"] in {t.__name__ for t in test_list}]
            for result in suite_results:
                status = "PASS" if result["passed"] else "FAIL"
                resp = (result.get("response") or "")[:80].replace("\n", " ")
                print(f"  [{status}] {result['test']}")
                print(f"         ground_truth : {result.get('ground_truth', '')}")
                print(f"         llm_response : {resp}")
            print(f"\n  -> {sum(r['passed'] for r in suite_results)}/{len(test_list)} passed")

    return results


# ---------------------------------------------------------------------------
# Comparison visualization
# ---------------------------------------------------------------------------

def plot_comparison(original_csv, new_csv, out_path):
    """Create a comparison scatter plot of original (gpt-4o) vs new (gpt-5.4) results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df_orig = pd.read_csv(original_csv)
    df_new = pd.read_csv(new_csv)

    # Clean data
    for df in [df_orig, df_new]:
        df.replace(["", "nan"], np.nan, inplace=True)
        df.dropna(subset=["mean_rank", "frac_interpretability_tests_passed", "model_name"], inplace=True)
        df["mean_rank"] = df["mean_rank"].astype(float)
        df["frac_interpretability_tests_passed"] = df["frac_interpretability_tests_passed"].astype(float)

    # Merge on model_name to compare
    merged = pd.merge(df_orig, df_new, on="model_name", suffixes=("_orig", "_new"), how="inner")

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # --- Plot 1: Side-by-side scatter (interp vs rank) ---
    ax = axes[0]
    ax.scatter(df_orig["frac_interpretability_tests_passed"], df_orig["mean_rank"],
               alpha=0.5, label="gpt-4o (original)", marker="o", s=40, c="steelblue")
    ax.scatter(df_new["frac_interpretability_tests_passed"], df_new["mean_rank"],
               alpha=0.5, label="gpt-5.4 (new)", marker="^", s=40, c="coral")
    ax.set_xlabel("Fraction Interpretability Tests Passed")
    ax.set_ylabel("Mean Rank (lower is better)")
    ax.set_title("Interpretability vs Performance")
    ax.invert_yaxis()
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Plot 2: Interp score comparison (gpt-4o vs gpt-5.4) ---
    ax = axes[1]
    if len(merged) > 0:
        ax.scatter(merged["frac_interpretability_tests_passed_orig"],
                   merged["frac_interpretability_tests_passed_new"],
                   alpha=0.6, s=40, c="teal")
        lims = [0, 1]
        ax.plot(lims, lims, '--', color='gray', alpha=0.5, label="y=x (no change)")
        ax.set_xlabel("Interp Score (gpt-4o)")
        ax.set_ylabel("Interp Score (gpt-5.4)")
        ax.set_title("Interpretability Score: gpt-4o vs gpt-5.4")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Label a few interesting points
        for _, row in merged.iterrows():
            diff = row["frac_interpretability_tests_passed_new"] - row["frac_interpretability_tests_passed_orig"]
            if abs(diff) > 0.15:
                ax.annotate(row["model_name"],
                            (row["frac_interpretability_tests_passed_orig"],
                             row["frac_interpretability_tests_passed_new"]),
                            fontsize=6, alpha=0.7)

    # --- Plot 3: Per-model interp difference ---
    ax = axes[2]
    if len(merged) > 0:
        merged["interp_diff"] = (merged["frac_interpretability_tests_passed_new"]
                                 - merged["frac_interpretability_tests_passed_orig"])
        sorted_df = merged.sort_values("interp_diff")

        # Show top/bottom 15
        n_show = min(15, len(sorted_df))
        show_df = pd.concat([sorted_df.head(n_show), sorted_df.tail(n_show)]).drop_duplicates()

        colors = ["coral" if d < 0 else "steelblue" for d in show_df["interp_diff"]]
        ax.barh(range(len(show_df)), show_df["interp_diff"], color=colors, alpha=0.7)
        ax.set_yticks(range(len(show_df)))
        ax.set_yticklabels(show_df["model_name"], fontsize=6)
        ax.set_xlabel("Change in Interp Score (gpt-5.4 - gpt-4o)")
        ax.set_title("Biggest Changes in Interpretability Score")
        ax.axvline(x=0, color='gray', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved -> {out_path}")


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
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()

    # --- Load evolved models ---
    print("=" * 60)
    print("  Loading models from interpretable_regressors_lib/success/")
    print("=" * 60)
    model_defs_full = load_all_models()
    print(f"\nLoaded {len(model_defs_full)} evolved models.\n")

    # --- Combine with baselines ---
    for bname, bmodel in BASELINE_DEFS:
        model_defs_full.append((bname, bmodel, BASELINE_DESCRIPTIONS.get(bname, bname)))

    model_defs_simple = [(name, model) for name, model, _ in model_defs_full]
    print(f"Total models (evolved + baselines): {len(model_defs_simple)}\n")

    # --- Interpretability tests with gpt-5.4 ---
    print("=" * 60)
    print(f"  Running interpretability tests with {GPT_5_4_CHECKPOINT}")
    print("=" * 60)
    interp_results = run_all_interp_tests_gpt54(model_defs_simple)

    # --- Performance evaluation ---
    print("\n" + "=" * 60)
    print("  Running performance evaluation (16 OpenML datasets)")
    print("=" * 60)
    dataset_rmses = evaluate_performance(model_defs_simple)

    # --- Save interpretability_results.csv ---
    interp_csv = str(RESULTS_DIR / "interpretability_results.csv")
    interp_fields = ["model", "test", "suite", "passed", "ground_truth", "response"]
    interp_rows = [{
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
        writer.writerows(interp_rows)
    print(f"\nInterpretability results saved -> {interp_csv}")

    # --- Save performance_results.csv ---
    perf_csv = str(RESULTS_DIR / "performance_results.csv")
    perf_fields = ["dataset", "model", "rmse", "rank"]

    perf_rows = []
    for ds_name, model_rmses in dataset_rmses.items():
        for name, rmse_val in model_rmses.items():
            perf_rows.append({
                "dataset": ds_name,
                "model": name,
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
        writer = csv.DictWriter(f, fieldnames=perf_fields)
        writer.writeheader()
        for ds_name in by_dataset:
            for row in by_dataset[ds_name]:
                writer.writerow(row)
    print(f"Performance results saved -> {perf_csv}")

    # --- Compute overall results ---
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

    model_interp = defaultdict(lambda: {"passed": 0, "total": 0})
    for r in interp_results:
        model_interp[r["model"]]["total"] += 1
        if r["passed"]:
            model_interp[r["model"]]["passed"] += 1

    baseline_names = {bname for bname, _ in BASELINE_DEFS}

    overall_rows = []
    for name, model, desc in model_defs_full:
        rank = avg_rank.get(name, float("nan"))
        mi = model_interp[name]
        frac = mi["passed"] / mi["total"] if mi["total"] > 0 else float("nan")
        overall_rows.append({
            "commit": "baseline" if name in baseline_names else "",
            "mean_rank": f"{rank:.2f}" if not np.isnan(rank) else "nan",
            "frac_interpretability_tests_passed": f"{frac:.4f}" if not np.isnan(frac) else "nan",
            "status": "baseline" if name in baseline_names else "",
            "model_name": name,
            "description": desc,
        })

    overall_csv = str(RESULTS_DIR / "overall_results.csv")
    with open(overall_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OVERALL_CSV_COLS)
        writer.writeheader()
        writer.writerows(overall_rows)
    print(f"Overall results saved -> {overall_csv}")

    # --- Standard interpretability vs performance plot ---
    plot_interp_vs_performance(
        overall_csv,
        str(RESULTS_DIR / "interpretability_vs_performance.png"),
    )

    # --- Comparison with original results ---
    original_overall = str(ORIGINAL_RESULTS_DIR / "overall_results.csv")
    if os.path.exists(original_overall):
        print("\n" + "=" * 60)
        print("  COMPARISON: gpt-4o (original) vs gpt-5.4 (new)")
        print("=" * 60)
        plot_comparison(
            original_overall,
            overall_csv,
            str(RESULTS_DIR / "comparison_gpt4o_vs_gpt54.png"),
        )

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  SUMMARY (gpt-5.4 results)")
    print("=" * 60)
    for row in sorted(overall_rows, key=lambda r: float(r["mean_rank"]) if r["mean_rank"] != "nan" else 999):
        status = " [baseline]" if row["status"] == "baseline" else ""
        print(f"  {row['model_name']:<25}  rank={row['mean_rank']:>6}  interp={row['frac_interpretability_tests_passed']}{status}")

    print(f"\nTotal time: {time.time() - t0:.1f}s")
