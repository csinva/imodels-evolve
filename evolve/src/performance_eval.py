"""
Shared evaluation utilities for regression benchmarking.

Datasets: 7 TabArena (OpenML) regression datasets + all 165 PMLB regression datasets.

Note that currently we normalize the outcome variable (y) to zero mean and unit std based on training set statistics only, to put all datasets on a similar scale and avoid leakage. This means that RMSE values are not directly comparable across datasets, but relative performance of models within each dataset is still meaningful.
We also subsample datasets to a maximum of 1000 training samples and 25 features (with a fixed random seed) and a minimum of 50 training samples and 3 features to speed up evaluation and focus on the low-data regime where interpretability is often most valuable.
We also only use the first 25 PMLB datasets for now.

Exports:
  MAX_SAMPLES, MAX_FEATURES, SUBSAMPLE_SEED
  subsample_dataset(X_train, X_test, y_train, y_test, ...) -> tuple
  evaluate_all_regressors(model_defs) -> {dataset: {model: rmse}}
  compute_rank_scores(dataset_rmses) -> (avg_rank, avg_rmse)
  upsert_overall_results(rows, results_dir) -> writes/updates overall_results.csv
  recompute_all_mean_ranks(results_dir) -> rewrites mean_rank for every row
    in overall_results.csv against the current performance_results.csv pool
"""

import os
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd
from joblib import Memory
from pmlb import fetch_data, regression_dataset_names
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import openml

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
MAX_SAMPLES = 1000
MAX_FEATURES = 50
MIN_SAMPLES = 10
MIN_FEATURES = 1
SUBSAMPLE_SEED = 42
MAX_PMLB_DATASETS = -1  # for now, to speed up evaluation (can remove this limit later)


OPENML_DATASET_NAMES = [
    "california",
    "abalone",
    "cpu_act",
    "house_16H",
    "elevators",
    "pol",
    "kin8nm",
]

PMLB_DATASET_NAMES = sorted(list(regression_dataset_names))

# filter our redundant pmlb friedman datasets
PMLB_DATASET_NAMES = [n for n in PMLB_DATASET_NAMES
                      if "_fri_" not in n]
if MAX_PMLB_DATASETS > 0:
    PMLB_DATASET_NAMES = PMLB_DATASET_NAMES[:MAX_PMLB_DATASETS]

_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "imodels-evolve")
_PMLB_CACHE_DIR = os.path.join(_CACHE_DIR, "pmlb")


def _load_openml_dataset(name):
    path = os.path.join(_CACHE_DIR, f"{name}.parquet")
    if not os.path.exists(path):
        os.makedirs(_CACHE_DIR, exist_ok=True)
        openml.config.cache_directory = os.path.join(_CACHE_DIR, "openml")
        datasets_list = openml.datasets.list_datasets(output_format="dataframe")
        matches = datasets_list[(datasets_list["name"] == name) & (datasets_list["status"] == "active")]
        dataset_id = int(matches.sort_values("did").iloc[-1]["did"])
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


def _load_pmlb_dataset(name):
    os.makedirs(_PMLB_CACHE_DIR, exist_ok=True)
    df = fetch_data(name, local_cache_dir=_PMLB_CACHE_DIR)

    y = df["target"].values.astype(float)
    X = df.drop(columns=["target"]).values.astype(np.float32)

    # Skip datasets with fewer than MIN_SAMPLES samples (too small to split)
    if len(X) < MIN_SAMPLES:
        raise ValueError(f"Too few samples: {len(X)}")

    if X.shape[1] < MIN_FEATURES:
        raise ValueError(f"Too few features: {X.shape[1]}")

    # Drop rows with NaN targets
    valid = ~np.isnan(y)
    X, y = X[valid], y[valid]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_tr, X_te, y_tr, y_te


def get_all_datasets():
    """Yield (name, X_train, X_test, y_train, y_test) for all datasets.

    Sources:
      - 7 TabArena/OpenML datasets (OPENML_DATASET_NAMES)
      - 165 PMLB regression datasets (PMLB_DATASET_NAMES)
    """
    for name in OPENML_DATASET_NAMES:
        try:
            yield (name, *_load_openml_dataset(name))
        except Exception as e:
            print(f"  WARNING: skipping openml '{name}': {e}")

    for name in PMLB_DATASET_NAMES:
        try:
            yield (f"pmlb/{name}", *_load_pmlb_dataset(name))
        except Exception as e:
            print(f"  WARNING: skipping pmlb '{name}': {e}")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
_memory = Memory(location=os.path.join(RESULTS_DIR, "cache"), verbose=0)


def subsample_dataset(X_train, X_test, y_train, y_test,
                      max_samples=MAX_SAMPLES, max_features=MAX_FEATURES,
                      seed=SUBSAMPLE_SEED):
    """Cap training samples and features with a fixed random seed."""
    rng = np.random.RandomState(seed)
    if X_train.shape[1] > max_features:
        feat_idx = rng.choice(X_train.shape[1], max_features, replace=False)
        feat_idx.sort()
        X_train = X_train[:, feat_idx]
        X_test  = X_test[:, feat_idx]
    if len(X_train) > max_samples:
        idx = rng.choice(len(X_train), max_samples, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]
    return X_train, X_test, y_train, y_test


@_memory.cache
def _run_one_regressor(model_name, ds_name, reg,
                       X_train, X_test, y_train, y_test):
    """Fit one regressor on one dataset and return RMSE. Cached by joblib."""
    try:
        m = deepcopy(reg)
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        print('Completed fitting!')
        return rmse
    except Exception as e:
        return str(e)   # cache the error message too


def _eval_one_dataset(ds_name, X_train, X_test, y_train, y_test, model_defs):
    """Evaluate all regressors on one dataset. Returns (ds_name, {model_name: rmse})."""
    X_train, X_test, y_train, y_test = subsample_dataset(X_train, X_test, y_train, y_test)

    if len(X_train) < MIN_SAMPLES:
        print(f"  Skipping {ds_name}: only {len(X_train)} training samples")
        return ds_name, {}
    if X_train.shape[1] < MIN_FEATURES:
        print(f"  Skipping {ds_name}: only {X_train.shape[1]} features")
        return ds_name, {}

    # Normalize outcome variable using training-set statistics only (avoid leakage)
    y_mean = float(y_train.mean())
    y_std = float(y_train.std())
    if y_std > 0:
        y_train = (y_train - y_mean) / y_std
        y_test = (y_test - y_mean) / y_std

    print(f"\n  Dataset: {ds_name} — {X_train.shape[1]} features, "
          f"{len(X_train)} train samples (y normalized: mean={y_mean:.3g}, std={y_std:.3g})")
    model_rmses = {}
    for name, reg in model_defs:
        result = _run_one_regressor(name, ds_name, reg, X_train, X_test, y_train, y_test)
        if isinstance(result, float):
            model_rmses[name] = result
            print(f"    {name:<15}: {result:.4f}")
        else:
            print(f"    {name:<15}: ERROR — {result}")
            model_rmses[name] = float("nan")
    return ds_name, model_rmses


def evaluate_all_regressors(model_defs):
    """Evaluate all regressors on every TabArena dataset (subsampled).

    Returns:
        dataset_rmses : {dataset_name: {model_name: rmse}}
    """
    from joblib import Parallel, delayed

    datasets = list(get_all_datasets())
    results = Parallel(n_jobs=-1)(
        delayed(_eval_one_dataset)(ds_name, X_train, X_test, y_train, y_test, model_defs)
        for ds_name, X_train, X_test, y_train, y_test in datasets
    )
    return dict(results)


def compute_rank_scores(dataset_rmses):
    """For each dataset rank models by RMSE (1=best/lowest), then average ranks."""
    all_model_names = set()
    for d in dataset_rmses.values():
        all_model_names.update(d.keys())

    ranks_per_model = {n: [] for n in all_model_names}
    mean_rmse_per_model = {n: [] for n in all_model_names}

    for ds_name, model_rmses in dataset_rmses.items():
        valid = [(n, v) for n, v in model_rmses.items() if not np.isnan(v)]
        sorted_models = sorted(valid, key=lambda x: x[1])  # ascending: lower RMSE = better rank
        rank_map = {n: r + 1 for r, (n, _) in enumerate(sorted_models)}
        for name in all_model_names:
            if name in model_rmses and not np.isnan(model_rmses[name]):
                ranks_per_model[name].append(rank_map[name])
                mean_rmse_per_model[name].append(model_rmses[name])

    n_datasets = len(dataset_rmses)
    avg_rank = {n: float(np.mean(v)) if len(v) == n_datasets else float("nan")
                for n, v in ranks_per_model.items() if v}
    avg_rmse = {n: float(np.mean(v)) if len(v) == n_datasets else float("nan")
                for n, v in mean_rmse_per_model.items() if v}
    return avg_rank, avg_rmse


# ---------------------------------------------------------------------------
# Overall results CSV
# ---------------------------------------------------------------------------

OVERALL_CSV_COLS = ["commit", "mean_rank", "frac_interpretability_tests_passed", "status", "model_name", "description"]


def upsert_overall_results(rows, results_dir):
    """Write or update overall_results.csv, replacing existing rows by (model_name, description).

    Args:
        rows: list of dicts with keys matching OVERALL_CSV_COLS
        results_dir: directory to write overall_results.csv into
    """
    import csv as _csv
    path = os.path.join(results_dir, "overall_results.csv")

    # Load existing rows, drop any being replaced (keyed by model_name AND description).
    existing = []
    new_keys = {(r["model_name"], r.get("description", "")) for r in rows}
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in _csv.DictReader(f):
                if (row.get("model_name"), row.get("description", "")) not in new_keys:
                    existing.append(row)

    all_rows = existing + [{k: r.get(k, "") for k in OVERALL_CSV_COLS} for r in rows]
    # all_rows.sort(key=lambda r: r.get("model_name", ""))

    with open(path, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=OVERALL_CSV_COLS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Overall results saved → {path}")


def recompute_all_mean_ranks(results_dir):
    """Rewrite mean_rank for every row in overall_results.csv against the
    current performance_results.csv pool, so all rows reflect the same pool.

    Returns the {model_name: mean_rank} mapping.
    """
    import csv as _csv
    perf_path = os.path.join(results_dir, "performance_results.csv")
    overall_path = os.path.join(results_dir, "overall_results.csv")
    if not (os.path.exists(perf_path) and os.path.exists(overall_path)):
        return {}

    dataset_rmses = defaultdict(dict)
    with open(perf_path, newline="") as f:
        for row in _csv.DictReader(f):
            rmse_str = row.get("rmse", "")
            try:
                val = float(rmse_str) if rmse_str not in ("", None) else float("nan")
            except ValueError:
                val = float("nan")
            dataset_rmses[row["dataset"]][row["model"]] = val

    avg_rank, _ = compute_rank_scores(dict(dataset_rmses))

    with open(overall_path, newline="") as f:
        rows = list(_csv.DictReader(f))

    for row in rows:
        mr = avg_rank.get(row.get("model_name"), float("nan"))
        row["mean_rank"] = "nan" if np.isnan(mr) else f"{mr:.2f}"

    with open(overall_path, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=OVERALL_CSV_COLS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Mean ranks recomputed for {len(rows)} rows → {overall_path}")
    return avg_rank

if __name__ == "__main__":
    # test get all datasets and print length and size of each dataset
    # for name, X_train, X_test, y_train, y_test in get_all_datasets():
        # print(f"{name}: {X_train.shape[0]} train samples, {X_test.shape[0]} test samples, {X_train.shape[1]} features")


    # print('Total', len(list(get_all_datasets())), 'datasets')
    benchmark_suite = openml.study.get_suite(335)
    suite_dataset_ids = set(benchmark_suite.data)
    print('Total', len(suite_dataset_ids), 'datasets in openml suite 335')

    # Resolve suite dataset IDs to names
    openml.config.cache_directory = os.path.join(_CACHE_DIR, "openml")
    datasets_list = openml.datasets.list_datasets(output_format="dataframe")
    suite_id_to_name = {}
    for did in suite_dataset_ids:
        matches = datasets_list[datasets_list["did"] == did]
        suite_id_to_name[did] = matches.iloc[0]["name"] if not matches.empty else "unknown"

    # Check which suite datasets are not in our OpenML or PMLB lists (by name)
    used_names = set(OPENML_DATASET_NAMES) | set(PMLB_DATASET_NAMES)
    missing = {did: name for did, name in suite_id_to_name.items() if name not in used_names}
    print(missing.keys())

    print(f"\n{len(missing)} suite 335 datasets NOT in our OpenML or PMLB lists:")
    for did in sorted(missing):
        print(f"  ID {did}: {missing[did]}")
