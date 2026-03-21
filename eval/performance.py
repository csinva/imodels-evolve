"""
Shared evaluation utilities for TabArena classification benchmarking.

Exports:
  MAX_SAMPLES, MAX_FEATURES, SUBSAMPLE_SEED
  subsample_dataset(X_train, X_test, y_train, y_test, ...) -> tuple
  evaluate_all_classifiers(model_defs) -> {dataset: {model: auc}}
  compute_rank_scores(dataset_aucs) -> (avg_rank, avg_auc)
"""

import os
from copy import deepcopy

import numpy as np
import pandas as pd
from joblib import Memory
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# ---------------------------------------------------------------------------
# Dataset loading (previously in prepare.py)
# ---------------------------------------------------------------------------

DATASET_NAMES = [
    "adult",
    "blood-transfusion-service-center",
    "breast-cancer",
    "california",
    "credit-g",
    "diabetes",
    "higgs",
    "jannis",
    "kr-vs-kp",
    "mfeat-factors",
    "numerai28.6",
    "phoneme",
    "sylvine",
    "volkert",
]

_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "imodels-evolve")


def _load_dataset(name):
    path = os.path.join(_CACHE_DIR, f"{name}.parquet")
    if not os.path.exists(path):
        import openml
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
    y = LabelEncoder().fit_transform(y_raw.astype(str))

    cat_cols = X_raw.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X_raw.columns if c not in cat_cols]

    X_tr, X_te, y_tr, y_te = train_test_split(X_raw, y, test_size=0.2, random_state=42, stratify=y)

    for col in num_cols:
        median = X_tr[col].median()
        X_tr[col] = pd.to_numeric(X_tr[col], errors="coerce").fillna(median)
        X_te[col] = pd.to_numeric(X_te[col], errors="coerce").fillna(median)

    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32)
        X_tr[cat_cols] = enc.fit_transform(X_tr[cat_cols].astype(str))
        X_te[cat_cols] = enc.transform(X_te[cat_cols].astype(str))

    return X_tr.astype(np.float32).values, X_te.astype(np.float32).values, y_tr, y_te


def get_all_datasets():
    """Yield (name, X_train, X_test, y_train, y_test) for each TabArena dataset."""
    for name in DATASET_NAMES:
        try:
            yield (name, *_load_dataset(name))
        except Exception as e:
            print(f"  WARNING: skipping '{name}': {e}")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
_memory = Memory(location=os.path.join(RESULTS_DIR, "cache"), verbose=0)

MAX_SAMPLES = 1000
MAX_FEATURES = 25
SUBSAMPLE_SEED = 42


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
def _run_one_classifier(model_name, ds_name, clf,
                        X_train, X_test, y_train, y_test):
    """Fit one classifier on one dataset and return AUC. Cached by joblib."""
    n_classes = len(np.unique(y_train))
    try:
        m = deepcopy(clf)
        m.fit(X_train, y_train)
        if n_classes == 2:
            proba = m.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, proba)
        else:
            proba = m.predict_proba(X_test)
            auc = roc_auc_score(y_test, proba, multi_class="ovr", average="macro")
        return float(auc)
    except Exception as e:
        return str(e)   # cache the error message too


def evaluate_all_classifiers(model_defs):
    """Evaluate all classifiers on every TabArena dataset (subsampled).

    Returns:
        dataset_aucs : {dataset_name: {model_name: auc}}
    """
    dataset_aucs = {}
    for ds_name, X_train, X_test, y_train, y_test in get_all_datasets():
        X_train, X_test, y_train, y_test = subsample_dataset(
            X_train, X_test, y_train, y_test)
        n_classes = len(np.unique(y_train))
        print(f"\n  Dataset: {ds_name} — {X_train.shape[1]} features, "
              f"{len(X_train)} train samples, {n_classes} classes")
        dataset_aucs[ds_name] = {}

        for name, clf in model_defs:
            result = _run_one_classifier(name, ds_name, clf,
                                         X_train, X_test, y_train, y_test)
            if isinstance(result, float):
                dataset_aucs[ds_name][name] = result
                print(f"    {name:<15}: {result:.4f}")
            else:
                print(f"    {name:<15}: ERROR — {result}")
                dataset_aucs[ds_name][name] = float("nan")

    return dataset_aucs


def compute_rank_scores(dataset_aucs):
    """For each dataset rank models by AUC (1=best), then average ranks."""
    all_model_names = set()
    for d in dataset_aucs.values():
        all_model_names.update(d.keys())

    ranks_per_model = {n: [] for n in all_model_names}
    mean_auc_per_model = {n: [] for n in all_model_names}

    for ds_name, model_aucs in dataset_aucs.items():
        valid = [(n, v) for n, v in model_aucs.items() if not np.isnan(v)]
        sorted_models = sorted(valid, key=lambda x: x[1], reverse=True)
        rank_map = {n: r + 1 for r, (n, _) in enumerate(sorted_models)}
        for name in all_model_names:
            if name in model_aucs and not np.isnan(model_aucs[name]):
                ranks_per_model[name].append(rank_map[name])
                mean_auc_per_model[name].append(model_aucs[name])

    avg_rank = {n: float(np.mean(v)) for n, v in ranks_per_model.items() if v}
    avg_auc  = {n: float(np.mean(v)) for n, v in mean_auc_per_model.items() if v}
    return avg_rank, avg_auc
