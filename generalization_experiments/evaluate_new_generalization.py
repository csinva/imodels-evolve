"""
Evaluate all models in interpretable_regressors_lib/ on NEW regression datasets
and NEW interpretability metrics, saving results to new_results/.

New regression datasets: 16 OpenML datasets from suite 335 (excluding abalone).
New interpretability tests: minor variations of the original tests with different
synthetic inputs (different seeds, coefficients, sample points).

Usage: uv run evaluate_new_generalization.py
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
    subsample_dataset, compute_rank_scores, upsert_overall_results, OVERALL_CSV_COLS,
)
from interp_eval import get_model_str, ask_llm, _safe_clone, CHECKPOINT
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

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_DIR = SCRIPT_DIR / "interpretable_regressors_lib" / "success"
NEW_RESULTS_DIR = SCRIPT_DIR / "new_results"
NEW_RESULTS_DIR.mkdir(exist_ok=True)

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
    """Import each .py file in LIB_DIR and extract (model_shorthand_name, model_instance)."""
    model_defs = []
    seen_names = set()
    for py_file in sorted(LIB_DIR.glob("*.py")):
        mod_name = py_file.stem
        spec = importlib.util.spec_from_file_location(mod_name, py_file)
        mod = importlib.util.module_from_spec(spec)
        # Make the module available for pickling by joblib under its unique stem name
        sys.modules[mod_name] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception as e:
            print(f"WARNING: failed to load {py_file.name}: {e}")
            continue

        # Fix pickling: ensure each model class's __module__ matches a registered sys.modules key
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
# NEW performance evaluation (16 OpenML datasets)
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


def _eval_one_dataset_new(ds_name, X_train, X_test, y_train, y_test, model_defs_simple):
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


def evaluate_new_performance(model_defs_simple):
    """Evaluate all models on the 16 new OpenML datasets.

    Runs sequentially to avoid pickling issues with dynamically-loaded model classes.
    """
    datasets = list(get_new_datasets())
    print(f"\nEvaluating on {len(datasets)} new datasets...")
    results = []
    for ds_name, X_tr, X_te, y_tr, y_te in datasets:
        results.append(
            _eval_one_dataset_new(ds_name, X_tr, X_te, y_tr, y_te, model_defs_simple)
        )
    return dict(results)


# ---------------------------------------------------------------------------
# NEW interpretability tests (variations of originals)
# ---------------------------------------------------------------------------
# Each test uses slightly different seeds, coefficients, or query points
# compared to the originals, while testing the same capabilities.

from joblib import Memory
_new_cache = Memory(location=os.path.join(str(NEW_RESULTS_DIR), "cache"), verbose=0)


# --- Synthetic data factories (varied from originals) ---

def _v_single_feature_data(n_features=5, true_feature=0, coef=10.0, n=300, seed=100):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features)
    y = coef * X[:, true_feature] + rng.randn(n) * 0.5
    return X, y

def _v_multi_feature_data(coefs, n=500, seed=101):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, len(coefs))
    y = X @ np.array(coefs) + rng.randn(n) * 0.5
    return X, y

def _v_threshold_data(threshold=0.3, n=400, seed=102):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 3)
    y = np.where(X[:, 0] > threshold, 2.5, 0.0) + rng.randn(n) * 0.1
    return X, y

def _v_signed_data(n=400, seed=103):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 4)
    y = 4.0 * X[:, 0] - 6.0 * X[:, 1] + rng.randn(n) * 0.5
    return X, y

def _v_hockey_stick_data(n=700, seed=130):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 3)
    y = 4.0 * np.maximum(0.0, X[:, 0]) + 0.2 * rng.randn(n)
    return X, y

def _v_sparse_ten_feature_data(n=600, seed=131):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 10)
    y = 6.0 * X[:, 0] + 2.5 * X[:, 1] + 0.3 * rng.randn(n)
    return X, y

def _v_mixed_sign_six_feature_data(n=700, seed=140):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 6)
    y = 3.5 * X[:, 0] - 2.5 * X[:, 1] + 3.0 * X[:, 2] - 1.0 * X[:, 3] + 0.6 * X[:, 4] - 0.5 * X[:, 5] + rng.randn(n) * 0.4
    return X, y

def _v_double_threshold_data(n=600, seed=141):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 4)
    y = np.where(X[:, 0] > 1.0, 3.5, np.where(X[:, 0] > -0.5, 1.5, 0.0)) + rng.randn(n) * 0.1
    return X, y

def _v_additive_nonlinear_data(n=800, seed=142):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 5)
    y = 2.5 * np.maximum(0.0, X[:, 0]) + 3.0 * np.sin(X[:, 1]) + 1.5 * X[:, 2] + rng.randn(n) * 0.3
    return X, y

def _v_interaction_data(n=700, seed=143):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 4)
    y = 2.5 * X[:, 0] + 3.0 * X[:, 1] + 2.0 * X[:, 0] * X[:, 1] + rng.randn(n) * 0.4
    return X, y

def _v_eight_feature_mixed_data(n=600, seed=160):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 8)
    coefs = np.array([5.0, -3.5, 4.0, -1.5, 2.0, -0.8, 0.3, 0.0])
    y = X @ coefs + rng.randn(n) * 0.5
    return X, y, coefs

def _v_fifteen_feature_sparse_data(n=800, seed=161):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 15)
    coefs = np.zeros(15)
    coefs[1] = 7.0
    coefs[6] = -3.5
    coefs[10] = 4.0
    y = X @ coefs + rng.randn(n) * 0.5
    return X, y, coefs

def _v_quadratic_data(n=800, seed=162):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 5)
    y = 2.5 * X[:, 0]**2 - 1.5 * X[:, 1]**2 + 2.0 * X[:, 2] + rng.randn(n) * 0.3
    return X, y

def _v_friedman1_data(n=1000, seed=164):
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 1, size=(n, 10))
    y = (10.0 * np.sin(np.pi * X[:, 0] * X[:, 1])
         + 20.0 * (X[:, 2] - 0.5)**2
         + 10.0 * X[:, 3]
         + 5.0 * X[:, 4]
         + rng.randn(n) * 0.5)
    return X, y

def _v_cascading_threshold_data(n=800, seed=165):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 6)
    y = np.where(X[:, 0] > 0, 2.5 * X[:, 1], -3.0 * X[:, 2]) + rng.randn(n) * 0.3
    return X, y

def _v_exponential_decay_data(n=700, seed=170):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 4)
    y = 4.0 * np.exp(-X[:, 0]) + 3.0 * X[:, 1] + rng.randn(n) * 0.3
    return X, y

def _v_piecewise_three_segment_data(n=800, seed=171):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 4)
    x0 = X[:, 0]
    y = np.where(x0 < -0.5, -2.0,
         np.where(x0 < 1.5, 2.0 * x0, 3.0 + 0.3 * (x0 - 1.5))) + rng.randn(n) * 0.2
    return X, y

def _v_twenty_feature_sparse_data(n=1000, seed=172):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 20)
    coefs = np.zeros(20)
    coefs[3] = 5.0
    coefs[8] = -4.0
    coefs[13] = 3.0
    coefs[17] = -2.0
    y = X @ coefs + rng.randn(n) * 0.5
    return X, y, coefs

def _v_sinusoidal_data(n=800, seed=173):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 5)
    y = 3.0 * np.sin(X[:, 0]) + 3.0 * np.cos(X[:, 1]) + 1.5 * X[:, 2] + rng.randn(n) * 0.3
    return X, y

def _v_abs_value_data(n=700, seed=174):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 5)
    y = 4.0 * np.abs(X[:, 0]) - 1.5 * np.abs(X[:, 1]) + 2.0 * X[:, 2] + rng.randn(n) * 0.3
    return X, y

def _v_twelve_feature_all_active_data(n=800, seed=175):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 12)
    coefs = np.array([4.0, -3.5, 3.0, -2.5, 2.0, -1.5, 1.0, -0.8, 0.6, -0.4, 0.3, -0.1])
    y = X @ coefs + rng.randn(n) * 0.5
    return X, y, coefs

def _v_nested_threshold_data(n=900, seed=176):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 5)
    y = np.where(
        X[:, 0] > 0,
        np.where(X[:, 1] > 0, 4.0, 1.5),
        -1.5
    ) + rng.randn(n) * 0.2
    return X, y


# --- New tests (variations of originals with prefix "new_") ---

def new_test_most_important_feature(model, llm):
    X, y = _v_single_feature_data(n_features=6, true_feature=0, coef=12.0)
    names = [f"x{i}" for i in range(6)]
    m = _safe_clone(model); m.fit(X, y)
    assert r2_score(y, m.predict(X)) > 0.5
    response = ask_llm(llm, get_model_str(m, names),
                       "Which single feature is most important for predicting the output? "
                       "Answer with just the feature name (e.g., 'x0', 'x3').")
    return dict(test="new_most_important_feature", passed=bool(response and "x0" in response.lower()),
                ground_truth="x0", response=response)

def new_test_point_prediction(model, llm):
    X, y = _v_single_feature_data(n_features=3, true_feature=0, coef=6.0, seed=100)
    names = [f"x{i}" for i in range(3)]
    m = _safe_clone(model); m.fit(X, y)
    true_pred = float(m.predict(np.array([[1.5, 0.0, 0.0]]))[0])
    response = ask_llm(llm, get_model_str(m, names),
                       "What does this model predict for the input x0=1.5, x1=0.0, x2=0.0? "
                       "Answer with just a single number (e.g., '10.5').")
    llm_val, passed = None, False
    nums = re.findall(r"-?\d+\.?\d*", response or "")
    if nums:
        try:
            llm_val = float(nums[0])
            passed = abs(llm_val - true_pred) < max(abs(true_pred) * 0.25, 1.5)
        except ValueError: pass
    return dict(test="new_point_prediction", passed=passed,
                ground_truth=round(true_pred, 3), response=response)

def new_test_direction_of_change(model, llm):
    X, y = _v_single_feature_data(n_features=4, true_feature=0, coef=7.0, seed=101)
    names = [f"x{i}" for i in range(4)]
    m = _safe_clone(model); m.fit(X, y)
    x1 = np.zeros((1, 4)); x1[0, 0] = 1.5
    true_change = float(m.predict(x1)[0]) - float(m.predict(np.zeros((1, 4)))[0])
    response = ask_llm(llm, get_model_str(m, names),
                       "By how much does the prediction change when x0 increases from 0.0 to 1.5 "
                       "(all other features stay at 0.0)? "
                       "Give just a number (positive if prediction increases, negative if it decreases).")
    llm_val, passed = None, False
    nums = re.findall(r"-?\d+\.?\d*", response or "")
    if nums:
        try:
            llm_val = float(nums[0])
            passed = abs(llm_val - true_change) < max(abs(true_change) * 0.25, 1.5)
        except ValueError: pass
    return dict(test="new_direction_of_change", passed=passed,
                ground_truth=round(true_change, 3), response=response)

def new_test_feature_ranking(model, llm):
    X, y = _v_multi_feature_data([6.0, 2.5, 1.0, 0.0, 0.0])
    names = [f"x{i}" for i in range(5)]
    m = _safe_clone(model); m.fit(X, y)
    response = ask_llm(llm, get_model_str(m, names),
                       "Rank the 3 most important features from most to least important. "
                       "Answer with just the feature names separated by commas (e.g., 'x2, x0, x4').")
    passed = False
    if response:
        r = response.lower()
        pos = {f"x{i}": r.find(f"x{i}") for i in range(5)}
        passed = pos["x0"] != -1 and pos["x1"] != -1 and pos["x0"] < pos["x1"]
    return dict(test="new_feature_ranking", passed=passed, ground_truth="x0, x1, x2", response=response)

def new_test_threshold_identification(model, llm):
    X, y = _v_threshold_data(threshold=0.3)
    names = [f"x{i}" for i in range(3)]
    m = _safe_clone(model); m.fit(X, y)
    response = ask_llm(llm, get_model_str(m, names),
                       "What approximate threshold value for x0 separates low predictions from "
                       "high predictions? Answer with just a number.")
    llm_threshold, passed = None, False
    nums = re.findall(r"-?\d+\.?\d*", response or "")
    if nums:
        try:
            llm_threshold = float(nums[0])
            passed = abs(llm_threshold - 0.3) < 0.35
        except ValueError: pass
    return dict(test="new_threshold_identification", passed=passed,
                ground_truth=0.3, response=response)

def new_test_irrelevant_features(model, llm):
    X, y = _v_single_feature_data(n_features=6, true_feature=0, coef=12.0, seed=104)
    names = [f"x{i}" for i in range(6)]
    m = _safe_clone(model); m.fit(X, y)
    response = ask_llm(llm, get_model_str(m, names),
                       "Which features appear to have little or no effect on the prediction? "
                       "List all such feature names, comma-separated.")
    passed = bool(response and sum(f"x{i}" in response.lower() for i in range(1, 6)) >= 3)
    return dict(test="new_irrelevant_features", passed=passed,
                ground_truth="x1, x2, x3, x4, x5 are irrelevant", response=response)

def new_test_sign_of_effect(model, llm):
    X, y = _v_signed_data()
    names = [f"x{i}" for i in range(4)]
    m = _safe_clone(model); m.fit(X, y)
    delta = (float(m.predict(np.array([[0.0, 1.0, 0.0, 0.0]]))[0])
             - float(m.predict(np.zeros((1, 4)))[0]))
    response = ask_llm(llm, get_model_str(m, names),
                       "By how much does the prediction change when x1 increases by 1 unit "
                       "(all other features at 0)? Give just a number.")
    llm_val, passed = None, False
    nums = re.findall(r"-?\d+\.?\d*", response or "")
    if nums:
        try:
            llm_val = float(nums[0])
            passed = abs(llm_val - delta) < max(abs(delta) * 0.25, 1.0)
        except ValueError: pass
    return dict(test="new_sign_of_effect", passed=passed,
                ground_truth=round(delta, 3), response=response)

def new_test_counterfactual_prediction(model, llm):
    X, y = _v_single_feature_data(n_features=3, true_feature=0, coef=5.0, seed=105)
    names = [f"x{i}" for i in range(3)]
    m = _safe_clone(model); m.fit(X, y)
    pred_a = float(m.predict(np.array([[0.5, 0.0, 0.0]]))[0])
    pred_b = float(m.predict(np.array([[2.5, 0.0, 0.0]]))[0])
    response = ask_llm(llm, get_model_str(m, names),
                       f"The model predicts {pred_a:.2f} for the input x0=0.5, x1=0.0, x2=0.0. "
                       f"Using the model shown above (without running any code), "
                       f"what would it predict for x0=2.5, x1=0.0, x2=0.0? Answer with just a number.")
    llm_pred, passed = None, False
    nums = re.findall(r"-?\d+\.?\d*", response or "")
    if nums:
        try:
            llm_pred = float(nums[0])
            passed = abs(llm_pred - pred_b) < max(abs(pred_b) * 0.25, 1.5)
        except ValueError: pass
    return dict(test="new_counterfactual_prediction", passed=passed,
                ground_truth=round(pred_b, 3), response=response)

# --- Hard tests (varied) ---

def new_hard_test_all_features_active(model, llm):
    X, y = _v_multi_feature_data([2.5, 3.0, 1.5], n=600, seed=110)
    names = [f"x{i}" for i in range(3)]
    m = _safe_clone(model); m.fit(X, y)
    assert r2_score(y, m.predict(X)) > 0.5
    true_pred = float(m.predict(np.array([[1.2, -0.6, 0.9]]))[0])
    response = ask_llm(llm, get_model_str(m, names),
                       "What does this model predict for the input x0=1.2, x1=-0.6, x2=0.9? "
                       "All three features are active. Answer with just a number.")
    llm_val, passed = None, False
    nums = re.findall(r"-?\d+\.?\d*", response or "")
    if nums:
        try:
            llm_val = float(nums[0])
            passed = abs(llm_val - true_pred) < max(abs(true_pred) * 0.15, 1.0)
        except ValueError: pass
    return dict(test="new_hard_all_features_active", passed=passed,
                ground_truth=round(true_pred, 3), response=response)

def new_hard_test_pairwise_anti_intuitive(model, llm):
    X, y = _v_multi_feature_data([4.0, 3.5, 0.0, 0.0, 0.0], n=600, seed=111)
    names = [f"x{i}" for i in range(5)]
    m = _safe_clone(model); m.fit(X, y)
    pred_a = float(m.predict(np.array([[1.5, 0.2, 0.0, 0.0, 0.0]]))[0])
    pred_b = float(m.predict(np.array([[0.3, 2.8, 0.0, 0.0, 0.0]]))[0])
    diff = pred_b - pred_a
    response = ask_llm(llm, get_model_str(m, names),
                       "Sample A has features: x0=1.5, x1=0.2, x2=0.0, x3=0.0, x4=0.0\n"
                       "Sample B has features: x0=0.3, x1=2.8, x2=0.0, x3=0.0, x4=0.0\n"
                       "What is the model's prediction for sample B minus the prediction for sample A? "
                       "Give just a number (positive if B is higher, negative if A is higher).")
    llm_val, passed = None, False
    nums = re.findall(r"-?\d+\.?\d*", response or "")
    if nums:
        try:
            llm_val = float(nums[0])
            passed = abs(llm_val - diff) < max(abs(diff) * 0.2, 1.0)
        except ValueError: pass
    return dict(test="new_hard_pairwise_anti_intuitive", passed=passed,
                ground_truth=round(diff, 3), response=response)

def new_hard_test_quantitative_sensitivity(model, llm):
    X, y = _v_single_feature_data(n_features=3, true_feature=0, coef=5.0, seed=112)
    names = [f"x{i}" for i in range(3)]
    m = _safe_clone(model); m.fit(X, y)
    delta = float(m.predict(np.array([[3.0, 0.0, 0.0]]))[0]) - float(m.predict(np.array([[1.0, 0.0, 0.0]]))[0])
    response = ask_llm(llm, get_model_str(m, names),
                       "By how much does the model's prediction change when x0 increases from 1.0 "
                       "to 3.0 (keeping x1=0, x2=0)? Answer with just a number (the difference).")
    llm_val, passed = None, False
    nums = re.findall(r"-?\d+\.?\d*", response or "")
    if nums:
        try:
            llm_val = float(nums[0])
            passed = abs(llm_val - delta) < max(abs(delta) * 0.15, 1.0)
        except ValueError: pass
    return dict(test="new_hard_quantitative_sensitivity", passed=passed,
                ground_truth=round(delta, 3), response=response)

def new_hard_test_mixed_sign_goes_negative(model, llm):
    X, y = _v_multi_feature_data([2.0, -3.0, 1.5], n=600, seed=113)
    names = [f"x{i}" for i in range(3)]
    m = _safe_clone(model); m.fit(X, y)
    assert r2_score(y, m.predict(X)) > 0.5
    true_pred = float(m.predict(np.array([[0.5, 2.0, 0.5]]))[0])
    response = ask_llm(llm, get_model_str(m, names),
                       "What does this model predict for x0=0.5, x1=2.0, x2=0.5? "
                       "Answer with just a number (can be negative).")
    llm_val, passed = None, False
    nums = re.findall(r"-?\d+\.?\d*", response or "")
    if nums:
        try:
            llm_val = float(nums[0])
            passed = abs(llm_val - true_pred) < max(abs(true_pred) * 0.2, 1.0)
        except ValueError: pass
    return dict(test="new_hard_mixed_sign_goes_negative", passed=passed,
                ground_truth=round(true_pred, 3), response=response)

def new_hard_test_two_feature_perturbation(model, llm):
    X, y = _v_multi_feature_data([2.5, 3.0, 0.0, 0.0], n=500, seed=114)
    names = [f"x{i}" for i in range(4)]
    m = _safe_clone(model); m.fit(X, y)
    pred_base = float(m.predict(np.zeros((1, 4)))[0])
    pred_new = float(m.predict(np.array([[1.5, 2.0, 0.0, 0.0]]))[0])
    response = ask_llm(llm, get_model_str(m, names),
                       f"The model predicts {pred_base:.2f} for the input x0=0, x1=0, x2=0, x3=0. "
                       f"What does it predict when both x0 changes to 1.5 AND x1 changes to 2.0 "
                       f"(x2 and x3 stay at 0)? Answer with just a number.")
    llm_val, passed = None, False
    nums = re.findall(r"-?\d+\.?\d*", response or "")
    if nums:
        try:
            llm_val = float(nums[0])
            passed = abs(llm_val - pred_new) < max(abs(pred_new) * 0.15, 1.0)
        except ValueError: pass
    return dict(test="new_hard_two_feature_perturbation", passed=passed,
                ground_truth=round(pred_new, 3), response=response)

# --- Insight tests (varied) ---

def new_insight_simulatability(model, llm):
    X, y = _v_multi_feature_data([4.0, 2.5, 0.0, 0.0], n=500, seed=120)
    names = [f"x{i}" for i in range(4)]
    m = _safe_clone(model); m.fit(X, y)
    true_pred = float(m.predict(np.array([[1.5, 1.0, 0.3, -0.8]]))[0])
    response = ask_llm(llm, get_model_str(m, names),
                       "What does this model predict for x0=1.5, x1=1.0, x2=0.3, x3=-0.8? "
                       "Answer with just the predicted value as a single number.")
    tol = max(abs(true_pred) * 0.15, 1.0)
    passed, llm_val = False, None
    for num_str in reversed(re.findall(r"-?\d+\.?\d*", response or "")):
        try:
            val = float(num_str)
            if abs(val - true_pred) < tol:
                llm_val = val; passed = True; break
        except ValueError: pass
    return dict(test="new_insight_simulatability", passed=passed,
                ground_truth=round(true_pred, 3), response=response)

def new_insight_sparse_feature_set(model, llm):
    X, y = _v_sparse_ten_feature_data()
    names = [f"x{i}" for i in range(10)]
    m = _safe_clone(model); m.fit(X, y)
    response = ask_llm(llm, get_model_str(m, names),
                       "This model was trained on 10 features (x0-x9). "
                       "Based solely on the model shown above, list ONLY the features that "
                       "contribute meaningfully to predictions. Exclude features with negligible "
                       "or zero effect. Give just a comma-separated list of feature names.")
    passed = False
    if response:
        listed_set = set(re.findall(r"x\d+", response.lower()))
        passed = "x0" in listed_set and "x1" in listed_set and len(listed_set - {"x0", "x1"}) <= 1
    return dict(test="new_insight_sparse_feature_set", passed=passed,
                ground_truth="x0, x1 only", response=response)

def new_insight_nonlinear_threshold(model, llm):
    X, y = _v_hockey_stick_data()
    names = [f"x{i}" for i in range(3)]
    m = _safe_clone(model); m.fit(X, y)
    r2 = r2_score(y, m.predict(X))
    response = ask_llm(llm, get_model_str(m, names),
                       "For x1=0 and x2=0, what is the approximate threshold value of x0 "
                       "below which x0 has little or no effect on the prediction? "
                       "Answer with just a number.")
    threshold_ok = False
    if response:
        r = response.lower()
        nums = re.findall(r"-?\d+\.?\d*", r)
        threshold_ok = (any(abs(float(n)) < 0.7 for n in nums) or
                        any(w in r for w in ["zero", "negative", "below zero", "0.0", "flat",
                                             "constant", "no effect", "hockey", "piecewise", "relu"]))
    return dict(test="new_insight_nonlinear_threshold", passed=threshold_ok and r2 > 0.5,
                ground_truth="~0 (flat for x0<0)", response=response)

def new_insight_nonlinear_direction(model, llm):
    X, y = _v_hockey_stick_data()
    names = [f"x{i}" for i in range(3)]
    m = _safe_clone(model); m.fit(X, y)
    r2 = r2_score(y, m.predict(X))
    true_pred = float(m.predict(np.array([[1.5, 0.0, 0.0]]))[0])
    response = ask_llm(llm, get_model_str(m, names),
                       "What does this model predict for x0=1.5, x1=0.0, x2=0.0? "
                       "Give just a number.")
    llm_val, passed = None, False
    nums = re.findall(r"-?\d+\.?\d*", response or "")
    if nums:
        try:
            llm_val = float(nums[0])
            passed = abs(llm_val - true_pred) < max(abs(true_pred) * 0.2, 1.0)
        except ValueError: pass
    return dict(test="new_insight_nonlinear_direction", passed=passed and r2 > 0.5,
                ground_truth=round(true_pred, 3), response=response)

def new_insight_counterfactual_target(model, llm):
    X, y = _v_multi_feature_data([3.5, 2.5, 0.0], n=500, seed=122)
    names = [f"x{i}" for i in range(3)]
    m = _safe_clone(model); m.fit(X, y)
    pred_base = float(m.predict(np.array([[0.5, 1.5, 0.0]]))[0])
    target = pred_base + 7.0
    lo, hi = -10.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if float(m.predict(np.array([[mid, 1.5, 0.0]]))[0]) < target: lo = mid
        else: hi = mid
    true_x0 = (lo + hi) / 2
    response = ask_llm(llm, get_model_str(m, names),
                       f"The model predicts {pred_base:.2f} for x0=0.5, x1=1.5, x2=0.0. "
                       f"What value of x0 (keeping x1=1.5 and x2=0.0 fixed) would make the model "
                       f"predict {target:.2f}? Answer with just a number.")
    llm_val, passed = None, False
    nums = re.findall(r"-?\d+\.?\d*", response or "")
    if nums:
        try:
            llm_val = float(nums[0])
            passed = abs(llm_val - true_x0) < max(abs(true_x0) * 0.15, 0.5)
        except ValueError: pass
    return dict(test="new_insight_counterfactual_target", passed=passed,
                ground_truth=round(true_x0, 3), response=response)

def new_insight_decision_region(model, llm):
    X, y = _v_single_feature_data(n_features=3, true_feature=0, coef=5.0, seed=123)
    names = [f"x{i}" for i in range(3)]
    m = _safe_clone(model); m.fit(X, y)
    threshold_y = 7.5
    lo, hi = -5.0, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if float(m.predict(np.array([[mid, 0.0, 0.0]]))[0]) < threshold_y: lo = mid
        else: hi = mid
    true_x0_boundary = (lo + hi) / 2
    response = ask_llm(llm, get_model_str(m, names),
                       "With x1=0 and x2=0, for what values of x0 does this model predict ABOVE 7.5? "
                       "Give the threshold value of x0 (e.g., 'x0 > 1.5'). "
                       "Answer with just the threshold number.")
    llm_val, passed = None, False
    nums = re.findall(r"-?\d+\.?\d*", response or "")
    if nums:
        try:
            llm_val = float(nums[0])
            passed = abs(llm_val - true_x0_boundary) < 0.4
        except ValueError: pass
    return dict(test="new_insight_decision_region", passed=passed,
                ground_truth=round(true_x0_boundary, 3), response=response)

# --- Discrimination tests (varied) ---

def new_discrim_test_simulate_all_active(model, llm):
    X, y = _v_multi_feature_data([3.5, 2.5, 1.5, 2.0, 0.8], n=700, seed=150)
    names = [f"x{i}" for i in range(5)]
    m = _safe_clone(model); m.fit(X, y)
    assert r2_score(y, m.predict(X)) > 0.5
    sample = np.array([[0.9, -1.2, 1.8, -0.7, 1.1]])
    true_pred = float(m.predict(sample)[0])
    response = ask_llm(llm, get_model_str(m, names),
                       "What does this model predict for x0=0.9, x1=-1.2, x2=1.8, x3=-0.7, x4=1.1? "
                       "Answer with just a single number.")
    tol = max(abs(true_pred) * 0.2, 1.5)
    passed = False
    for num_str in reversed(re.findall(r"-?\d+\.?\d*", response or "")):
        try:
            if abs(float(num_str) - true_pred) < tol: passed = True; break
        except ValueError: pass
    return dict(test="new_discrim_simulate_all_active", passed=passed,
                ground_truth=round(true_pred, 3), response=response)

def new_discrim_test_compactness(model, llm):
    X, y = _v_multi_feature_data([7.0, 0.0, 0.0, 0.0, 0.0, 0.0], n=500, seed=151)
    names = [f"x{i}" for i in range(6)]
    m = _safe_clone(model); m.fit(X, y)
    response = ask_llm(llm, get_model_str(m, names),
                       "Can this entire model be computed in 10 or fewer rules or arithmetic operations "
                       "starting from the feature values? Answer with exactly 'yes' or 'no'.",
                       max_tokens=5)
    passed = bool(response and "yes" in response.lower())
    return dict(test="new_discrim_compactness", passed=passed,
                ground_truth=None, response=response)

def new_discrim_test_dominant_feature_sample(model, llm):
    X, y = _v_multi_feature_data([8.0, 0.5, 1.0, 0.0], n=500, seed=152)
    names = [f"x{i}" for i in range(4)]
    m = _safe_clone(model); m.fit(X, y)
    response = ask_llm(llm, get_model_str(m, names),
                       "For the sample x0=1.5, x1=0.2, x2=0.1, x3=0.0, "
                       "which single feature contributes the MOST to the prediction? "
                       "Answer with just the feature name (e.g., 'x0', 'x3').")
    passed = bool(response and "x0" in response.lower())
    return dict(test="new_discrim_dominant_feature_sample", passed=passed,
                ground_truth="x0", response=response)

def new_discrim_test_unit_sensitivity(model, llm):
    X, y = _v_multi_feature_data([4.0, 2.5, 0.0, 0.0], n=500, seed=153)
    names = [f"x{i}" for i in range(4)]
    m = _safe_clone(model); m.fit(X, y)
    delta = (float(m.predict(np.array([[1.0, 0.0, 0.0, 0.0]]))[0])
             - float(m.predict(np.array([[0.0, 0.0, 0.0, 0.0]]))[0]))
    response = ask_llm(llm, get_model_str(m, names),
                       "With x1=0, x2=0, x3=0, by exactly how much does the model's prediction change "
                       "when x0 increases from 0 to 1? Give just a single number.")
    llm_val, passed = None, False
    nums = re.findall(r"-?\d+\.?\d*", response or "")
    if nums:
        try:
            llm_val = float(nums[0])
            passed = abs(llm_val - delta) < max(abs(delta) * 0.10, 0.5)
        except ValueError: pass
    return dict(test="new_discrim_unit_sensitivity", passed=passed,
                ground_truth=round(delta, 3), response=response)

def new_discrim_test_predict_above_threshold(model, llm):
    X, y = _v_threshold_data(threshold=0.8, n=600, seed=154)
    names = [f"x{i}" for i in range(3)]
    m = _safe_clone(model); m.fit(X, y)
    assert r2_score(y, m.predict(X)) > 0.5
    pred_a = round(float(m.predict(np.array([[1.5, 0.0, 0.0]]))[0]), 2)
    response = ask_llm(llm, get_model_str(m, names),
                       "What does this model predict for x0=1.5, x1=0.0, x2=0.0? "
                       "Answer with just a single number.")
    tol_a = max(abs(pred_a) * 0.2, 0.5)
    passed = False
    nums = re.findall(r"-?\d+\.?\d*", response or "")
    if nums:
        try: passed = abs(float(nums[0]) - pred_a) < tol_a
        except ValueError: pass
    return dict(test="new_discrim_predict_above_threshold", passed=passed,
                ground_truth=pred_a, response=response)

def new_discrim_test_predict_below_threshold(model, llm):
    X, y = _v_threshold_data(threshold=0.8, n=600, seed=154)
    names = [f"x{i}" for i in range(3)]
    m = _safe_clone(model); m.fit(X, y)
    assert r2_score(y, m.predict(X)) > 0.5
    pred_b = round(float(m.predict(np.array([[-1.0, 0.0, 0.0]]))[0]), 2)
    response = ask_llm(llm, get_model_str(m, names),
                       "What does this model predict for x0=-1.0, x1=0.0, x2=0.0? "
                       "Answer with just a single number.")
    tol_b = max(abs(pred_b) * 0.2, 0.5)
    passed = False
    nums = re.findall(r"-?\d+\.?\d*", response or "")
    if nums:
        try: passed = abs(float(nums[0]) - pred_b) < tol_b
        except ValueError: pass
    return dict(test="new_discrim_predict_below_threshold", passed=passed,
                ground_truth=pred_b, response=response)

def new_discrim_test_simulate_mixed_sign(model, llm):
    X, y = _v_mixed_sign_six_feature_data()
    names = [f"x{i}" for i in range(6)]
    m = _safe_clone(model); m.fit(X, y)
    assert r2_score(y, m.predict(X)) > 0.5
    sample = np.array([[1.2, -0.8, 1.0, 1.5, -0.3, 0.9]])
    true_pred = float(m.predict(sample)[0])
    response = ask_llm(llm, get_model_str(m, names),
                       "What does this model predict for x0=1.2, x1=-0.8, x2=1.0, x3=1.5, x4=-0.3, x5=0.9? "
                       "Answer with just a single number.")
    tol = max(abs(true_pred) * 0.2, 1.5)
    passed = False
    for num_str in reversed(re.findall(r"-?\d+\.?\d*", response or "")):
        try:
            if abs(float(num_str) - true_pred) < tol: passed = True; break
        except ValueError: pass
    return dict(test="new_discrim_simulate_mixed_sign", passed=passed,
                ground_truth=round(true_pred, 3), response=response)

def new_discrim_test_simulate_double_threshold(model, llm):
    X, y = _v_double_threshold_data()
    names = [f"x{i}" for i in range(4)]
    m = _safe_clone(model); m.fit(X, y)
    assert r2_score(y, m.predict(X)) > 0.5
    sample = np.array([[0.5, 0.0, 0.0, 0.0]])
    true_pred = float(m.predict(sample)[0])
    response = ask_llm(llm, get_model_str(m, names),
                       "What does this model predict for x0=0.5, x1=0.0, x2=0.0, x3=0.0? "
                       "Answer with just a single number.")
    tol = max(abs(true_pred) * 0.2, 0.6)
    passed = False
    nums = re.findall(r"-?\d+\.?\d*", response or "")
    if nums:
        try: passed = abs(float(nums[0]) - true_pred) < tol
        except ValueError: pass
    return dict(test="new_discrim_simulate_double_threshold", passed=passed,
                ground_truth=round(true_pred, 3), response=response)

def new_discrim_test_simulate_additive_nonlinear(model, llm):
    X, y = _v_additive_nonlinear_data()
    names = [f"x{i}" for i in range(5)]
    m = _safe_clone(model); m.fit(X, y)
    assert r2_score(y, m.predict(X)) > 0.4
    sample = np.array([[1.0, 1.5, -0.3, 0.0, 0.0]])
    true_pred = float(m.predict(sample)[0])
    response = ask_llm(llm, get_model_str(m, names),
                       "What does this model predict for x0=1.0, x1=1.5, x2=-0.3, x3=0.0, x4=0.0? "
                       "Answer with just a single number.")
    tol = max(abs(true_pred) * 0.2, 1.0)
    passed = False
    for num_str in reversed(re.findall(r"-?\d+\.?\d*", response or "")):
        try:
            if abs(float(num_str) - true_pred) < tol: passed = True; break
        except ValueError: pass
    return dict(test="new_discrim_simulate_additive_nonlinear", passed=passed,
                ground_truth=round(true_pred, 3), response=response)

def new_discrim_test_simulate_interaction(model, llm):
    X, y = _v_interaction_data()
    names = [f"x{i}" for i in range(4)]
    m = _safe_clone(model); m.fit(X, y)
    assert r2_score(y, m.predict(X)) > 0.4
    sample = np.array([[1.5, 2.0, 0.0, 0.0]])
    true_pred = float(m.predict(sample)[0])
    response = ask_llm(llm, get_model_str(m, names),
                       "What does this model predict for x0=1.5, x1=2.0, x2=0.0, x3=0.0? "
                       "Answer with just a single number.")
    tol = max(abs(true_pred) * 0.2, 1.5)
    passed = False
    for num_str in reversed(re.findall(r"-?\d+\.?\d*", response or "")):
        try:
            if abs(float(num_str) - true_pred) < tol: passed = True; break
        except ValueError: pass
    return dict(test="new_discrim_simulate_interaction", passed=passed,
                ground_truth=round(true_pred, 3), response=response)

# --- Simulatability tests (varied) ---

def new_simulatability_eight_features(model, llm):
    X, y, _ = _v_eight_feature_mixed_data()
    names = [f"x{i}" for i in range(8)]
    m = _safe_clone(model); m.fit(X, y)
    sample = np.array([[0.8, -1.2, 0.3, 1.5, -0.5, 0.9, -0.7, 0.4]])
    true_pred = float(m.predict(sample)[0])
    response = ask_llm(llm, get_model_str(m, names),
                       "What does this model predict for x0=0.8, x1=-1.2, x2=0.3, x3=1.5, "
                       "x4=-0.5, x5=0.9, x6=-0.7, x7=0.4? Answer with just a single number.")
    tol = max(abs(true_pred) * 0.15, 1.5)
    passed = False
    for num_str in reversed(re.findall(r"-?\d+\.?\d*", response or "")):
        try:
            if abs(float(num_str) - true_pred) < tol: passed = True; break
        except ValueError: pass
    return dict(test="new_simulatability_eight_features", passed=passed,
                ground_truth=round(true_pred, 3), response=response)

def new_simulatability_fifteen_features_sparse(model, llm):
    X, y, _ = _v_fifteen_feature_sparse_data()
    names = [f"x{i}" for i in range(15)]
    m = _safe_clone(model); m.fit(X, y)
    sample = np.zeros((1, 15))
    sample[0, 1] = 1.0   # active: coef 7.0
    sample[0, 6] = -1.5  # active: coef -3.5
    sample[0, 10] = 0.8  # active: coef 4.0
    sample[0, 4] = 0.5   # noise
    sample[0, 12] = -0.3 # noise
    true_pred = float(m.predict(sample)[0])
    feat_str = ", ".join(f"x{i}={sample[0,i]}" for i in range(15) if sample[0, i] != 0)
    response = ask_llm(llm, get_model_str(m, names),
                       f"What does this model predict for the input where {feat_str} "
                       f"and all other features are 0? Answer with just a single number.")
    tol = max(abs(true_pred) * 0.15, 2.0)
    passed = False
    for num_str in reversed(re.findall(r"-?\d+\.?\d*", response or "")):
        try:
            if abs(float(num_str) - true_pred) < tol: passed = True; break
        except ValueError: pass
    return dict(test="new_simulatability_fifteen_features_sparse", passed=passed,
                ground_truth=round(true_pred, 3), response=response)

def new_simulatability_quadratic(model, llm):
    X, y = _v_quadratic_data()
    names = [f"x{i}" for i in range(5)]
    m = _safe_clone(model); m.fit(X, y)
    sample = np.array([[1.0, -1.5, 0.8, 0.0, 0.0]])
    true_pred = float(m.predict(sample)[0])
    response = ask_llm(llm, get_model_str(m, names),
                       "What does this model predict for x0=1.0, x1=-1.5, x2=0.8, x3=0.0, x4=0.0? "
                       "Answer with just a single number.")
    tol = max(abs(true_pred) * 0.2, 1.5)
    passed = False
    for num_str in reversed(re.findall(r"-?\d+\.?\d*", response or "")):
        try:
            if abs(float(num_str) - true_pred) < tol: passed = True; break
        except ValueError: pass
    return dict(test="new_simulatability_quadratic", passed=passed,
                ground_truth=round(true_pred, 3), response=response)

def new_simulatability_friedman1(model, llm):
    X, y = _v_friedman1_data()
    names = [f"x{i}" for i in range(10)]
    m = _safe_clone(model); m.fit(X, y)
    sample = np.array([[0.5, 0.4, 0.6, 0.7, 0.3, 0.2, 0.8, 0.1, 0.5, 0.6]])
    true_pred = float(m.predict(sample)[0])
    response = ask_llm(llm, get_model_str(m, names),
                       "What does this model predict for x0=0.5, x1=0.4, x2=0.6, x3=0.7, x4=0.3, "
                       "x5=0.2, x6=0.8, x7=0.1, x8=0.5, x9=0.6? "
                       "Answer with just a single number.")
    tol = max(abs(true_pred) * 0.2, 2.0)
    passed = False
    for num_str in reversed(re.findall(r"-?\d+\.?\d*", response or "")):
        try:
            if abs(float(num_str) - true_pred) < tol: passed = True; break
        except ValueError: pass
    return dict(test="new_simulatability_friedman1", passed=passed,
                ground_truth=round(true_pred, 3), response=response)

def new_simulatability_cascading_threshold(model, llm):
    X, y = _v_cascading_threshold_data()
    names = [f"x{i}" for i in range(6)]
    m = _safe_clone(model); m.fit(X, y)
    sample = np.array([[0.8, 1.2, -0.3, 0.5, 0.0, 0.0]])
    true_pred = float(m.predict(sample)[0])
    response = ask_llm(llm, get_model_str(m, names),
                       "What does this model predict for x0=0.8, x1=1.2, x2=-0.3, x3=0.5, x4=0.0, x5=0.0? "
                       "Answer with just a single number.")
    tol = max(abs(true_pred) * 0.2, 1.0)
    passed = False
    for num_str in reversed(re.findall(r"-?\d+\.?\d*", response or "")):
        try:
            if abs(float(num_str) - true_pred) < tol: passed = True; break
        except ValueError: pass
    return dict(test="new_simulatability_cascading_threshold", passed=passed,
                ground_truth=round(true_pred, 3), response=response)

def new_simulatability_exponential_decay(model, llm):
    X, y = _v_exponential_decay_data()
    names = [f"x{i}" for i in range(4)]
    m = _safe_clone(model); m.fit(X, y)
    sample = np.array([[1.0, 0.5, 0.0, 0.0]])
    true_pred = float(m.predict(sample)[0])
    response = ask_llm(llm, get_model_str(m, names),
                       "What does this model predict for x0=1.0, x1=0.5, x2=0.0, x3=0.0? "
                       "Answer with just a single number.")
    tol = max(abs(true_pred) * 0.2, 1.0)
    passed = False
    for num_str in reversed(re.findall(r"-?\d+\.?\d*", response or "")):
        try:
            if abs(float(num_str) - true_pred) < tol: passed = True; break
        except ValueError: pass
    return dict(test="new_simulatability_exponential_decay", passed=passed,
                ground_truth=round(true_pred, 3), response=response)

def new_simulatability_piecewise_three_segment(model, llm):
    X, y = _v_piecewise_three_segment_data()
    names = [f"x{i}" for i in range(4)]
    m = _safe_clone(model); m.fit(X, y)
    sample = np.array([[0.8, 0.0, 0.0, 0.0]])
    true_pred = float(m.predict(sample)[0])
    response = ask_llm(llm, get_model_str(m, names),
                       "What does this model predict for x0=0.8, x1=0.0, x2=0.0, x3=0.0? "
                       "Answer with just a single number.")
    tol = max(abs(true_pred) * 0.2, 0.8)
    passed = False
    for num_str in reversed(re.findall(r"-?\d+\.?\d*", response or "")):
        try:
            if abs(float(num_str) - true_pred) < tol: passed = True; break
        except ValueError: pass
    return dict(test="new_simulatability_piecewise_three_segment", passed=passed,
                ground_truth=round(true_pred, 3), response=response)

def new_simulatability_twenty_features_sparse(model, llm):
    X, y, _ = _v_twenty_feature_sparse_data()
    names = [f"x{i}" for i in range(20)]
    m = _safe_clone(model); m.fit(X, y)
    sample = np.zeros((1, 20))
    sample[0, 3] = 1.0    # active: coef 5.0
    sample[0, 8] = -1.2   # active: coef -4.0
    sample[0, 13] = 0.5   # active: coef 3.0
    sample[0, 17] = -0.8  # active: coef -2.0
    sample[0, 5] = 0.4    # noise
    sample[0, 14] = -0.7  # noise
    true_pred = float(m.predict(sample)[0])
    feat_str = ", ".join(f"x{i}={sample[0,i]}" for i in range(20) if sample[0, i] != 0)
    response = ask_llm(llm, get_model_str(m, names),
                       f"What does this model predict for the input where {feat_str} "
                       f"and all other features are 0? Answer with just a single number.")
    tol = max(abs(true_pred) * 0.15, 2.0)
    passed = False
    for num_str in reversed(re.findall(r"-?\d+\.?\d*", response or "")):
        try:
            if abs(float(num_str) - true_pred) < tol: passed = True; break
        except ValueError: pass
    return dict(test="new_simulatability_twenty_features_sparse", passed=passed,
                ground_truth=round(true_pred, 3), response=response)

def new_simulatability_sinusoidal(model, llm):
    X, y = _v_sinusoidal_data()
    names = [f"x{i}" for i in range(5)]
    m = _safe_clone(model); m.fit(X, y)
    sample = np.array([[0.5, 1.0, -0.5, 0.0, 0.0]])
    true_pred = float(m.predict(sample)[0])
    response = ask_llm(llm, get_model_str(m, names),
                       "What does this model predict for x0=0.5, x1=1.0, x2=-0.5, x3=0.0, x4=0.0? "
                       "Answer with just a single number.")
    tol = max(abs(true_pred) * 0.2, 1.0)
    passed = False
    for num_str in reversed(re.findall(r"-?\d+\.?\d*", response or "")):
        try:
            if abs(float(num_str) - true_pred) < tol: passed = True; break
        except ValueError: pass
    return dict(test="new_simulatability_sinusoidal", passed=passed,
                ground_truth=round(true_pred, 3), response=response)

def new_simulatability_abs_value(model, llm):
    X, y = _v_abs_value_data()
    names = [f"x{i}" for i in range(5)]
    m = _safe_clone(model); m.fit(X, y)
    sample = np.array([[-1.0, 1.2, 0.3, 0.0, 0.0]])
    true_pred = float(m.predict(sample)[0])
    response = ask_llm(llm, get_model_str(m, names),
                       "What does this model predict for x0=-1.0, x1=1.2, x2=0.3, x3=0.0, x4=0.0? "
                       "Answer with just a single number.")
    tol = max(abs(true_pred) * 0.2, 1.0)
    passed = False
    for num_str in reversed(re.findall(r"-?\d+\.?\d*", response or "")):
        try:
            if abs(float(num_str) - true_pred) < tol: passed = True; break
        except ValueError: pass
    return dict(test="new_simulatability_abs_value", passed=passed,
                ground_truth=round(true_pred, 3), response=response)

def new_simulatability_twelve_features_all_active(model, llm):
    X, y, _ = _v_twelve_feature_all_active_data()
    names = [f"x{i}" for i in range(12)]
    m = _safe_clone(model); m.fit(X, y)
    sample = np.array([[0.8, -0.3, 1.0, 0.5, -0.7, 0.4, -1.2, 0.6, -0.1, 0.9, -0.5, 0.2]])
    true_pred = float(m.predict(sample)[0])
    response = ask_llm(llm, get_model_str(m, names),
                       "What does this model predict for x0=0.8, x1=-0.3, x2=1.0, x3=0.5, "
                       "x4=-0.7, x5=0.4, x6=-1.2, x7=0.6, x8=-0.1, x9=0.9, x10=-0.5, x11=0.2? "
                       "Answer with just a single number.")
    tol = max(abs(true_pred) * 0.15, 1.5)
    passed = False
    for num_str in reversed(re.findall(r"-?\d+\.?\d*", response or "")):
        try:
            if abs(float(num_str) - true_pred) < tol: passed = True; break
        except ValueError: pass
    return dict(test="new_simulatability_twelve_features_all_active", passed=passed,
                ground_truth=round(true_pred, 3), response=response)

def new_simulatability_nested_threshold(model, llm):
    X, y = _v_nested_threshold_data()
    names = [f"x{i}" for i in range(5)]
    m = _safe_clone(model); m.fit(X, y)
    sample = np.array([[1.0, -0.3, 0.0, 0.0, 0.0]])  # x0>0 but x1<0
    true_pred = float(m.predict(sample)[0])
    response = ask_llm(llm, get_model_str(m, names),
                       "What does this model predict for x0=1.0, x1=-0.3, x2=0.0, x3=0.0, x4=0.0? "
                       "Answer with just a single number.")
    tol = max(abs(true_pred) * 0.2, 0.8)
    passed = False
    for num_str in reversed(re.findall(r"-?\d+\.?\d*", response or "")):
        try:
            if abs(float(num_str) - true_pred) < tol: passed = True; break
        except ValueError: pass
    return dict(test="new_simulatability_nested_threshold", passed=passed,
                ground_truth=round(true_pred, 3), response=response)


# --- Test lists ---

NEW_STANDARD_TESTS = [
    new_test_most_important_feature,
    new_test_point_prediction,
    new_test_direction_of_change,
    new_test_feature_ranking,
    new_test_threshold_identification,
    new_test_irrelevant_features,
    new_test_sign_of_effect,
    new_test_counterfactual_prediction,
]

NEW_HARD_TESTS = [
    new_hard_test_all_features_active,
    new_hard_test_pairwise_anti_intuitive,
    new_hard_test_quantitative_sensitivity,
    new_hard_test_mixed_sign_goes_negative,
    new_hard_test_two_feature_perturbation,
]

NEW_INSIGHT_TESTS = [
    new_insight_simulatability,
    new_insight_sparse_feature_set,
    new_insight_nonlinear_threshold,
    new_insight_nonlinear_direction,
    new_insight_counterfactual_target,
    new_insight_decision_region,
]

NEW_DISCRIM_TESTS = [
    new_discrim_test_simulate_all_active,
    new_discrim_test_compactness,
    new_discrim_test_dominant_feature_sample,
    new_discrim_test_unit_sensitivity,
    new_discrim_test_predict_above_threshold,
    new_discrim_test_predict_below_threshold,
    new_discrim_test_simulate_mixed_sign,
    new_discrim_test_simulate_double_threshold,
    new_discrim_test_simulate_additive_nonlinear,
    new_discrim_test_simulate_interaction,
]

NEW_SIMULATABILITY_TESTS = [
    new_simulatability_eight_features,
    new_simulatability_fifteen_features_sparse,
    new_simulatability_quadratic,
    new_simulatability_friedman1,
    new_simulatability_cascading_threshold,
    new_simulatability_exponential_decay,
    new_simulatability_piecewise_three_segment,
    new_simulatability_twenty_features_sparse,
    new_simulatability_sinusoidal,
    new_simulatability_abs_value,
    new_simulatability_twelve_features_all_active,
    new_simulatability_nested_threshold,
]

ALL_NEW_TESTS = (NEW_STANDARD_TESTS + NEW_HARD_TESTS + NEW_INSIGHT_TESTS
                 + NEW_DISCRIM_TESTS + NEW_SIMULATABILITY_TESTS)

_ALL_NEW_TEST_FNS = {fn.__name__: fn for fn in ALL_NEW_TESTS}


# --- Cached runner for new tests ---

@_new_cache.cache
def _run_one_new_test(model_name, test_fn_name, model):
    llm = imodelsx.llm.get_llm(CHECKPOINT)
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


def run_all_new_interp_tests(model_defs_simple):
    """Run all new interpretability tests on all models."""
    from joblib import Parallel, delayed

    tasks = [(name, reg, test_fn)
             for name, reg in model_defs_simple
             for test_fn in ALL_NEW_TESTS]

    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_run_one_new_test)(name, test_fn.__name__, reg)
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
    model_defs_full = load_all_models()  # list of (name, model_instance, description)
    print(f"\nLoaded {len(model_defs_full)} evolved models.\n")

    # --- Combine with baselines ---
    # model_defs_full: list of (name, model, description)
    # Add baselines to the full list
    for bname, bmodel in BASELINE_DEFS:
        model_defs_full.append((bname, bmodel, BASELINE_DESCRIPTIONS.get(bname, bname)))

    # model_defs_simple: list of (name, model) for evaluation functions
    model_defs_simple = [(name, model) for name, model, _ in model_defs_full]
    print(f"Total models (evolved + baselines): {len(model_defs_simple)}\n")

    # --- Interpretability tests ---
    print("=" * 60)
    print("  Running NEW interpretability tests")
    print("=" * 60)
    interp_results = run_all_new_interp_tests(model_defs_simple)

    # --- Performance evaluation ---
    print("\n" + "=" * 60)
    print("  Running NEW performance evaluation (16 OpenML datasets)")
    print("=" * 60)
    dataset_rmses = evaluate_new_performance(model_defs_simple)

    # --- Save interpretability_results.csv ---
    interp_csv = str(NEW_RESULTS_DIR / "interpretability_results.csv")
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
    perf_csv = str(NEW_RESULTS_DIR / "performance_results.csv")
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

    # Compute ranks per dataset
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

    # --- Compute overall results (lenient: average over available datasets) ---
    def _lenient_rank_scores(dataset_rmses):
        """Like compute_rank_scores but averages over available datasets instead of requiring all."""
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

    # Compute per-model interp pass rates
    model_interp = defaultdict(lambda: {"passed": 0, "total": 0})
    for r in interp_results:
        model_interp[r["model"]]["total"] += 1
        if r["passed"]:
            model_interp[r["model"]]["passed"] += 1

    # Determine baseline names for status field
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

    # Write overall_results.csv
    overall_csv = str(NEW_RESULTS_DIR / "overall_results.csv")
    with open(overall_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OVERALL_CSV_COLS)
        writer.writeheader()
        writer.writerows(overall_rows)
    print(f"Overall results saved -> {overall_csv}")

    # --- Plot ---
    plot_interp_vs_performance(
        overall_csv,
        str(NEW_RESULTS_DIR / "interpretability_vs_performance.png"),
    )

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for row in sorted(overall_rows, key=lambda r: float(r["mean_rank"]) if r["mean_rank"] != "nan" else 999):
        status = " [baseline]" if row["status"] == "baseline" else ""
        print(f"  {row['model_name']:<25}  rank={row['mean_rank']:>6}  interp={row['frac_interpretability_tests_passed']}{status}")

    print(f"\nTotal time: {time.time() - t0:.1f}s")
