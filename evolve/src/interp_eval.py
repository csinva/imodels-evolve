"""
Interpretability tests for sklearn-style regressors, plus cached test runner.

Uses an LLM (gpt-4o) to probe whether a model's string representation
conveys meaningful, usable information about the model's behavior.

Tests:
  Standard (8): most important feature, point prediction, direction of change,
    feature ranking, threshold identification, irrelevant features, sign of effect,
    counterfactual prediction
  Hard (5): all features active, pairwise anti-intuitive, quantitative sensitivity,
    mixed sign goes negative, two-feature perturbation
  Insight (6): simulatability, sparse feature set, nonlinear threshold,
    nonlinear direction, counterfactual target, decision region
  Discrim (10): simulate complex sample, compactness, dominant feature for sample,
    unit sensitivity, predict above threshold, predict below threshold — designed to separate interpretable
    models (sparse linear, GAM, shallow tree) from black-box models (MLP, GBDT)
    and reward finer degrees of interpretability
  Simulatability (14): point predictions on increasingly complex data (8-feature
    mixed-sign, 15-feature sparse, quadratic, triple interaction, Friedman #1,
    cascading threshold, quadratic counterfactual, exponential decay, piecewise
    3-segment, 20-feature sparse, sinusoidal, abs-value, 12-feature all-active,
    nested threshold) — simple models (linear, shallow tree) remain traceable
    from their string; GBDTs and MLPs do not

Notes:
    Each test should ask only one question.
    It should ask the model to directly output an answer, not a reasoning chain. 
    Each test should ask for a specific, easily checkable answer (e.g., a feature name, a number).
    The answer should not be able to be guessed easily (e.g. do not ask binary questions like increase/decrease or A vs B).

Exports:
  ALL_TESTS, HARD_TESTS, INSIGHT_TESTS, DISCRIM_TESTS, SIMULATABILITY_TESTS — lists of test functions
  run_all_interp_tests(model_defs)       — cached runner, returns list of result dicts
"""

import os
import re
import sys
from copy import deepcopy

import numpy as np
from joblib import Memory
from imodels import FIGSRegressor, FIGSRegressorCV, HSTreeRegressor, HSTreeRegressorCV, RuleFitRegressor, TreeGAMRegressor
from pygam import LinearGAM
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor, export_text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import imodelsx.llm

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
_memory = Memory(location=os.path.join(RESULTS_DIR, "cache"), verbose=0)

CHECKPOINT = "gpt-4o"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_clone(model):
    try:
        return clone(model)
    except Exception:
        return deepcopy(model)


# ---------------------------------------------------------------------------
# Model → string
# ---------------------------------------------------------------------------

def get_model_str(model, feature_names=None):
    """Return a human-readable string for a fitted sklearn model."""
    if feature_names is None and hasattr(model, "n_features_in_"):
        feature_names = [f"x{i}" for i in range(model.n_features_in_)]

    if isinstance(model, DecisionTreeRegressor):
        tree_text = export_text(model, feature_names=feature_names, max_depth=6)
        return f"Decision Tree Regressor (max_depth={model.max_depth}):\n{tree_text}"

    if isinstance(model, GradientBoostingRegressor):
        importances = model.feature_importances_
        names = feature_names or [f"x{i}" for i in range(len(importances))]
        order = np.argsort(importances)[::-1]
        lines = [
            f"Gradient Boosted Trees ({model.n_estimators} estimators, "
            f"max_depth={model.max_depth}, lr={model.learning_rate}):",
            "Feature Importances (aggregate over all trees; individual tree interactions not shown):",
        ]
        for i in order:
            lines.append(f"  {names[i]}: {importances[i]:.4f}")
        lines.append("\nFirst estimator tree (depth ≤ 3):")
        lines.append(export_text(model.estimators_[0, 0], feature_names=feature_names, max_depth=3))
        return "\n".join(lines)

    if isinstance(model, RandomForestRegressor):
        importances = model.feature_importances_
        names = feature_names or [f"x{i}" for i in range(len(importances))]
        order = np.argsort(importances)[::-1]
        lines = ["Random Forest Regressor — Feature Importances (higher = more important):"]
        for i in order:
            lines.append(f"  {names[i]}: {importances[i]:.4f}")
        lines.append("\nFirst estimator tree (depth ≤ 3):")
        lines.append(export_text(model.estimators_[0], feature_names=feature_names, max_depth=3))
        return "\n".join(lines)

    if isinstance(model, Lasso):
        names = feature_names or [f"x{i}" for i in range(len(model.coef_))]
        active = [(n, c) for n, c in zip(names, model.coef_) if abs(c) > 1e-8]
        zeroed = [n for n, c in zip(names, model.coef_) if abs(c) <= 1e-8]
        lines = [f"LASSO Regression (L1 regularization, α={model.alpha:.3f} — promotes sparsity):"]
        if active:
            lines.append(f"  Active features ({len(active)} non-zero coefficients):")
            for n, c in active:
                lines.append(f"    {n}: {c:.4f}")
        if zeroed:
            lines.append(f"  Features with zero coefficients (excluded): {', '.join(zeroed)}")
        lines.append(f"  intercept: {model.intercept_:.4f}")
        return "\n".join(lines)

    if isinstance(model, LinearRegression):
        names = feature_names or [f"x{i}" for i in range(len(model.coef_))]
        equation = " + ".join(f"{c:.4f}*{n}" for c, n in zip(model.coef_, names))
        equation += f" + {model.intercept_:.4f}"
        lines = [f"OLS Linear Regression:  y = {equation}", "", "Coefficients:"]
        for n, c in zip(names, model.coef_):
            lines.append(f"  {n}: {c:.4f}")
        lines.append(f"  intercept: {model.intercept_:.4f}")
        return "\n".join(lines)

    if isinstance(model, RidgeCV):
        names = feature_names or [f"x{i}" for i in range(len(model.coef_))]
        equation = " + ".join(f"{c:.4f}*{n}" for c, n in zip(model.coef_, names))
        equation += f" + {model.intercept_:.4f}"
        lines = [f"Ridge Regression (L2 regularization, α={model.alpha_:.4g} chosen by CV):",
                 f"  y = {equation}", "", "Coefficients:"]
        for n, c in zip(names, model.coef_):
            lines.append(f"  {n}: {c:.4f}")
        lines.append(f"  intercept: {model.intercept_:.4f}")
        return "\n".join(lines)

    if isinstance(model, LassoCV):
        names = feature_names or [f"x{i}" for i in range(len(model.coef_))]
        active = [(n, c) for n, c in zip(names, model.coef_) if abs(c) > 1e-8]
        zeroed = [n for n, c in zip(names, model.coef_) if abs(c) <= 1e-8]
        lines = [f"LASSO Regression (L1 regularization, α={model.alpha_:.4g} chosen by CV — promotes sparsity):"]
        if active:
            lines.append(f"  Active features ({len(active)} non-zero coefficients):")
            for n, c in active:
                lines.append(f"    {n}: {c:.4f}")
        if zeroed:
            lines.append(f"  Features with zero coefficients (excluded): {', '.join(zeroed)}")
        lines.append(f"  intercept: {model.intercept_:.4f}")
        return "\n".join(lines)

    if isinstance(model, MLPRegressor):
        names = feature_names or [f"x{i}" for i in range(model.n_features_in_)]
        hidden = (model.hidden_layer_sizes
                  if isinstance(model.hidden_layer_sizes, (list, tuple))
                  else (model.hidden_layer_sizes,))
        arch = f"{model.n_features_in_} → {' → '.join(str(h) for h in hidden)} → 1"
        lines = [f"MLP Regressor (architecture: {arch}, activation: {model.activation})"]
        W0 = model.coefs_[0]
        lines.append(f"\nFirst-layer weight matrix ({W0.shape[0]} features × {W0.shape[1]} neurons):")
        lines.append("(Each row shows the weights from one input feature to all hidden neurons.)")
        for fi, name in enumerate(names):
            w = W0[fi]
            l2 = np.linalg.norm(w)
            w_str = ("[" + ", ".join(f"{v:.3f}" for v in w) + "]" if len(w) <= 8
                     else f"mean={w.mean():.3f}, std={w.std():.3f}")
            lines.append(f"  {name}: {w_str}  (L2={l2:.3f})")
        return "\n".join(lines)

    if isinstance(model, FIGSRegressorCV):
        return str(model.figs)
    if isinstance(model, HSTreeRegressorCV):
        return str(model)
    if isinstance(model, (FIGSRegressor, RuleFitRegressor, HSTreeRegressor)):
        return str(model)
    if isinstance(model, TreeGAMRegressor):
        return _tree_gam_str(model, feature_names)

    if isinstance(model, LinearGAM):
        if feature_names is None and hasattr(model, "statistics_"):
            n_f = model.statistics_.get("m_features", 0)
            feature_names = [f"x{i}" for i in range(n_f)]
        return _gam_str(model, feature_names)

    return str(model)


def _gam_str(model, feature_names=None):
    n_features = model.statistics_["m_features"]
    names = feature_names or [f"x{i}" for i in range(n_features)]
    lines = [
        "Generalized Additive Model (GAM):",
        "  y = intercept + f0(x0) + f1(x1) + ...  (each feature's effect is INDEPENDENT)",
        f"  intercept: {model.coef_[-1]:.4f}",
        "",
        "Feature partial effects (each feature's independent contribution):",
    ]
    for i, name in enumerate(names):
        try:
            XX = model.generate_X_grid(term=i, n=7)
            pdp = model.partial_dependence(term=i, X=XX)
            x_vals = XX[:, i]
            lines.append(f"\n  {name}:")
            for xv, yv in zip(x_vals, pdp):
                lines.append(f"    {name}={xv:+.2f}  →  effect={yv:+.3f}")
            if pdp[-1] > pdp[0] + 0.5:       shape = "increasing"
            elif pdp[-1] < pdp[0] - 0.5:     shape = "decreasing"
            elif max(pdp) - min(pdp) < 0.3:  shape = "flat/negligible"
            else:                              shape = "non-monotone"
            lines.append(f"    (shape: {shape})")
        except Exception:
            lines.append(f"\n  {name}: (partial effect not available)")
    return "\n".join(lines)


def _tree_gam_str(model, feature_names=None):
    n_features = len(model.estimators_)
    names = feature_names or [f"x{i}" for i in range(n_features)]
    lines = [
        "TreeGAM Regressor (additive: one interpretable tree per feature):",
        "  y = bias + tree_0(x0) + tree_1(x1) + ...  (each tree uses ONE feature)",
        f"  bias: {model.bias_:.4f}",
        "",
        "Per-feature trees (each feature's independent contribution):",
    ]
    for i, tree in enumerate(model.estimators_):
        tree_obj = tree.tree_
        feat_used = set()
        for j in range(tree_obj.node_count):
            if tree_obj.children_left[j] != -1:
                feat_used.add(int(tree_obj.feature[j]))
        feat_idx = next(iter(feat_used)) if len(feat_used) == 1 else i
        fname = names[feat_idx] if feat_idx < len(names) else f"x{feat_idx}"
        lines.append(f"\n  Tree for {fname} (feature {feat_idx}):")
        tree_text = export_text(tree, feature_names=names, max_depth=4)
        for line in tree_text.strip().split("\n"):
            lines.append("    " + line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

def ask_llm(llm, model_str, question, max_tokens=200):
    prompt = f"Here is a trained regression model:\n\n{model_str}\n\n{question}"
    return llm(prompt, max_completion_tokens=max_tokens, stop=['cannot', 'I do not have enough', "I'm sorry", ])


# ---------------------------------------------------------------------------
# Synthetic dataset factories
# ---------------------------------------------------------------------------

def _single_feature_data(n_features=5, true_feature=0, coef=10.0, n=300, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features)
    y = coef * X[:, true_feature] + rng.randn(n) * 0.5
    return X, y

def _multi_feature_data(coefs, n=500, seed=1):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, len(coefs))
    y = X @ np.array(coefs) + rng.randn(n) * 0.5
    return X, y

def _threshold_data(threshold=0.5, n=400, seed=2):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 3)
    y = np.where(X[:, 0] > threshold, 2.0, 0.0) + rng.randn(n) * 0.1
    return X, y

def _signed_data(n=400, seed=3):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 4)
    y = 5.0 * X[:, 0] - 5.0 * X[:, 1] + rng.randn(n) * 0.5
    return X, y

def _hockey_stick_data(n=700, seed=30):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 3)
    y = 3.0 * np.maximum(0.0, X[:, 0]) + 0.2 * rng.randn(n)
    return X, y

def _sparse_ten_feature_data(n=600, seed=31):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 10)
    y = 5.0 * X[:, 0] + 3.0 * X[:, 1] + 0.3 * rng.randn(n)
    return X, y

def _mixed_sign_six_feature_data(n=700, seed=40):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 6)
    y = 4.0 * X[:, 0] - 3.0 * X[:, 1] + 2.5 * X[:, 2] - 1.5 * X[:, 3] + 0.8 * X[:, 4] - 0.3 * X[:, 5] + rng.randn(n) * 0.4
    return X, y

def _double_threshold_data(n=600, seed=41):
    """Two thresholds on x0: y jumps at x0=0 and again at x0=1.5."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 4)
    y = np.where(X[:, 0] > 1.5, 4.0, np.where(X[:, 0] > 0.0, 2.0, 0.0)) + rng.randn(n) * 0.1
    return X, y

def _additive_nonlinear_data(n=800, seed=42):
    """y = 3*max(0, x0) + 2*sin(x1) + x2, additive but nonlinear."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 5)
    y = 3.0 * np.maximum(0.0, X[:, 0]) + 2.0 * np.sin(X[:, 1]) + X[:, 2] + rng.randn(n) * 0.3
    return X, y

def _interaction_data(n=700, seed=43):
    """y = 3*x0 + 2*x1 + 1.5*x0*x1, includes a pairwise interaction."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 4)
    y = 3.0 * X[:, 0] + 2.0 * X[:, 1] + 1.5 * X[:, 0] * X[:, 1] + rng.randn(n) * 0.4
    return X, y


def _eight_feature_mixed_data(n=600, seed=60):
    """8 features with mixed positive/negative coefficients of varying magnitude."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 8)
    coefs = np.array([6.0, -4.0, 3.0, -2.0, 1.5, -1.0, 0.5, 0.0])
    y = X @ coefs + rng.randn(n) * 0.5
    return X, y, coefs


def _fifteen_feature_sparse_data(n=800, seed=61):
    """15 features but only 3 are active — tests reading through a large model."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 15)
    coefs = np.zeros(15)
    coefs[0] = 8.0
    coefs[5] = -4.0
    coefs[12] = 3.0
    y = X @ coefs + rng.randn(n) * 0.5
    return X, y, coefs


def _quadratic_data(n=800, seed=62):
    """y = 3*x0^2 - 2*x1^2 + x2, quadratic nonlinearity."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 5)
    y = 3.0 * X[:, 0]**2 - 2.0 * X[:, 1]**2 + X[:, 2] + rng.randn(n) * 0.3
    return X, y


def _triple_interaction_data(n=1000, seed=63):
    """y = 2*x0*x1 + 3*x1*x2 + x0*x2*x3, multi-way interactions."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 6)
    y = (2.0 * X[:, 0] * X[:, 1] + 3.0 * X[:, 1] * X[:, 2]
         + X[:, 0] * X[:, 2] * X[:, 3] + rng.randn(n) * 0.4)
    return X, y


def _friedman1_data(n=1000, seed=64):
    """Friedman #1: y = 10*sin(pi*x0*x1) + 20*(x2-0.5)^2 + 10*x3 + 5*x4, 10 features."""
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 1, size=(n, 10))
    y = (10.0 * np.sin(np.pi * X[:, 0] * X[:, 1])
         + 20.0 * (X[:, 2] - 0.5)**2
         + 10.0 * X[:, 3]
         + 5.0 * X[:, 4]
         + rng.randn(n) * 0.5)
    return X, y


def _cascading_threshold_data(n=800, seed=65):
    """y depends on a cascade: if x0>0 then y~3*x1 else y~-2*x2. 6 features, 3 irrelevant."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 6)
    y = np.where(X[:, 0] > 0, 3.0 * X[:, 1], -2.0 * X[:, 2]) + rng.randn(n) * 0.3
    return X, y


def _exponential_decay_data(n=700, seed=70):
    """y = 5*exp(-x0) + 2*x1, mix of exponential and linear."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 4)
    y = 5.0 * np.exp(-X[:, 0]) + 2.0 * X[:, 1] + rng.randn(n) * 0.3
    return X, y


def _piecewise_three_segment_data(n=800, seed=71):
    """y is piecewise linear in x0 with 3 segments: slope 0 for x0<-1, slope 3 for -1<x0<1, slope 0.5 for x0>1."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 4)
    x0 = X[:, 0]
    y = np.where(x0 < -1, -3.0,
         np.where(x0 < 1, 3.0 * x0, 3.0 + 0.5 * (x0 - 1.0))) + rng.randn(n) * 0.2
    return X, y


def _twenty_feature_sparse_data(n=1000, seed=72):
    """20 features, only 4 active with varying coefficients."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 20)
    coefs = np.zeros(20)
    coefs[2] = 6.0
    coefs[7] = -3.5
    coefs[11] = 2.0
    coefs[18] = -1.5
    y = X @ coefs + rng.randn(n) * 0.5
    return X, y, coefs


def _sinusoidal_data(n=800, seed=73):
    """y = 4*sin(x0) + 2*cos(x1) + x2, trigonometric nonlinearity."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 5)
    y = 4.0 * np.sin(X[:, 0]) + 2.0 * np.cos(X[:, 1]) + X[:, 2] + rng.randn(n) * 0.3
    return X, y


def _abs_value_data(n=700, seed=74):
    """y = 3*|x0| - 2*|x1| + x2, V-shaped nonlinearity."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 5)
    y = 3.0 * np.abs(X[:, 0]) - 2.0 * np.abs(X[:, 1]) + X[:, 2] + rng.randn(n) * 0.3
    return X, y


def _twelve_feature_all_active_data(n=800, seed=75):
    """12 features all active with decreasing coefficients."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 12)
    coefs = np.array([5.0, -4.0, 3.5, -3.0, 2.5, -2.0, 1.5, -1.0, 0.8, -0.6, 0.4, -0.2])
    y = X @ coefs + rng.randn(n) * 0.5
    return X, y, coefs


def _nested_threshold_data(n=900, seed=76):
    """Nested thresholds: if x0>0 and x1>0 then y~5, elif x0>0 then y~2, else y~-1. Plus noise features."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 5)
    y = np.where(
        X[:, 0] > 0,
        np.where(X[:, 1] > 0, 5.0, 2.0),
        -1.0
    ) + rng.randn(n) * 0.2
    return X, y


# ---------------------------------------------------------------------------
# Standard tests
# ---------------------------------------------------------------------------

def test_most_important_feature(model, llm):
    X, y = _single_feature_data(n_features=5, true_feature=0, coef=10.0)
    names = [f"x{i}" for i in range(5)]
    m = _safe_clone(model); m.fit(X, y)
    assert r2_score(y, m.predict(X)) > 0.5
    response = ask_llm(llm, get_model_str(m, names),
                       "Which single feature is most important for predicting the output? "
                       "Answer with just the feature name (e.g., 'x0', 'x3').")
    return dict(test="most_important_feature", passed=bool(response and "x0" in response.lower()),
                ground_truth="x0", response=response)

def test_point_prediction(model, llm):
    X, y = _single_feature_data(n_features=3, true_feature=0, coef=5.0)
    names = [f"x{i}" for i in range(3)]
    m = _safe_clone(model); m.fit(X, y)
    true_pred = float(m.predict(np.array([[2.0, 0.0, 0.0]]))[0])
    response = ask_llm(llm, get_model_str(m, names),
                       "What does this model predict for the input x0=2.0, x1=0.0, x2=0.0? "
                       "Answer with just a single number (e.g., '10.5').")
    llm_val, passed = None, False
    nums = re.findall(r"-?\d+\.?\d*", response or "")
    if nums:
        try:
            llm_val = float(nums[0])
            passed = abs(llm_val - true_pred) < max(abs(true_pred) * 0.25, 1.5)
        except ValueError: pass
    return dict(test="point_prediction", passed=passed,
                ground_truth=round(true_pred, 3), response=response)

def test_direction_of_change(model, llm):
    X, y = _single_feature_data(n_features=4, true_feature=0, coef=8.0)
    names = [f"x{i}" for i in range(4)]
    m = _safe_clone(model); m.fit(X, y)
    x1 = np.zeros((1, 4)); x1[0, 0] = 1.0
    true_change = float(m.predict(x1)[0]) - float(m.predict(np.zeros((1, 4)))[0])
    response = ask_llm(llm, get_model_str(m, names),
                       "By how much does the prediction change when x0 increases from 0.0 to 1.0 "
                       "(all other features stay at 0.0)? "
                       "Give just a number (positive if prediction increases, negative if it decreases).")
    llm_val, passed = None, False
    nums = re.findall(r"-?\d+\.?\d*", response or "")
    if nums:
        try:
            llm_val = float(nums[0])
            passed = abs(llm_val - true_change) < max(abs(true_change) * 0.25, 1.5)
        except ValueError: pass
    return dict(test="direction_of_change", passed=passed,
                ground_truth=round(true_change, 3), response=response)

def test_feature_ranking(model, llm):
    X, y = _multi_feature_data([5.0, 3.0, 1.5, 0.0, 0.0])
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
    return dict(test="feature_ranking", passed=passed, ground_truth="x0, x1, x2", response=response)

def test_threshold_identification(model, llm):
    X, y = _threshold_data(threshold=0.5)
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
            passed = abs(llm_threshold - 0.5) < 0.35
        except ValueError: pass
    return dict(test="threshold_identification", passed=passed,
                ground_truth=0.5, response=response)

def test_irrelevant_features(model, llm):
    X, y = _single_feature_data(n_features=5, true_feature=0, coef=10.0, seed=4)
    names = [f"x{i}" for i in range(5)]
    m = _safe_clone(model); m.fit(X, y)
    response = ask_llm(llm, get_model_str(m, names),
                       "Which features appear to have little or no effect on the prediction? "
                       "List all such feature names, comma-separated.")
    passed = bool(response and sum(f"x{i}" in response.lower() for i in range(1, 5)) >= 2)
    return dict(test="irrelevant_features", passed=passed,
                ground_truth="x1, x2, x3, x4 are irrelevant", response=response)

def test_sign_of_effect(model, llm):
    X, y = _signed_data()
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
    return dict(test="sign_of_effect", passed=passed,
                ground_truth=round(delta, 3), response=response)

def test_counterfactual_prediction(model, llm):
    X, y = _single_feature_data(n_features=3, true_feature=0, coef=4.0, seed=5)
    names = [f"x{i}" for i in range(3)]
    m = _safe_clone(model); m.fit(X, y)
    pred_a = float(m.predict(np.array([[1.0, 0.0, 0.0]]))[0])
    pred_b = float(m.predict(np.array([[3.0, 0.0, 0.0]]))[0])
    response = ask_llm(llm, get_model_str(m, names),
                       f"The model predicts {pred_a:.2f} for the input x0=1.0, x1=0.0, x2=0.0. "
                       f"Using the model shown above (without running any code), "
                       f"what would it predict for x0=3.0, x1=0.0, x2=0.0? Answer with just a number.")
    llm_pred, passed = None, False
    nums = re.findall(r"-?\d+\.?\d*", response or "")
    if nums:
        try:
            llm_pred = float(nums[0])
            passed = abs(llm_pred - pred_b) < max(abs(pred_b) * 0.25, 1.5)
        except ValueError: pass
    return dict(test="counterfactual_prediction", passed=passed,
                ground_truth=round(pred_b, 3), response=response)


# ---------------------------------------------------------------------------
# Hard tests
# ---------------------------------------------------------------------------

def hard_test_all_features_active(model, llm):
    X, y = _multi_feature_data([3.0, 2.0, 1.0], n=600, seed=10)
    names = [f"x{i}" for i in range(3)]
    m = _safe_clone(model); m.fit(X, y)
    assert r2_score(y, m.predict(X)) > 0.5
    true_pred = float(m.predict(np.array([[1.7, 0.8, -0.5]]))[0])
    response = ask_llm(llm, get_model_str(m, names),
                       "What does this model predict for the input x0=1.7, x1=0.8, x2=-0.5? "
                       "All three features are active. Answer with just a number.")
    llm_val, passed = None, False
    nums = re.findall(r"-?\d+\.?\d*", response or "")
    if nums:
        try:
            llm_val = float(nums[0])
            passed = abs(llm_val - true_pred) < max(abs(true_pred) * 0.15, 1.0)
        except ValueError: pass
    return dict(test="hard_all_features_active", passed=passed,
                ground_truth=round(true_pred, 3), response=response)

def hard_test_pairwise_anti_intuitive(model, llm):
    X, y = _multi_feature_data([5.0, 3.0, 0.0, 0.0, 0.0], n=600, seed=11)
    names = [f"x{i}" for i in range(5)]
    m = _safe_clone(model); m.fit(X, y)
    pred_a = float(m.predict(np.array([[2.0, 0.1, 0.0, 0.0, 0.0]]))[0])
    pred_b = float(m.predict(np.array([[0.5, 3.3, 0.0, 0.0, 0.0]]))[0])
    diff = pred_b - pred_a
    response = ask_llm(llm, get_model_str(m, names),
                       "Sample A has features: x0=2.0, x1=0.1, x2=0.0, x3=0.0, x4=0.0\n"
                       "Sample B has features: x0=0.5, x1=3.3, x2=0.0, x3=0.0, x4=0.0\n"
                       "What is the model's prediction for sample B minus the prediction for sample A? "
                       "Give just a number (positive if B is higher, negative if A is higher).")
    llm_val, passed = None, False
    nums = re.findall(r"-?\d+\.?\d*", response or "")
    if nums:
        try:
            llm_val = float(nums[0])
            passed = abs(llm_val - diff) < max(abs(diff) * 0.2, 1.0)
        except ValueError: pass
    return dict(test="hard_pairwise_anti_intuitive", passed=passed,
                ground_truth=round(diff, 3), response=response)

def hard_test_quantitative_sensitivity(model, llm):
    X, y = _single_feature_data(n_features=3, true_feature=0, coef=4.0, seed=12)
    names = [f"x{i}" for i in range(3)]
    m = _safe_clone(model); m.fit(X, y)
    delta = float(m.predict(np.array([[2.5, 0.0, 0.0]]))[0]) - float(m.predict(np.array([[0.5, 0.0, 0.0]]))[0])
    response = ask_llm(llm, get_model_str(m, names),
                       "By how much does the model's prediction change when x0 increases from 0.5 "
                       "to 2.5 (keeping x1=0, x2=0)? Answer with just a number (the difference).")
    llm_val, passed = None, False
    nums = re.findall(r"-?\d+\.?\d*", response or "")
    if nums:
        try:
            llm_val = float(nums[0])
            passed = abs(llm_val - delta) < max(abs(delta) * 0.15, 1.0)
        except ValueError: pass
    return dict(test="hard_quantitative_sensitivity", passed=passed,
                ground_truth=round(delta, 3), response=response)

def hard_test_mixed_sign_goes_negative(model, llm):
    X, y = _multi_feature_data([3.0, -2.0, 1.0], n=600, seed=13)
    names = [f"x{i}" for i in range(3)]
    m = _safe_clone(model); m.fit(X, y)
    assert r2_score(y, m.predict(X)) > 0.5
    true_pred = float(m.predict(np.array([[1.0, 2.5, 1.0]]))[0])
    response = ask_llm(llm, get_model_str(m, names),
                       "What does this model predict for x0=1.0, x1=2.5, x2=1.0? "
                       "Answer with just a number (can be negative).")
    llm_val, passed = None, False
    nums = re.findall(r"-?\d+\.?\d*", response or "")
    if nums:
        try:
            llm_val = float(nums[0])
            passed = abs(llm_val - true_pred) < max(abs(true_pred) * 0.2, 1.0)
        except ValueError: pass
    return dict(test="hard_mixed_sign_goes_negative", passed=passed,
                ground_truth=round(true_pred, 3), response=response)

def hard_test_two_feature_perturbation(model, llm):
    X, y = _multi_feature_data([3.0, 2.0, 0.0, 0.0], n=500, seed=14)
    names = [f"x{i}" for i in range(4)]
    m = _safe_clone(model); m.fit(X, y)
    pred_base = float(m.predict(np.zeros((1, 4)))[0])
    pred_new  = float(m.predict(np.array([[2.0, 1.5, 0.0, 0.0]]))[0])
    response = ask_llm(llm, get_model_str(m, names),
                       f"The model predicts {pred_base:.2f} for the input x0=0, x1=0, x2=0, x3=0. "
                       f"What does it predict when both x0 changes to 2.0 AND x1 changes to 1.5 "
                       f"(x2 and x3 stay at 0)? Answer with just a number.")
    llm_val, passed = None, False
    nums = re.findall(r"-?\d+\.?\d*", response or "")
    if nums:
        try:
            llm_val = float(nums[0])
            passed = abs(llm_val - pred_new) < max(abs(pred_new) * 0.15, 1.0)
        except ValueError: pass
    return dict(test="hard_two_feature_perturbation", passed=passed,
                ground_truth=round(pred_new, 3), response=response)


# ---------------------------------------------------------------------------
# Insight tests
# ---------------------------------------------------------------------------

def insight_simulatability(model, llm):
    X, y = _multi_feature_data([5.0, 3.0, 0.0, 0.0], n=500, seed=20)
    names = [f"x{i}" for i in range(4)]
    m = _safe_clone(model); m.fit(X, y)
    true_pred = float(m.predict(np.array([[1.0, 2.0, 0.5, -0.5]]))[0])
    response = ask_llm(llm, get_model_str(m, names),
                       "What does this model predict for x0=1.0, x1=2.0, x2=0.5, x3=-0.5? "
                       "Answer with just the predicted value as a single number.")
    tol = max(abs(true_pred) * 0.15, 1.0)
    passed, llm_val = False, None
    for num_str in reversed(re.findall(r"-?\d+\.?\d*", response or "")):
        try:
            val = float(num_str)
            if abs(val - true_pred) < tol:
                llm_val = val; passed = True; break
        except ValueError: pass
    return dict(test="insight_simulatability", passed=passed,
                ground_truth=round(true_pred, 3), response=response)

def insight_sparse_feature_set(model, llm):
    X, y = _sparse_ten_feature_data()
    names = [f"x{i}" for i in range(10)]
    m = _safe_clone(model); m.fit(X, y)
    response = ask_llm(llm, get_model_str(m, names),
                       "This model was trained on 10 features (x0–x9). "
                       "Based solely on the model shown above, list ONLY the features that "
                       "contribute meaningfully to predictions. Exclude features with negligible "
                       "or zero effect. Give just a comma-separated list of feature names.")
    passed = False
    if response:
        listed_set = set(re.findall(r"x\d+", response.lower()))
        passed = "x0" in listed_set and "x1" in listed_set and len(listed_set - {"x0", "x1"}) <= 1
    return dict(test="insight_sparse_feature_set", passed=passed,
                ground_truth="x0, x1 only", response=response)

def insight_nonlinear_threshold(model, llm):
    X, y = _hockey_stick_data()
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
    return dict(test="insight_nonlinear_threshold", passed=threshold_ok and r2 > 0.5,
                ground_truth="~0 (flat for x0<0)", response=response)

def insight_nonlinear_direction(model, llm):
    X, y = _hockey_stick_data()
    names = [f"x{i}" for i in range(3)]
    m = _safe_clone(model); m.fit(X, y)
    r2 = r2_score(y, m.predict(X))
    true_pred = float(m.predict(np.array([[2.0, 0.0, 0.0]]))[0])
    response = ask_llm(llm, get_model_str(m, names),
                       "What does this model predict for x0=2.0, x1=0.0, x2=0.0? "
                       "Give just a number.")
    llm_val, passed = None, False
    nums = re.findall(r"-?\d+\.?\d*", response or "")
    if nums:
        try:
            llm_val = float(nums[0])
            passed = abs(llm_val - true_pred) < max(abs(true_pred) * 0.2, 1.0)
        except ValueError: pass
    return dict(test="insight_nonlinear_direction", passed=passed and r2 > 0.5,
                ground_truth=round(true_pred, 3), response=response)

def insight_counterfactual_target(model, llm):
    X, y = _multi_feature_data([4.0, 2.0, 0.0], n=500, seed=22)
    names = [f"x{i}" for i in range(3)]
    m = _safe_clone(model); m.fit(X, y)
    pred_base = float(m.predict(np.array([[1.0, 1.0, 0.0]]))[0])
    target = pred_base + 8.0
    lo, hi = -10.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        (lo if float(m.predict(np.array([[mid, 1.0, 0.0]]))[0]) < target else hi).__class__  # dummy
        if float(m.predict(np.array([[mid, 1.0, 0.0]]))[0]) < target: lo = mid
        else: hi = mid
    true_x0 = (lo + hi) / 2
    response = ask_llm(llm, get_model_str(m, names),
                       f"The model predicts {pred_base:.2f} for x0=1.0, x1=1.0, x2=0.0. "
                       f"What value of x0 (keeping x1=1.0 and x2=0.0 fixed) would make the model "
                       f"predict {target:.2f}? Answer with just a number.")
    llm_val, passed = None, False
    nums = re.findall(r"-?\d+\.?\d*", response or "")
    if nums:
        try:
            llm_val = float(nums[0])
            passed = abs(llm_val - true_x0) < max(abs(true_x0) * 0.15, 0.5)
        except ValueError: pass
    return dict(test="insight_counterfactual_target", passed=passed,
                ground_truth=round(true_x0, 3), response=response)

def insight_decision_region(model, llm):
    X, y = _single_feature_data(n_features=3, true_feature=0, coef=4.0, seed=23)
    names = [f"x{i}" for i in range(3)]
    m = _safe_clone(model); m.fit(X, y)
    threshold_y = 6.0
    lo, hi = -5.0, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if float(m.predict(np.array([[mid, 0.0, 0.0]]))[0]) < threshold_y: lo = mid
        else: hi = mid
    true_x0_boundary = (lo + hi) / 2
    response = ask_llm(llm, get_model_str(m, names),
                       "With x1=0 and x2=0, for what values of x0 does this model predict ABOVE 6.0? "
                       "Give the threshold value of x0 (e.g., 'x0 > 1.5'). "
                       "Answer with just the threshold number.")
    llm_val, passed = None, False
    nums = re.findall(r"-?\d+\.?\d*", response or "")
    if nums:
        try:
            llm_val = float(nums[0])
            passed = abs(llm_val - true_x0_boundary) < 0.4
        except ValueError: pass
    return dict(test="insight_decision_region", passed=passed,
                ground_truth=round(true_x0_boundary, 3), response=response)


# ---------------------------------------------------------------------------
# Discrimination tests (interpretable vs. black-box, degrees of interpretability)
# ---------------------------------------------------------------------------


def discrim_test_simulate_all_active(model, llm):
    """Simulate prediction on a complex sample where all five features are active
    at non-round values.

    Tests direct readability and simulatability of the model string on harder inputs.
    Interpretable models (sparse linear, shallow tree, GAM) can trace their own
    representation; MLPs and deep GBDTs cannot.
    """
    X, y = _multi_feature_data([4.0, 3.0, 2.0, 1.5, 0.5], n=700, seed=50)
    names = [f"x{i}" for i in range(5)]
    m = _safe_clone(model)
    m.fit(X, y)
    assert r2_score(y, m.predict(X)) > 0.5, "Model failed to fit"
    sample = np.array([[1.3, -0.7, 2.1, -1.5, 0.8]])
    true_pred = float(m.predict(sample)[0])
    response = ask_llm(
        llm, get_model_str(m, names),
        "What does this model predict for x0=1.3, x1=-0.7, x2=2.1, x3=-1.5, x4=0.8? "
        "Answer with just a single number.",
    )
    tol = max(abs(true_pred) * 0.2, 1.5)
    passed, llm_val = False, None
    for num_str in reversed(re.findall(r"-?\d+\.?\d*", response or "")):
        try:
            val = float(num_str)
            if abs(val - true_pred) < tol:
                llm_val = val
                passed = True
                break
        except ValueError:
            pass
    return dict(test="discrim_simulate_all_active", passed=passed,
                ground_truth=round(true_pred, 3), response=response)


def discrim_test_compactness(model, llm):
    """Pass if the LLM says 'yes' the model is compact (≤10 rules/operations)."""
    X, y = _multi_feature_data([6.0, 0.0, 0.0, 0.0, 0.0, 0.0], n=500, seed=51)
    names = [f"x{i}" for i in range(6)]
    m = _safe_clone(model)
    m.fit(X, y)
    response = ask_llm(
        llm, get_model_str(m, names),
        "Can this entire model be computed in 10 or fewer rules or arithmetic operations "
        "starting from the feature values? "
        "(Example: a 2-term linear equation takes ~3 operations; "
        "a model with 50 trees or 100 neurons takes many more.) "
        "Answer with exactly 'yes' or 'no'.",
        max_tokens=5,
    )
    passed = bool(response and "yes" in response.lower())
    return dict(test="discrim_compactness", passed=passed,
                ground_truth=None, response=response)


def discrim_test_dominant_feature_sample(model, llm):
    """Ask which single feature contributes most to a specific sample prediction.

    The sample is designed so x0 dominates overwhelmingly (coefficient ≈7, value=2.0).
    Additive models (linear, GAM) and shallow trees can read this directly from their
    string; MLPs cannot reason about per-feature contributions from weight matrices.
    Sparser models that highlight x0's large coefficient pass more reliably than dense ones.
    """
    X, y = _multi_feature_data([7.0, 1.0, 0.5, 0.0], n=500, seed=52)
    names = [f"x{i}" for i in range(4)]
    m = _safe_clone(model)
    m.fit(X, y)
    response = ask_llm(
        llm, get_model_str(m, names),
        "For the sample x0=2.0, x1=0.1, x2=0.1, x3=0.0, "
        "which single feature contributes the MOST to the prediction? "
        "Answer with just the feature name (e.g., 'x0', 'x3').",
    )
    passed = bool(response and "x0" in response.lower())
    return dict(test="discrim_dominant_feature_sample", passed=passed,
                ground_truth="x0 (coefficient ≈7, value=2.0 >> others)",
                response=response)


def discrim_test_unit_sensitivity(model, llm):
    """Ask the exact change in prediction when x0 increases by 1 unit.

    For a linear model this equals the coefficient and can be read directly.
    For a GAM, the partial-effect table gives the answer.
    For a shallow decision tree the LLM can trace the relevant path.
    For an MLP or deep GBDT, computing this from the raw weight/tree strings is
    intractable. A tight tolerance (10 %) rewards exact, readable representations.
    """
    X, y = _multi_feature_data([5.0, 2.0, 0.0, 0.0], n=500, seed=53)
    names = [f"x{i}" for i in range(4)]
    m = _safe_clone(model)
    m.fit(X, y)
    delta = (float(m.predict(np.array([[1.0, 0.0, 0.0, 0.0]]))[0])
             - float(m.predict(np.array([[0.0, 0.0, 0.0, 0.0]]))[0]))
    response = ask_llm(
        llm, get_model_str(m, names),
        "With x1=0, x2=0, x3=0, by exactly how much does the model's prediction change "
        "when x0 increases from 0 to 1? Give just a single number.",
    )
    llm_val, passed = None, False
    nums = re.findall(r"-?\d+\.?\d*", response or "")
    if nums:
        try:
            llm_val = float(nums[0])
            passed = abs(llm_val - delta) < max(abs(delta) * 0.10, 0.5)
        except ValueError:
            pass
    return dict(test="discrim_unit_sensitivity", passed=passed,
                ground_truth=round(delta, 3), response=response)


def discrim_test_predict_above_threshold(model, llm):
    """Predict a sample above the threshold (x0=2.0 > threshold=1.0, true output ≈2.0)."""
    X, y = _threshold_data(threshold=1.0, n=600, seed=54)
    names = [f"x{i}" for i in range(3)]
    m = _safe_clone(model)
    m.fit(X, y)
    assert r2_score(y, m.predict(X)) > 0.5, "Model failed to fit"
    pred_a = round(float(m.predict(np.array([[2.0, 0.0, 0.0]]))[0]), 2)
    response = ask_llm(
        llm, get_model_str(m, names),
        "What does this model predict for x0=2.0, x1=0.0, x2=0.0? "
        "Answer with just a single number.",
    )
    tol_a = max(abs(pred_a) * 0.2, 0.5)
    passed = False
    nums = re.findall(r"-?\d+\.?\d*", response or "")
    if nums:
        try: passed = abs(float(nums[0]) - pred_a) < tol_a
        except ValueError: pass
    return dict(test="discrim_predict_above_threshold", passed=passed,
                ground_truth=pred_a, response=response)


def discrim_test_predict_below_threshold(model, llm):
    """Predict a sample below the threshold (x0=-0.5 < threshold=1.0, true output ≈0.0)."""
    X, y = _threshold_data(threshold=1.0, n=600, seed=54)
    names = [f"x{i}" for i in range(3)]
    m = _safe_clone(model)
    m.fit(X, y)
    assert r2_score(y, m.predict(X)) > 0.5, "Model failed to fit"
    pred_b = round(float(m.predict(np.array([[-0.5, 0.0, 0.0]]))[0]), 2)
    response = ask_llm(
        llm, get_model_str(m, names),
        "What does this model predict for x0=-0.5, x1=0.0, x2=0.0? "
        "Answer with just a single number.",
    )
    tol_b = max(abs(pred_b) * 0.2, 0.5)
    passed = False
    nums = re.findall(r"-?\d+\.?\d*", response or "")
    if nums:
        try: passed = abs(float(nums[0]) - pred_b) < tol_b
        except ValueError: pass
    return dict(test="discrim_predict_below_threshold", passed=passed,
                ground_truth=pred_b, response=response)


def discrim_test_simulate_mixed_sign(model, llm):
    """Simulate prediction on 6 features with mixed positive/negative coefficients.

    Requires tracing through more terms with sign changes; interpretable additive
    models can still compute this from their representation while black-box models
    cannot.
    """
    X, y = _mixed_sign_six_feature_data()
    names = [f"x{i}" for i in range(6)]
    m = _safe_clone(model)
    m.fit(X, y)
    assert r2_score(y, m.predict(X)) > 0.5, "Model failed to fit"
    sample = np.array([[1.5, -1.0, 0.8, 2.0, -0.5, 1.2]])
    true_pred = float(m.predict(sample)[0])
    response = ask_llm(
        llm, get_model_str(m, names),
        "What does this model predict for x0=1.5, x1=-1.0, x2=0.8, x3=2.0, x4=-0.5, x5=1.2? "
        "Answer with just a single number.",
    )
    tol = max(abs(true_pred) * 0.2, 1.5)
    passed, llm_val = False, None
    for num_str in reversed(re.findall(r"-?\d+\.?\d*", response or "")):
        try:
            val = float(num_str)
            if abs(val - true_pred) < tol:
                llm_val = val; passed = True; break
        except ValueError: pass
    return dict(test="discrim_simulate_mixed_sign", passed=passed,
                ground_truth=round(true_pred, 3), response=response)


def discrim_test_simulate_double_threshold(model, llm):
    """Simulate prediction on data with two step thresholds on x0.

    y jumps at x0=0 and x0=1.5, creating three output levels. Interpretable
    models (shallow trees, piecewise representations) expose these steps directly;
    black-box models obscure them.
    """
    X, y = _double_threshold_data()
    names = [f"x{i}" for i in range(4)]
    m = _safe_clone(model)
    m.fit(X, y)
    assert r2_score(y, m.predict(X)) > 0.5, "Model failed to fit"
    sample = np.array([[0.8, 0.0, 0.0, 0.0]])
    true_pred = float(m.predict(sample)[0])
    response = ask_llm(
        llm, get_model_str(m, names),
        "What does this model predict for x0=0.8, x1=0.0, x2=0.0, x3=0.0? "
        "Answer with just a single number.",
    )
    tol = max(abs(true_pred) * 0.2, 0.6)
    passed = False
    nums = re.findall(r"-?\d+\.?\d*", response or "")
    if nums:
        try: passed = abs(float(nums[0]) - true_pred) < tol
        except ValueError: pass
    return dict(test="discrim_simulate_double_threshold", passed=passed,
                ground_truth=round(true_pred, 3), response=response)


def discrim_test_simulate_additive_nonlinear(model, llm):
    """Simulate prediction on additive nonlinear data: y = 3*max(0,x0) + 2*sin(x1) + x2.

    GAMs and piecewise-linear models can read partial effects directly.
    Linear models approximate but may be inaccurate at the nonlinear regions.
    Black-box models cannot trace through their representations.
    """
    X, y = _additive_nonlinear_data()
    names = [f"x{i}" for i in range(5)]
    m = _safe_clone(model)
    m.fit(X, y)
    assert r2_score(y, m.predict(X)) > 0.4, "Model failed to fit"
    sample = np.array([[1.5, 1.0, -0.5, 0.0, 0.0]])
    true_pred = float(m.predict(sample)[0])
    response = ask_llm(
        llm, get_model_str(m, names),
        "What does this model predict for x0=1.5, x1=1.0, x2=-0.5, x3=0.0, x4=0.0? "
        "Answer with just a single number.",
    )
    tol = max(abs(true_pred) * 0.2, 1.0)
    passed, llm_val = False, None
    for num_str in reversed(re.findall(r"-?\d+\.?\d*", response or "")):
        try:
            val = float(num_str)
            if abs(val - true_pred) < tol:
                llm_val = val; passed = True; break
        except ValueError: pass
    return dict(test="discrim_simulate_additive_nonlinear", passed=passed,
                ground_truth=round(true_pred, 3), response=response)


def discrim_test_simulate_interaction(model, llm):
    """Simulate prediction on data with a pairwise interaction: y = 3*x0 + 2*x1 + 1.5*x0*x1.

    Models that capture interactions (trees, FIGS, GBDT) may represent this;
    purely additive models (linear, GAM) will approximate but miss the interaction.
    Tests whether the LLM can trace through the model's representation to get the
    right answer on a sample where the interaction term matters.
    """
    X, y = _interaction_data()
    names = [f"x{i}" for i in range(4)]
    m = _safe_clone(model)
    m.fit(X, y)
    assert r2_score(y, m.predict(X)) > 0.4, "Model failed to fit"
    sample = np.array([[2.0, 1.5, 0.0, 0.0]])
    true_pred = float(m.predict(sample)[0])
    response = ask_llm(
        llm, get_model_str(m, names),
        "What does this model predict for x0=2.0, x1=1.5, x2=0.0, x3=0.0? "
        "Answer with just a single number.",
    )
    tol = max(abs(true_pred) * 0.2, 1.5)
    passed, llm_val = False, None
    for num_str in reversed(re.findall(r"-?\d+\.?\d*", response or "")):
        try:
            val = float(num_str)
            if abs(val - true_pred) < tol:
                llm_val = val; passed = True; break
        except ValueError: pass
    return dict(test="discrim_simulate_interaction", passed=passed,
                ground_truth=round(true_pred, 3), response=response)


# ---------------------------------------------------------------------------
# Simulatability tests — increasingly complex data
# ---------------------------------------------------------------------------

def simulatability_eight_features(model, llm):
    """Simulate on 8-feature linear data with mixed signs.

    More terms to trace through than basic tests. Linear models / shallow trees
    remain readable; GBDTs and MLPs do not.
    """
    X, y, _ = _eight_feature_mixed_data()
    names = [f"x{i}" for i in range(8)]
    m = _safe_clone(model); m.fit(X, y)
    sample = np.array([[1.2, -0.8, 0.5, 1.0, -0.3, 0.7, -1.5, 0.2]])
    true_pred = float(m.predict(sample)[0])
    response = ask_llm(
        llm, get_model_str(m, names),
        "What does this model predict for x0=1.2, x1=-0.8, x2=0.5, x3=1.0, "
        "x4=-0.3, x5=0.7, x6=-1.5, x7=0.2? Answer with just a single number.",
    )
    tol = max(abs(true_pred) * 0.15, 1.5)
    passed = False
    for num_str in reversed(re.findall(r"-?\d+\.?\d*", response or "")):
        try:
            if abs(float(num_str) - true_pred) < tol:
                passed = True; break
        except ValueError: pass
    return dict(test="simulatability_eight_features", passed=passed,
                ground_truth=round(true_pred, 3), response=response)


def simulatability_fifteen_features_sparse(model, llm):
    """Simulate on 15 features where only 3 matter.

    The model string is large (15 features), but an interpretable model reveals
    which features are zeroed out, making simulation feasible. Black-box models
    present all 15 features opaquely.
    """
    X, y, _ = _fifteen_feature_sparse_data()
    names = [f"x{i}" for i in range(15)]
    m = _safe_clone(model); m.fit(X, y)
    sample = np.zeros((1, 15))
    sample[0, 0] = 1.5    # active: coef 8.0
    sample[0, 5] = -1.0   # active: coef -4.0
    sample[0, 12] = 2.0   # active: coef 3.0
    sample[0, 3] = 0.7    # noise feature
    sample[0, 9] = -0.4   # noise feature
    true_pred = float(m.predict(sample)[0])
    feat_str = ", ".join(f"x{i}={sample[0,i]}" for i in range(15) if sample[0, i] != 0)
    response = ask_llm(
        llm, get_model_str(m, names),
        f"What does this model predict for the input where {feat_str} "
        f"and all other features are 0? Answer with just a single number.",
    )
    tol = max(abs(true_pred) * 0.15, 2.0)
    passed = False
    for num_str in reversed(re.findall(r"-?\d+\.?\d*", response or "")):
        try:
            if abs(float(num_str) - true_pred) < tol:
                passed = True; break
        except ValueError: pass
    return dict(test="simulatability_fifteen_features_sparse", passed=passed,
                ground_truth=round(true_pred, 3), response=response)


def simulatability_quadratic(model, llm):
    """Simulate on quadratic data: y = 3*x0^2 - 2*x1^2 + x2.

    Models that capture nonlinearity (trees, GAMs) will have piecewise
    representations the LLM can trace. Linear models approximate poorly but
    their string is still readable. GBDTs/MLPs are opaque.
    """
    X, y = _quadratic_data()
    names = [f"x{i}" for i in range(5)]
    m = _safe_clone(model); m.fit(X, y)
    sample = np.array([[1.5, -1.0, 0.5, 0.0, 0.0]])
    true_pred = float(m.predict(sample)[0])
    response = ask_llm(
        llm, get_model_str(m, names),
        "What does this model predict for x0=1.5, x1=-1.0, x2=0.5, x3=0.0, x4=0.0? "
        "Answer with just a single number.",
    )
    tol = max(abs(true_pred) * 0.2, 1.5)
    passed = False
    for num_str in reversed(re.findall(r"-?\d+\.?\d*", response or "")):
        try:
            if abs(float(num_str) - true_pred) < tol:
                passed = True; break
        except ValueError: pass
    return dict(test="simulatability_quadratic", passed=passed,
                ground_truth=round(true_pred, 3), response=response)


def simulatability_triple_interaction(model, llm):
    """Simulate on data with multi-way interactions: 2*x0*x1 + 3*x1*x2 + x0*x2*x3.

    Very hard for any model to represent readably. Shallow trees approximate
    but remain traceable. GBDTs and MLPs are intractable to trace.
    """
    X, y = _triple_interaction_data()
    names = [f"x{i}" for i in range(6)]
    m = _safe_clone(model); m.fit(X, y)
    sample = np.array([[1.0, -0.5, 1.5, 0.8, 0.0, 0.0]])
    true_pred = float(m.predict(sample)[0])
    response = ask_llm(
        llm, get_model_str(m, names),
        "What does this model predict for x0=1.0, x1=-0.5, x2=1.5, x3=0.8, x4=0.0, x5=0.0? "
        "Answer with just a single number.",
    )
    tol = max(abs(true_pred) * 0.25, 1.5)
    passed = False
    for num_str in reversed(re.findall(r"-?\d+\.?\d*", response or "")):
        try:
            if abs(float(num_str) - true_pred) < tol:
                passed = True; break
        except ValueError: pass
    return dict(test="simulatability_triple_interaction", passed=passed,
                ground_truth=round(true_pred, 3), response=response)


def simulatability_friedman1(model, llm):
    """Simulate on Friedman #1 data (10 features, sin/polynomial/linear mix).

    This is a classic hard regression benchmark. The data-generating process is
    highly nonlinear. Only models whose string representation allows step-by-step
    tracing (shallow trees, GAMs with partial-effect tables) can be simulated.
    GBDTs and MLPs have opaque representations at this complexity.
    """
    X, y = _friedman1_data()
    names = [f"x{i}" for i in range(10)]
    m = _safe_clone(model); m.fit(X, y)
    sample = np.array([[0.7, 0.3, 0.8, 0.5, 0.6, 0.1, 0.9, 0.2, 0.4, 0.5]])
    true_pred = float(m.predict(sample)[0])
    response = ask_llm(
        llm, get_model_str(m, names),
        "What does this model predict for x0=0.7, x1=0.3, x2=0.8, x3=0.5, x4=0.6, "
        "x5=0.1, x6=0.9, x7=0.2, x8=0.4, x9=0.5? "
        "Answer with just a single number.",
    )
    tol = max(abs(true_pred) * 0.2, 2.0)
    passed = False
    for num_str in reversed(re.findall(r"-?\d+\.?\d*", response or "")):
        try:
            if abs(float(num_str) - true_pred) < tol:
                passed = True; break
        except ValueError: pass
    return dict(test="simulatability_friedman1", passed=passed,
                ground_truth=round(true_pred, 3), response=response)


def simulatability_cascading_threshold(model, llm):
    """Simulate on cascading threshold data: if x0>0 then y~3*x1 else y~-2*x2.

    Shallow decision trees naturally represent this branching structure and can
    be traced. Linear models/GAMs approximate as additive effects (still readable).
    GBDTs with many trees and MLPs are opaque.
    """
    X, y = _cascading_threshold_data()
    names = [f"x{i}" for i in range(6)]
    m = _safe_clone(model); m.fit(X, y)
    # Test a sample in the x0>0 branch
    sample = np.array([[1.2, 0.8, -0.5, 0.3, 0.0, 0.0]])
    true_pred = float(m.predict(sample)[0])
    response = ask_llm(
        llm, get_model_str(m, names),
        "What does this model predict for x0=1.2, x1=0.8, x2=-0.5, x3=0.3, x4=0.0, x5=0.0? "
        "Answer with just a single number.",
    )
    tol = max(abs(true_pred) * 0.2, 1.0)
    passed = False
    for num_str in reversed(re.findall(r"-?\d+\.?\d*", response or "")):
        try:
            if abs(float(num_str) - true_pred) < tol:
                passed = True; break
        except ValueError: pass
    return dict(test="simulatability_cascading_threshold", passed=passed,
                ground_truth=round(true_pred, 3), response=response)


def simulatability_quadratic_counterfactual(model, llm):
    """Counterfactual on quadratic data: how does prediction change when x0 goes from 0 to 2?

    On y=3*x0^2-2*x1^2+x2, the change from x0=0→2 is large and nonlinear.
    Interpretable models expose this directly; black-box models cannot be traced.
    """
    X, y = _quadratic_data()
    names = [f"x{i}" for i in range(5)]
    m = _safe_clone(model); m.fit(X, y)
    base = np.array([[0.0, 0.5, 1.0, 0.0, 0.0]])
    changed = np.array([[2.0, 0.5, 1.0, 0.0, 0.0]])
    delta = float(m.predict(changed)[0]) - float(m.predict(base)[0])
    response = ask_llm(
        llm, get_model_str(m, names),
        "By how much does the prediction change when x0 goes from 0.0 to 2.0, "
        "with x1=0.5, x2=1.0, x3=0.0, x4=0.0 held fixed? "
        "Give just a number (positive if prediction increases).",
    )
    tol = max(abs(delta) * 0.2, 1.5)
    passed = False
    nums = re.findall(r"-?\d+\.?\d*", response or "")
    if nums:
        try:
            passed = abs(float(nums[0]) - delta) < tol
        except ValueError: pass
    return dict(test="simulatability_quadratic_counterfactual", passed=passed,
                ground_truth=round(delta, 3), response=response)


def simulatability_exponential_decay(model, llm):
    """Simulate on exponential + linear data: y = 5*exp(-x0) + 2*x1.

    Trees and GAMs approximate with piecewise steps that the LLM can trace.
    Linear models give a linear approximation (still readable). GBDTs/MLPs are opaque.
    """
    X, y = _exponential_decay_data()
    names = [f"x{i}" for i in range(4)]
    m = _safe_clone(model); m.fit(X, y)
    sample = np.array([[0.5, 1.0, 0.0, 0.0]])
    true_pred = float(m.predict(sample)[0])
    response = ask_llm(
        llm, get_model_str(m, names),
        "What does this model predict for x0=0.5, x1=1.0, x2=0.0, x3=0.0? "
        "Answer with just a single number.",
    )
    tol = max(abs(true_pred) * 0.2, 1.0)
    passed = False
    for num_str in reversed(re.findall(r"-?\d+\.?\d*", response or "")):
        try:
            if abs(float(num_str) - true_pred) < tol:
                passed = True; break
        except ValueError: pass
    return dict(test="simulatability_exponential_decay", passed=passed,
                ground_truth=round(true_pred, 3), response=response)


def simulatability_piecewise_three_segment(model, llm):
    """Simulate on piecewise-linear data with 3 segments on x0.

    Shallow trees naturally capture the breakpoints. Linear models give a single
    slope (readable but inaccurate). GBDTs/MLPs are opaque.
    """
    X, y = _piecewise_three_segment_data()
    names = [f"x{i}" for i in range(4)]
    m = _safe_clone(model); m.fit(X, y)
    sample = np.array([[0.5, 0.0, 0.0, 0.0]])  # middle segment, expected ~1.5
    true_pred = float(m.predict(sample)[0])
    response = ask_llm(
        llm, get_model_str(m, names),
        "What does this model predict for x0=0.5, x1=0.0, x2=0.0, x3=0.0? "
        "Answer with just a single number.",
    )
    tol = max(abs(true_pred) * 0.2, 0.8)
    passed = False
    for num_str in reversed(re.findall(r"-?\d+\.?\d*", response or "")):
        try:
            if abs(float(num_str) - true_pred) < tol:
                passed = True; break
        except ValueError: pass
    return dict(test="simulatability_piecewise_three_segment", passed=passed,
                ground_truth=round(true_pred, 3), response=response)


def simulatability_twenty_features_sparse(model, llm):
    """Simulate on 20 features where only 4 matter.

    Even larger than the 15-feature test. Interpretable models highlight which
    features have zero effect; black-box models present all 20 opaquely.
    """
    X, y, _ = _twenty_feature_sparse_data()
    names = [f"x{i}" for i in range(20)]
    m = _safe_clone(model); m.fit(X, y)
    sample = np.zeros((1, 20))
    sample[0, 2] = 1.5    # active: coef 6.0
    sample[0, 7] = -0.8   # active: coef -3.5
    sample[0, 11] = 1.0   # active: coef 2.0
    sample[0, 18] = -0.5  # active: coef -1.5
    sample[0, 4] = 0.3    # noise
    sample[0, 15] = -0.6  # noise
    true_pred = float(m.predict(sample)[0])
    feat_str = ", ".join(f"x{i}={sample[0,i]}" for i in range(20) if sample[0, i] != 0)
    response = ask_llm(
        llm, get_model_str(m, names),
        f"What does this model predict for the input where {feat_str} "
        f"and all other features are 0? Answer with just a single number.",
    )
    tol = max(abs(true_pred) * 0.15, 2.0)
    passed = False
    for num_str in reversed(re.findall(r"-?\d+\.?\d*", response or "")):
        try:
            if abs(float(num_str) - true_pred) < tol:
                passed = True; break
        except ValueError: pass
    return dict(test="simulatability_twenty_features_sparse", passed=passed,
                ground_truth=round(true_pred, 3), response=response)


def simulatability_sinusoidal(model, llm):
    """Simulate on sinusoidal data: y = 4*sin(x0) + 2*cos(x1) + x2.

    GAMs with partial-effect tables can be traced through. Trees approximate
    with piecewise steps. GBDTs/MLPs are opaque.
    """
    X, y = _sinusoidal_data()
    names = [f"x{i}" for i in range(5)]
    m = _safe_clone(model); m.fit(X, y)
    sample = np.array([[1.0, 0.5, -0.3, 0.0, 0.0]])
    true_pred = float(m.predict(sample)[0])
    response = ask_llm(
        llm, get_model_str(m, names),
        "What does this model predict for x0=1.0, x1=0.5, x2=-0.3, x3=0.0, x4=0.0? "
        "Answer with just a single number.",
    )
    tol = max(abs(true_pred) * 0.2, 1.0)
    passed = False
    for num_str in reversed(re.findall(r"-?\d+\.?\d*", response or "")):
        try:
            if abs(float(num_str) - true_pred) < tol:
                passed = True; break
        except ValueError: pass
    return dict(test="simulatability_sinusoidal", passed=passed,
                ground_truth=round(true_pred, 3), response=response)


def simulatability_abs_value(model, llm):
    """Simulate on V-shaped data: y = 3*|x0| - 2*|x1| + x2.

    Similar to hockey-stick but with symmetric V shapes. Trees capture the
    breakpoint at 0 naturally. Linear models approximate with a single slope.
    """
    X, y = _abs_value_data()
    names = [f"x{i}" for i in range(5)]
    m = _safe_clone(model); m.fit(X, y)
    sample = np.array([[-1.5, 0.8, 0.5, 0.0, 0.0]])
    true_pred = float(m.predict(sample)[0])
    response = ask_llm(
        llm, get_model_str(m, names),
        "What does this model predict for x0=-1.5, x1=0.8, x2=0.5, x3=0.0, x4=0.0? "
        "Answer with just a single number.",
    )
    tol = max(abs(true_pred) * 0.2, 1.0)
    passed = False
    for num_str in reversed(re.findall(r"-?\d+\.?\d*", response or "")):
        try:
            if abs(float(num_str) - true_pred) < tol:
                passed = True; break
        except ValueError: pass
    return dict(test="simulatability_abs_value", passed=passed,
                ground_truth=round(true_pred, 3), response=response)


def simulatability_twelve_features_all_active(model, llm):
    """Simulate on 12 features, all active with decreasing coefficients.

    Requires tracing through 12 terms. A serious stress test for readability.
    Linear models are still computable from their string; GBDTs/MLPs are not.
    """
    X, y, _ = _twelve_feature_all_active_data()
    names = [f"x{i}" for i in range(12)]
    m = _safe_clone(model); m.fit(X, y)
    sample = np.array([[1.0, -0.5, 0.8, 1.2, -0.3, 0.6, -1.0, 0.4, -0.2, 0.7, -0.8, 0.3]])
    true_pred = float(m.predict(sample)[0])
    response = ask_llm(
        llm, get_model_str(m, names),
        "What does this model predict for x0=1.0, x1=-0.5, x2=0.8, x3=1.2, "
        "x4=-0.3, x5=0.6, x6=-1.0, x7=0.4, x8=-0.2, x9=0.7, x10=-0.8, x11=0.3? "
        "Answer with just a single number.",
    )
    tol = max(abs(true_pred) * 0.15, 1.5)
    passed = False
    for num_str in reversed(re.findall(r"-?\d+\.?\d*", response or "")):
        try:
            if abs(float(num_str) - true_pred) < tol:
                passed = True; break
        except ValueError: pass
    return dict(test="simulatability_twelve_features_all_active", passed=passed,
                ground_truth=round(true_pred, 3), response=response)


def simulatability_nested_threshold(model, llm):
    """Simulate on nested-threshold data: if x0>0 and x1>0 → 5, elif x0>0 → 2, else → -1.

    Shallow decision trees represent this branching structure perfectly.
    Linear models approximate as additive effects. GBDTs/MLPs are opaque.
    """
    X, y = _nested_threshold_data()
    names = [f"x{i}" for i in range(5)]
    m = _safe_clone(model); m.fit(X, y)
    sample = np.array([[0.8, -0.5, 0.0, 0.0, 0.0]])  # x0>0 but x1<0, expected ~2
    true_pred = float(m.predict(sample)[0])
    response = ask_llm(
        llm, get_model_str(m, names),
        "What does this model predict for x0=0.8, x1=-0.5, x2=0.0, x3=0.0, x4=0.0? "
        "Answer with just a single number.",
    )
    tol = max(abs(true_pred) * 0.2, 0.8)
    passed = False
    for num_str in reversed(re.findall(r"-?\d+\.?\d*", response or "")):
        try:
            if abs(float(num_str) - true_pred) < tol:
                passed = True; break
        except ValueError: pass
    return dict(test="simulatability_nested_threshold", passed=passed,
                ground_truth=round(true_pred, 3), response=response)


SIMULATABILITY_TESTS = [
    simulatability_eight_features,
    simulatability_fifteen_features_sparse,
    simulatability_quadratic,
    simulatability_triple_interaction,
    simulatability_friedman1,
    simulatability_cascading_threshold,
    simulatability_quadratic_counterfactual,
    simulatability_exponential_decay,
    simulatability_piecewise_three_segment,
    simulatability_twenty_features_sparse,
    simulatability_sinusoidal,
    simulatability_abs_value,
    simulatability_twelve_features_all_active,
    simulatability_nested_threshold,
]


DISCRIM_TESTS = [
    discrim_test_simulate_all_active,
    discrim_test_compactness,
    discrim_test_dominant_feature_sample,
    discrim_test_unit_sensitivity,
    discrim_test_predict_above_threshold,
    discrim_test_predict_below_threshold,
    discrim_test_simulate_mixed_sign,
    discrim_test_simulate_double_threshold,
    discrim_test_simulate_additive_nonlinear,
    discrim_test_simulate_interaction,
]


# ---------------------------------------------------------------------------
# Test lists
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_most_important_feature,
    test_point_prediction,
    test_direction_of_change,
    test_feature_ranking,
    test_threshold_identification,
    test_irrelevant_features,
    test_sign_of_effect,
    test_counterfactual_prediction,
]

HARD_TESTS = [
    hard_test_all_features_active,
    hard_test_pairwise_anti_intuitive,
    hard_test_quantitative_sensitivity,
    hard_test_mixed_sign_goes_negative,
    hard_test_two_feature_perturbation,
]

INSIGHT_TESTS = [
    insight_simulatability,
    insight_sparse_feature_set,
    insight_nonlinear_threshold,
    insight_nonlinear_direction,
    insight_counterfactual_target,
    insight_decision_region,
]

_ALL_TEST_FNS = {fn.__name__: fn for fn in ALL_TESTS + HARD_TESTS + INSIGHT_TESTS + DISCRIM_TESTS + SIMULATABILITY_TESTS}


# ---------------------------------------------------------------------------
# Cached runner
# ---------------------------------------------------------------------------

@_memory.cache
def _run_one_test(model_name, test_fn_name, model, checkpoint=None):
    llm = imodelsx.llm.get_llm(checkpoint or CHECKPOINT)
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


def run_all_interp_tests(model_defs, checkpoint=None):
    """Run standard + hard + insight tests on all models, with per-test caching.

    Args:
        model_defs: list of (name, regressor) tuples
        checkpoint: LLM checkpoint to use (default: CHECKPOINT global, i.e. gpt-4o)
    Returns:
        list of result dicts with keys: model, test, passed, ground_truth, response
    """
    from joblib import Parallel, delayed

    all_test_fns = ALL_TESTS + HARD_TESTS + INSIGHT_TESTS + DISCRIM_TESTS + SIMULATABILITY_TESTS
    tasks = [(name, reg, test_fn) for name, reg in model_defs for test_fn in all_test_fns]

    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_run_one_test)(name, test_fn.__name__, reg, checkpoint=checkpoint)
        for name, reg, test_fn in tasks
    )

    # Print results grouped by model and suite
    for name, reg in model_defs:
        print(f"\n{'='*60}\n  Model: {name}\n{'='*60}")
        for test_list, label in [(ALL_TESTS, "standard"), (HARD_TESTS, "hard"), (INSIGHT_TESTS, "insight"), (DISCRIM_TESTS, "discrim"), (SIMULATABILITY_TESTS, "simulatability")]:
            print(f"\n  [{label}]")
            suite_results = [r for r in results if r["model"] == name and r["test"] in {t.__name__ for t in test_list}]
            for result in suite_results:
                status = "PASS" if result["passed"] else "FAIL"
                resp = (result.get("response") or "")[:80].replace("\n", " ")
                print(f"  [{status}] {result['test']}")
                print(f"         ground_truth : {result.get('ground_truth', '')}")
                print(f"         llm_response : {resp}")
            print(f"\n  → {sum(r['passed'] for r in suite_results)}/{len(test_list)} passed")

    return results
