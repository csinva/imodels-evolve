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
  Insight (5): simulatability, sparse feature set, nonlinear shape,
    counterfactual target, decision region

Exports:
  ALL_TESTS, HARD_TESTS, INSIGHT_TESTS  — lists of test functions
  run_all_interp_tests(model_defs)       — cached runner, returns list of result dicts
"""

import os
import re
import sys
from copy import deepcopy

import numpy as np
from joblib import Memory
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor, export_text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import imodelsx.llm

try:
    from pygam import LinearGAM
    _HAS_PYGAM = True
except ImportError:
    _HAS_PYGAM = False

try:
    import imodels as _imodels
    _HAS_IMODELS = True
except ImportError:
    _HAS_IMODELS = False

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
    """Return a human-readable string for a fitted sklearn regressor."""
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

    if _HAS_IMODELS:
        from imodels import FIGSRegressor, FIGSRegressorCV, RuleFitRegressor, HSTreeRegressor, HSTreeRegressorCV, TreeGAMRegressor
        if isinstance(model, FIGSRegressorCV):
            return str(model.figs)
        if isinstance(model, HSTreeRegressorCV):
            return str(model)
        if isinstance(model, (FIGSRegressor, RuleFitRegressor, HSTreeRegressor)):
            return str(model)
        if isinstance(model, TreeGAMRegressor):
            return _tree_gam_str(model, feature_names)

    if _HAS_PYGAM:
        from pygam import LinearGAM
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
    return llm(prompt, max_completion_tokens=max_tokens)


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
    true_dir = "increase" if float(m.predict(x1)[0]) > float(m.predict(np.zeros((1, 4)))[0]) else "decrease"
    response = ask_llm(llm, get_model_str(m, names),
                       "If we change x0 from 0.0 to 1.0 (all other features stay at 0.0), "
                       "will the prediction increase or decrease? Answer with just 'increase' or 'decrease'.")
    return dict(test="direction_of_change", passed=bool(response and true_dir in response.lower()),
                ground_truth=true_dir, response=response)

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
    response = ask_llm(llm, get_model_str(m, names),
                       "Does increasing x1 tend to increase or decrease the predicted value? "
                       "Answer with just 'increase' or 'decrease'.")
    return dict(test="sign_of_effect", passed=bool(response and "decrease" in response.lower()),
                ground_truth="decrease (x1 has coefficient -5.0)", response=response)

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
    ground_truth = "B" if pred_b > pred_a else "A"
    response = ask_llm(llm, get_model_str(m, names),
                       "Sample A has features: x0=2.0, x1=0.1, x2=0.0, x3=0.0, x4=0.0\n"
                       "Sample B has features: x0=0.5, x1=3.3, x2=0.0, x3=0.0, x4=0.0\n"
                       "Which sample does the model predict a higher value for? Answer with just 'A' or 'B'.")
    return dict(test="hard_pairwise_anti_intuitive",
                passed=bool(response and ground_truth in response.upper()[:5]),
                ground_truth=f"{ground_truth} (pred_A={pred_a:.2f}, pred_B={pred_b:.2f})",
                response=response)

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
                       "Walk through the model step by step to predict for "
                       "x0=1.0, x1=2.0, x2=0.5, x3=-0.5. "
                       "Show each step of your reasoning, citing specific values from the model. "
                       "End your answer with the final predicted value on its own line.",
                       max_tokens=350)
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

def insight_nonlinear_shape(model, llm):
    X, y = _hockey_stick_data()
    names = [f"x{i}" for i in range(3)]
    m = _safe_clone(model); m.fit(X, y)
    r2 = r2_score(y, m.predict(X))
    response = ask_llm(llm, get_model_str(m, names),
                       "Describe the relationship between x0 and the predicted output "
                       "(holding x1=0 and x2=0). Specifically answer: "
                       "(1) Is there a threshold value of x0 below which x0 has little or no effect? "
                       "If so, what is it approximately? "
                       "(2) What happens to the prediction as x0 increases above that threshold?",
                       max_tokens=250)
    threshold_ok, increasing_ok = False, False
    if response:
        r = response.lower()
        nums = re.findall(r"-?\d+\.?\d*", r)
        threshold_ok = (any(abs(float(n)) < 0.7 for n in nums) or
                        any(w in r for w in ["zero", "negative", "below zero", "0.0", "flat",
                                             "constant", "no effect", "hockey", "piecewise", "relu"]))
        increasing_ok = any(w in r for w in ["increase", "linear", "positive slope", "grows",
                                              "rises", "higher", "greater", "upward", "positively"])
    return dict(test="insight_nonlinear_shape", passed=threshold_ok and increasing_ok and r2 > 0.5,
                ground_truth="flat for x0<0, linearly increasing for x0>0", response=response)

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
    insight_nonlinear_shape,
    insight_counterfactual_target,
    insight_decision_region,
]

_ALL_TEST_FNS = {fn.__name__: fn for fn in ALL_TESTS + HARD_TESTS + INSIGHT_TESTS}


# ---------------------------------------------------------------------------
# Cached runner
# ---------------------------------------------------------------------------

@_memory.cache
def _run_one_test(model_name, test_fn_name, model):
    llm = imodelsx.llm.get_llm(CHECKPOINT)
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


def run_all_interp_tests(model_defs):
    """Run standard + hard + insight tests on all models, with per-test caching.

    Args:
        model_defs: list of (name, regressor) tuples
    Returns:
        list of result dicts with keys: model, test, passed, ground_truth, response
    """
    all_results = []
    for name, reg in model_defs:
        print(f"\n{'='*60}\n  Model: {name}\n{'='*60}")
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
            print(f"\n  → {sum(r['passed'] for r in suite_results)}/{len(test_list)} passed")
            all_results.extend(suite_results)
    return all_results
