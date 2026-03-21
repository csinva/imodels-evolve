"""
Interpretability tests for sklearn-style regressors.

Uses an LLM (gpt-4o) to probe whether a model's string representation
conveys meaningful, usable information about the model's behavior.

Tests:
  1. Most important feature identification
  2. Point prediction (no code execution)
  3. Direction of change when perturbing the key feature
  4. Feature importance ranking
  5. Decision threshold identification (for tree-based models)
  6. Irrelevant feature detection
  7. Sign of a feature's effect (positive vs negative)
  8. Counterfactual prediction (given one prediction, infer another)

Run with: uv run interpretability.py
"""

import re
import sys
from copy import deepcopy

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor, export_text

import importlib.util as _ilu
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

CHECKPOINT = "gpt-4o"


def _safe_clone(model):
    """Clone a model; fall back to deepcopy for non-sklearn models (e.g. pygam)."""
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
        lines.append(
            export_text(model.estimators_[0, 0], feature_names=feature_names, max_depth=3)
        )
        return "\n".join(lines)

    if isinstance(model, RandomForestRegressor):
        importances = model.feature_importances_
        names = feature_names or [f"x{i}" for i in range(len(importances))]
        order = np.argsort(importances)[::-1]
        lines = ["Random Forest Regressor — Feature Importances (higher = more important):"]
        for i in order:
            lines.append(f"  {names[i]}: {importances[i]:.4f}")
        lines.append("\nFirst estimator tree (depth ≤ 3):")
        lines.append(
            export_text(model.estimators_[0], feature_names=feature_names, max_depth=3)
        )
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

    if isinstance(model, MLPRegressor):
        names = feature_names or [f"x{i}" for i in range(model.n_features_in_)]
        hidden = (
            model.hidden_layer_sizes
            if isinstance(model.hidden_layer_sizes, (list, tuple))
            else (model.hidden_layer_sizes,)
        )
        arch = f"{model.n_features_in_} → {' → '.join(str(h) for h in hidden)} → 1"
        lines = [f"MLP Regressor (architecture: {arch}, activation: {model.activation})"]
        W0 = model.coefs_[0]  # shape: (n_features, n_hidden_1)
        lines.append(
            f"\nFirst-layer weight matrix ({W0.shape[0]} features × {W0.shape[1]} neurons):"
        )
        lines.append("(Each row shows the weights from one input feature to all hidden neurons.)")
        for fi, name in enumerate(names):
            w = W0[fi]
            l2 = np.linalg.norm(w)
            if len(w) <= 8:
                w_str = "[" + ", ".join(f"{v:.3f}" for v in w) + "]"
            else:
                w_str = f"mean={w.mean():.3f}, std={w.std():.3f}"
            lines.append(f"  {name}: {w_str}  (L2={l2:.3f})")
        return "\n".join(lines)

    # imodels: FIGSRegressor, RuleFitRegressor, HSTreeRegressor — use built-in __str__
    if _HAS_IMODELS:
        from imodels import FIGSRegressor, RuleFitRegressor, HSTreeRegressor, TreeGAMRegressor
        if isinstance(model, (FIGSRegressor, RuleFitRegressor, HSTreeRegressor)):
            return str(model)
        if isinstance(model, TreeGAMRegressor):
            return _tree_gam_str(model, feature_names)

    # GAM (pygam.LinearGAM)
    if _HAS_PYGAM:
        from pygam import LinearGAM
        if isinstance(model, LinearGAM):
            # GAM doesn't set n_features_in_, derive from statistics after fit
            if feature_names is None and hasattr(model, "statistics_"):
                n_f = model.statistics_.get("m_features", 0)
                feature_names = [f"x{i}" for i in range(n_f)]
            return _gam_str(model, feature_names)

    return str(model)


def _gam_str(model, feature_names=None):
    """Human-readable representation of a fitted pygam LinearGAM.

    Shows each feature's partial effect (shape function) at evenly spaced points.
    GAMs are additive — f(x) = intercept + f0(x0) + f1(x1) + ... — so each
    feature's contribution can be read independently, making them highly modular.
    """
    from pygam import LinearGAM
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
            # describe shape
            if pdp[-1] > pdp[0] + 0.5:
                shape = "increasing"
            elif pdp[-1] < pdp[0] - 0.5:
                shape = "decreasing"
            elif max(pdp) - min(pdp) < 0.3:
                shape = "flat/negligible"
            else:
                shape = "non-monotone"
            lines.append(f"    (shape: {shape})")
        except Exception:
            lines.append(f"\n  {name}: (partial effect not available)")
    return "\n".join(lines)


def _tree_gam_str(model, feature_names=None):
    """Human-readable representation of a fitted TreeGAMRegressor.

    TreeGAM uses cyclic boosting: one tree per feature, each tree fits only
    that feature's residuals. The prediction is:
      y = bias + tree0(x0) + tree1(x1) + ...
    This is highly modular — each feature's effect is independently readable.
    """
    from sklearn.tree import export_text
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
        # Identify which feature this tree splits on
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
    prompt = (
        f"Here is a trained regression model:\n\n"
        f"{model_str}\n\n"
        f"{question}"
    )
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
    """y = +5*x0 - 5*x1 + noise"""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 4)
    y = 5.0 * X[:, 0] - 5.0 * X[:, 1] + rng.randn(n) * 0.5
    return X, y


# ---------------------------------------------------------------------------
# Individual tests
# ---------------------------------------------------------------------------


def test_most_important_feature(model, llm):
    """Test 1 — Identify the single most important feature."""
    X, y = _single_feature_data(n_features=5, true_feature=0, coef=10.0)
    names = [f"x{i}" for i in range(5)]
    m = _safe_clone(model)
    m.fit(X, y)
    r2 = r2_score(y, m.predict(X))
    assert r2 > 0.5, f"Model R²={r2:.3f} too low — test invalid"

    mstr = get_model_str(m, names)
    q = (
        "Which single feature is most important for predicting the output? "
        "Answer with just the feature name (e.g., 'x0', 'x3')."
    )
    response = ask_llm(llm, mstr, q)
    passed = bool(response and "x0" in response.lower())
    return dict(test="most_important_feature", passed=passed, r2=r2,
                ground_truth="x0", response=response)


def test_point_prediction(model, llm):
    """Test 2 — Predict the output for a specific input without running code."""
    X, y = _single_feature_data(n_features=3, true_feature=0, coef=5.0)
    names = [f"x{i}" for i in range(3)]
    m = _safe_clone(model)
    m.fit(X, y)

    x_test = np.array([[2.0, 0.0, 0.0]])
    true_pred = float(m.predict(x_test)[0])
    mstr = get_model_str(m, names)

    q = (
        f"What does this model predict for the input x0=2.0, x1=0.0, x2=0.0? "
        f"Answer with just a single number (e.g., '10.5')."
    )
    response = ask_llm(llm, mstr, q)

    llm_val = None
    passed = False
    nums = re.findall(r"-?\d+\.?\d*", response or "")
    if nums:
        try:
            llm_val = float(nums[0])
            tol = max(abs(true_pred) * 0.25, 1.5)
            passed = abs(llm_val - true_pred) < tol
        except ValueError:
            pass

    return dict(test="point_prediction", passed=passed,
                ground_truth=round(true_pred, 3), llm_guess=llm_val, response=response)


def test_direction_of_change(model, llm):
    """Test 3 — Predict whether the output increases or decreases when x0 goes 0→1."""
    X, y = _single_feature_data(n_features=4, true_feature=0, coef=8.0)
    names = [f"x{i}" for i in range(4)]
    m = _safe_clone(model)
    m.fit(X, y)

    pred_0 = float(m.predict(np.zeros((1, 4)))[0])
    x1 = np.zeros((1, 4)); x1[0, 0] = 1.0
    pred_1 = float(m.predict(x1)[0])
    true_dir = "increase" if pred_1 > pred_0 else "decrease"

    mstr = get_model_str(m, names)
    q = (
        "If we change x0 from 0.0 to 1.0 (all other features stay at 0.0), "
        "will the prediction increase or decrease? "
        "Answer with just 'increase' or 'decrease'."
    )
    response = ask_llm(llm, mstr, q)
    passed = bool(response and true_dir in response.lower())

    return dict(test="direction_of_change", passed=passed,
                ground_truth=true_dir, pred_at_0=round(pred_0, 3),
                pred_at_1=round(pred_1, 3), response=response)


def test_feature_ranking(model, llm):
    """Test 4 — Rank the top 3 features by importance (ground truth: x0 > x1 > x2)."""
    coefs = [5.0, 3.0, 1.5, 0.0, 0.0]
    X, y = _multi_feature_data(coefs)
    names = [f"x{i}" for i in range(5)]
    m = _safe_clone(model)
    m.fit(X, y)

    mstr = get_model_str(m, names)
    q = (
        "Rank the 3 most important features from most to least important. "
        "Answer with just the feature names separated by commas (e.g., 'x2, x0, x4')."
    )
    response = ask_llm(llm, mstr, q)

    # Pass if x0 is mentioned before x1, and x1 before x2
    passed = False
    if response:
        r = response.lower()
        pos = {f"x{i}": r.find(f"x{i}") for i in range(5)}
        passed = (
            pos["x0"] != -1
            and pos["x1"] != -1
            and pos["x0"] < pos["x1"]
        )

    return dict(test="feature_ranking", passed=passed,
                ground_truth="x0, x1, x2", response=response)


def test_threshold_identification(model, llm):
    """Test 5 — Identify the threshold for x0 that separates low/high predictions."""
    threshold = 0.5
    X, y = _threshold_data(threshold=threshold)
    names = [f"x{i}" for i in range(3)]
    m = _safe_clone(model)
    m.fit(X, y)

    mstr = get_model_str(m, names)
    q = (
        "What approximate threshold value for x0 separates low predictions from "
        "high predictions? Answer with just a number."
    )
    response = ask_llm(llm, mstr, q)

    llm_threshold = None
    passed = False
    nums = re.findall(r"-?\d+\.?\d*", response or "")
    if nums:
        try:
            llm_threshold = float(nums[0])
            passed = abs(llm_threshold - threshold) < 0.35
        except ValueError:
            pass

    return dict(test="threshold_identification", passed=passed,
                ground_truth=threshold, llm_threshold=llm_threshold, response=response)


def test_irrelevant_features(model, llm):
    """Test 6 — Identify features that have little/no effect on the prediction."""
    X, y = _single_feature_data(n_features=5, true_feature=0, coef=10.0, seed=4)
    names = [f"x{i}" for i in range(5)]
    m = _safe_clone(model)
    m.fit(X, y)

    mstr = get_model_str(m, names)
    q = (
        "Which features appear to have little or no effect on the prediction? "
        "List all such feature names, comma-separated."
    )
    response = ask_llm(llm, mstr, q)

    if response:
        found = [f"x{i}" in response.lower() for i in range(1, 5)]
        passed = sum(found) >= 2  # at least 2 of 4 irrelevant features identified
    else:
        passed = False

    return dict(test="irrelevant_features", passed=passed,
                ground_truth="x1, x2, x3, x4 are irrelevant", response=response)


def test_sign_of_effect(model, llm):
    """Test 7 — Predict the sign of x1's effect (negative: y decreases when x1 increases)."""
    X, y = _signed_data()
    names = [f"x{i}" for i in range(4)]
    m = _safe_clone(model)
    m.fit(X, y)

    mstr = get_model_str(m, names)
    q = (
        "Does increasing x1 tend to increase or decrease the predicted value? "
        "Answer with just 'increase' or 'decrease'."
    )
    response = ask_llm(llm, mstr, q)
    passed = bool(response and "decrease" in response.lower())

    return dict(test="sign_of_effect", passed=passed,
                ground_truth="decrease (x1 has coefficient -5.0)", response=response)


def test_counterfactual_prediction(model, llm):
    """Test 8 — Given prediction at x0=1, infer prediction at x0=3 (Δx0=+2)."""
    X, y = _single_feature_data(n_features=3, true_feature=0, coef=4.0, seed=5)
    names = [f"x{i}" for i in range(3)]
    m = _safe_clone(model)
    m.fit(X, y)

    x_a = np.array([[1.0, 0.0, 0.0]])
    x_b = np.array([[3.0, 0.0, 0.0]])
    pred_a = float(m.predict(x_a)[0])
    pred_b = float(m.predict(x_b)[0])

    mstr = get_model_str(m, names)
    q = (
        f"The model predicts {pred_a:.2f} for the input x0=1.0, x1=0.0, x2=0.0. "
        f"Using the model shown above (without running any code), "
        f"what would it predict for x0=3.0, x1=0.0, x2=0.0? "
        f"Answer with just a number."
    )
    response = ask_llm(llm, mstr, q)

    llm_pred = None
    passed = False
    nums = re.findall(r"-?\d+\.?\d*", response or "")
    if nums:
        try:
            llm_pred = float(nums[0])
            tol = max(abs(pred_b) * 0.25, 1.5)
            passed = abs(llm_pred - pred_b) < tol
        except ValueError:
            pass

    return dict(test="counterfactual_prediction", passed=passed,
                ground_truth=round(pred_b, 3), llm_pred=llm_pred, response=response)


# ---------------------------------------------------------------------------
# Hard tests — designed to stress black-box models (RF, MLP)
#
# Rationale:
#   OLS has an explicit equation → LLM can always plug in numbers.
#   DT has explicit rules → LLM can trace a single tree.
#   RF requires averaging 50 trees → practically impossible without code.
#   MLP requires propagating through non-linear layers → impossible without code.
#
# These tests require actual arithmetic, not just pattern-matching on importances.
# ---------------------------------------------------------------------------


def hard_test_all_features_active(model, llm):
    """Hard A — Predict when all 3 features are non-zero and non-round.

    Requires computing 3*1.7 + 2*0.8 + (-0.5) for OLS, or tracing a 3-feature
    tree for DT.  RF must average 50 trees; MLP must propagate layers.
    """
    coefs = [3.0, 2.0, 1.0]
    X, y = _multi_feature_data(coefs, n=600, seed=10)
    names = [f"x{i}" for i in range(3)]
    m = _safe_clone(model)
    m.fit(X, y)
    r2 = r2_score(y, m.predict(X))
    assert r2 > 0.5, f"R²={r2:.3f} too low"

    x_test = np.array([[1.7, 0.8, -0.5]])
    true_pred = float(m.predict(x_test)[0])
    mstr = get_model_str(m, names)

    q = (
        "What does this model predict for the input x0=1.7, x1=0.8, x2=-0.5? "
        "All three features are active. Answer with just a number."
    )
    response = ask_llm(llm, mstr, q)

    llm_val = None
    passed = False
    nums = re.findall(r"-?\d+\.?\d*", response or "")
    if nums:
        try:
            llm_val = float(nums[0])
            tol = max(abs(true_pred) * 0.15, 1.0)
            passed = abs(llm_val - true_pred) < tol
        except ValueError:
            pass

    return dict(test="hard_all_features_active", passed=passed,
                ground_truth=round(true_pred, 3), llm_guess=llm_val,
                r2=r2, response=response)


def hard_test_pairwise_anti_intuitive(model, llm):
    """Hard B — Which sample has a higher prediction when naive importance-ranking
    gives the WRONG answer?

    y = 5*x0 + 3*x1.  x0 has the larger coefficient, so a naive LLM might say
    'A wins because x0 is more important and A has x0=2.0 > B's x0=0.5'.
    But B has x1=3.3, so pred_B = 5*0.5 + 3*3.3 = 12.4 > pred_A = 5*2.0 + 3*0.1 = 10.3.
    Correct answer requires computing both predictions, not just comparing x0.
    """
    coefs = [5.0, 3.0, 0.0, 0.0, 0.0]
    X, y = _multi_feature_data(coefs, n=600, seed=11)
    names = [f"x{i}" for i in range(5)]
    m = _safe_clone(model)
    m.fit(X, y)

    x_a = np.array([[2.0, 0.1, 0.0, 0.0, 0.0]])
    x_b = np.array([[0.5, 3.3, 0.0, 0.0, 0.0]])
    pred_a = float(m.predict(x_a)[0])
    pred_b = float(m.predict(x_b)[0])
    ground_truth = "B" if pred_b > pred_a else "A"

    mstr = get_model_str(m, names)
    q = (
        f"Sample A has features: x0=2.0, x1=0.1, x2=0.0, x3=0.0, x4=0.0\n"
        f"Sample B has features: x0=0.5, x1=3.3, x2=0.0, x3=0.0, x4=0.0\n"
        f"Which sample does the model predict a higher value for? "
        f"Answer with just 'A' or 'B'."
    )
    response = ask_llm(llm, mstr, q)
    passed = bool(response and ground_truth in response.upper()[:5])

    return dict(test="hard_pairwise_anti_intuitive", passed=passed,
                ground_truth=f"{ground_truth} (pred_A={pred_a:.2f}, pred_B={pred_b:.2f})",
                response=response)


def hard_test_quantitative_sensitivity(model, llm):
    """Hard C — By exactly how much does the output change when x0 goes from 0.5 to 2.5?

    For OLS with coef ≈ 4, the answer is 4*(2.5-0.5) = 8.  For DT the LLM must
    find the leaf values at both points and subtract.  RF must average the delta
    over 50 trees.  MLP must propagate both inputs.
    """
    X, y = _single_feature_data(n_features=3, true_feature=0, coef=4.0, seed=12)
    names = [f"x{i}" for i in range(3)]
    m = _safe_clone(model)
    m.fit(X, y)

    x_lo = np.array([[0.5, 0.0, 0.0]])
    x_hi = np.array([[2.5, 0.0, 0.0]])
    delta = float(m.predict(x_hi)[0]) - float(m.predict(x_lo)[0])

    mstr = get_model_str(m, names)
    q = (
        "By how much does the model's prediction change when x0 increases from 0.5 "
        "to 2.5 (keeping x1=0, x2=0)? Answer with just a number (the difference)."
    )
    response = ask_llm(llm, mstr, q)

    llm_val = None
    passed = False
    nums = re.findall(r"-?\d+\.?\d*", response or "")
    if nums:
        try:
            llm_val = float(nums[0])
            tol = max(abs(delta) * 0.15, 1.0)
            passed = abs(llm_val - delta) < tol
        except ValueError:
            pass

    return dict(test="hard_quantitative_sensitivity", passed=passed,
                ground_truth=round(delta, 3), llm_guess=llm_val, response=response)


def hard_test_mixed_sign_goes_negative(model, llm):
    """Hard D — Predict an output that is negative even though all inputs are positive.

    y = 3*x0 - 2*x1 + x2.  At x0=1.0, x1=2.5, x2=1.0 the true value is
    3 - 5 + 1 = -1, which is negative despite all inputs being positive.
    Naive 'positive inputs → positive output' reasoning fails.
    """
    coefs = [3.0, -2.0, 1.0]
    X, y = _multi_feature_data(coefs, n=600, seed=13)
    names = [f"x{i}" for i in range(3)]
    m = _safe_clone(model)
    m.fit(X, y)
    r2 = r2_score(y, m.predict(X))
    assert r2 > 0.5, f"R²={r2:.3f} too low"

    x_test = np.array([[1.0, 2.5, 1.0]])
    true_pred = float(m.predict(x_test)[0])
    mstr = get_model_str(m, names)

    q = (
        "What does this model predict for x0=1.0, x1=2.5, x2=1.0? "
        "Answer with just a number (can be negative)."
    )
    response = ask_llm(llm, mstr, q)

    llm_val = None
    passed = False
    nums = re.findall(r"-?\d+\.?\d*", response or "")
    if nums:
        try:
            llm_val = float(nums[0])
            tol = max(abs(true_pred) * 0.2, 1.0)
            passed = abs(llm_val - true_pred) < tol
        except ValueError:
            pass

    return dict(test="hard_mixed_sign_goes_negative", passed=passed,
                ground_truth=round(true_pred, 3), llm_guess=llm_val,
                r2=r2, response=response)


def hard_test_two_feature_perturbation(model, llm):
    """Hard E — Predict after simultaneously changing two features.

    Given the base prediction at (0,0,...), what is the prediction when BOTH
    x0 and x1 change to non-trivial values?  Requires computing the combined
    effect; for OLS this is additive arithmetic; for DT the LLM must re-trace;
    for RF/MLP it requires running 50 trees / propagating layers.
    """
    coefs = [3.0, 2.0, 0.0, 0.0]
    X, y = _multi_feature_data(coefs, n=500, seed=14)
    names = [f"x{i}" for i in range(4)]
    m = _safe_clone(model)
    m.fit(X, y)

    x_base = np.zeros((1, 4))
    x_new = np.array([[2.0, 1.5, 0.0, 0.0]])
    pred_base = float(m.predict(x_base)[0])
    pred_new = float(m.predict(x_new)[0])

    mstr = get_model_str(m, names)
    q = (
        f"The model predicts {pred_base:.2f} for the input x0=0, x1=0, x2=0, x3=0. "
        f"What does it predict when both x0 changes to 2.0 AND x1 changes to 1.5 "
        f"(x2 and x3 stay at 0)? Answer with just a number."
    )
    response = ask_llm(llm, mstr, q)

    llm_val = None
    passed = False
    nums = re.findall(r"-?\d+\.?\d*", response or "")
    if nums:
        try:
            llm_val = float(nums[0])
            tol = max(abs(pred_new) * 0.15, 1.0)
            passed = abs(llm_val - pred_new) < tol
        except ValueError:
            pass

    return dict(test="hard_two_feature_perturbation", passed=passed,
                ground_truth=round(pred_new, 3), llm_guess=llm_val,
                pred_base=round(pred_base, 3), response=response)


# ---------------------------------------------------------------------------
# Insight tests — grounded in the interpretability literature
#
# Murdoch et al. (2019) PDR framework key properties:
#   • Simulatability: can a human manually trace the full decision process?
#   • Sparsity: fewer active features → easier to understand
#   • Modularity: can each feature's effect be understood independently? (GAMs)
#
# Kaur et al. (2020) practitioner expectations (from Fig. 2):
#   • Counterfactual what-ifs (~23% of practitioners expect this)
#   • Decision regions / global feature effects (~60-68% expect this)
#   • Detecting feature correlations / data issues
#
# These tests reward DEGREES of interpretability:
#   GAM ≈ sparse DT > sparse OLS > deep DT > dense OLS > RF ≈ GBM > MLP
# ---------------------------------------------------------------------------


def _hockey_stick_data(n=700, seed=30):
    """y = 3*max(0, x0) + noise — flat for x0 < 0, linear for x0 > 0."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 3)
    y = 3.0 * np.maximum(0.0, X[:, 0]) + 0.2 * rng.randn(n)
    return X, y


def _sparse_ten_feature_data(n=600, seed=31):
    """y = 5*x0 + 3*x1 + noise — 10 features, only two matter."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 10)
    y = 5.0 * X[:, 0] + 3.0 * X[:, 1] + 0.3 * rng.randn(n)
    return X, y


def insight_simulatability(model, llm):
    """Insight A — Step-by-step reasoning trace.

    Murdoch et al. (2019): 'A model is simulatable if a human is able to
    internally simulate and reason about its entire decision-making process.'
    This property is strong for OLS (plug into equation) and shallow DT
    (follow rules), weak for RF (50 trees), and absent for MLP.

    Graded by correctness of the final answer AND whether reasoning
    explicitly cites model values (e.g., '5 × 1.0 = 5.0').
    """
    coefs = [5.0, 3.0, 0.0, 0.0]
    X, y = _multi_feature_data(coefs, n=500, seed=20)
    names = [f"x{i}" for i in range(4)]
    m = _safe_clone(model)
    m.fit(X, y)

    x_test = np.array([[1.0, 2.0, 0.5, -0.5]])
    true_pred = float(m.predict(x_test)[0])
    mstr = get_model_str(m, names)

    q = (
        "Walk through the model step by step to predict for "
        "x0=1.0, x1=2.0, x2=0.5, x3=-0.5. "
        "Show each step of your reasoning, citing specific values from the model. "
        "End your answer with the final predicted value on its own line."
    )
    response = ask_llm(llm, mstr, q, max_tokens=350)

    # Search all numbers in response; prefer last occurrence within tolerance
    nums = re.findall(r"-?\d+\.?\d*", response or "")
    tol = max(abs(true_pred) * 0.15, 1.0)
    passed = False
    llm_val = None
    for num_str in reversed(nums):  # last match wins (final answer typically at end)
        try:
            val = float(num_str)
            if abs(val - true_pred) < tol:
                llm_val = val
                passed = True
                break
        except ValueError:
            pass

    return dict(test="insight_simulatability", passed=passed,
                ground_truth=round(true_pred, 3), llm_guess=llm_val, response=response)


def insight_sparse_feature_set(model, llm):
    """Insight B — Does the model string clearly reveal which features matter?

    Murdoch et al. (2019): sparsity is a key property — 'limiting the number
    of non-zero parameters so the model depends on a small number of features'.
    LASSO explicitly zeros out irrelevant features; dense OLS retains all.
    Shallow DTs only split on relevant features; deep DTs may split on noise.

    Ground truth: only x0 and x1 are relevant (8 features are pure noise).
    """
    X, y = _sparse_ten_feature_data()
    names = [f"x{i}" for i in range(10)]
    m = _safe_clone(model)
    m.fit(X, y)
    mstr = get_model_str(m, names)

    q = (
        "This model was trained on 10 features (x0–x9). "
        "Based solely on the model shown above, list ONLY the features that "
        "contribute meaningfully to predictions. Exclude features with negligible "
        "or zero effect. Give just a comma-separated list of feature names."
    )
    response = ask_llm(llm, mstr, q)

    passed = False
    listed = []
    if response:
        listed = re.findall(r"x\d+", response.lower())
        listed_set = set(listed)
        has_x0 = "x0" in listed_set
        has_x1 = "x1" in listed_set
        extras = listed_set - {"x0", "x1"}
        # Pass if both key features named and at most 1 false positive
        passed = has_x0 and has_x1 and len(extras) <= 1

    return dict(test="insight_sparse_feature_set", passed=passed,
                ground_truth="x0, x1 only", listed=listed, response=response)


def insight_nonlinear_shape(model, llm):
    """Insight C — Describe the shape of the x0 → output relationship.

    Murdoch et al. (2019): modularity lets each feature's effect be understood
    independently. GAMs excel here: their string explicitly shows a shape function
    for each feature. OLS cannot capture non-linearity at all. DTs show piecewise steps.

    Dataset: y = 3*max(0, x0) — flat for x0 < 0, linearly increasing for x0 > 0.
    An interpretable model should communicate:
      (1) there is a threshold near x0 = 0
      (2) the output is flat below the threshold
      (3) the output increases above the threshold
    """
    X, y = _hockey_stick_data()
    names = [f"x{i}" for i in range(3)]
    m = _safe_clone(model)
    m.fit(X, y)
    r2 = r2_score(y, m.predict(X))

    mstr = get_model_str(m, names)
    q = (
        "Describe the relationship between x0 and the predicted output "
        "(holding x1=0 and x2=0). Specifically answer: "
        "(1) Is there a threshold value of x0 below which x0 has little or no effect? "
        "If so, what is it approximately? "
        "(2) What happens to the prediction as x0 increases above that threshold?"
    )
    response = ask_llm(llm, mstr, q, max_tokens=250)

    # Pass if: identifies threshold near 0, AND identifies increasing trend above it
    threshold_ok = False
    increasing_ok = False
    if response:
        r = response.lower()
        # threshold near 0: a number close to 0, or relevant keywords
        nums = re.findall(r"-?\d+\.?\d*", r)
        near_zero = any(abs(float(n)) < 0.7 for n in nums)
        threshold_ok = near_zero or any(
            w in r for w in ["zero", "negative", "below zero", "0.0", "flat",
                              "constant", "no effect", "no change", "no impact",
                              "hockey", "piecewise", "kink", "relu"]
        )
        increasing_ok = any(
            w in r for w in ["increase", "linear", "positive slope", "grows",
                              "rises", "higher", "greater", "upward", "positively"]
        )

    passed = threshold_ok and increasing_ok and r2 > 0.5

    return dict(test="insight_nonlinear_shape", passed=passed, r2=r2,
                ground_truth="flat for x0<0, linearly increasing for x0>0",
                response=response)


def insight_counterfactual_target(model, llm):
    """Insight D — Targeted counterfactual: what x0 value reaches a specific target?

    Kaur et al. (2020): ~23% of practitioners expect counterfactual explanations.
    Murdoch et al. (2019): modularity/simulatability enable inverting the model.
    OLS: solve (target - intercept - coef_x1*x1) / coef_x0.
    DT: trace to the leaf containing the target value.
    RF/MLP: cannot invert from the string representation.
    """
    coefs = [4.0, 2.0, 0.0]
    X, y = _multi_feature_data(coefs, n=500, seed=22)
    names = [f"x{i}" for i in range(3)]
    m = _safe_clone(model)
    m.fit(X, y)

    x_base = np.array([[1.0, 1.0, 0.0]])
    pred_base = float(m.predict(x_base)[0])
    target = pred_base + 8.0

    # Binary-search for true x0 that hits target
    lo, hi = -10.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if float(m.predict(np.array([[mid, 1.0, 0.0]]))[0]) < target:
            lo = mid
        else:
            hi = mid
    true_x0 = (lo + hi) / 2

    mstr = get_model_str(m, names)
    q = (
        f"The model predicts {pred_base:.2f} for x0=1.0, x1=1.0, x2=0.0. "
        f"What value of x0 (keeping x1=1.0 and x2=0.0 fixed) would make the model "
        f"predict {target:.2f}? Answer with just a number."
    )
    response = ask_llm(llm, mstr, q)

    llm_val = None
    passed = False
    nums = re.findall(r"-?\d+\.?\d*", response or "")
    if nums:
        try:
            llm_val = float(nums[0])
            tol = max(abs(true_x0) * 0.15, 0.5)
            passed = abs(llm_val - true_x0) < tol
        except ValueError:
            pass

    return dict(test="insight_counterfactual_target", passed=passed,
                ground_truth=round(true_x0, 3), llm_guess=llm_val,
                pred_base=round(pred_base, 2), target=round(target, 2),
                response=response)


def insight_decision_region(model, llm):
    """Insight E — Extract the decision region: above what x0 does pred exceed a threshold?

    Kaur et al. (2020): 60%+ of practitioners want 'individual features' global effects'.
    Murdoch et al. (2019): simulatability enables rule extraction.
    OLS: solve threshold_y = coef*x0 + intercept → one inequality.
    Shallow DT: identify the leaf boundary directly from the tree text.
    RF/GBM/MLP: require running the model — cannot extract from string.
    """
    X, y = _single_feature_data(n_features=3, true_feature=0, coef=4.0, seed=23)
    names = [f"x{i}" for i in range(3)]
    m = _safe_clone(model)
    m.fit(X, y)

    # Binary-search for x0 where pred crosses 6.0
    threshold_y = 6.0
    lo, hi = -5.0, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if float(m.predict(np.array([[mid, 0.0, 0.0]]))[0]) < threshold_y:
            lo = mid
        else:
            hi = mid
    true_x0_boundary = (lo + hi) / 2

    mstr = get_model_str(m, names)
    q = (
        "With x1=0 and x2=0, for what values of x0 does this model predict ABOVE 6.0? "
        "Give the threshold value of x0 (e.g., 'x0 > 1.5'). "
        "Answer with just the threshold number."
    )
    response = ask_llm(llm, mstr, q)

    llm_val = None
    passed = False
    nums = re.findall(r"-?\d+\.?\d*", response or "")
    if nums:
        try:
            llm_val = float(nums[0])
            passed = abs(llm_val - true_x0_boundary) < 0.4
        except ValueError:
            pass

    return dict(test="insight_decision_region", passed=passed,
                ground_truth=round(true_x0_boundary, 3), llm_guess=llm_val,
                response=response)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

INSIGHT_TESTS = [
    insight_simulatability,
    insight_sparse_feature_set,
    insight_nonlinear_shape,
    insight_counterfactual_target,
    insight_decision_region,
]

HARD_TESTS = [
    hard_test_all_features_active,
    hard_test_pairwise_anti_intuitive,
    hard_test_quantitative_sensitivity,
    hard_test_mixed_sign_goes_negative,
    hard_test_two_feature_perturbation,
]

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


def _run_test_list(model, model_name, llm, test_list, label, verbose=True):
    if verbose:
        print(f"\n{'='*65}")
        print(f"  Model: {model_name}  [{label}]")
        print("=" * 65)

    results = []
    for test_fn in test_list:
        try:
            result = test_fn(model, llm)
        except AssertionError as e:
            result = dict(test=test_fn.__name__, passed=False,
                          error=f"Assertion: {e}", response=None)
        except Exception as e:
            result = dict(test=test_fn.__name__, passed=False,
                          error=str(e), response=None)

        result["model"] = model_name
        result.setdefault("test", test_fn.__name__)
        results.append(result)

        if verbose:
            status = "PASS" if result["passed"] else "FAIL"
            resp_preview = (result.get("response") or "")[:80].replace("\n", " ")
            gt = result.get("ground_truth", "")
            print(f"  [{status}] {result['test']}")
            print(f"         ground_truth : {gt}")
            print(f"         llm_response : {resp_preview}")

    n_passed = sum(r["passed"] for r in results)
    if verbose:
        print(f"\n  → {n_passed}/{len(test_list)} passed")

    return results


def run_all_tests(model, model_name, llm, verbose=True):
    return _run_test_list(model, model_name, llm, ALL_TESTS, "standard", verbose)


def run_hard_tests(model, model_name, llm, verbose=True):
    return _run_test_list(model, model_name, llm, HARD_TESTS, "hard", verbose)


def run_insight_tests(model, model_name, llm, verbose=True):
    return _run_test_list(model, model_name, llm, INSIGHT_TESTS, "insight", verbose)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    llm = imodelsx.llm.get_llm(CHECKPOINT)

    # Original models for standard/hard tests
    orig_models = [
        (DecisionTreeRegressor(max_depth=5, random_state=42), "DecisionTree"),
        (RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42), "RandomForest"),
        (LinearRegression(), "OLS"),
        (MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=1000,
                      random_state=42, learning_rate_init=0.01), "MLP"),
    ]

    # Extended model set for insight tests — covers degrees of interpretability
    # Expected ranking: GAM ≈ DT_shallow > LASSO > OLS > DT_deep > RF ≈ GBM > MLP
    insight_models = [
        (DecisionTreeRegressor(max_depth=2, random_state=42), "DT_shallow"),
        (DecisionTreeRegressor(max_depth=8, random_state=42), "DT_deep"),
        (Lasso(alpha=0.1), "LASSO"),
        (LinearRegression(), "OLS"),
        (RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42), "RandomForest"),
        (GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42), "GBM"),
        (MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=1000,
                      random_state=42, learning_rate_init=0.01), "MLP"),
    ]
    if _HAS_PYGAM:
        from pygam import LinearGAM
        insight_models.insert(0, (LinearGAM(n_splines=10), "GAM"))

    if _HAS_IMODELS:
        from imodels import FIGSRegressor, RuleFitRegressor, HSTreeRegressor, TreeGAMRegressor
        imodels_models = [
            (FIGSRegressor(max_rules=12), "FIGS"),
            (RuleFitRegressor(max_rules=20, random_state=42), "RuleFit"),
            (HSTreeRegressor(max_leaf_nodes=16, random_state=42), "HSTree"),
            (TreeGAMRegressor(n_boosting_rounds=5, max_leaf_nodes=4, random_state=42), "TreeGAM"),
        ]
        insight_models.extend(imodels_models)

    def print_summary(results, title):
        model_names = list(dict.fromkeys(r["model"] for r in results))
        test_names  = list(dict.fromkeys(r["test"]  for r in results))
        col = 16
        print("\n\n" + "=" * (36 + col * len(model_names)))
        print(f"SUMMARY — {title}")
        print("=" * (36 + col * len(model_names)))
        header = f"{'Test':<36}" + "".join(f"{n:<{col}}" for n in model_names)
        print(header)
        print("-" * len(header))
        for tname in test_names:
            row = f"{tname:<36}"
            for mname in model_names:
                match = [r for r in results if r["test"] == tname and r["model"] == mname]
                row += f"{'PASS' if match and match[0]['passed'] else 'FAIL':<{col}}"
            print(row)
        print("\nTotal passed per model:")
        for mname in model_names:
            subset = [r for r in results if r["model"] == mname]
            n = sum(r["passed"] for r in subset)
            print(f"  {mname:<22}: {n}/{len(subset)}")

    std_results, hard_results = [], []
    for mdl, name in orig_models:
        std_results.extend(run_all_tests(mdl, name, llm))
        hard_results.extend(run_hard_tests(mdl, name, llm))

    insight_results = []
    for mdl, name in insight_models:
        insight_results.extend(run_insight_tests(mdl, name, llm))

    print_summary(std_results,     "STANDARD TESTS (8)")
    print_summary(hard_results,    "HARD TESTS (5) — require arithmetic through model")
    print_summary(insight_results, "INSIGHT TESTS (5) — grounded in interpretability literature")
