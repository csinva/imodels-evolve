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

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor, export_text

import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "imodelsx_llm", "/home/chansingh/imodelsX/imodelsx/llm.py"
)
llm_module = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(llm_module)

CHECKPOINT = "gpt-4o"


# ---------------------------------------------------------------------------
# Model → string
# ---------------------------------------------------------------------------


def get_model_str(model, feature_names=None):
    """Return a human-readable string for a fitted sklearn regressor."""
    if feature_names is None and hasattr(model, "n_features_in_"):
        feature_names = [f"x{i}" for i in range(model.n_features_in_)]

    if isinstance(model, DecisionTreeRegressor):
        tree_text = export_text(model, feature_names=feature_names, max_depth=6)
        return f"Decision Tree Regressor:\n{tree_text}"

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

    return str(model)


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
    m = clone(model)
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
    m = clone(model)
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
    m = clone(model)
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
    m = clone(model)
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
    m = clone(model)
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
    m = clone(model)
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
    m = clone(model)
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
    m = clone(model)
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
# Runner
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


def run_all_tests(model, model_name, llm, verbose=True):
    if verbose:
        print(f"\n{'='*65}")
        print(f"  Model: {model_name}")
        print("=" * 65)

    results = []
    for test_fn in ALL_TESTS:
        try:
            result = test_fn(model, llm)
        except AssertionError as e:
            result = dict(test=test_fn.__name__, passed=False,
                          error=f"Assertion: {e}", response=None)
        except Exception as e:
            result = dict(test=test_fn.__name__, passed=False,
                          error=str(e), response=None)

        result["model"] = model_name
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
        print(f"\n  → {n_passed}/{len(ALL_TESTS)} passed")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    llm = llm_module.get_llm(CHECKPOINT)

    models = [
        (DecisionTreeRegressor(max_depth=5, random_state=42), "DecisionTree"),
        (RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42), "RandomForest"),
        (LinearRegression(), "OLS"),
        (MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=1000,
                      random_state=42, learning_rate_init=0.01), "MLP"),
    ]

    all_results = []
    for mdl, name in models:
        results = run_all_tests(mdl, name, llm)
        all_results.extend(results)

    # ---- Summary table ----
    model_names = list(dict.fromkeys(r["model"] for r in all_results))
    test_names = list(dict.fromkeys(r["test"] for r in all_results))

    col = 20
    print("\n\n" + "=" * 65)
    print("SUMMARY TABLE  (PASS / FAIL per model)")
    print("=" * 65)
    header = f"{'Test':<30}" + "".join(f"{n:<{col}}" for n in model_names)
    print(header)
    print("-" * len(header))
    for tname in test_names:
        row = f"{tname:<30}"
        for mname in model_names:
            match = [r for r in all_results if r["test"] == tname and r["model"] == mname]
            row += f"{'PASS' if match and match[0]['passed'] else 'FAIL':<{col}}"
        print(row)

    print("\nTotal passed per model:")
    for mname in model_names:
        subset = [r for r in all_results if r["model"] == mname]
        n = sum(r["passed"] for r in subset)
        print(f"  {mname:<20}: {n}/{len(subset)}")
