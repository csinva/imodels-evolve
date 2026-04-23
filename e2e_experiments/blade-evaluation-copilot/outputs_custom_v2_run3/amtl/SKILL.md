---
name: agentic-imodels
description: Interpretable tabular regressors with a sklearn API; print the fitted model to get a human- and LLM-readable equation / tree / rule set. Pick a model based on the interpretability-vs-performance tradeoff table below.
---

# agentic-imodels SKILL

## When to use this skill

The user has a **tabular regression** task (continuous target, ≤ a few thousand
rows, ≤ ~50 features) and wants a model whose fitted form can be read and
reasoned about — either by a human or by another LLM. Typical phrasings:

- "fit an interpretable regressor on …"
- "train a GAM / sparse linear / piecewise model on …"
- "give me a model whose equation I can show to a stakeholder"
- "I need a model whose predictions I can explain without SHAP"

Prefer this skill over raw scikit-learn when the user specifically wants
interpretability, or when the downstream consumer of the model is another
LLM (agentic-analysis pipelines).

## Installation

```bash
uv add agentic-imodels        # inside a uv project
# or
pip install agentic-imodels
```

Runtime dependencies: `numpy`, `scikit-learn`, `interpret` (pulled in for
the EBM-backed models).

## API — it's just a sklearn regressor

Every class in `agentic_imodels` inherits from
`sklearn.base.BaseEstimator + RegressorMixin`. The usage is uniform:

```python
from agentic_imodels import HingeEBMRegressor

model = HingeEBMRegressor()     # no required args
model.fit(X_train, y_train)     # X: 2d array/DataFrame, y: 1d array
y_hat = model.predict(X_test)
print(model)                    # human-readable form — SHOW THIS to the user
```

Because they are sklearn estimators, you can drop them into:

- `sklearn.pipeline.Pipeline` — useful if you need imputation / scaling
- `sklearn.model_selection.cross_val_score`, `GridSearchCV`
- `sklearn.ensemble.VotingRegressor` / `StackingRegressor`

**Always call `print(model)` after fitting** — that is the whole point of
the library. The output is also cheap to feed back into a downstream LLM:
it is short (typically 10–70 lines), in plain ASCII, and each class was
explicitly optimized to be readable by an LLM evaluator.

## Picking a model

All models were evolved against two axes:

- **Rank** — RMSE rank averaged globally across 65 regression datasets
  (lower is better). Baselines: TabPFN 94.5, EBM 154.1, RF 259.7,
  RidgeCV 358.1, OLS 354.5.
- **Test interp** — fraction of 157 held-out LLM-graded tests passed
  (higher is better). Baselines: OLS 0.69, HSTree 0.66, EBM 0.49,
  TabPFN 0.17.

| Class | Rank ↓ | Test interp ↑ | When to pick | Category |
| --- | ---: | ---: | --- | --- |
| `HingeEBMRegressor` | **108** | 0.71 | Best predictive rank at ~70% interp. Good default when you want both. | display-predict decoupled |
| `DistilledTreeBlendAtlasRegressor` | 140 | 0.71 | Very predictive; displays a probe-answer "atlas" card. | display-predict decoupled |
| `DualPathSparseSymbolicRegressor` | 164 | 0.71 | Sparse symbolic single-row equation, GBM/RF/Ridge blend behind it. | display-predict decoupled |
| `HybridGAM` | 164 | 0.68 | SmartAdditiveGAM display + hidden RF corrector. | display-predict decoupled |
| `TeacherStudentRuleSplineRegressor` | 204 | **0.80** | Highest held-out interpretability in the library. | display-predict decoupled |
| `SparseSignedBasisPursuitRegressor` | 273 | 0.76 | Honest: `predict == display`. Forward-selected signed basis + ridge. | honest |
| `HingeGAMRegressor` | 280 | 0.78 | Honest pure hinge GAM, 10 breakpoints, no hidden corrector. | honest |
| `WinsorizedSparseOLSRegressor` | 327 | 0.73 | Honest sparse linear: clip outliers, LassoCV select, OLS refit. | honest |
| `TinyDTDepth2Regressor` | 334 | 0.71 | Simplest — a 4-leaf decision tree. Easy to eyeball. | honest |
| `SmartAdditiveRegressor` | 354 | 0.73 | Adaptive-linearization GAM (shape shown as line if linear, else short piecewise table). | honest |

### Default-choice heuristic

- User just says **"fit an interpretable model"** → `HingeEBMRegressor`
  (best rank while still > 0.70 interp).
- User says **"I need the equation to be what the model actually computes"**
  (or mentions auditing / regulatory / medical) → pick an **honest**
  model: `SmartAdditiveRegressor`, `HingeGAMRegressor`,
  `SparseSignedBasisPursuitRegressor`, or `WinsorizedSparseOLSRegressor`.
  The decoupled models have a hidden corrector and their `__str__` will
  understate what `predict` does.
- User needs **maximum LLM-readability** (agent-driven analysis
  pipeline) → `TeacherStudentRuleSplineRegressor`.
- User asks for **a simple tree** → `TinyDTDepth2Regressor`.

### Honest vs. decoupled — read this before recommending

Five of the ten models use **display-predict decoupling**: the printed
equation is a clean sparse linear/additive formula, but `predict` also
adds a hidden corrector (EBM, RF, or GBM) trained on residuals. The
displayed formula therefore understates the model. This is noted
explicitly in the paper that produced these models as a known tradeoff:
it maximizes LLM-graded interpretability scores but can mislead readers
who assume the formula *is* the model.

If the user asks "is what you printed really the model?", the answer is
yes for `SmartAdditiveRegressor`, `HingeGAMRegressor`,
`SparseSignedBasisPursuitRegressor`, `WinsorizedSparseOLSRegressor`,
`TinyDTDepth2Regressor`; and **no** for the other five — you should
say so.

## Worked example

```python
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from agentic_imodels import SmartAdditiveRegressor, HingeEBMRegressor

X, y = fetch_california_housing(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0)

for cls in (SmartAdditiveRegressor, HingeEBMRegressor):
    m = cls().fit(X_tr, y_tr)
    print(f"=== {cls.__name__} — R^2={r2_score(y_te, m.predict(X_te)):.3f} ===")
    print(m)
    print()
```

Each printed block will be a compact, plain-text description of the
fitted model — a short equation, a short piecewise table, or a small
decision tree. Return that text verbatim to the user; do not paraphrase
or truncate it unless it exceeds the user's context budget.

## Recommended analysis workflow

When using this skill to answer a data-driven research question (e.g. "does
X affect Y, controlling for Z?"), follow these steps. They are the pattern
this library was evolved to support and are what makes the printed form
useful — skipping them tends to waste the interpretability gains.

### Step 1 — Frame the question

- Identify the **dependent variable** (DV / outcome) and the candidate
  **independent variable** (IV / predictor) the question is really asking
  about.
- List plausible **controls / confounders** from the other columns.
- Choose the right estimator family: OLS for continuous DV, logistic /
  `GLM(Binomial)` for binary DV, Poisson / NegativeBinomial for counts.
  `agentic_imodels` is a regressor library — if the DV is binary, fit the
  regressors on the 0/1 outcome for *shape* discovery but keep a
  `statsmodels` logistic regression alongside for the formal test.

### Step 2 — Classical statistical tests (with controls)

Before touching the interpretable models, establish the baseline
statistical picture. This is what produces the *p*-values and confidence
intervals that will anchor the conclusion.

```python
import statsmodels.api as sm
X = sm.add_constant(df[[iv_col, *control_cols]])
model = sm.OLS(df[dv_col], X).fit()   # or sm.Logit / sm.GLM
print(model.summary())
```

Report: sign and magnitude of the IV coefficient, its *p*-value, and
whether it survives adding the controls. If the bivariate effect
disappears under controls, that is itself an answer.

### Step 3 — Interpretable models for shape, direction, importance

Classical regression tells you *whether* an effect is present. The
interpretable models in this library tell you *how* it behaves. **Fit at
least two models from different categories** so you can check that the
story is robust; pick one "honest" model and one high-rank model:

```python
from agentic_imodels import (
    SmartAdditiveRegressor,     # honest GAM, reveals nonlinear shapes
    HingeGAMRegressor,           # honest pure hinge GAM
    WinsorizedSparseOLSRegressor,# honest sparse linear
    HingeEBMRegressor,           # high-rank, decoupled (hidden EBM corrector)
    TinyDTDepth2Regressor,       # 4-leaf tree, easy to sanity-check
)

X = df[numeric_feature_cols]   # keep as DataFrame so column names flow through
y = df[dv_col]

for cls in (SmartAdditiveRegressor, HingeEBMRegressor):
    m = cls().fit(X, y)
    print(f"=== {cls.__name__} ===")
    print(m)                     # the fitted form — SHOW THIS verbatim
```

From the printed form you should be able to read off, for every relevant
feature, four things the research question usually needs:

- **Direction** — is the effect positive, negative, zero, or non-monotone?
- **Magnitude / rank** — which features dominate? A hinge/Lasso model that
  *zeroes out* a feature is strong null evidence, usually stronger than a
  non-significant *p*-value alone.
- **Shape** — linear, threshold, diminishing-returns, U-shaped?
  `SmartAdditiveRegressor` is best at surfacing this, since it falls back
  to a piecewise table when a feature's shape is non-linear.
- **Robustness** — do the `SmartAdditive` and `HingeEBM` (or OLS) stories
  agree? If they disagree, say so explicitly.

### Step 4 — Write a rich, calibrated conclusion

A good answer goes beyond "significant / not significant":

- **Direction + magnitude**: "X has a positive effect on Y (OLS β=0.34,
  *p*=0.002; rank 2 / 8 in SmartAdditive importance at 19.8%)."
- **Shape**: "Effect is roughly linear for X₁ but shows a threshold at
  X₂≈25 (SmartAdditive switches from the linear form to a piecewise
  table)."
- **Robustness**: "Effect persists after controlling for Z₁, Z₂, Z₃;
  confirmed by both SmartAdditive and HingeEBM."
- **Competing predictors**: "Z₁ is actually the strongest predictor
  (importance 45.2%), so the IV matters but is not the dominant driver."
- **Null findings** should cite the Lasso/hinge zeroing, not just a
  non-significant *p*-value.

When the downstream consumer is a Likert-style score (e.g. 0 = strong
"No", 100 = strong "Yes"), calibrate it to the evidence:

- Strong significant effect that persists across models and is top-ranked
  in importance → 75–100
- Moderate / partially significant / mid-rank → 40–70
- Weak, inconsistent, or marginal → 15–40
- Zero coefficient in Lasso AND non-significant AND low importance → 0–15

Weigh bivariate **and** controlled results. A bivariate effect that
weakens but does not vanish after controls is a *moderate* score, not a
zero.

## Limitations

- Regression only — no classification support.
- Designed and evaluated on datasets with ≤ 1,000 training rows and ≤ 50
  features. Some models (e.g. `DistilledTreeBlendAtlasRegressor`) will be
  slow on larger data.
- No built-in handling of categorical features; encode them upstream
  (one-hot / target encoding) before `.fit`.
- No built-in missing-value imputation; use a `SimpleImputer` in a
  `Pipeline`.

## Inspecting provenance

Each module in `agentic_imodels/` records the exact source file it was
extracted from (commit hash + experiment run), along with its rank and
interp scores. If the user wants to see the raw evolved script, read the
top of the module file.
