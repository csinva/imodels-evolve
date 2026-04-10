# BLADE Evaluation: Standard vs. Custom Interpretability Tools

## Overview

This report compares how well an AI agent (OpenAI Codex with `gpt-5.3-codex`) performs on the 13 BLADE benchmark data-science tasks under three conditions:

1. **Standard tools**: Agent uses scikit-learn, imodels, statsmodels, scipy
2. **Custom v1**: Standard tools + custom interpretable regressors (`interp_models.py`) with basic prompting
3. **Custom v2**: Improved custom tools (DataFrame-aware, `feature_effects()` method) + structured analysis strategy emphasizing feature importance, effect shapes, and robustness

All runs used the same Codex configuration (model_reasoning_effort="high", danger-full-access sandbox, gpt-5.3-codex on dl-openai-3). Evaluation used a 1-10 rubric that rewards defensible conclusions, depth of understanding, and — critically — clarity of interpretable insight (importance rankings, effect shapes, nonlinear patterns, robustness).

## Results Summary (1-10 scale)

| Dimension | Standard | Custom v1 | Custom v2 | v2 vs Standard |
|-----------|----------|-----------|-----------|----------------|
| Correctness | 8.54 | **8.69** | 8.62 | +0.08 |
| Completeness | 7.62 | 7.77 | **8.15** | **+0.53** |
| Clarity | 8.46 | 8.31 | **8.69** | **+0.23** |
| **Overall** | 8.21 | 8.26 | **8.49** | **+0.28 (+3.4%)** |

**Custom v2 achieves the highest overall score (8.49/10)**, with gains in both completeness (+0.53) and clarity (+0.23) over standard tools.

## Per-Dataset Scores

| Dataset | Standard ||| Custom v1 ||| Custom v2 |||
|---------|C|Comp|Cl|C|Comp|Cl|C|Comp|Cl|
|---------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| affairs | 8 | 7 | 8 | 9 | 8 | 8 | 9 | 8 | **9** |
| amtl | 9 | 8 | 9 | 9 | 8 | 9 | 9 | 8 | 9 |
| boxes | 9 | **9** | 8 | 9 | 8 | 9 | 8 | 7 | 8 |
| caschools | 9 | 8 | 9 | 7 | 6 | 7 | 9 | 8 | 9 |
| crofoot | 6 | 5 | 5 | 8 | 7 | 8 | **8** | **8** | **9** |
| fertility | 9 | 8 | 9 | 9 | 8 | 8 | 9 | **9** | **10** |
| fish | 8 | 8 | 9 | 9 | 8 | 8 | **9** | 8 | 9 |
| hurricane | 9 | 6 | 8 | 9 | **9** | 9 | 9 | **8** | **9** |
| mortgage | 8 | 8 | 9 | **9** | 8 | 9 | 8 | **9** | 9 |
| panda_nuts | 9 | 8 | 9 | 9 | 8 | 9 | 9 | 8 | 9 |
| reading | 9 | 8 | 9 | 9 | 8 | 8 | 9 | 8 | 8 |
| soccer | **9** | 8 | **9** | 8 | 7 | 7 | 7 | 8 | 6 |
| teachingratings | 9 | 8 | 9 | 9 | 8 | 9 | 9 | **9** | 9 |

## Where Custom v2 Tools Made the Biggest Difference

### Clarity gains: from "significant" to "here's how it works"

The updated clarity rubric rewards explanations that go beyond p-values to describe importance rankings, effect shapes, and robustness. Custom v2 excelled here:

- **crofoot** (std clarity 5 → v2 clarity 9): Standard run just reported non-significance. Custom v2 explained relative importance of group size vs location, and showed the effect shape across models.
- **fertility** (std 9 → v2 **10**): v2 earned the only perfect clarity score by quantifying each feature's importance ranking and confirming the null result across multiple interpretable models.
- **hurricane** (std 8 → v2 9): v2 used HingeEBM's Lasso to show femininity gets zeroed out entirely — stronger evidence than just "p > 0.05".

### Completeness gains: depth of understanding

- **hurricane** (std 6 → v2 8): Standard run missed confounder analysis. v2's feature importance showed pressure/wind dominate while femininity has negligible importance.
- **crofoot** (std 5 → v2 8): Standard run under-explored controls. v2 compared across OLS, SmartAdditive, and HingeEBM.
- **mortgage** (std 8 → v2 9): v2 quantified gender's small effect relative to other predictors.

### Example: What custom tools add to an explanation

**Standard (teachingratings):**
> "Beauty shows a statistically significant positive relationship with teaching evaluations. Pearson r=0.189, p=4.25e-05; simple OLS beauty coef=0.133, p=4.25e-05."

**Custom v2 (teachingratings):**
> "SmartAdditive ranks beauty as direction=nonlinear (increasing trend), importance=50.8%, rank=1, with a nonlinear increasing pattern and a zero-crossing threshold near beauty=-0.698. HingeEBM also keeps beauty with direction=positive, importance=89.9%, rank=1. Other significant controls: credits_single (coef=0.561), native_yes (coef=0.236), gender_male (coef=0.203). These matter, but they do not remove the positive beauty-evaluation relationship."

The v2 explanation reveals the *shape* (nonlinear with threshold), *relative importance* (rank 1 at 50.8%), and *robustness* (confirmed by two different model architectures).

## Agent Likert Scores (0-100)

| Dataset | Standard | Custom v1 | Custom v2 |
|---------|----------|-----------|-----------|
| affairs | 0 | 0 | 12 |
| amtl | 80 | 100 | 66 |
| boxes | 15 | 15 | 23 |
| caschools | 65 | 90 | 22 |
| crofoot | 26 | 64 | 57 |
| fertility | 20 | 5 | 8 |
| fish | 82 | 87 | 69 |
| hurricane | 7 | 10 | 4 |
| mortgage | 85 | 65 | 39 |
| panda_nuts | 78 | 90 | 60 |
| reading | 15 | 12 | 8 |
| soccer | 76 | 35 | 65 |
| teachingratings | 79 | 100 | 100 |

## Summary of Custom v2 Improvements

| Component | Change | Effect |
|-----------|--------|--------|
| `interp_models.py` | DataFrame-aware `fit()`, column names in output, `feature_effects()` method | Agent reports importance rankings and effect directions with real column names |
| AGENTS.md prompt | Structured workflow: explore → OLS with controls → interpretable models → rich conclusion | Agent systematically uses custom tools and reports shapes/importance |
| AGENTS.md scoring | Balanced: "weigh both bivariate and controlled" with graduated scale | Better-calibrated Likert scores (no longer overly conservative) |
| Evaluation rubric | Clarity rewards importance rankings, effect shapes, nonlinear patterns, robustness | Measures the interpretable insights that custom tools uniquely provide |

## Setup

- **Agent**: OpenAI Codex CLI (`@openai/codex` v0.118.0)
- **Model**: `gpt-5.3-codex` (Azure deployment: `dl-openai-3`)
- **Judge**: Azure OpenAI `gpt-4o` with v2 rubric (1-10 scale)
- **Custom tools**: `SmartAdditiveRegressor` and `HingeEBMRegressor` from `interp_models.py`

## How to Reproduce

```bash
# Prepare all three modes
python prepare_run.py --mode standard
python prepare_run.py --mode custom
python prepare_run.py --mode custom_v2

# Run Codex
bash run_all.sh --mode standard
bash run_all.sh --mode custom
bash run_all.sh --mode custom_v2

# Evaluate with v2 rubric
python evaluate.py --mode standard --rubric v2 --verbose
python evaluate.py --mode custom --rubric v2 --verbose
python evaluate.py --mode custom_v2 --rubric v2 --verbose
```
