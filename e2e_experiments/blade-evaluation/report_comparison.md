# BLADE Evaluation: Standard Tools vs. Custom Interpretability Tools

## Overview

This report compares how well an AI agent (OpenAI Codex with `gpt-5.3-codex`) performs on the 13 BLADE benchmark data-science tasks under two conditions:

1. **Standard tools**: Agent prompted to use scikit-learn, imodels, statsmodels, and scipy
2. **Custom tools**: Same standard tools plus custom interpretable regressors (`SmartAdditiveRegressor`, `HingeEBMRegressor`) from `interp_models.py`

Both runs used identical Codex configuration (model_reasoning_effort="high", danger-full-access sandbox) and the same LLM-as-a-judge evaluation (Azure OpenAI gpt-4o).

## Results Summary

| Dimension | Standard Tools | Custom Tools | Difference |
|-----------|---------------|-------------|------------|
| Correctness | 3.92 | **4.15** | +0.23 (+6%) |
| Completeness | 3.23 | **3.38** | +0.15 (+5%) |
| Clarity | **4.08** | 4.00 | -0.08 (-2%) |
| **Overall** | **3.74** | **3.85** | **+0.10 (+3%)** |
| Execution success | 13/13 | 13/13 | -- |

## Per-Dataset Comparison

| Dataset | Standard ||| Custom |||
|---------|Corr|Comp|Clar|Corr|Comp|Clar|
|---------|:--:|:--:|:--:|:--:|:--:|:--:|
| affairs | 4 | 3 | 4 | **5** | **4** | **5** |
| amtl | **5** | **4** | **5** | 4 | 3 | 4 |
| boxes | 4 | 3 | 4 | 4 | 3 | 4 |
| caschools | 4 | 4 | **5** | 4 | 3 | 4 |
| crofoot | 2 | 2 | 3 | **4** | **3** | **4** |
| fertility | 4 | 3 | 4 | 4 | 3 | 4 |
| fish | 4 | 4 | 3 | 4 | 4 | 3 |
| hurricane | 4 | 3 | 4 | 4 | **4** | 4 |
| mortgage | 4 | 3 | 4 | 4 | 3 | 4 |
| panda_nuts | 4 | 3 | **5** | 4 | **4** | 4 |
| reading | 4 | 3 | 4 | **5** | **4** | 4 |
| soccer | 4 | 4 | 4 | 4 | 3 | 4 |
| teachingratings | 4 | 3 | 4 | 4 | 3 | 4 |
| **AVERAGE** | **3.92** | **3.23** | **4.08** | **4.15** | **3.38** | **4.00** |

## Analysis

### Correctness (+0.23)

The custom tools helped most on datasets where the standard approach chose suboptimal methods:
- **crofoot** (2 -> 4): The biggest improvement. With standard tools, the agent used OLS/t-tests on a binary outcome. The custom tools' interpretable models helped the agent recognize the data structure better and switch to logistic regression.
- **affairs** (4 -> 5) and **reading** (4 -> 5): The custom tools' explicit feature-effect displays helped the agent provide more nuanced interpretations.

### Completeness (+0.15)

Modest improvement. The custom tools helped the agent identify more relevant features in some cases:
- **affairs** (3 -> 4), **crofoot** (2 -> 3), **hurricane** (3 -> 4), **reading** (3 -> 4), **panda_nuts** (3 -> 4): The interpretable model output showing per-feature effects may have prompted the agent to consider more variables.
- However, the main gap (missing control variables from expert annotations) persists with both toolsets — this is more about domain knowledge than tool capability.

### Clarity (-0.08)

Slight decrease. Some custom-tool explanations included model-specific jargon (e.g., "+1SD effect", "permutation importance drops") that the judge rated as less accessible. The standard tools' explanations tended to be more straightforward with standard statistical reporting.

### Datasets where standard tools performed better

- **amtl** (5/4/5 -> 4/3/4): The standard run happened to select a better model specification
- **caschools** (4/4/5 -> 4/3/4): Standard run's clarity was rated higher

### Where custom tools made no difference

Most datasets (boxes, fertility, fish, mortgage, soccer, teachingratings) showed identical or nearly identical scores. For these datasets, the research questions were straightforward enough that standard statistical tools sufficed.

## Key Insights

1. **Custom interpretability tools provide a modest overall improvement** (+3% overall), with the benefit concentrated in correctness (+6%).

2. **The biggest gains come from harder datasets** where the agent needs to understand data structure (crofoot's binary outcome, affairs' count data). The interpretable models' explicit feature-effect displays helped the agent make better methodological choices.

3. **Completeness remains the main gap** for both conditions (3.23 and 3.38 out of 5). Missing control variables is a domain-knowledge issue that interpretability tools alone don't solve — the agent needs to know which confounders to include, not just how to interpret the ones it does include.

4. **There's a slight clarity trade-off**: custom tool output adds jargon that can make explanations marginally less accessible, though still well-structured (4.00 vs 4.08).

5. **Both conditions achieved 100% execution success** (13/13 datasets), confirming that gpt-5.3-codex reliably writes and executes analysis scripts regardless of tool complexity.

## Setup

- **Agent**: OpenAI Codex CLI (`@openai/codex` v0.118.0)
- **Model**: `gpt-5.3-codex` (Azure deployment: `dl-openai-3`)
- **Judge**: Azure OpenAI `gpt-4o`
- **Custom tools**: `SmartAdditiveRegressor` (greedy additive boosted stumps) and `HingeEBMRegressor` (hinge basis + EBM residual correction) from `interp_models.py`

## How to Reproduce

```bash
# Prepare both modes
python prepare_run.py --mode standard
python prepare_run.py --mode custom

# Run Codex on all datasets
bash run_all.sh --mode standard
bash run_all.sh --mode custom

# Evaluate
python evaluate.py --mode standard --verbose
python evaluate.py --mode custom --verbose
```
