# BLADE Evaluation Report: AI Agent on End-to-End Data Science Tasks

## Overview

This evaluation assesses how well an AI agent system performs on end-to-end data science tasks from the [BLADE benchmark](https://github.com/behavioral-data/blade) (Benchmark for Language-model-based Analysis of Data Experiments). BLADE contains 13 datasets, each paired with a research question and expert-annotated statistical analysis specifications.

## Setup

- **Agent**: OpenAI Codex CLI (`@openai/codex` v0.118.0) via Azure OpenAI with keyless Entra ID authentication
- **Model**: `gpt-5.3-codex` (Azure deployment: `dl-openai-3`)
- **Configuration**: `model_reasoning_effort="high"`, sandbox mode `danger-full-access`
- **Evaluation**: LLM-as-a-judge (Azure OpenAI `gpt-4o` via `dl-openai-3`) scoring correctness, completeness, and clarity (1-5 each)
- **Authentication**: Keyless via `ChainedTokenCredential` (AzureCli -> ManagedIdentity)

## Datasets

| # | Dataset | Research Question (abbreviated) |
|---|---------|-------------------------------|
| 1 | affairs | Extramarital affairs and relationship factors |
| 2 | amtl | AMTL analysis |
| 3 | boxes | Boxes experiment |
| 4 | caschools | California school test scores and class size |
| 5 | crofoot | Crofoot behavioral study |
| 6 | fertility | Fertility dataset analysis |
| 7 | fish | Fish dataset analysis |
| 8 | hurricane | Hurricane name femininity and fatalities |
| 9 | mortgage | Mortgage data analysis |
| 10 | panda_nuts | Panda nuts experiment |
| 11 | reading | Reading study |
| 12 | soccer | Skin tone and red cards in soccer |
| 13 | teachingratings | Teaching ratings and instructor characteristics |

## Methodology

For each dataset, the Codex agent:
1. Receives an `AGENTS.md` prompt, `info.json` (metadata + research question), and the dataset CSV
2. Autonomously explores the data, selects statistical methods, writes and executes a Python analysis script
3. Outputs a `conclusion.txt` with a Likert score (0-100) and written explanation

The LLM-as-a-judge evaluation compares each conclusion against human expert annotations (from `annotations.csv`) on three criteria:
- **Correctness** (1-5): Does the analysis use sound methodology and reach a defensible conclusion?
- **Completeness** (1-5): Does it consider confounders, data issues, and alternative explanations?
- **Clarity** (1-5): Is the explanation well-structured and precise?

## Results

### Summary Scores

| Dataset | Response | Correctness | Completeness | Clarity |
|---------|----------|-------------|--------------|---------|
| affairs | 3 | 5 | 4 | 5 |
| amtl | 96 | 4 | 3 | 4 |
| boxes | 10 | 4 | 3 | 4 |
| caschools | 18 | 4 | 3 | 4 |
| crofoot | 15 | 4 | 3 | 4 |
| fertility | 12 | 4 | 4 | 4 |
| fish | 92 | 4 | 3 | 4 |
| hurricane | 10 | 4 | 3 | 4 |
| mortgage | 2 | 5 | 2 | 4 |
| panda_nuts | 80 | 3 | 3 | 4 |
| reading | 8 | 4 | 3 | 4 |
| soccer | 66 | 4 | 3 | 4 |
| teachingratings | 95 | 4 | 3 | 4 |
| **AVERAGE** | | **4.08** | **3.08** | **4.08** |

**Overall average score: 3.74 / 5.00**

### Key Findings

1. **Correctness (avg 4.08/5)**: The agent generally chose appropriate statistical methods — logistic regression for binary outcomes (boxes, crofoot), OLS with controls (caschools, hurricane, teachingratings), non-parametric tests when distributions were skewed (panda_nuts), and rate-based z-tests (soccer). Two datasets scored 5/5 (affairs, mortgage).

2. **Completeness (avg 3.08/5)**: The main gap. While the agent typically identified core IVs and DVs, it sometimes missed control variables identified by human experts (e.g., confounders like player height/weight in soccer, sockets in amtl, class size specifics in teachingratings). Mortgage scored lowest (2/5) for not considering confounders like employment status and skin color.

3. **Clarity (avg 4.08/5)**: Consistently strong. All explanations were well-structured with clear statistical reporting (test statistics, p-values, confidence intervals). Affairs scored 5/5.

### Comparison: o4-mini vs gpt-5.3-codex

| Dimension | o4-mini (prev run) | gpt-5.3-codex | Improvement |
|-----------|-------------------|---------------|-------------|
| Correctness | 2.15 | **4.08** | +90% |
| Completeness | 1.62 | **3.08** | +90% |
| Clarity | 3.00 | **4.08** | +36% |
| **Overall** | **2.26** | **3.74** | **+65%** |
| Execution success | 3/13 (23%) | **13/13 (100%)** | +77pp |

The switch from `o4-mini` to `gpt-5.3-codex` resolved both the execution reliability issue (100% vs 23% success) and dramatically improved analysis quality across all dimensions.

## How to Run

```bash
cd blade-evaluation

# 1. Ensure Azure CLI is logged in
az login

# 2. Prepare dataset directories
python prepare_run.py

# 3. Run Codex on all datasets (uses ~/.codex/config.toml for dl-openai-3 / gpt-5.3-codex)
bash run_all.sh

# 4. Run LLM-as-a-judge evaluation
python evaluate.py --verbose
```

To run a single dataset: `bash run_all.sh hurricane`
To skip already-completed datasets: `bash run_all.sh --skip-existing`

## File Structure

```
blade-evaluation/
├── prepare_run.py      # Prepares dataset directories under outputs/
├── run_all.sh          # Runs Codex on all 13 datasets (keyless Azure auth)
├── run_analysis.py     # Generic statistical analysis fallback (optional)
├── evaluate.py         # LLM-as-a-judge evaluation (keyless Azure auth)
├── report.md           # This report
├── results.csv         # Generated evaluation results
└── outputs/
    ├── affairs/        # Each contains: AGENTS.md, info.json, {name}.csv,
    ├── amtl/           #   packages.txt, analysis.py, conclusion.txt
    ├── ...
    └── teachingratings/
```
