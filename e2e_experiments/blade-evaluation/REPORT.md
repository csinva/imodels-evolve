# BLADE Evaluation Report: OpenAI Codex on End-to-End Data Science Tasks

## Overview

This evaluation assesses how well the OpenAI Codex agent performs on end-to-end data science tasks from the [BLADE benchmark](https://github.com/behavioral-data/blade) (Benchmark for Language-model-based Analysis of Data Experiments). BLADE contains 13 datasets, each paired with a research question and expert-annotated statistical analysis specifications.

## Setup

- **Agent**: OpenAI Codex CLI (`@openai/codex`) via Azure OpenAI with Entra ID authentication
- **Model**: `gpt-5.1` (Azure deployment: `fxdata-eastus2`)
- **Configuration**: `model_reasoning_effort="high"`, sandbox mode `workspace-write`
- **Evaluation**: LLM-as-a-judge (Azure OpenAI `gpt-5-mini`) scoring correctness, completeness, and clarity (1-5 each)

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
2. Autonomously explores the data, selects statistical methods, and produces an analysis
3. Outputs a `conclusion.txt` with a Likert score (0-100) and written explanation

The LLM-as-a-judge evaluation compares each conclusion against human expert annotations (from `annotations.csv`) on three criteria:
- **Correctness**: Does the analysis use sound methodology and reach a defensible conclusion?
- **Completeness**: Does it consider confounders, data issues, and alternative explanations?
- **Clarity**: Is the explanation well-structured and precise?

## How to Run

```bash
cd blade-evaluation

# 1. Setup Azure OpenAI authentication
source setup_azure.sh

# 2. Prepare dataset directories (already done)
python prepare_run.py

# 3. Run Codex on all datasets
bash run_all.sh

# 4. Run LLM-as-a-judge evaluation
source refresh_token.sh
python evaluate.py --verbose
```

To run a single dataset: `bash run_all.sh hurricane`
To skip already-completed datasets: `bash run_all.sh --skip-existing`

## Results

*Results will be populated after running the pipeline. Run `python evaluate.py --verbose` to generate `results.csv` and the summary table.*

## File Structure

```
blade-evaluation/
├── setup_azure.sh      # Azure OpenAI + Codex CLI configuration
├── refresh_token.sh    # Entra ID token refresh
├── prepare_run.py      # Prepares dataset directories under outputs/
├── run_all.sh          # Runs Codex on all 13 datasets
├── evaluate.py         # LLM-as-a-judge evaluation
├── REPORT.md           # This report
├── results.csv         # Generated evaluation results
└── outputs/
    ├── affairs/        # Each contains: AGENTS.md, info.json, {name}.csv,
    ├── amtl/           #   packages.txt, and (after run) conclusion.txt
    ├── ...
    └── teachingratings/
```
