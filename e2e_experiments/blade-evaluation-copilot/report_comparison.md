# Blade Evaluation — Claude Code Agent

This folder mirrors `blade-evaluation/` but drives the agent with the Claude
Code CLI (`claude -p`, model `sonnet`) instead of OpenAI Codex. The Blade task
prompts, dataset preparation, LLM-as-a-judge scoring (Azure OpenAI), and
aggregation scripts are otherwise unchanged so results are directly
comparable to the Codex pipeline.

## Setup

- Agent: `claude -p --model sonnet --permission-mode bypassPermissions`,
  hard-capped at `timeout 900s` per dataset.
- Judge: Azure OpenAI `gpt-4o` via keyless Entra ID auth (unchanged from
  original pipeline)
- Datasets: 13 Blade tasks, sourced from
  `../blade-evaluation/outputs_standard_run1/` (info.json + CSV per dataset)
- Modes:
  - `standard` — agent instructed to use scikit-learn / imodels / statsmodels
  - `custom_v2` — `standard` plus the `agentic_imodels` package (10 evolved
    regressors) and `SKILL.md` (API + recommended analysis workflow) copied
    into each run directory.
- Runs: 3 agent runs per mode × 3 judge runs = **9 evaluations per dataset
  per mode**.

## Results (mean over 13 datasets, 1–10 scale, mean ± SE across 9 evaluations)

| Dimension     | standard         | custom_v2        | Δ (custom − std) |
| ------------- | ---------------- | ---------------- | ---------------- |
| Correctness   | 6.59 ± 0.10      | **8.38 ± 0.14**  | **+1.79**        |
| Completeness  | 5.61 ± 0.08      | **7.72 ± 0.10**  | **+2.11**        |
| Clarity       | 6.27 ± 0.10      | **8.33 ± 0.10**  | **+2.06**        |
| **Overall**   | **6.16 ± 0.09**  | **8.15 ± 0.11**  | **+1.99 (+32.3%)** |

**All 13 / 13 datasets improved** under custom_v2. SE bars do not overlap on
any dimension.

## Per-Dataset Scores (mean ± SE, n=9)

| Dataset         | Std Corr | Std Comp | Std Clar | Cus Corr | Cus Comp | Cus Clar |
|-----------------|----------|----------|----------|----------|----------|----------|
| affairs         | 5.9±0.5  | 5.0±0.4  | 5.3±0.5  | 6.7±1.0  | 6.4±0.7  | 7.1±0.9  |
| amtl            | 6.8±0.5  | 5.9±0.5  | 6.6±0.6  | **9.0±0.0** | **8.3±0.2** | **8.9±0.1** |
| boxes           | 4.8±0.5  | 3.6±0.5  | 4.2±0.4  | **9.0±0.2** | **8.0±0.2** | **8.9±0.1** |
| caschools       | 8.1±0.5  | 7.1±0.4  | 7.8±0.5  | 8.3±0.2  | 7.7±0.3  | 8.2±0.3  |
| crofoot         | 4.6±0.3  | 3.2±0.3  | 3.9±0.3  | **6.9±0.6** | **5.9±0.7** | **6.7±0.6** |
| fertility       | 8.7±0.2  | 7.0±0.2  | 8.1±0.2  | 8.8±0.1  | 8.0±0.2  | 8.7±0.2  |
| fish            | 5.4±0.2  | 4.9±0.1  | 5.7±0.2  | **8.7±0.2** | **8.3±0.2** | **8.7±0.2** |
| hurricane       | 7.8±0.3  | 6.8±0.3  | 7.3±0.3  | 9.0±0.2  | 8.1±0.2  | 9.0±0.0  |
| mortgage        | 6.0±0.4  | 5.7±0.3  | 6.0±0.3  | **8.0±0.4** | **7.4±0.3** | **7.9±0.5** |
| panda_nuts      | 6.9±0.6  | 6.0±0.6  | 6.7±0.7  | 7.9±0.1  | 7.3±0.3  | 8.1±0.1  |
| reading         | 5.1±0.4  | 3.9±0.4  | 4.7±0.2  | **8.8±0.1** | **8.0±0.2** | **8.3±0.2** |
| soccer          | 7.3±0.3  | 6.4±0.2  | 7.0±0.4  | **8.9±0.1** | **8.6±0.2** | **8.9±0.1** |
| teachingratings | 8.3±0.2  | 7.4±0.2  | 8.3±0.2  | 9.1±0.1  | 8.2±0.1  | 9.0±0.0  |

## Observations

- Claude's `standard` baseline is much lower than Codex's because Claude
  defaults to terser, less-structured analyses. The gap on the low-baseline
  datasets (`boxes`, `crofoot`, `fish`, `reading`) is especially large.
- Custom-tool conclusions consistently quote `str(model)` output for the
  sparse/GAM models in `agentic_imodels`, and report feature importance
  rankings, effect shapes, and robustness across models — the dimensions
  the judge rubric rewards explicitly.
- `affairs` shows the largest per-run variance under custom_v2
  (SE≈0.7–1.0): the agent sometimes writes a "strong Yes" (well-calibrated
  negation) and sometimes a "mild No," depending on how it integrates the
  bivariate-vs-controlled evidence.

## Files

- `run_all.sh` — runs `claude -p` on each dataset's run directory (timeout 15 min per dataset)
- `prepare_run.py` — builds per-dataset run dirs (uses sibling repo's data)
- `evaluate.py` — Azure OpenAI LLM-as-a-judge (unchanged from original)
- `aggregate_results.py` — aggregates judge CSVs (unchanged)
- `outputs_standard_run{1,2,3}/`, `outputs_custom_v2_run{1,2,3}/` — per-dataset working dirs with `analysis.py`, `conclusion.txt`, Claude CLI logs
- `judge_results/results_{mode}_run{R}_judge{J}.csv` — per-dataset scores
- `judge_results/results_aggregated.csv` — summary across modes
