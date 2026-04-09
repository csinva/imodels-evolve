# imodels-evolve

Autonomous AI research on interpretable scikit-learn regressors.

The idea: give an AI agent a training setup and let it experiment autonomously. It modifies `interpretable_regressor.py`, trains a regressor, checks if both metrics improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model.

## How it works

The repo has three files that matter:

- **`run_baselines.py`** — evaluates a fixed set of baseline regressors across two metrics: (1) `frac_interpretability_tests_passed` — LLM-graded interpretability tests, and (2) `mean_rank` — average rank of prediction performance on regression datasets. Results saved to `results/overall_results.csv`. **Not modified by the agent.**
- **`interpretable_regressor.py`** — the single file the agent edits. Defines `InterpretableRegressor` (a scikit-learn compatible model) and an evaluation loop that runs the same metrics and updates `results/overall_results.csv`. **This file is edited and iterated on by the agent.**
- **`program.md`** — instructions for the agent. Point your agent here and let it go. **This file is edited and iterated on by the human.**

## Metrics

Two metrics are tracked in `results/overall_results.csv`:

- **`mean_rank`** — mean rank across regression datasets (lower is better, evaluated on held-out test sets)
- **`frac_interpretability_tests_passed`** — fraction of LLM-graded interpretability tests passed (higher is better)

## Quick start

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Run baseline evaluation (interpretability tests + regression datasets RMSE, ~2 min with caching)
uv run run_baselines.py

# 4. Manually run a single training experiment
uv run interpretable_regressor.py
```

If the above commands all work, your setup is working and you can go into autonomous research mode. Then, ask claude to do the following:

```bash
First, read the src/performance_eval.py, results/cached_runs/apr9/performance_results.csv, and the results/cached_runs/apr9/interpretability_vs_performance.png plot. Include the plot then discuss how different variations effected performance/interpretability and why.

Next, read the tests in src/interp_eval.py and the results in results/cached_runs/apr9/interpretability_results.csv. Write a concise report titled results/cached_runs/apr9/results_report.md. It should describe the interp_eval tests and show a detailed example of how a single test is conducted. Then, it should show a table where each row is a test, column gives a short description, next column gives a detailed description, next column gives pass rate, and final columns show which models passed.

Finally, the report should end with a string visualization of three different model types (e.g. decision tree, linear model, random forest) when fit to the synthetic data from the first interpretability test with some text discussion.
```

## Running the agent

Spin up Claude Code (or any LLM agent) in this repo and prompt:

```
Hi have a look at program.md and let's kick off a new experiment!
```

The `program.md` file is the lightweight "skill" that instructs the agent.

## Project structure

```
run_baselines.py  — baseline evaluation: interpretability tests + regression datasets RMSE (do not modify)
interpretable_regressor.py          — regressor definition (agent modifies this)
program.md        — agent instructions
pyproject.toml    — dependencies
results/          — output plots, CSVs, and scores
  overall_results.csv — mean_rank + frac_interpretability_tests_passed per model
src/             — supporting modules (interpretability tests, performance eval)
```

## Design choices

- **Single file to modify.** The agent only touches `interpretable_regressor.py`. Diffs are small and reviewable.
- **Fixed dataset split.** Train/test splits are fixed (seed=42, 80/20), so experiments are comparable regardless of what the agent changes.
- **Two metrics.** `mean_rank` measures predictive performance; `frac_interpretability_tests_passed` measures how well the model communicates its behavior to an LLM. Both are tracked — neither is a hard constraint.
