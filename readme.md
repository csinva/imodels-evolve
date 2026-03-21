# imodels-evolve

Autonomous AI research on interpretable scikit-learn regressors.

The idea: give an AI agent a training setup and let it experiment autonomously. It modifies `model.py`, trains a regressor, checks if both metrics improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model.

## How it works

The repo has three files that matter:

- **`run_baselines.py`** — evaluates a fixed set of baseline regressors/classifiers across two metrics: (1) `frac_interpretability_tests_passed` — LLM-graded interpretability tests, and (2) `mean_auc` — AUC on subsampled TabArena regression datasets. Results saved to `results/overall_results.csv`. **Not modified by the agent.**
- **`model.py`** — the single file the agent edits. Defines `InterpretableRegressor` (a scikit-learn compatible model) and an evaluation loop that runs the same metrics and updates `baselines/overall_results.csv`. **This file is edited and iterated on by the agent.**
- **`program.md`** — instructions for the agent. Point your agent here and let it go. **This file is edited and iterated on by the human.**

## Metrics

Two metrics are tracked in `results/overall_results.csv`:

- **`mean_auc`** — mean one-vs-rest AUC across TabArena classification datasets (higher is better, evaluated on held-out test sets)
- **`frac_interpretability_tests_passed`** — fraction of LLM-graded interpretability tests passed (higher is better)

## Quick start

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Run baseline evaluation (interpretability tests + TabArena AUC, ~2 min with caching)
uv run run_baselines.py

# 4. Manually run a single training experiment
uv run model.py
```

If the above commands all work, your setup is working and you can go into autonomous research mode.

## Running the agent

Spin up Claude Code (or any LLM agent) in this repo and prompt:

```
Hi have a look at program.md and let's kick off a new experiment!
```

The `program.md` file is the lightweight "skill" that instructs the agent.

## Project structure

```
run_baselines.py  — baseline evaluation: interpretability tests + TabArena AUC (do not modify)
model.py          — regressor definition (agent modifies this)
program.md        — agent instructions
pyproject.toml    — dependencies
results/          — output plots, CSVs, and scores
  overall_results.csv — mean_auc + frac_interpretability_tests_passed per model
eval/             — supporting modules (interpretability tests, performance eval)
```

## Design choices

- **Single file to modify.** The agent only touches `model.py`. Diffs are small and reviewable.
- **Fixed dataset split.** Train/test splits are fixed (seed=42, 80/20), so experiments are comparable regardless of what the agent changes.
- **Two metrics.** `mean_auc` measures predictive performance; `frac_interpretability_tests_passed` measures how well the model communicates its behavior to an LLM. Both are tracked — neither is a hard constraint.
