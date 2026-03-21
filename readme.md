# imodels-evolve

Autonomous AI research on interpretable scikit-learn classifiers.

The idea: give an AI agent a classification training setup and let it experiment autonomously. It modifies `model.py`, trains a classifier, checks if AUC improved across TabArena datasets, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better interpretable model.

## How it works

The repo has three files that matter:

- **`run_baselines.py`** — evaluates a fixed set of baseline regressors/classifiers across two dimensions: (1) interpretability tests graded by an LLM and (2) AUC on subsampled TabArena datasets. Writes results to `results/`. **Not modified by the agent.**
- **`model.py`** — the single file the agent edits. Defines `InterpretableClassifier` (a scikit-learn compatible model), a `model_factory`, and a training + evaluation loop. Everything is fair game: algorithm, hyperparameters, feature engineering, etc. **This file is edited and iterated on by the agent.**
- **`program.md`** — baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human.**

The metric is **mean AUC** (area under the ROC curve) across all TabArena classification datasets — higher is better, and evaluated on held-out test sets so architectural changes are fairly compared.

**Interpretability constraint**: models must remain human-interpretable. Decision trees, rule lists, sparse linear models, and GAMs are all fair game. Black-box ensembles are not.

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
model.py          — interpretable classifier definition (agent modifies this)
program.md        — agent instructions
pyproject.toml    — dependencies
results/          — output plots, CSVs, and scores from run_baselines.py
baselines/        — supporting modules (interpretability tests, performance eval)
```

## Design choices

- **Single file to modify.** The agent only touches `model.py`. Diffs are small and reviewable.
- **Fixed dataset split.** Train/test splits are fixed (seed=42, 80/20), so experiments are comparable regardless of what the agent changes.
- **Interpretability as a hard constraint.** The agent is instructed not to use black-box models. This keeps the research focused on the goal: finding the best *explainable* classifier, not just the most accurate one.
- **AUC as metric.** Mean one-vs-rest AUC across datasets handles both binary and multi-class problems uniformly.
