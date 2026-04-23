# imodels-evolve (Codex edition)

Autonomous AI research on interpretable scikit-learn regressors, powered by OpenAI Codex CLI.

## How it works

The folder has three files that matter (symlinks to the `../evolve` folder):

- **`run_baselines.py`** — evaluates a fixed set of baseline regressors across two metrics: (1) `frac_interpretability_tests_passed` — LLM-graded interpretability tests, and (2) `mean_rank` — average rank of prediction performance on regression datasets. Results saved to `results/overall_results.csv`. **Not modified by the agent.**
- **`interpretable_regressor.py`** — the single file the agent edits. Defines `InterpretableRegressor` (a scikit-learn compatible model) and an evaluation loop that runs the same metrics and updates `results/overall_results.csv`. **This file is edited and iterated on by the agent.**
- **`program.md`** — instructions for the agent. The launch script feeds this to Codex.

## Quick start

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/), [Codex CLI](https://github.com/openai/codex).

```bash
# 1. Install dependencies
uv sync

# 2. Run baseline evaluation (first time only, ~2 min with caching)
cd evolve-codex && uv run run_baselines.py

# 3. Start the autonomous research loop
./run_loop.sh
```

## How the loop works

Unlike Claude Code (which runs an infinite loop inside a single session), Codex runs in non-interactive `exec` mode. The `run_loop.sh` script handles the outer loop:

1. Invokes `codex exec` with the prompt from `program.md`
2. Codex runs one experiment iteration (edit, commit, evaluate, record)
3. Codex applies its changes via `codex apply`
4. The script loops back to step 1

Each Codex invocation gets fresh context but reads `results/overall_results.csv` to see prior experiment results.

## Configuration

Edit `run_loop.sh` to change:

- `MAX_ITERATIONS` — how many experiments to run (default: 50)
- `MODEL` — which model to use (default: from ~/.codex/config.toml)
- Sandbox permissions and approval policy
