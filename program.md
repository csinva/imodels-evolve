# autoresearch — interpretable regressors

This is an experiment to have the LLM autonomously research scikit-learn regressors that score well on two metrics: predictive performance (mean AUC) and interpretability (fraction of LLM-graded tests passed).

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar20`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `readme.md` — repository context.
   - `run_baselines.py` — the fixed baseline evaluation harness. This is NOT modified by the agent.
   - `model.py` — the file you modify. Regressor definition and evaluation loop.
   - `results/overall_results.csv` — current scores for all models including baselines.
4. **Initialize the results/overall_results.csv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
5. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Run an experiment with: `uv run model.py`

This trains `InterpretableRegressor`, runs interpretability tests, and updates `results/overall_results.csv`.

**What you CAN do:**
- Modify `model.py` — this is the only file you edit. Everything is fair game:
  - The `InterpretableRegressor` class definition (algorithm, structure, hyperparameters)
  - Switching to another model type (rule lists, linear models, GAMs, sparse models, etc.)
  - Feature engineering or preprocessing inside the regressor
  - Hyperparameter values
  - The training loop structure

**What you CANNOT do:**
- Modify `run_baselines.py`. It is read-only.
- Modify anything in the `eval/` folder. It contains the ground truth tests.
- Install new packages. You can only use what's already in `pyproject.toml`.

## Goal

Maximize both metrics in `results/overall_results.csv`:

- **`mean_auc`** — mean AUC across TabArena classification datasets (higher is better)
- **`frac_interpretability_tests_passed`** — fraction of LLM-graded interpretability tests passed (higher is better)

Both metrics matter. A model that scores well on AUC but poorly on interpretability tests, or vice versa, is not ideal. Look at the baseline scores in `overall_results.csv` to understand the trade-off space.

**Simplicity criterion**: All else being equal, simpler is better. A tiny metric gain that doubles code complexity is not worth it.

**The first run**: Your very first run should always be to establish the baseline — run the script as is, record the results.

## Output format

Once the script finishes it prints a summary like this:

```
---
tests_passed:  5/18 (27.78%)  [std 0/8  hard 0/5  insight 0/5]
total_seconds: 0.8s
```

It also updates `results/overall_results.csv` with the row for `InterpretableRegressor`.

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 4 columns:

```
commit	mean_auc	frac_interpretability_tests_passed	status	description
```

1. git commit hash (short, 7 chars)
2. mean_auc achieved — use 0.000000 for crashes (note: model.py does not compute mean_auc; check overall_results.csv for the baseline value to compare against)
3. frac_interpretability_tests_passed — from the script output
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar20`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Edit `model.py` with an experimental idea
3. git commit
4. Run the experiment: `uv run model.py > run.log 2>&1`
5. Read results: `tail -n 5 run.log` and `grep InterpretableRegressor results/overall_results.csv`
6. If the run crashed, check `tail -n 50 run.log` for the stack trace and attempt a fix
7. Record results in `results.tsv` (do not commit this file)
8. If either metric improved without the other getting significantly worse, keep the commit
9. Otherwise, `git reset --hard` back to the previous commit

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. Run until manually stopped.

**Ideas to try** (not exhaustive — be creative):
- Tune depth, leaf size, splitting criteria
- Try sparse linear models (LASSO, ElasticNet) with feature selection
- Try rule-based models (OneR, RIPPER, RuleFit)
- Try Generalized Additive Models (pygam)
- Try imodels: FIGS, HSTree, TreeGAM
- Feature preprocessing: binning, interaction terms, polynomial features
- Per-dataset hyperparameter tuning via cross-validation
- Combining simple models in an interpretable ensemble
