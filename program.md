# autoresearch — interpretable classifiers

This is an experiment to have the LLM autonomously research interpretable scikit-learn classifiers.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar20`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `model.py` — the single file you edit edits. Defines `InterpretableRegressor` (a scikit-learn compatible model), a `model_factory`, and a training + evaluation loop. Everything is fair game: algorithm, hyperparameters, feature engineering, etc. 
   - `model.py` — the file you modify. Classifier definition, hyperparameters.
4. **Verify data exists**: Check that `~/.cache/imodels-evolve/` contains cached dataset parquet files. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs in the fixed time budget. You launch it as: `uv run train.py`.

**What you CAN do:**
- Modify `model.py` — this is the only file you edit. Everything is fair game:
  - The `InterpretableClassifier` class definition (algorithm, structure, hyperparameters)
  - Switching from a decision tree to another interpretable model (rule lists, logistic regression, GA2M, sparse linear models, etc.)
  - Feature engineering or preprocessing inside the classifier
  - Hyperparameter values (`MAX_DEPTH`, `MIN_SAMPLES_LEAF`, `CRITERION`, etc.)
  - The `model_factory()` function
  - The training loop structure (cross-validation, ensembling, etc.)

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation harness (`evaluate_auc`), dataset list, and preprocessing.
- Install new packages. You can only use what's already in `pyproject.toml`.
- Modify the `evaluate_auc` function. It is the ground truth metric.

**The goal is simple: maximize mean_auc across all TabArena classification datasets.**

**Interpretability constraint**: The classifier must remain interpretable. Acceptable models include:
- Decision trees and rule sets
- Sparse linear models (LASSO, logistic regression with few features)
- Generalized additive models (GAMs)
- Any model that a domain expert can inspect and understand

Do NOT use black-box models like random forests, gradient boosting, or neural networks. The spirit of this research is to find the best *interpretable* model.

**Simplicity criterion**: All else being equal, simpler is better. A 0.001 mean_auc gain that doubles code complexity is not worth it. A simplification that maintains AUC is a win.

**The first run**: Your very first run should always be to establish the baseline — run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
mean_auc:         0.847321
total_seconds:    42.3
max_depth:        4
min_samples_leaf: 10
criterion:        gini
num_datasets:     15
```

Extract the key metric with:

```
grep "^mean_auc:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	mean_auc	status	description
```

1. git commit hash (short, 7 chars)
2. mean_auc achieved (e.g. 0.847321) — use 0.000000 for crashes
3. status: `keep`, `discard`, or `crash`
4. short text description of what this experiment tried

Example:

```
commit	mean_auc	status	description
a1b2c3d	0.847321	keep	baseline: decision tree depth=4
b2c3d4e	0.851200	keep	increase depth to 6
c3d4e5f	0.840000	discard	switch to depth=2 (too shallow)
d4e5f6g	0.000000	crash	CART with custom splitter (bug)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar12`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^mean_auc:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If mean_auc improved (higher), you "advance" the branch, keeping the git commit
9. If mean_auc is equal or worse, you git reset back to where you started

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human might be asleep. You are autonomous. Run until manually stopped.

**Crashes**: If a run crashes, use your judgment. Easy typos — fix and re-run. Fundamentally broken idea — log "crash" and move on.

**Ideas to try** (not exhaustive — be creative):
- Tune max_depth, min_samples_leaf, criterion
- Try CART with different splitting criteria
- Try sparse logistic regression (L1-penalized) with interpretable feature selection
- Try rule-based classifiers (OneR, RIPPER/IREP, CN2)
- Try optimal decision trees (GOSDT, DL8.5)
- Try Generalized Additive Models (pygam, interpret)
- Feature preprocessing: binning, interaction terms, polynomial features
- Per-dataset hyperparameter tuning (cross-validate within fit)
- Calibration for better probability estimates
- Combining simple models with a meta-learner
