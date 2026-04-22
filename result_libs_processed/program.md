This repo contains many different tabular regression models that have been generated using an agentic loop. See `../paper-imodels-agentic/content.tex` for high-level context and `../evolve/program.md` for precisely what was run. The results have been stored into `../result_libs`.

## Your task

Your task is to analyze the results stored in `../result_libs` and build a usable python library of the best generated tabular models.

## Analyzing results

Your first step should be to write and run a script to write a `combined_results.csv` file into `../result_libs` that contains all models along with their:

- Development set predictive performance & interpretability score (from `overall_results.csv`)
  - Make sure the predictive performance is computed by averaging over ranks globally across all models, like is done in fig 3 of `../paper-imodels-agentic/content.tex`.
- Held-out set interpretability score (from `overall_results_test.csv`)
- Interpretability scores broken down by the 6 categories for both the development set (derived from `interpretability_results.csv`) and the held-out set (from `interpretability_results_test.csv`)
- Include a link to each python file containing the model implementation
- Do not include the "status" column and drop any rows with missing mean_rank_global values (due to models that failed to run or did not produce valid results)

## Selecting models

Using the newly built `combined_results.csv`, select a set of top-performing models based on a combination of predictive performance and interpretability scores.

- First write code to filter to models achieving good tradeoffs (ideally being pareto optimal over the baselines, but do not be too strict in this step)
- Then, read through and select diverse models that cover different trade-offs between these two metrics and that use different ideas (do not include models that are extremely similar variations of each other).
- Aim for about 10 finally selected models

### Library specifications

Now, build the library here in a folder named `agentic-imodels`.

- The library should follow the general structure of [this repo https://github.com/csinva/imodels](https://github.com/csinva/imodels)
  - Include the readme, the uv setup, and the source structure
  - There is no need to write notebooks/test/CI.
- Additionally, write a `SKILL.md` file to be used by an agent. This file should summarize the performance of the generated tabular regression models and describe exactly how to use the library.
- Within the library create a `results` folder that contains the `combined_results.csv` with only the results for the selected models, along with an interpretability_vs_performance_test.png plot which shows the interpretability (for the held-out set) vs the performance
  - Don't invert y-axes, it's okay for lower to be better
