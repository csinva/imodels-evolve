# Background

The goal is to evaluate how well different classes in the `interpretable_regressors_lib` folder perform on interpretability metrics. The original results are stored in the `original_results` folder, particularly the `original_results/overall_results.csv` file.

The original results were obtained by running the `../evolve/interpretable_regressor.py` script. That script makes calls to `../evolve/src/performance_eval.py` and `../evolve/src/interp_eval.py` to compute the performance and interpretability metrics, respectively.

# Your task

Your task is to evaluate all of the classes in the `interpretable_regressors_lib` folder on the same interpretability tests but use the llm checkpoint `gpt-5.4` instead of `gpt-4o`. Save the results in a new folder called `new_results_gpt_5_4`. To do so, you should modify the original `../evolve/src/interp_eval.py` script to allow passing a checkpoint (maintain the original default of `gpt-4o`), and then create a new script called `evaluate_new_generalization_gpt_5_4.py` that performs this evaluation. Include all of the original models from the `../evolve/run_baselines.py` script in your evaluation and all the new models from the `interpretable_regressors_lib` folder. Save youre results in the same format as the original results, using the `original_results/overall_results.csv` file as a template for how to format your new results. Finally, visualize the results and save the visualizations in the `new_results_gpt_5_4` folder. Also make a comparison of the results with the original results and include that in your visualizations.
