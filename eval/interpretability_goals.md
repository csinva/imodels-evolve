The goal is to evaluate the interpretability of an interpretable regressor. To do so, write synthetic tests, fit a model to them, then have an LLM try to recover something about them from the model string.

Example 1: Given a small synthetic dataset with only one important variable, fit the model, verify it's accuracy is above some threshold, then ask an LLM to guess which variable was the important one using the string representation.

Example 2: Take the fitted model from example 1. Guess the output prediction for a particular simple example without running any code.

Example 3: Given the prediction for a particular sample without running any code, guess the prediction when changing the most important feature by a small amount.

Brainstorm several more examples and write code for all of them. These tests should work with a scikit-learn style interface (e.g. DecisionTreeRegressor). The llm calls should use gpt-4o and be called through imodelsx.llm (see reference file at /home/chansingh/imodelsX/imodelsx/llm.py).

Finally, test these functions using a decision tree, a random forest, an OLS model, and an MLP (all from sklearn).



# Followup

Read these two papers: Interpreting Interpretability: Understanding Data Scientists' Use of Interpretability Tools for Machine Learning https://dl.acm.org/doi/abs/10.1145/3313831.3376219 and Interpretable machine learning: definitions, methods, and applications https://arxiv.org/abs/1901.04592. Then brainstorm and implement some more widespread tests. The tests should distinguish between interpretable models (like decision trees, GAMs, and sparse linear models) and black-box models (like MLPs and gradient-boosted decision trees). Ideally, the tests should also differentiate degrees of interpretability, e.g. sparser linear models should score higher than dense ones, shallow decision trees should score higher than deep ones, and GAMs should score high.

# Followup


Read and edit eval/interp_eval.py with the following changes:

Add more tests that can help distinguish between interpretable models (like decision trees, GAMs, and sparse linear models) and black-box models (like MLPs and gradient-boosted decision trees). Ideally, the tests should also differentiate degrees of interpretability, e.g. sparser linear models should score higher than dense ones, shallow decision trees should score higher than deep ones, and GAMs should score high.

Here are some examples:

Example 1: Can the model easily be summarized with less than 10 rules or arithmetic operations?

Example 2: After fitting on some complex data, try to simulate the model's prediction on a difficult sample without running code.