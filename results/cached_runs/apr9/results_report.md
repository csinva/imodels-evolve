# apr9 Experiment Results Report

## 1. Interpretability Evaluation Overview

The interpretability evaluation (`src/interp_eval.py`) tests whether an LLM (GPT-4o) can extract meaningful, actionable information from a model's string representation. Each test:

1. **Generates synthetic data** with a known ground-truth relationship (e.g., `y = 10*x0 + noise`)
2. **Fits the model** being evaluated on this synthetic data
3. **Converts the fitted model to a string** using `get_model_str()` (e.g., showing coefficients, tree structure, or partial effects)
4. **Prompts GPT-4o** with the model string and a question about the model's behavior
5. **Checks the LLM's answer** against the ground truth with a tolerance

If GPT-4o can answer correctly, the model's representation is considered interpretable for that task.

### Detailed Example: `test_point_prediction`

This test evaluates whether an LLM can compute a specific prediction by reading the model string.

**Step 1 -- Generate data:**
```python
# y = 5.0 * x0 + noise, with 3 features
rng = np.random.RandomState(0)
X = rng.randn(300, 3)
y = 5.0 * X[:, 0] + rng.randn(300) * 0.5
```

**Step 2 -- Fit model and get string:** For a linear regression, `get_model_str()` produces:
```
OLS Linear Regression:  y = 4.9876*x0 + 0.0490*x1 + 0.0311*x2 + 0.0122
```

**Step 3 -- Prompt GPT-4o:**
> "Here is a trained regression model: [model string above]
> What does this model predict for the input x0=2.0, x1=0.0, x2=0.0?
> Answer with just a single number."

**Step 4 -- Check answer:**
- True prediction: `model.predict([[2.0, 0.0, 0.0]])` = 9.987
- GPT-4o response: "9.99"
- Tolerance: `max(|true| * 0.25, 1.5)` = 2.50
- `|9.99 - 9.987| = 0.003 < 2.50` --> **PASS**

For an MLP, the model string shows a weight matrix, making it nearly impossible for the LLM to trace the forward pass --> **FAIL**.

---

## 2. Test Descriptions and Pass Rates

The 43 core tests span 5 suites. The table below shows results across the 17 baseline models (177 total models were tested, but the table focuses on baselines).

### Standard Tests (8)

| Test | Short Description | Detailed Description | Pass Rate | Models Passed |
|------|-------------------|---------------------|-----------|---------------|
| most_important_feature | Identify top feature | Fit on `y=10*x0+noise` (5 features), ask which single feature matters most | 98% (172/176) | All except TabPFN, EBM, NeuralNAM_v1, AdaptiveTreeGAM_v1 |
| point_prediction | Predict a value | Fit on `y=5*x0+noise` (3 features), ask for prediction at x0=2.0 | 95% (167/176) | Most models; failures: PyGAM, RF, MLP, GBM, TabPFN, EBM, some HingeEBM |
| direction_of_change | Quantify change | Fit on `y=8*x0+noise` (4 features), ask how prediction changes when x0: 0->1 | 90% (158/176) | Most additive/tree models; failures: DT_large, MLP, TabPFN, EBM, NeuralNAM |
| feature_ranking | Rank features | Fit on `y=5*x0+3*x1+1.5*x2` (5 features), ask for top-3 ranking | 98% (174/177) | Nearly all; failures: MLP, EBM, TabPFN |
| threshold_identification | Find decision boundary | Fit on step function at x0=0.5, ask for threshold value | 86% (152/177) | Most; failures: DT_large, LassoCV, MLP, TabPFN, EBM, some evolved |
| irrelevant_features | Identify noise features | Fit on `y=10*x0+noise` (5 features), ask which features don't matter | 89% (158/177) | Most additive; failures: DT_mini, DT_large, RF, MLP, TabPFN, EBM |
| sign_of_effect | Direction and magnitude | Fit on `y=5*x0-5*x1+noise`, ask effect of +1 unit of x1 | 77% (135/176) | Linear, tree, SmoothGAM; failures: OLS variants sometimes, RF, MLP, TabPFN, EBM |
| counterfactual_prediction | Predict at new point | Given prediction at x0=1, predict at x0=3 | 86% (151/176) | Most additive/tree; failures: PyGAM, MLP, TabPFN, EBM |

### Hard Tests (5)

| Test | Short Description | Detailed Description | Pass Rate | Models Passed |
|------|-------------------|---------------------|-----------|---------------|
| hard_all_features_active | 3-feature prediction | All 3 features active at non-zero values, predict `y` at x0=1.7, x1=0.8, x2=-0.5 | 86% (151/175) | Most; failures: PyGAM, MLP, TabPFN, EBM, some evolved |
| hard_pairwise_anti_intuitive | Compare two samples | Predict difference between sample A (x0=2.0,x1=0.1) and B (x0=0.5,x1=3.3) | 3% (5/176) | Only CompactAdditive_v1, CyclicAdditive_v1, GBM, HSTree_large, HSTree_mini |
| hard_quantitative_sensitivity | Sensitivity over range | Predict change when x0 goes from 0.5 to 2.5 | 91% (161/176) | Nearly all; failures: MLP, TabPFN, EBM, NeuralNAM |
| hard_mixed_sign_goes_negative | Negative prediction | Predict at x0=1.0, x1=2.5, x2=1.0 where y < 0 due to negative coefficient | 69% (120/175) | SmoothGAM, HingeGBM, linear; failures: OLS, RF, MLP, TabPFN, EBM |
| hard_two_feature_perturbation | Two features change | Given baseline prediction, predict when x0->2.0 and x1->1.5 simultaneously | 82% (144/176) | Most additive; failures: MLP, TabPFN, EBM, some evolved |

### Insight Tests (6)

| Test | Short Description | Detailed Description | Pass Rate | Models Passed |
|------|-------------------|---------------------|-----------|---------------|
| insight_simulatability | 4-feature prediction | Predict at x0=1.0, x1=2.0, x2=0.5, x3=-0.5 on linear data | 24% (43/177) | Linear, some SmoothGAM; failures: most trees, PyGAM, RF, MLP, TabPFN, EBM |
| insight_sparse_feature_set | Identify active features | 10 features, only x0 and x1 active; list meaningful features | 90% (160/177) | Nearly all; failures: MLP, TabPFN, EBM, NeuralNAM |
| insight_nonlinear_threshold | Hockey-stick threshold | Identify that x0<0 has no effect (ReLU-like function) | 74% (131/177) | Tree, GAM, SmoothGAM; failures: OLS, RF, MLP, TabPFN, EBM |
| insight_nonlinear_direction | Predict on hockey-stick | Predict at x0=2.0 on `y=3*max(0,x0)` data | 64% (114/177) | SmoothGAM, linear, some trees; failures: DT_large, RF, MLP, TabPFN, EBM |
| insight_counterfactual_target | Inverse prediction | Find x0 value that yields a target prediction (inverse problem) | 16% (28/177) | OLS, LassoCV, some SmoothGAM; failures: most trees, RF, MLP, TabPFN, EBM |
| insight_decision_region | Decision boundary | Find x0 threshold for predictions above 6.0 | 92% (163/177) | Nearly all; failures: DT_large, MLP, TabPFN, EBM |

### Discrimination Tests (10)

| Test | Short Description | Detailed Description | Pass Rate | Models Passed |
|------|-------------------|---------------------|-----------|---------------|
| discrim_simulate_all_active | 5-feature complex sample | Predict at non-round values with 5 active features | 21% (37/175) | Linear, some SmoothGAM, DT_mini; failures: PyGAM, RF, MLP, TabPFN, EBM |
| discrim_compactness | Model size check | Ask if model is computable in <=10 operations | 83% (147/177) | Most additive/tree; failures: DT_large, RF, GBM, MLP, TabPFN, EBM |
| discrim_dominant_feature_sample | Identify dominant feature | Which feature contributes most to a specific sample (x0 dominates) | 100% (177/177) | All models |
| discrim_unit_sensitivity | Exact unit change | Exact prediction change when x0: 0->1 (tight 10% tolerance) | 87% (153/176) | Linear, SmoothGAM, most trees; failures: MLP, TabPFN, EBM |
| discrim_predict_above_threshold | Predict above threshold | Predict at x0=2.0 on step-function data (above threshold) | 77% (129/167) | Most; failures: MLP, TabPFN, EBM, NeuralNAM |
| discrim_predict_below_threshold | Predict below threshold | Predict at x0=-0.5 on step-function data (below threshold) | 77% (129/167) | Most; failures: MLP, TabPFN, EBM, NeuralNAM |
| discrim_simulate_mixed_sign | 6-feature mixed signs | Predict with 6 features having mixed +/- coefficients | 75% (132/175) | SmoothGAM, linear, DT_mini; failures: PyGAM, RF, MLP, TabPFN, EBM |
| discrim_simulate_double_threshold | Two-step prediction | Predict on data with two step thresholds (3 output levels) | 69% (120/175) | Tree, SmoothGAM; failures: OLS, MLP, TabPFN, EBM |
| discrim_simulate_additive_nonlinear | Nonlinear additive | Predict on `y=3*max(0,x0)+2*sin(x1)+x2` | 62% (109/175) | SmoothGAM, DT, linear approx; failures: RF, MLP, TabPFN, EBM |
| discrim_simulate_interaction | Interaction prediction | Predict on `y=3*x0+2*x1+1.5*x0*x1` | 54% (94/175) | DT, linear, some SmoothGAM; failures: PyGAM, RF, MLP, TabPFN, EBM |

### Simulatability Tests (14)

| Test | Short Description | Detailed Description | Pass Rate | Models Passed |
|------|-------------------|---------------------|-----------|---------------|
| simulatability_eight_features | 8-feature mixed data | Mixed +/- coefficients, predict at specific point | 61% (108/177) | SmoothGAM, linear, some trees; failures: RF, MLP, TabPFN, EBM |
| simulatability_fifteen_features_sparse | 15-feature sparse | Only 3 of 15 features active, predict at specific point | 4% (7/177) | DT_mini, OLS, LassoCV, a few SmoothGAM |
| simulatability_quadratic | Quadratic nonlinearity | `y=3*x0^2 - 2*x1^2 + x2`, predict at point | 46% (81/177) | SmoothGAM, OLS (approx), some trees; failures: RF, MLP, TabPFN, EBM |
| simulatability_triple_interaction | Multi-way interaction | `y=2*x0*x1+3*x1*x2+x0*x2*x3`, predict | 76% (134/177) | Most SmoothGAM, linear (approx), DT_large; failures: RF, MLP, TabPFN, EBM |
| simulatability_friedman1 | Friedman #1 benchmark | `y=10*sin(pi*x0*x1)+20*(x2-0.5)^2+10*x3+5*x4`, 10 features | 73% (129/177) | SmoothGAM, OLS (approx), some trees; failures: RF, MLP, TabPFN, EBM |
| simulatability_cascading_threshold | Cascading if-else | `if x0>0: y~3*x1 else y~-2*x2`, predict | 75% (133/177) | SmoothGAM, DT, linear; failures: RF, MLP, TabPFN, EBM |
| simulatability_quadratic_counterfactual | Quadratic change | How much does prediction change on quadratic data when x0: 0->2 | 51% (91/177) (approx) | Linear (exact calc), SmoothGAM; failures: RF, MLP, TabPFN, EBM |
| simulatability_exponential_decay | Exponential + linear | `y=5*exp(-x0)+2*x1`, predict | 21% (38/177) | DT_mini, OLS, a few SmoothGAM; failures: most models |
| simulatability_piecewise_three_segment | Piecewise linear | 3-segment piecewise in x0, predict | 88% (156/177) | Most; failures: MLP, TabPFN, EBM, NeuralNAM |
| simulatability_twenty_features_sparse | 20-feature sparse | 4 of 20 features active, predict | 22% (39/177) | DT_mini, DT_large, OLS, a few SmoothGAM; failures: most |
| simulatability_sinusoidal | Trig nonlinearity | `y=4*sin(x0)+2*cos(x1)+x2`, predict | 20% (35/177) | DT_mini, OLS, a few SmoothGAM; failures: most |
| simulatability_abs_value | V-shaped nonlinearity | `y=3*|x0|-2*|x1|+x2`, predict | 55% (97/177) | SmoothGAM, OLS (approx), some DT; failures: RF, MLP, TabPFN, EBM |
| simulatability_twelve_features_all_active | 12 features all active | 12 features with decreasing coefficients, predict | 12% (22/177) | PyGAM, OLS, a few SmoothGAM; failures: most models |
| simulatability_nested_threshold | Nested if-else | `if x0>0 and x1>0: y~5, elif x0>0: y~2, else y~-1`, predict | 51% (91/177) | DT, SmoothGAM, some linear; failures: RF, MLP, TabPFN, EBM |

---

## 3. Model String Visualizations

Below are the string representations of three model types when fit to the first interpretability test's synthetic data (`y = 10*x0 + noise`, 5 features, 300 samples). These strings are what GPT-4o reads to answer interpretability questions.

### Decision Tree (max_depth=3)

```
|--- x0 <= -0.07
|   |--- x0 <= -1.10
|   |   |--- x0 <= -1.87
|   |   |   |--- value: [-23.34]
|   |   |--- x0 >  -1.87
|   |   |   |--- value: [-14.23]
|   |--- x0 >  -1.10
|   |   |--- x0 <= -0.53
|   |   |   |--- value: [-7.96]
|   |   |--- x0 >  -0.53
|   |   |   |--- value: [-3.13]
|--- x0 >  -0.07
|   |--- x0 <= 1.16
|   |   |--- x0 <= 0.60
|   |   |   |--- value: [3.18]
|   |   |--- x0 >  0.60
|   |   |   |--- value: [8.33]
|   |--- x0 >  1.16
|   |   |--- x0 <= 1.80
|   |   |   |--- value: [15.12]
|   |   |--- x0 >  1.80
|   |   |   |--- value: [20.61]
```

**Interpretability analysis:** The tree structure makes it immediately clear that only `x0` is used for splitting (features x1-x4 are absent). An LLM can trace any input by following the branch conditions. For x0=2.0: `x0 > -0.07` -> `x0 > 1.16` -> `x0 > 1.80` -> prediction = **20.61**. The piecewise-constant nature means predictions are approximate but the reasoning is fully transparent. However, the tree only has 8 leaf nodes, so it discretizes the continuous relationship -- it cannot represent the exact slope of 10.0.

### Linear Regression (OLS)

```
OLS Linear Regression:  y = 9.9876*x0 + 0.0490*x1 + 0.0311*x2 + 0.0406*x3 + -0.0084*x4 + 0.0122

Coefficients:
  x0: 9.9876
  x1: 0.0490
  x2: 0.0311
  x3: 0.0406
  x4: -0.0084
  intercept: 0.0122
```

**Interpretability analysis:** The equation is fully readable. The coefficient for x0 (9.99) clearly dominates -- matching the true coefficient of 10.0. All other coefficients are near zero, making it obvious that x1-x4 are irrelevant. For any input, the LLM simply performs arithmetic: at x0=2.0, `y = 9.9876*2.0 + 0.0122 = 19.99`. This is the gold standard for interpretability: exact, closed-form, one-line computation. The weakness is that linear models cannot capture nonlinear patterns (thresholds, interactions), causing them to fail on more complex datasets.

### Random Forest (3 estimators, max_depth=2)

```
Random Forest Regressor -- Feature Importances:
  x0: 1.0000
  x1: 0.0000
  x2: 0.0000
  x3: 0.0000
  x4: 0.0000

Tree 0:                          Tree 1:                          Tree 2:
|--- x0 <= -0.07                 |--- x0 <= -0.09                 |--- x0 <= 0.02
|   |--- x0 <= -1.13             |   |--- x0 <= -1.02             |   |--- x0 <= -0.90
|   |   |--- value: [-15.86]     |   |   |--- value: [-15.52]     |   |   |--- value: [-13.33]
|   |--- x0 >  -1.13             |   |--- x0 >  -1.02             |   |--- x0 >  -0.90
|   |   |--- value: [-5.79]      |   |   |--- value: [-5.19]      |   |   |--- value: [-4.34]
|--- x0 >  -0.07                 |--- x0 >  -0.09                 |--- x0 >  0.02
|   |--- x0 <= 0.95              |   |--- x0 <= 0.90              |   |--- x0 <= 1.16
|   |   |--- value: [4.70]       |   |   |--- value: [3.74]       |   |   |--- value: [5.30]
|   |--- x0 >  0.95              |   |--- x0 >  0.90              |   |--- x0 >  1.16
|   |   |--- value: [15.78]      |   |   |--- value: [13.68]      |   |   |--- value: [18.36]
```

**Interpretability analysis:** Each individual tree is traceable, but the LLM must average predictions across all 3 trees. Feature importances correctly show x0 as the only relevant feature. For x0=2.0: Tree 0 predicts 15.78, Tree 1 predicts 13.68, Tree 2 predicts 18.36, average = **15.94**. This is less accurate than the linear model (true is ~20) because max_depth=2 limits resolution. With realistic forests (100+ trees), the model string becomes far too long for an LLM to trace, which is why RF scores only 23.3% on interpretability tests. The feature importances provide global insight, but per-sample predictions require tracing every tree.

### Key takeaway

The fundamental interpretability-performance tradeoff is visible in these representations:
- **Linear models** are perfectly readable but miss nonlinear patterns
- **Shallow trees** are traceable but discretize continuous relationships
- **Ensembles** (RF, GBM) improve predictions by averaging many models, but this makes the string representation intractable for an LLM to simulate

The evolved SmoothGAM family attempts to bridge this gap: additive structure (like a linear model) with smooth nonlinear components (like a GAM), keeping the representation readable while capturing more complex patterns.
