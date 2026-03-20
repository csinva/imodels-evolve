# Interpretability Test Results

## Tests

| # | Test | What the LLM is asked |
|---|------|-----------------------|
| 1 | **most_important_feature** | Which single feature drives the output? |
| 2 | **point_prediction** | What is the model's output for a specific input? |
| 3 | **direction_of_change** | If x0 goes 0→1, does the prediction increase or decrease? |
| 4 | **feature_ranking** | Rank the top 3 features by importance |
| 5 | **threshold_identification** | What x0 threshold separates low/high predictions? |
| 6 | **irrelevant_features** | Which features have little/no effect? |
| 7 | **sign_of_effect** | Does increasing x1 increase or decrease the output? (negative coef) |
| 8 | **counterfactual_prediction** | Given pred at x0=1, what is pred at x0=3? |

## Results

| Test | DecisionTree | RandomForest | OLS | MLP |
|------|:---:|:---:|:---:|:---:|
| most_important_feature | ✅ | ✅ | ✅ | ✅ |
| point_prediction | ✅ | ❌ | ✅ | ❌ |
| direction_of_change | ✅ | ✅ | ✅ | ✅ |
| feature_ranking | ✅ | ✅ | ✅ | ✅ |
| threshold_identification | ✅ | ✅ | ✅ | ❌ |
| irrelevant_features | ❌ | ✅ | ✅ | ✅ |
| sign_of_effect | ✅ | ❌ | ✅ | ✅ |
| counterfactual_prediction | ✅ | ✅ | ✅ | ✅ |
| **Score** | **7/8** | **6/8** | **8/8** | **6/8** |

## Raw LLM Responses

### DecisionTree (7/8)

| Test | Ground truth | LLM response | Pass |
|------|-------------|--------------|------|
| most_important_feature | x0 | x0 | ✅ |
| point_prediction | 9.287 | 9.29 | ✅ |
| direction_of_change | increase | increase | ✅ |
| feature_ranking | x0, x1, x2 | x0, x1, x2 | ✅ |
| threshold_identification | 0.5 | 0.50 | ✅ |
| irrelevant_features | x1, x2, x3, x4 | "The features that appear to have little or no effect..." (no names extracted) | ❌ |
| sign_of_effect | decrease | decrease | ✅ |
| counterfactual_prediction | 9.1 | 9.10 | ✅ |

### RandomForest (6/8)

| Test | Ground truth | LLM response | Pass |
|------|-------------|--------------|------|
| most_important_feature | x0 | x0 | ✅ |
| point_prediction | 9.452 | 2.0 | ❌ |
| direction_of_change | increase | increase | ✅ |
| feature_ranking | x0, x1, x2 | x0, x1, x2 | ✅ |
| threshold_identification | 0.5 | 0.50 | ✅ |
| irrelevant_features | x1, x2, x3, x4 | x1, x2, x3, x4 | ✅ |
| sign_of_effect | decrease | increase | ❌ |
| counterfactual_prediction | 9.299 | 7.51 | ✅ |

### OLS (8/8)

| Test | Ground truth | LLM response | Pass |
|------|-------------|--------------|------|
| most_important_feature | x0 | x0 | ✅ |
| point_prediction | 10.061 | 10.0600 | ✅ |
| direction_of_change | increase | increase | ✅ |
| feature_ranking | x0, x1, x2 | x0, x1, x2 | ✅ |
| threshold_identification | 0.5 | 0.846 | ✅ |
| irrelevant_features | x1, x2, x3, x4 | "To determine which features have little or no effect..." | ✅ |
| sign_of_effect | decrease | decrease | ✅ |
| counterfactual_prediction | 11.996 | 12.00 | ✅ |

### MLP (6/8)

| Test | Ground truth | LLM response | Pass |
|------|-------------|--------------|------|
| most_important_feature | x0 | x0 | ✅ |
| point_prediction | 10.115 | "I cannot directly compute the output of the model with the given information." | ❌ |
| direction_of_change | increase | increase | ✅ |
| feature_ranking | x0, x1, x2 | x0, x1, x4 | ✅ |
| threshold_identification | 0.5 | 0. | ❌ |
| irrelevant_features | x1, x2, x3, x4 | "To determine which features have little or no effect..." | ✅ |
| sign_of_effect | decrease | decrease | ✅ |
| counterfactual_prediction | 11.624 | 12.21 | ✅ |
