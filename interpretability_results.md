# Interpretability Test Results

Tests use `gpt-4o` (via `imodelsx.llm`) to probe whether a model's string representation
conveys meaningful, usable information. Ground truth comes from known synthetic data.

---

## Complete Results — All 18 Tests × 12 Models

All models evaluated on all three test suites. Regressors used for interpretability tests;
classifier counterparts evaluated on TabArena. Plot: `interpretability_vs_performance.png`.

### Test suites

| Suite | Tests (n) | What they measure |
|-------|:---------:|-------------------|
| **Standard** | 8 | Basic feature importance, direction of change, ranking, thresholds |
| **Hard** | 5 | Require arithmetic *through* the model (averaging trees, propagating layers) |
| **Insight** | 5 | Paper-grounded: simulatability, sparsity, modularity, counterfactuals, global effects |

### Full results table

| Model | Standard (8) | Hard (5) | Insight (5) | **Total (18)** | TabArena avg rank ↓ |
|-------|:---:|:---:|:---:|:---:|:---:|
| **OLS** | **8/8** | 3/5 | **4/5** | **15/18 (83%)** | 4.57 |
| **HSTree** | **8/8** | **4/5** | 3/5 | **15/18 (83%)** | 9.69 |
| **GBM** | 7/8 | **4/5** | 2/5 | 13/18 (72%) | **2.50** |
| **DT_shallow** | 6/8 | **4/5** | 3/5 | 13/18 (72%) | 8.50 |
| **DT_deep** | 7/8 | 3/5 | 2/5 | 12/18 (67%) | 6.21 |
| **LASSO** | **8/8** | 2/5 | 2/5 | 12/18 (67%) | 5.21 |
| **FIGS** | 7/8 | 2/5 | 3/5 | 12/18 (67%) | 6.86 |
| **RuleFit** | 6/8 | 3/5 | 3/5 | 12/18 (67%) | 5.77* |
| **GAM** | 5/8 | 2/5 | 2/5 | 9/18 (50%) | — |
| **RF** | 6/8 | 1/5 | 1/5 | 8/18 (44%) | 3.71 |
| **MLP** | 6/8 | 0/5 | 1/5 | 7/18 (39%) | 6.21 |
| **TreeGAM** | 3/8 | 0/5 | 1/5 | 4/18 (22%) | 6.15* |

*RuleFit and TreeGAM failed on `volkert` (10-class); ranks computed over 13/14 datasets.
HSTree also failed on `volkert`; rank computed over 13/14 datasets.

---

## Standard Tests (8)

| Test | What the LLM is asked |
|------|-----------------------|
| most_important_feature | Which single feature drives the output? |
| point_prediction | What is the model's output for a specific input? |
| direction_of_change | If x0 goes 0→1, does the prediction increase or decrease? |
| feature_ranking | Rank the top 3 features by importance |
| threshold_identification | What x0 threshold separates low/high predictions? |
| irrelevant_features | Which features have little/no effect? |
| sign_of_effect | Does increasing x1 increase or decrease the output? (negative coef) |
| counterfactual_prediction | Given pred at x0=1, what is pred at x0=3? |

| Test | GAM | DT_sh | DT_dp | OLS | LASSO | RF | GBM | MLP | FIGS | RuleFit | HSTree | TreeGAM |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| most_important_feature | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| point_prediction | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ |
| direction_of_change | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| feature_ranking | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| threshold_identification | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ |
| irrelevant_features | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ |
| sign_of_effect | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ | ❌ |
| counterfactual_prediction | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| **Score** | **5/8** | **6/8** | **7/8** | **8/8** | **8/8** | **6/8** | **7/8** | **6/8** | **7/8** | **6/8** | **8/8** | **3/8** |

---

## Hard Tests (5)

Require actual arithmetic *through* the model — averaging 50 trees or propagating layers is impractical without code.

| Test | GAM | DT_sh | DT_dp | OLS | LASSO | RF | GBM | MLP | FIGS | RuleFit | HSTree | TreeGAM |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| hard_all_features_active | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ |
| hard_pairwise_anti_intuitive | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ |
| hard_quantitative_sensitivity | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ |
| hard_mixed_sign_goes_negative | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| hard_two_feature_perturbation | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| **Score** | **2/5** | **4/5** | **3/5** | **3/5** | **2/5** | **1/5** | **4/5** | **0/5** | **2/5** | **3/5** | **4/5** | **0/5** |

---

## Insight Tests (5) — grounded in interpretability literature

| Test | GAM | DT_sh | DT_dp | LASSO | OLS | RF | GBM | MLP | FIGS | RuleFit | HSTree | TreeGAM |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| insight_simulatability | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ |
| insight_sparse_feature_set | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ |
| insight_nonlinear_shape | ❌ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ | ❌ |
| insight_counterfactual_target | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| insight_decision_region | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ |
| **Score** | **2/5** | **3/5** | **2/5** | **2/5** | **4/5** | **1/5** | **2/5** | **1/5** | **3/5** | **3/5** | **3/5** | **1/5** |

---

## TabArena Classification Performance

Classifier counterparts evaluated on 14 TabArena datasets. Performance rank per dataset averaged across datasets (lower = better).

*Note: RuleFit, HSTree, TreeGAM do not support multiclass classification (volkert, 10 classes). Ranks for these models computed over 13/14 datasets.*

| Model | Avg Rank ↓ | Mean AUC | Group |
|-------|:---:|:---:|-------|
| **GBM** | **2.50** | **0.844** | Black-box |
| **RF** | 3.71 | 0.834 | Black-box |
| **OLS** | 4.57 | 0.814 | Linear |
| LASSO | 5.21 | 0.814 | Linear |
| RuleFit* | 5.77 | 0.816 | imodels |
| TreeGAM* | 6.15 | 0.814 | imodels |
| DT_deep | 6.21 | 0.801 | Tree |
| MLP | 6.21 | 0.756 | Black-box |
| FIGS | 6.86 | 0.796 | imodels |
| DT_shallow | 8.50 | 0.755 | Tree |
| HSTree* | 9.69 | 0.710 | imodels |

---

## Key Findings

### 1. No monotonic interpretability–performance trade-off

The plot reveals a nuanced picture rather than a clean trade-off:
- **Upper-right** (high interpretability, good performance): **OLS** (83%, rank 4.6) — the sweet spot
- **Upper-left** (high interpretability, poor performance): **HSTree** (83%, rank 9.7) and **DT_shallow** (72%, rank 8.5)
- **Lower-right** (low interpretability, good performance): **GBM** (72%, rank 2.5) and **RF** (44%, rank 3.7)
- **Lower-center** (low interpretability, middling performance): **MLP** (39%, rank 6.2), **TreeGAM** (22%, rank 6.2)

### 2. GBM anomaly: a "black box" that reads well

GBM scores 13/18 (72%) — higher than many explicitly interpretable models. This is because:
- Its string includes feature importances (enough for standard tests)
- Its first estimator tree is shown (enough for some hard tests: the first tree captures most signal)
- The LLM exploits these two representations effectively

However GBM fails `hard_mixed_sign_goes_negative` and `hard_quantitative_sensitivity` — tasks that require tracing the full ensemble.

### 3. HSTree achieves 15/18 interpretability — matching OLS

HSTree (hierarchical shrinkage decision tree) scores 8/8 standard, 4/5 hard, 3/5 insight = 15/18, matching OLS. This is because:
- Its string representation **is a decision tree** (exact same format as DT), just with shrinkage-smoothed leaf values
- The LLM can trace paths directly, just as with a regular tree
- This makes it the only model that combines tree-style readability with the numeric precision that hard tests require

However, HSTree's **classification performance is poor** (avg rank 9.7) — the best performance–interpretability combination comes from OLS and RuleFit.

### 4. OLS remains the best balanced model

OLS: 8/8 standard, 3/5 hard, 4/5 insight, TabArena rank 4.6. The explicit linear equation enables:
- Direct arithmetic (hard tests)
- Counterfactual solving (insight tests)
- Rule extraction (decision region)

Only `hard_pairwise_anti_intuitive` and `hard_mixed_sign_goes_negative` fail — both are LLM reasoning failures, not model transparency failures.

### 5. Black-box collapse on hard tests

| Model | Standard | Hard | Δ |
|-------|:---:|:---:|:---:|
| MLP | 6/8 (75%) | 0/5 (0%) | −75pp |
| RF | 6/8 (75%) | 1/5 (20%) | −55pp |
| GBM | 7/8 (88%) | 4/5 (80%) | −8pp |

MLP has the most dramatic collapse. GBM is the exception — its first tree provides enough signal for many hard tasks.

### 6. `hard_mixed_sign_goes_negative` fails universally (1/12)

Only GBM passes this test (a near-miss: the tree happens to capture the right region). Every other model fails. The LLM is fundamentally biased toward predicting positive outputs when all inputs are positive, regardless of what the negative coefficient or negative leaf value shows. This is an LLM reasoning failure.

### 7. imodels models cluster at 67% interpretability

FIGS, RuleFit (67% each) and HSTree (83%) from imodels all score well. TreeGAM (22%) is the outlier:
- TreeGAM's additive structure (one tree per feature) is theoretically highly modular
- But the LLM fails `insight_sparse_feature_set` — it lists all 10 features as important because each feature has a tree in the model, even if that tree has near-zero predictions
- And fails most hard/standard tests because the model's predictions are near 0 on simple synthetic data (too many boosting rounds with small max_leaf_nodes)

### 8. Multiclass support gap in imodels

RuleFit, HSTree, and TreeGAM classifiers do not support multiclass classification (`volkert`, 10 classes), limiting their applicability to binary tasks. This is a practical disadvantage vs sklearn models and FIGS.
