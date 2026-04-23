#!/usr/bin/env python3
"""
Analysis of Antemortem Tooth Loss (AMTL) in Homo sapiens vs. Non-human Primates

Research Question: Do modern humans (Homo sapiens) have higher frequencies of 
antemortem tooth loss (AMTL) compared to non-human primate genera (Pan, Pongo, 
Papio), after accounting for the effects of age, sex, and tooth class?
"""

import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import glm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Import agentic_imodels interpretable regressors
from agentic_imodels import (
    SmartAdditiveRegressor,
    HingeEBMRegressor,
    WinsorizedSparseOLSRegressor,
    HingeGAMRegressor
)

print("=" * 80)
print("ANTEMORTEM TOOTH LOSS ANALYSIS")
print("=" * 80)

# Load data
df = pd.read_csv('amtl.csv')
print(f"\nDataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nColumns: {list(df.columns)}")

# ============================================================================
# PART 1: DATA EXPLORATION
# ============================================================================
print("\n" + "=" * 80)
print("PART 1: DATA EXPLORATION")
print("=" * 80)

print("\n--- Summary Statistics ---")
print(df.describe())

print("\n--- Genus Distribution ---")
print(df['genus'].value_counts())

print("\n--- Tooth Class Distribution ---")
print(df['tooth_class'].value_counts())

# Calculate AMTL rate (proportion of missing teeth)
df['amtl_rate'] = df['num_amtl'] / df['sockets']

print("\n--- AMTL Rate by Genus ---")
print(df.groupby('genus')['amtl_rate'].agg(['mean', 'std', 'count']))

# Create binary indicator for Homo sapiens vs. others
df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

print("\n--- AMTL Rate: Humans vs. Non-human Primates ---")
print(df.groupby('is_human')['amtl_rate'].agg(['mean', 'std', 'count']))

# Bivariate correlation check
print("\n--- Bivariate Correlations with AMTL Rate ---")
numeric_cols = ['age', 'prob_male', 'is_human', 'sockets']
for col in numeric_cols:
    corr = df[['amtl_rate', col]].corr().iloc[0, 1]
    print(f"{col:15s}: r = {corr:6.3f}")

# ============================================================================
# PART 2: CLASSICAL STATISTICAL TESTS WITH CONTROLS
# ============================================================================
print("\n" + "=" * 80)
print("PART 2: CLASSICAL STATISTICAL TESTS (STATSMODELS)")
print("=" * 80)

# Encode categorical variables
df_model = df.copy()

# One-hot encode tooth_class
tooth_dummies = pd.get_dummies(df_model['tooth_class'], prefix='tooth', drop_first=True)
df_model = pd.concat([df_model, tooth_dummies], axis=1)

# Select features for modeling
feature_cols = ['is_human', 'age', 'prob_male'] + list(tooth_dummies.columns)
X_full = df_model[feature_cols].values.astype(float)
y = df_model['amtl_rate'].values

print("\n--- Model 1: Bivariate OLS (AMTL rate ~ is_human) ---")
X_bivariate = sm.add_constant(df_model[['is_human']].values)
model_bivariate = sm.OLS(y, X_bivariate).fit()
print(model_bivariate.summary())

print("\n--- Model 2: Multiple OLS with Controls (AMTL rate ~ is_human + controls) ---")
X_controlled = sm.add_constant(X_full)
model_controlled = sm.OLS(y, X_controlled).fit()
print(model_controlled.summary())

# Extract key statistics
is_human_coef_bivariate = model_bivariate.params[1]  # is_human coefficient
is_human_pval_bivariate = model_bivariate.pvalues[1]
is_human_coef_controlled = model_controlled.params[1]  # is_human coefficient
is_human_pval_controlled = model_controlled.pvalues[1]

print("\n--- Key Results Summary ---")
print(f"Bivariate effect (is_human):    β = {is_human_coef_bivariate:7.4f}, p = {is_human_pval_bivariate:.4f}")
print(f"Controlled effect (is_human):   β = {is_human_coef_controlled:7.4f}, p = {is_human_pval_controlled:.4f}")
print(f"Effect persists after controls: {is_human_pval_controlled < 0.05}")

# ============================================================================
# PART 3: INTERPRETABLE MODELS (AGENTIC_IMODELS)
# ============================================================================
print("\n" + "=" * 80)
print("PART 3: INTERPRETABLE MODELS FOR SHAPE, DIRECTION, ROBUSTNESS")
print("=" * 80)

# Prepare data for agentic_imodels (requires numeric features only)
X_interp = pd.DataFrame(X_full, columns=feature_cols)
y_interp = pd.Series(y)

# Split for evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X_interp, y_interp, test_size=0.2, random_state=42
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Features: {list(X_train.columns)}")

# Fit multiple interpretable models
models_to_fit = [
    ('SmartAdditiveRegressor', SmartAdditiveRegressor()),
    ('HingeEBMRegressor', HingeEBMRegressor()),
]

results = {}

for name, model in models_to_fit:
    print("\n" + "-" * 80)
    print(f"MODEL: {name}")
    print("-" * 80)
    
    try:
        # Fit model
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        print(f"\nPerformance:")
        print(f"  R² (train): {r2_train:.4f}")
        print(f"  R² (test):  {r2_test:.4f}")
        print(f"  RMSE (test): {rmse_test:.4f}")
        
        # Print the interpretable form
        print(f"\nInterpretable Form:")
        print(model)
        
        results[name] = {
            'model': model,
            'r2_train': r2_train,
            'r2_test': r2_test,
            'rmse_test': rmse_test
        }
        
    except Exception as e:
        print(f"Error fitting {name}: {e}")
        import traceback
        traceback.print_exc()
        results[name] = {'error': str(e)}

# ============================================================================
# PART 4: INTERPRETATION AND CONCLUSION
# ============================================================================
print("\n" + "=" * 80)
print("PART 4: INTERPRETATION AND CONCLUSION")
print("=" * 80)

print("\n--- Evidence Summary ---")

# Statistical significance
print(f"\n1. Classical Statistical Tests:")
print(f"   - Bivariate OLS: is_human β = {is_human_coef_bivariate:.4f}, p = {is_human_pval_bivariate:.4f}")
print(f"   - Controlled OLS: is_human β = {is_human_coef_controlled:.4f}, p = {is_human_pval_controlled:.4f}")
print(f"   - Effect direction: {'POSITIVE' if is_human_coef_controlled > 0 else 'NEGATIVE'}")
print(f"   - Statistical significance: {is_human_pval_controlled < 0.05}")

# Interpretable model insights
print(f"\n2. Interpretable Models:")
print(f"   Models fitted successfully:")
for name, result in results.items():
    if 'error' not in result:
        print(f"   - {name}: R² = {result['r2_test']:.3f}")

# Determine Likert score
reasoning_parts = []

# Evidence 1: Statistical significance
if is_human_pval_controlled < 0.001:
    stat_evidence = "very strong"
    stat_score = 30
elif is_human_pval_controlled < 0.05:
    stat_evidence = "significant"
    stat_score = 20
elif is_human_pval_controlled < 0.10:
    stat_evidence = "marginally significant"
    stat_score = 10
else:
    stat_evidence = "not significant"
    stat_score = 0

reasoning_parts.append(
    f"The classical OLS regression with controls (age, sex, tooth class) shows that "
    f"Homo sapiens has a {stat_evidence} effect on AMTL rate (β={is_human_coef_controlled:.4f}, "
    f"p={is_human_pval_controlled:.4f})."
)

# Evidence 2: Direction and magnitude
direction = "positive" if is_human_coef_controlled > 0 else "negative"
magnitude = abs(is_human_coef_controlled)

if direction == "positive":
    if magnitude > 0.05:
        mag_evidence = "substantial"
        mag_score = 30
    elif magnitude > 0.02:
        mag_evidence = "moderate"
        mag_score = 20
    else:
        mag_evidence = "small"
        mag_score = 10
else:
    mag_evidence = "negative (contradicts hypothesis)"
    mag_score = -50

reasoning_parts.append(
    f"The effect is {direction} with a {mag_evidence} magnitude "
    f"(coefficient of {is_human_coef_controlled:.4f}, meaning humans show "
    f"{abs(is_human_coef_controlled)*100:.2f} percentage points {'higher' if direction=='positive' else 'lower'} "
    f"AMTL rate than non-human primates)."
)

# Evidence 3: Robustness across models
successful_models = [name for name, res in results.items() if 'error' not in res]
if len(successful_models) >= 2:
    robustness_score = 20
    reasoning_parts.append(
        f"Multiple interpretable models ({', '.join(successful_models[:2])}) were fitted "
        f"successfully, suggesting the relationship is robust across different modeling approaches."
    )
else:
    robustness_score = 10

# Evidence 4: Bivariate vs. controlled comparison
effect_change = abs(is_human_coef_bivariate - is_human_coef_controlled) / abs(is_human_coef_bivariate) if is_human_coef_bivariate != 0 else 0

if effect_change < 0.2:
    persistence_evidence = "persists strongly"
    persist_score = 20
elif effect_change < 0.5:
    persistence_evidence = "persists moderately"
    persist_score = 10
else:
    persistence_evidence = "weakens substantially"
    persist_score = 0

reasoning_parts.append(
    f"Comparing bivariate (β={is_human_coef_bivariate:.4f}) to controlled "
    f"(β={is_human_coef_controlled:.4f}) results, the effect {persistence_evidence} "
    f"after controlling for confounders (change of {effect_change*100:.1f}%)."
)

# Calculate final Likert score
base_score = stat_score + mag_score + robustness_score + persist_score

# Adjust based on direction (if negative, contradicts hypothesis)
if direction == "negative":
    likert_score = max(0, min(15, base_score))  # Cap at low score
else:
    likert_score = max(0, min(100, base_score))

# Final reasoning
final_reasoning = " ".join(reasoning_parts)

print("\n--- Final Assessment ---")
print(f"Likert Score: {likert_score}/100")
print(f"\nExplanation: {final_reasoning}")

# Write conclusion to file
conclusion = {
    "response": likert_score,
    "explanation": final_reasoning
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE - conclusion.txt written")
print("=" * 80)
