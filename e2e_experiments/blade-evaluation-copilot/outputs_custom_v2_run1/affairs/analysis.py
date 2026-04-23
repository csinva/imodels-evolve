import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from agentic_imodels import SmartAdditiveRegressor, HingeEBMRegressor, WinsorizedSparseOLSRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Load data
df = pd.read_csv('affairs.csv')

print("="*80)
print("EXPLORATORY DATA ANALYSIS")
print("="*80)

# Basic statistics
print("\nDataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

print("\nSummary statistics:")
print(df.describe())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

# Research question: Does having children decrease engagement in extramarital affairs?
print("\n" + "="*80)
print("RESEARCH QUESTION: Does having children decrease engagement in extramarital affairs?")
print("="*80)

# Convert categorical variables to binary
df['children_yes'] = (df['children'] == 'yes').astype(int)
df['gender_male'] = (df['gender'] == 'male').astype(int)

# Key variables for analysis
outcome = 'affairs'
key_predictor = 'children_yes'

print(f"\n{outcome} by children status:")
print(df.groupby('children')[outcome].describe())

# Bivariate analysis
print("\n" + "-"*80)
print("BIVARIATE ANALYSIS")
print("-"*80)

children_yes = df[df['children'] == 'yes'][outcome]
children_no = df[df['children'] == 'no'][outcome]

print(f"\nMean {outcome} with children: {children_yes.mean():.3f}")
print(f"Mean {outcome} without children: {children_no.mean():.3f}")
print(f"Difference: {children_yes.mean() - children_no.mean():.3f}")

# T-test
t_stat, p_val = stats.ttest_ind(children_yes, children_no)
print(f"\nT-test: t={t_stat:.3f}, p={p_val:.4f}")

# Point-biserial correlation (equivalent for binary predictor)
corr, p_corr = stats.pointbiserialr(df['children_yes'], df[outcome])
print(f"Point-biserial correlation: r={corr:.3f}, p={p_corr:.4f}")

# Classical regression with controls
print("\n" + "-"*80)
print("CLASSICAL REGRESSION WITH CONTROLS")
print("-"*80)

# First, simple bivariate OLS
X_simple = sm.add_constant(df[[key_predictor]])
model_simple = sm.OLS(df[outcome], X_simple).fit()
print("\n*** BIVARIATE OLS (affairs ~ children_yes) ***")
print(model_simple.summary())

# Full model with controls
controls = ['gender_male', 'age', 'yearsmarried', 'religiousness', 
            'education', 'occupation', 'rating']
X_full = sm.add_constant(df[[key_predictor] + controls])
model_full = sm.OLS(df[outcome], X_full).fit()
print("\n*** FULL OLS WITH CONTROLS ***")
print(model_full.summary())

print("\n*** KEY FINDINGS FROM OLS ***")
print(f"Bivariate: children_yes coefficient = {model_simple.params[key_predictor]:.4f}, p = {model_simple.pvalues[key_predictor]:.4f}")
print(f"Controlled: children_yes coefficient = {model_full.params[key_predictor]:.4f}, p = {model_full.pvalues[key_predictor]:.4f}")

# Interpretable models
print("\n" + "="*80)
print("INTERPRETABLE MODELS (agentic_imodels)")
print("="*80)

# Prepare features for interpretable models
feature_cols = [key_predictor] + controls
X = df[feature_cols]
y = df[outcome]

# Model 1: SmartAdditiveRegressor (honest GAM)
print("\n" + "-"*80)
print("MODEL 1: SmartAdditiveRegressor (honest GAM)")
print("-"*80)
model1 = SmartAdditiveRegressor()
model1.fit(X, y)
y_pred1 = model1.predict(X)
r2_1 = r2_score(y, y_pred1)
rmse_1 = np.sqrt(mean_squared_error(y, y_pred1))
print(f"R² = {r2_1:.4f}, RMSE = {rmse_1:.4f}")
print("\nFitted model:")
print(model1)

# Model 2: HingeEBMRegressor (high-rank, decoupled)
print("\n" + "-"*80)
print("MODEL 2: HingeEBMRegressor (high-rank, decoupled)")
print("-"*80)
model2 = HingeEBMRegressor()
model2.fit(X, y)
y_pred2 = model2.predict(X)
r2_2 = r2_score(y, y_pred2)
rmse_2 = np.sqrt(mean_squared_error(y, y_pred2))
print(f"R² = {r2_2:.4f}, RMSE = {rmse_2:.4f}")
print("\nFitted model:")
print(model2)

# Model 3: WinsorizedSparseOLSRegressor (honest sparse linear)
print("\n" + "-"*80)
print("MODEL 3: WinsorizedSparseOLSRegressor (honest sparse linear)")
print("-"*80)
model3 = WinsorizedSparseOLSRegressor()
model3.fit(X, y)
y_pred3 = model3.predict(X)
r2_3 = r2_score(y, y_pred3)
rmse_3 = np.sqrt(mean_squared_error(y, y_pred3))
print(f"R² = {r2_3:.4f}, RMSE = {rmse_3:.4f}")
print("\nFitted model:")
print(model3)

# Synthesis and conclusion
print("\n" + "="*80)
print("SYNTHESIS AND CONCLUSION")
print("="*80)

print("\n*** Evidence Summary ***")
print(f"1. Bivariate difference: Children YES mean={children_yes.mean():.3f}, NO mean={children_no.mean():.3f}")
print(f"   Difference = {children_yes.mean() - children_no.mean():.3f} (t={t_stat:.3f}, p={p_val:.4f})")
print(f"   → {'Significant' if p_val < 0.05 else 'Not significant'} at α=0.05")

print(f"\n2. Bivariate OLS: β={model_simple.params[key_predictor]:.4f}, p={model_simple.pvalues[key_predictor]:.4f}")
print(f"   → {'Significant' if model_simple.pvalues[key_predictor] < 0.05 else 'Not significant'} at α=0.05")

print(f"\n3. Controlled OLS (with gender, age, yearsmarried, religiousness, education, occupation, rating):")
print(f"   β={model_full.params[key_predictor]:.4f}, p={model_full.pvalues[key_predictor]:.4f}")
print(f"   → {'Significant' if model_full.pvalues[key_predictor] < 0.05 else 'Not significant'} at α=0.05")

print("\n4. Interpretable models (from printed forms above):")
print("   Check the coefficient/importance of 'children_yes' in each model")
print("   - SmartAdditiveRegressor shows feature importance and shape")
print("   - HingeEBMRegressor shows binned relationships")
print("   - WinsorizedSparseOLSRegressor shows sparse linear coefficients (zeroing = strong null evidence)")

# Determine the Likert score based on evidence
bivariate_sig = p_val < 0.05
controlled_sig = model_full.pvalues[key_predictor] < 0.05
bivariate_coef = model_simple.params[key_predictor]
controlled_coef = model_full.params[key_predictor]

# Decision logic for Likert score
# 0 = strong "No", 100 = strong "Yes"
# Question: Does having children DECREASE affairs? (i.e., is the effect negative?)

print("\n*** Decision Logic ***")
print(f"Direction of effect in bivariate OLS: {bivariate_coef:.4f} ({'negative' if bivariate_coef < 0 else 'positive'})")
print(f"Direction of effect in controlled OLS: {controlled_coef:.4f} ({'negative' if controlled_coef < 0 else 'positive'})")

if bivariate_coef > 0 and controlled_coef > 0:
    # Effect is positive (children INCREASE affairs, opposite of question)
    if bivariate_sig or controlled_sig:
        score = 10  # Strong "No" - opposite direction is significant
        explanation = "Evidence shows children are associated with MORE extramarital affairs, not fewer. Bivariate and controlled OLS both show positive coefficients, contradicting the hypothesis."
    else:
        score = 25  # Moderate "No" - weak positive effect
        explanation = "The relationship is slightly positive (opposite of hypothesis) but not statistically significant. Children do not decrease affairs."
elif bivariate_coef < 0 and controlled_coef < 0:
    # Effect is negative (children DECREASE affairs, as hypothesized)
    if bivariate_sig and controlled_sig:
        # Check magnitude
        if abs(controlled_coef) > 0.5:
            score = 85  # Strong "Yes"
            explanation = "Strong evidence: children are significantly associated with fewer extramarital affairs in both bivariate and controlled analyses. The effect is statistically significant and persists after controlling for gender, age, years married, religiousness, education, occupation, and marital satisfaction."
        else:
            score = 70  # Moderate-strong "Yes"
            explanation = "Moderate to strong evidence: children are significantly associated with fewer extramarital affairs in both bivariate and controlled analyses, though the magnitude is modest. Effect persists after controlling for confounders."
    elif controlled_sig:
        score = 60  # Moderate "Yes"
        explanation = "Moderate evidence: children are associated with fewer affairs in the controlled model (significant), though the bivariate relationship is weaker. The effect emerges or strengthens after accounting for confounders."
    elif bivariate_sig:
        score = 50  # Moderate "Yes" but weakens with controls
        explanation = "Mixed evidence: the bivariate relationship shows children associated with fewer affairs (significant), but the effect weakens and loses significance when controlling for other factors. This suggests the relationship may be partially confounded."
    else:
        score = 35  # Weak "Yes" - direction right but not significant
        explanation = "Weak evidence: the direction suggests children may decrease affairs, but the relationship is not statistically significant in either bivariate or controlled models. The effect is uncertain."
else:
    # Mixed signs (positive bivariate, negative controlled, or vice versa)
    score = 30  # Weak/inconsistent
    explanation = "Inconsistent evidence: the relationship changes direction between bivariate and controlled analyses, indicating confounding or model instability. No clear conclusion can be drawn."

print(f"\nFinal Likert score: {score}/100")
print(f"Explanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "="*80)
print("CONCLUSION WRITTEN TO conclusion.txt")
print("="*80)
