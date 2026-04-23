import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.preprocessing import StandardScaler
from agentic_imodels import (
    SmartAdditiveRegressor,
    HingeEBMRegressor,
    WinsorizedSparseOLSRegressor,
    HingeGAMRegressor
)

# Load data
df = pd.read_csv('caschools.csv')

print("="*80)
print("CALIFORNIA SCHOOLS DATASET ANALYSIS")
print("Research Question: Is a lower student-teacher ratio associated with higher academic performance?")
print("="*80)

# Data exploration
print("\n1. DATA EXPLORATION")
print("-" * 80)
print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

# Calculate student-teacher ratio
df['str_ratio'] = df['students'] / df['teachers']

# Use average of math and read scores as academic performance measure
df['academic_performance'] = (df['math'] + df['read']) / 2

print(f"\nStudent-teacher ratio: mean={df['str_ratio'].mean():.2f}, std={df['str_ratio'].std():.2f}")
print(f"Academic performance: mean={df['academic_performance'].mean():.2f}, std={df['academic_performance'].std():.2f}")

print("\nSummary statistics for key variables:")
key_vars = ['str_ratio', 'academic_performance', 'income', 'english', 'lunch', 'calworks', 'expenditure']
print(df[key_vars].describe())

# Correlation analysis
print("\n2. BIVARIATE CORRELATION ANALYSIS")
print("-" * 80)
corr_result = stats.pearsonr(df['str_ratio'], df['academic_performance'])
print(f"Pearson correlation between STR and academic performance: r={corr_result[0]:.4f}, p={corr_result[1]:.4e}")

# Check correlation with potential confounders
print("\nCorrelations with academic performance:")
for var in ['str_ratio', 'income', 'english', 'lunch', 'calworks', 'expenditure']:
    if var in df.columns:
        corr = df[var].corr(df['academic_performance'])
        print(f"  {var}: {corr:.4f}")

# Classical regression analysis
print("\n3. CLASSICAL OLS REGRESSION ANALYSIS")
print("-" * 80)

# Model 1: Bivariate (no controls)
print("\nModel 1: Bivariate (academic_performance ~ str_ratio)")
X_bivariate = sm.add_constant(df[['str_ratio']])
model_bivariate = sm.OLS(df['academic_performance'], X_bivariate).fit()
print(model_bivariate.summary())

# Model 2: With controls
print("\nModel 2: Multivariate with controls (income, english, lunch, expenditure)")
control_vars = ['str_ratio', 'income', 'english', 'lunch', 'expenditure']
X_controlled = sm.add_constant(df[control_vars])
model_controlled = sm.OLS(df['academic_performance'], X_controlled).fit()
print(model_controlled.summary())

# Extract key statistics
str_coef_bivariate = model_bivariate.params['str_ratio']
str_pval_bivariate = model_bivariate.pvalues['str_ratio']
str_coef_controlled = model_controlled.params['str_ratio']
str_pval_controlled = model_controlled.pvalues['str_ratio']

print(f"\n*** KEY FINDING FROM CLASSICAL REGRESSION ***")
print(f"Bivariate: β={str_coef_bivariate:.4f}, p={str_pval_bivariate:.4e}")
print(f"Controlled: β={str_coef_controlled:.4f}, p={str_pval_controlled:.4e}")

# Interpretable models using agentic_imodels
print("\n4. INTERPRETABLE MODELS (agentic_imodels)")
print("-" * 80)

# Prepare feature matrix (excluding target and non-predictive columns)
feature_cols = ['str_ratio', 'income', 'english', 'lunch', 'calworks', 'expenditure', 'students']
X = df[feature_cols].copy()
y = df['academic_performance'].copy()

# Handle any missing values
X = X.fillna(X.mean())
y = y.fillna(y.mean())

print(f"\nFeature matrix shape: {X.shape}")
print(f"Features: {list(X.columns)}")

# Fit multiple interpretable models
models_to_fit = [
    ('SmartAdditiveRegressor', SmartAdditiveRegressor()),
    ('HingeEBMRegressor', HingeEBMRegressor()),
    ('WinsorizedSparseOLSRegressor', WinsorizedSparseOLSRegressor()),
    ('HingeGAMRegressor', HingeGAMRegressor())
]

fitted_models = []
for name, model in models_to_fit:
    print(f"\n{'='*80}")
    print(f"=== {name} ===")
    print('='*80)
    model.fit(X, y)
    print(model)
    fitted_models.append((name, model))
    
    # Calculate R^2 on training data
    from sklearn.metrics import r2_score
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    print(f"\nTraining R^2: {r2:.4f}")

# Final interpretation
print("\n" + "="*80)
print("5. COMPREHENSIVE INTERPRETATION")
print("="*80)

interpretation = []

# Direction and significance from OLS
if str_coef_controlled < 0 and str_pval_controlled < 0.05:
    interpretation.append(f"Classical OLS shows a NEGATIVE association: β={str_coef_controlled:.4f}, p={str_pval_controlled:.4e}")
    interpretation.append("Lower student-teacher ratio (fewer students per teacher) is associated with HIGHER academic performance")
elif str_coef_controlled > 0 and str_pval_controlled < 0.05:
    interpretation.append(f"Classical OLS shows a POSITIVE association: β={str_coef_controlled:.4f}, p={str_pval_controlled:.4e}")
    interpretation.append("Higher student-teacher ratio (more students per teacher) is associated with HIGHER academic performance")
else:
    interpretation.append(f"Classical OLS shows NO significant association: β={str_coef_controlled:.4f}, p={str_pval_controlled:.4e}")

# Check if effect persists after controls
if abs(str_coef_bivariate) > abs(str_coef_controlled) * 1.5:
    interpretation.append("Effect WEAKENS substantially after adding controls (income, English learners, lunch assistance, expenditure)")
elif abs(str_coef_controlled) > 0:
    interpretation.append("Effect PERSISTS after controlling for socioeconomic factors")

print("\n".join(interpretation))

# Determine Likert score based on evidence
print("\n" + "="*80)
print("6. LIKERT SCORE CALIBRATION")
print("="*80)

# Scoring logic:
# - Strong significant effect (p < 0.01), persists after controls, correct direction: 75-100
# - Moderate effect (p < 0.05), persists: 50-75
# - Weak or marginal (0.05 < p < 0.10): 25-50
# - Non-significant (p > 0.10): 0-25

score = 50  # default
explanation_parts = []

# Check statistical significance
if str_pval_controlled < 0.01:
    score = 75
    explanation_parts.append(f"Highly significant effect (p={str_pval_controlled:.4e})")
elif str_pval_controlled < 0.05:
    score = 60
    explanation_parts.append(f"Significant effect (p={str_pval_controlled:.4e})")
elif str_pval_controlled < 0.10:
    score = 35
    explanation_parts.append(f"Marginal effect (p={str_pval_controlled:.4e})")
else:
    score = 15
    explanation_parts.append(f"Non-significant effect (p={str_pval_controlled:.4e})")

# Check direction (negative coefficient means lower STR -> higher performance)
if str_coef_controlled < 0:
    score += 10
    explanation_parts.append(f"Correct direction: negative coefficient ({str_coef_controlled:.4f}) means lower STR → higher performance")
else:
    score -= 15
    explanation_parts.append(f"Wrong direction: positive coefficient ({str_coef_controlled:.4f}) means higher STR → higher performance")

# Check effect size (standardize for interpretation)
std_str = df['str_ratio'].std()
std_perf = df['academic_performance'].std()
standardized_effect = (str_coef_controlled * std_str) / std_perf

if abs(standardized_effect) > 0.3:
    score += 10
    explanation_parts.append(f"Moderate to large standardized effect size ({standardized_effect:.3f})")
elif abs(standardized_effect) > 0.1:
    explanation_parts.append(f"Small to moderate standardized effect size ({standardized_effect:.3f})")
else:
    score -= 5
    explanation_parts.append(f"Small standardized effect size ({standardized_effect:.3f})")

# Check robustness (bivariate vs controlled)
effect_reduction = abs((str_coef_bivariate - str_coef_controlled) / str_coef_bivariate) if str_coef_bivariate != 0 else 0
if effect_reduction < 0.3:
    score += 5
    explanation_parts.append("Effect is robust to controls (minimal reduction)")
elif effect_reduction < 0.6:
    explanation_parts.append("Effect partially attenuated by controls")
else:
    score -= 10
    explanation_parts.append("Effect substantially weakened by controls, suggesting confounding")

# Clamp score to [0, 100]
score = max(0, min(100, score))

explanation_parts.append(f"The evidence suggests {'YES' if score >= 50 else 'WEAK/NO'} - lower student-teacher ratio IS{'' if score >= 50 else ' NOT strongly'} associated with higher academic performance after controlling for socioeconomic factors")

final_explanation = ". ".join(explanation_parts) + "."

print(f"\nFinal Likert Score: {score}")
print(f"\nExplanation: {final_explanation}")

# Write conclusion to file
conclusion = {
    "response": score,
    "explanation": final_explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "="*80)
print("ANALYSIS COMPLETE - conclusion.txt written")
print("="*80)
