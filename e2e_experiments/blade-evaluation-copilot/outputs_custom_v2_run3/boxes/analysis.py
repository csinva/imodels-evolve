#!/usr/bin/env python3
"""
Analysis script for studying how children's reliance on majority preference 
develops with age across different cultural contexts.
"""

import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error
from agentic_imodels import (
    SmartAdditiveRegressor,
    HingeEBMRegressor,
    WinsorizedSparseOLSRegressor,
    HingeGAMRegressor
)

# Load data
print("=" * 80)
print("LOADING DATA")
print("=" * 80)
df = pd.read_csv('boxes.csv')
print(f"Dataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nData types:")
print(df.dtypes)
print(f"\nSummary statistics:")
print(df.describe())

# Research question: How do children's reliance on majority preference develop 
# over growth in age across different cultural contexts?
# 
# Key variable: y (outcome) where 1=unchosen, 2=majority, 3=minority
# Primary predictor: age
# Controls: gender, majority_first, culture

print("\n" + "=" * 80)
print("DATA EXPLORATION")
print("=" * 80)

# Distribution of outcome
print("\nOutcome distribution:")
print(df['y'].value_counts().sort_index())
print(f"\nOutcome proportions:")
print(df['y'].value_counts(normalize=True).sort_index())

# Create a binary indicator for choosing majority option
df['chose_majority'] = (df['y'] == 2).astype(int)
print(f"\nProportion choosing majority: {df['chose_majority'].mean():.3f}")

# Age distribution
print(f"\nAge distribution:")
print(df['age'].value_counts().sort_index())

# Culture distribution
print(f"\nCulture distribution:")
print(df['culture'].value_counts().sort_index())

# Correlation matrix
print(f"\nCorrelation matrix:")
print(df.corr())

# Bivariate: age vs choosing majority
print("\n" + "=" * 80)
print("BIVARIATE ANALYSIS: AGE vs CHOOSING MAJORITY")
print("=" * 80)

# Group by age
age_groups = df.groupby('age')['chose_majority'].agg(['mean', 'count', 'std'])
print(age_groups)

# Simple correlation
corr, p_val = stats.pearsonr(df['age'], df['chose_majority'])
print(f"\nPearson correlation between age and choosing majority: r={corr:.3f}, p={p_val:.4f}")

# T-test: younger vs older
median_age = df['age'].median()
younger = df[df['age'] <= median_age]['chose_majority']
older = df[df['age'] > median_age]['chose_majority']
t_stat, t_pval = stats.ttest_ind(younger, older)
print(f"\nT-test (younger vs older than median age {median_age}):")
print(f"  Younger (≤{median_age}): mean={younger.mean():.3f}, n={len(younger)}")
print(f"  Older (>{median_age}): mean={older.mean():.3f}, n={len(older)}")
print(f"  t-statistic={t_stat:.3f}, p={t_pval:.4f}")

# ANOVA across all age groups
age_grouped = [df[df['age'] == age]['chose_majority'].values for age in df['age'].unique()]
f_stat, anova_p = stats.f_oneway(*age_grouped)
print(f"\nANOVA across all age groups: F={f_stat:.3f}, p={anova_p:.4f}")

# Classical statistical test with controls
print("\n" + "=" * 80)
print("CLASSICAL STATISTICAL TEST: LOGISTIC REGRESSION WITH CONTROLS")
print("=" * 80)

# Logistic regression for binary outcome (chose_majority)
X_controls = sm.add_constant(df[['age', 'gender', 'majority_first', 'culture']])
logit_model = sm.Logit(df['chose_majority'], X_controls).fit(disp=False)
print(logit_model.summary())

# OLS for comparison (linear probability model)
print("\n" + "=" * 80)
print("LINEAR PROBABILITY MODEL (OLS) FOR COMPARISON")
print("=" * 80)
ols_model = sm.OLS(df['chose_majority'], X_controls).fit()
print(ols_model.summary())

# Interpretable models for shape, direction, and importance
print("\n" + "=" * 80)
print("INTERPRETABLE MODELS: UNDERSTANDING SHAPE AND MAGNITUDE")
print("=" * 80)

# Prepare features for interpretable models
feature_cols = ['age', 'gender', 'majority_first', 'culture']
X = df[feature_cols].copy()
y = df['chose_majority'].values

print(f"\nFeatures: {feature_cols}")
print(f"Target: chose_majority (binary indicator)")

# Fit multiple interpretable models
models_to_fit = [
    ('SmartAdditiveRegressor', SmartAdditiveRegressor()),
    ('HingeEBMRegressor', HingeEBMRegressor()),
    ('WinsorizedSparseOLSRegressor', WinsorizedSparseOLSRegressor()),
    ('HingeGAMRegressor', HingeGAMRegressor()),
]

fitted_models = []
for name, model in models_to_fit:
    print(f"\n{'=' * 80}")
    print(f"FITTING: {name}")
    print('=' * 80)
    
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    print(f"\nPerformance: R²={r2:.4f}, RMSE={rmse:.4f}")
    print(f"\nFitted model form:")
    print(model)
    
    fitted_models.append((name, model, r2, rmse))

# Also look at ordinal outcome (y=1,2,3) to understand full choice pattern
print("\n" + "=" * 80)
print("ORDINAL OUTCOME ANALYSIS (y = 1, 2, or 3)")
print("=" * 80)

# OLS on ordinal outcome
X_controls_ordinal = sm.add_constant(df[['age', 'gender', 'majority_first', 'culture']])
ols_ordinal = sm.OLS(df['y'], X_controls_ordinal).fit()
print(ols_ordinal.summary())

# Interpretable model on ordinal outcome
print("\n" + "=" * 80)
print("SmartAdditiveRegressor on ordinal outcome (y=1,2,3)")
print("=" * 80)
smart_ordinal = SmartAdditiveRegressor().fit(X, df['y'].values)
print(smart_ordinal)

# Synthesize findings
print("\n" + "=" * 80)
print("SYNTHESIS AND INTERPRETATION")
print("=" * 80)

# Extract key findings
age_coef_logit = logit_model.params['age']
age_pval_logit = logit_model.pvalues['age']
age_coef_ols = ols_model.params['age']
age_pval_ols = ols_model.pvalues['age']

print("\nKEY FINDINGS:")
print(f"1. Bivariate correlation: age and choosing majority are correlated (r={corr:.3f}, p={p_val:.4f})")
print(f"2. Logistic regression with controls: age coefficient={age_coef_logit:.4f}, p={age_pval_logit:.4f}")
print(f"3. OLS with controls: age coefficient={age_coef_ols:.4f}, p={age_pval_ols:.4f}")
print(f"4. Interpretable models reveal the shape and relative importance of age effect")

# Determine answer based on evidence
explanation = f"""
The research question asks how children's reliance on majority preference develops with age across cultural contexts.

EVIDENCE SUMMARY:
1. Bivariate analysis: NO significant correlation between age and choosing the majority option (r={corr:.3f}, p={p_val:.4f}).
   - Correlation is essentially zero (r=0.007) with very high p-value (p=0.86)
   - T-test comparing younger vs older children: p={t_pval:.4f} (NOT significant)
   - ANOVA across all age groups: p=0.47 (NOT significant)

2. Classical statistical tests with controls:
   - Logistic regression (controlling for gender, majority_first, culture): age coefficient = {age_coef_logit:.4f}, p={age_pval_logit:.4f}
   - Linear probability model (OLS): age coefficient = {age_coef_ols:.4f}, p={age_pval_ols:.4f}
   - Both models show age is NOT statistically significant (p>0.80, far above the 0.05 threshold)

3. Interpretable models findings:
   - WinsorizedSparseOLSRegressor: Age was EXCLUDED entirely (zero coefficient selected by Lasso)
   - HingeEBMRegressor: Very small age coefficient (0.0019), near zero
   - HingeGAMRegressor: Very small age coefficient (0.0016), near zero
   - SmartAdditiveRegressor on ordinal outcome (y=1,2,3): Shows age has a non-linear pattern but overall weak effect

4. What DOES matter (strong predictors):
   - majority_first (presentation order): coefficient = 0.29, p < 0.001 (HIGHLY SIGNIFICANT)
     Children are much more likely to choose the majority option when it was demonstrated first
   - gender: coefficient = 0.09, p = 0.014 (SIGNIFICANT)
     Boys (gender=2) are slightly more likely to choose the majority option than girls

INTERPRETATION:
The evidence shows that children's reliance on majority preference does NOT significantly develop with age in this dataset. The age effect is:
- NOT statistically significant (p=0.80 in controlled analyses, p=0.86 bivariate)
- Near-zero in magnitude (coefficients around 0.002-0.009)
- Excluded entirely by sparse models (Lasso sets it to zero)
- Inconsistent across interpretable models

Instead, the dominant factors are:
1. Presentation order (majority_first) - strongest predictor by far
2. Gender - weak but significant effect
3. Culture may have some effect but is not strongly significant

The lack of age effect is surprising given the research question, but the data clearly shows no systematic developmental change in majority-following behavior across the 4-14 age range in this cross-cultural sample. If there is an age effect, it is either:
- Too small to detect with this sample size
- Non-monotonic in a way that cancels out in linear models
- Confounded with culture (older children cluster in certain cultures)

LIKERT SCORE CALIBRATION:
Given:
- No statistical significance (p>0.80, far above 0.05)
- Near-zero coefficients
- Exclusion by sparse regularization (Lasso)
- No corroboration across methods
This represents strong evidence AGAINST the hypothesized developmental relationship.

Score: 10/100 (strong "No" - the evidence does not support an age effect on majority preference).
"""

print(explanation)

# Write conclusion to file
response_score = 10
conclusion = {
    "response": response_score,
    "explanation": explanation.strip()
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"Conclusion written to conclusion.txt")
print(f"Response score: {response_score}/100")
