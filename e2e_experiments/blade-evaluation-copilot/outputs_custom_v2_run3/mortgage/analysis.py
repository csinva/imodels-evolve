#!/usr/bin/env python3
"""
Analysis script to examine the effect of gender on mortgage approval.

Research question: How does gender affect whether banks approve an individual's 
mortgage application?
"""

import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error

import sys
sys.path.insert(0, '/home/chansingh/imodels-evolve/e2e_experiments/blade-evaluation-copilot/outputs_custom_v2_run3/mortgage')
from agentic_imodels import (
    SmartAdditiveRegressor,
    HingeEBMRegressor,
    WinsorizedSparseOLSRegressor,
    HingeGAMRegressor
)

print("=" * 80)
print("MORTGAGE APPROVAL ANALYSIS: Effect of Gender")
print("=" * 80)

# Load data
df = pd.read_csv('mortgage.csv')
print(f"\nDataset shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")

# Step 1: Exploratory Data Analysis
print("\n" + "=" * 80)
print("STEP 1: EXPLORATORY DATA ANALYSIS")
print("=" * 80)

print("\nBasic statistics:")
print(df.describe())

print("\n--- Target variable (accept) ---")
print(df['accept'].value_counts())
print(f"Acceptance rate: {df['accept'].mean():.3f}")

print("\n--- Gender distribution ---")
print(df['female'].value_counts())
print(f"Proportion female: {df['female'].mean():.3f}")

print("\n--- Acceptance rate by gender ---")
gender_approval = df.groupby('female')['accept'].agg(['mean', 'count', 'sum'])
gender_approval.index = ['Male (0)', 'Female (1)']
print(gender_approval)

# Step 2: Bivariate analysis
print("\n" + "=" * 80)
print("STEP 2: BIVARIATE ANALYSIS")
print("=" * 80)

male_acceptance = df[df['female'] == 0]['accept'].values
female_acceptance = df[df['female'] == 1]['accept'].values

# Two-sample t-test
t_stat, p_value = stats.ttest_ind(male_acceptance, female_acceptance)
print(f"\nTwo-sample t-test (Male vs Female acceptance):")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value:.4f}")

# Chi-square test for independence
contingency = pd.crosstab(df['female'], df['accept'])
chi2, chi_p, dof, expected = stats.chi2_contingency(contingency)
print(f"\nChi-square test of independence:")
print(f"  χ²: {chi2:.4f}")
print(f"  p-value: {chi_p:.4f}")

# Point-biserial correlation
correlation, corr_p = stats.pointbiserialr(df['female'], df['accept'])
print(f"\nPoint-biserial correlation (gender-acceptance):")
print(f"  r: {correlation:.4f}")
print(f"  p-value: {corr_p:.4f}")

# Step 3: Classical statistical test with controls
print("\n" + "=" * 80)
print("STEP 3: CLASSICAL STATISTICAL TEST WITH CONTROLS")
print("=" * 80)

# Since accept is binary (0/1), we use logistic regression
# But also run OLS on 0/1 for comparison (linear probability model)

# Define features
iv = 'female'  # independent variable of interest
controls = ['black', 'housing_expense_ratio', 'self_employed', 'married', 
            'mortgage_credit', 'consumer_credit', 'bad_history', 'PI_ratio', 
            'loan_to_value', 'denied_PMI']
dv = 'accept'  # dependent variable

# Remove rows with missing values
df_clean = df[[iv] + controls + [dv]].dropna()
print(f"\nClean dataset shape: {df_clean.shape}")

print("\n--- Model 1: Bivariate Logistic Regression (gender only) ---")
X_bivariate = sm.add_constant(df_clean[[iv]])
logit_bivariate = sm.Logit(df_clean[dv], X_bivariate).fit(disp=0)
print(logit_bivariate.summary())

print("\n--- Model 2: Multivariate Logistic Regression (gender + controls) ---")
X_multivariate = sm.add_constant(df_clean[[iv] + controls])
logit_multivariate = sm.Logit(df_clean[dv], X_multivariate).fit(disp=0)
print(logit_multivariate.summary())

print("\n--- Model 3: Linear Probability Model (OLS on 0/1) with controls ---")
ols_model = sm.OLS(df_clean[dv], X_multivariate).fit()
print(ols_model.summary())

# Extract key coefficients
coef_bivariate = logit_bivariate.params[iv]
pval_bivariate = logit_bivariate.pvalues[iv]
coef_multivariate = logit_multivariate.params[iv]
pval_multivariate = logit_multivariate.pvalues[iv]
coef_ols = ols_model.params[iv]
pval_ols = ols_model.pvalues[iv]

print("\n" + "-" * 80)
print("SUMMARY OF GENDER EFFECTS:")
print("-" * 80)
print(f"Bivariate logistic:     coef={coef_bivariate:.4f}, p={pval_bivariate:.4f}")
print(f"Controlled logistic:    coef={coef_multivariate:.4f}, p={pval_multivariate:.4f}")
print(f"Controlled OLS (LPM):   coef={coef_ols:.4f}, p={pval_ols:.4f}")

# Step 4: Interpretable models for shape, direction, importance
print("\n" + "=" * 80)
print("STEP 4: INTERPRETABLE MODELS")
print("=" * 80)

# Prepare data for agentic_imodels
X_features = df_clean[[iv] + controls]
y_target = df_clean[dv].values

print(f"\nFitting interpretable models on {X_features.shape[0]} samples, {X_features.shape[1]} features")
print(f"Features: {list(X_features.columns)}")

# Fit multiple interpretable models
models_to_fit = [
    ('SmartAdditiveRegressor', SmartAdditiveRegressor()),
    ('HingeEBMRegressor', HingeEBMRegressor()),
    ('WinsorizedSparseOLSRegressor', WinsorizedSparseOLSRegressor()),
]

for name, model in models_to_fit:
    print(f"\n{'=' * 80}")
    print(f"MODEL: {name}")
    print('=' * 80)
    
    model.fit(X_features, y_target)
    y_pred = model.predict(X_features)
    r2 = r2_score(y_target, y_pred)
    rmse = np.sqrt(mean_squared_error(y_target, y_pred))
    
    print(f"\nPerformance: R²={r2:.4f}, RMSE={rmse:.4f}")
    print(f"\nFitted model form:")
    print(model)

# Step 5: Synthesize findings and create conclusion
print("\n" + "=" * 80)
print("STEP 5: SYNTHESIS AND CONCLUSION")
print("=" * 80)

print("\n--- Evidence Summary ---")

print("\n1. BIVARIATE EVIDENCE:")
print(f"   - Female acceptance rate: {df[df['female']==1]['accept'].mean():.3f}")
print(f"   - Male acceptance rate: {df[df['female']==0]['accept'].mean():.3f}")
print(f"   - Difference: {df[df['female']==1]['accept'].mean() - df[df['female']==0]['accept'].mean():.3f}")
print(f"   - t-test p-value: {p_value:.4f}")
print(f"   - Chi-square p-value: {chi_p:.4f}")

print("\n2. CONTROLLED REGRESSION EVIDENCE:")
print(f"   - Logistic regression (bivariate) coefficient: {coef_bivariate:.4f}, p={pval_bivariate:.4f}")
print(f"   - Logistic regression (controlled) coefficient: {coef_multivariate:.4f}, p={pval_multivariate:.4f}")
print(f"   - OLS linear probability coefficient: {coef_ols:.4f}, p={pval_ols:.4f}")

# Determine conclusion based on evidence
# Score calibration per SKILL.md:
# - Strong significant effect, persists across models, top-ranked → 75-100
# - Moderate / partially significant / mid-rank → 40-70
# - Weak, inconsistent, or marginal → 15-40
# - Zero in Lasso AND non-significant AND low importance → 0-15

interpretation = ""
score = 50  # default midpoint

# Check significance and effect direction
if pval_multivariate < 0.05:
    if coef_multivariate > 0:
        # Positive coefficient means female (1) has higher odds of acceptance
        interpretation = (
            "Yes, gender affects mortgage approval. Female applicants have "
            f"significantly higher odds of approval (logistic coef={coef_multivariate:.3f}, "
            f"p={pval_multivariate:.4f}) even after controlling for creditworthiness, "
            "financial ratios, and other key variables. "
        )
        score = 70
    else:
        # Negative coefficient means female (1) has lower odds of acceptance
        interpretation = (
            "Yes, gender affects mortgage approval. Female applicants have "
            f"significantly lower odds of approval (logistic coef={coef_multivariate:.3f}, "
            f"p={pval_multivariate:.4f}) even after controlling for creditworthiness, "
            "financial ratios, and other key variables. "
        )
        score = 70
else:
    # Not significant after controls
    if pval_bivariate < 0.05:
        # Was significant bivariate but not controlled
        interpretation = (
            "Gender shows a bivariate relationship with mortgage approval "
            f"(p={pval_bivariate:.4f}), but this effect becomes non-significant "
            f"(p={pval_multivariate:.4f}) when controlling for creditworthiness and "
            "financial variables. This suggests the bivariate relationship is largely "
            "explained by confounding factors. The evidence for a direct gender effect "
            "is weak. "
        )
        score = 30
    else:
        # Not significant even bivariate
        interpretation = (
            "Gender does not show a statistically significant effect on mortgage approval. "
            f"Both the bivariate (p={pval_bivariate:.4f}) and controlled "
            f"(p={pval_multivariate:.4f}) logistic regressions show non-significant "
            "coefficients for gender. "
        )
        score = 15

# Add interpretable model insights
interpretation += (
    "The interpretable models (SmartAdditiveRegressor, HingeEBMRegressor, "
    "WinsorizedSparseOLSRegressor) were fitted to examine feature importance and shape. "
    "These models help verify the robustness of the gender effect and identify "
    "the relative importance of gender compared to other predictors like credit scores, "
    "debt ratios, and employment status."
)

print("\n" + "=" * 80)
print("FINAL CONCLUSION")
print("=" * 80)
print(f"\nLikert Score (0-100): {score}")
print(f"\nExplanation:\n{interpretation}")

# Write conclusion to file
conclusion = {
    "response": score,
    "explanation": interpretation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "=" * 80)
print("Analysis complete. Conclusion written to conclusion.txt")
print("=" * 80)
