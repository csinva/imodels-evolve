#!/usr/bin/env python3
"""
Analysis script to investigate the relationship between hurricane name femininity 
and deaths/precautionary measures.
"""

import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from imodels import RuleFitRegressor, FIGSRegressor
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('hurricane.csv')

print("="*80)
print("HURRICANE NAME FEMININITY AND DEATHS ANALYSIS")
print("="*80)
print(f"\nDataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

print("\n" + "="*80)
print("DATA EXPLORATION")
print("="*80)

# Summary statistics
print("\nSummary statistics for key variables:")
print(df[['masfem', 'alldeaths', 'category', 'min', 'wind', 'ndam']].describe())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Correlation analysis
print("\n" + "="*80)
print("CORRELATION ANALYSIS")
print("="*80)
numeric_cols = ['masfem', 'alldeaths', 'category', 'min', 'wind', 'ndam', 'year']
corr_matrix = df[numeric_cols].corr()
print("\nCorrelation with alldeaths:")
print(corr_matrix['alldeaths'].sort_values(ascending=False))

# Direct correlation test between femininity and deaths
print("\n" + "="*80)
print("DIRECT RELATIONSHIP: FEMININITY vs DEATHS")
print("="*80)
corr_fem_deaths, p_fem_deaths = stats.pearsonr(df['masfem'], df['alldeaths'])
print(f"Pearson correlation (masfem vs alldeaths): r={corr_fem_deaths:.4f}, p={p_fem_deaths:.4f}")

# Spearman correlation (non-parametric, robust to outliers)
spearman_corr, spearman_p = stats.spearmanr(df['masfem'], df['alldeaths'])
print(f"Spearman correlation (masfem vs alldeaths): rho={spearman_corr:.4f}, p={spearman_p:.4f}")

# Compare deaths by gender binary
print("\n" + "="*80)
print("COMPARISON BY BINARY GENDER CLASSIFICATION")
print("="*80)
male_deaths = df[df['gender_mf'] == 0]['alldeaths']
female_deaths = df[df['gender_mf'] == 1]['alldeaths']
print(f"Male-named hurricanes: n={len(male_deaths)}, mean deaths={male_deaths.mean():.2f}, std={male_deaths.std():.2f}")
print(f"Female-named hurricanes: n={len(female_deaths)}, mean deaths={female_deaths.mean():.2f}, std={female_deaths.std():.2f}")

# T-test
t_stat, t_p = stats.ttest_ind(male_deaths, female_deaths)
print(f"T-test: t={t_stat:.4f}, p={t_p:.4f}")

# Mann-Whitney U test (non-parametric alternative)
u_stat, u_p = stats.mannwhitneyu(male_deaths, female_deaths, alternative='two-sided')
print(f"Mann-Whitney U test: U={u_stat:.4f}, p={u_p:.4f}")

# LINEAR REGRESSION ANALYSIS
print("\n" + "="*80)
print("LINEAR REGRESSION: FEMININITY PREDICTING DEATHS")
print("="*80)

# Simple linear regression
X_simple = sm.add_constant(df['masfem'])
model_simple = sm.OLS(df['alldeaths'], X_simple).fit()
print("\nSimple regression (masfem -> alldeaths):")
print(model_simple.summary())

# Multiple regression controlling for hurricane severity
print("\n" + "="*80)
print("MULTIPLE REGRESSION: CONTROLLING FOR HURRICANE SEVERITY")
print("="*80)

# Control for hurricane characteristics
control_vars = ['masfem', 'category', 'min', 'wind', 'ndam']
df_reg = df[control_vars + ['alldeaths']].dropna()

X_multi = sm.add_constant(df_reg[control_vars])
model_multi = sm.OLS(df_reg['alldeaths'], X_multi).fit()
print("\nMultiple regression (controlling for severity):")
print(model_multi.summary())

# INTERPRETABLE MACHINE LEARNING MODELS
print("\n" + "="*80)
print("INTERPRETABLE ML MODELS")
print("="*80)

# Prepare data for sklearn models
X_ml = df_reg[control_vars].values
y_ml = df_reg['alldeaths'].values

# Decision Tree for interpretability
print("\nDecision Tree Regressor:")
dt = DecisionTreeRegressor(max_depth=4, min_samples_leaf=5, random_state=42)
dt.fit(X_ml, y_ml)
print("Feature importances:")
for feat, imp in zip(control_vars, dt.feature_importances_):
    print(f"  {feat}: {imp:.4f}")

# Ridge regression with standardized features
print("\nRidge Regression (standardized coefficients):")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_ml)
ridge = Ridge(alpha=1.0)
ridge.fit(X_scaled, y_ml)
print("Standardized coefficients:")
for feat, coef in zip(control_vars, ridge.coef_):
    print(f"  {feat}: {coef:.4f}")

# Try RuleFit for interpretable rules
print("\nRuleFit Regressor (rule-based model):")
try:
    rulefit = RuleFitRegressor(max_rules=10, random_state=42)
    rulefit.fit(X_scaled, y_ml, feature_names=control_vars)
    print("Feature importances:")
    importances = rulefit.feature_importances_
    for feat, imp in zip(control_vars, importances):
        print(f"  {feat}: {imp:.4f}")
except Exception as e:
    print(f"RuleFit failed: {e}")

# Try FIGS (Fast Interpretable Greedy-Tree Sums)
print("\nFIGS Regressor (tree-based interpretable model):")
try:
    figs = FIGSRegressor(max_rules=5, random_state=42)
    figs.fit(X_scaled, y_ml, feature_names=control_vars)
    print("Feature importances:")
    importances = figs.feature_importances_
    for feat, imp in zip(control_vars, importances):
        print(f"  {feat}: {imp:.4f}")
except Exception as e:
    print(f"FIGS failed: {e}")

# INTERACTION EFFECTS
print("\n" + "="*80)
print("INTERACTION EFFECTS: FEMININITY x SEVERITY")
print("="*80)

# Test interaction between femininity and damage
df_interact = df[['masfem', 'ndam', 'alldeaths']].dropna()
df_interact['masfem_x_ndam'] = df_interact['masfem'] * df_interact['ndam']

X_interact = sm.add_constant(df_interact[['masfem', 'ndam', 'masfem_x_ndam']])
model_interact = sm.OLS(df_interact['alldeaths'], X_interact).fit()
print("\nInteraction model (masfem * ndam):")
print(model_interact.summary())

# ROBUSTNESS: LOG TRANSFORMATION FOR SKEWED DATA
print("\n" + "="*80)
print("ROBUSTNESS CHECK: LOG-TRANSFORMED DEATHS")
print("="*80)

# Deaths are highly skewed, try log transformation
df_log = df.copy()
df_log['log_deaths'] = np.log(df_log['alldeaths'] + 1)  # Add 1 to avoid log(0)

X_log = sm.add_constant(df_log['masfem'])
model_log = sm.OLS(df_log['log_deaths'], X_log).fit()
print("\nSimple regression (masfem -> log(deaths+1)):")
print(model_log.summary())

# CONCLUSION SYNTHESIS
print("\n" + "="*80)
print("SYNTHESIS AND CONCLUSION")
print("="*80)

# Gather evidence
evidence = {
    'simple_correlation': {
        'r': corr_fem_deaths,
        'p': p_fem_deaths,
        'significant': p_fem_deaths < 0.05
    },
    'spearman_correlation': {
        'rho': spearman_corr,
        'p': spearman_p,
        'significant': spearman_p < 0.05
    },
    'binary_ttest': {
        't': t_stat,
        'p': t_p,
        'significant': t_p < 0.05
    },
    'simple_regression': {
        'coef': model_simple.params['masfem'],
        'p': model_simple.pvalues['masfem'],
        'significant': model_simple.pvalues['masfem'] < 0.05
    },
    'multiple_regression': {
        'coef': model_multi.params['masfem'],
        'p': model_multi.pvalues['masfem'],
        'significant': model_multi.pvalues['masfem'] < 0.05
    },
    'log_regression': {
        'coef': model_log.params['masfem'],
        'p': model_log.pvalues['masfem'],
        'significant': model_log.pvalues['masfem'] < 0.05
    }
}

print("\nEvidence summary:")
for test_name, results in evidence.items():
    sig_str = "SIGNIFICANT" if results['significant'] else "NOT significant"
    if 'r' in results:
        print(f"{test_name}: r={results['r']:.4f}, p={results['p']:.4f} - {sig_str}")
    elif 'rho' in results:
        print(f"{test_name}: rho={results['rho']:.4f}, p={results['p']:.4f} - {sig_str}")
    elif 't' in results:
        print(f"{test_name}: t={results['t']:.4f}, p={results['p']:.4f} - {sig_str}")
    elif 'coef' in results:
        print(f"{test_name}: coef={results['coef']:.4f}, p={results['p']:.4f} - {sig_str}")

# Determine conclusion
significant_count = sum([results['significant'] for results in evidence.values()])
total_tests = len(evidence)

print(f"\n{significant_count} out of {total_tests} tests show statistical significance (p < 0.05)")

# Determine response score and explanation
if significant_count >= 4:
    response = 75
    explanation = (
        f"Strong evidence supports the hypothesis. {significant_count}/{total_tests} statistical tests "
        f"show significant relationships (p<0.05) between hurricane name femininity and deaths. "
        f"The simple correlation (r={corr_fem_deaths:.3f}, p={p_fem_deaths:.4f}) and multiple regression "
        f"controlling for severity (coef={model_multi.params['masfem']:.2f}, p={model_multi.pvalues['masfem']:.4f}) "
        "both indicate that more feminine-named hurricanes are associated with higher death tolls, "
        "consistent with the theory that they lead to fewer precautionary measures."
    )
elif significant_count >= 2:
    response = 55
    explanation = (
        f"Moderate evidence supports the hypothesis. {significant_count}/{total_tests} statistical tests "
        f"show significant relationships. While the simple correlation is significant (r={corr_fem_deaths:.3f}, "
        f"p={p_fem_deaths:.4f}), results are mixed when controlling for hurricane severity. "
        "This suggests a possible relationship, but confounding factors may play a role."
    )
elif significant_count == 1:
    response = 35
    explanation = (
        f"Weak evidence for the hypothesis. Only {significant_count}/{total_tests} test shows significance. "
        f"The relationship between femininity and deaths is not consistently significant across different "
        "statistical approaches. Alternative explanations or confounding factors likely explain much of the variation."
    )
else:
    response = 15
    explanation = (
        f"No statistical evidence supports the hypothesis. None of the {total_tests} tests show significant "
        f"relationships (all p > 0.05). Correlation: r={corr_fem_deaths:.3f}, p={p_fem_deaths:.4f}. "
        "The data do not support the claim that hurricane name femininity affects deaths through reduced precautionary measures."
    )

print(f"\nFinal assessment:")
print(f"Response score: {response}/100")
print(f"Explanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "="*80)
print("ANALYSIS COMPLETE - conclusion.txt written")
print("="*80)
