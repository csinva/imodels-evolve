import pandas as pd
import numpy as np
import json
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
import statsmodels.api as sm
from imodels import RuleFitRegressor, FIGSRegressor
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('hurricane.csv')

# Basic exploration
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nSummary statistics:")
print(df.describe())

# Focus on key variables: masfem (femininity), alldeaths (outcome)
print("\n" + "="*80)
print("ANALYZING RELATIONSHIP BETWEEN FEMININITY OF NAME AND DEATHS")
print("="*80)

# Remove any missing values in key columns
df_clean = df[['masfem', 'alldeaths', 'min', 'ndam', 'category', 'wind']].dropna()
print(f"\nClean dataset: {len(df_clean)} hurricanes")

# 1. CORRELATION ANALYSIS
print("\n--- Correlation Analysis ---")
corr = df_clean['masfem'].corr(df_clean['alldeaths'])
print(f"Pearson correlation between masfem and alldeaths: {corr:.4f}")

# Spearman correlation (non-parametric)
spearman_corr, spearman_p = stats.spearmanr(df_clean['masfem'], df_clean['alldeaths'])
print(f"Spearman correlation: {spearman_corr:.4f}, p-value: {spearman_p:.4f}")

# 2. SIMPLE LINEAR REGRESSION
print("\n--- Simple Linear Regression (Deaths ~ Femininity) ---")
X_simple = df_clean[['masfem']]
y = df_clean['alldeaths']
model_simple = LinearRegression()
model_simple.fit(X_simple, y)
print(f"Coefficient: {model_simple.coef_[0]:.4f}")
print(f"Intercept: {model_simple.intercept_:.4f}")
print(f"R-squared: {model_simple.score(X_simple, y):.4f}")

# Statistical significance test using statsmodels
X_sm = sm.add_constant(df_clean['masfem'])
model_ols = sm.OLS(df_clean['alldeaths'], X_sm).fit()
print("\nOLS Regression Summary:")
print(model_ols.summary())
print(f"\nCoefficient p-value: {model_ols.pvalues['masfem']:.6f}")

# 3. MULTIPLE REGRESSION CONTROLLING FOR CONFOUNDERS
print("\n--- Multiple Regression (Controlling for Hurricane Severity) ---")
# Control for minimum pressure, damage, category, and wind speed
X_multiple = df_clean[['masfem', 'min', 'ndam', 'category', 'wind']]
model_multiple = LinearRegression()
model_multiple.fit(X_multiple, y)
print("Coefficients:")
for i, col in enumerate(X_multiple.columns):
    print(f"  {col}: {model_multiple.coef_[i]:.4f}")
print(f"R-squared: {model_multiple.score(X_multiple, y):.4f}")

# Statistical significance with multiple regression
X_sm_multiple = sm.add_constant(X_multiple)
model_ols_multiple = sm.OLS(y, X_sm_multiple).fit()
print(f"\nMasfem coefficient p-value (controlling for severity): {model_ols_multiple.pvalues['masfem']:.6f}")
print(f"Is masfem significant at 0.05 level? {model_ols_multiple.pvalues['masfem'] < 0.05}")

# 4. INTERPRETABLE MODELS
print("\n--- Interpretable Models ---")

# Decision Tree
tree_model = DecisionTreeRegressor(max_depth=3, min_samples_leaf=10, random_state=42)
tree_model.fit(X_multiple, y)
print("\nDecision Tree Feature Importances:")
for i, col in enumerate(X_multiple.columns):
    print(f"  {col}: {tree_model.feature_importances_[i]:.4f}")

# Try imodels - FIGS
try:
    figs_model = FIGSRegressor(max_rules=5, random_state=42)
    figs_model.fit(X_multiple, y)
    print("\nFIGS Model Feature Importances:")
    for i, col in enumerate(X_multiple.columns):
        print(f"  {col}: {figs_model.feature_importances_[i]:.4f}")
except Exception as e:
    print(f"FIGS model failed: {e}")

# 5. T-TEST: Compare high femininity vs low femininity
print("\n--- T-test: High vs Low Femininity Names ---")
median_masfem = df_clean['masfem'].median()
high_fem = df_clean[df_clean['masfem'] >= median_masfem]['alldeaths']
low_fem = df_clean[df_clean['masfem'] < median_masfem]['alldeaths']
t_stat, t_pval = stats.ttest_ind(high_fem, low_fem)
print(f"High femininity deaths (mean): {high_fem.mean():.2f}")
print(f"Low femininity deaths (mean): {low_fem.mean():.2f}")
print(f"T-statistic: {t_stat:.4f}, p-value: {t_pval:.4f}")

# 6. ANALYZE INTERACTION EFFECTS
print("\n--- Interaction with Hurricane Severity ---")
# Create interaction term: femininity * damage
df_clean['masfem_x_ndam'] = df_clean['masfem'] * df_clean['ndam']
X_interaction = df_clean[['masfem', 'ndam', 'masfem_x_ndam', 'min', 'category']]
X_sm_interaction = sm.add_constant(X_interaction)
model_interaction = sm.OLS(y, X_sm_interaction).fit()
print(f"Interaction term (masfem x ndam) p-value: {model_interaction.pvalues['masfem_x_ndam']:.6f}")

# CONCLUSION
print("\n" + "="*80)
print("FINAL ANALYSIS")
print("="*80)

# Key findings
simple_regression_pval = model_ols.pvalues['masfem']
multiple_regression_pval = model_ols_multiple.pvalues['masfem']
simple_coef = model_simple.coef_[0]
multiple_coef = model_multiple.coef_[0]

print(f"\n1. Simple regression: masfem coefficient = {simple_coef:.4f}, p-value = {simple_regression_pval:.6f}")
print(f"2. Multiple regression (controlling for severity): masfem coefficient = {multiple_coef:.4f}, p-value = {multiple_regression_pval:.6f}")
print(f"3. T-test comparing high vs low femininity: p-value = {t_pval:.4f}")
print(f"4. Spearman correlation: rho = {spearman_corr:.4f}, p-value = {spearman_p:.4f}")

# Determine response score
# The hypothesis is that more feminine names lead to more deaths (fewer precautions)
# We need positive coefficient AND statistical significance

# Consider both simple and controlled analyses
if simple_regression_pval < 0.05 and simple_coef > 0:
    # Significant positive relationship in simple model
    if multiple_regression_pval < 0.05 and multiple_coef > 0:
        # Also significant when controlling for confounders - STRONG evidence
        response = 85
        explanation = f"Strong evidence: Femininity positively predicts deaths in both simple (coef={simple_coef:.2f}, p={simple_regression_pval:.4f}) and controlled regression (coef={multiple_coef:.2f}, p={multiple_regression_pval:.4f}), suggesting feminine-named hurricanes do lead to more deaths, consistent with fewer precautions."
    else:
        # Significant in simple but not controlled - MODERATE evidence
        response = 60
        explanation = f"Moderate evidence: Femininity significantly predicts deaths in simple regression (coef={simple_coef:.2f}, p={simple_regression_pval:.4f}), but relationship weakens when controlling for hurricane severity (p={multiple_regression_pval:.4f}). Some support for the hypothesis but confounded by other factors."
elif simple_coef > 0 and simple_regression_pval < 0.10:
    # Marginal significance
    response = 50
    explanation = f"Weak evidence: Positive but marginally significant relationship (coef={simple_coef:.2f}, p={simple_regression_pval:.4f}). The trend is in the hypothesized direction but not statistically robust."
elif simple_coef > 0:
    # Positive direction but not significant
    response = 30
    explanation = f"Insufficient evidence: Coefficient is positive (coef={simple_coef:.2f}) but not statistically significant (p={simple_regression_pval:.4f}). Cannot conclude that femininity affects precautionary behavior."
else:
    # Negative or no relationship
    response = 10
    explanation = f"No evidence: The relationship between femininity and deaths is not in the hypothesized direction (coef={simple_coef:.2f}, p={simple_regression_pval:.4f}). No support for the hypothesis."

print(f"\nFinal Score: {response}/100")
print(f"Explanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n✓ Conclusion written to conclusion.txt")
