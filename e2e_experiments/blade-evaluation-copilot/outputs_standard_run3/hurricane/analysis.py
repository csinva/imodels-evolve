import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
import statsmodels.api as sm
import json
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('hurricane.csv')

print("=" * 80)
print("HURRICANE DATA ANALYSIS")
print("=" * 80)
print("\nResearch Question: Do hurricanes with more feminine names lead to fewer")
print("precautionary measures (resulting in more deaths)?")
print("\n")

# Data overview
print("Dataset shape:", df.shape)
print("\nKey variables:")
print("- masfem: Masculinity-Femininity index (1=masculine, 11=feminine)")
print("- alldeaths: Total deaths caused by hurricane")
print("- Other controls: min (pressure), category, wind, ndam (damage)")
print("\n")

# Summary statistics
print("=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)
print("\nMasculinity-Femininity Index (masfem):")
print(df['masfem'].describe())
print("\nAll Deaths:")
print(df['alldeaths'].describe())

# Check for missing values
print("\nMissing values:")
print(df[['masfem', 'alldeaths', 'min', 'category', 'wind', 'ndam']].isnull().sum())

# Remove rows with missing values for key variables
df_clean = df[['masfem', 'alldeaths', 'min', 'category', 'wind', 'ndam', 'gender_mf']].dropna()
print(f"\nClean dataset: {len(df_clean)} hurricanes (removed {len(df) - len(df_clean)} with missing data)")

# Correlation analysis
print("\n" + "=" * 80)
print("CORRELATION ANALYSIS")
print("=" * 80)
corr_masfem_deaths = df_clean['masfem'].corr(df_clean['alldeaths'])
print(f"\nPearson correlation (masfem vs alldeaths): {corr_masfem_deaths:.4f}")

# Spearman correlation (non-parametric)
spearman_corr, spearman_p = stats.spearmanr(df_clean['masfem'], df_clean['alldeaths'])
print(f"Spearman correlation: {spearman_corr:.4f} (p-value: {spearman_p:.4f})")

# Simple linear regression: deaths ~ masfem
print("\n" + "=" * 80)
print("SIMPLE LINEAR REGRESSION: Deaths ~ Femininity")
print("=" * 80)
X_simple = sm.add_constant(df_clean['masfem'])
y = df_clean['alldeaths']
model_simple = sm.OLS(y, X_simple).fit()
print(model_simple.summary())

print(f"\nInterpretation:")
print(f"- Coefficient for masfem: {model_simple.params['masfem']:.4f}")
print(f"- P-value: {model_simple.pvalues['masfem']:.4f}")
if model_simple.pvalues['masfem'] < 0.05:
    print(f"- SIGNIFICANT at p < 0.05")
    if model_simple.params['masfem'] > 0:
        print(f"- More feminine names ARE associated with MORE deaths")
    else:
        print(f"- More feminine names ARE associated with FEWER deaths")
else:
    print(f"- NOT SIGNIFICANT at p < 0.05")

# Multiple regression controlling for hurricane severity
print("\n" + "=" * 80)
print("MULTIPLE REGRESSION: Controlling for Hurricane Severity")
print("=" * 80)
print("Model: Deaths ~ Femininity + Min Pressure + Category + Wind Speed + Damage")

X_multi = sm.add_constant(df_clean[['masfem', 'min', 'category', 'wind', 'ndam']])
model_multi = sm.OLS(df_clean['alldeaths'], X_multi).fit()
print(model_multi.summary())

print(f"\nInterpretation (controlling for severity):")
print(f"- Coefficient for masfem: {model_multi.params['masfem']:.4f}")
print(f"- P-value: {model_multi.pvalues['masfem']:.4f}")
if model_multi.pvalues['masfem'] < 0.05:
    print(f"- SIGNIFICANT at p < 0.05 (even after controlling for severity)")
    if model_multi.params['masfem'] > 0:
        print(f"- More feminine names ARE associated with MORE deaths")
    else:
        print(f"- More feminine names ARE associated with FEWER deaths")
else:
    print(f"- NOT SIGNIFICANT at p < 0.05 (effect disappears when controlling for severity)")

# Log transformation (deaths are heavily skewed)
print("\n" + "=" * 80)
print("LOG-TRANSFORMED REGRESSION")
print("=" * 80)
print("Using log(deaths + 1) to handle skewness in death counts")

df_clean['log_deaths'] = np.log(df_clean['alldeaths'] + 1)
df_clean['log_ndam'] = np.log(df_clean['ndam'] + 1)

X_log = sm.add_constant(df_clean[['masfem', 'min', 'category', 'wind', 'log_ndam']])
model_log = sm.OLS(df_clean['log_deaths'], X_log).fit()
print(model_log.summary())

print(f"\nInterpretation (log-transformed):")
print(f"- Coefficient for masfem: {model_log.params['masfem']:.4f}")
print(f"- P-value: {model_log.pvalues['masfem']:.4f}")

# T-test comparing male vs female named hurricanes
print("\n" + "=" * 80)
print("T-TEST: Male vs Female Named Hurricanes")
print("=" * 80)

male_deaths = df_clean[df_clean['gender_mf'] == 0]['alldeaths']
female_deaths = df_clean[df_clean['gender_mf'] == 1]['alldeaths']

print(f"Male named hurricanes (n={len(male_deaths)}): Mean deaths = {male_deaths.mean():.2f} (SD = {male_deaths.std():.2f})")
print(f"Female named hurricanes (n={len(female_deaths)}): Mean deaths = {female_deaths.mean():.2f} (SD = {female_deaths.std():.2f})")

t_stat, t_pval = stats.ttest_ind(male_deaths, female_deaths)
print(f"\nT-test: t = {t_stat:.4f}, p-value = {t_pval:.4f}")

if t_pval < 0.05:
    print("- SIGNIFICANT difference at p < 0.05")
else:
    print("- NOT SIGNIFICANT at p < 0.05")

# Mann-Whitney U test (non-parametric alternative)
u_stat, u_pval = stats.mannwhitneyu(male_deaths, female_deaths, alternative='two-sided')
print(f"\nMann-Whitney U test (non-parametric): U = {u_stat:.2f}, p-value = {u_pval:.4f}")

# Interpretable model: Decision Tree
print("\n" + "=" * 80)
print("INTERPRETABLE MODEL: Decision Tree Regressor")
print("=" * 80)

X_tree = df_clean[['masfem', 'min', 'category', 'wind', 'ndam']]
y_tree = df_clean['alldeaths']

tree = DecisionTreeRegressor(max_depth=3, min_samples_split=10, random_state=42)
tree.fit(X_tree, y_tree)

print("Feature importances:")
for feat, imp in sorted(zip(X_tree.columns, tree.feature_importances_), key=lambda x: x[1], reverse=True):
    print(f"  {feat}: {imp:.4f}")

# Final conclusion
print("\n" + "=" * 80)
print("FINAL CONCLUSION")
print("=" * 80)

# Gather all p-values for masfem
p_values = {
    'simple_regression': model_simple.pvalues['masfem'],
    'multiple_regression': model_multi.pvalues['masfem'],
    'log_regression': model_log.pvalues['masfem'],
    'spearman_correlation': spearman_p,
    't_test': t_pval
}

print("\nSummary of statistical tests for femininity effect:")
for test, pval in p_values.items():
    sig = "SIGNIFICANT" if pval < 0.05 else "NOT SIGNIFICANT"
    print(f"  {test}: p = {pval:.4f} ({sig})")

# Determine response
significant_tests = sum(1 for p in p_values.values() if p < 0.05)
total_tests = len(p_values)

print(f"\n{significant_tests} out of {total_tests} tests show statistical significance (p < 0.05)")

# Check direction of effect in significant tests
simple_sig = model_simple.pvalues['masfem'] < 0.05
multi_sig = model_multi.pvalues['masfem'] < 0.05
log_sig = model_log.pvalues['masfem'] < 0.05

simple_coef = model_simple.params['masfem']
multi_coef = model_multi.params['masfem']
log_coef = model_log.params['masfem']

print("\nEffect direction in regression models:")
print(f"  Simple regression: {simple_coef:.4f} ({'positive' if simple_coef > 0 else 'negative'})")
print(f"  Multiple regression: {multi_coef:.4f} ({'positive' if multi_coef > 0 else 'negative'})")
print(f"  Log regression: {log_coef:.4f} ({'positive' if log_coef > 0 else 'negative'})")

# Decision logic
explanation_parts = []

if simple_sig and simple_coef > 0:
    explanation_parts.append(f"Simple linear regression shows a significant positive relationship (p={model_simple.pvalues['masfem']:.4f})")
    
if multi_sig and multi_coef > 0:
    explanation_parts.append(f"Multiple regression controlling for hurricane severity shows significant positive effect (p={model_multi.pvalues['masfem']:.4f})")
elif not multi_sig:
    explanation_parts.append(f"When controlling for hurricane severity (pressure, category, wind, damage), the effect becomes non-significant (p={model_multi.pvalues['masfem']:.4f})")

if spearman_p < 0.05:
    explanation_parts.append(f"Spearman correlation is significant (p={spearman_p:.4f})")
    
tree_masfem_importance = tree.feature_importances_[0]
if tree_masfem_importance < 0.1:
    explanation_parts.append(f"Decision tree model shows low importance for femininity ({tree_masfem_importance:.4f})")

# Determine final score
if significant_tests >= 3 and simple_coef > 0:
    # Strong evidence for the hypothesis
    if multi_sig:
        response = 75  # Strong support, even after controls
        explanation_parts.append("The evidence supports that more feminine names are associated with more deaths, suggesting fewer precautionary measures.")
    else:
        response = 50  # Marginal support, effect disappears with controls
        explanation_parts.append("However, the effect is largely explained by confounding factors (hurricane severity).")
elif significant_tests >= 2 and simple_coef > 0:
    response = 40  # Weak support
    explanation_parts.append("There is weak evidence for the hypothesis, but it's not robust across all tests.")
else:
    response = 20  # Little to no support
    explanation_parts.append("The evidence does not strongly support the hypothesis that feminine names lead to fewer precautionary measures.")

explanation = " ".join(explanation_parts)

print(f"\n{'='*80}")
print(f"LIKERT SCALE RESPONSE: {response}")
print(f"\nExplanation: {explanation}")
print(f"{'='*80}")

# Write conclusion to file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n✓ conclusion.txt written successfully!")
