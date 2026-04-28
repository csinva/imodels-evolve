import pandas as pd
import numpy as np
import json
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from interpret.glassbox import ExplainableBoostingRegressor

# Load the dataset
df = pd.read_csv('hurricane.csv')

print("=" * 80)
print("HURRICANE ANALYSIS: Femininity of Names and Deaths")
print("=" * 80)
print("\nResearch Question:")
print("Do hurricanes with more feminine names lead to more deaths (due to fewer")
print("precautionary measures by the public)?")
print("\n")

# Exploratory Data Analysis
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nBasic statistics:")
print(df[['masfem', 'alldeaths', 'category', 'ndam', 'wind', 'min']].describe())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Key variables of interest
print("\n" + "=" * 80)
print("CORRELATION ANALYSIS")
print("=" * 80)

# Pearson correlation between femininity and deaths
corr_masfem_deaths, p_value_corr = stats.pearsonr(df['masfem'], df['alldeaths'])
print(f"\nPearson correlation between masfem and alldeaths: {corr_masfem_deaths:.4f}")
print(f"P-value: {p_value_corr:.4f}")

# Spearman correlation (more robust to outliers)
spearman_corr, spearman_p = stats.spearmanr(df['masfem'], df['alldeaths'])
print(f"\nSpearman correlation between masfem and alldeaths: {spearman_corr:.4f}")
print(f"P-value: {spearman_p:.4f}")

# Check correlation with other femininity measure (mturk)
if 'masfem_mturk' in df.columns:
    corr_mturk_deaths, p_mturk = stats.pearsonr(df['masfem_mturk'], df['alldeaths'])
    print(f"\nPearson correlation between masfem_mturk and alldeaths: {corr_mturk_deaths:.4f}")
    print(f"P-value: {p_mturk:.4f}")

# Simple regression: deaths ~ femininity
print("\n" + "=" * 80)
print("SIMPLE LINEAR REGRESSION: alldeaths ~ masfem")
print("=" * 80)

X_simple = sm.add_constant(df['masfem'])
y = df['alldeaths']
model_simple = sm.OLS(y, X_simple).fit()
print(model_simple.summary())

# Multiple regression controlling for hurricane severity
print("\n" + "=" * 80)
print("MULTIPLE REGRESSION: Controlling for Hurricane Severity")
print("=" * 80)

# Control for category, wind speed, minimum pressure, and damage
control_vars = ['masfem', 'category', 'wind', 'ndam']
df_model = df[control_vars + ['alldeaths']].dropna()

X_multi = sm.add_constant(df_model[control_vars])
y_multi = df_model['alldeaths']
model_multi = sm.OLS(y_multi, X_multi).fit()
print(model_multi.summary())

# Interpretable model using EBM
print("\n" + "=" * 80)
print("EXPLAINABLE BOOSTING MACHINE (EBM)")
print("=" * 80)

# Prepare data for EBM
X_ebm = df_model[control_vars].values
y_ebm = df_model['alldeaths'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_ebm, y_ebm, test_size=0.2, random_state=42)

# Fit EBM
ebm = ExplainableBoostingRegressor(random_state=42, interactions=0)
ebm.fit(X_train, y_train)

# Get feature importances
feature_names = control_vars
importances = ebm.term_importances()
print("\nFeature Importances from EBM:")
for name, imp in zip(feature_names, importances):
    print(f"  {name}: {imp:.4f}")

# Test with and without controlling for outliers
print("\n" + "=" * 80)
print("SENSITIVITY ANALYSIS: Removing High-Death Outliers")
print("=" * 80)

# Identify outliers (deaths > 95th percentile)
death_threshold = df['alldeaths'].quantile(0.95)
print(f"Deaths 95th percentile: {death_threshold}")

df_no_outliers = df[df['alldeaths'] <= death_threshold].copy()
print(f"Sample size without outliers: {len(df_no_outliers)} (removed {len(df) - len(df_no_outliers)} storms)")

# Re-run correlation without outliers
corr_no_outliers, p_no_outliers = stats.pearsonr(df_no_outliers['masfem'], df_no_outliers['alldeaths'])
print(f"\nPearson correlation (without outliers): {corr_no_outliers:.4f}")
print(f"P-value: {p_no_outliers:.4f}")

# Re-run regression without outliers
X_no_outliers = sm.add_constant(df_no_outliers['masfem'])
y_no_outliers = df_no_outliers['alldeaths']
model_no_outliers = sm.OLS(y_no_outliers, X_no_outliers).fit()
print(f"\nRegression coefficient for masfem (without outliers): {model_no_outliers.params['masfem']:.4f}")
print(f"P-value: {model_no_outliers.pvalues['masfem']:.4f}")

# Group comparison: Male vs Female named hurricanes
print("\n" + "=" * 80)
print("T-TEST: Male vs Female Named Hurricanes")
print("=" * 80)

if 'gender_mf' in df.columns:
    male_deaths = df[df['gender_mf'] == 0]['alldeaths']
    female_deaths = df[df['gender_mf'] == 1]['alldeaths']
    
    print(f"Male-named hurricanes (n={len(male_deaths)}): mean deaths = {male_deaths.mean():.2f}, std = {male_deaths.std():.2f}")
    print(f"Female-named hurricanes (n={len(female_deaths)}): mean deaths = {female_deaths.mean():.2f}, std = {female_deaths.std():.2f}")
    
    t_stat, t_pvalue = stats.ttest_ind(male_deaths, female_deaths)
    print(f"\nT-test: t={t_stat:.4f}, p-value={t_pvalue:.4f}")

# Summary and Conclusion
print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

# Key findings
sig_level = 0.05

findings = []
findings.append(f"1. Simple correlation (Pearson): r={corr_masfem_deaths:.4f}, p={p_value_corr:.4f}")
findings.append(f"2. Spearman correlation: rho={spearman_corr:.4f}, p={spearman_p:.4f}")
findings.append(f"3. Simple regression coefficient: {model_simple.params['masfem']:.4f}, p={model_simple.pvalues['masfem']:.4f}")
findings.append(f"4. Multiple regression coefficient (controlling for severity): {model_multi.params['masfem']:.4f}, p={model_multi.pvalues['masfem']:.4f}")
findings.append(f"5. Without outliers: r={corr_no_outliers:.4f}, p={p_no_outliers:.4f}")

print("\nKey Findings:")
for finding in findings:
    print(finding)

# Determine conclusion based on statistical evidence
# The research question is about whether feminine names lead to MORE deaths
# A positive relationship (more feminine = more deaths) would support the hypothesis
# Statistical significance at p < 0.05 is the threshold

simple_regression_significant = model_simple.pvalues['masfem'] < sig_level
simple_regression_positive = model_simple.params['masfem'] > 0

multi_regression_significant = model_multi.pvalues['masfem'] < sig_level
multi_regression_positive = model_multi.params['masfem'] > 0

correlation_significant = p_value_corr < sig_level
correlation_positive = corr_masfem_deaths > 0

# The most important test is the multiple regression controlling for hurricane severity
# because it accounts for confounding factors

if multi_regression_significant and multi_regression_positive:
    response_score = 75  # Strong evidence when controlling for confounds
    explanation = (
        f"YES - There is statistically significant evidence (p={model_multi.pvalues['masfem']:.4f} < 0.05) "
        f"that hurricanes with more feminine names are associated with MORE deaths, even when controlling "
        f"for hurricane severity (category, wind speed, damage). The multiple regression coefficient for "
        f"femininity is {model_multi.params['masfem']:.2f}, meaning each unit increase in femininity score "
        f"is associated with approximately {model_multi.params['masfem']:.2f} more deaths, holding severity constant. "
        f"This supports the hypothesis that feminine-named hurricanes may lead to fewer precautionary measures."
    )
elif simple_regression_significant and simple_regression_positive:
    # Significant in simple but not multiple regression - suggests confounding
    response_score = 40  # Weak to moderate evidence
    explanation = (
        f"MIXED - There is a significant positive correlation (r={corr_masfem_deaths:.4f}, p={p_value_corr:.4f}) "
        f"between femininity and deaths in simple analysis (p={model_simple.pvalues['masfem']:.4f}), but when "
        f"controlling for hurricane severity in multiple regression, the effect becomes less clear "
        f"(p={model_multi.pvalues['masfem']:.4f}). This suggests the relationship may be confounded by hurricane "
        f"severity and provides only weak support for the hypothesis."
    )
elif correlation_positive:
    # Positive but not significant
    response_score = 25
    explanation = (
        f"WEAK/NO - While there is a positive correlation (r={corr_masfem_deaths:.4f}), it is not "
        f"statistically significant (p={p_value_corr:.4f} > 0.05). The multiple regression controlling for "
        f"severity also shows no significant effect (p={model_multi.pvalues['masfem']:.4f}). There is insufficient "
        f"statistical evidence to conclude that feminine-named hurricanes lead to more deaths."
    )
else:
    # Negative or no relationship
    response_score = 10
    explanation = (
        f"NO - There is no statistical evidence supporting the hypothesis. The correlation is not positive and "
        f"not significant (r={corr_masfem_deaths:.4f}, p={p_value_corr:.4f}). Multiple regression controlling for "
        f"severity shows no significant effect (p={model_multi.pvalues['masfem']:.4f}). The data do not support "
        f"the claim that feminine-named hurricanes lead to more deaths."
    )

print(f"\nFinal Assessment:")
print(f"Score: {response_score}/100")
print(f"Explanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response_score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "=" * 80)
print("Analysis complete. Results written to conclusion.txt")
print("=" * 80)
