import pandas as pd
import numpy as np
import json
from scipy import stats
from scipy.stats import pearsonr, spearmanr, ttest_ind
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from interpret.glassbox import ExplainableBoostingRegressor

# Load data
df = pd.read_csv('hurricane.csv')

# Research question: Hurricanes with more feminine names are perceived as less threatening 
# and hence lead to fewer precautionary measures by the general public.
# Hypothesis: More feminine names -> less precaution -> more deaths

print("=" * 80)
print("HURRICANE NAME FEMININITY AND DEATHS ANALYSIS")
print("=" * 80)

print("\nDataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

print("\n" + "=" * 80)
print("1. DESCRIPTIVE STATISTICS")
print("=" * 80)

print("\nKey variables summary:")
print(df[['masfem', 'alldeaths', 'category', 'min', 'wind', 'ndam']].describe())

print("\nCorrelation between femininity (masfem) and deaths:")
corr_masfem_deaths, p_corr = pearsonr(df['masfem'], df['alldeaths'])
print(f"Pearson correlation: {corr_masfem_deaths:.4f}, p-value: {p_corr:.4f}")

spearman_corr, p_spearman = spearmanr(df['masfem'], df['alldeaths'])
print(f"Spearman correlation: {spearman_corr:.4f}, p-value: {p_spearman:.4f}")

print("\n" + "=" * 80)
print("2. REGRESSION ANALYSIS: Femininity -> Deaths")
print("=" * 80)

# Simple regression: deaths ~ masfem
X_simple = df[['masfem']].values
y = df['alldeaths'].values

X_simple_sm = sm.add_constant(X_simple)
model_simple = sm.OLS(y, X_simple_sm).fit()
print("\nSimple regression (deaths ~ masfem):")
print(model_simple.summary())

print("\n" + "=" * 80)
print("3. CONTROL FOR HURRICANE SEVERITY")
print("=" * 80)

# Multiple regression controlling for hurricane severity
# Handle missing values
df_clean = df[['masfem', 'min', 'wind', 'ndam', 'alldeaths']].dropna()
X_controls = df_clean[['masfem', 'min', 'wind', 'ndam']].values
y_clean = df_clean['alldeaths'].values
X_controls_sm = sm.add_constant(X_controls)
model_controls = sm.OLS(y_clean, X_controls_sm).fit()
print("\nMultiple regression (deaths ~ masfem + min + wind + ndam):")
print(model_controls.summary())

print("\n" + "=" * 80)
print("4. INTERPRETABLE MODEL: EXPLAINABLE BOOSTING")
print("=" * 80)

# Use Explainable Boosting Regressor for interpretable prediction
X_ebm = df[['masfem', 'min', 'wind', 'ndam']].fillna(df[['masfem', 'min', 'wind', 'ndam']].mean())
y_ebm = df['alldeaths'].values

ebm = ExplainableBoostingRegressor(random_state=42, interactions=0)
ebm.fit(X_ebm, y_ebm)

print("\nEBM Feature Importances:")
feature_names = ['masfem', 'min', 'wind', 'ndam']
for name, importance in zip(feature_names, ebm.term_importances()):
    print(f"  {name}: {importance:.4f}")

print("\n" + "=" * 80)
print("5. GROUP COMPARISON: MALE vs FEMALE NAMED HURRICANES")
print("=" * 80)

# Split by binary gender
male_deaths = df[df['gender_mf'] == 0]['alldeaths']
female_deaths = df[df['gender_mf'] == 1]['alldeaths']

print(f"\nMale-named hurricanes (n={len(male_deaths)}): mean deaths = {male_deaths.mean():.2f}, median = {male_deaths.median():.2f}")
print(f"Female-named hurricanes (n={len(female_deaths)}): mean deaths = {female_deaths.mean():.2f}, median = {female_deaths.median():.2f}")

t_stat, p_ttest = ttest_ind(female_deaths, male_deaths)
print(f"\nT-test (female vs male): t = {t_stat:.4f}, p-value: {p_ttest:.4f}")

# Also control for severity in group comparison
male_severity = df[df['gender_mf'] == 0][['min', 'wind', 'ndam']].mean()
female_severity = df[df['gender_mf'] == 1][['min', 'wind', 'ndam']].mean()

print("\nMean severity indicators by gender:")
print("Male-named hurricanes:")
print(f"  Min pressure: {male_severity['min']:.2f}")
print(f"  Wind speed: {male_severity['wind']:.2f}")
print(f"  Damage: {male_severity['ndam']:.2f}")
print("Female-named hurricanes:")
print(f"  Min pressure: {female_severity['min']:.2f}")
print(f"  Wind speed: {female_severity['wind']:.2f}")
print(f"  Damage: {female_severity['ndam']:.2f}")

print("\n" + "=" * 80)
print("6. CONCLUSION")
print("=" * 80)

# Analysis of evidence
evidence = []

# 1. Simple correlation
if p_corr < 0.05:
    evidence.append(f"Significant correlation between femininity and deaths (r={corr_masfem_deaths:.3f}, p={p_corr:.4f})")
else:
    evidence.append(f"NO significant correlation between femininity and deaths (r={corr_masfem_deaths:.3f}, p={p_corr:.4f})")

# 2. Simple regression
masfem_coef_simple = model_simple.params[1]
masfem_pval_simple = model_simple.pvalues[1]
if masfem_pval_simple < 0.05:
    evidence.append(f"Significant effect in simple regression (coef={masfem_coef_simple:.3f}, p={masfem_pval_simple:.4f})")
else:
    evidence.append(f"NO significant effect in simple regression (coef={masfem_coef_simple:.3f}, p={masfem_pval_simple:.4f})")

# 3. Controlled regression
masfem_coef_controls = model_controls.params[1]
masfem_pval_controls = model_controls.pvalues[1]
if masfem_pval_controls < 0.05:
    evidence.append(f"Significant effect controlling for severity (coef={masfem_coef_controls:.3f}, p={masfem_pval_controls:.4f})")
else:
    evidence.append(f"NO significant effect controlling for severity (coef={masfem_coef_controls:.3f}, p={masfem_pval_controls:.4f})")

# 4. Group comparison
if p_ttest < 0.05:
    evidence.append(f"Significant difference between male and female named hurricanes (p={p_ttest:.4f})")
else:
    evidence.append(f"NO significant difference between male and female named hurricanes (p={p_ttest:.4f})")

print("\nEvidence summary:")
for i, e in enumerate(evidence, 1):
    print(f"{i}. {e}")

# Final determination
significant_results = sum([
    p_corr < 0.05,
    masfem_pval_simple < 0.05,
    masfem_pval_controls < 0.05,
    p_ttest < 0.05
])

print(f"\nSignificant results: {significant_results}/4")

# Determine response score
if significant_results == 0:
    response = 10  # Strong "No"
    explanation = "No statistically significant relationship found between hurricane name femininity and deaths in any of the tests (correlation, simple regression, controlled regression, or group comparison). The hypothesis is not supported by the data."
elif significant_results == 1:
    if masfem_pval_controls < 0.05:
        response = 60
        explanation = "Mixed evidence: while some tests show significance, the most important test (controlled for hurricane severity) shows a significant relationship. However, the lack of consistent significance across all tests suggests the relationship may be weak or context-dependent."
    else:
        response = 40
        explanation = "Weak evidence: only one of four statistical tests shows significance, and it does not hold when controlling for hurricane severity. This suggests the relationship may be spurious or confounded by other factors."
elif significant_results == 2:
    if masfem_pval_controls < 0.05:
        response = 70
        explanation = "Moderate evidence: half of the statistical tests show significance, including the controlled regression that accounts for hurricane severity. This suggests there may be a real relationship, though it's not uniformly strong across all analyses."
    else:
        response = 45
        explanation = "Mixed evidence: while some tests show significance, the relationship disappears when controlling for hurricane severity, suggesting the apparent relationship may be confounded by other factors."
elif significant_results == 3:
    response = 80
    explanation = "Strong evidence: three of four statistical tests show significant relationships between hurricane name femininity and deaths. The preponderance of evidence supports the hypothesis, though not all tests confirm it."
else:  # all 4 significant
    response = 90
    explanation = "Very strong evidence: all statistical tests (correlation, simple regression, controlled regression, and group comparison) show significant relationships between hurricane name femininity and deaths. The consistent significance across multiple analytical approaches strongly supports the hypothesis."

print(f"\nFinal assessment: {response}/100")
print(f"Explanation: {explanation}")

# Write conclusion
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "=" * 80)
print("Analysis complete. Conclusion written to conclusion.txt")
print("=" * 80)
